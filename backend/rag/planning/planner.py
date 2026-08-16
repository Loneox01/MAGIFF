import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import DEFAULT_INDEX_PATH, DEFAULT_PLANNER_MODEL
from .schema_values import (
    Conference,
    DepthChartPosition,
    Division,
    ECRLeagueFormat,
    ECRPosition,
    ECRScoringFormat,
    Formation,
    PlayerPosition,
    PositionGroup,
    RosterStatus,
    TeamCode,
)


PLANNER_PROMPT_VERSION = "14"

PLANNER_INSTRUCTIONS = """Plan evidence retrieval from a fantasy-football report corpus.

Return a retrieval plan, not an answer. Do not assert that an event occurred or
invent entities, dates, attributes, or relationships. Do not include conversational text.

Interpret the complete user question:
- preserve its subjects, comparisons, exclusions, uncertainty, and time scope
- for every reference to one specific player, preserve the exact phrase from the
  question in reference_text and enter the best official full-name guess in
  names, even when uncertain
- assign identity_confidence from 0 to 1 and classify resolution_basis as:
  exact_name when the phrase is already the official name; known_alias when a
  broadly recognized nickname, initialism, shortened name, or common name
  variant identifies the player without context; contextual_alias when team,
  position, season, or other context identifies the player; inferred when the
  identity is only the best available interpretation
- for a player group rather than one specific player, use not_applicable, set
  identity_confidence to 0, and leave names empty
- avoid adding topics or constraints not stated or strongly implied

Build focused retrieval inputs:
- semantic_query: a natural-language statement of the evidence needed, without
  planner instructions or commentary about resolving identities
- keyword_query: compact, discriminative names, concepts, events, and time terms
- mentions: official candidate names only, without aliases or duplicates
- selectors: one per independently selected or compared entity; use only fields,
  values, operators, and entity combinations permitted by the schema
- keep every objective constraint describing the same player or player group in
  that player selector; a separate team selector or team_mentions entry does
  not constrain a player selector
- use a team selector only when the team itself is independently selected or
  compared; when a team limits a player group, express that relationship with
  a team filter inside the player selector
- when teams are selected according to a condition about a player group or
  position-specific role, represent both subjects: use a team selector for the
  teams being selected and a player-group selector containing the relevant
  position or attribute constraints plus its team, division, or conference scope
- player selectors: follow the player-reference rules above and never put
  identity uncertainty or resolution instructions in either search query
- hard_filters: only objective constraints explicitly stated in the question or
  unambiguously normalized or entailed by it; hard filters may exclude evidence
- soft_filters: optional structured context inferred from background knowledge;
  soft filters never constrain database lookup or exclude evidence, and should
  usually be omitted when the fact is unstable or unnecessary
- do not copy a soft-only value into semantic_query or keyword_query unless the
  same concept is present in the user's question; soft context must not bias the
  candidate pool before database grounding
- never place a player's team, position, status, or other relationship in
  hard_filters merely because you believe it to be true; if the question does
  not supply that relationship, either omit it or place it in soft_filters
- team_mentions contains only teams explicitly mentioned or unambiguously
  normalized from the question; put inferred team context in soft_team_mentions
- database enrichment happens after planning and must not be anticipated as a
  hard filter
- semantic qualifiers: subjective or report-derived descriptions requiring text
  evidence rather than exact database matching

Represent time deliberately. Use latest/current for present-status questions,
date boundaries when supplied, and timeline for change across time. Resolve an
unambiguous partial date using the supplied current date. Set needs_baseline when
the requested conclusion requires both earlier and later evidence. Do not set
start and end dates unless provided in the user prompt.

Choose the smallest evidence strategy that can answer the whole question:
single_document for one sufficient report, multiple_documents for synthesis or
corroboration, per_entity for independent coverage of multiple subjects, and
timeline for chronological change. Treat negative_focus as de-emphasis rather
than an unconditional exclusion.
"""


class EntityFilterField(StrEnum):
    TEAM = "team"
    POSITION = "position"
    POSITION_GROUP = "position_group"
    ROOKIE_SEASON = "rookie_season"
    DRAFT_YEAR = "draft_year"
    DRAFT_ROUND = "draft_round"
    DRAFT_PICK = "draft_pick"
    COLLEGE = "college"
    HEIGHT = "height"
    WEIGHT = "weight"
    LAST_SEASON = "last_season"
    ROSTER_STATUS = "roster_status"
    YEARS_EXPERIENCE = "years_experience"
    DEPTH_CHART_POSITION = "depth_chart_position"
    DEPTH_RANK = "depth_rank"
    FORMATION = "formation"
    ECR_RANK = "ecr_rank"
    ECR_POSITION = "ecr_position"
    ECR_SCORING_FORMAT = "ecr_scoring_format"
    ECR_LEAGUE_FORMAT = "ecr_league_format"
    GAMES = "games"
    FANTASY_POINTS = "fantasy_points"
    FANTASY_POINTS_PPR = "fantasy_points_ppr"
    TARGETS = "targets"
    CARRIES = "carries"
    RECEPTIONS = "receptions"
    OFFENSE_SNAP_PCT = "offense_snap_pct"
    OPPONENT = "opponent"
    CONFERENCE = "conference"
    DIVISION = "division"


class PlannerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlayerResolutionBasis(StrEnum):
    EXACT_NAME = "exact_name"
    KNOWN_ALIAS = "known_alias"
    CONTEXTUAL_ALIAS = "contextual_alias"
    INFERRED = "inferred"
    NOT_APPLICABLE = "not_applicable"


class TeamCodeFilter(PlannerModel):
    field: Literal[EntityFilterField.TEAM]
    operator: Literal["eq", "in"]
    values: list[TeamCode] = Field(
        min_length=1,
        description="Canonical team codes used by processed nflverse data.",
    )


class PositionFilter(PlannerModel):
    field: Literal[EntityFilterField.POSITION]
    operator: Literal["eq", "in"]
    values: list[PlayerPosition] = Field(min_length=1)


class PositionGroupFilter(PlannerModel):
    field: Literal[EntityFilterField.POSITION_GROUP]
    operator: Literal["eq", "in"]
    values: list[PositionGroup] = Field(min_length=1)


class RosterStatusFilter(PlannerModel):
    field: Literal[EntityFilterField.ROSTER_STATUS]
    operator: Literal["eq", "in"]
    values: list[RosterStatus] = Field(min_length=1)


class DepthChartPositionFilter(PlannerModel):
    field: Literal[EntityFilterField.DEPTH_CHART_POSITION]
    operator: Literal["eq", "in"]
    values: list[DepthChartPosition] = Field(min_length=1)


class FormationFilter(PlannerModel):
    field: Literal[EntityFilterField.FORMATION] = Field(
        description=(
            "Normalized depth-chart side: Offense, Defense, or Special Teams. "
            "This is not a personnel package or a measure of snaps, usage, or "
            "workload. It requires QueryPlan.season."
        )
    )
    operator: Literal["eq", "in"]
    values: list[Formation] = Field(min_length=1)


class ECRPositionFilter(PlannerModel):
    field: Literal[EntityFilterField.ECR_POSITION]
    operator: Literal["eq", "in"]
    values: list[ECRPosition] = Field(min_length=1)


class ECRScoringFormatFilter(PlannerModel):
    field: Literal[EntityFilterField.ECR_SCORING_FORMAT]
    operator: Literal["eq", "in"]
    values: list[ECRScoringFormat] = Field(min_length=1)


class ECRLeagueFormatFilter(PlannerModel):
    field: Literal[EntityFilterField.ECR_LEAGUE_FORMAT]
    operator: Literal["eq", "in"]
    values: list[ECRLeagueFormat] = Field(min_length=1)


class OpponentFilter(PlannerModel):
    field: Literal[EntityFilterField.OPPONENT]
    operator: Literal["eq", "in"]
    values: list[TeamCode] = Field(min_length=1)


class OpenTextFilter(PlannerModel):
    field: Literal[EntityFilterField.COLLEGE]
    operator: Literal["eq", "in"]
    values: list[str] = Field(min_length=1)


class NumericFilter(PlannerModel):
    field: Literal[
        EntityFilterField.ROOKIE_SEASON,
        EntityFilterField.DRAFT_YEAR,
        EntityFilterField.DRAFT_ROUND,
        EntityFilterField.DRAFT_PICK,
        EntityFilterField.HEIGHT,
        EntityFilterField.WEIGHT,
        EntityFilterField.LAST_SEASON,
        EntityFilterField.YEARS_EXPERIENCE,
        EntityFilterField.DEPTH_RANK,
        EntityFilterField.ECR_RANK,
        EntityFilterField.GAMES,
        EntityFilterField.FANTASY_POINTS,
        EntityFilterField.FANTASY_POINTS_PPR,
        EntityFilterField.TARGETS,
        EntityFilterField.CARRIES,
        EntityFilterField.RECEPTIONS,
        EntityFilterField.OFFENSE_SNAP_PCT,
    ] = Field(
        description=(
            "Numeric database field. `last_season` means the latest season "
            "recorded for the player's career in the dataset. It is not the "
            "season requested by the user or shorthand for the previous "
            "season. Use it only when the question explicitly constrains that "
            "property."
        )
    )
    operator: Literal["eq", "in", "gte", "lte"]
    values: list[str] = Field(
        min_length=1,
        description="Numeric values encoded as strings for database coercion."
    )


class ConferenceFilter(PlannerModel):
    field: Literal[EntityFilterField.CONFERENCE]
    operator: Literal["eq", "in"]
    values: list[Conference] = Field(min_length=1)


class DivisionFilter(PlannerModel):
    field: Literal[EntityFilterField.DIVISION]
    operator: Literal["eq", "in"]
    values: list[Division] = Field(min_length=1)


PlayerFilter = (
    TeamCodeFilter
    | PositionFilter
    | PositionGroupFilter
    | RosterStatusFilter
    | DepthChartPositionFilter
    | FormationFilter
    | ECRPositionFilter
    | ECRScoringFormatFilter
    | ECRLeagueFormatFilter
    | OpponentFilter
    | OpenTextFilter
    | NumericFilter
    | ConferenceFilter
    | DivisionFilter
)
TeamFilter = ConferenceFilter | DivisionFilter
EntityFilter = PlayerFilter | TeamFilter


class FilteredSelectorModel(PlannerModel):
    """Shared hard/soft filter boundary with legacy input migration."""

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_filters(cls, value):
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        legacy = migrated.pop("filters", None)
        if "hard_filters" not in migrated:
            migrated["hard_filters"] = legacy if legacy is not None else []
        if "soft_filters" not in migrated:
            migrated["soft_filters"] = []
        return migrated

    @property
    def filters(self):
        """Compatibility alias; execution code must use hard_filters."""
        return self.hard_filters


class PlayerSelector(FilteredSelectorModel):
    entity_type: Literal["player"]
    reference_text: str = Field(
        min_length=1,
        description=(
            "Exact phrase copied from the question that refers to this player "
            "or player group."
        )
    )
    names: list[str] = Field(
        max_length=1,
        description=(
            "Exactly one best official full-name candidate for a single-player "
            "reference; empty only for a player group."
        )
    )
    identity_confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence from 0 to 1 in the official name candidate.",
    )
    resolution_basis: PlayerResolutionBasis = Field(
        description=(
            "How the candidate was obtained: official name as written, broadly "
            "recognized alias or name variant, identity derived from surrounding "
            "context, best unsupported inference, or not applicable to a group."
        )
    )
    hard_filters: list[PlayerFilter] = Field(
        description=(
            "Prompt-grounded objective constraints allowed to exclude players "
            "or reports."
        )
    )
    soft_filters: list[PlayerFilter] = Field(
        description=(
            "Optional inferred structured context. Never used as an exclusion "
            "constraint."
        )
    )
    semantic_qualifiers: list[str]


class TeamSelector(FilteredSelectorModel):
    entity_type: Literal["team"]
    names: list[TeamCode] = Field(
        description="Canonical codes, such as SEA, NYJ, PHI, and SF."
    )
    hard_filters: list[TeamFilter] = Field(
        description="Prompt-grounded constraints allowed to exclude teams."
    )
    soft_filters: list[TeamFilter] = Field(
        description=(
            "Optional inferred team context. Never used as an exclusion "
            "constraint."
        )
    )
    semantic_qualifiers: list[str]


EntitySelector = PlayerSelector | TeamSelector


class QueryPlan(PlannerModel):
    @model_validator(mode="before")
    @classmethod
    def _migrate_soft_team_mentions(cls, value):
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "soft_team_mentions" not in migrated:
            migrated["soft_team_mentions"] = []
        return migrated

    semantic_query: str
    keyword_query: str
    intent: Literal[
        "fact",
        "current_status",
        "timeline",
        "comparison",
        "projection",
        "yes_no",
        "other",
    ]
    player_mentions: list[str]
    team_mentions: list[TeamCode] = Field(
        description=(
            "Canonical codes only for teams explicitly mentioned or "
            "unambiguously normalized from the question."
        )
    )
    soft_team_mentions: list[TeamCode] = Field(
        description=(
            "Optional inferred team context that must never constrain lookup "
            "or exclude evidence."
        )
    )
    negative_focus: list[str]
    entity_selectors: list[EntitySelector]
    season: int | None
    week: int | None
    temporal_mode: Literal[
        "none",
        "latest",
        "current",
        "before",
        "after",
        "between",
        "timeline",
    ]
    start_date: str | None
    end_date: str | None
    needs_baseline: bool
    evidence_strategy: Literal[
        "single_document",
        "multiple_documents",
        "per_entity",
        "timeline",
    ]


@dataclass(frozen=True)
class QueryPlanResult:
    plan: QueryPlan
    model: str
    cached: bool
    input_tokens: int
    output_tokens: int
    scope_retries: int = 0
    scope_issues: tuple[str, ...] = ()


def _scope_review_issues(plan: QueryPlan) -> tuple[str, ...]:
    """Identify plans that may have detached a team from a player group.

    This only requests a model review. It does not assume a relationship or
    mutate selectors because separate team and league-wide player subjects can
    be valid in comparison questions.
    """
    has_team_context = bool(plan.team_mentions) or any(
        selector.entity_type == "team" for selector in plan.entity_selectors
    )
    if not has_team_context:
        return ()

    issues = []
    for index, selector in enumerate(plan.entity_selectors):
        if selector.entity_type != "player" or selector.names:
            continue
        has_team_scope = any(
            item.field in {"team", "conference", "division"}
            for item in selector.hard_filters
        )
        if not has_team_scope:
            issues.append(
                f"player-group selector {index} has no team, division, or "
                "conference scope while the "
                "plan contains separate team context"
            )
    return tuple(issues)


class QueryPlanner:
    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        model: str = DEFAULT_PLANNER_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.model = model
        self.client = client

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.index_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS query_plans (
                        query_hash TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        planning_date TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (query_hash, model, prompt_version)
                    )
                    """
                )
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _query_hash(query: str, planning_date: date) -> str:
        cache_input = f"{planning_date.isoformat()}\n{query.strip()}"
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()

    def _cached_plan(
        self,
        query_hash: str,
    ) -> QueryPlan | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT plan_json
                FROM query_plans
                WHERE query_hash = ? AND model = ? AND prompt_version = ?
                """,
                (query_hash, self.model, PLANNER_PROMPT_VERSION),
            ).fetchone()
        if row is None:
            return None
        return QueryPlan.model_validate_json(row["plan_json"])

    def _cache_plan(
        self,
        query_hash: str,
        planning_date: date,
        plan: QueryPlan,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO query_plans (
                    query_hash, model, prompt_version, planning_date, plan_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query_hash,
                    self.model,
                    PLANNER_PROMPT_VERSION,
                    planning_date.isoformat(),
                    plan.model_dump_json(),
                ),
            )
            connection.commit()

    def _request_plan(
        self,
        client: OpenAI,
        query: str,
        current_date: date,
        *,
        previous_plan: QueryPlan | None = None,
        scope_issues: tuple[str, ...] = (),
    ) -> tuple[QueryPlan, int, int]:
        messages = [
            {"role": "system", "content": PLANNER_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Current date: {current_date.isoformat()}\n"
                    f"Question: {query}"
                ),
            },
        ]
        if previous_plan is not None:
            issue_text = "\n".join(f"- {issue}" for issue in scope_issues)
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": previous_plan.model_dump_json(),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Review the selector relationships in the previous "
                            "plan and return a complete plan. Potential issue:\n"
                            f"{issue_text}\n"
                            "If the team constrains the player group, place the "
                            "hard team filter inside that player selector. If the "
                            "team and player group are independent subjects, "
                            "preserve them independently. Do not merge scopes "
                            "automatically."
                        ),
                    },
                ]
            )

        response = client.responses.parse(
            model=self.model,
            reasoning={"effort": "none"},
            input=messages,
            text_format=QueryPlan,
        )
        plan = response.output_parsed
        if plan is None:
            raise RuntimeError("Query planner returned no structured plan")

        usage = response.usage
        return (
            plan,
            usage.input_tokens if usage else 0,
            usage.output_tokens if usage else 0,
        )

    def plan(
        self,
        query: str,
        *,
        planning_date: date | None = None,
        use_cache: bool = True,
    ) -> QueryPlanResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Planner query must not be empty")

        current_date = planning_date or date.today()
        query_hash = self._query_hash(normalized_query, current_date)

        if use_cache:
            cached_plan = self._cached_plan(query_hash)
            if cached_plan is not None:
                return QueryPlanResult(
                    plan=cached_plan,
                    model=self.model,
                    cached=True,
                    input_tokens=0,
                    output_tokens=0,
                )

        try:
            client = self.client or OpenAI()
            plan, input_tokens, output_tokens = self._request_plan(
                client,
                normalized_query,
                current_date,
            )
            scope_retries = 0
            scope_issues = _scope_review_issues(plan)
            if scope_issues:
                plan, retry_input_tokens, retry_output_tokens = self._request_plan(
                    client,
                    normalized_query,
                    current_date,
                    previous_plan=plan,
                    scope_issues=scope_issues,
                )
                input_tokens += retry_input_tokens
                output_tokens += retry_output_tokens
                scope_retries = 1
                scope_issues = _scope_review_issues(plan)
        except Exception as error:
            raise RuntimeError(f"Query planner request failed: {error}") from error

        self._cache_plan(query_hash, current_date, plan)
        return QueryPlanResult(
            plan=plan,
            model=self.model,
            cached=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            scope_retries=scope_retries,
            scope_issues=scope_issues,
        )
