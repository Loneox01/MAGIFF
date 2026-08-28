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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from prompts import DIRECT_REPORT_PLANNER_INSTRUCTIONS

from ..config import DEFAULT_INDEX_PATH, DEFAULT_PLANNER_MODEL
from .lookups import (
    ContextScopePolicy,
    PLAYER_ANCHORED_LOOKUP_OPERATIONS,
    PLAYER_RANKING_LOOKUP_OPERATIONS,
    StructuredLookup,
    TEAM_ANCHORED_LOOKUP_OPERATIONS,
    TEAM_RANKING_LOOKUP_OPERATIONS,
)
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


PLANNER_PROMPT_VERSION = "25"


def _nfl_season_for_date(current_date: date) -> int:
    """Return the season containing the supplied date.

    NFL seasons cross the calendar-year boundary. January and February belong
    to the season that began in the prior calendar year; the new league-year
    context applies from March onward.
    """
    return current_date.year if current_date.month >= 3 else current_date.year - 1


def _planner_runtime_context(model: str, current_date: date) -> str:
    """Build dynamic temporal context for the selected direct planner model."""
    lines = [f"Current date: {current_date.isoformat()}"]
    if "luna" in model.lower():
        lines.append(f"Current NFL season: {_nfl_season_for_date(current_date)}")
    return "\n".join(lines)


def _normalize_relative_week(
    plan: "QueryPlan",
    current_date: date,
) -> "QueryPlan":
    """Ensure a relative week has the season required by downstream tools."""
    if plan.week is None or plan.season is not None:
        return plan
    return QueryPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "season": _nfl_season_for_date(current_date),
        }
    )


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


class TemporalBasis(StrEnum):
    """Provenance for report-time preferences and boundaries."""

    EXPLICIT_USER = "explicit_user"
    NORMALIZED_USER = "normalized_user"
    INFERRED = "inferred"
    NOT_APPLICABLE = "not_applicable"


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
    structured_lookups: list[StructuredLookup] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Bounded read-only structured lookups that enrich this explicit "
            "target. Their results may add query terms and reranker context but "
            "must not silently become global exclusion constraints."
        ),
    )


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
    structured_lookups: list[StructuredLookup] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Bounded read-only structured lookups that enrich this explicit "
            "target without silently changing its required scope."
        ),
    )


EntitySelector = PlayerSelector | TeamSelector


class ContextRelation(StrEnum):
    SAME_TEAM = "same_team"
    ENVIRONMENT = "environment"
    DEPENDENCY = "dependency"
    MATCHUP = "matchup"
    COMPARISON = "comparison"


class ContextRequest(PlannerModel):
    """A bounded request for indirect evidence around a resolved subject."""

    anchor_selector_index: int = Field(
        ge=0,
        description=(
            "Index of a specific player, an objectively bounded player group, "
            "or a bounded team selector in entity_selectors."
        ),
    )
    relation: ContextRelation = Field(
        description=(
            "Broad reason indirect evidence is related to the anchor. This is "
            "provenance for retrieval and reranking, not a database predicate."
        )
    )
    semantic_query: str = Field(
        min_length=1,
        description="Natural-language query for the indirect evidence branch.",
    )
    keyword_query: str = Field(
        min_length=1,
        description="Compact keyword query for the indirect evidence branch.",
    )
    semantic_qualifiers: list[str]
    scope_policy: ContextScopePolicy = Field(
        default=ContextScopePolicy.ANCHOR_TEAMS,
        description=(
            "Branch-local metadata scope. Derived scope never constrains the "
            "target branch or any other contextual branch."
        ),
    )
    structured_lookups: list[StructuredLookup] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Read-only structured operations needed to discover related "
            "entities, relationships, or compact facts for this branch."
        ),
    )


class QueryPlan(PlannerModel):
    @model_validator(mode="before")
    @classmethod
    def _migrate_added_fields(cls, value):
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "soft_team_mentions" not in migrated:
            migrated["soft_team_mentions"] = []
        if "context_requests" not in migrated:
            migrated["context_requests"] = []
        if "temporal_basis" not in migrated:
            migrated["temporal_basis"] = TemporalBasis.NOT_APPLICABLE
        return migrated

    @model_validator(mode="after")
    def _validate_context_requests(self):
        if len(self.context_requests) > 3:
            raise ValueError("A query plan may contain at most three context requests")
        for selector in self.entity_selectors:
            if (
                selector.entity_type == "player"
                and not selector.names
                and selector.resolution_basis
                == PlayerResolutionBasis.NOT_APPLICABLE
                and not selector.hard_filters
            ):
                raise ValueError(
                    "Player groups require objective hard filters that bound "
                    "their membership. A nickname or other uncertain reference "
                    "to one individual must instead supply the best official "
                    "name hypothesis, confidence, and identity resolution basis."
                )
        for request in self.context_requests:
            if request.anchor_selector_index >= len(self.entity_selectors):
                raise ValueError("Context request anchor selector is out of range")
            anchor = self.entity_selectors[request.anchor_selector_index]
            if (
                anchor.entity_type == "player"
                and not anchor.names
                and not anchor.hard_filters
            ):
                raise ValueError(
                    "Player-group context requests require objective hard "
                    "filters that bound the anchor group"
                )
            if (
                anchor.entity_type == "team"
                and not anchor.names
                and not anchor.hard_filters
            ):
                raise ValueError(
                    "Team context requests must anchor to named teams or a "
                    "prompt-grounded team group"
                )
            has_scope_lookup = any(
                lookup.purpose
                in {
                    "resolve_relationship",
                    "expand_candidates",
                }
                for lookup in request.structured_lookups
            )
            if (
                request.scope_policy
                in {
                    ContextScopePolicy.LOOKUP_ENTITIES,
                    ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS,
                }
                and not has_scope_lookup
            ):
                raise ValueError(
                    "lookup-derived context scope requires a relationship or "
                    "candidate-expansion lookup"
                )
        lookup_ids = [
            lookup.lookup_id
            for selector in self.entity_selectors
            for lookup in selector.structured_lookups
        ] + [
            lookup.lookup_id
            for request in self.context_requests
            for lookup in request.structured_lookups
        ]
        if len(lookup_ids) > 8:
            raise ValueError("A query plan may contain at most eight lookups")
        if len(lookup_ids) != len(set(lookup_ids)):
            raise ValueError("Structured lookup IDs must be unique within a plan")

        team_anchored_lookups = [
            lookup
            for selector in self.entity_selectors
            for lookup in selector.structured_lookups
            if lookup.operation in TEAM_ANCHORED_LOOKUP_OPERATIONS
        ] + [
            lookup
            for request in self.context_requests
            for lookup in request.structured_lookups
            if lookup.operation in TEAM_ANCHORED_LOOKUP_OPERATIONS
        ]
        if team_anchored_lookups and self.season is None:
            raise ValueError(
                "Team-anchored structured lookups require QueryPlan.season"
            )
        if any(
            lookup.season != self.season for lookup in team_anchored_lookups
        ):
            raise ValueError(
                "Team-anchored structured lookup seasons must match "
                "QueryPlan.season"
            )
        for selector in self.entity_selectors:
            if any(
                lookup.purpose == "resolve_relationship"
                for lookup in selector.structured_lookups
            ):
                raise ValueError(
                    "Relationship-resolution lookups belong in a context "
                    "request, not an explicit target"
                )
            operations = {
                lookup.operation for lookup in selector.structured_lookups
            }
            if selector.entity_type == "player" and selector.names:
                invalid = operations - PLAYER_ANCHORED_LOOKUP_OPERATIONS
                if invalid:
                    raise ValueError(
                        "Specific-player target lookups must operate on that "
                        f"player; use context for related scope: {sorted(invalid)}"
                    )
            elif selector.entity_type == "player":
                allowed = (
                    PLAYER_RANKING_LOOKUP_OPERATIONS
                    | TEAM_ANCHORED_LOOKUP_OPERATIONS
                )
                invalid = operations - allowed
                if invalid:
                    raise ValueError(
                        "Player-group target has incompatible structured "
                        f"lookups: {sorted(invalid)}"
                    )
            else:
                allowed = (
                    TEAM_ANCHORED_LOOKUP_OPERATIONS
                    | TEAM_RANKING_LOOKUP_OPERATIONS
                )
                invalid = operations - allowed
                if invalid:
                    raise ValueError(
                        "Team target has incompatible structured lookups: "
                        f"{sorted(invalid)}"
                    )

        for request in self.context_requests:
            anchor = self.entity_selectors[request.anchor_selector_index]
            is_group_anchor = (
                anchor.entity_type == "team"
                or (anchor.entity_type == "player" and not anchor.names)
            )
            if not is_group_anchor:
                continue
            invalid = {
                lookup.operation
                for lookup in request.structured_lookups
                if lookup.operation in PLAYER_ANCHORED_LOOKUP_OPERATIONS
            }
            if invalid:
                raise ValueError(
                    "Grouped context cannot execute player-specific "
                    f"lookups: {sorted(invalid)}"
                )
        return self

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
    entity_selectors: list[EntitySelector] = Field(
        max_length=8,
        description=(
            "Independently retrieved target branches. Filters within one "
            "selector constrain that target only."
        ),
    )
    context_requests: list[ContextRequest] = Field(
        description=(
            "Optional bounded retrieval branches for materially related "
            "indirect evidence around a resolved target."
        )
    )
    season: int | None = Field(
        description=(
            "NFL season. Required whenever week is not null; an unqualified "
            "relative week uses the season containing the supplied current date."
        )
    )
    week: int | None = Field(
        description="NFL week within season, or null when no week is requested."
    )
    temporal_mode: Literal[
        "none",
        "latest",
        "current",
        "before",
        "after",
        "between",
        "timeline",
    ]
    temporal_basis: TemporalBasis = Field(
        default=TemporalBasis.NOT_APPLICABLE,
        description=(
            "Origin of the requested report-time scope. explicit_user is "
            "reserved for a closed start/end range actually stated by the "
            "user; normalized_user covers partial or relative periods; "
            "inferred covers recency preferences; not_applicable means no "
            "temporal preference."
        ),
    )
    start_date: str | None
    end_date: str | None
    needs_baseline: bool
    evidence_strategy: Literal[
        "single_document",
        "multiple_documents",
        "per_entity",
        "timeline",
    ]


class DirectQueryPlan(PlannerModel):
    """Luna's direct-only output before contextual expansion.

    ``QueryPlan`` remains the stable executor contract. Keeping a separate model
    here prevents the direct planner from seeing or attempting to fill context
    fields while reusing the mature combined-plan validation at the merge
    boundary.
    """

    @model_validator(mode="before")
    @classmethod
    def _migrate_added_fields(cls, value):
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "soft_team_mentions" not in migrated:
            migrated["soft_team_mentions"] = []
        if "temporal_basis" not in migrated:
            migrated["temporal_basis"] = TemporalBasis.NOT_APPLICABLE
        migrated.pop("context_requests", None)
        return migrated

    @model_validator(mode="after")
    def _validate_as_combined_plan(self):
        QueryPlan.model_validate(
            {**self.model_dump(), "context_requests": []}
        )
        return self

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
    entity_selectors: list[EntitySelector] = Field(
        max_length=8,
        description=(
            "Independently retrieved direct target branches. Filters within "
            "one selector constrain that target only."
        ),
    )
    season: int | None = Field(
        description=(
            "NFL season. Required whenever week is not null; an unqualified "
            "relative week uses the season containing the supplied current date."
        )
    )
    week: int | None = Field(
        description="NFL week within season, or null when no week is requested."
    )
    temporal_mode: Literal[
        "none",
        "latest",
        "current",
        "before",
        "after",
        "between",
        "timeline",
    ]
    temporal_basis: TemporalBasis = Field(
        default=TemporalBasis.NOT_APPLICABLE,
        description=(
            "Origin of the requested report-time scope. explicit_user is "
            "reserved for a closed start/end range actually stated by the "
            "user; normalized_user covers partial or relative periods; "
            "inferred covers recency preferences; not_applicable means no "
            "temporal preference."
        ),
    )
    start_date: str | None
    end_date: str | None
    needs_baseline: bool
    evidence_strategy: Literal[
        "single_document",
        "multiple_documents",
        "per_entity",
        "timeline",
    ]

    def combined(self) -> QueryPlan:
        return QueryPlan.model_validate(
            {**self.model_dump(), "context_requests": []}
        )


@dataclass(frozen=True)
class QueryPlanResult:
    plan: QueryPlan
    model: str
    cached: bool
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    attempts: int = 1
    retried: bool = False
    retry_reason: str | None = None


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
    def _query_hash(
        query: str,
        planning_date: date,
        source_question: str,
    ) -> str:
        cache_input = (
            f"{planning_date.isoformat()}\n{source_question.strip()}\n"
            f"{query.strip()}"
        )
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
        source_question: str,
        current_date: date,
        *,
        correction_error: str | None = None,
    ) -> tuple[QueryPlan, int, int, int]:
        user_content = (
            f"{_planner_runtime_context(self.model, current_date)}\n"
            "Original user question (authoritative for hard constraints): "
            f"{source_question}\n"
            "Report retrieval query (semantic guidance only): "
            f"{query}"
        )
        if correction_error is not None:
            user_content += (
                "\n\nCorrection required: The previous direct retrieval plan "
                "was incompatible with the schema. Return a complete corrected "
                "plan. Preserve the question and change only what is necessary "
                "to satisfy the validation contract.\n"
                f"Validation error: {correction_error[:2400]}"
            )
        messages = [
            {
                "role": "system",
                "content": DIRECT_REPORT_PLANNER_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = client.responses.parse(
            model=self.model,
            reasoning={"effort": "none"},
            input=messages,
            text_format=DirectQueryPlan,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Query planner returned no structured plan")
        if isinstance(parsed, DirectQueryPlan):
            plan = parsed.combined()
        else:
            # Keeps injected test clients and older local fixtures compatible
            # while the live structured-output contract remains direct-only.
            raw = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
            plan = DirectQueryPlan.model_validate(raw).combined()
        plan = _normalize_relative_week(plan, current_date)

        usage = response.usage
        cached_tokens = 0
        if usage is not None:
            details = getattr(usage, "input_tokens_details", None)
            cached_tokens = getattr(details, "cached_tokens", 0) or 0
        return (
            plan,
            usage.input_tokens if usage else 0,
            cached_tokens,
            usage.output_tokens if usage else 0,
        )

    def plan(
        self,
        query: str,
        *,
        source_question: str | None = None,
        planning_date: date | None = None,
        use_cache: bool = True,
    ) -> QueryPlanResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Planner query must not be empty")
        normalized_source_question = (source_question or query).strip()
        if not normalized_source_question:
            raise ValueError("Planner source question must not be empty")

        current_date = planning_date or date.today()
        query_hash = self._query_hash(
            normalized_query,
            current_date,
            normalized_source_question,
        )

        if use_cache:
            cached_plan = self._cached_plan(query_hash)
            if cached_plan is not None:
                return QueryPlanResult(
                    plan=cached_plan,
                    model=self.model,
                    cached=True,
                    input_tokens=0,
                    output_tokens=0,
                    attempts=0,
                )

        retry_reason = None
        attempts = 1
        try:
            client = self.client or OpenAI()
            try:
                (
                    plan,
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                ) = self._request_plan(
                    client,
                    normalized_query,
                    normalized_source_question,
                    current_date,
                )
            except ValidationError as error:
                retry_reason = str(error)
                attempts += 1
                (
                    plan,
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                ) = self._request_plan(
                    client,
                    normalized_query,
                    normalized_source_question,
                    current_date,
                    correction_error=retry_reason,
                )
        except Exception as error:
            raise RuntimeError(f"Query planner request failed: {error}") from error

        self._cache_plan(query_hash, current_date, plan)
        return QueryPlanResult(
            plan=plan,
            model=self.model,
            cached=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            attempts=attempts,
            retried=retry_reason is not None,
            retry_reason=retry_reason,
        )
