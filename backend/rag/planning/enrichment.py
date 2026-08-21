"""Execute bounded structured lookups for target and contextual retrieval.

This module is the deterministic boundary between model-authored retrieval plans
and the existing NFL tools. The model selects an allowlisted operation and
declares why it is needed; application code supplies grounded player IDs or team
codes, validates the public tool contract, caps outputs, and records provenance.
Lookup results may enrich queries or a single contextual branch, but never mutate
global target constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from repositories import nfl_supabase as repository
from tools import nfl as nfl_tools

from .lookups import (
    ECRRankingLookup,
    LookupPurpose,
    PlayerFormulaRankingLookup,
    PlayerSeasonStatsLookup,
    PlayerSnapCountsLookup,
    PlayerWeeklyStatsLookup,
    StructuredLookup,
    TeamDepthChartLookup,
    TeamFormulaRankingLookup,
    TeamRosterLookup,
    TeamScheduleLookup,
    TeamWeeklyStatsLookup,
)
from .planner import QueryPlan
from .resolver import ResolutionResult, ResolvedEntity


MAX_LOOKUP_ROWS = 20
MAX_LOOKUP_FACTS = 8
MAX_QUERY_TERMS = 12


class EnrichmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LookupExecution(EnrichmentModel):
    lookup_id: str
    operation: str
    purpose: LookupPurpose
    status: Literal["resolved", "empty", "error"]
    entities: list[ResolvedEntity]
    teams: list[str]
    query_terms: list[str]
    facts: list[dict[str, Any]]
    error: str | None
    fallback_used: bool = False
    fallback_reason: str | None = None


class TargetEnrichment(EnrichmentModel):
    selector_index: int
    lookups: list[LookupExecution]

    @property
    def query_terms(self) -> list[str]:
        return list(
            dict.fromkeys(
                term
                for lookup in self.lookups
                if lookup.purpose != LookupPurpose.RERANKER_CONTEXT
                for term in lookup.query_terms
            )
        )


class ContextEnrichment(EnrichmentModel):
    request_index: int
    lookups: list[LookupExecution]

    @property
    def entities(self) -> list[ResolvedEntity]:
        return _unique_entities(
            entity for lookup in self.lookups for entity in lookup.entities
        )

    @property
    def teams(self) -> list[str]:
        return list(
            dict.fromkeys(team for lookup in self.lookups for team in lookup.teams)
        )

    @property
    def scope_entities(self) -> list[ResolvedEntity]:
        return _unique_entities(
            entity
            for lookup in self.lookups
            if lookup.status == "resolved"
            and lookup.purpose
            in {
                LookupPurpose.RESOLVE_RELATIONSHIP,
                LookupPurpose.EXPAND_CANDIDATES,
            }
            for entity in lookup.entities
        )

    @property
    def scope_teams(self) -> list[str]:
        return list(
            dict.fromkeys(
                team
                for lookup in self.lookups
                if lookup.status == "resolved"
                and lookup.purpose
                in {
                    LookupPurpose.RESOLVE_RELATIONSHIP,
                    LookupPurpose.EXPAND_CANDIDATES,
                }
                for team in lookup.teams
            )
        )

    @property
    def query_terms(self) -> list[str]:
        return list(
            dict.fromkeys(
                term
                for lookup in self.lookups
                if lookup.purpose != LookupPurpose.RERANKER_CONTEXT
                for term in lookup.query_terms
            )
        )


class StructuredEnrichment(EnrichmentModel):
    targets: list[TargetEnrichment] = Field(default_factory=list)
    contexts: list[ContextEnrichment] = Field(default_factory=list)

    def target(self, selector_index: int) -> TargetEnrichment | None:
        return next(
            (
                item
                for item in self.targets
                if item.selector_index == selector_index
            ),
            None,
        )

    def context(self, request_index: int) -> ContextEnrichment | None:
        return next(
            (
                item
                for item in self.contexts
                if item.request_index == request_index
            ),
            None,
        )

    @property
    def errors(self) -> list[str]:
        return [
            f"{lookup.lookup_id}: {lookup.error}"
            for group in [*self.targets, *self.contexts]
            for lookup in group.lookups
            if lookup.error
        ]


def _unique_entities(values) -> list[ResolvedEntity]:
    unique: dict[tuple[str, str], ResolvedEntity] = {}
    for entity in values:
        unique[(entity.entity_type, entity.entity_id)] = entity
    return list(unique.values())


def _bounded_rows(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        nested = value.get("results")
        rows = nested if isinstance(nested, list) else [value]
    else:
        return []
    return [dict(row) for row in rows[:MAX_LOOKUP_ROWS] if isinstance(row, dict)]


def _fact_rows(value: object) -> list[dict[str, Any]]:
    """Keep enough structured evidence for reranking without copying large tables."""
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        metadata = {
            key: item
            for key, item in value.items()
            if key != "results" and not isinstance(item, (list, dict))
        }
        facts = [{"summary": metadata}] if metadata else []
        facts.extend(
            {"row": dict(row)}
            for row in value["results"]
            if isinstance(row, dict)
        )
        return facts[:MAX_LOOKUP_FACTS]
    return _bounded_rows(value)[:MAX_LOOKUP_FACTS]


def _player_entity(row: dict[str, Any], names: dict[str, str]) -> ResolvedEntity | None:
    player_id = row.get("player_id")
    if not player_id:
        return None
    display_name = row.get("display_name") or row.get("player_name") or names.get(
        str(player_id)
    )
    if not display_name:
        return None
    team = row.get("team") or row.get("latest_team") or row.get("last_team")
    return ResolvedEntity(
        entity_type="player",
        entity_id=str(player_id),
        display_name=str(display_name),
        team=str(team) if team else None,
        position=row.get("position"),
        position_group=row.get("position_group"),
        roster_status=row.get("status"),
    )


def _team_entities(team_codes: list[str]) -> list[ResolvedEntity]:
    codes = list(dict.fromkeys(code for code in team_codes if code))
    return [
        ResolvedEntity(
            entity_type="team",
            entity_id=code,
            display_name=code,
            team=code,
            position=None,
        )
        for code in codes
    ]


def _rows_to_context(
    value: object,
    *,
    anchor_players: list[ResolvedEntity],
) -> tuple[list[ResolvedEntity], list[str], list[str], list[dict[str, Any]]]:
    rows = _bounded_rows(value)
    player_ids = list(
        dict.fromkeys(str(row["player_id"]) for row in rows if row.get("player_id"))
    )
    names = {
        player.entity_id: player.display_name
        for player in anchor_players
    }
    missing_names = [player_id for player_id in player_ids if player_id not in names]
    if missing_names:
        names.update(repository.get_player_names(missing_names))
    players = [
        entity
        for entity in (_player_entity(row, names) for row in rows)
        if entity is not None
    ]

    teams: list[str] = []
    for row in rows:
        for key in (
            "team",
            "latest_team",
            "last_team",
            "opponent",
            "opponent_team",
            "home_team",
            "away_team",
            "ecr_team",
            "result_team",
        ):
            team = row.get(key)
            if team:
                teams.append(str(team))
    teams = list(dict.fromkeys(teams))

    team_entities = _team_entities(teams)
    entities = _unique_entities([*players, *team_entities])
    query_terms = list(
        dict.fromkeys(
            [
                *(entity.display_name for entity in players),
                *teams,
                *(entity.display_name for entity in team_entities),
            ]
        )
    )[:MAX_QUERY_TERMS]
    return entities, teams, query_terms, _fact_rows(value)


class StructuredLookupGateway(Protocol):
    def execute(
        self,
        lookup: StructuredLookup,
        *,
        players: list[ResolvedEntity],
        teams: list[str],
    ) -> LookupExecution: ...


@dataclass(frozen=True)
class StructuredToolGateway:
    """Invoke the existing read-only NFL tool behavior with grounded anchors."""

    current_date: date | None = None

    def execute(
        self,
        lookup: StructuredLookup,
        *,
        players: list[ResolvedEntity],
        teams: list[str],
    ) -> LookupExecution:
        try:
            value = self._call(lookup, players=players, teams=teams)
            entities, related_teams, terms, facts = _rows_to_context(
                value,
                anchor_players=players,
            )
            has_value = bool(_bounded_rows(value))
            fallback_used = bool(
                isinstance(value, dict) and value.get("fallback_used")
            )
            fallback_reason = (
                str(value["fallback_reason"])
                if isinstance(value, dict) and value.get("fallback_reason")
                else None
            )
            return LookupExecution(
                lookup_id=lookup.lookup_id,
                operation=lookup.operation,
                purpose=lookup.purpose,
                status="resolved" if has_value else "empty",
                entities=entities,
                teams=related_teams,
                query_terms=terms,
                facts=facts,
                error=None,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        except Exception as error:
            return LookupExecution(
                lookup_id=lookup.lookup_id,
                operation=lookup.operation,
                purpose=lookup.purpose,
                status="error",
                entities=[],
                teams=[],
                query_terms=[],
                facts=[],
                error=str(error),
                fallback_used=False,
                fallback_reason=None,
            )

    @staticmethod
    def _require_players(players: list[ResolvedEntity], operation: str) -> None:
        if not players:
            raise ValueError(f"{operation} requires a resolved player anchor")

    @staticmethod
    def _require_teams(teams: list[str], operation: str) -> None:
        if not teams:
            raise ValueError(f"{operation} requires a grounded team scope")

    def _current_nfl_season(self) -> int:
        current_date = self.current_date or date.today()
        return (
            current_date.year
            if current_date.month >= 3
            else current_date.year - 1
        )

    def _team_depth_chart(
        self,
        lookup: TeamDepthChartLookup,
        teams: list[str],
    ) -> object:
        rows: list[dict[str, Any]] = []
        fallback_teams: list[str] = []
        for team in teams:
            team_rows = nfl_tools.get_team_depth_chart(
                team,
                lookup.season,
                lookup.week,
                lookup.position,
            )
            if (
                not team_rows
                and lookup.week is not None
                and lookup.season == self._current_nfl_season()
            ):
                team_rows = nfl_tools.get_team_depth_chart(
                    team,
                    lookup.season,
                    None,
                    lookup.position,
                )
                if team_rows:
                    fallback_teams.append(team)
            rows.extend(team_rows)

        if not fallback_teams:
            return rows
        return {
            "results": rows,
            "fallback_used": True,
            "fallback_reason": (
                "Requested current-season weekly depth chart was empty; used "
                "the latest current snapshot instead."
            ),
            "requested_week": lookup.week,
            "fallback_team_count": len(fallback_teams),
        }

    def _team_roster(
        self,
        lookup: TeamRosterLookup,
        teams: list[str],
    ) -> object:
        rows: list[dict[str, Any]] = []
        fallback_teams: list[str] = []
        fallback_reasons: list[str] = []
        sources: list[str] = []
        for team in teams:
            result = nfl_tools.get_team_roster(
                team,
                lookup.season,
                lookup.week,
                lookup.position,
                lookup.status,
                current_date=self.current_date,
            )
            team_rows = result.get("results", [])
            rows.extend(team_rows)
            source = result.get("source_snapshot")
            if source:
                sources.append(str(source))
            if result.get("fallback_used"):
                fallback_teams.append(team)
                reason = result.get("fallback_reason")
                if reason:
                    fallback_reasons.append(str(reason))

        return {
            "results": rows,
            "source_snapshot": ",".join(dict.fromkeys(sources)) or None,
            "fallback_used": bool(fallback_teams),
            "fallback_reason": (
                "; ".join(dict.fromkeys(fallback_reasons))
                if fallback_reasons
                else None
            ),
            "fallback_team_count": len(fallback_teams),
        }

    def _call(
        self,
        lookup: StructuredLookup,
        *,
        players: list[ResolvedEntity],
        teams: list[str],
    ) -> object:
        if isinstance(lookup, TeamRosterLookup):
            self._require_teams(teams, lookup.operation)
            return self._team_roster(lookup, teams)
        if isinstance(lookup, TeamDepthChartLookup):
            self._require_teams(teams, lookup.operation)
            return self._team_depth_chart(lookup, teams)
        if isinstance(lookup, TeamScheduleLookup):
            self._require_teams(teams, lookup.operation)
            return [
                row
                for team in teams
                for row in nfl_tools.get_team_games(
                    team,
                    lookup.season,
                    lookup.week,
                )
            ]
        if isinstance(lookup, PlayerSeasonStatsLookup):
            self._require_players(players, lookup.operation)
            return [
                nfl_tools.get_player_season_stats(
                    player.entity_id,
                    lookup.season,
                    lookup.season_type,
                    lookup.fields,
                )
                for player in players
            ]
        if isinstance(lookup, PlayerWeeklyStatsLookup):
            self._require_players(players, lookup.operation)
            return [
                row
                for player in players
                for row in nfl_tools.get_player_weekly_stats(
                    player.entity_id,
                    lookup.season,
                    lookup.week,
                    lookup.fields,
                )
            ]
        if isinstance(lookup, PlayerSnapCountsLookup):
            self._require_players(players, lookup.operation)
            return [
                row
                for player in players
                for row in nfl_tools.get_player_snap_counts(
                    player.entity_id,
                    lookup.season,
                    lookup.week,
                )
            ]
        if isinstance(lookup, TeamWeeklyStatsLookup):
            self._require_teams(teams, lookup.operation)
            return [
                row
                for team in teams
                for row in nfl_tools.get_team_weekly_stats(
                    team,
                    lookup.season,
                    lookup.week,
                    lookup.fields,
                )
            ]
        if isinstance(lookup, ECRRankingLookup):
            return nfl_tools.rank_players_by_ecr(
                season=lookup.season,
                positions=(
                    [str(position) for position in lookup.positions]
                    if lookup.positions is not None
                    else None
                ),
                scoring_format=lookup.scoring_format,
                league_format=lookup.league_format,
                snapshot_type=lookup.snapshot_type,
                as_of_date=lookup.as_of_date,
                sort_by="overall_rank",
                sort_direction="asc",
                minimum_overall_rank=lookup.minimum_overall_rank,
                maximum_overall_rank=lookup.maximum_overall_rank,
                limit=lookup.limit,
            )
        if isinstance(lookup, PlayerFormulaRankingLookup):
            return nfl_tools.rank_players_by_formula(
                season=lookup.season,
                formula=lookup.formula,
                season_type=lookup.season_type,
                position=lookup.position,
                minimum_field=lookup.minimum_field,
                minimum_value=lookup.minimum_value,
                sort_direction=lookup.sort_direction,
                limit=lookup.limit,
            )
        if isinstance(lookup, TeamFormulaRankingLookup):
            return nfl_tools.rank_teams_by_formula(
                season=lookup.season,
                perspective=lookup.perspective,
                formula=lookup.formula,
                season_type=lookup.season_type,
                minimum_games=lookup.minimum_games,
                sort_direction=lookup.sort_direction,
                limit=lookup.limit,
            )
        raise ValueError(f"Unsupported structured lookup: {lookup.operation}")


class StructuredLookupExecutor:
    def __init__(self, gateway: StructuredLookupGateway | None = None) -> None:
        self.gateway = gateway or StructuredToolGateway()

    @staticmethod
    def _anchor_teams(entities: list[ResolvedEntity]) -> list[str]:
        return list(
            dict.fromkeys(
                team
                for entity in entities
                for team in [
                    entity.entity_id if entity.entity_type == "team" else entity.team
                ]
                if team
            )
        )

    def execute(
        self,
        plan: QueryPlan,
        resolution: ResolutionResult,
    ) -> StructuredEnrichment:
        selector_results = {
            item.selector_index: item for item in resolution.selectors
        }
        targets: list[TargetEnrichment] = []
        for index, selector in enumerate(plan.entity_selectors):
            anchor = selector_results.get(index)
            entities = list(anchor.matches) if anchor is not None else []
            players = [item for item in entities if item.entity_type == "player"]
            teams = self._anchor_teams(entities)
            if anchor is not None:
                teams = list(
                    dict.fromkeys(
                        [
                            *teams,
                            *(
                                str(value)
                                for item in anchor.selector.hard_filters
                                if item.field == "team"
                                for value in item.values
                            ),
                        ]
                    )
                )
            targets.append(
                TargetEnrichment(
                    selector_index=index,
                    lookups=[
                        self.gateway.execute(
                            lookup,
                            players=players,
                            teams=teams,
                        )
                        for lookup in selector.structured_lookups
                    ],
                )
            )

        contexts_by_index = {
            item.request_index: item for item in resolution.contexts
        }
        contexts: list[ContextEnrichment] = []
        for request_index, request in enumerate(plan.context_requests):
            resolved = contexts_by_index.get(request_index)
            anchors = list(resolved.anchor_entities) if resolved is not None else []
            players = [
                item for item in anchors if item.entity_type == "player"
            ]
            teams = (
                list(resolved.teams)
                if resolved is not None
                else self._anchor_teams(anchors)
            )
            contexts.append(
                ContextEnrichment(
                    request_index=request_index,
                    lookups=[
                        self.gateway.execute(
                            lookup,
                            players=players,
                            teams=teams,
                        )
                        for lookup in request.structured_lookups
                    ],
                )
            )

        return StructuredEnrichment(targets=targets, contexts=contexts)


__all__ = [
    "ContextEnrichment",
    "LookupExecution",
    "StructuredEnrichment",
    "StructuredLookupGateway",
    "StructuredLookupExecutor",
    "StructuredToolGateway",
    "TargetEnrichment",
]
