"""Resolve planner descriptions to canonical entities from structured NFL data."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel

from database.client import get_supabase_client

from .planner import (
    EntityFilter,
    EntityFilterField,
    EntitySelector,
    PlayerResolutionBasis,
    PlayerSelector,
    QueryPlan,
    TeamSelector,
)


MAX_RESOLVED_ENTITIES = 200


@dataclass(frozen=True)
class FieldSpec:
    table: str
    column: str
    value_type: Literal["text", "integer", "number"]


# These mappings are the security and correctness boundary. Planner fields never
# become table or column names directly; only verified fields in this registry do.
PLAYER_FIELD_SPECS = {
    "position": FieldSpec("players", "position", "text"),
    "position_group": FieldSpec("players", "position_group", "text"),
    "rookie_season": FieldSpec("players", "rookie_season", "integer"),
    "draft_year": FieldSpec("players", "draft_year", "integer"),
    "draft_round": FieldSpec("players", "draft_round", "integer"),
    "draft_pick": FieldSpec("players", "draft_pick", "integer"),
    "college": FieldSpec("players", "college_name", "text"),
    "height": FieldSpec("players", "height", "integer"),
    "weight": FieldSpec("players", "weight", "integer"),
    "last_season": FieldSpec("players", "last_season", "integer"),
    "team": FieldSpec("player_status", "latest_team", "text"),
    "roster_status": FieldSpec("player_status", "status", "text"),
    "years_experience": FieldSpec(
        "player_status", "years_of_experience", "integer"
    ),
    "depth_chart_position": FieldSpec(
        "current_depth_chart_entries", "position", "text"
    ),
    "depth_rank": FieldSpec(
        "current_depth_chart_entries", "depth_rank", "integer"
    ),
    "formation": FieldSpec(
        "current_depth_chart_entries", "formation", "text"
    ),
    "ecr_rank": FieldSpec("player_ecr", "overall_rank", "number"),
    "ecr_position": FieldSpec("player_ecr", "position", "text"),
    "ecr_scoring_format": FieldSpec(
        "player_ecr", "scoring_format", "text"
    ),
    "ecr_league_format": FieldSpec("player_ecr", "league_format", "text"),
    "games": FieldSpec("player_season_stats", "games", "integer"),
    "fantasy_points": FieldSpec(
        "player_season_stats", "fantasy_points", "number"
    ),
    "fantasy_points_ppr": FieldSpec(
        "player_season_stats", "fantasy_points_ppr", "number"
    ),
    "targets": FieldSpec("player_season_stats", "targets", "integer"),
    "carries": FieldSpec("player_season_stats", "carries", "integer"),
    "receptions": FieldSpec("player_season_stats", "receptions", "integer"),
    "offense_snap_pct": FieldSpec(
        "player_snap_counts", "offense_pct", "number"
    ),
    "opponent": FieldSpec("player_snap_counts", "opponent", "text"),
}

TEAM_FILTER_FIELDS = {
    "conference": "team_conf",
    "division": "team_division",
}

_implemented_fields = {*PLAYER_FIELD_SPECS, *TEAM_FILTER_FIELDS}
_planner_fields = {field.value for field in EntityFilterField}
if _implemented_fields != _planner_fields:
    missing = sorted(_planner_fields - _implemented_fields)
    extra = sorted(_implemented_fields - _planner_fields)
    raise RuntimeError(
        f"Entity selector registry mismatch; missing={missing}, extra={extra}"
    )

SEASONAL_TABLES = {
    "current_depth_chart_entries",
    "player_ecr",
    "player_season_stats",
    "player_snap_counts",
}


class ResolvedEntity(BaseModel):
    entity_type: Literal["player", "team"]
    entity_id: str
    display_name: str
    team: str | None
    position: str | None
    position_group: str | None = None
    jersey_number: str | None = None
    roster_status: str | None = None
    rookie_season: int | None = None
    draft_year: int | None = None


class SelectorResolution(BaseModel):
    selector_index: int
    selector: EntitySelector
    status: Literal["resolved", "multiple", "unresolved"]
    matches: list[ResolvedEntity]
    unresolved_filters: list[str]
    semantic_qualifiers: list[str]
    truncated: bool


class ResolutionResult(BaseModel):
    selectors: list[SelectorResolution]

    @property
    def players(self) -> list[ResolvedEntity]:
        return _unique_entities(
            match
            for result in self.selectors
            for match in result.matches
            if match.entity_type == "player"
        )

    @property
    def teams(self) -> list[ResolvedEntity]:
        return _unique_entities(
            match
            for result in self.selectors
            for match in result.matches
            if match.entity_type == "team"
        )


def _unique_entities(entities) -> list[ResolvedEntity]:
    unique: dict[tuple[str, str], ResolvedEntity] = {}
    for entity in entities:
        unique[(entity.entity_type, entity.entity_id)] = entity
    return list(unique.values())


class EntityRepository(Protocol):
    def list_teams(self) -> list[dict]: ...

    def resolve_player_selector(
        self,
        selector: PlayerSelector,
        *,
        season: int | None,
        week: int | None,
    ) -> tuple[list[dict], list[str], bool]: ...


def _coerce_value(value: str, value_type: str) -> str | int | float:
    if value_type == "integer":
        return int(value)
    if value_type == "number":
        return float(value)
    return value


def _apply_filter(query, column: str, entity_filter: EntityFilter, value_type: str):
    if not entity_filter.values:
        raise ValueError(f"{entity_filter.field} requires at least one value")

    values = [_coerce_value(value, value_type) for value in entity_filter.values]
    if entity_filter.operator == "eq":
        return query.eq(column, values[0])
    if entity_filter.operator == "in":
        return query.in_(column, values)
    if entity_filter.operator == "gte":
        return query.gte(column, values[0])
    if entity_filter.operator == "lte":
        return query.lte(column, values[0])
    raise ValueError(f"Unsupported entity filter operator: {entity_filter.operator}")


class SupabaseEntityRepository:
    """Read-only, allowlisted entity lookups against the existing NFL tables."""

    def __init__(self, client=None) -> None:
        self.client = client or get_supabase_client()
        self._team_cache: list[dict] | None = None

    def list_teams(self) -> list[dict]:
        if self._team_cache is None:
            self._team_cache = (
                self.client.table("teams")
                .select(
                    "team_abbr,team_id,team_name,team_nick,team_conf,team_division"
                )
                .limit(100)
                .execute()
                .data
            )
        return self._team_cache

    def _base_player_rows(
        self,
        names: list[str],
        filters: list[EntityFilter],
        candidate_ids: set[str] | None,
    ) -> list[dict]:
        player_filters = [
            item for item in filters if PLAYER_FIELD_SPECS[item.field].table == "players"
        ]
        status_filters = [
            item
            for item in filters
            if PLAYER_FIELD_SPECS[item.field].table == "player_status"
        ]
        status_relation = "player_status!player_status_player_id_fkey"
        if status_filters:
            status_relation += "!inner"
        selected = (
            "player_id,display_name,common_first_name,football_name,last_name,"
            "position,position_group,rookie_season,draft_year,draft_round,draft_pick,"
            f"{status_relation}(latest_team,jersey_number,status,years_of_experience)"
        )

        def execute(
            name: str | None,
            name_column: str = "display_name",
            ids: set[str] | None = None,
        ) -> list[dict]:
            query = self.client.table("players").select(selected)
            if name:
                query = query.ilike(name_column, f"%{name}%")
            if ids is not None:
                if not ids:
                    return []
                query = query.in_("player_id", sorted(ids))
            for item in player_filters:
                spec = PLAYER_FIELD_SPECS[item.field]
                query = _apply_filter(query, spec.column, item, spec.value_type)
            for item in status_filters:
                spec = PLAYER_FIELD_SPECS[item.field]
                query = _apply_filter(
                    query,
                    f"player_status.{spec.column}",
                    item,
                    spec.value_type,
                )
            return query.limit(MAX_RESOLVED_ENTITIES + 1).execute().data

        base_is_constrained = bool(names or player_filters or status_filters)
        ids_for_query = None if base_is_constrained else candidate_ids

        if not names:
            if ids_for_query is not None and len(ids_for_query) > 50:
                rows_by_id: dict[str, dict] = {}
                ordered_ids = sorted(ids_for_query)
                for start in range(0, len(ordered_ids), 50):
                    batch = set(ordered_ids[start : start + 50])
                    for row in execute(None, ids=batch):
                        rows_by_id[row["player_id"]] = row
                    if len(rows_by_id) > MAX_RESOLVED_ENTITIES:
                        break
                rows = list(rows_by_id.values())
            else:
                rows = execute(None, ids=ids_for_query)
            if candidate_ids is not None:
                rows = [row for row in rows if row["player_id"] in candidate_ids]
            return rows

        rows_by_id: dict[str, dict] = {}
        for name in names:
            matched = execute(name, ids=ids_for_query)
            if not matched and " " not in name.strip():
                matched = execute(name, "last_name", ids=ids_for_query)
            for row in matched:
                if candidate_ids is None or row["player_id"] in candidate_ids:
                    rows_by_id[row["player_id"]] = row
        return list(rows_by_id.values())

    def _advanced_candidate_ids(
        self,
        filters: list[EntityFilter],
        *,
        season: int | None,
        week: int | None,
    ) -> tuple[set[str] | None, list[str]]:
        grouped: dict[str, list[EntityFilter]] = defaultdict(list)
        for item in filters:
            spec = PLAYER_FIELD_SPECS[item.field]
            if spec.table not in {"players", "player_status"}:
                grouped[spec.table].append(item)

        candidate_ids: set[str] | None = None
        unresolved: list[str] = []
        for table, table_filters in grouped.items():
            if table in SEASONAL_TABLES and season is None:
                unresolved.extend(
                    f"{item.field}: requires a season" for item in table_filters
                )
                continue

            query = self.client.table(table).select("player_id")
            if season is not None:
                query = query.eq("season", season)
            if week is not None and table in {
                "current_depth_chart_entries",
                "player_snap_counts",
            }:
                query = query.eq("week", week)
            if table == "player_season_stats":
                query = query.eq("season_type", "REG")

            for item in table_filters:
                spec = PLAYER_FIELD_SPECS[item.field]
                try:
                    query = _apply_filter(
                        query, spec.column, item, spec.value_type
                    )
                except ValueError as error:
                    unresolved.append(f"{item.field}: {error}")

            rows = query.limit(5000).execute().data
            table_ids = {row["player_id"] for row in rows if row.get("player_id")}
            candidate_ids = (
                table_ids if candidate_ids is None else candidate_ids & table_ids
            )

        return candidate_ids, unresolved

    def _historical_team_candidate_ids(
        self,
        filters: list[EntityFilter],
        *,
        season: int,
    ) -> tuple[set[str], list[str]]:
        query = (
            self.client.table("player_season_stats")
            .select("player_id")
            .eq("season", season)
            .eq("season_type", "REG")
        )
        unresolved: list[str] = []
        for item in filters:
            if item.operator == "eq":
                query = query.contains("teams", [item.values[0]])
            elif item.operator == "in":
                query = query.overlaps("teams", item.values)
            else:
                unresolved.append(
                    "team: historical team lookup only supports eq/in"
                )
        if unresolved:
            return set(), unresolved
        rows = query.limit(5000).execute().data
        return {row["player_id"] for row in rows if row.get("player_id")}, []

    def resolve_player_selector(
        self,
        selector: PlayerSelector,
        *,
        season: int | None,
        week: int | None,
    ) -> tuple[list[dict], list[str], bool]:
        supported: list[EntityFilter] = []
        unresolved: list[str] = []
        for item in selector.filters:
            spec = PLAYER_FIELD_SPECS.get(item.field)
            if spec is None:
                unresolved.append(f"{item.field}: not valid for a player selector")
                continue
            if not item.values:
                unresolved.append(f"{item.field}: requires at least one value")
                continue
            if item.operator != "in" and len(item.values) != 1:
                unresolved.append(
                    f"{item.field}: {item.operator} requires exactly one value"
                )
                continue
            try:
                for value in item.values:
                    _coerce_value(value, spec.value_type)
            except ValueError:
                unresolved.append(
                    f"{item.field}: values must be {spec.value_type}"
                )
                continue
            supported.append(item)

        historical_team_filters: list[EntityFilter] = []
        if season is not None and season < date.today().year:
            historical_team_filters = [
                item for item in supported if item.field == "team"
            ]
            supported = [item for item in supported if item.field != "team"]

        candidate_ids, advanced_unresolved = self._advanced_candidate_ids(
            supported,
            season=season,
            week=week,
        )
        unresolved.extend(advanced_unresolved)
        if historical_team_filters and season is not None:
            historical_ids, historical_unresolved = (
                self._historical_team_candidate_ids(
                    historical_team_filters,
                    season=season,
                )
            )
            unresolved.extend(historical_unresolved)
            candidate_ids = (
                historical_ids
                if candidate_ids is None
                else candidate_ids & historical_ids
            )
        rows = self._base_player_rows(selector.names, supported, candidate_ids)
        truncated = len(rows) > MAX_RESOLVED_ENTITIES
        return rows[:MAX_RESOLVED_ENTITIES], unresolved, truncated


def _normalize_team_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


class EntityResolver:
    def __init__(self, repository: EntityRepository | None = None) -> None:
        self.repository = repository or SupabaseEntityRepository()

    def _team_aliases(self) -> tuple[list[dict], dict[str, list[dict]]]:
        teams = self.repository.list_teams()
        aliases: dict[str, list[dict]] = defaultdict(list)
        city_alias_counts: dict[str, int] = defaultdict(int)
        city_by_team: dict[str, str] = {}

        for team in teams:
            team_name = str(team["team_name"])
            nickname = str(team["team_nick"])
            city = team_name.removesuffix(nickname).strip()
            city_by_team[str(team["team_abbr"])] = city
            city_alias_counts[_normalize_team_text(city)] += 1

            for value in (team["team_abbr"], team["team_id"], team_name, nickname):
                normalized = _normalize_team_text(str(value))
                if normalized:
                    aliases[normalized].append(team)

        for team in teams:
            city = city_by_team[str(team["team_abbr"])]
            normalized = _normalize_team_text(city)
            if normalized and city_alias_counts[normalized] == 1:
                aliases[normalized].append(team)
        return teams, aliases

    def _resolve_team_values(self, values: list[str]) -> tuple[list[str], list[str]]:
        _, aliases = self._team_aliases()
        resolved: list[str] = []
        unresolved: list[str] = []
        for value in values:
            matches = aliases.get(_normalize_team_text(value), [])
            unique = {str(match["team_abbr"]): match for match in matches}
            if len(unique) == 1:
                resolved.append(next(iter(unique)))
            else:
                unresolved.append(value)
        return list(dict.fromkeys(resolved)), unresolved

    def _resolve_team_selector(
        self, selector: TeamSelector, selector_index: int
    ) -> SelectorResolution:
        teams, aliases = self._team_aliases()
        candidates = {str(team["team_abbr"]): team for team in teams}
        unresolved_filters: list[str] = []

        if selector.names:
            named: dict[str, dict] = {}
            for name in selector.names:
                for team in aliases.get(_normalize_team_text(name), []):
                    named[str(team["team_abbr"])] = team
            candidates = named

        for item in selector.filters:
            column = TEAM_FILTER_FIELDS.get(item.field)
            if column is None:
                unresolved_filters.append(
                    f"{item.field}: not valid for a team selector"
                )
                continue
            normalized_values = {_normalize_team_text(value) for value in item.values}
            if item.operator not in {"eq", "in"}:
                unresolved_filters.append(
                    f"{item.field}: team fields only support eq/in"
                )
                continue
            candidates = {
                key: team
                for key, team in candidates.items()
                if _normalize_team_text(str(team.get(column, ""))) in normalized_values
            }

        matches = [
            ResolvedEntity(
                entity_type="team",
                entity_id=str(team["team_abbr"]),
                display_name=str(team["team_name"]),
                team=str(team["team_abbr"]),
                position=None,
            )
            for team in candidates.values()
        ]
        return SelectorResolution(
            selector_index=selector_index,
            selector=selector,
            status=_resolution_status(matches),
            matches=matches,
            unresolved_filters=unresolved_filters,
            semantic_qualifiers=selector.semantic_qualifiers,
            truncated=False,
        )

    def _normalize_player_selector(
        self, selector: PlayerSelector
    ) -> tuple[PlayerSelector, list[str]]:
        filters: list[EntityFilter] = []
        unresolved: list[str] = []
        for item in selector.filters:
            if item.field != "team":
                filters.append(item)
                continue
            values, missing = self._resolve_team_values(item.values)
            if values:
                filters.append(item.model_copy(update={"values": values}))
            unresolved.extend(f"team: could not resolve {value!r}" for value in missing)
        return selector.model_copy(update={"filters": filters}), unresolved

    @staticmethod
    def _selectors_with_mentions(plan: QueryPlan) -> list[EntitySelector]:
        selectors = list(plan.entity_selectors)
        covered_players = {
            name.casefold()
            for selector in selectors
            if selector.entity_type == "player"
            for name in selector.names
        }
        covered_teams = {
            name.casefold()
            for selector in selectors
            if selector.entity_type == "team"
            for name in selector.names
        }
        selectors.extend(
            PlayerSelector(
                entity_type="player",
                reference_text=name,
                names=[name],
                identity_confidence=1.0,
                resolution_basis=PlayerResolutionBasis.EXACT_NAME,
                filters=[],
                semantic_qualifiers=[],
            )
            for name in plan.player_mentions
            if name.casefold() not in covered_players
        )
        selectors.extend(
            TeamSelector(
                entity_type="team",
                names=[name],
                filters=[],
                semantic_qualifiers=[],
            )
            for name in plan.team_mentions
            if name.casefold() not in covered_teams
        )
        return selectors

    def resolve(self, plan: QueryPlan) -> ResolutionResult:
        results: list[SelectorResolution] = []
        for index, selector in enumerate(self._selectors_with_mentions(plan)):
            if selector.entity_type == "team":
                results.append(self._resolve_team_selector(selector, index))
                continue

            normalized, team_errors = self._normalize_player_selector(selector)
            requires_identity = (
                normalized.resolution_basis
                != PlayerResolutionBasis.NOT_APPLICABLE
            )
            if requires_identity and not normalized.names:
                rows = []
                unresolved = ["player identity could not be grounded"]
                truncated = False
            else:
                rows, unresolved, truncated = (
                    self.repository.resolve_player_selector(
                        normalized,
                        season=plan.season,
                        week=plan.week,
                    )
                )
            resolution_errors = [*team_errors, *unresolved]
            if resolution_errors and not normalized.names:
                rows = []
            matches = []
            for row in rows:
                status = row.get("player_status") or {}
                if isinstance(status, list):
                    status = status[0] if status else {}
                matches.append(
                    ResolvedEntity(
                        entity_type="player",
                        entity_id=str(row["player_id"]),
                        display_name=str(row["display_name"]),
                        team=status.get("latest_team"),
                        position=row.get("position"),
                        position_group=row.get("position_group"),
                        jersey_number=status.get("jersey_number"),
                        roster_status=status.get("status"),
                        rookie_season=row.get("rookie_season"),
                        draft_year=row.get("draft_year"),
                    )
                )
            results.append(
                SelectorResolution(
                    selector_index=index,
                    selector=normalized,
                    status=_resolution_status(matches),
                    matches=matches,
                    unresolved_filters=resolution_errors,
                    semantic_qualifiers=normalized.semantic_qualifiers,
                    truncated=truncated,
                )
            )
        return ResolutionResult(selectors=results)


def _resolution_status(
    matches: list[ResolvedEntity],
) -> Literal["resolved", "multiple", "unresolved"]:
    if not matches:
        return "unresolved"
    if len(matches) == 1:
        return "resolved"
    return "multiple"
