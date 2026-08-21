"""Resolve planner descriptions to canonical entities from structured NFL data."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from database.client import get_supabase_client

from .lookups import ContextScopePolicy, TEAM_ANCHORED_LOOKUP_OPERATIONS
from .planner import (
    ContextRequest,
    EntityFilter,
    EntityFilterField,
    EntitySelector,
    PlayerResolutionBasis,
    PlayerSelector,
    QueryPlan,
    TeamSelector,
    TeamCodeFilter,
)


MAX_RESOLVED_ENTITIES = 200

# Reference data intentionally preserves historical franchises. For seasons in
# which a relocation has already happened, resolve the old code to the active
# franchise and deduplicate it with the current row.
FRANCHISE_RELOCATIONS = {
    "OAK": ("LV", 2020),
    "SD": ("LAC", 2017),
    "STL": ("LA", 2016),
}

OPTIONAL_PLAYER_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _player_name_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _without_optional_player_suffix(value: str) -> str | None:
    """Return a lookup name only when the final token is a known suffix."""
    tokens = _player_name_tokens(value)
    if len(tokens) < 2 or tokens[-1] not in OPTIONAL_PLAYER_NAME_SUFFIXES:
        return None
    return " ".join(tokens[:-1])


def _suffix_insensitive_player_name(value: str) -> str:
    tokens = _player_name_tokens(value)
    if tokens and tokens[-1] in OPTIONAL_PLAYER_NAME_SUFFIXES:
        tokens = tokens[:-1]
    return "".join(tokens)


class ResolutionValidationError(RuntimeError):
    """Temporary typed boundary for unsafe resolved retrieval plans."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
    # The planner's finite Formation vocabulary represents the normalized
    # offense/defense/special-teams side. The processed `formation` column is a
    # source formation such as `3WR 1TE` or `Base 4-3 D` in newer snapshots.
    "formation": FieldSpec(
        "current_depth_chart_entries", "position_group", "text"
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


class ContextResolution(BaseModel):
    request_index: int
    request: ContextRequest
    status: Literal["resolved", "unresolved"]
    anchor_entities: list[ResolvedEntity]
    teams: list[str]
    unresolved: list[str]
    truncated: bool


class ResolutionResult(BaseModel):
    selectors: list[SelectorResolution]
    contexts: list[ContextResolution] = Field(default_factory=list)

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


def validate_resolution_bounds(result: ResolutionResult) -> None:
    """Stop an incomplete player-group lookup from becoming a hard filter."""
    truncated_groups = [
        selector_result.selector.reference_text
        for selector_result in result.selectors
        if selector_result.selector.entity_type == "player"
        and not selector_result.selector.names
        and selector_result.truncated
    ]
    truncated_contexts = [
        context.request_index for context in result.contexts if context.truncated
    ]
    if not truncated_groups and not truncated_contexts:
        return

    if truncated_contexts:
        raise ResolutionValidationError(
            "unbounded_context_group",
            "Contextual player expansion exceeded the safe resolution limit for "
            f"request(s) {truncated_contexts}.",
        )

    references = ", ".join(repr(value) for value in truncated_groups)
    raise ResolutionValidationError(
        "unbounded_player_group",
        f"Player group {references} exceeded the safe resolution limit. "
        "Its objective scope must be made more specific before retrieval.",
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

    def resolve_player_teams(
        self,
        player_ids: list[str],
        *,
        season: int,
        week: int | None,
    ) -> dict[str, str]: ...


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

        if not names:
            if candidate_ids is not None and len(candidate_ids) > 50:
                rows_by_id: dict[str, dict] = {}
                ordered_ids = sorted(candidate_ids)
                for start in range(0, len(ordered_ids), 50):
                    batch = set(ordered_ids[start : start + 50])
                    for row in execute(None, ids=batch):
                        rows_by_id[row["player_id"]] = row
                    if len(rows_by_id) > MAX_RESOLVED_ENTITIES:
                        break
                rows = list(rows_by_id.values())
            else:
                rows = execute(None, ids=candidate_ids)
            if candidate_ids is not None:
                rows = [row for row in rows if row["player_id"] in candidate_ids]
            return rows

        rows_by_id: dict[str, dict] = {}
        for name in names:
            matched = execute(name)
            if not matched and " " not in name.strip():
                matched = execute(name, "last_name")
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

    def _current_team_candidate_ids(
        self,
        filters: list[EntityFilter],
        *,
        season: int | None,
    ) -> tuple[set[str], list[str]]:
        """Resolve active team membership from the current depth-chart snapshot."""
        allowed_teams: set[str] | None = None
        unresolved: list[str] = []
        for item in filters:
            values = {str(value) for value in item.values}
            if item.operator not in {"eq", "in"}:
                unresolved.append("team: current team lookup only supports eq/in")
                continue
            allowed_teams = (
                values if allowed_teams is None else allowed_teams & values
            )

        if not allowed_teams or unresolved:
            return set(), unresolved

        selected_season = season
        if selected_season is None:
            newest = (
                self.client.table("current_depth_chart_entries")
                .select("season")
                .in_("team", sorted(allowed_teams))
                .order("season", desc=True)
                .limit(1)
                .execute()
                .data
            )
            if not newest:
                return set(), []
            selected_season = int(newest[0]["season"])

        rows = (
            self.client.table("current_depth_chart_entries")
            .select("player_id")
            .eq("season", selected_season)
            .in_("team", sorted(allowed_teams))
            .limit(5000)
            .execute()
            .data
        )
        return {row["player_id"] for row in rows if row.get("player_id")}, []

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
        # Soft filters are retained on the selector for downstream context but
        # intentionally never become database predicates.
        for item in selector.hard_filters:
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

        team_filters = [item for item in supported if item.field == "team"]
        current_team_filters: list[EntityFilter] = []
        historical_team_filters: list[EntityFilter] = []
        if team_filters:
            if not selector.names and (
                season is None or season >= date.today().year
            ):
                current_team_filters = team_filters
            elif season is not None and season < date.today().year:
                historical_team_filters = team_filters
            if current_team_filters or historical_team_filters:
                supported = [item for item in supported if item.field != "team"]

        candidate_ids, advanced_unresolved = self._advanced_candidate_ids(
            supported,
            season=season,
            week=week,
        )
        unresolved.extend(advanced_unresolved)
        if current_team_filters:
            current_ids, current_unresolved = self._current_team_candidate_ids(
                current_team_filters,
                season=season,
            )
            unresolved.extend(current_unresolved)
            candidate_ids = (
                current_ids
                if candidate_ids is None
                else candidate_ids & current_ids
            )
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

    def resolve_player_teams(
        self,
        player_ids: list[str],
        *,
        season: int,
        week: int | None,
    ) -> dict[str, str]:
        """Resolve season-appropriate team membership for contextual anchors."""
        if not player_ids:
            return {}
        query = (
            self.client.table("player_weekly_rosters")
            .select("player_id,team,week")
            .eq("season", season)
            .in_("player_id", player_ids)
        )
        if week is not None:
            query = query.eq("week", week)
        rows = query.order("week", desc=True).limit(1000).execute().data
        teams: dict[str, str] = {}
        for row in rows:
            player_id = row.get("player_id")
            team = row.get("team")
            if player_id and team and player_id not in teams:
                teams[str(player_id)] = str(team)
        return teams


def _normalize_team_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


class EntityResolver:
    def __init__(self, repository: EntityRepository | None = None) -> None:
        self.repository = repository or SupabaseEntityRepository()

    @staticmethod
    def _player_entity(row: dict) -> ResolvedEntity:
        status = row.get("player_status") or {}
        if isinstance(status, list):
            status = status[0] if status else {}
        return ResolvedEntity(
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

    def suffix_variant_candidates(
        self,
        selector: PlayerSelector,
        *,
        season: int | None,
        week: int | None,
    ) -> list[ResolvedEntity]:
        """Propose DB candidates for Sol without silently grounding a suffix miss.

        The normal resolver remains exact-first. This fallback is deliberately
        narrow: it runs only for one model-proposed name ending in a recognized
        generational suffix, preserves all selector filters, and retains only
        rows whose full names are equal after optional-suffix normalization.
        """
        if len(selector.names) != 1:
            return []
        candidate_name = selector.names[0]
        lookup_name = _without_optional_player_suffix(candidate_name)
        if lookup_name is None:
            return []

        lookup_selector = selector.model_copy(update={"names": [lookup_name]})
        rows, _, _ = self.repository.resolve_player_selector(
            lookup_selector,
            season=season,
            week=week,
        )
        expected = _suffix_insensitive_player_name(candidate_name)
        return _unique_entities(
            self._player_entity(row)
            for row in rows
            if _suffix_insensitive_player_name(str(row.get("display_name", "")))
            == expected
        )

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

    @staticmethod
    def _canonical_team_code(team_code: str, season: int | None) -> str:
        relocation = FRANCHISE_RELOCATIONS.get(team_code)
        if relocation is None:
            return team_code
        current_code, first_season = relocation
        if season is None or season >= first_season:
            return current_code
        return team_code

    def _canonical_team_candidates(
        self,
        candidates: list[dict],
        all_teams: list[dict],
        *,
        season: int | None,
    ) -> dict[str, dict]:
        rows_by_code = {str(team["team_abbr"]): team for team in all_teams}
        canonical: dict[str, dict] = {}
        for team in candidates:
            code = self._canonical_team_code(str(team["team_abbr"]), season)
            canonical[code] = rows_by_code.get(code, team)
        return canonical

    def _resolve_team_values(
        self, values: list[str], *, season: int | None
    ) -> tuple[list[str], list[str]]:
        _, aliases = self._team_aliases()
        resolved: list[str] = []
        unresolved: list[str] = []
        for value in values:
            matches = aliases.get(_normalize_team_text(value), [])
            unique = {str(match["team_abbr"]): match for match in matches}
            if len(unique) == 1:
                resolved.append(
                    self._canonical_team_code(next(iter(unique)), season)
                )
            else:
                unresolved.append(value)
        return list(dict.fromkeys(resolved)), unresolved

    def _resolve_team_selector(
        self,
        selector: TeamSelector,
        selector_index: int,
        *,
        season: int | None,
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

        for item in selector.hard_filters:
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

        candidates = self._canonical_team_candidates(
            list(candidates.values()),
            teams,
            season=season,
        )

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
        self, selector: PlayerSelector, *, season: int | None
    ) -> tuple[PlayerSelector, list[str]]:
        filters = []
        unresolved: list[str] = []
        team_scopes: list[set[str]] = []
        for item in selector.hard_filters:
            if item.field in {"conference", "division"}:
                teams = self.repository.list_teams()
                column = TEAM_FILTER_FIELDS[item.field]
                normalized_values = {
                    _normalize_team_text(str(value)) for value in item.values
                }
                matched = [
                    team
                    for team in teams
                    if _normalize_team_text(str(team.get(column, "")))
                    in normalized_values
                ]
                codes = sorted(
                    self._canonical_team_candidates(
                        matched,
                        teams,
                        season=season,
                    )
                )
                if codes:
                    team_scopes.append(set(codes))
                else:
                    unresolved.append(
                        f"{item.field}: could not resolve {item.values!r}"
                    )
                continue
            if item.field != "team":
                filters.append(item)
                continue
            values, missing = self._resolve_team_values(
                item.values,
                season=season,
            )
            if values:
                team_scopes.append(set(values))
            unresolved.extend(f"team: could not resolve {value!r}" for value in missing)

        if team_scopes:
            team_codes = sorted(set.intersection(*team_scopes))
            if team_codes:
                filters.append(
                    TeamCodeFilter(
                        field="team",
                        operator="in",
                        values=team_codes,
                    )
                )
            else:
                unresolved.append("team: structured scopes do not overlap")
        normalized = PlayerSelector.model_validate(
            {**selector.model_dump(), "hard_filters": filters}
        )
        return normalized, unresolved

    @staticmethod
    def _identity_lookup_selector(selector: PlayerSelector) -> PlayerSelector:
        """Try a literal full-name phrase alongside Luna's canonical guess.

        This preserves meaningful spelling and punctuation distinctions such as
        `DJ Moore` versus `D.J. Moore` without turning short nicknames like CMC
        or K9 into broad database searches. The public selector remains limited
        to one model candidate; the extra names exist only inside repository
        lookup.
        """
        reference = selector.reference_text.strip()
        if not selector.names or " " not in reference:
            return selector
        lookup_names = list(dict.fromkeys([reference, *selector.names]))
        if lookup_names == selector.names:
            return selector
        return selector.model_copy(update={"names": lookup_names})

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
                hard_filters=[],
                soft_filters=[],
                semantic_qualifiers=[],
            )
            for name in plan.player_mentions
            if name.casefold() not in covered_players
        )
        selectors.extend(
            TeamSelector(
                entity_type="team",
                names=[name],
                hard_filters=[],
                soft_filters=[],
                semantic_qualifiers=[],
            )
            for name in plan.team_mentions
            if name.casefold() not in covered_teams
        )
        return selectors

    def resolve_contexts(
        self,
        plan: QueryPlan,
        selector_results: list[SelectorResolution],
    ) -> list[ContextResolution]:
        by_index = {item.selector_index: item for item in selector_results}
        contexts: list[ContextResolution] = []
        for request_index, request in enumerate(plan.context_requests):
            anchor_result = by_index.get(request.anchor_selector_index)
            anchors = list(anchor_result.matches) if anchor_result is not None else []
            unresolved: list[str] = []
            anchor_type = (
                anchor_result.selector.entity_type
                if anchor_result is not None
                else None
            )
            if anchor_type == "player":
                anchors = [
                    anchor for anchor in anchors if anchor.entity_type == "player"
                ]
                is_specific_player = bool(anchor_result.selector.names)
                if is_specific_player and len(anchors) != 1:
                    unresolved.append(
                        "specific-player context anchor must resolve to exactly "
                        "one player"
                    )
                    anchors = []
                elif not is_specific_player and not anchors:
                    unresolved.append(
                        "player-group context anchor must resolve to at least "
                        "one player"
                    )
            elif anchor_type == "team":
                anchors = [
                    anchor for anchor in anchors if anchor.entity_type == "team"
                ]
                if not anchors:
                    unresolved.append(
                        "context anchor must resolve to at least one team"
                    )
            elif anchor_type is None:
                unresolved.append("context anchor selector could not be resolved")

            requires_anchor_teams = (
                request.scope_policy
                in {
                    ContextScopePolicy.ANCHOR_TEAMS,
                    ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS,
                }
                or any(
                    lookup.operation in TEAM_ANCHORED_LOOKUP_OPERATIONS
                    for lookup in request.structured_lookups
                )
            )

            teams: list[str] = []
            if anchor_type == "team":
                teams.extend(anchor.entity_id for anchor in anchors)
            elif anchors and requires_anchor_teams:
                explicit_teams = [
                    str(value)
                    for item in anchor_result.selector.hard_filters
                    if item.field == "team"
                    for value in item.values
                ]
                if explicit_teams:
                    teams.extend(explicit_teams)
                elif (
                    plan.season is not None
                    and plan.season < date.today().year
                ):
                    player_ids = [anchor.entity_id for anchor in anchors]
                    seasonal = self.repository.resolve_player_teams(
                        player_ids,
                        season=plan.season,
                        week=plan.week,
                    )
                    unresolved_players = []
                    for anchor in anchors:
                        seasonal_team = seasonal.get(anchor.entity_id)
                        if seasonal_team:
                            teams.append(seasonal_team)
                        else:
                            unresolved_players.append(anchor.display_name)
                    if unresolved_players:
                        unresolved.append(
                            "context anchor team membership is incomplete for "
                            "the requested season"
                        )
                else:
                    teams.extend(anchor.team for anchor in anchors if anchor.team)
                    missing_anchors = [anchor for anchor in anchors if not anchor.team]
                    seasonal_missing: dict[str, str] = {}
                    if missing_anchors and plan.season is not None:
                        seasonal_missing = self.repository.resolve_player_teams(
                            [anchor.entity_id for anchor in missing_anchors],
                            season=plan.season,
                            week=plan.week,
                        )
                        for anchor in missing_anchors:
                            seasonal_team = seasonal_missing.get(anchor.entity_id)
                            if seasonal_team:
                                teams.append(seasonal_team)
                    unresolved_players = [
                        anchor.display_name
                        for anchor in missing_anchors
                        if not seasonal_missing.get(anchor.entity_id)
                    ]
                    if unresolved_players:
                        unresolved.append(
                            "context anchor current team membership is incomplete"
                        )

            canonical_teams: list[str] = []
            if teams:
                canonical_teams, missing = self._resolve_team_values(
                    teams,
                    season=plan.season,
                )
                unresolved.extend(
                    f"context team could not resolve {team!r}" for team in missing
                )

            anchor_resolved = bool(anchors)
            if request.scope_policy in {
                ContextScopePolicy.ANCHOR_TEAMS,
                ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS,
            }:
                scope_resolved = bool(canonical_teams)
            elif request.scope_policy == ContextScopePolicy.LOOKUP_ENTITIES:
                # The structured-enrichment phase validates whether the
                # requested operation actually returns bounded entities.
                scope_resolved = bool(request.structured_lookups)
                if not scope_resolved:
                    unresolved.append(
                        "lookup_entities scope requires a structured lookup"
                    )
            else:
                # A semantic-only context branch deliberately has no metadata
                # boundary beyond its resolved anchor and query provenance.
                scope_resolved = True

            status = (
                "resolved"
                if anchor_resolved and scope_resolved and not unresolved
                else "unresolved"
            )
            contexts.append(
                ContextResolution(
                    request_index=request_index,
                    request=request,
                    status=status,
                    anchor_entities=anchors,
                    teams=canonical_teams,
                    unresolved=unresolved,
                    truncated=(
                        anchor_result.truncated
                        if anchor_result is not None
                        else False
                    ),
                )
            )
        return contexts

    def resolve(self, plan: QueryPlan) -> ResolutionResult:
        results: list[SelectorResolution] = []
        for index, selector in enumerate(self._selectors_with_mentions(plan)):
            if selector.entity_type == "team":
                results.append(
                    self._resolve_team_selector(
                        selector,
                        index,
                        season=plan.season,
                    )
                )
                continue

            normalized, team_errors = self._normalize_player_selector(
                selector,
                season=plan.season,
            )
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
                        self._identity_lookup_selector(normalized),
                        season=plan.season,
                        week=plan.week,
                    )
                )
            resolution_errors = [*team_errors, *unresolved]
            if resolution_errors and not normalized.names:
                rows = []
            matches = [self._player_entity(row) for row in rows]
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
        return ResolutionResult(
            selectors=results,
            contexts=self.resolve_contexts(plan, results),
        )


def _resolution_status(
    matches: list[ResolvedEntity],
) -> Literal["resolved", "multiple", "unresolved"]:
    if not matches:
        return "unresolved"
    if len(matches) == 1:
        return "resolved"
    return "multiple"
