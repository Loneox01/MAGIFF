"""Deterministic structured-stat service used by the Discord /stats command."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from repositories import nfl_supabase
from services.news import (
    PlayerCandidate,
    TeamCandidate,
    resolve_player,
    resolve_team,
)
from tools import nfl
from tools.field_catalog import (
    PLAYER_FORMULA_FIELDS,
    PLAYER_WEEKLY_STAT_FIELDS,
)
from tools.formulas import (
    FormulaError,
    ZeroDenominatorError,
    evaluate_formula,
    parse_formula,
)
from tools.team_analytics import (
    TEAM_DEFENSE_FORMULA_FIELDS,
    TEAM_OFFENSE_FORMULA_FIELDS,
    build_team_season_rows,
)


DEFAULT_STATS_COUNT = 5
MAX_STATS_COUNT = 10
POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K"})


class StatsOutcome(StrEnum):
    SUCCESS = "success"
    PLAYER_NOT_FOUND = "player_not_found"
    PLAYER_AMBIGUOUS = "player_ambiguous"
    TEAM_NOT_FOUND = "team_not_found"
    TEAM_AMBIGUOUS = "team_ambiguous"
    NO_STATS = "no_stats"
    INVALID_FORMULA = "invalid_formula"
    ZERO_DENOMINATOR = "zero_denominator"


class StatsScope(StrEnum):
    PLAYER_SEASON = "player-season"
    PLAYER_WEEKLY = "player-weekly"
    TEAM_OFFENSE = "team-offense"
    TEAM_DEFENSE = "team-defense"


@dataclass(frozen=True)
class PlayerStatsQuery:
    player: str
    season: int | None = None
    week: int | None = None
    season_type: str = "REG"
    view: str = "summary"
    formula: str | None = None


@dataclass(frozen=True)
class PlayerLeadersQuery:
    formula: str
    season: int | None = None
    season_type: str = "REG"
    position: str | None = None
    minimum_field: str | None = None
    minimum_value: float | None = None
    sort_direction: str = "desc"
    count: int = DEFAULT_STATS_COUNT


@dataclass(frozen=True)
class TeamStatsQuery:
    team: str
    season: int | None = None
    week: int | None = None
    season_type: str = "REG"
    perspective: str = "offense"
    view: str = "summary"
    formula: str | None = None


@dataclass(frozen=True)
class TeamLeadersQuery:
    formula: str
    season: int | None = None
    season_type: str = "REG"
    perspective: str = "offense"
    minimum_games: int | None = None
    sort_direction: str = "desc"
    count: int = DEFAULT_STATS_COUNT


@dataclass(frozen=True)
class StatsFieldsQuery:
    scope: StatsScope
    search: str | None = None
    count: int = 25


StatsQuery = (
    PlayerStatsQuery
    | PlayerLeadersQuery
    | TeamStatsQuery
    | TeamLeadersQuery
    | StatsFieldsQuery
)


@dataclass(frozen=True)
class StatsResult:
    outcome: StatsOutcome
    query: StatsQuery
    season: int | None = None
    rows: tuple[dict, ...] = ()
    resolved_player: PlayerCandidate | None = None
    resolved_team: TeamCandidate | None = None
    player_candidates: tuple[PlayerCandidate, ...] = ()
    team_candidates: tuple[TeamCandidate, ...] = ()
    resolution_note: str | None = None
    formula: str | None = None
    error: str | None = None


class StatsRepository(Protocol):
    def find_players(self, name: str) -> list[PlayerCandidate]: ...
    def list_teams(self) -> list[TeamCandidate]: ...
    def latest_player_season(self, player_id: str, season_type: str) -> int | None: ...
    def latest_leader_season(self, season_type: str) -> int | None: ...
    def latest_team_season(self, season_type: str) -> int | None: ...
    def player_season(self, player_id: str, season: int, season_type: str, fields: list[str]) -> dict: ...
    def player_week(self, player_id: str, season: int, week: int, fields: list[str]) -> list[dict]: ...
    def player_leaders(self, query: PlayerLeadersQuery, season: int) -> dict: ...
    def team_rows(self, season: int, season_type: str, week: int | None) -> list[dict]: ...
    def team_leaders(self, query: TeamLeadersQuery, season: int) -> dict: ...


class SupabaseStatsRepository:
    def __init__(self, client=None) -> None:
        from database.client import get_supabase_client

        self.client = client or get_supabase_client()
        self._teams: list[TeamCandidate] | None = None

    def find_players(self, name: str) -> list[PlayerCandidate]:
        return [
            PlayerCandidate(
                str(row["player_id"]),
                str(row["display_name"]),
                None if row.get("position") is None else str(row["position"]),
                None if row.get("latest_team") is None else str(row["latest_team"]),
                None if row.get("status") is None else str(row["status"]),
            )
            for row in nfl_supabase.find_players(name)
        ]

    def list_teams(self) -> list[TeamCandidate]:
        if self._teams is None:
            rows = self.client.table("teams").select("team_abbr,team_name,team_nick").limit(100).execute().data
            self._teams = [
                TeamCandidate(str(row["team_abbr"]), str(row["team_name"]), str(row["team_nick"]))
                for row in rows
            ]
        return self._teams

    def _latest(self, table: str, season_type: str, player_id: str | None = None) -> int | None:
        query = self.client.table(table).select("season").eq("season_type", season_type)
        if player_id is not None:
            query = query.eq("player_id", player_id)
        rows = query.order("season", desc=True).limit(1).execute().data
        return int(rows[0]["season"]) if rows else None

    def latest_player_season(self, player_id: str, season_type: str) -> int | None:
        return self._latest("player_season_stats", season_type, player_id)

    def latest_leader_season(self, season_type: str) -> int | None:
        return self._latest("player_season_stats", season_type)

    def latest_team_season(self, season_type: str) -> int | None:
        return self._latest("team_weekly_stats", season_type)

    def player_season(self, player_id: str, season: int, season_type: str, fields: list[str]) -> dict:
        return nfl.get_player_season_stats(player_id, season, season_type, fields)

    def player_week(self, player_id: str, season: int, week: int, fields: list[str]) -> list[dict]:
        return nfl.get_player_weekly_stats(player_id, season, week, fields)

    def player_leaders(self, query: PlayerLeadersQuery, season: int) -> dict:
        return nfl.rank_players_by_formula(
            season, query.formula, query.season_type, query.position,
            query.minimum_field, query.minimum_value, query.sort_direction,
            query.count,
        )

    def team_rows(self, season: int, season_type: str, week: int | None) -> list[dict]:
        weekly, games = nfl_supabase.get_team_formula_inputs(season, season_type)
        if week is not None:
            game_ids = {str(row["game_id"]) for row in weekly if row.get("week") == week}
            weekly = [row for row in weekly if row.get("week") == week]
            games = [row for row in games if str(row.get("game_id")) in game_ids]
        return build_team_season_rows(weekly, games, season, season_type)

    def team_leaders(self, query: TeamLeadersQuery, season: int) -> dict:
        return nfl.rank_teams_by_formula(
            season, query.perspective, query.formula, query.season_type,
            query.minimum_games, query.sort_direction, query.count,
        )


PLAYER_VIEWS = {
    "fantasy": ("games", "fantasy_points", "fantasy_points_ppr", "fantasy_points_per_game", "fantasy_points_ppr_per_game"),
    "passing": ("games", "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "completion_percentage", "passing_yards_per_attempt"),
    "rushing": ("games", "carries", "rushing_yards", "rushing_tds", "rushing_yards_per_carry"),
    "receiving": ("games", "targets", "receptions", "receiving_yards", "receiving_tds", "catch_percentage", "receiving_yards_per_reception", "receiving_yards_per_target"),
    "usage": ("games", "attempts", "carries", "targets", "receptions"),
}
PLAYER_WEEKLY_VIEWS = {
    key: tuple(field for field in fields if field in PLAYER_WEEKLY_STAT_FIELDS)
    for key, fields in PLAYER_VIEWS.items()
}
TEAM_VIEWS = {
    "passing": ("games", "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered"),
    "rushing": ("games", "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost"),
    "summary": ("games", "points_scored", "passing_yards", "passing_tds", "rushing_yards", "rushing_tds"),
}


def formula_fields(scope: StatsScope) -> tuple[str, ...]:
    mapping = {
        StatsScope.PLAYER_SEASON: PLAYER_FORMULA_FIELDS,
        StatsScope.PLAYER_WEEKLY: PLAYER_WEEKLY_STAT_FIELDS,
        StatsScope.TEAM_OFFENSE: TEAM_OFFENSE_FORMULA_FIELDS,
        StatsScope.TEAM_DEFENSE: TEAM_DEFENSE_FORMULA_FIELDS,
    }
    return tuple(sorted(mapping[scope]))


def _season_type(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"REG", "POST"}:
        raise ValueError("season_type must be REG or POST")
    return normalized


def _validate_common(season: int | None, week: int | None, count: int | None = None) -> None:
    if season is not None and not 1999 <= season <= 2100:
        raise ValueError("season must be between 1999 and 2100")
    if week is not None and not 1 <= week <= 22:
        raise ValueError("week must be between 1 and 22")
    if count is not None and not 1 <= count <= MAX_STATS_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_STATS_COUNT}")


def _summary_view(position: str | None) -> str:
    if position == "QB":
        return "passing"
    if position in {"RB", "FB"}:
        return "rushing"
    if position in {"WR", "TE"}:
        return "receiving"
    return "fantasy"


class StatsService:
    def __init__(self, repository: StatsRepository | None = None) -> None:
        self.repository = repository or SupabaseStatsRepository()

    def execute(self, query: StatsQuery) -> StatsResult:
        if isinstance(query, StatsFieldsQuery):
            if not 1 <= query.count <= 25:
                raise ValueError("field count must be between 1 and 25")
            fields = formula_fields(query.scope)
            if query.search:
                needle = query.search.casefold()
                fields = tuple(field for field in fields if needle in field.casefold())
            return StatsResult(StatsOutcome.SUCCESS, query, rows=tuple({"field": field} for field in fields[: query.count]))
        if isinstance(query, PlayerStatsQuery):
            return self._player(query)
        if isinstance(query, PlayerLeadersQuery):
            return self._player_leaders(query)
        if isinstance(query, TeamStatsQuery):
            return self._team(query)
        return self._team_leaders(query)

    def _player(self, query: PlayerStatsQuery) -> StatsResult:
        _validate_common(query.season, query.week)
        season_type = _season_type(query.season_type)
        player, candidates, note = resolve_player(self.repository, query.player)
        if player is None:
            return StatsResult(
                StatsOutcome.PLAYER_AMBIGUOUS if candidates else StatsOutcome.PLAYER_NOT_FOUND,
                query, player_candidates=candidates,
            )
        season = query.season or self.repository.latest_player_season(player.player_id, season_type)
        if season is None:
            return StatsResult(StatsOutcome.NO_STATS, query, resolved_player=player)
        allowed = PLAYER_WEEKLY_STAT_FIELDS if query.week is not None else PLAYER_FORMULA_FIELDS
        view = _summary_view(player.position) if query.view == "summary" else query.view
        views = PLAYER_WEEKLY_VIEWS if query.week is not None else PLAYER_VIEWS
        if view not in views:
            raise ValueError("view must be summary, fantasy, passing, rushing, receiving, or usage")
        try:
            parsed = parse_formula(query.formula, allowed) if query.formula else None
        except FormulaError as error:
            return StatsResult(StatsOutcome.INVALID_FORMULA, query, season=season, resolved_player=player, resolution_note=note, error=str(error))
        fields = list(parsed.fields if parsed else views[view])
        raw = (
            self.repository.player_week(player.player_id, season, query.week, fields)
            if query.week is not None
            else [self.repository.player_season(player.player_id, season, season_type, fields)]
        )
        rows = [row for row in raw if row]
        if not rows:
            return StatsResult(StatsOutcome.NO_STATS, query, season=season, resolved_player=player, resolution_note=note)
        row = rows[0]
        if parsed:
            try:
                metric = evaluate_formula(parsed, row)
            except ZeroDenominatorError:
                return StatsResult(StatsOutcome.ZERO_DENOMINATOR, query, season=season, resolved_player=player, resolution_note=note, formula=parsed.canonical)
            except (TypeError, ValueError) as error:
                return StatsResult(StatsOutcome.NO_STATS, query, season=season, resolved_player=player, resolution_note=note, formula=parsed.canonical, error=str(error))
            row = {"metric_value": metric, "inputs": {field: row.get(field) for field in parsed.fields}, **{key: row.get(key) for key in ("team", "opponent_team", "week") if key in row}}
        else:
            row = {field: row.get(field) for field in views[view]}
        return StatsResult(StatsOutcome.SUCCESS, query, season=season, rows=(row,), resolved_player=player, resolution_note=note, formula=parsed.canonical if parsed else None)

    def _player_leaders(self, query: PlayerLeadersQuery) -> StatsResult:
        _validate_common(query.season, None, query.count)
        _season_type(query.season_type)
        if query.position is not None and query.position.upper() not in POSITIONS:
            raise ValueError("position must be QB, RB, WR, TE, K, or omitted")
        if (query.minimum_field is None) != (query.minimum_value is None):
            raise ValueError("minimum_field and minimum_value must both be provided")
        try:
            parsed = parse_formula(query.formula, PLAYER_FORMULA_FIELDS)
        except FormulaError as error:
            return StatsResult(StatsOutcome.INVALID_FORMULA, query, error=str(error))
        season = query.season or self.repository.latest_leader_season(query.season_type)
        if season is None:
            return StatsResult(StatsOutcome.NO_STATS, query)
        ranked = self.repository.player_leaders(query, season)
        rows = tuple(ranked.get("results") or ())
        return StatsResult(StatsOutcome.SUCCESS if rows else StatsOutcome.NO_STATS, query, season=season, rows=rows, formula=parsed.canonical)

    def _team(self, query: TeamStatsQuery) -> StatsResult:
        _validate_common(query.season, query.week)
        season_type = _season_type(query.season_type)
        if query.perspective not in {"offense", "defense"}:
            raise ValueError("perspective must be offense or defense")
        team, candidates, ambiguous = resolve_team(self.repository, query.team)
        if team is None:
            return StatsResult(StatsOutcome.TEAM_AMBIGUOUS if ambiguous else StatsOutcome.TEAM_NOT_FOUND, query, team_candidates=candidates)
        season = query.season or self.repository.latest_team_season(season_type)
        if season is None:
            return StatsResult(StatsOutcome.NO_STATS, query, resolved_team=team)
        allowed = TEAM_OFFENSE_FORMULA_FIELDS if query.perspective == "offense" else TEAM_DEFENSE_FORMULA_FIELDS
        try:
            parsed = parse_formula(query.formula, allowed) if query.formula else None
        except FormulaError as error:
            return StatsResult(StatsOutcome.INVALID_FORMULA, query, season=season, resolved_team=team, error=str(error))
        rows = self.repository.team_rows(season, season_type, query.week)
        row = next((value for value in rows if value.get("team") == team.code), None)
        if row is None:
            return StatsResult(StatsOutcome.NO_STATS, query, season=season, resolved_team=team)
        if parsed:
            try:
                value = evaluate_formula(parsed, row)
            except ZeroDenominatorError:
                return StatsResult(StatsOutcome.ZERO_DENOMINATOR, query, season=season, resolved_team=team, formula=parsed.canonical)
            row = {"metric_value": value, "inputs": {field: row.get(field) for field in parsed.fields}, "games": row.get("games")}
        else:
            selected = TEAM_VIEWS.get(query.view)
            if selected is None:
                raise ValueError("view must be summary, passing, or rushing")
            if query.perspective == "defense":
                selected = tuple("points_allowed" if field == "points_scored" else field if field == "games" else f"{field}_allowed" for field in selected)
            row = {field: row.get(field) for field in selected}
        return StatsResult(StatsOutcome.SUCCESS, query, season=season, rows=(row,), resolved_team=team, formula=parsed.canonical if parsed else None)

    def _team_leaders(self, query: TeamLeadersQuery) -> StatsResult:
        _validate_common(query.season, None, query.count)
        _season_type(query.season_type)
        if query.perspective not in {"offense", "defense"}:
            raise ValueError("perspective must be offense or defense")
        if query.minimum_games is not None and not 1 <= query.minimum_games <= 25:
            raise ValueError("minimum_games must be between 1 and 25")
        if query.sort_direction not in {"asc", "desc"}:
            raise ValueError("direction must be highest or lowest")
        allowed = TEAM_OFFENSE_FORMULA_FIELDS if query.perspective == "offense" else TEAM_DEFENSE_FORMULA_FIELDS
        try:
            parsed = parse_formula(query.formula, allowed)
        except FormulaError as error:
            return StatsResult(StatsOutcome.INVALID_FORMULA, query, error=str(error))
        season = query.season or self.repository.latest_team_season(query.season_type)
        if season is None:
            return StatsResult(StatsOutcome.NO_STATS, query)
        ranked = self.repository.team_leaders(query, season)
        rows = tuple(ranked.get("results") or ())
        return StatsResult(StatsOutcome.SUCCESS if rows else StatsOutcome.NO_STATS, query, season=season, rows=rows, formula=parsed.canonical)
