"""Model-facing NFL tools backed by the selected data repository."""

from repositories import nfl_supabase as repository
from tools.field_catalog import (
    PLAYER_FORMULA_FIELDS,
    PLAYER_SEASON_STAT_FIELDS,
    PLAYER_WEEKLY_STAT_FIELDS,
    TEAM_WEEKLY_STAT_FIELDS,
)
from tools.formulas import ZeroDenominatorError, evaluate_formula, parse_formula
from tools.team_analytics import (
    TEAM_DEFENSE_FORMULA_FIELDS,
    TEAM_OFFENSE_FORMULA_FIELDS,
    build_team_season_rows,
)


ALLOWED_STAT_FIELDS = set(PLAYER_WEEKLY_STAT_FIELDS)
DEFAULT_STAT_FIELDS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points_ppr",
]
SEASON_STAT_FIELDS = set(PLAYER_SEASON_STAT_FIELDS)
DEFAULT_SEASON_FIELDS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points_ppr",
]
TEAM_STAT_FIELDS = set(TEAM_WEEKLY_STAT_FIELDS)
DEFAULT_TEAM_FIELDS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
]
def _stat_fields(fields: list[str] | None) -> list[str]:
    selected = DEFAULT_STAT_FIELDS if fields is None else list(dict.fromkeys(fields))
    invalid = set(selected) - ALLOWED_STAT_FIELDS
    if invalid:
        raise ValueError(f"Unsupported stat fields: {sorted(invalid)}")
    return selected


def _fields(
    fields: list[str] | None,
    allowed: set[str],
    defaults: list[str],
) -> list[str]:
    selected = defaults if fields is None else list(dict.fromkeys(fields))
    invalid = set(selected) - allowed
    if invalid:
        raise ValueError(f"Unsupported fields: {sorted(invalid)}")
    return selected


def _team_code(team: str) -> str:
    code = team.strip().upper()
    if not code.isalpha() or not 2 <= len(code) <= 3:
        raise ValueError("team must be a two- or three-letter NFL abbreviation")
    return code


def _season_type(season_type: str | None) -> str:
    value = "REG" if season_type is None else season_type.strip().upper()
    if value not in {"REG", "POST"}:
        raise ValueError("season_type must be REG, POST, or null")
    return value


def find_players(name: str) -> list[dict]:
    """Find players by a case-insensitive partial name."""
    query = name.strip()
    if not query:
        raise ValueError("name cannot be empty")
    return repository.find_players(query)


def get_player_weekly_stats(
    player_id: str,
    season: int,
    week: int | None = None,
    fields: list[str] | None = None,
) -> list[dict]:
    """Return one player's weekly stats, optionally for one week."""
    return repository.get_player_weekly_stats(
        player_id, season, week, _stat_fields(fields)
    )


def get_team_games(team: str, season: int, week: int | None = None) -> list[dict]:
    """Return recent schedule/results data for a team."""
    return repository.get_team_games(_team_code(team), season, week)


def get_player_season_stats(
    player_id: str,
    season: int,
    season_type: str | None = None,
    fields: list[str] | None = None,
) -> dict:
    """Return stored season totals and efficiency metrics for one player."""
    selected = _fields(fields, SEASON_STAT_FIELDS, DEFAULT_SEASON_FIELDS)
    return repository.get_player_season_stats(
        player_id, season, _season_type(season_type), selected
    )


def get_team_weekly_stats(
    team: str,
    season: int,
    week: int | None = None,
    fields: list[str] | None = None,
) -> list[dict]:
    """Return a team's offensive totals for one week or an entire season."""
    selected = _fields(fields, TEAM_STAT_FIELDS, DEFAULT_TEAM_FIELDS)
    return repository.get_team_weekly_stats(
        _team_code(team), season, week, selected
    )


def get_team_depth_chart(
    team: str,
    season: int,
    week: int | None = None,
    position: str | None = None,
) -> list[dict]:
    """Return a current or historical team depth chart."""
    position_code = position.strip().upper() if position else None
    return repository.get_team_depth_chart(
        _team_code(team), season, week, position_code
    )


def get_player_snap_counts(
    player_id: str,
    season: int,
    week: int | None = None,
) -> list[dict]:
    """Return weekly offensive, defensive, and special-teams usage."""
    return repository.get_player_snap_counts(player_id, season, week)


def get_team_roster(
    team: str,
    season: int,
    week: int | None = None,
    position: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Return a weekly roster, defaulting to the latest available week."""
    position_code = position.strip().upper() if position else None
    status_code = status.strip().upper() if status else None
    return repository.get_team_roster(
        _team_code(team), season, week, position_code, status_code
    )


def rank_players_by_formula(
    season: int,
    formula: str,
    season_type: str | None = None,
    position: str | None = None,
    minimum_field: str | None = None,
    minimum_value: float | None = None,
    sort_direction: str = "desc",
    limit: int = 10,
) -> dict:
    """Rank player seasons using a safe, model-provided arithmetic formula."""
    parsed = parse_formula(formula, PLAYER_FORMULA_FIELDS)
    if (minimum_field is None) != (minimum_value is None):
        raise ValueError("minimum_field and minimum_value must both be set or null")
    if minimum_field is not None and minimum_field not in PLAYER_FORMULA_FIELDS:
        raise ValueError(f"Unsupported minimum field: {minimum_field}")
    if minimum_value is not None and minimum_value < 0:
        raise ValueError("minimum_value cannot be negative")
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be asc or desc")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    position_code = position.strip().upper() if position else None
    candidates = repository.get_player_season_candidates(
        season,
        _season_type(season_type),
        position_code,
        list(parsed.fields),
        minimum_field,
        minimum_value,
    )

    ranked: list[tuple[float, dict]] = []
    zero_denominator: list[dict] = []
    invalid_rows = 0
    for row in candidates:
        try:
            value = evaluate_formula(parsed, row)
        except ZeroDenominatorError:
            zero_denominator.append(row)
        except (TypeError, ValueError):
            invalid_rows += 1
        else:
            ranked.append((value, row))

    ranked.sort(key=lambda item: item[1]["player_id"])
    ranked.sort(key=lambda item: item[0], reverse=sort_direction == "desc")
    zero_denominator.sort(key=lambda row: row["player_id"])
    selected = ranked[:limit]
    selected_zero = zero_denominator[:limit]
    names = repository.get_player_names(
        [row["player_id"] for _value, row in selected]
        + [row["player_id"] for row in selected_zero]
    )

    input_fields = list(parsed.fields)
    if minimum_field and minimum_field not in input_fields:
        input_fields.append(minimum_field)

    def player_result(row: dict) -> dict:
        return {
            "player_id": row["player_id"],
            "display_name": names.get(row["player_id"]),
            "position": row.get("position"),
            "team": row.get("last_team"),
            "games": row.get("games"),
            "inputs": {field: row.get(field) for field in input_fields},
        }

    return {
        "formula": parsed.canonical,
        "season": season,
        "season_type": _season_type(season_type),
        "position": position_code,
        "minimum": (
            {"field": minimum_field, "value": minimum_value}
            if minimum_field is not None
            else None
        ),
        "sort_direction": sort_direction,
        "eligible_players": len(candidates),
        "excluded_invalid_rows": invalid_rows,
        "zero_denominator_qualifiers": [
            player_result(row) for row in selected_zero
        ],
        "results": [
            {
                "rank": rank,
                "metric_value": round(value, 6),
                **player_result(row),
            }
            for rank, (value, row) in enumerate(selected, start=1)
        ],
    }


def rank_teams_by_formula(
    season: int,
    perspective: str,
    formula: str,
    season_type: str | None = None,
    minimum_games: int | None = None,
    sort_direction: str = "desc",
    limit: int = 10,
) -> dict:
    """Rank team seasons using offense or opponent-derived defense metrics."""
    if perspective not in {"offense", "defense"}:
        raise ValueError("perspective must be offense or defense")
    allowed_fields = (
        TEAM_OFFENSE_FORMULA_FIELDS
        if perspective == "offense"
        else TEAM_DEFENSE_FORMULA_FIELDS
    )
    parsed = parse_formula(formula, allowed_fields)
    if minimum_games is not None and minimum_games < 1:
        raise ValueError("minimum_games must be positive or null")
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be asc or desc")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    normalized_season_type = _season_type(season_type)
    weekly_rows, games = repository.get_team_formula_inputs(
        season, normalized_season_type
    )
    candidates = build_team_season_rows(
        weekly_rows, games, season, normalized_season_type
    )
    if minimum_games is not None:
        candidates = [
            row for row in candidates if row["games"] >= minimum_games
        ]

    ranked: list[tuple[float, dict]] = []
    zero_denominator: list[dict] = []
    invalid_rows = 0
    for row in candidates:
        try:
            value = evaluate_formula(parsed, row)
        except ZeroDenominatorError:
            zero_denominator.append(row)
        except (TypeError, ValueError):
            invalid_rows += 1
        else:
            ranked.append((value, row))

    ranked.sort(key=lambda item: item[1]["team"])
    ranked.sort(key=lambda item: item[0], reverse=sort_direction == "desc")
    zero_denominator.sort(key=lambda row: row["team"])
    selected = ranked[:limit]
    selected_zero = zero_denominator[:limit]
    names = repository.get_team_names(
        [row["team"] for _value, row in selected]
        + [row["team"] for row in selected_zero]
    )
    input_fields = list(parsed.fields)
    if "games" not in input_fields:
        input_fields.append("games")

    def team_result(row: dict) -> dict:
        return {
            "team": row["team"],
            "team_name": names.get(row["team"]),
            "games": row["games"],
            "inputs": {field: row.get(field) for field in input_fields},
        }

    return {
        "formula": parsed.canonical,
        "season": season,
        "season_type": normalized_season_type,
        "perspective": perspective,
        "minimum_games": minimum_games,
        "sort_direction": sort_direction,
        "eligible_teams": len(candidates),
        "excluded_invalid_rows": invalid_rows,
        "zero_denominator_qualifiers": [
            team_result(row) for row in selected_zero
        ],
        "results": [
            {
                "rank": rank,
                "metric_value": round(value, 6),
                **team_result(row),
            }
            for rank, (value, row) in enumerate(selected, start=1)
        ],
    }


TOOL_HANDLERS = {
    "find_players": find_players,
    "get_player_weekly_stats": get_player_weekly_stats,
    "get_team_games": get_team_games,
    "get_player_season_stats": get_player_season_stats,
    "get_team_weekly_stats": get_team_weekly_stats,
    "get_team_depth_chart": get_team_depth_chart,
    "get_player_snap_counts": get_player_snap_counts,
    "get_team_roster": get_team_roster,
    "rank_players_by_formula": rank_players_by_formula,
    "rank_teams_by_formula": rank_teams_by_formula,
}
