"""Model-facing NFL tools backed by the selected data repository."""

from repositories import nfl_supabase as repository


ALLOWED_STAT_FIELDS = {
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points",
    "fantasy_points_ppr",
}
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


def _stat_fields(fields: list[str] | None) -> list[str]:
    selected = DEFAULT_STAT_FIELDS if fields is None else list(dict.fromkeys(fields))
    invalid = set(selected) - ALLOWED_STAT_FIELDS
    if invalid:
        raise ValueError(f"Unsupported stat fields: {sorted(invalid)}")
    return selected


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


def get_player_season_totals(
    player_id: str,
    season: int,
    fields: list[str] | None = None,
) -> dict:
    """Sum additive player statistics across a season."""
    return repository.get_player_season_totals(
        player_id, season, _stat_fields(fields)
    )


def get_team_games(team: str, season: int, week: int | None = None) -> list[dict]:
    """Return recent schedule/results data for a team."""
    team_code = team.strip().upper()
    if not team_code.isalpha() or not 2 <= len(team_code) <= 3:
        raise ValueError("team must be a two- or three-letter NFL abbreviation")
    return repository.get_team_games(team_code, season, week)


TOOL_HANDLERS = {
    "find_players": find_players,
    "get_player_weekly_stats": get_player_weekly_stats,
    "get_player_season_totals": get_player_season_totals,
    "get_team_games": get_team_games,
}
