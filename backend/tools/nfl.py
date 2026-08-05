"""Small model-facing query tools backed by processed Parquet files. Temporary for testing locally."""

from pathlib import Path

import polars as pl


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

BASE_STAT_FIELDS = [
    "player_id",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
]
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
    query = name.strip().lower()
    if not query:
        raise ValueError("name cannot be empty")

    players = pl.scan_parquet(PROCESSED_DIR / "reference" / "players.parquet")
    status = pl.scan_parquet(PROCESSED_DIR / "current" / "player_status.parquet")

    return (
        players.join(status.select("player_id", "latest_team", "status"), on="player_id")
        .filter(pl.col("display_name").str.to_lowercase().str.contains(query, literal=True))
        .select(
            "player_id",
            "display_name",
            "position",
            "latest_team",
            "status",
        )
        .limit(10)
        .collect()
        .to_dicts()
    )


def get_player_weekly_stats(
    player_id: str,
    season: int,
    week: int | None = None,
    fields: list[str] | None = None,
) -> list[dict]:
    """Return one player's weekly stats, optionally for one week."""
    selected = _stat_fields(fields)
    stats = pl.scan_parquet(
        PROCESSED_DIR / "seasons" / str(season) / "player_weekly_stats.parquet"
    ).filter(pl.col("player_id") == player_id)
    if week is not None:
        stats = stats.filter(pl.col("week") == week)

    return stats.select(*BASE_STAT_FIELDS, *selected).sort("week").collect().to_dicts()


def get_player_season_totals(
    player_id: str,
    season: int,
    fields: list[str] | None = None,
) -> dict:
    """Sum additive player statistics across a season."""
    selected = _stat_fields(fields)
    stats = pl.scan_parquet(
        PROCESSED_DIR / "seasons" / str(season) / "player_weekly_stats.parquet"
    ).filter(pl.col("player_id") == player_id)

    result = stats.select(
        pl.len().alias("games"),
        *[pl.col(field).sum().alias(field) for field in selected],
    ).collect().to_dicts()[0]
    return {"player_id": player_id, "season": season, **result}


def get_team_games(team: str, season: int, week: int | None = None) -> list[dict]:
    """Return recent schedule/results data for a team."""
    games = pl.scan_parquet(
        PROCESSED_DIR / "seasons" / str(season) / "games.parquet"
    ).filter((pl.col("home_team") == team.upper()) | (pl.col("away_team") == team.upper()))
    if week is not None:
        games = games.filter(pl.col("week") == week)

    return (
        games.select(
            "game_id",
            "season",
            "game_type",
            "week",
            "gameday",
            "away_team",
            "away_score",
            "home_team",
            "home_score",
            "overtime",
            "spread_line",
            "total_line",
        )
        .sort("gameday")
        .collect()
        .with_columns(pl.col("gameday").cast(pl.String))
        .to_dicts()
    )


TOOL_HANDLERS = {
    "find_players": find_players,
    "get_player_weekly_stats": get_player_weekly_stats,
    "get_player_season_totals": get_player_season_totals,
    "get_team_games": get_team_games,
}
