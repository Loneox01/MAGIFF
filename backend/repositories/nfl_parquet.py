"""NFL repository backed by locally processed Parquet files."""

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


def find_players(name: str) -> list[dict]:
    players = pl.scan_parquet(PROCESSED_DIR / "reference" / "players.parquet")
    status = pl.scan_parquet(PROCESSED_DIR / "current" / "player_status.parquet")

    return (
        players.join(status.select("player_id", "latest_team", "status"), on="player_id")
        .filter(
            pl.col("display_name")
            .str.to_lowercase()
            .str.contains(name.lower(), literal=True)
        )
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
    week: int | None,
    fields: list[str],
) -> list[dict]:
    stats = pl.scan_parquet(
        PROCESSED_DIR / "seasons" / str(season) / "player_weekly_stats.parquet"
    ).filter(pl.col("player_id") == player_id)
    if week is not None:
        stats = stats.filter(pl.col("week") == week)

    return stats.select(*BASE_STAT_FIELDS, *fields).sort("week").collect().to_dicts()


def get_player_season_totals(
    player_id: str,
    season: int,
    fields: list[str],
) -> dict:
    stats = pl.scan_parquet(
        PROCESSED_DIR / "seasons" / str(season) / "player_weekly_stats.parquet"
    ).filter(pl.col("player_id") == player_id)

    result = stats.select(
        pl.len().alias("games"),
        *[pl.col(field).sum().alias(field) for field in fields],
    ).collect().to_dicts()[0]
    return {"player_id": player_id, "season": season, **result}


def get_team_games(team: str, season: int, week: int | None) -> list[dict]:
    games = pl.scan_parquet(
        PROCESSED_DIR / "seasons" / str(season) / "games.parquet"
    ).filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
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
