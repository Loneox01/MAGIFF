"""Process one completed NFL season from the latest historical snapshot."""

import argparse

import polars as pl

from .common import (
    PROCESSED_DIR,
    historical_season_dir,
    latest_reference_dir,
    validate_unique,
    write_outputs,
)
from .season_tables import process_season_tables


def process_historical(season: int) -> None:
    raw_dir = historical_season_dir(season)
    raw_players = pl.read_parquet(latest_reference_dir() / "players.parquet")
    outputs, dropped = process_season_tables(
        raw_dir,
        season,
        raw_players,
        historical=True,
        strict=True,
    )

    validate_unique(outputs["games"], ["game_id"], "games")
    validate_unique(
        outputs["player_weekly_stats"],
        ["player_id", "season", "week", "season_type", "game_id"],
        "player_weekly_stats",
    )
    validate_unique(
        outputs["player_season_stats"],
        ["player_id", "season", "season_type"],
        "player_season_stats",
    )
    write_outputs(
        PROCESSED_DIR / "seasons" / str(season),
        outputs,
        workflow=f"historical_{season}",
        dropped=dropped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    process_historical(args.season)


if __name__ == "__main__":
    main()
