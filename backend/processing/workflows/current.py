"""Process overwriteable data for the active NFL season."""

import argparse

import polars as pl

from .common import (
    PROCESSED_DIR,
    current_season_dir,
    latest_reference_dir,
    write_outputs,
)
from .season_tables import process_season_tables


def process_current(season: int) -> None:
    raw_dir = current_season_dir(season)
    raw_players = pl.read_parquet(latest_reference_dir() / "players.parquet")
    outputs, dropped = process_season_tables(
        raw_dir,
        season,
        raw_players,
        historical=False,
        strict=False,
    )

    # Current files are intentionally overwritten on every refresh. Missing
    # preseason datasets are omitted without deleting previously valid files.
    write_outputs(
        PROCESSED_DIR / "current",
        outputs,
        workflow=f"current_{season}",
        dropped=dropped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    process_current(args.season)


if __name__ == "__main__":
    main()
