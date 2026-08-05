"""Ingest raw data from nflverse API into .parquet files in backend/data/raw/nflverse

Creates the following files:

GENERAL
players.parquet
teams.parquet

BY SEASON
weekly_stats.parquet
weekly_rosters.parquet
schedules.parquet
snap_counts.parquet
team_weekly_stats.parquet

Includes a function to preview output headers, 
"""

from pathlib import Path

import nflreadpy as nfl
import polars as pl


BACKEND_DIR = Path(__file__).resolve().parent.parent
SEASON = 2025
NFLVERSE_DIR = BACKEND_DIR / "data" / "raw" / "nflverse"


def get_season_dir(season: int) -> Path:
    return NFLVERSE_DIR / str(season)


def download_data(season: int) -> None:
    season_dir = get_season_dir(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    players = nfl.load_players()
    teams = nfl.load_teams()

    weekly_stats = nfl.load_player_stats(
        seasons=[season],
        summary_level="week",
    )

    weekly_rosters = nfl.load_rosters_weekly(
        seasons=[season],
    )

    schedules = nfl.load_schedules(
        seasons=[season],
    )

    snap_counts = nfl.load_snap_counts(
        seasons=[season],
    )

    team_weekly_stats = nfl.load_team_stats(
        seasons=[season],
        summary_level="week",
    )

    # Latest data
    players.write_parquet(NFLVERSE_DIR / "players.parquet")
    teams.write_parquet(NFLVERSE_DIR / "teams.parquet")

    # Historical data
    weekly_stats.write_parquet(
        season_dir / "weekly_stats.parquet"
    )
    weekly_rosters.write_parquet(
        season_dir / "weekly_rosters.parquet"
    )
    schedules.write_parquet(
        season_dir / "schedules.parquet"
    )
    snap_counts.write_parquet(
        season_dir / "snap_counts.parquet"
    )
    team_weekly_stats.write_parquet(
        season_dir / "team_weekly_stats.parquet"
    )

    print(f"Reference data saved to: {NFLVERSE_DIR}")
    print(f"Season data saved to: {season_dir}")


def preview_headers(season: int) -> None:
    season_dir = get_season_dir(season)
    paths = {
        "PLAYERS": NFLVERSE_DIR / "players.parquet",
        "TEAMS": NFLVERSE_DIR / "teams.parquet",
        "WEEKLY STATS": season_dir / "weekly_stats.parquet",
        "WEEKLY ROSTERS": season_dir / "weekly_rosters.parquet",
        "SCHEDULES": season_dir / "schedules.parquet",
        "SNAP COUNTS": season_dir / "snap_counts.parquet",
        "TEAM WEEKLY STATS": season_dir / "team_weekly_stats.parquet",
    }

    for name, path in paths.items():
        schema = pl.read_parquet_schema(path)

        print(f"\n{name}")
        print("-" * 50)

        for column, dtype in schema.items():
            print(f"{column}: {dtype}")


if __name__ == "__main__":
    download_data(SEASON)
    preview_headers(SEASON)
