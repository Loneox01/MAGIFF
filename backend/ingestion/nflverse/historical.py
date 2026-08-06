"""Ingest a complete historical nflverse dataset for a requested season."""

import argparse

import nflreadpy as nfl

from .common import write_historical_snapshot


def season_loaders(season: int) -> dict:
    """Return downloads shared by historical and current workflows."""
    return {
        "weekly_stats": lambda: nfl.load_player_stats(
            seasons=[season],
            summary_level="week",
        ),
        "weekly_rosters": lambda: nfl.load_rosters_weekly(seasons=[season]),
        "schedules": lambda: nfl.load_schedules(seasons=[season]),
        "snap_counts": lambda: nfl.load_snap_counts(seasons=[season]),
        "team_weekly_stats": lambda: nfl.load_team_stats(
            seasons=[season],
            summary_level="week",
        ),
        "depth_charts": lambda: nfl.load_depth_charts(seasons=[season]),
    }


def load_season_frames(season: int) -> dict:
    """Download every selected historical dataset, failing on any error."""
    return {name: loader() for name, loader in season_loaders(season).items()}


def ingest_historical(season: int):
    frames = load_season_frames(season)
    output_dir = write_historical_snapshot(frames, season)
    print(f"Historical season {season} saved to: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    ingest_historical(args.season)


if __name__ == "__main__":
    main()
