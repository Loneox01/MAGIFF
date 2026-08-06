"""
Refresh the latest nflverse player/team reference data for a reference year.

Excludes current depth-charts
"""

import argparse

import nflreadpy as nfl

from .common import reference_dir, write_dataset


def ingest_reference(season: int | None = None):
    reference_season = season or nfl.get_current_season(roster=True)
    frames = {
        "players": nfl.load_players(),
        "teams": nfl.load_teams(),
    }
    output_dir = write_dataset(
        frames,
        reference_dir(reference_season),
        category="reference",
        season=reference_season,
    )
    print(f"Reference data saved to: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        help="Reference year; defaults to the active roster season.",
    )
    args = parser.parse_args()
    ingest_reference(args.season)


if __name__ == "__main__":
    main()
