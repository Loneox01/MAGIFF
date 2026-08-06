"""Refresh overwriteable nflverse data for the active NFL season."""

import argparse

import nflreadpy as nfl

from .common import current_dir, write_dataset
from .historical import season_loaders


def ingest_current(season: int | None = None):
    active_season = season or nfl.get_current_season(roster=True)
    frames = {}
    failures = {}
    for name, loader in season_loaders(active_season).items():
        try:
            frames[name] = loader()
        except Exception as error:
            failures[name] = str(error)
            print(f"{name}: unavailable ({error})")

    if not frames:
        raise RuntimeError(f"No current-season datasets were available: {failures}")

    output_dir = write_dataset(
        frames,
        current_dir(active_season),
        category="current",
        season=active_season,
    )
    print(f"Current season {active_season} saved to: {output_dir}")
    if failures:
        print(f"Unavailable datasets left unchanged: {sorted(failures)}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        help="Override the automatically detected active roster season.",
    )
    args = parser.parse_args()
    ingest_current(args.season)


if __name__ == "__main__":
    main()
