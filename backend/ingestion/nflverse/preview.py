"""Preview schemas saved in the latest reference and requested season data."""

import argparse
from pathlib import Path

import polars as pl

from .common import latest_reference_dir, resolve_season_dir


def print_schemas(snapshot_dir: Path) -> None:
    for path in sorted(snapshot_dir.glob("*.parquet")):
        print(f"\n{path.stem.upper().replace('_', ' ')}")
        print("-" * 50)
        for column, dtype in pl.read_parquet_schema(path).items():
            print(f"{column}: {dtype}")


def preview_latest(season: int) -> None:
    print("\nREFERENCE DATA")
    print_schemas(latest_reference_dir())
    print(f"\nSEASON {season} DATA")
    print_schemas(resolve_season_dir(season))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    preview_latest(args.season)


if __name__ == "__main__":
    main()
