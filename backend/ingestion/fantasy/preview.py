"""Preview schemas and row counts for saved fantasy-market snapshots."""

import argparse
from pathlib import Path

import polars as pl

from .common import RAW_DIR


def latest_snapshot_dir(root: Path) -> Path:
    snapshots_dir = root / "snapshots"
    candidates = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No snapshots found under: {snapshots_dir}")
    return candidates[-1]


def print_parquet_preview(path: Path) -> None:
    schema = pl.read_parquet_schema(path)
    row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
    print(f"\n{path.stem.upper().replace('_', ' ')}")
    print(f"Path: {path}")
    print(f"Rows: {row_count}")
    print("-" * 50)
    for column, dtype in schema.items():
        print(f"{column}: {dtype}")


def preview_ecr(season: int, include_archive: bool = False) -> None:
    current = latest_snapshot_dir(RAW_DIR / "nflverse" / "ecr" / str(season))
    for path in sorted(current.glob("*.parquet")):
        print_parquet_preview(path)

    player_ids = latest_snapshot_dir(
        RAW_DIR / "nflverse" / "player_ids" / str(season)
    )
    for path in sorted(player_ids.glob("*.parquet")):
        print_parquet_preview(path)

    if include_archive:
        archive = latest_snapshot_dir(RAW_DIR / "nflverse" / "ecr" / "archive")
        for path in sorted(archive.glob("*.parquet")):
            print_parquet_preview(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also preview the complete historical ECR archive.",
    )
    args = parser.parse_args()
    preview_ecr(args.season, args.include_archive)


if __name__ == "__main__":
    main()

