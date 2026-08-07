"""Shared raw snapshot discovery for fantasy-market processing."""

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_FANTASY_DIR = BACKEND_DIR / "data" / "raw" / "fantasy"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def latest_snapshot_file(root: Path, filename: str) -> Path:
    snapshots = root / "snapshots"
    candidates = sorted(
        path / filename
        for path in snapshots.iterdir()
        if path.is_dir() and (path / filename).exists()
    ) if snapshots.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No {filename} snapshots found under: {snapshots}")
    return candidates[-1]


def player_ids_path(season: int) -> Path:
    return latest_snapshot_file(
        RAW_FANTASY_DIR / "nflverse" / "player_ids" / str(season),
        "fantasy_player_ids.parquet",
    )


def current_ecr_path(season: int) -> Path:
    return latest_snapshot_file(
        RAW_FANTASY_DIR / "nflverse" / "ecr" / str(season),
        "draft_ecr.parquet",
    )


def ecr_archive_path() -> Path:
    return latest_snapshot_file(
        RAW_FANTASY_DIR / "nflverse" / "ecr" / "archive",
        "ecr_archive.parquet",
    )

