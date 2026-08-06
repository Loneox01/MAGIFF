"""Shared paths and writing helpers for nflverse ingestion workflows."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl


BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_DIR / "data" / "raw" / "nflverse"
REFERENCE_DIR = RAW_DIR / "reference"
HISTORICAL_DIR = RAW_DIR / "historical"
CURRENT_DIR = RAW_DIR / "current"


def reference_dir(season: int) -> Path:
    return REFERENCE_DIR / str(season)


def historical_dir(season: int) -> Path:
    return HISTORICAL_DIR / str(season)


def current_dir(season: int) -> Path:
    return CURRENT_DIR / str(season)


def write_dataset(
    frames: dict[str, pl.DataFrame],
    target_dir: Path,
    category: str,
    season: int,
) -> Path:
    """Write the newest complete version of a raw dataset collection."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in frames.items():
        frame.write_parquet(target_dir / f"{name}.parquet")

    metadata = {
        "source": "nflverse",
        "category": category,
        "season": season,
        "acquired_at": datetime.now(UTC).isoformat(),
        "datasets": {
            name: {"rows": frame.height, "columns": frame.width}
            for name, frame in frames.items()
        },
    }
    (target_dir / "snapshot.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_dir


def write_historical_snapshot(
    frames: dict[str, pl.DataFrame],
    season: int,
) -> Path:
    """Archive one dated acquisition of a completed season."""
    acquired_at = datetime.now(UTC).isoformat()
    snapshot_id = date.today().isoformat()
    root = historical_dir(season)
    snapshot_dir = root / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    for name, frame in frames.items():
        frame.write_parquet(snapshot_dir / f"{name}.parquet")

    metadata = {
        "source": "nflverse",
        "category": "historical",
        "season": season,
        "snapshot_id": snapshot_id,
        "acquired_at": acquired_at,
        "datasets": {
            name: {"rows": frame.height, "columns": frame.width}
            for name, frame in frames.items()
        },
    }
    (snapshot_dir / "snapshot.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "latest.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "acquired_at": acquired_at,
                "season": season,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot_dir


def latest_reference_dir() -> Path:
    """Return the highest season-numbered reference directory."""
    candidates = [
        path for path in REFERENCE_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    ] if REFERENCE_DIR.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No reference data found under: {REFERENCE_DIR}")
    return max(candidates, key=lambda path: int(path.name))


def resolve_season_dir(season: int) -> Path:
    """Prefer active data, then the normalized historical season layout."""
    active = current_dir(season)
    if active.exists():
        return active

    historical = historical_dir(season)
    if (historical / "weekly_stats.parquet").exists():
        return historical

    # Compatibility with the earlier dated-snapshot experiment.
    pointer_path = historical / "latest.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        snapshot = historical / "snapshots" / pointer["snapshot_id"]
        if snapshot.exists():
            return snapshot

    raise FileNotFoundError(f"No current or historical data found for season {season}")
