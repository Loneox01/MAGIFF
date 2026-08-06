"""Shared paths, identity mapping, validation, and output helpers."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

import polars as pl


BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_DIR / "data" / "raw" / "nflverse"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
PLAYER_ID_NAMESPACE = UUID("f0cf9ef7-d56d-4a6f-a8e1-197be003c7a8")


def latest_reference_dir() -> Path:
    root = RAW_DIR / "reference"
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No reference data found under: {root}")
    return max(candidates, key=lambda path: int(path.name))


def current_season_dir(season: int) -> Path:
    path = RAW_DIR / "current" / str(season)
    if not path.exists():
        raise FileNotFoundError(f"No current raw data found at: {path}")
    return path


def historical_season_dir(season: int) -> Path:
    root = RAW_DIR / "historical" / str(season)
    pointer_path = root / "latest.json"
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        path = root / "snapshots" / pointer["snapshot_id"]
        if path.exists():
            return path
    if (root / "weekly_stats.parquet").exists():
        return root
    raise FileNotFoundError(f"No historical raw data found under: {root}")


def internal_player_id(gsis_id: str) -> str:
    return str(uuid5(PLAYER_ID_NAMESPACE, f"gsis:{gsis_id}"))


def player_identities(raw_players: pl.DataFrame) -> pl.DataFrame:
    return raw_players.select("gsis_id", "espn_id").with_columns(
        pl.col("gsis_id")
        .map_elements(internal_player_id, return_dtype=pl.String)
        .alias("player_id")
    )


def attach_player_id(
    frame: pl.DataFrame,
    identities: pl.DataFrame,
    external_column: str = "gsis_id",
) -> tuple[pl.DataFrame, int]:
    mapping = identities.select(
        pl.col("gsis_id").alias(external_column), "player_id"
    ).unique(external_column)
    joined = frame.join(mapping, on=external_column, how="left")
    unmatched = joined.filter(pl.col("player_id").is_null()).height
    return joined.filter(pl.col("player_id").is_not_null()), unmatched


def attach_depth_chart_player_id(
    frame: pl.DataFrame,
    identities: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Prefer GSIS identity and use ESPN identity when GSIS is absent."""
    by_gsis = identities.select("gsis_id", "player_id").unique("gsis_id")
    by_espn = (
        identities.filter(pl.col("espn_id").is_not_null())
        .select("espn_id", pl.col("player_id").alias("espn_player_id"))
        .unique("espn_id")
    )
    joined = (
        frame.join(by_gsis, on="gsis_id", how="left")
        .join(by_espn, on="espn_id", how="left")
        .with_columns(
            pl.coalesce("player_id", "espn_player_id").alias("player_id")
        )
        .drop("espn_player_id")
    )
    unmatched = joined.filter(pl.col("player_id").is_null()).height
    return joined.filter(pl.col("player_id").is_not_null()), unmatched


def validate_unique(frame: pl.DataFrame, columns: list[str], name: str) -> None:
    duplicates = frame.group_by(columns).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(f"{name} has {duplicates.height} duplicate logical keys")


def write_outputs(
    target_dir: Path,
    outputs: dict[str, pl.DataFrame],
    workflow: str,
    dropped: dict[str, int] | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.write_parquet(target_dir / f"{name}.parquet")

    manifest_path = PROCESSED_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest.setdefault("workflows", {})[workflow] = {
        "processed_at": datetime.now(UTC).isoformat(),
        "outputs": {name: list(frame.shape) for name, frame in outputs.items()},
        "dropped_unmatched_rows": dropped or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name, frame in outputs.items():
        print(f"{name}: {frame.height} rows, {frame.width} columns")
    for name, count in (dropped or {}).items():
        print(f"{name}: dropped {count} rows without an internal player match")
    print(f"Processed data saved to: {target_dir}")
