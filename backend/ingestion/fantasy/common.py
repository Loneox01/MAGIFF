"""Shared paths and snapshot helpers for fantasy-market ingestion."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl


BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_DIR / "data" / "raw" / "fantasy"


def snapshot_dir(provider: str, dataset: str, season: int | str) -> Path:
    """Return today's raw snapshot directory for one provider dataset."""
    return (
        RAW_DIR
        / provider.lower()
        / dataset.lower()
        / str(season)
        / "snapshots"
        / date.today().isoformat()
    )


def write_snapshot(
    frame: pl.DataFrame,
    *,
    provider: str,
    dataset: str,
    season: int | str,
    filename: str,
    acquisition_method: str,
    scoring: str | None = None,
    source_reference: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a provider-specific frame and acquisition metadata.

    A repeat run on the same date replaces only the matching provider file. This
    avoids uncontrolled intraday snapshot growth while retaining daily history.
    """
    target_dir = snapshot_dir(provider, dataset, season)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename
    frame.write_parquet(output_path)

    metadata_path = target_dir / f"{output_path.stem}.snapshot.json"
    metadata = {
        "provider": provider,
        "dataset": dataset,
        "season": season,
        "scoring": scoring,
        "acquisition_method": acquisition_method,
        "source_reference": source_reference,
        "acquired_at": datetime.now(UTC).isoformat(),
        "rows": frame.height,
        "columns": frame.width,
        "column_names": frame.columns,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path

