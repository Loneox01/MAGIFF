"""Import a provider ADP download into the raw snapshot structure.

This adapter deliberately preserves provider-specific columns. Column mapping,
player reconciliation, and consensus calculation belong in processing.
"""

import argparse
from pathlib import Path

import polars as pl

from .common import write_snapshot


PROVIDERS = ("espn", "sleeper", "yahoo", "fantasypros", "other")
SCORING_FORMATS = ("ppr", "half_ppr", "standard")


def read_source(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=None)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pl.read_ndjson(path) if suffix != ".json" else pl.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError("Supported input types: .csv, .json, .jsonl, .ndjson, .parquet")


def ingest_adp_file(
    path: Path,
    *,
    provider: str,
    season: int,
    scoring: str,
    source_reference: str | None = None,
):
    """Store one provider download without imposing a shared raw schema."""
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = read_source(path)
    output = write_snapshot(
        frame,
        provider=provider,
        dataset="adp",
        season=season,
        filename=f"adp_{scoring}.parquet",
        acquisition_method="provider file import",
        scoring=scoring,
        source_reference=source_reference or path.name,
        extra_metadata={"original_filename": path.name},
    )
    print(f"{provider.title()} {scoring} ADP saved to: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--provider", required=True, choices=PROVIDERS)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--scoring", required=True, choices=SCORING_FORMATS)
    parser.add_argument(
        "--source-reference",
        help="Public page, export name, or other provenance note.",
    )
    args = parser.parse_args()
    ingest_adp_file(
        args.path,
        provider=args.provider,
        season=args.season,
        scoring=args.scoring,
        source_reference=args.source_reference,
    )


if __name__ == "__main__":
    main()

