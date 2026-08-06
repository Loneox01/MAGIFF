"""
Upload processed NFL Parquet tables to Supabase in dependency order.

Examples:
    python -m backend.database.load_supabase --dry-run
    python -m backend.database.load_supabase --workflow all
    python -m backend.database.load_supabase --workflow reference
    python -m backend.database.load_supabase --workflow current --season 2026
    python -m backend.database.load_supabase --workflow historical --season 2025
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Literal

import polars as pl
from dotenv import load_dotenv
from postgrest.types import ReturnMethod
from supabase import Client, create_client


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
SEASONS_DIR = PROCESSED_DIR / "seasons"

LoadMode = Literal["upsert", "replace_season"]
Workflow = Literal["all", "reference", "current", "historical"]


@dataclass(frozen=True)
class LoadSpec:
    table: str
    path: Path
    conflict_columns: str | None
    mode: LoadMode = "upsert"
    season: int | None = None


def processed_seasons() -> list[int]:
    """Return processed season directories in chronological order."""
    if not SEASONS_DIR.exists():
        return []
    return sorted(
        int(path.name)
        for path in SEASONS_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def add_if_present(specs: list[LoadSpec], spec: LoadSpec) -> None:
    """Add optional current outputs only when processing produced them."""
    if spec.path.exists():
        specs.append(spec)


def reference_specs() -> list[LoadSpec]:
    return [
        LoadSpec(
            "players",
            PROCESSED_DIR / "reference" / "players.parquet",
            "player_id",
        ),
        LoadSpec(
            "teams",
            PROCESSED_DIR / "reference" / "teams.parquet",
            "team_abbr",
        ),
        LoadSpec(
            "player_external_ids",
            PROCESSED_DIR / "reference" / "player_external_ids.parquet",
            "provider,external_id",
        ),
        LoadSpec(
            "player_status",
            PROCESSED_DIR / "current" / "player_status.parquet",
            "player_id",
        ),
    ]


def season_specs(directory: Path, season: int) -> list[LoadSpec]:
    """Return dependency-ordered tables shared by current and history."""
    definitions = [
        ("games", "games.parquet", "game_id"),
        ("player_weekly_stats", "player_weekly_stats.parquet", "player_id,game_id"),
        (
            "player_season_stats",
            "player_season_stats.parquet",
            "player_id,season,season_type",
        ),
        (
            "player_weekly_rosters",
            "player_weekly_rosters.parquet",
            "player_id,season,week,game_type,team",
        ),
        ("player_snap_counts", "player_snap_counts.parquet", "player_id,game_id"),
        ("team_weekly_stats", "team_weekly_stats.parquet", "team,game_id"),
    ]
    return [
        LoadSpec(table, directory / filename, conflict, season=season)
        for table, filename, conflict in definitions
        if (directory / filename).exists()
    ]


def historical_specs(season: int) -> list[LoadSpec]:
    directory = SEASONS_DIR / str(season)
    if not directory.exists():
        raise FileNotFoundError(f"Processed season not found: {directory}")
    specs = season_specs(directory, season)
    depth_path = directory / "depth_chart_entries.parquet"
    if depth_path.exists():
        specs.append(
            LoadSpec(
                "depth_chart_entries",
                depth_path,
                (
                    "season,season_type,week,team,player_id,formation,"
                    "position,position_slot,depth_rank"
                ),
                season=season,
            )
        )
    return specs


def current_specs(season: int) -> list[LoadSpec]:
    """Return active-season tables, including weekly files when available."""
    directory = PROCESSED_DIR / "current"
    specs: list[LoadSpec] = []
    add_if_present(
        specs,
        LoadSpec("player_status", directory / "player_status.parquet", "player_id"),
    )
    specs.extend(season_specs(directory, season))
    add_if_present(
        specs,
        LoadSpec(
            "current_depth_chart_entries",
            directory / "depth_chart_entries.parquet",
            conflict_columns=None,
            mode="replace_season",
            season=season,
        ),
    )
    return specs


def build_load_plan(workflow: Workflow, season: int | None) -> list[LoadSpec]:
    """Build the requested frequency-aware, dependency-ordered upload plan."""
    if workflow in {"current", "historical"} and season is None:
        raise ValueError(f"--season is required for the {workflow} workflow")

    if workflow == "reference":
        return reference_specs()
    if workflow == "current":
        return current_specs(season)  # type: ignore[arg-type]
    if workflow == "historical":
        return historical_specs(season)  # type: ignore[arg-type]

    specs = reference_specs()
    for historical_season in processed_seasons():
        specs.extend(historical_specs(historical_season))

    current_depth_path = PROCESSED_DIR / "current" / "depth_chart_entries.parquet"
    if current_depth_path.exists():
        current_frame = pl.read_parquet(current_depth_path, columns=["season"])
        current_seasons = current_frame.get_column("season").drop_nulls().unique()
        if len(current_seasons) != 1:
            raise ValueError(
                "Current depth data must contain exactly one season; got "
                f"{current_seasons.to_list()}"
            )
        specs.extend(
            spec
            for spec in current_specs(int(current_seasons.item()))
            if spec.table != "player_status"
        )
    return specs


def batches(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def read_json_rows(path: Path) -> list[dict]:
    """Read Parquet and convert values into JSON-compatible objects."""
    frame = pl.read_parquet(path)
    conversions = []

    for column, dtype in frame.schema.items():
        if dtype.is_temporal():
            conversions.append(pl.col(column).cast(pl.String))
        elif dtype.is_float():
            conversions.append(pl.col(column).fill_nan(None))

    if conversions:
        frame = frame.with_columns(conversions)
    return frame.to_dicts()


def upload_batches(
    client: Client,
    spec: LoadSpec,
    rows: list[dict],
    batch_size: int,
) -> None:
    """Insert or upsert prepared rows in bounded API requests."""
    total_batches = (len(rows) + batch_size - 1) // batch_size
    for number, batch in enumerate(batches(rows, batch_size), start=1):
        query = client.table(spec.table)
        if spec.mode == "upsert":
            query.upsert(
                batch,
                on_conflict=spec.conflict_columns,
                returning=ReturnMethod.minimal,
            ).execute()
        else:
            query.insert(batch, returning=ReturnMethod.minimal).execute()
        print(f"{spec.table}: uploaded batch {number}/{total_batches}")


def replace_season_rows(
    client: Client,
    spec: LoadSpec,
    rows: list[dict],
    batch_size: int,
) -> None:
    """Replace one current-season snapshot, then insert its complete new state."""
    seasons = {row.get("season") for row in rows}
    if None in seasons or len(seasons) != 1:
        raise ValueError(
            f"{spec.path} must contain exactly one non-null season; got {seasons}"
        )
    season = seasons.pop()

    # TODO(production weekly refresh): replace this two-request delete/insert
    # workflow with a transactional Supabase RPC or staging-table swap. If an
    # insert currently fails after deletion, rerun this loader from the local
    # Parquet snapshot to restore the season.
    (
        client.table(spec.table)
        .delete(returning=ReturnMethod.minimal)
        .eq("season", season)
        .execute()
    )
    print(f"{spec.table}: removed previous season {season} snapshot")
    upload_batches(client, spec, rows, batch_size)


def load_data(
    batch_size: int = 500,
    dry_run: bool = False,
    workflow: Workflow = "all",
    season: int | None = None,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    plan = build_load_plan(workflow, season)
    prepared: list[tuple[LoadSpec, list[dict]]] = []
    for spec in plan:
        if not spec.path.exists():
            raise FileNotFoundError(f"Processed file not found: {spec.path}")
        rows = read_json_rows(spec.path)
        row_seasons = {
            row["season"] for row in rows
            if "season" in row and row["season"] is not None
        }
        if spec.season is not None and row_seasons and row_seasons != {spec.season}:
            raise ValueError(
                f"{spec.path} expected season {spec.season}; got {row_seasons}"
            )
        prepared.append((spec, rows))
        season_label = f", season {spec.season}" if spec.season else ""
        print(
            f"{spec.table}: prepared {len(rows)} rows "
            f"from {spec.path.relative_to(PROCESSED_DIR)}"
            f" ({spec.mode}{season_label})"
        )

    if dry_run:
        print("Dry run complete; no data was uploaded or deleted.")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not secret_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY not set in root .env"
        )

    client = create_client(url, secret_key)
    for spec, rows in prepared:
        if spec.mode == "replace_season":
            replace_season_rows(client, spec, rows, batch_size)
        else:
            upload_batches(client, spec, rows, batch_size)

    print("Supabase upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--workflow",
        choices=("all", "reference", "current", "historical"),
        default="all",
        help="Upload only data belonging to this refresh frequency.",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Required for current and historical workflows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate local files without contacting Supabase.",
    )
    args = parser.parse_args()
    load_data(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        workflow=args.workflow,
        season=args.season,
    )


if __name__ == "__main__":
    main()
