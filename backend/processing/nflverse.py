"""Normalize raw nflverse snapshots into application-ready Parquet tables.

The raw layer remains an untouched copy of nflverse. This module selects the
fields the first version of Magiff can query, assigns deterministic internal
player UUIDs, validates joins, and writes source-independent processed tables.

Deferred for now:
- Individual defensive player statistics
- Punt and kick return statistics
- Detailed kicking distance buckets
- Provider-specific duplicate name and status fields
- Extra team branding assets and provider-specific schedule IDs
"""

from datetime import UTC, datetime
import json
from pathlib import Path
from uuid import UUID, uuid5

import polars as pl

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BACKEND_DIR / "data" / "raw" / "nflverse"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
SEASON = 2025

# A fixed namespace makes a GSIS ID resolve to the same internal UUID on every run.
PLAYER_ID_NAMESPACE = UUID("f0cf9ef7-d56d-4a6f-a8e1-197be003c7a8")

from columns import (
    EXTERNAL_ID_COLUMNS,
    GAME_COLUMNS,
    PLAYER_COLUMNS,
    PLAYER_STATUS_COLUMNS,
    SNAP_COUNT_COLUMNS,
    TEAM_COLUMNS,
    TEAM_WEEKLY_STAT_COLUMNS,
    WEEKLY_ROSTER_COLUMNS,
    WEEKLY_STAT_COLUMNS,
)


def internal_player_id(gsis_id: str) -> str:
    return str(uuid5(PLAYER_ID_NAMESPACE, f"gsis:{gsis_id}"))


def player_id_map(players: pl.DataFrame) -> pl.DataFrame:
    return players.select("gsis_id").with_columns(
        pl.col("gsis_id")
        .map_elements(internal_player_id, return_dtype=pl.String)
        .alias("player_id")
    )


def join_player_id(
    frame: pl.DataFrame,
    ids: pl.DataFrame,
    external_column: str,
) -> tuple[pl.DataFrame, int]:
    joined = frame.join(
        ids,
        left_on=external_column,
        right_on="gsis_id",
        how="left",
    )
    unmatched = joined.filter(pl.col("player_id").is_null()).height
    return joined.filter(pl.col("player_id").is_not_null()), unmatched


def process_players(
    raw_players: pl.DataFrame,
    ids: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    players = (
        raw_players.select(PLAYER_COLUMNS)
        .join(ids, on="gsis_id", how="inner")
        .with_columns(
            pl.col("birth_date").str.to_date(strict=False),
        )
        .select("player_id", *PLAYER_COLUMNS)
    )

    external_ids = (
        raw_players.select(*EXTERNAL_ID_COLUMNS)
        .join(ids, on="gsis_id", how="inner")
        .unpivot(
            index="player_id",
            on=EXTERNAL_ID_COLUMNS,
            variable_name="provider",
            value_name="external_id",
        )
        .filter(
            pl.col("external_id").is_not_null()
            & (pl.col("external_id").str.strip_chars() != "")
        )
        .with_columns(
            pl.col("provider").str.replace(r"_id$", ""),
        )
        .unique(subset=["provider", "external_id"])
        .sort("player_id", "provider")
    )

    current_status = (
        raw_players.select(PLAYER_STATUS_COLUMNS)
        .join(ids, on="gsis_id", how="inner")
        .select("player_id", *PLAYER_STATUS_COLUMNS)
    )

    return players, external_ids, current_status


def process_season(
    season: int,
    ids: pl.DataFrame,
    raw_players: pl.DataFrame,
) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    raw_season_dir = RAW_DIR / str(season)

    raw_weekly_stats = pl.read_parquet(raw_season_dir / "weekly_stats.parquet")
    raw_weekly_rosters = pl.read_parquet(raw_season_dir / "weekly_rosters.parquet")
    raw_schedules = pl.read_parquet(raw_season_dir / "schedules.parquet")
    raw_snap_counts = pl.read_parquet(raw_season_dir / "snap_counts.parquet")
    raw_team_stats = pl.read_parquet(raw_season_dir / "team_weekly_stats.parquet")

    # nflverse calls the external GSIS value player_id in this one dataset.
    weekly_stats_input = raw_weekly_stats.select(WEEKLY_STAT_COLUMNS).rename(
        {"player_id": "gsis_id"}
    )
    weekly_stats, unmatched_stats = join_player_id(
        weekly_stats_input,
        ids,
        "gsis_id",
    )
    weekly_stats = weekly_stats.select(
        "player_id",
        "gsis_id",
        *[column for column in WEEKLY_STAT_COLUMNS if column != "player_id"],
    )

    weekly_rosters, unmatched_rosters = join_player_id(
        raw_weekly_rosters.select(WEEKLY_ROSTER_COLUMNS),
        ids,
        "gsis_id",
    )
    weekly_rosters = weekly_rosters.select(
        "player_id",
        *WEEKLY_ROSTER_COLUMNS,
    )

    pfr_ids = raw_players.select("gsis_id", "pfr_id").join(ids, on="gsis_id").select(
        "pfr_id", "player_id"
    )
    snap_counts = raw_snap_counts.select(SNAP_COUNT_COLUMNS).join(
        pfr_ids,
        left_on="pfr_player_id",
        right_on="pfr_id",
        how="left",
    )
    unmatched_snaps = snap_counts.filter(pl.col("player_id").is_null()).height
    snap_counts = snap_counts.filter(pl.col("player_id").is_not_null()).select(
        "player_id",
        *SNAP_COUNT_COLUMNS,
    )

    games = raw_schedules.select(GAME_COLUMNS).with_columns(
        pl.col("gameday").str.to_date(strict=False),
    )
    team_weekly_stats = raw_team_stats.select(TEAM_WEEKLY_STAT_COLUMNS)

    outputs = {
        "games": games,
        "player_weekly_stats": weekly_stats,
        "player_weekly_rosters": weekly_rosters,
        "player_snap_counts": snap_counts,
        "team_weekly_stats": team_weekly_stats,
    }
    dropped = {
        "player_weekly_stats": unmatched_stats,
        "player_weekly_rosters": unmatched_rosters,
        "player_snap_counts": unmatched_snaps,
    }
    return outputs, dropped


def validate_unique(
    frame: pl.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    duplicates = frame.group_by(columns).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(f"{name} has {duplicates.height} duplicate logical keys")


def process_data(season: int) -> None:
    raw_players = pl.read_parquet(RAW_DIR / "players.parquet")
    raw_teams = pl.read_parquet(RAW_DIR / "teams.parquet")
    ids = player_id_map(raw_players)

    players, external_ids, current_status = process_players(raw_players, ids)
    teams = raw_teams.select(TEAM_COLUMNS)
    season_outputs, dropped = process_season(season, ids, raw_players)

    validate_unique(players, ["player_id"], "players")
    validate_unique(
        season_outputs["player_weekly_stats"],
        ["player_id", "season", "week", "season_type", "game_id"],
        "player_weekly_stats",
    )
    validate_unique(
        season_outputs["player_weekly_rosters"],
        ["player_id", "season", "week", "team"],
        "player_weekly_rosters",
    )
    validate_unique(
        season_outputs["player_snap_counts"],
        ["player_id", "game_id"],
        "player_snap_counts",
    )
    validate_unique(season_outputs["games"], ["game_id"], "games")
    validate_unique(
        season_outputs["team_weekly_stats"],
        ["team", "season", "week", "season_type", "game_id"],
        "team_weekly_stats",
    )

    reference_dir = PROCESSED_DIR / "reference"
    current_dir = PROCESSED_DIR / "current"
    season_dir = PROCESSED_DIR / "seasons" / str(season)
    for directory in (reference_dir, current_dir, season_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reference_outputs = {
        "players": players,
        "player_external_ids": external_ids,
        "teams": teams,
    }
    current_outputs = {"player_status": current_status}

    for name, frame in reference_outputs.items():
        frame.write_parquet(reference_dir / f"{name}.parquet")
    for name, frame in current_outputs.items():
        frame.write_parquet(current_dir / f"{name}.parquet")
    for name, frame in season_outputs.items():
        frame.write_parquet(season_dir / f"{name}.parquet")

    manifest = {
        "season": season,
        "processed_at": datetime.now(UTC).isoformat(),
        "outputs": {
            **{name: list(frame.shape) for name, frame in reference_outputs.items()},
            **{name: list(frame.shape) for name, frame in current_outputs.items()},
            **{name: list(frame.shape) for name, frame in season_outputs.items()},
        },
        "dropped_unmatched_rows": dropped,
    }
    (PROCESSED_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for name, shape in manifest["outputs"].items():
        print(f"{name}: {shape[0]} rows, {shape[1]} columns")
    for name, count in dropped.items():
        print(f"{name}: dropped {count} rows without an internal player match")
    print(f"Processed data saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    process_data(SEASON)
