"""Shared transformations used by historical and current season workflows."""

from pathlib import Path

import polars as pl

from ..columns import (
    GAME_COLUMNS,
    SNAP_COUNT_COLUMNS,
    TEAM_WEEKLY_STAT_COLUMNS,
    WEEKLY_ROSTER_COLUMNS,
    WEEKLY_STAT_COLUMNS,
)
from ..normalization.depth_charts import (
    normalize_depth_charts,
    select_current_depth_chart,
    select_historical_depth_charts,
)
from ..player_season_stats import build_player_season_stats
from .common import (
    attach_depth_chart_player_id,
    attach_player_id,
    player_identities,
)


SEASON_DATASETS = (
    "weekly_stats",
    "weekly_rosters",
    "schedules",
    "snap_counts",
    "team_weekly_stats",
)


def process_season_tables(
    raw_dir: Path,
    season: int,
    raw_players: pl.DataFrame,
    *,
    historical: bool,
    strict: bool,
) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    """Process all season datasets present in ``raw_dir``.

    Historical runs are strict and require the completed-season datasets.
    Current runs process only data that is available so preseason refreshes
    can still publish schedules and depth charts.
    """
    if strict:
        missing = [
            name for name in SEASON_DATASETS
            if not (raw_dir / f"{name}.parquet").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required season datasets in {raw_dir}: {missing}"
            )

    identities = player_identities(raw_players)
    outputs: dict[str, pl.DataFrame] = {}
    dropped: dict[str, int] = {}

    schedules_path = raw_dir / "schedules.parquet"
    raw_schedules = (
        pl.read_parquet(schedules_path) if schedules_path.exists() else None
    )
    if raw_schedules is not None:
        outputs["games"] = raw_schedules.select(GAME_COLUMNS).with_columns(
            pl.col("gameday").str.to_date(strict=False)
        )

    weekly_stats_path = raw_dir / "weekly_stats.parquet"
    if weekly_stats_path.exists():
        weekly_input = (
            pl.read_parquet(weekly_stats_path)
            .select(WEEKLY_STAT_COLUMNS)
            .rename({"player_id": "gsis_id"})
        )
        weekly_stats, unmatched = attach_player_id(weekly_input, identities)
        weekly_stats = weekly_stats.select(
            "player_id",
            "gsis_id",
            *[column for column in WEEKLY_STAT_COLUMNS if column != "player_id"],
        )
        outputs["player_weekly_stats"] = weekly_stats
        outputs["player_season_stats"] = build_player_season_stats(weekly_stats)
        dropped["player_weekly_stats"] = unmatched

    rosters_path = raw_dir / "weekly_rosters.parquet"
    if rosters_path.exists():
        rosters, unmatched = attach_player_id(
            pl.read_parquet(rosters_path).select(WEEKLY_ROSTER_COLUMNS),
            identities,
        )
        outputs["player_weekly_rosters"] = rosters.select(
            "player_id", *WEEKLY_ROSTER_COLUMNS
        )
        dropped["player_weekly_rosters"] = unmatched

    snap_counts_path = raw_dir / "snap_counts.parquet"
    if snap_counts_path.exists():
        pfr_ids = (
            raw_players.select("gsis_id", "pfr_id")
            .join(identities.select("gsis_id", "player_id"), on="gsis_id")
            .select("pfr_id", "player_id")
        )
        snap_counts = (
            pl.read_parquet(snap_counts_path)
            .select(SNAP_COUNT_COLUMNS)
            .join(
                pfr_ids,
                left_on="pfr_player_id",
                right_on="pfr_id",
                how="left",
            )
        )
        unmatched = snap_counts.filter(pl.col("player_id").is_null()).height
        outputs["player_snap_counts"] = snap_counts.filter(
            pl.col("player_id").is_not_null()
        ).select("player_id", *SNAP_COUNT_COLUMNS)
        dropped["player_snap_counts"] = unmatched

    team_stats_path = raw_dir / "team_weekly_stats.parquet"
    if team_stats_path.exists():
        outputs["team_weekly_stats"] = pl.read_parquet(team_stats_path).select(
            TEAM_WEEKLY_STAT_COLUMNS
        )

    depth_path = raw_dir / "depth_charts.parquet"
    if depth_path.exists():
        depth_charts = normalize_depth_charts(
            pl.read_parquet(depth_path), season
        )
        if historical:
            if raw_schedules is None:
                raise FileNotFoundError(
                    "Historical timestamped depth charts require schedules"
                )
            depth_charts = select_historical_depth_charts(
                depth_charts, raw_schedules
            )
        else:
            depth_charts = select_current_depth_chart(depth_charts)

        depth_charts, unmatched = attach_depth_chart_player_id(
            depth_charts, identities
        )
        outputs["depth_chart_entries"] = depth_charts.select(
            "player_id",
            *[column for column in depth_charts.columns if column != "player_id"],
        ).sort(
            "season", "week", "team", "position_slot", "depth_rank"
        )
        dropped["depth_chart_entries"] = unmatched

    return outputs, dropped
