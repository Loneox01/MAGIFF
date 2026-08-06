"""Normalize nflverse depth-chart schema generations.

To support a future schema, add one adapter to ``ADAPTERS`` with the columns
that uniquely identify it. Processing workflows can continue calling
``normalize_depth_charts`` without knowing which raw format was downloaded.
"""

from collections.abc import Callable

import polars as pl

from ..columns import (
    LEGACY_DEPTH_CHART_COLUMNS,
    NORMALIZED_DEPTH_CHART_COLUMNS,
    TIMESTAMPED_DEPTH_CHART_COLUMNS,
)


DepthChartAdapter = Callable[[pl.DataFrame, int], pl.DataFrame]


def _normalize_legacy_weekly(frame: pl.DataFrame, season: int) -> pl.DataFrame:
    """Normalize the weekly NFL depth-chart format used through 2024."""
    return (
        frame.select(LEGACY_DEPTH_CHART_COLUMNS)
        .with_columns(
            pl.lit(season, dtype=pl.Int32).alias("season"),
            pl.col("week").cast(pl.Int32),
            pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth_rank"),
            pl.col("depth_position").str.strip_chars().replace("", None),
        )
        .with_columns(
            pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("snapshot_at"),
            pl.col("club_code").alias("team"),
            pl.coalesce("full_name", "football_name").alias("player_name"),
            pl.lit(None, dtype=pl.String).alias("espn_id"),
            pl.col("formation").alias("position_group"),
            pl.lit(None, dtype=pl.String).alias("position_name"),
            pl.col("depth_position").alias("position"),
            pl.lit(None, dtype=pl.Int32).alias("position_slot"),
        )
        .rename({"game_type": "season_type"})
        .select(NORMALIZED_DEPTH_CHART_COLUMNS)
    )


def _normalize_timestamped(frame: pl.DataFrame, season: int) -> pl.DataFrame:
    """Normalize the timestamped ESPN-backed format introduced in 2025."""
    timestamp = pl.col("dt")
    if frame.schema["dt"] == pl.String:
        timestamp = timestamp.str.to_datetime(time_zone="UTC", strict=False)

    return (
        frame.select(TIMESTAMPED_DEPTH_CHART_COLUMNS)
        .with_columns(timestamp.alias("snapshot_at"))
        .with_columns(
            pl.lit(season, dtype=pl.Int32).alias("season"),
            pl.lit(None, dtype=pl.Int32).alias("week"),
            pl.lit(None, dtype=pl.String).alias("season_type"),
            pl.col("pos_grp").alias("formation"),
            pl.when(pl.col("pos_grp") == "Special Teams")
            .then(pl.lit("Special Teams"))
            .when(pl.col("pos_grp").str.ends_with("D"))
            .then(pl.lit("Defense"))
            .otherwise(pl.lit("Offense"))
            .alias("position_group"),
            pl.col("pos_name").alias("position_name"),
            pl.col("pos_abb").alias("position"),
            pl.col("pos_slot").alias("position_slot"),
            pl.col("pos_rank").alias("depth_rank"),
            pl.lit(None, dtype=pl.String).alias("jersey_number"),
        )
        .select(NORMALIZED_DEPTH_CHART_COLUMNS)
    )


# Order adapters from most specific to least specific. A future schema only
# needs a new required-column set and normalizer added here.
ADAPTERS: tuple[tuple[str, frozenset[str], DepthChartAdapter], ...] = (
    (
        "legacy_weekly",
        frozenset(LEGACY_DEPTH_CHART_COLUMNS),
        _normalize_legacy_weekly,
    ),
    (
        "timestamped",
        frozenset(TIMESTAMPED_DEPTH_CHART_COLUMNS),
        _normalize_timestamped,
    ),
)


def normalize_depth_charts(frame: pl.DataFrame, season: int) -> pl.DataFrame:
    """Dispatch a raw depth chart to its matching schema adapter."""
    available = set(frame.columns)
    for _name, required, adapter in ADAPTERS:
        if required <= available:
            return adapter(frame, season)

    expected = {name: sorted(columns) for name, columns, _adapter in ADAPTERS}
    raise ValueError(
        "Unsupported depth-chart schema. "
        f"Available columns: {sorted(available)}. Known schemas: {expected}"
    )


def select_current_depth_chart(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the newest complete timestamped snapshot available for each team."""
    timestamped = frame.filter(pl.col("snapshot_at").is_not_null())
    if timestamped.is_empty():
        return frame

    newest = timestamped.group_by("team").agg(
        pl.col("snapshot_at").max()
    )
    return (
        timestamped.join(newest, on=["team", "snapshot_at"], how="inner")
        .unique()
        .sort("team", "position_slot", "depth_rank", "player_name")
    )


def select_historical_depth_charts(
    frame: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    """Reduce depth charts to one pregame snapshot per team and NFL week.

    Legacy depth charts already contain an NFL week and pass through unchanged.
    Timestamped snapshots are assigned to the team's next scheduled game; only
    the newest snapshot before that game is retained.
    """
    if frame.get_column("week").is_not_null().any():
        return frame.unique().sort(
            "season", "season_type", "week", "team", "position", "depth_rank"
        )

    games = pl.concat(
        [
            schedules.select(
                "season",
                "week",
                pl.col("game_type").alias("season_type"),
                pl.col("gameday").str.to_date(strict=False).alias("game_date"),
                pl.col("away_team").alias("team"),
            ),
            schedules.select(
                "season",
                "week",
                pl.col("game_type").alias("season_type"),
                pl.col("gameday").str.to_date(strict=False).alias("game_date"),
                pl.col("home_team").alias("team"),
            ),
        ]
    ).unique()

    snapshots = (
        frame.select("team", "snapshot_at")
        .unique()
        .with_columns(pl.col("snapshot_at").dt.date().alias("snapshot_date"))
        .sort("snapshot_date")
    )
    mapped = snapshots.join_asof(
        games.sort("game_date"),
        left_on="snapshot_date",
        right_on="game_date",
        by="team",
        strategy="forward",
        check_sortedness=False,
    ).filter(pl.col("week").is_not_null())

    selected = mapped.group_by(
        "season", "week", "season_type", "team"
    ).agg(pl.col("snapshot_at").max())

    return (
        frame.drop("season", "week", "season_type")
        .join(selected, on=["team", "snapshot_at"], how="inner")
        .select(NORMALIZED_DEPTH_CHART_COLUMNS)
        .unique()
        .sort(
            "season",
            "season_type",
            "week",
            "team",
            "position_slot",
            "depth_rank",
            "player_name",
        )
    )
