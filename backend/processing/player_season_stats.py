"""
Build player season totals from processed weekly player statistics.

Include arg --season xxxx for processing non 2025 seasons when running.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from processing.columns import PLAYER_SEASON_ADDITIVE_COLUMNS


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
DEFAULT_SEASON = 2025

def ratio(numerator: str, denominator: str, alias: str, scale: float = 1.0) -> pl.Expr:
    """Return a null-safe ratio calculated from season totals."""
    return (
        pl.when(pl.col(denominator) != 0)
        .then(pl.col(numerator) * scale / pl.col(denominator))
        .otherwise(None)
        .round(3)
        .alias(alias)
    )


def build_player_season_stats(weekly_stats: pl.DataFrame) -> pl.DataFrame:
    """Aggregate one row per player, season, and season type."""
    season_stats = (
        weekly_stats.sort("season", "season_type", "player_id", "week")
        .group_by("player_id", "gsis_id", "season", "season_type")
        .agg(
            pl.col("game_id").n_unique().alias("games"),
            pl.col("team").unique(maintain_order=True).alias("teams"),
            pl.col("team").last().alias("last_team"),
            pl.col("position").last(),
            pl.col("position_group").last(),
            *[
                pl.col(column).sum().alias(column)
                for column in PLAYER_SEASON_ADDITIVE_COLUMNS
            ],
            pl.col("passing_epa").sum().alias("passing_epa"),
            (pl.col("passing_cpoe") * pl.col("attempts"))
            .sum()
            .alias("passing_cpoe_weighted_sum"),
            pl.col("rushing_epa").sum().alias("rushing_epa"),
            pl.col("receiving_epa").sum().alias("receiving_epa"),
        )
        .with_columns(
            pl.when(pl.col("attempts") > 0)
            .then(pl.col("passing_epa"))
            .otherwise(None)
            .alias("passing_epa"),
            pl.when(pl.col("carries") > 0)
            .then(pl.col("rushing_epa"))
            .otherwise(None)
            .alias("rushing_epa"),
            pl.when(pl.col("targets") > 0)
            .then(pl.col("receiving_epa"))
            .otherwise(None)
            .alias("receiving_epa"),
            pl.col("fantasy_points").round(2),
            pl.col("fantasy_points_ppr").round(2),
        )
        .with_columns(
            ratio("completions", "attempts", "completion_percentage", 100),
            ratio("passing_yards", "attempts", "passing_yards_per_attempt"),
            ratio("passing_epa", "attempts", "passing_epa_per_attempt"),
            ratio(
                "passing_cpoe_weighted_sum",
                "attempts",
                "passing_cpoe",
            ),
            ratio("passing_yards", "passing_air_yards", "pacr"),
            ratio("rushing_yards", "carries", "rushing_yards_per_carry"),
            ratio("rushing_epa", "carries", "rushing_epa_per_carry"),
            ratio("receptions", "targets", "catch_percentage", 100),
            ratio(
                "receiving_yards",
                "receptions",
                "receiving_yards_per_reception",
            ),
            ratio("receiving_yards", "targets", "receiving_yards_per_target"),
            ratio("receiving_epa", "targets", "receiving_epa_per_target"),
            ratio("receiving_yards", "receiving_air_yards", "racr"),
            ratio("fantasy_points", "games", "fantasy_points_per_game"),
            ratio(
                "fantasy_points_ppr",
                "games",
                "fantasy_points_ppr_per_game",
            ),
        )
        .drop("passing_cpoe_weighted_sum")
        .sort("season", "season_type", "player_id")
    )

    duplicates = (
        season_stats.group_by("player_id", "season", "season_type")
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ValueError(
            f"player_season_stats has {duplicates.height} duplicate logical keys"
        )

    return season_stats


def process_player_season_stats(season: int) -> Path:
    season_dir = PROCESSED_DIR / "seasons" / str(season)
    weekly_path = season_dir / "player_weekly_stats.parquet"
    output_path = season_dir / "player_season_stats.parquet"

    if not weekly_path.exists():
        raise FileNotFoundError(f"Processed weekly stats not found: {weekly_path}")

    weekly_stats = pl.read_parquet(weekly_path)
    season_stats = build_player_season_stats(weekly_stats)
    season_stats.write_parquet(output_path)

    manifest_path = PROCESSED_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("outputs", {})["player_season_stats"] = list(
            season_stats.shape
        )
        manifest["season_stats_processed_at"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"player_season_stats: {season_stats.height} rows, "
        f"{season_stats.width} columns"
    )
    print(f"Saved to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    args = parser.parse_args()
    process_player_season_stats(args.season)


if __name__ == "__main__":
    main()
