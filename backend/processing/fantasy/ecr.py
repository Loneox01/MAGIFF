"""Process supported overall nflverse ECR ranking pages."""

import argparse
from dataclasses import dataclass
from datetime import date

import polars as pl

from ..normalization.team_codes import normalize_team_codes
from ..workflows.common import validate_unique, write_outputs
from .common import PROCESSED_DIR, current_ecr_path, ecr_archive_path, player_ids_path


@dataclass(frozen=True)
class RankingConfig:
    page_type: str
    scoring_format: str
    league_format: str
    historical_snapshot_type: str


# Complete ranking pools only; positional pages duplicate these player pools.
RANKING_CONFIGS = (
    RankingConfig("redraft-overall", "ppr", "redraft_1qb", "final_preseason"),
    RankingConfig("redraft-op", "ppr", "redraft_superflex", "final_preseason"),
    RankingConfig("dynasty-overall", "source_default", "dynasty_1qb", "season_opening"),
    RankingConfig("dynasty-op", "source_default", "dynasty_superflex", "season_opening"),
    RankingConfig("dynasty-rk", "source_default", "dynasty_rookie", "season_opening"),
    RankingConfig("best-overall", "source_default", "best_ball", "season_opening"),
    RankingConfig("redraft-idp", "source_default", "redraft_idp", "final_preseason"),
    RankingConfig("dynasty-idp", "source_default", "dynasty_idp", "season_opening"),
)


def ecr_identity_map(reference_season: int) -> pl.DataFrame:
    """Map FantasyPros IDs to internal UUIDs through GSIS IDs."""
    crosswalk = pl.read_parquet(
        player_ids_path(reference_season),
        columns=["fantasypros_id", "gsis_id"],
    ).with_columns(pl.col("fantasypros_id").cast(pl.String, strict=False))
    players = pl.read_parquet(
        PROCESSED_DIR / "reference" / "players.parquet",
        columns=["player_id", "gsis_id"],
    )
    return (
        crosswalk.filter(
            pl.col("fantasypros_id").is_not_null()
            & pl.col("gsis_id").is_not_null()
        )
        .join(players, on="gsis_id", how="inner")
        .select("fantasypros_id", "player_id")
        .unique("fantasypros_id")
    )


def normalize_ranking_page(
    frame: pl.DataFrame,
    config: RankingConfig,
    *,
    season: int,
    identity_map: pl.DataFrame,
    snapshot_type: str,
) -> tuple[pl.DataFrame, int]:
    """Normalize one complete ranking page and attach internal player IDs."""
    filtered = (
        frame.filter(pl.col("page_type") == config.page_type)
        .with_columns(
            pl.col("id").cast(pl.String, strict=False).alias("fantasypros_id"),
            pl.col("scrape_date").str.to_date(strict=False),
        )
    )
    joined = filtered.join(identity_map, on="fantasypros_id", how="left")
    unmatched = joined.filter(pl.col("player_id").is_null()).height
    result = normalize_team_codes(
        joined.filter(pl.col("player_id").is_not_null())
        .select(
            "player_id",
            pl.lit(season).cast(pl.Int32).alias("season"),
            "scrape_date",
            pl.lit(config.scoring_format).alias("scoring_format"),
            pl.lit(config.league_format).alias("league_format"),
            pl.lit(snapshot_type).alias("snapshot_type"),
            pl.lit(config.page_type).alias("ranking_page"),
            pl.col("ecr").alias("overall_rank"),
            pl.col("best").cast(pl.Int32, strict=False).alias("best_rank"),
            pl.col("worst").cast(pl.Int32, strict=False).alias("worst_rank"),
            pl.col("sd").alias("rank_sd"),
            pl.col("rank_delta").cast(pl.Int32, strict=False),
            pl.col("pos").alias("position"),
            pl.coalesce("team", "tm").alias("team"),
            pl.lit("fantasypros_ecr_via_nflverse").alias("source"),
        )
        .sort("overall_rank")
    )
    return result, unmatched


def validate_ecr(frame: pl.DataFrame) -> None:
    validate_unique(
        frame,
        ["player_id", "season", "scrape_date", "scoring_format", "league_format"],
        "player_ecr",
    )


def process_current_ecr(season: int, reference_season: int | None = None) -> None:
    raw = pl.read_parquet(current_ecr_path(season))
    identities = ecr_identity_map(reference_season or season)
    outputs = []
    dropped = {}
    for config in RANKING_CONFIGS:
        output, unmatched = normalize_ranking_page(
            raw,
            config,
            season=season,
            identity_map=identities,
            snapshot_type="current",
        )
        if not output.is_empty():
            outputs.append(output)
        dropped[config.page_type] = unmatched

    combined = pl.concat(outputs, how="vertical_relaxed").sort(
        "league_format", "overall_rank"
    )
    validate_ecr(combined)
    write_outputs(
        PROCESSED_DIR / "current",
        {"player_ecr": combined},
        workflow=f"current_ecr_{season}",
        dropped=dropped,
    )


def first_regular_season_game(season: int) -> date:
    games_path = PROCESSED_DIR / "seasons" / str(season) / "games.parquet"
    games = pl.read_parquet(games_path, columns=["game_type", "gameday"])
    first_game = (
        games.filter(pl.col("game_type") == "REG")
        .select(pl.col("gameday").min())
        .item()
    )
    if first_game is None:
        raise ValueError(f"No regular-season games found for {season}")
    return first_game


def process_historical_ecr(
    season: int,
    reference_season: int,
    cutoff: date | None = None,
) -> None:
    """Select each page's last snapshot before the first regular game."""
    cutoff_date = cutoff or first_regular_season_game(season)
    archive = (
        pl.scan_parquet(ecr_archive_path())
        .with_columns(pl.col("scrape_date").str.to_date(strict=False).alias("_date"))
        .filter(
            pl.col("page_type").is_in(
                [config.page_type for config in RANKING_CONFIGS]
            )
            & (pl.col("_date").dt.year() == season)
            & (pl.col("_date") < cutoff_date)
        )
        .collect()
    )
    identities = ecr_identity_map(reference_season)
    outputs = []
    dropped = {}
    for config in RANKING_CONFIGS:
        page = archive.filter(pl.col("page_type") == config.page_type)
        snapshot_date = page.select(pl.col("_date").max()).item()
        if snapshot_date is None:
            dropped[f"{config.page_type}_missing_snapshot"] = 0
            continue
        snapshot = (
            page.filter(pl.col("_date") == snapshot_date)
            .drop("_date")
            .with_columns(pl.col("scrape_date").cast(pl.String))
        )
        output, unmatched = normalize_ranking_page(
            snapshot,
            config,
            season=season,
            identity_map=identities,
            snapshot_type=config.historical_snapshot_type,
        )
        if not output.is_empty():
            outputs.append(output)
        dropped[config.page_type] = unmatched

    combined = pl.concat(outputs, how="vertical_relaxed").sort(
        "league_format", "overall_rank"
    )
    validate_ecr(combined)
    write_outputs(
        PROCESSED_DIR / "seasons" / str(season),
        {"player_ecr": combined},
        workflow=f"historical_ecr_{season}",
        dropped=dropped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    current_parser = subparsers.add_parser("current")
    current_parser.add_argument("--season", type=int, required=True)
    current_parser.add_argument("--reference-season", type=int)

    historical_parser = subparsers.add_parser("historical")
    historical_parser.add_argument("--season", type=int, required=True)
    historical_parser.add_argument("--reference-season", type=int, required=True)
    historical_parser.add_argument("--cutoff", type=date.fromisoformat)

    args = parser.parse_args()
    if args.command == "current":
        process_current_ecr(args.season, args.reference_season)
    else:
        process_historical_ecr(args.season, args.reference_season, args.cutoff)


if __name__ == "__main__":
    main()
