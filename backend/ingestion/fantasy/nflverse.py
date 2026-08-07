"""Ingest nflverse fantasy ECR rankings and cross-platform player IDs."""

import argparse

import nflreadpy as nfl

from .common import write_snapshot


def ingest_current_ecr(season: int | None = None):
    """Download the latest FantasyPros draft ECR exposed by nflverse."""
    target_season = season or nfl.get_current_season(roster=True)
    frame = nfl.load_ff_rankings(type="draft")
    output = write_snapshot(
        frame,
        provider="nflverse",
        dataset="ecr",
        season=target_season,
        filename="draft_ecr.parquet",
        acquisition_method="nflreadpy.load_ff_rankings(type='draft')",
        source_reference="DynastyProcess/FantasyPros ECR via nflverse",
        extra_metadata={"ranking_scope": "latest_draft"},
    )
    print(f"Current ECR saved to: {output}")
    return output


def ingest_ecr_archive():
    """Download the complete historical ECR archive for later processing."""
    frame = nfl.load_ff_rankings(type="all")
    output = write_snapshot(
        frame,
        provider="nflverse",
        dataset="ecr",
        season="archive",
        filename="ecr_archive.parquet",
        acquisition_method="nflreadpy.load_ff_rankings(type='all')",
        source_reference="DynastyProcess/FantasyPros ECR archive via nflverse",
        extra_metadata={"ranking_scope": "historical_archive"},
    )
    print(f"ECR archive saved to: {output}")
    return output


def ingest_player_ids(season: int | None = None):
    """Download IDs used to reconcile names across ranking providers."""
    reference_season = season or nfl.get_current_season(roster=True)
    frame = nfl.load_ff_playerids()
    output = write_snapshot(
        frame,
        provider="nflverse",
        dataset="player_ids",
        season=reference_season,
        filename="fantasy_player_ids.parquet",
        acquisition_method="nflreadpy.load_ff_playerids()",
        source_reference="DynastyProcess player ID database via nflverse",
    )
    print(f"Fantasy player IDs saved to: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    current_parser = subparsers.add_parser("current-ecr")
    current_parser.add_argument("--season", type=int)

    subparsers.add_parser("ecr-archive")

    ids_parser = subparsers.add_parser("player-ids")
    ids_parser.add_argument("--season", type=int)

    args = parser.parse_args()
    if args.command == "current-ecr":
        ingest_current_ecr(args.season)
    elif args.command == "ecr-archive":
        ingest_ecr_archive()
    else:
        ingest_player_ids(args.season)


if __name__ == "__main__":
    main()

