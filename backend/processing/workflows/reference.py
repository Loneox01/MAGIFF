"""Process slowly changing player and team reference data."""

import polars as pl

from ..columns import (
    EXTERNAL_ID_COLUMNS,
    PLAYER_COLUMNS,
    PLAYER_STATUS_COLUMNS,
    TEAM_COLUMNS,
)
from .common import (
    PROCESSED_DIR,
    latest_reference_dir,
    player_identities,
    validate_unique,
    write_outputs,
)


def process_reference() -> None:
    raw_dir = latest_reference_dir()
    raw_players = pl.read_parquet(raw_dir / "players.parquet")
    raw_teams = pl.read_parquet(raw_dir / "teams.parquet")
    identities = player_identities(raw_players)

    players = (
        raw_players.select(PLAYER_COLUMNS)
        .join(identities.select("gsis_id", "player_id"), on="gsis_id")
        .with_columns(pl.col("birth_date").str.to_date(strict=False))
        .select("player_id", *PLAYER_COLUMNS)
    )
    external_ids = (
        raw_players.select(EXTERNAL_ID_COLUMNS)
        .join(identities.select("gsis_id", "player_id"), on="gsis_id")
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
        .with_columns(pl.col("provider").str.replace(r"_id$", ""))
        .unique(subset=["provider", "external_id"])
        .sort("player_id", "provider")
    )
    teams = raw_teams.select(TEAM_COLUMNS)
    player_status = (
        raw_players.select(PLAYER_STATUS_COLUMNS)
        .join(identities.select("gsis_id", "player_id"), on="gsis_id")
        .select("player_id", *PLAYER_STATUS_COLUMNS)
    )

    validate_unique(players, ["player_id"], "players")
    write_outputs(
        PROCESSED_DIR / "reference",
        {
            "players": players,
            "player_external_ids": external_ids,
            "teams": teams,
        },
        workflow="reference",
    )
    # This is the latest status embedded in the reference player snapshot.
    # The current workflow may replace it with fresher weekly-roster status.
    write_outputs(
        PROCESSED_DIR / "current",
        {"player_status": player_status},
        workflow="reference_player_status",
    )


if __name__ == "__main__":
    process_reference()
