"""Convert the nflverse fantasy ID crosswalk to the project's long format."""

import polars as pl

from .common import player_ids_path


# Keep IDs for likely integrations. The complete source crosswalk remains in raw.
FANTASY_ID_COLUMNS = {
    "fantasypros_id": "fantasypros",
    "sleeper_id": "sleeper",
    "espn_id": "espn",
    "yahoo_id": "yahoo",
    "cbs_id": "cbs",
    "mfl_id": "mfl",
    "pfr_id": "pfr",
}


def build_fantasy_external_ids(
    identities: pl.DataFrame,
    reference_season: int,
) -> tuple[pl.DataFrame, int]:
    """Return provider IDs matched to an existing internal player UUID."""
    raw = pl.read_parquet(player_ids_path(reference_season))
    id_columns = [column for column in FANTASY_ID_COLUMNS if column in raw.columns]
    typed = raw.select("gsis_id", *id_columns).with_columns(
        *(pl.col(column).cast(pl.String, strict=False) for column in id_columns)
    )
    joined = typed.join(
        identities.select("gsis_id", "player_id").unique("gsis_id"),
        on="gsis_id",
        how="left",
    )
    unmatched = joined.filter(
        pl.col("gsis_id").is_not_null() & pl.col("player_id").is_null()
    ).height
    external_ids = (
        joined.filter(pl.col("player_id").is_not_null())
        .unpivot(
            index="player_id",
            on=id_columns,
            variable_name="provider_column",
            value_name="external_id",
        )
        .filter(
            pl.col("external_id").is_not_null()
            & (pl.col("external_id").str.strip_chars() != "")
        )
        .with_columns(
            pl.col("provider_column")
            .replace(FANTASY_ID_COLUMNS)
            .alias("provider")
        )
        .select("player_id", "provider", "external_id")
        .unique(subset=["provider", "external_id"], maintain_order=True)
    )
    return external_ids, unmatched

