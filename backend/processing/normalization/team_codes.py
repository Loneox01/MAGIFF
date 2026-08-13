"""Normalize source aliases to the canonical team codes used in processing."""

import polars as pl


# AZ appears in nflverse player snapshots while schedules and statistics use
# ARI. Historical relocation codes (OAK, SD, STL) remain distinct on purpose.
TEAM_CODE_ALIASES = {
    "AZ": "ARI",
}

TEAM_CODE_COLUMNS = frozenset(
    {
        "team_abbr",
        "team",
        "opponent",
        "opponent_team",
        "away_team",
        "home_team",
        "latest_team",
        "draft_team",
        "last_team",
    }
)


def normalize_team_codes(frame: pl.DataFrame) -> pl.DataFrame:
    """Replace known aliases in every scalar team-code column present."""
    columns = TEAM_CODE_COLUMNS.intersection(frame.columns)
    if not columns:
        return frame
    return frame.with_columns(
        pl.col(column).replace(TEAM_CODE_ALIASES).alias(column)
        for column in sorted(columns)
    )
