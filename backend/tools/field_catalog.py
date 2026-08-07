"""Canonical model-facing fields derived from processed NFL schemas."""

from processing.columns import (
    PLAYER_SEASON_ADDITIVE_COLUMNS,
    PLAYER_SEASON_DERIVED_COLUMNS,
    TEAM_WEEKLY_STAT_COLUMNS,
    WEEKLY_STAT_COLUMNS,
)


PLAYER_WEEKLY_CONTEXT_FIELDS = {
    "player_id", "gsis_id", "season", "week", "season_type", "game_id",
    "team", "opponent_team", "position", "position_group",
}
TEAM_WEEKLY_CONTEXT_FIELDS = {
    "season", "week", "season_type", "game_id", "team", "opponent_team",
}

# Every statistical value retained by the corresponding processed weekly file.
PLAYER_WEEKLY_STAT_FIELDS = frozenset(WEEKLY_STAT_COLUMNS) - PLAYER_WEEKLY_CONTEXT_FIELDS
TEAM_WEEKLY_STAT_FIELDS = frozenset(TEAM_WEEKLY_STAT_COLUMNS) - TEAM_WEEKLY_CONTEXT_FIELDS

# Numeric columns written by player_season_stats.py. Identifiers and dimensions
# are intentionally excluded because formulas must be numeric.
PLAYER_SEASON_STAT_FIELDS = frozenset(
    ["games", *PLAYER_SEASON_ADDITIVE_COLUMNS, *PLAYER_SEASON_DERIVED_COLUMNS]
)

PLAYER_FORMULA_FIELDS = PLAYER_SEASON_STAT_FIELDS


def field_names(fields: set[str] | frozenset[str]) -> str:
    """Return a stable field list suitable for a tool-parameter description."""
    return ", ".join(sorted(fields))
