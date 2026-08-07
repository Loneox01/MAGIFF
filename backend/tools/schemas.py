"""OpenAI function-tool schemas for local NFL data."""

from tools.field_catalog import (
    PLAYER_FORMULA_FIELDS,
    PLAYER_SEASON_STAT_FIELDS,
    PLAYER_WEEKLY_STAT_FIELDS,
    TEAM_WEEKLY_STAT_FIELDS,
    field_names,
)
from tools.team_analytics import (
    TEAM_DEFENSE_FORMULA_FIELDS,
    TEAM_OFFENSE_FORMULA_FIELDS,
)


STAT_FIELDS = sorted(PLAYER_WEEKLY_STAT_FIELDS)
SEASON_STAT_FIELDS = sorted(PLAYER_SEASON_STAT_FIELDS)
TEAM_STAT_FIELDS = sorted(TEAM_WEEKLY_STAT_FIELDS)
PLAYER_FORMULA_FIELD_LIST = sorted(PLAYER_FORMULA_FIELDS)

NULLABLE_WEEK = {
    "anyOf": [{"type": "integer", "minimum": 1, "maximum": 22}, {"type": "null"}],
    "description": "NFL week, or null for every week.",
}
NULLABLE_FIELDS = {
    "anyOf": [
        {
            "type": "array",
            "items": {"type": "string", "enum": STAT_FIELDS},
        },
        {"type": "null"},
    ],
    "description": "Stats to return, or null for the default fantasy-relevant set.",
}
NULLABLE_SEASON_TYPE = {
    "anyOf": [
        {"type": "string", "enum": ["REG", "POST"]},
        {"type": "null"},
    ],
    "description": "REG or POST; null defaults to REG.",
}
NULLABLE_POSITION = {
    "anyOf": [{"type": "string"}, {"type": "null"}],
    "description": "Position abbreviation such as QB or WR, or null.",
}


def nullable_field_list(values: list[str], description: str) -> dict:
    return {
        "anyOf": [
            {"type": "array", "items": {"type": "string", "enum": values}},
            {"type": "null"},
        ],
        "description": description,
    }

FIND_PLAYERS_TOOL = {
    "type": "function",
    "name": "find_players",
    "description": "Find an NFL player and their internal player ID by name.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full or partial player name."},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_PLAYER_WEEKLY_STATS_TOOL = {
    "type": "function",
    "name": "get_player_weekly_stats",
    "description": "Get recent weekly NFL statistics for a player identified by find_players.",
    "parameters": {
        "type": "object",
        "properties": {
            "player_id": {"type": "string", "description": "Internal player UUID."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "week": NULLABLE_WEEK,
            "fields": NULLABLE_FIELDS,
        },
        "required": ["player_id", "season", "week", "fields"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_TEAM_GAMES_TOOL = {
    "type": "function",
    "name": "get_team_games",
    "description": "Get schedule and result data for an NFL team.",
    "parameters": {
        "type": "object",
        "properties": {
            "team": {"type": "string", "description": "NFL team abbreviation, such as BUF."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "week": NULLABLE_WEEK,
        },
        "required": ["team", "season", "week"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_PLAYER_SEASON_STATS_TOOL = {
    "type": "function",
    "name": "get_player_season_stats",
    "description": "Get stored NFL season totals and efficiency metrics for a player.",
    "parameters": {
        "type": "object",
        "properties": {
            "player_id": {"type": "string", "description": "Internal player UUID."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "season_type": NULLABLE_SEASON_TYPE,
            "fields": nullable_field_list(
                SEASON_STAT_FIELDS,
                "Season statistics to return, or null for common fantasy totals.",
            ),
        },
        "required": ["player_id", "season", "season_type", "fields"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_TEAM_WEEKLY_STATS_TOOL = {
    "type": "function",
    "name": "get_team_weekly_stats",
    "description": "Get an NFL team's offensive totals for one week or every week in a season.",
    "parameters": {
        "type": "object",
        "properties": {
            "team": {"type": "string", "description": "Team abbreviation such as BUF."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "week": NULLABLE_WEEK,
            "fields": nullable_field_list(
                TEAM_STAT_FIELDS,
                "Team statistics to return, or null for common offensive totals.",
            ),
        },
        "required": ["team", "season", "week", "fields"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_TEAM_DEPTH_CHART_TOOL = {
    "type": "function",
    "name": "get_team_depth_chart",
    "description": (
        "Get a team depth chart. Use a week for historical seasons; null gets "
        "the current active-season snapshot."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "team": {"type": "string", "description": "Team abbreviation such as SF."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "week": NULLABLE_WEEK,
            "position": NULLABLE_POSITION,
        },
        "required": ["team", "season", "week", "position"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_PLAYER_SNAP_COUNTS_TOOL = {
    "type": "function",
    "name": "get_player_snap_counts",
    "description": "Get a player's weekly offense, defense, and special-teams snap usage.",
    "parameters": {
        "type": "object",
        "properties": {
            "player_id": {"type": "string", "description": "Internal player UUID."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "week": NULLABLE_WEEK,
        },
        "required": ["player_id", "season", "week"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_TEAM_ROSTER_TOOL = {
    "type": "function",
    "name": "get_team_roster",
    "description": "Get a team's weekly roster; null week uses its latest available week.",
    "parameters": {
        "type": "object",
        "properties": {
            "team": {"type": "string", "description": "Team abbreviation such as BUF."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "week": NULLABLE_WEEK,
            "position": NULLABLE_POSITION,
            "status": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Roster status code, or null for every status.",
            },
        },
        "required": ["team", "season", "week", "position", "status"],
        "additionalProperties": False,
    },
    "strict": True,
}

RANK_PLAYERS_BY_FORMULA_TOOL = {
    "type": "function",
    "name": "rank_players_by_formula",
    "description": (
        "Rank player seasons in one call using safe arithmetic over stored stats. "
        "Use for basic leaders and derived efficiency metrics, and provide a "
        "meaningful workload minimum for ratios. Formula supports field names, "
        "numbers, parentheses, +, -, *, and /."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "season": {
                "type": "integer",
                "minimum": 1999,
                "description": "Four-digit NFL season.",
            },
            "formula": {
                "type": "string",
                "description": (
                    "Arithmetic metric, such as rushing_yards / carries, "
                    "fantasy_points_ppr / targets, or passing_yards. Allowed "
                    f"fields: {field_names(PLAYER_FORMULA_FIELDS)}."
                ),
            },
            "season_type": NULLABLE_SEASON_TYPE,
            "position": NULLABLE_POSITION,
            "minimum_field": {
                "anyOf": [
                    {"type": "string", "enum": PLAYER_FORMULA_FIELD_LIST},
                    {"type": "null"},
                ],
                "description": (
                    "Workload field such as carries, targets, attempts, or games; "
                    "null only when no minimum is appropriate."
                ),
            },
            "minimum_value": {
                "anyOf": [
                    {"type": "number", "minimum": 0},
                    {"type": "null"},
                ],
                "description": "Minimum workload value, paired with minimum_field.",
            },
            "sort_direction": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "desc finds highest values; asc finds lowest.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [
            "season",
            "formula",
            "season_type",
            "position",
            "minimum_field",
            "minimum_value",
            "sort_direction",
            "limit",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}

RANK_TEAMS_BY_FORMULA_TOOL = {
    "type": "function",
    "name": "rank_teams_by_formula",
    "description": (
        "Rank team seasons in one call using safe arithmetic. For offense use "
        "fields such as points_scored, passing_yards, rushing_yards, attempts, "
        "carries, passing_epa, and rushing_epa. For defense use points_allowed "
        "or append _allowed to a team stat, such as passing_yards_allowed. "
        "Formula supports field names, numbers, parentheses, +, -, *, and /."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "season": {
                "type": "integer",
                "minimum": 1999,
                "description": "Four-digit NFL season.",
            },
            "perspective": {
                "type": "string",
                "enum": ["offense", "defense"],
                "description": "Whether to rank production or production allowed.",
            },
            "formula": {
                "type": "string",
                "description": (
                    "Arithmetic metric, such as points_scored / games, "
                    "(passing_yards + rushing_yards) / games, or "
                    "points_allowed / games. Offense fields: "
                    f"{field_names(TEAM_OFFENSE_FORMULA_FIELDS)}. Defense "
                    f"fields: {field_names(TEAM_DEFENSE_FORMULA_FIELDS)}."
                ),
            },
            "season_type": NULLABLE_SEASON_TYPE,
            "minimum_games": {
                "anyOf": [
                    {"type": "integer", "minimum": 1, "maximum": 25},
                    {"type": "null"},
                ],
                "description": "Minimum games played, or null for every team.",
            },
            "sort_direction": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "desc finds highest values; asc finds lowest.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [
            "season",
            "perspective",
            "formula",
            "season_type",
            "minimum_games",
            "sort_direction",
            "limit",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}

NFL_TOOLS = [
    FIND_PLAYERS_TOOL,
    GET_PLAYER_WEEKLY_STATS_TOOL,
    GET_TEAM_GAMES_TOOL,
    GET_PLAYER_SEASON_STATS_TOOL,
    GET_TEAM_WEEKLY_STATS_TOOL,
    GET_TEAM_DEPTH_CHART_TOOL,
    GET_PLAYER_SNAP_COUNTS_TOOL,
    GET_TEAM_ROSTER_TOOL,
    RANK_PLAYERS_BY_FORMULA_TOOL,
    RANK_TEAMS_BY_FORMULA_TOOL,
]
