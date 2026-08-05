"""OpenAI function-tool schemas for local NFL data."""


STAT_FIELDS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points",
    "fantasy_points_ppr",
]

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

GET_PLAYER_SEASON_TOTALS_TOOL = {
    "type": "function",
    "name": "get_player_season_totals",
    "description": "Get summed season statistics for a player identified by find_players.",
    "parameters": {
        "type": "object",
        "properties": {
            "player_id": {"type": "string", "description": "Internal player UUID."},
            "season": {"type": "integer", "description": "Four-digit NFL season."},
            "fields": NULLABLE_FIELDS,
        },
        "required": ["player_id", "season", "fields"],
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

NFL_TOOLS = [
    FIND_PLAYERS_TOOL,
    GET_PLAYER_WEEKLY_STATS_TOOL,
    GET_PLAYER_SEASON_TOTALS_TOOL,
    GET_TEAM_GAMES_TOOL,
]
