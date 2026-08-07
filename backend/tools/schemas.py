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
from tools.weekly_analytics import COMPARISONS, PARTICIPATION_BASES, WEEKLY_RANK_FIELDS


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

RANK_PLAYERS_BY_WEEKLY_THRESHOLD_TOOL = {
    "type": "function",
    "name": "rank_players_by_weekly_threshold",
    "description": (
        "Rank players by how often a safe weekly-stat formula meets a threshold. "
        "Use for boom/bust, milestone, and consistency questions. Choose the "
        "participation denominator explicitly: stat_row evaluates games with a "
        "stats row; active_roster includes active zero-production games; "
        "team_games includes every rostered team-game week, including injury or "
        "reserve absences. Pass player_ids from a preceding player-ranking call "
        "when the question limits analysis to a specific candidate pool. Formula "
        "supports numbers, parentheses, +, -, *, and /."
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
                    "Weekly arithmetic metric, such as fantasy_points_ppr, "
                    "fantasy_points + 0.5 * receptions, receiving_yards, or "
                    "attempts. Allowed fields: "
                    f"{field_names(PLAYER_WEEKLY_STAT_FIELDS)}."
                ),
            },
            "comparison": {
                "type": "string",
                "enum": sorted(COMPARISONS),
                "description": (
                    "Threshold operator: gte >=, gt >, lte <=, lt <, or eq ==."
                ),
            },
            "threshold": {
                "type": "number",
                "description": "Numeric threshold applied to each weekly result.",
            },
            "participation_basis": {
                "type": "string",
                "enum": sorted(PARTICIPATION_BASES),
                "description": (
                    "stat_row for recorded-stat games; active_roster for weeks "
                    "listed ACT; team_games for every weekly roster record, "
                    "including inactive/reserve weeks."
                ),
            },
            "season_type": NULLABLE_SEASON_TYPE,
            "position": NULLABLE_POSITION,
            "player_ids": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    {"type": "null"},
                ],
                "description": (
                    "Optional candidate pool of 1-20 internal player IDs, usually "
                    "copied from rank_players_by_formula; null searches everyone."
                ),
            },
            "minimum_games": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "description": (
                    "Minimum denominator games under the selected participation basis."
                ),
            },
            "rank_by": {
                "type": "string",
                "enum": sorted(WEEKLY_RANK_FIELDS),
                "description": (
                    "qualifying_games ranks total matching weeks; "
                    "qualifying_rate ranks the share of denominator games."
                ),
            },
            "sort_direction": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "desc finds most/highest; asc finds fewest/lowest.",
            },
            "include_week_details": {
                "type": "boolean",
                "description": (
                    "False returns compact ranking rows. Use true only when the "
                    "question explicitly needs qualifying weeks, averages, "
                    "medians, or participation diagnostics."
                ),
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": [
            "season",
            "formula",
            "comparison",
            "threshold",
            "participation_basis",
            "season_type",
            "position",
            "player_ids",
            "minimum_games",
            "rank_by",
            "sort_direction",
            "include_week_details",
            "limit",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}

ECR_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DL", "LB", "DB"]
ECR_COMPARISON_POSITIONS = ["QB", "RB", "WR", "TE", "K"]
NULLABLE_ECR_POSITIONS = {
    "anyOf": [
        {
            "type": "array",
            "items": {"type": "string", "enum": ECR_POSITIONS},
            "minItems": 1,
            "maxItems": len(ECR_POSITIONS),
        },
        {"type": "null"},
    ],
    "description": (
        "One or more positions to include; null includes every available player "
        "position. Team defenses are not stored as players."
    ),
}
NULLABLE_ECR_RANK = {
    "anyOf": [{"type": "number", "minimum": 1}, {"type": "null"}],
    "description": "Inclusive overall ECR boundary, or null for no boundary.",
}


def nullable_ecr_positions(values: list[str], description: str) -> dict:
    return {
        "anyOf": [
            {
                "type": "array",
                "items": {"type": "string", "enum": values},
                "minItems": 1,
                "maxItems": len(values),
            },
            {"type": "null"},
        ],
        "description": description,
    }

RANK_PLAYERS_BY_ECR_TOOL = {
    "type": "function",
    "name": "rank_players_by_ecr",
    "description": (
        "Rank players from one expert-consensus-ranking snapshot. Use for draft "
        "rankings, positional rankings, expert disagreement, and ranking movement. "
        "The tool automatically uses the latest snapshot on or before as_of_date."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "season": {"type": "integer", "minimum": 2019},
            "positions": NULLABLE_ECR_POSITIONS,
            "scoring_format": {
                "type": "string",
                "enum": ["ppr", "source_default"],
                "description": (
                    "Use ppr for redraft formats. Use source_default for dynasty, "
                    "rookie, best-ball, and IDP pages whose upstream scoring is "
                    "not explicitly stated."
                ),
            },
            "league_format": {
                "type": "string",
                "enum": [
                    "redraft_1qb",
                    "redraft_superflex",
                    "dynasty_1qb",
                    "dynasty_superflex",
                    "dynasty_rookie",
                    "best_ball",
                    "redraft_idp",
                    "dynasty_idp",
                ],
                "description": "League/ranking format represented by this ECR pool.",
            },
            "snapshot_type": {
                "type": "string",
                "enum": ["current", "final_preseason", "season_opening"],
                "description": (
                    "current for the active ranking snapshot; final_preseason for "
                    "the selected historical draft-day snapshot."
                ),
            },
            "as_of_date": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ],
                "description": "YYYY-MM-DD cutoff, or null for the latest snapshot.",
            },
            "sort_by": {
                "type": "string",
                "enum": [
                    "overall_rank",
                    "position_rank",
                    "rank_sd",
                    "rank_range",
                    "rank_delta",
                ],
                "description": (
                    "rank_sd and rank_range measure expert disagreement; rank_delta "
                    "measures source-provided movement."
                ),
            },
            "sort_direction": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Use asc for best/smallest ECR; desc for largest values.",
            },
            "minimum_overall_rank": NULLABLE_ECR_RANK,
            "maximum_overall_rank": NULLABLE_ECR_RANK,
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [
            "season",
            "positions",
            "scoring_format",
            "league_format",
            "snapshot_type",
            "as_of_date",
            "sort_by",
            "sort_direction",
            "minimum_overall_rank",
            "maximum_overall_rank",
            "limit",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}

COMPARE_ECR_TO_RESULTS_TOOL = {
    "type": "function",
    "name": "compare_ecr_to_results",
    "description": (
        "Compare a stored ECR snapshot with actual regular-season fantasy results "
        "in one call. Use for draft steals, busts, ECR accuracy, and positional "
        "finish analysis. Positive rank_difference means the player outperformed ECR."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "season": {"type": "integer", "minimum": 2019},
            "positions": nullable_ecr_positions(
                ECR_COMPARISON_POSITIONS,
                "Offensive positions to compare; null includes all supported positions.",
            ),
            "scoring_format": {
                "type": "string",
                "enum": ["ppr"],
            },
            "league_format": {
                "type": "string",
                "enum": ["redraft_1qb", "redraft_superflex"],
            },
            "snapshot_type": {
                "type": "string",
                "enum": ["final_preseason"],
                "description": "Completed-season comparisons use final preseason ECR.",
            },
            "comparison_basis": {
                "type": "string",
                "enum": ["overall", "position"],
                "description": (
                    "Compare overall ECR/finish or ranks within each player position."
                ),
            },
            "sort_by": {
                "type": "string",
                "enum": [
                    "rank_difference",
                    "ecr_rank",
                    "actual_finish",
                    "fantasy_points",
                    "fantasy_points_per_game",
                ],
            },
            "sort_direction": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": (
                    "For rank_difference, desc finds outperformers and asc finds busts."
                ),
            },
            "minimum_games": {
                "type": "integer",
                "minimum": 0,
                "maximum": 25,
                "description": (
                    "Minimum games played; use 0 to retain complete-season busts."
                ),
            },
            "minimum_overall_rank": NULLABLE_ECR_RANK,
            "maximum_overall_rank": NULLABLE_ECR_RANK,
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [
            "season",
            "positions",
            "scoring_format",
            "league_format",
            "snapshot_type",
            "comparison_basis",
            "sort_by",
            "sort_direction",
            "minimum_games",
            "minimum_overall_rank",
            "maximum_overall_rank",
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
    RANK_PLAYERS_BY_WEEKLY_THRESHOLD_TOOL,
    RANK_PLAYERS_BY_ECR_TOOL,
    COMPARE_ECR_TO_RESULTS_TOOL,
]
