"""Select compact tool subsets from a universal request route."""

from tools.nfl import TOOL_HANDLERS as NFL_TOOL_HANDLERS
from tools.reports import REPORT_TOOL_HANDLERS, SEARCH_REPORTS_TOOL
from tools.schemas import NFL_TOOLS

from .router import Capability, RequestRoute, StructuredDomain


_NFL_SCHEMAS = {schema["name"]: schema for schema in NFL_TOOLS}

STRUCTURED_DOMAIN_TOOLS = {
    StructuredDomain.PLAYER_LOOKUP: (
        "find_players",
    ),
    StructuredDomain.PLAYER_STATS: (
        "find_players",
        "get_player_weekly_stats",
        "get_player_season_stats",
        "get_player_snap_counts",
        "rank_players_by_formula",
        "rank_players_by_weekly_threshold",
    ),
    StructuredDomain.TEAM_STATS: (
        "get_team_weekly_stats",
        "rank_teams_by_formula",
    ),
    StructuredDomain.SCHEDULES: (
        "get_team_games",
    ),
    StructuredDomain.ROSTERS_DEPTH_CHARTS: (
        "find_players",
        "get_team_depth_chart",
        "get_team_roster",
        "get_player_snap_counts",
    ),
    StructuredDomain.ECR: (
        "find_players",
        "get_player_ecr",
        "rank_players_by_ecr",
        "compare_ecr_to_results",
    ),
}


def tool_names_for_route(route: RequestRoute) -> list[str]:
    names: list[str] = []
    if Capability.STRUCTURED_DATA in route.capabilities:
        for domain in route.structured_domains:
            names.extend(STRUCTURED_DOMAIN_TOOLS[domain])
    if Capability.REPORTS in route.capabilities:
        names.append("search_reports")
    return list(dict.fromkeys(names))


def tool_schemas_for_route(route: RequestRoute) -> list[dict]:
    schemas = {**_NFL_SCHEMAS, "search_reports": SEARCH_REPORTS_TOOL}
    return [schemas[name] for name in tool_names_for_route(route)]


TOOL_HANDLERS = {**NFL_TOOL_HANDLERS, **REPORT_TOOL_HANDLERS}
