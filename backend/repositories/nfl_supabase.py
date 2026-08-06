"""NFL repository backed by Supabase/PostgreSQL."""

from database.client import get_supabase_client


BASE_STAT_FIELDS = [
    "player_id",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
]
GAME_FIELDS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "overtime",
    "spread_line",
    "total_line",
]


def find_players(name: str) -> list[dict]:
    client = get_supabase_client()
    players = (
        client.table("players")
        .select("player_id,display_name,position")
        .ilike("display_name", f"%{name}%")
        .limit(10)
        .execute()
        .data
    )
    if not players:
        return []

    player_ids = [player["player_id"] for player in players]
    statuses = (
        client.table("player_status")
        .select("player_id,latest_team,status")
        .in_("player_id", player_ids)
        .execute()
        .data
    )
    status_by_player = {status["player_id"]: status for status in statuses}

    return [
        {
            **player,
            "latest_team": status_by_player.get(player["player_id"], {}).get(
                "latest_team"
            ),
            "status": status_by_player.get(player["player_id"], {}).get("status"),
        }
        for player in players
    ]


def get_player_weekly_stats(
    player_id: str,
    season: int,
    week: int | None,
    fields: list[str],
) -> list[dict]:
    client = get_supabase_client()
    columns = ",".join([*BASE_STAT_FIELDS, *fields])
    query = (
        client.table("player_weekly_stats")
        .select(columns)
        .eq("player_id", player_id)
        .eq("season", season)
    )
    if week is not None:
        query = query.eq("week", week)

    return query.order("week").execute().data


def get_player_season_totals(
    player_id: str,
    season: int,
    fields: list[str],
) -> dict:
    client = get_supabase_client()
    columns = ",".join(fields) if fields else "player_id"
    rows = (
        client.table("player_weekly_stats")
        .select(columns)
        .eq("player_id", player_id)
        .eq("season", season)
        .execute()
        .data
    )
    totals = {field: sum(row.get(field) or 0 for row in rows) for field in fields}
    return {
        "player_id": player_id,
        "season": season,
        "games": len(rows),
        **totals,
    }


def get_team_games(team: str, season: int, week: int | None) -> list[dict]:
    client = get_supabase_client()
    query = (
        client.table("games")
        .select(",".join(GAME_FIELDS))
        .eq("season", season)
        .or_(f"home_team.eq.{team},away_team.eq.{team}")
    )
    if week is not None:
        query = query.eq("week", week)

    return query.order("gameday").execute().data
