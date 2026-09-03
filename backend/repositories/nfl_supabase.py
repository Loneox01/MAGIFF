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
    "gametime",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "overtime",
    "spread_line",
    "total_line",
]
SEASON_BASE_FIELDS = [
    "player_id",
    "season",
    "season_type",
    "games",
    "teams",
    "last_team",
    "position",
    "position_group",
]
TEAM_STAT_BASE_FIELDS = [
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
]
DEPTH_CHART_FIELDS = [
    "player_id",
    "season",
    "week",
    "season_type",
    "snapshot_at",
    "team",
    "player_name",
    "formation",
    "position_group",
    "position_name",
    "position",
    "position_slot",
    "depth_rank",
    "jersey_number",
]
SNAP_COUNT_FIELDS = [
    "player_id",
    "game_id",
    "season",
    "game_type",
    "week",
    "position",
    "team",
    "opponent",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
]
ROSTER_FIELDS = [
    "player_id",
    "season",
    "week",
    "game_type",
    "team",
    "position",
    "depth_chart_position",
    "jersey_number",
    "status",
    "status_description_abbr",
    "years_exp",
]
ECR_FIELDS = [
    "player_id",
    "season",
    "scrape_date",
    "scoring_format",
    "league_format",
    "snapshot_type",
    "overall_rank",
    "best_rank",
    "worst_rank",
    "rank_sd",
    "rank_delta",
    "position",
    "team",
    "source",
    "ranking_page",
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


def get_week_games(season: int, week: int) -> list[dict]:
    """Return one NFL week's schedule with exact stored game times."""
    return (
        get_supabase_client()
        .table("games")
        .select(",".join(GAME_FIELDS))
        .eq("season", season)
        .eq("week", week)
        .order("gameday")
        .order("gametime")
        .execute()
        .data
    )


def get_player_season_stats(
    player_id: str,
    season: int,
    season_type: str,
    fields: list[str],
) -> dict:
    columns = ",".join(
        [*SEASON_BASE_FIELDS, *[field for field in fields if field not in SEASON_BASE_FIELDS]]
    )
    rows = (
        get_supabase_client()
        .table("player_season_stats")
        .select(columns)
        .eq("player_id", player_id)
        .eq("season", season)
        .eq("season_type", season_type)
        .limit(1)
        .execute()
        .data
    )
    return rows[0] if rows else {}


def get_team_weekly_stats(
    team: str,
    season: int,
    week: int | None,
    fields: list[str],
) -> list[dict]:
    query = (
        get_supabase_client()
        .table("team_weekly_stats")
        .select(",".join([*TEAM_STAT_BASE_FIELDS, *fields]))
        .eq("team", team)
        .eq("season", season)
    )
    if week is not None:
        query = query.eq("week", week)
    return query.order("week").execute().data


def get_team_depth_chart(
    team: str,
    season: int,
    week: int | None,
    position: str | None,
) -> list[dict]:
    table = "current_depth_chart_entries" if week is None else "depth_chart_entries"
    query = (
        get_supabase_client()
        .table(table)
        .select(",".join(DEPTH_CHART_FIELDS))
        .eq("team", team)
        .eq("season", season)
    )
    if week is not None:
        query = query.eq("week", week)
    if position is not None:
        query = query.eq("position", position)
    return (
        query.order("formation")
        .order("position_slot")
        .order("depth_rank")
        .limit(150)
        .execute()
        .data
    )


def get_player_snap_counts(
    player_id: str,
    season: int,
    week: int | None,
) -> list[dict]:
    query = (
        get_supabase_client()
        .table("player_snap_counts")
        .select(",".join(SNAP_COUNT_FIELDS))
        .eq("player_id", player_id)
        .eq("season", season)
    )
    if week is not None:
        query = query.eq("week", week)
    return query.order("week").execute().data


def get_team_roster(
    team: str,
    season: int,
    week: int | None,
    position: str | None,
    status: str | None,
) -> list[dict]:
    client = get_supabase_client()
    selected_week = week
    if selected_week is None:
        newest = (
            client.table("player_weekly_rosters")
            .select("week")
            .eq("team", team)
            .eq("season", season)
            .order("week", desc=True)
            .limit(1)
            .execute()
            .data
        )
        if not newest:
            return []
        selected_week = newest[0]["week"]

    query = (
        client.table("player_weekly_rosters")
        .select(",".join(ROSTER_FIELDS))
        .eq("team", team)
        .eq("season", season)
        .eq("week", selected_week)
    )
    if position is not None:
        query = query.eq("position", position)
    if status is not None:
        query = query.eq("status", status)
    rows = query.order("position").limit(100).execute().data
    names = get_player_names(
        [str(row["player_id"]) for row in rows if row.get("player_id")]
    )
    return [
        {
            **row,
            "player_name": names.get(str(row.get("player_id"))),
        }
        for row in rows
    ]


def get_current_team_roster(
    team: str,
    season: int,
    position: str | None,
    status: str | None,
) -> list[dict]:
    """Return the latest player-status snapshot in weekly-roster shape."""
    query = (
        get_supabase_client()
        .table("player_status")
        .select(
            "player_id,latest_team,jersey_number,position,status,ngs_status,"
            "years_of_experience,last_season"
        )
        .eq("latest_team", team)
        # player_status includes historical players whose latest career team
        # was this team. last_season is therefore the required current-snapshot
        # eligibility boundary, not merely useful metadata.
        .eq("last_season", season)
    )
    if position is not None:
        query = query.eq("position", position)
    if status is not None:
        query = query.eq("status", status)
    rows = query.order("position").limit(100).execute().data
    names = get_player_names(
        [str(row["player_id"]) for row in rows if row.get("player_id")]
    )
    return [
        {
            "player_id": row["player_id"],
            "player_name": names.get(str(row["player_id"])),
            "season": season,
            "week": None,
            "game_type": None,
            "team": row["latest_team"],
            "position": row["position"],
            "depth_chart_position": None,
            "jersey_number": row.get("jersey_number"),
            "status": row["status"],
            "status_description_abbr": row.get("ngs_status"),
            "years_exp": row.get("years_of_experience"),
        }
        for row in rows
    ]


def get_player_season_candidates(
    season: int,
    season_type: str,
    position: str | None,
    fields: list[str],
    minimum_field: str | None,
    minimum_value: float | None,
) -> list[dict]:
    """Fetch bounded pages of season rows for application-side analytics."""
    normalized_minimum = (
        int(minimum_value)
        if isinstance(minimum_value, float) and minimum_value.is_integer()
        else minimum_value
    )
    selected = list(
        dict.fromkeys(
            [
                "player_id",
                "season",
                "season_type",
                "games",
                "last_team",
                "position",
                "position_group",
                *fields,
                *([minimum_field] if minimum_field else []),
            ]
        )
    )
    client = get_supabase_client()
    rows: list[dict] = []
    page_size = 1000
    start = 0

    while True:
        query = (
            client.table("player_season_stats")
            .select(",".join(selected))
            .eq("season", season)
            .eq("season_type", season_type)
        )
        if position is not None:
            query = query.eq("position", position)
        if minimum_field is not None and normalized_minimum is not None:
            query = query.gte(minimum_field, normalized_minimum)

        page = (
            query.order("player_id")
            .range(start, start + page_size - 1)
            .execute()
            .data
        )
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    return rows


def get_player_weekly_analysis_inputs(
    season: int,
    season_type: str,
    position: str | None,
    player_ids: list[str] | None,
    fields: list[str],
    include_rosters: bool,
) -> tuple[list[dict], list[dict]]:
    """Fetch paginated weekly stats and optional roster participation rows."""
    client = get_supabase_client()

    def fetch_pages(query, order_fields: list[str]) -> list[dict]:
        rows: list[dict] = []
        page_size = 1000
        start = 0
        while True:
            ordered = query
            for field in order_fields:
                ordered = ordered.order(field)
            page = (
                ordered.range(start, start + page_size - 1).execute().data
            )
            rows.extend(page)
            if len(page) < page_size:
                return rows
            start += page_size

    weekly_columns = list(
        dict.fromkeys(
            [
                "player_id",
                "week",
                "team",
                "position",
                *fields,
            ]
        )
    )
    weekly_query = (
        client.table("player_weekly_stats")
        .select(",".join(weekly_columns))
        .eq("season", season)
        .eq("season_type", season_type)
    )
    if position is not None:
        weekly_query = weekly_query.eq("position", position)
    if player_ids is not None:
        weekly_query = weekly_query.in_("player_id", player_ids)
    weekly_rows = fetch_pages(weekly_query, ["player_id", "week"])

    if not include_rosters:
        return weekly_rows, []

    roster_query = (
        client.table("player_weekly_rosters")
        .select("player_id,week,team,position,status")
        .eq("season", season)
    )
    if season_type == "REG":
        roster_query = roster_query.eq("game_type", "REG")
    else:
        roster_query = roster_query.in_("game_type", ["WC", "DIV", "CON", "SB"])
    if position is not None:
        roster_query = roster_query.eq("position", position)
    if player_ids is not None:
        roster_query = roster_query.in_("player_id", player_ids)
    roster_rows = fetch_pages(roster_query, ["player_id", "week"])
    return weekly_rows, roster_rows


def get_player_names(player_ids: list[str]) -> dict[str, str]:
    """Resolve the small final ranking result to display names."""
    if not player_ids:
        return {}
    rows = (
        get_supabase_client()
        .table("players")
        .select("player_id,display_name")
        .in_("player_id", list(dict.fromkeys(player_ids)))
        .execute()
        .data
    )
    return {row["player_id"]: row["display_name"] for row in rows}


def get_player_external_ids(
    player_ids: list[str],
    provider: str,
) -> dict[str, str]:
    """Return one provider identifier for each requested internal player ID."""
    if not player_ids:
        return {}
    rows = (
        get_supabase_client()
        .table("player_external_ids")
        .select("player_id,external_id")
        .eq("provider", provider)
        .in_("player_id", list(dict.fromkeys(player_ids)))
        .execute()
        .data
    )
    return {
        str(row["player_id"]): str(row["external_id"])
        for row in rows
        if row.get("player_id") and row.get("external_id")
    }


def _get_ecr_snapshot_date(
    client,
    season: int,
    scoring_format: str,
    league_format: str,
    snapshot_type: str,
    as_of_date: str | None,
) -> str | None:
    """Return the newest stored snapshot date matching one ECR format."""
    date_query = (
        client.table("player_ecr")
        .select("scrape_date")
        .eq("season", season)
        .eq("scoring_format", scoring_format)
        .eq("league_format", league_format)
        .eq("snapshot_type", snapshot_type)
    )
    if as_of_date is not None:
        date_query = date_query.lte("scrape_date", as_of_date)
    dates = date_query.order("scrape_date", desc=True).limit(1).execute().data
    return dates[0]["scrape_date"] if dates else None


def get_ecr_rows(
    season: int,
    scoring_format: str,
    league_format: str,
    snapshot_type: str,
    as_of_date: str | None,
) -> tuple[str | None, list[dict]]:
    """Fetch the latest qualifying ECR snapshot and all its player rows."""
    client = get_supabase_client()
    selected_date = _get_ecr_snapshot_date(
        client,
        season,
        scoring_format,
        league_format,
        snapshot_type,
        as_of_date,
    )
    if selected_date is None:
        return None, []

    rows = (
        client.table("player_ecr")
        .select(",".join(ECR_FIELDS))
        .eq("season", season)
        .eq("scoring_format", scoring_format)
        .eq("league_format", league_format)
        .eq("snapshot_type", snapshot_type)
        .eq("scrape_date", selected_date)
        .order("overall_rank")
        .limit(1000)
        .execute()
        .data
    )
    return selected_date, rows


def get_player_ecr_row(
    player_id: str,
    season: int,
    scoring_format: str,
    league_format: str,
    snapshot_type: str,
    as_of_date: str | None,
) -> tuple[str | None, dict | None]:
    """Fetch one player's ECR from the latest qualifying stored snapshot."""
    client = get_supabase_client()
    selected_date = _get_ecr_snapshot_date(
        client,
        season,
        scoring_format,
        league_format,
        snapshot_type,
        as_of_date,
    )
    if selected_date is None:
        return None, None

    filters = {
        "season": season,
        "scoring_format": scoring_format,
        "league_format": league_format,
        "snapshot_type": snapshot_type,
        "scrape_date": selected_date,
    }
    row_query = client.table("player_ecr").select(",".join(ECR_FIELDS))
    for field, value in filters.items():
        row_query = row_query.eq(field, value)
    rows = row_query.eq("player_id", player_id).limit(1).execute().data
    if not rows:
        return selected_date, None

    row = rows[0]
    row["rank_range"] = (
        row["worst_rank"] - row["best_rank"]
        if row.get("worst_rank") is not None and row.get("best_rank") is not None
        else None
    )
    position = row.get("position")
    if position and row.get("overall_rank") is not None:
        rank_query = client.table("player_ecr").select(
            "player_id", count="exact", head=True
        )
        for field, value in filters.items():
            rank_query = rank_query.eq(field, value)
        rank_response = (
            rank_query.eq("position", position)
            .lt("overall_rank", row["overall_rank"])
            .execute()
        )
        row["position_rank"] = (rank_response.count or 0) + 1
    else:
        row["position_rank"] = None

    return selected_date, row


def get_season_fantasy_results(season: int) -> list[dict]:
    """Fetch regular-season scoring inputs used for ECR comparisons."""
    client = get_supabase_client()
    rows: list[dict] = []
    page_size = 1000
    start = 0
    while True:
        page = (
            client.table("player_season_stats")
            .select(
                "player_id,season,games,last_team,position,receptions,"
                "fantasy_points,fantasy_points_ppr"
            )
            .eq("season", season)
            .eq("season_type", "REG")
            .order("player_id")
            .range(start, start + page_size - 1)
            .execute()
            .data
        )
        rows.extend(page)
        if len(page) < page_size:
            return rows
        start += page_size


def get_team_formula_inputs(
    season: int,
    season_type: str,
) -> tuple[list[dict], list[dict]]:
    """Fetch weekly team statistics and matching game scores for aggregation."""
    client = get_supabase_client()
    weekly_rows: list[dict] = []
    page_size = 1000
    start = 0
    while True:
        page = (
            client.table("team_weekly_stats")
            .select("*")
            .eq("season", season)
            .eq("season_type", season_type)
            .order("game_id")
            .order("team")
            .range(start, start + page_size - 1)
            .execute()
            .data
        )
        weekly_rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    games = (
        client.table("games")
        .select("game_id,home_team,home_score,away_team,away_score")
        .eq("season", season)
        .execute()
        .data
    )
    return weekly_rows, games


def get_team_names(team_abbrs: list[str]) -> dict[str, str]:
    if not team_abbrs:
        return {}
    rows = (
        get_supabase_client()
        .table("teams")
        .select("team_abbr,team_name")
        .in_("team_abbr", list(dict.fromkeys(team_abbrs)))
        .execute()
        .data
    )
    return {row["team_abbr"]: row["team_name"] for row in rows}
