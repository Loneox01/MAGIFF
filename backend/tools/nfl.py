"""Model-facing NFL tools backed by the selected data repository."""

from repositories import nfl_supabase as repository
from tools.field_catalog import (
    PLAYER_FORMULA_FIELDS,
    PLAYER_SEASON_STAT_FIELDS,
    PLAYER_WEEKLY_STAT_FIELDS,
    TEAM_WEEKLY_STAT_FIELDS,
)
from tools.formulas import ZeroDenominatorError, evaluate_formula, parse_formula
from tools.ecr_analytics import (
    ECR_COMPARISON_SORT_FIELDS,
    ECR_COMPARISON_POSITIONS,
    ECR_POSITIONS,
    ECR_SORT_FIELDS,
    add_ecr_position_ranks,
    build_ecr_comparisons,
)
from tools.team_analytics import (
    TEAM_DEFENSE_FORMULA_FIELDS,
    TEAM_OFFENSE_FORMULA_FIELDS,
    build_team_season_rows,
)
from tools.weekly_analytics import (
    COMPARISONS,
    PARTICIPATION_BASES,
    WEEKLY_RANK_FIELDS,
    build_weekly_threshold_rows,
)


ALLOWED_STAT_FIELDS = set(PLAYER_WEEKLY_STAT_FIELDS)
DEFAULT_STAT_FIELDS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points_ppr",
]
SEASON_STAT_FIELDS = set(PLAYER_SEASON_STAT_FIELDS)
DEFAULT_SEASON_FIELDS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points_ppr",
]
TEAM_STAT_FIELDS = set(TEAM_WEEKLY_STAT_FIELDS)
DEFAULT_TEAM_FIELDS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
]
def _stat_fields(fields: list[str] | None) -> list[str]:
    selected = DEFAULT_STAT_FIELDS if fields is None else list(dict.fromkeys(fields))
    invalid = set(selected) - ALLOWED_STAT_FIELDS
    if invalid:
        raise ValueError(f"Unsupported stat fields: {sorted(invalid)}")
    return selected


def _fields(
    fields: list[str] | None,
    allowed: set[str],
    defaults: list[str],
) -> list[str]:
    selected = defaults if fields is None else list(dict.fromkeys(fields))
    invalid = set(selected) - allowed
    if invalid:
        raise ValueError(f"Unsupported fields: {sorted(invalid)}")
    return selected


def _team_code(team: str) -> str:
    code = team.strip().upper()
    if not code.isalpha() or not 2 <= len(code) <= 3:
        raise ValueError("team must be a two- or three-letter NFL abbreviation")
    return code


def _season_type(season_type: str | None) -> str:
    value = "REG" if season_type is None else season_type.strip().upper()
    if value not in {"REG", "POST"}:
        raise ValueError("season_type must be REG, POST, or null")
    return value


def find_players(name: str) -> list[dict]:
    """Find players by a case-insensitive partial name."""
    query = name.strip()
    if not query:
        raise ValueError("name cannot be empty")
    return repository.find_players(query)


def get_player_weekly_stats(
    player_id: str,
    season: int,
    week: int | None = None,
    fields: list[str] | None = None,
) -> list[dict]:
    """Return one player's weekly stats, optionally for one week."""
    return repository.get_player_weekly_stats(
        player_id, season, week, _stat_fields(fields)
    )


def get_team_games(team: str, season: int, week: int | None = None) -> list[dict]:
    """Return recent schedule/results data for a team."""
    return repository.get_team_games(_team_code(team), season, week)


def get_player_season_stats(
    player_id: str,
    season: int,
    season_type: str | None = None,
    fields: list[str] | None = None,
) -> dict:
    """Return stored season totals and efficiency metrics for one player."""
    selected = _fields(fields, SEASON_STAT_FIELDS, DEFAULT_SEASON_FIELDS)
    return repository.get_player_season_stats(
        player_id, season, _season_type(season_type), selected
    )


def get_team_weekly_stats(
    team: str,
    season: int,
    week: int | None = None,
    fields: list[str] | None = None,
) -> list[dict]:
    """Return a team's offensive totals for one week or an entire season."""
    selected = _fields(fields, TEAM_STAT_FIELDS, DEFAULT_TEAM_FIELDS)
    return repository.get_team_weekly_stats(
        _team_code(team), season, week, selected
    )


def get_team_depth_chart(
    team: str,
    season: int,
    week: int | None = None,
    position: str | None = None,
) -> list[dict]:
    """Return a current or historical team depth chart."""
    position_code = position.strip().upper() if position else None
    return repository.get_team_depth_chart(
        _team_code(team), season, week, position_code
    )


def get_player_snap_counts(
    player_id: str,
    season: int,
    week: int | None = None,
) -> list[dict]:
    """Return weekly offensive, defensive, and special-teams usage."""
    return repository.get_player_snap_counts(player_id, season, week)


def get_team_roster(
    team: str,
    season: int,
    week: int | None = None,
    position: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Return a weekly roster, defaulting to the latest available week."""
    position_code = position.strip().upper() if position else None
    status_code = status.strip().upper() if status else None
    return repository.get_team_roster(
        _team_code(team), season, week, position_code, status_code
    )


def rank_players_by_formula(
    season: int,
    formula: str,
    season_type: str | None = None,
    position: str | None = None,
    minimum_field: str | None = None,
    minimum_value: float | None = None,
    sort_direction: str = "desc",
    limit: int = 10,
) -> dict:
    """Rank player seasons using a safe, model-provided arithmetic formula."""
    parsed = parse_formula(formula, PLAYER_FORMULA_FIELDS)
    if (minimum_field is None) != (minimum_value is None):
        raise ValueError("minimum_field and minimum_value must both be set or null")
    if minimum_field is not None and minimum_field not in PLAYER_FORMULA_FIELDS:
        raise ValueError(f"Unsupported minimum field: {minimum_field}")
    if minimum_value is not None and minimum_value < 0:
        raise ValueError("minimum_value cannot be negative")
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be asc or desc")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    position_code = position.strip().upper() if position else None
    candidates = repository.get_player_season_candidates(
        season,
        _season_type(season_type),
        position_code,
        list(parsed.fields),
        minimum_field,
        minimum_value,
    )

    ranked: list[tuple[float, dict]] = []
    zero_denominator: list[dict] = []
    invalid_rows = 0
    for row in candidates:
        try:
            value = evaluate_formula(parsed, row)
        except ZeroDenominatorError:
            zero_denominator.append(row)
        except (TypeError, ValueError):
            invalid_rows += 1
        else:
            ranked.append((value, row))

    ranked.sort(key=lambda item: item[1]["player_id"])
    ranked.sort(key=lambda item: item[0], reverse=sort_direction == "desc")
    zero_denominator.sort(key=lambda row: row["player_id"])
    selected = ranked[:limit]
    selected_zero = zero_denominator[:limit]
    names = repository.get_player_names(
        [row["player_id"] for _value, row in selected]
        + [row["player_id"] for row in selected_zero]
    )

    input_fields = list(parsed.fields)
    if minimum_field and minimum_field not in input_fields:
        input_fields.append(minimum_field)

    def player_result(row: dict) -> dict:
        return {
            "player_id": row["player_id"],
            "display_name": names.get(row["player_id"]),
            "position": row.get("position"),
            "team": row.get("last_team"),
            "games": row.get("games"),
            "inputs": {field: row.get(field) for field in input_fields},
        }

    return {
        "formula": parsed.canonical,
        "season": season,
        "season_type": _season_type(season_type),
        "position": position_code,
        "minimum": (
            {"field": minimum_field, "value": minimum_value}
            if minimum_field is not None
            else None
        ),
        "sort_direction": sort_direction,
        "eligible_players": len(candidates),
        "excluded_invalid_rows": invalid_rows,
        "zero_denominator_qualifiers": [
            player_result(row) for row in selected_zero
        ],
        "results": [
            {
                "rank": rank,
                "metric_value": round(value, 6),
                **player_result(row),
            }
            for rank, (value, row) in enumerate(selected, start=1)
        ],
    }


def rank_teams_by_formula(
    season: int,
    perspective: str,
    formula: str,
    season_type: str | None = None,
    minimum_games: int | None = None,
    sort_direction: str = "desc",
    limit: int = 10,
) -> dict:
    """Rank team seasons using offense or opponent-derived defense metrics."""
    if perspective not in {"offense", "defense"}:
        raise ValueError("perspective must be offense or defense")
    allowed_fields = (
        TEAM_OFFENSE_FORMULA_FIELDS
        if perspective == "offense"
        else TEAM_DEFENSE_FORMULA_FIELDS
    )
    parsed = parse_formula(formula, allowed_fields)
    if minimum_games is not None and minimum_games < 1:
        raise ValueError("minimum_games must be positive or null")
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be asc or desc")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    normalized_season_type = _season_type(season_type)
    weekly_rows, games = repository.get_team_formula_inputs(
        season, normalized_season_type
    )
    candidates = build_team_season_rows(
        weekly_rows, games, season, normalized_season_type
    )
    if minimum_games is not None:
        candidates = [
            row for row in candidates if row["games"] >= minimum_games
        ]

    ranked: list[tuple[float, dict]] = []
    zero_denominator: list[dict] = []
    invalid_rows = 0
    for row in candidates:
        try:
            value = evaluate_formula(parsed, row)
        except ZeroDenominatorError:
            zero_denominator.append(row)
        except (TypeError, ValueError):
            invalid_rows += 1
        else:
            ranked.append((value, row))

    ranked.sort(key=lambda item: item[1]["team"])
    ranked.sort(key=lambda item: item[0], reverse=sort_direction == "desc")
    zero_denominator.sort(key=lambda row: row["team"])
    selected = ranked[:limit]
    selected_zero = zero_denominator[:limit]
    names = repository.get_team_names(
        [row["team"] for _value, row in selected]
        + [row["team"] for row in selected_zero]
    )
    input_fields = list(parsed.fields)
    if "games" not in input_fields:
        input_fields.append("games")

    def team_result(row: dict) -> dict:
        return {
            "team": row["team"],
            "team_name": names.get(row["team"]),
            "games": row["games"],
            "inputs": {field: row.get(field) for field in input_fields},
        }

    return {
        "formula": parsed.canonical,
        "season": season,
        "season_type": normalized_season_type,
        "perspective": perspective,
        "minimum_games": minimum_games,
        "sort_direction": sort_direction,
        "eligible_teams": len(candidates),
        "excluded_invalid_rows": invalid_rows,
        "zero_denominator_qualifiers": [
            team_result(row) for row in selected_zero
        ],
        "results": [
            {
                "rank": rank,
                "metric_value": round(value, 6),
                **team_result(row),
            }
            for rank, (value, row) in enumerate(selected, start=1)
        ],
    }


def rank_players_by_weekly_threshold(
    season: int,
    formula: str,
    comparison: str,
    threshold: float,
    participation_basis: str,
    season_type: str | None = None,
    position: str | None = None,
    player_ids: list[str] | None = None,
    minimum_games: int = 1,
    rank_by: str = "qualifying_games",
    sort_direction: str = "desc",
    include_week_details: bool = False,
    limit: int = 5,
) -> dict:
    """Rank players by games or game rate meeting a weekly threshold."""
    parsed = parse_formula(formula, set(PLAYER_WEEKLY_STAT_FIELDS))
    if comparison not in COMPARISONS:
        raise ValueError(f"comparison must be one of {sorted(COMPARISONS)}")
    if participation_basis not in PARTICIPATION_BASES:
        raise ValueError(
            f"participation_basis must be one of {sorted(PARTICIPATION_BASES)}"
        )
    if rank_by not in WEEKLY_RANK_FIELDS:
        raise ValueError(f"rank_by must be one of {sorted(WEEKLY_RANK_FIELDS)}")
    if not 1 <= minimum_games <= 25:
        raise ValueError("minimum_games must be between 1 and 25")
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be asc or desc")
    selected_player_ids = None
    if player_ids is not None:
        selected_player_ids = list(dict.fromkeys(player_id.strip() for player_id in player_ids))
        if not selected_player_ids or any(not player_id for player_id in selected_player_ids):
            raise ValueError("player_ids must contain at least one non-empty ID or be null")
        if len(selected_player_ids) > 20:
            raise ValueError("player_ids cannot contain more than 20 IDs")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")

    normalized_season_type = _season_type(season_type)
    position_code = position.strip().upper() if position else None
    weekly_rows, roster_rows = repository.get_player_weekly_analysis_inputs(
        season,
        normalized_season_type,
        position_code,
        selected_player_ids,
        list(parsed.fields),
        include_rosters=participation_basis != "stat_row",
    )
    rows = build_weekly_threshold_rows(
        weekly_rows,
        roster_rows,
        parsed,
        participation_basis,
        comparison,
        threshold,
    )
    eligible = [
        row for row in rows if row["denominator_games"] >= minimum_games
    ]
    eligible.sort(key=lambda row: row["player_id"])
    eligible.sort(
        key=lambda row: (
            row[rank_by] is not None,
            row[rank_by] if row[rank_by] is not None else 0,
        ),
        reverse=sort_direction == "desc",
    )
    selected = eligible[:limit]
    names = repository.get_player_names([row["player_id"] for row in selected])

    return {
        "formula": parsed.canonical,
        "comparison": comparison,
        "threshold": threshold,
        "season": season,
        "season_type": normalized_season_type,
        "position": position_code,
        "candidate_player_ids": selected_player_ids,
        "participation_basis": participation_basis,
        "minimum_games": minimum_games,
        "rank_by": rank_by,
        "sort_direction": sort_direction,
        "eligible_players": len(eligible),
        "results": [
            {
                "rank": rank,
                "player_id": row["player_id"],
                "display_name": names.get(row["player_id"]),
                "position": row["position"],
                "team": row["team"],
                "qualifying_games": row["qualifying_games"],
                "denominator_games": row["denominator_games"],
                "qualifying_rate": (
                    round(row["qualifying_rate"], 6)
                    if row["qualifying_rate"] is not None
                    else None
                ),
                **(
                    {
                        "stat_row_games": row["stat_row_games"],
                        "active_games": row["active_games"],
                        "team_games": row["team_games"],
                        "evaluated_games": row["evaluated_games"],
                        "undefined_formula_games": row["undefined_formula_games"],
                        "undefined_formula_weeks": row["undefined_formula_weeks"],
                        "qualifying_weeks": row["qualifying_weeks"],
                        "average": (
                            round(row["average"], 6)
                            if row["average"] is not None
                            else None
                        ),
                        "median": (
                            round(row["median"], 6)
                            if row["median"] is not None
                            else None
                        ),
                    }
                    if include_week_details
                    else {}
                ),
            }
            for rank, row in enumerate(selected, start=1)
        ],
    }


ECR_FORMATS = {
    "redraft_1qb": ("ppr", "final_preseason"),
    "redraft_superflex": ("ppr", "final_preseason"),
    "dynasty_1qb": ("source_default", "season_opening"),
    "dynasty_superflex": ("source_default", "season_opening"),
    "dynasty_rookie": ("source_default", "season_opening"),
    "best_ball": ("source_default", "season_opening"),
    "redraft_idp": ("source_default", "final_preseason"),
    "dynasty_idp": ("source_default", "season_opening"),
}


def _ecr_options(
    scoring_format: str,
    league_format: str,
    snapshot_type: str,
) -> None:
    if league_format not in ECR_FORMATS:
        raise ValueError(f"league_format must be one of {sorted(ECR_FORMATS)}")
    expected_scoring, historical_snapshot = ECR_FORMATS[league_format]
    if scoring_format != expected_scoring:
        raise ValueError(
            f"{league_format} requires scoring_format={expected_scoring}"
        )
    if snapshot_type not in {"current", historical_snapshot}:
        raise ValueError(
            f"{league_format} supports current or {historical_snapshot} snapshots"
        )


def _positions(positions: list[str] | None) -> list[str] | None:
    if positions is None:
        return None
    selected = list(dict.fromkeys(position.strip().upper() for position in positions))
    if not selected or set(selected) - ECR_POSITIONS:
        raise ValueError(f"positions must use {sorted(ECR_POSITIONS)}")
    return selected


def rank_players_by_ecr(
    season: int,
    positions: list[str] | None,
    scoring_format: str,
    league_format: str,
    snapshot_type: str,
    as_of_date: str | None,
    sort_by: str,
    sort_direction: str,
    minimum_overall_rank: float | None,
    maximum_overall_rank: float | None,
    limit: int,
) -> dict:
    """Rank one ECR snapshot by consensus, movement, or disagreement."""
    _ecr_options(scoring_format, league_format, snapshot_type)
    selected_positions = _positions(positions)
    if sort_by not in ECR_SORT_FIELDS:
        raise ValueError(f"sort_by must be one of {sorted(ECR_SORT_FIELDS)}")
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be asc or desc")
    if minimum_overall_rank is not None and minimum_overall_rank < 1:
        raise ValueError("minimum_overall_rank must be at least 1 or null")
    if maximum_overall_rank is not None and maximum_overall_rank < 1:
        raise ValueError("maximum_overall_rank must be at least 1 or null")
    if (
        minimum_overall_rank is not None
        and maximum_overall_rank is not None
        and minimum_overall_rank > maximum_overall_rank
    ):
        raise ValueError("minimum_overall_rank cannot exceed maximum_overall_rank")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    selected_date, rows = repository.get_ecr_rows(
        season, scoring_format, league_format, snapshot_type, as_of_date
    )
    rows = add_ecr_position_ranks(rows)
    if selected_positions is not None:
        rows = [row for row in rows if row.get("position") in selected_positions]
    if minimum_overall_rank is not None:
        rows = [row for row in rows if row["overall_rank"] >= minimum_overall_rank]
    if maximum_overall_rank is not None:
        rows = [row for row in rows if row["overall_rank"] <= maximum_overall_rank]

    # Null analytical values sort behind defined values in either direction.
    rows.sort(key=lambda row: row["player_id"])
    defined = [row for row in rows if row.get(sort_by) is not None]
    undefined = [row for row in rows if row.get(sort_by) is None]
    defined.sort(key=lambda row: row[sort_by], reverse=sort_direction == "desc")
    selected = (defined + undefined)[:limit]
    names = repository.get_player_names([row["player_id"] for row in selected])
    return {
        "season": season,
        "scrape_date": selected_date,
        "scoring_format": scoring_format,
        "league_format": league_format,
        "snapshot_type": snapshot_type,
        "positions": selected_positions,
        "sort_by": sort_by,
        "sort_direction": sort_direction,
        "eligible_players": len(rows),
        "results": [
            {
                "rank": index,
                "player_id": row["player_id"],
                "display_name": names.get(row["player_id"]),
                "position": row.get("position"),
                "team": row.get("team"),
                "overall_rank": row.get("overall_rank"),
                "position_rank": row.get("position_rank"),
                "best_rank": row.get("best_rank"),
                "worst_rank": row.get("worst_rank"),
                "rank_sd": row.get("rank_sd"),
                "rank_range": row.get("rank_range"),
                "rank_delta": row.get("rank_delta"),
            }
            for index, row in enumerate(selected, start=1)
        ],
    }


def compare_ecr_to_results(
    season: int,
    positions: list[str] | None,
    scoring_format: str,
    league_format: str,
    snapshot_type: str,
    comparison_basis: str,
    sort_by: str,
    sort_direction: str,
    minimum_games: int,
    minimum_overall_rank: float | None,
    maximum_overall_rank: float | None,
    limit: int,
) -> dict:
    """Compare a preseason/current ECR snapshot with actual season results."""
    _ecr_options(scoring_format, league_format, snapshot_type)
    selected_positions = _positions(positions)
    if league_format not in {"redraft_1qb", "redraft_superflex"}:
        raise ValueError("Result comparisons currently support redraft formats only")
    if snapshot_type != "final_preseason":
        raise ValueError("Result comparisons require a final_preseason snapshot")
    if (
        selected_positions is not None
        and set(selected_positions) - ECR_COMPARISON_POSITIONS
    ):
        raise ValueError(
            "Result comparisons support QB, RB, WR, TE, and K positions only"
        )
    if comparison_basis not in {"overall", "position"}:
        raise ValueError("comparison_basis must be overall or position")
    if sort_by not in ECR_COMPARISON_SORT_FIELDS:
        raise ValueError(
            f"sort_by must be one of {sorted(ECR_COMPARISON_SORT_FIELDS)}"
        )
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be asc or desc")
    if not 0 <= minimum_games <= 25:
        raise ValueError("minimum_games must be between 0 and 25")
    if minimum_overall_rank is not None and minimum_overall_rank < 1:
        raise ValueError("minimum_overall_rank must be at least 1 or null")
    if maximum_overall_rank is not None and maximum_overall_rank < 1:
        raise ValueError("maximum_overall_rank must be at least 1 or null")
    if (
        minimum_overall_rank is not None
        and maximum_overall_rank is not None
        and minimum_overall_rank > maximum_overall_rank
    ):
        raise ValueError("minimum_overall_rank cannot exceed maximum_overall_rank")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    selected_date, ecr_rows = repository.get_ecr_rows(
        season, scoring_format, league_format, snapshot_type, None
    )
    comparisons = build_ecr_comparisons(
        ecr_rows,
        repository.get_season_fantasy_results(season),
        scoring_format,
    )
    if selected_positions is not None:
        comparisons = [
            row for row in comparisons if row.get("position") in selected_positions
        ]
    comparisons = [row for row in comparisons if row["games"] >= minimum_games]
    if minimum_overall_rank is not None:
        comparisons = [
            row for row in comparisons
            if row["overall_rank"] >= minimum_overall_rank
        ]
    if maximum_overall_rank is not None:
        comparisons = [
            row for row in comparisons
            if row["overall_rank"] <= maximum_overall_rank
        ]

    basis_fields = {
        "rank_difference": f"{comparison_basis}_rank_difference",
        "ecr_rank": "overall_rank" if comparison_basis == "overall" else "position_rank",
        "actual_finish": (
            "actual_overall_finish"
            if comparison_basis == "overall"
            else "actual_position_finish"
        ),
        "fantasy_points": "fantasy_points",
        "fantasy_points_per_game": "fantasy_points_per_game",
    }
    field = basis_fields[sort_by]
    comparisons.sort(key=lambda row: row["player_id"])
    defined = [row for row in comparisons if row.get(field) is not None]
    undefined = [row for row in comparisons if row.get(field) is None]
    defined.sort(key=lambda row: row[field], reverse=sort_direction == "desc")
    selected = (defined + undefined)[:limit]
    names = repository.get_player_names([row["player_id"] for row in selected])
    return {
        "season": season,
        "scrape_date": selected_date,
        "scoring_format": scoring_format,
        "league_format": league_format,
        "snapshot_type": snapshot_type,
        "positions": selected_positions,
        "comparison_basis": comparison_basis,
        "sort_by": sort_by,
        "sort_direction": sort_direction,
        "minimum_games": minimum_games,
        "positive_rank_difference_means": "outperformed ECR",
        "eligible_players": len(comparisons),
        "results": [
            {
                "rank": index,
                "player_id": row["player_id"],
                "display_name": names.get(row["player_id"]),
                "position": row.get("position"),
                "ecr_team": row.get("team"),
                "result_team": row.get("result_team"),
                "games": row.get("games"),
                "ecr_overall_rank": row.get("overall_rank"),
                "ecr_position_rank": row.get("position_rank"),
                "actual_overall_finish": row.get("actual_overall_finish"),
                "actual_position_finish": row.get("actual_position_finish"),
                "overall_rank_difference": row.get("overall_rank_difference"),
                "position_rank_difference": row.get("position_rank_difference"),
                "fantasy_points": row.get("fantasy_points"),
                "fantasy_points_per_game": row.get("fantasy_points_per_game"),
            }
            for index, row in enumerate(selected, start=1)
        ],
    }


TOOL_HANDLERS = {
    "find_players": find_players,
    "get_player_weekly_stats": get_player_weekly_stats,
    "get_team_games": get_team_games,
    "get_player_season_stats": get_player_season_stats,
    "get_team_weekly_stats": get_team_weekly_stats,
    "get_team_depth_chart": get_team_depth_chart,
    "get_player_snap_counts": get_player_snap_counts,
    "get_team_roster": get_team_roster,
    "rank_players_by_formula": rank_players_by_formula,
    "rank_teams_by_formula": rank_teams_by_formula,
    "rank_players_by_weekly_threshold": rank_players_by_weekly_threshold,
    "rank_players_by_ecr": rank_players_by_ecr,
    "compare_ecr_to_results": compare_ecr_to_results,
}
