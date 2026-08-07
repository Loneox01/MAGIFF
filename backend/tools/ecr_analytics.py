"""Application-side ECR ranking and season-result comparison helpers."""

from collections import defaultdict


ECR_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DL", "LB", "DB"}
ECR_COMPARISON_POSITIONS = {"QB", "RB", "WR", "TE", "K"}
ECR_SORT_FIELDS = {
    "overall_rank",
    "position_rank",
    "rank_sd",
    "rank_range",
    "rank_delta",
}
ECR_COMPARISON_SORT_FIELDS = {
    "rank_difference",
    "ecr_rank",
    "actual_finish",
    "fantasy_points",
    "fantasy_points_per_game",
}


def competition_ranks(
    rows: list[dict],
    value_key: str,
    *,
    descending: bool,
) -> dict[str, int]:
    """Return stable competition ranks (1, 2, 2, 4) by player ID."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -(row[value_key] or 0) if descending else (row[value_key] or 0),
            row["player_id"],
        ),
    )
    ranks: dict[str, int] = {}
    previous = object()
    rank = 0
    for index, row in enumerate(ordered, start=1):
        value = row[value_key]
        if value != previous:
            rank = index
            previous = value
        ranks[row["player_id"]] = rank
    return ranks


def add_ecr_position_ranks(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("position") or "UNKNOWN"].append(row)
    for group in groups.values():
        ranks = competition_ranks(group, "overall_rank", descending=False)
        for row in group:
            row["position_rank"] = ranks[row["player_id"]]
            row["rank_range"] = (
                row["worst_rank"] - row["best_rank"]
                if row.get("worst_rank") is not None
                and row.get("best_rank") is not None
                else None
            )
    return rows


def fantasy_value(row: dict, scoring_format: str) -> float:
    if scoring_format == "ppr":
        return float(row.get("fantasy_points_ppr") or 0)
    if scoring_format == "half_ppr":
        return float(row.get("fantasy_points") or 0) + 0.5 * float(
            row.get("receptions") or 0
        )
    return float(row.get("fantasy_points") or 0)


def build_ecr_comparisons(
    ecr_rows: list[dict],
    season_rows: list[dict],
    scoring_format: str,
) -> list[dict]:
    """Join ECR to results and calculate overall/positional finishes."""
    add_ecr_position_ranks(ecr_rows)
    season_by_player = {row["player_id"]: row for row in season_rows}

    performance_rows = []
    for row in season_rows:
        if row.get("position") not in ECR_COMPARISON_POSITIONS:
            continue
        performance_rows.append(
            {
                **row,
                "fantasy_points_value": fantasy_value(row, scoring_format),
            }
        )

    # Preserve drafted players who recorded no season-stat row so bust queries
    # can include injuries, cuts, and other zero-game outcomes.
    known_ids = {row["player_id"] for row in performance_rows}
    for ecr in ecr_rows:
        if ecr["player_id"] not in known_ids:
            performance_rows.append(
                {
                    "player_id": ecr["player_id"],
                    "position": ecr.get("position"),
                    "last_team": None,
                    "games": 0,
                    "fantasy_points_value": 0.0,
                }
            )

    overall_finishes = competition_ranks(
        performance_rows, "fantasy_points_value", descending=True
    )
    position_finishes: dict[str, int] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in performance_rows:
        groups[row.get("position") or "UNKNOWN"].append(row)
    for group in groups.values():
        position_finishes.update(
            competition_ranks(group, "fantasy_points_value", descending=True)
        )

    comparisons = []
    for ecr in ecr_rows:
        result = season_by_player.get(ecr["player_id"], {})
        games = int(result.get("games") or 0)
        points = fantasy_value(result, scoring_format)
        comparisons.append(
            {
                **ecr,
                "games": games,
                "result_team": result.get("last_team"),
                "fantasy_points": round(points, 6),
                "fantasy_points_per_game": (
                    round(points / games, 6) if games else None
                ),
                "actual_overall_finish": overall_finishes[ecr["player_id"]],
                "actual_position_finish": position_finishes[ecr["player_id"]],
                "overall_rank_difference": round(
                    ecr["overall_rank"]
                    - overall_finishes[ecr["player_id"]],
                    6,
                ),
                "position_rank_difference": (
                    ecr["position_rank"]
                    - position_finishes[ecr["player_id"]]
                ),
            }
        )
    return comparisons
