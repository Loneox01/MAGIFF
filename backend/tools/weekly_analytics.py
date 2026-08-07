"""Weekly threshold and participation analysis for player seasons."""

from collections import defaultdict
from statistics import mean, median
from typing import Any

from tools.formulas import ZeroDenominatorError, evaluate_formula, ParsedFormula


PARTICIPATION_BASES = {"stat_row", "active_roster", "team_games"}
COMPARISONS = {"gte", "gt", "lte", "lt", "eq"}
WEEKLY_RANK_FIELDS = {"qualifying_games", "qualifying_rate"}


def _qualifies(value: float, comparison: str, threshold: float) -> bool:
    if comparison == "gte":
        return value >= threshold
    if comparison == "gt":
        return value > threshold
    if comparison == "lte":
        return value <= threshold
    if comparison == "lt":
        return value < threshold
    return value == threshold


def build_weekly_threshold_rows(
    weekly_rows: list[dict],
    roster_rows: list[dict],
    parsed: ParsedFormula,
    participation_basis: str,
    comparison: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Build one threshold-analysis record per player."""
    stats_by_player: defaultdict[str, dict[int, dict]] = defaultdict(dict)
    roster_by_player: defaultdict[str, dict[int, dict]] = defaultdict(dict)

    for row in weekly_rows:
        stats_by_player[row["player_id"]][row["week"]] = row
    for row in roster_rows:
        roster_by_player[row["player_id"]][row["week"]] = row

    player_ids = set(stats_by_player) | set(roster_by_player)
    results: list[dict[str, Any]] = []
    for player_id in player_ids:
        stats = stats_by_player[player_id]
        roster = roster_by_player[player_id]
        stat_weeks = set(stats)
        active_weeks = {
            week for week, row in roster.items() if row.get("status") == "ACT"
        }
        rostered_weeks = set(roster)

        if participation_basis == "stat_row":
            denominator_weeks = stat_weeks
        elif participation_basis == "active_roster":
            denominator_weeks = active_weeks
        else:
            denominator_weeks = rostered_weeks

        values: list[float] = []
        qualifying_weeks: list[int] = []
        undefined_weeks: list[int] = []
        for week in sorted(denominator_weeks):
            stat_row = stats.get(week)
            # An active/rostered week without a stats row represents zero
            # production. Preserve nulls in real rows so unavailable inputs are
            # not silently converted into zero.
            inputs = (
                stat_row
                if stat_row is not None
                else {field: 0 for field in parsed.fields}
            )
            try:
                value = evaluate_formula(parsed, inputs)
            except (ZeroDenominatorError, TypeError, ValueError):
                undefined_weeks.append(week)
                continue
            values.append(value)
            if _qualifies(value, comparison, threshold):
                qualifying_weeks.append(week)

        denominator_games = len(denominator_weeks)
        qualifying_games = len(qualifying_weeks)
        latest_week = max(denominator_weeks) if denominator_weeks else None
        latest = roster.get(latest_week) or stats.get(latest_week) or {}
        results.append(
            {
                "player_id": player_id,
                "position": latest.get("position"),
                "team": latest.get("team"),
                "stat_row_games": len(stat_weeks),
                "active_games": len(active_weeks),
                "team_games": len(rostered_weeks),
                "denominator_games": denominator_games,
                "evaluated_games": len(values),
                "undefined_formula_games": len(undefined_weeks),
                "undefined_formula_weeks": undefined_weeks,
                "qualifying_games": qualifying_games,
                "qualifying_rate": (
                    qualifying_games / denominator_games
                    if denominator_games
                    else None
                ),
                "qualifying_weeks": qualifying_weeks,
                "average": mean(values) if values else None,
                "median": median(values) if values else None,
            }
        )

    return results
