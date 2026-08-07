"""Build team-season offense and opponent-derived defense aggregates."""

from collections import defaultdict
from typing import Any

from tools.field_catalog import TEAM_WEEKLY_STAT_FIELDS

TEAM_SOURCE_FIELDS = set(TEAM_WEEKLY_STAT_FIELDS)

TEAM_OFFENSE_FORMULA_FIELDS = {"games", "points_scored", *TEAM_SOURCE_FIELDS}
TEAM_DEFENSE_FORMULA_FIELDS = {
    "games",
    "points_allowed",
    *(f"{field}_allowed" for field in TEAM_SOURCE_FIELDS),
}
ADDITIVE_TEAM_FIELDS = TEAM_SOURCE_FIELDS - {"passing_cpoe"}


def _new_team_row(team: str, season: int, season_type: str) -> dict[str, Any]:
    return {
        "team": team,
        "season": season,
        "season_type": season_type,
        "games": 0,
        "points_scored": 0,
        "points_allowed": 0,
        **{field: 0 for field in TEAM_SOURCE_FIELDS},
        **{f"{field}_allowed": 0 for field in TEAM_SOURCE_FIELDS},
        "_passing_cpoe_weight": 0,
        "_passing_cpoe_allowed_weight": 0,
    }


def build_team_season_rows(
    weekly_rows: list[dict],
    games: list[dict],
    season: int,
    season_type: str,
) -> list[dict]:
    """Aggregate offense and mirror each opponent row into defense allowed."""
    game_scores = {
        game["game_id"]: {
            game["home_team"]: game.get("home_score"),
            game["away_team"]: game.get("away_score"),
        }
        for game in games
    }
    teams: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    seen_games: defaultdict[str, set[str]] = defaultdict(set)

    def team_row(team: str) -> dict[str, Any]:
        if not teams[team]:
            teams[team] = _new_team_row(team, season, season_type)
        return teams[team]

    for weekly in weekly_rows:
        team = weekly["team"]
        opponent = weekly["opponent_team"]
        offense = team_row(team)
        defense = team_row(opponent)

        if weekly["game_id"] not in seen_games[team]:
            seen_games[team].add(weekly["game_id"])
            offense["games"] += 1

        for field in ADDITIVE_TEAM_FIELDS:
            value = weekly.get(field)
            if value is not None:
                offense[field] += value
                defense[f"{field}_allowed"] += value

        passing_cpoe = weekly.get("passing_cpoe")
        attempts = weekly.get("attempts") or 0
        if passing_cpoe is not None and attempts:
            offense["_passing_cpoe_weight"] += passing_cpoe * attempts
            defense["_passing_cpoe_allowed_weight"] += passing_cpoe * attempts

        scores = game_scores.get(weekly["game_id"], {})
        team_score = scores.get(team)
        opponent_score = scores.get(opponent)
        if team_score is not None:
            offense["points_scored"] += team_score
        if opponent_score is not None:
            offense["points_allowed"] += opponent_score

    for row in teams.values():
        attempts = row["attempts"]
        attempts_allowed = row["attempts_allowed"]
        row["passing_cpoe"] = (
            row.pop("_passing_cpoe_weight") / attempts if attempts else None
        )
        row["passing_cpoe_allowed"] = (
            row.pop("_passing_cpoe_allowed_weight") / attempts_allowed
            if attempts_allowed
            else None
        )

    return sorted(teams.values(), key=lambda row: row["team"])
