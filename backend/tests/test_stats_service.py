import unittest

from services.news import PlayerCandidate, TeamCandidate
from services.stats import (
    PlayerLeadersQuery,
    PlayerStatsQuery,
    StatsFieldsQuery,
    StatsOutcome,
    StatsScope,
    StatsService,
    TeamLeadersQuery,
    TeamStatsQuery,
)


class FakeStatsRepository:
    def __init__(self) -> None:
        self.players = [PlayerCandidate("player-1", "A.J. Brown", "WR", "PHI", "ACT")]
        self.teams = [
            TeamCandidate("PHI", "Philadelphia Eagles", "Eagles"),
            TeamCandidate("DAL", "Dallas Cowboys", "Cowboys"),
        ]
        self.player_leader_query = None
        self.team_leader_query = None

    def find_players(self, name):
        needle = name.casefold().replace(".", "")
        return [player for player in self.players if needle in player.display_name.casefold().replace(".", "")]

    def list_teams(self):
        return self.teams

    def latest_player_season(self, player_id, season_type):
        return 2025

    def latest_leader_season(self, season_type):
        return 2025

    def latest_team_season(self, season_type):
        return 2025

    def player_season(self, player_id, season, season_type, fields):
        source = {
            "games": 17,
            "targets": 150,
            "receptions": 100,
            "receiving_yards": 1500,
            "receiving_tds": 10,
            "catch_percentage": 2 / 3,
            "receiving_yards_per_reception": 15,
            "receiving_yards_per_target": 10,
            "fantasy_points": 210,
            "fantasy_points_ppr": 310,
            "fantasy_points_per_game": 12.35,
            "fantasy_points_ppr_per_game": 18.24,
        }
        return {field: source.get(field) for field in fields}

    def player_week(self, player_id, season, week, fields):
        source = {"targets": 10, "receptions": 8, "receiving_yards": 120, "week": week, "team": "PHI", "opponent_team": "DAL"}
        return [{**{field: source.get(field) for field in fields}, "week": week, "team": "PHI", "opponent_team": "DAL"}]

    def player_leaders(self, query, season):
        self.player_leader_query = query
        return {"results": [{"rank": 1, "player_id": "player-1", "display_name": "A.J. Brown", "position": "WR", "team": "PHI", "metric_value": 10.0, "inputs": {"receiving_yards": 1500, "targets": 150}}]}

    def team_rows(self, season, season_type, week):
        return [{"team": "PHI", "games": 17, "points_scored": 450, "points_allowed": 300, "attempts": 500, "carries": 500, "passing_yards": 4000, "passing_tds": 30, "rushing_yards": 2400, "rushing_tds": 20, "passing_yards_allowed": 3200, "passing_tds_allowed": 18, "rushing_yards_allowed": 1600, "rushing_tds_allowed": 10}]

    def team_leaders(self, query, season):
        self.team_leader_query = query
        return {"results": [{"rank": 1, "team": "PHI", "team_name": "Philadelphia Eagles", "games": 17, "metric_value": 26.47, "inputs": {"points_scored": 450, "games": 17}}]}


class StatsServiceTests(unittest.TestCase):
    def setUp(self):
        self.repository = FakeStatsRepository()
        self.service = StatsService(self.repository)

    def test_player_formula_uses_latest_available_season(self):
        result = self.service.execute(PlayerStatsQuery(player="AJ Brown", formula="receiving_yards / targets"))

        self.assertEqual(result.outcome, StatsOutcome.SUCCESS)
        self.assertEqual(result.season, 2025)
        self.assertEqual(result.resolved_player.display_name, "A.J. Brown")
        self.assertEqual(result.formula, "receiving_yards / targets")
        self.assertEqual(result.rows[0]["metric_value"], 10)

    def test_player_week_uses_weekly_formula_catalog(self):
        result = self.service.execute(PlayerStatsQuery(player="A.J. Brown", season=2025, week=1, formula="receptions + receiving_yards"))

        self.assertEqual(result.outcome, StatsOutcome.SUCCESS)
        self.assertEqual(result.rows[0]["metric_value"], 128)
        self.assertEqual(result.rows[0]["opponent_team"], "DAL")

    def test_invalid_formula_is_a_user_facing_outcome(self):
        result = self.service.execute(PlayerStatsQuery(player="A.J. Brown", formula="drop table players"))

        self.assertEqual(result.outcome, StatsOutcome.INVALID_FORMULA)

    def test_player_leaders_preserve_eligibility_inputs(self):
        query = PlayerLeadersQuery(formula="receiving_yards / targets", position="WR", minimum_field="targets", minimum_value=50)
        result = self.service.execute(query)

        self.assertEqual(result.outcome, StatsOutcome.SUCCESS)
        self.assertEqual(result.rows[0]["display_name"], "A.J. Brown")
        self.assertEqual(self.repository.player_leader_query.minimum_value, 50)

    def test_team_defense_formula_uses_allowed_fields(self):
        result = self.service.execute(TeamStatsQuery(team="Eagles", perspective="defense", formula="points_allowed / games"))

        self.assertEqual(result.outcome, StatsOutcome.SUCCESS)
        self.assertAlmostEqual(result.rows[0]["metric_value"], 300 / 17)

    def test_team_leaders_delegate_to_existing_analytics(self):
        result = self.service.execute(TeamLeadersQuery(formula="points_scored / games"))

        self.assertEqual(result.outcome, StatsOutcome.SUCCESS)
        self.assertEqual(result.rows[0]["team"], "PHI")

    def test_fields_are_derived_from_scope_catalog(self):
        result = self.service.execute(StatsFieldsQuery(StatsScope.TEAM_DEFENSE, search="passing_yards"))

        self.assertEqual(result.outcome, StatsOutcome.SUCCESS)
        self.assertEqual(result.rows, ({"field": "passing_yards_after_catch_allowed"}, {"field": "passing_yards_allowed"}))


if __name__ == "__main__":
    unittest.main()
