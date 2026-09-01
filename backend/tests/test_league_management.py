import unittest

import httpx

from drafting.models import DraftCandidate
from integrations.sleeper import SleeperLeagueClient
from league_management.context import LeagueContextBuilder


class FakeSleeperLeagueSource:
    def __init__(self, snapshot):
        self.value = snapshot
        self.calls = []

    def snapshot(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


class FakeLeaguePlayerRepository:
    def __init__(self, profiles):
        self.profiles = profiles
        self.requested = []

    def resolve_players(self, sleeper_player_ids):
        self.requested.append(list(sleeper_player_ids))
        return {
            key: value
            for key, value in self.profiles.items()
            if key in sleeper_player_ids
        }


class FakeMarketRepository:
    def __init__(self, candidates):
        self.candidates = candidates

    def load_candidates(self, **_kwargs):
        return "2026-08-28", list(self.candidates), "FantasyPros", "redraft-overall"


def _candidate(index, sleeper_id, name, position="RB"):
    return DraftCandidate(
        player_id=f"internal-{index}",
        external_id=sleeper_id,
        display_name=name,
        position=position,
        team="TST",
        overall_rank=float(index),
        position_rank=index,
    )


def _snapshot():
    return {
        "user": {"user_id": "user-magiff", "display_name": "Magiff"},
        "league": {
            "league_id": "league-1",
            "name": "Test League",
            "season": "2026",
            "status": "in_season",
            "total_rosters": 2,
            "season_type": "regular",
            "roster_positions": ["QB", "RB", "FLEX", "DEF", "BN"],
            "scoring_settings": {"rec": 1, "pass_td": 4},
            "settings": {
                "waiver_budget": 100,
                "waiver_type": 2,
                "waiver_clear_days": 2,
                "waiver_day_of_week": 2,
                "trade_deadline": 11,
                "trade_review_days": 2,
            },
        },
        "users": [
            {"user_id": "user-other", "display_name": "Opponent"},
            {"user_id": "user-magiff", "display_name": "Magiff"},
        ],
        "rosters": [
            {
                "roster_id": 1,
                "owner_id": "user-other",
                "players": ["s3"],
                "starters": ["s3", "0", "0", "0"],
                "settings": {"waiver_position": 1},
            },
            {
                "roster_id": 6,
                "owner_id": "user-magiff",
                "players": ["s1", "s2", "JAX"],
                "starters": ["s1", "s2", "0", "JAX"],
                "settings": {
                    "wins": 1,
                    "fpts": 120,
                    "fpts_decimal": 35,
                    "waiver_position": 2,
                    "waiver_budget_used": 17,
                },
            },
        ],
        "nfl_state": {
            "week": 1,
            "season_type": "regular",
            "season_start_date": "2026-09-09",
        },
        "week": 1,
        "matchups": [
            {"roster_id": 1, "matchup_id": 4, "points": 80.5},
            {"roster_id": 6, "matchup_id": 4, "points": 91.2},
        ],
        "transactions": [
            {
                "transaction_id": "tx-1",
                "type": "waiver",
                "status": "pending",
                "leg": 1,
                "created": 1_788_133_457_971,
                "roster_ids": [6],
                "adds": {"s4": 6},
                "drops": {"s2": 6},
                "settings": {"waiver_bid": 12},
                "draft_picks": [],
            }
        ],
        "trending_adds": [{"player_id": "s4", "count": 100}],
        "trending_drops": [{"player_id": "s2", "count": 25}],
    }


def _profiles():
    return {
        value: {
            "player_id": f"internal-{index}",
            "display_name": name,
            "position": position,
            "team": "TST",
            "roster_status": "ACT",
        }
        for index, (value, name, position) in enumerate(
            [
                ("s1", "Quarterback One", "QB"),
                ("s2", "Running Back Two", "RB"),
                ("s3", "Opponent Player", "QB"),
                ("s4", "Available Player", "WR"),
                ("s5", "Second Available", "TE"),
                ("JAX", "Jacksonville Jaguars D/ST", "DEF"),
            ],
            start=1,
        )
    }


class LeagueContextBuilderTests(unittest.TestCase):
    def test_build_verifies_owner_and_normalizes_actionable_state(self):
        source = FakeSleeperLeagueSource(_snapshot())
        context = LeagueContextBuilder(
            sleeper=source,
            players=FakeLeaguePlayerRepository(_profiles()),
            market=FakeMarketRepository(
                [
                    _candidate(1, "s1", "Quarterback One", "QB"),
                    _candidate(2, "s3", "Opponent Player", "QB"),
                    _candidate(3, "s4", "Available Player", "WR"),
                    _candidate(4, "s5", "Second Available", "TE"),
                ]
            ),
        ).build(
            league_id="league-1",
            user_reference="Magiff",
            available_limit=10,
        )

        self.assertEqual(context.managed_user_id, "user-magiff")
        self.assertEqual(context.managed_roster_id, 6)
        self.assertEqual(context.managed_roster.starters[0].slot, "QB")
        self.assertEqual(
            context.managed_roster.starters[0].player.display_name,
            "Quarterback One",
        )
        self.assertIsNone(context.managed_roster.starters[2].player)
        self.assertEqual(context.managed_roster.points_for, 120.35)
        self.assertEqual(context.matchup.opponent_name, "Opponent")
        self.assertEqual(context.transactions[0].waiver_bid, 12)
        self.assertEqual(context.trending_adds[0].player.display_name, "Available Player")
        self.assertEqual(
            [item.display_name for item in context.available_candidates],
            ["Available Player", "Second Available"],
        )
        payload = context.agent_payload()
        self.assertEqual(payload["managed_team"]["waiver_budget_remaining"], 83)
        self.assertEqual(payload["ecr"]["source_roster_assumption"], "3 starting WR slots")

    def test_build_rejects_user_outside_league(self):
        snapshot = _snapshot()
        snapshot["users"] = [snapshot["users"][0]]
        with self.assertRaisesRegex(ValueError, "not a member"):
            LeagueContextBuilder(
                sleeper=FakeSleeperLeagueSource(snapshot),
                players=FakeLeaguePlayerRepository(_profiles()),
                market=FakeMarketRepository([]),
            ).build(
                league_id="league-1",
                user_reference="Magiff",
                available_limit=10,
            )


class SleeperLeagueClientTests(unittest.TestCase):
    def test_snapshot_fetches_public_league_resources(self):
        requested = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(request.url)
            path = request.url.path
            if path == "/v1/user/Magiff":
                return httpx.Response(200, json={"user_id": "user-magiff"})
            if path == "/v1/league/league-1":
                return httpx.Response(200, json={"league_id": "league-1"})
            if path == "/v1/league/league-1/users":
                return httpx.Response(200, json=[])
            if path == "/v1/league/league-1/rosters":
                return httpx.Response(200, json=[])
            if path == "/v1/state/nfl":
                return httpx.Response(200, json={"week": 3})
            if path == "/v1/league/league-1/matchups/3":
                return httpx.Response(200, json=[])
            if path == "/v1/league/league-1/transactions/3":
                return httpx.Response(200, json=[])
            if path.endswith("/trending/add") or path.endswith("/trending/drop"):
                return httpx.Response(200, json=[])
            return httpx.Response(404)

        client = SleeperLeagueClient(
            client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        snapshot = client.snapshot(
            league_id="league-1",
            user_reference="Magiff",
            trending_lookback_hours=12,
            trending_limit=15,
        )

        self.assertEqual(snapshot["week"], 3)
        self.assertEqual(len(requested), 9)
        trending = [url for url in requested if "/trending/" in url.path]
        self.assertEqual(len(trending), 2)
        self.assertTrue(all(url.params["lookback_hours"] == "12" for url in trending))
        self.assertTrue(all(url.params["limit"] == "15" for url in trending))
        self.assertTrue(all("magiff_refresh" in url.params for url in trending))


if __name__ == "__main__":
    unittest.main()
