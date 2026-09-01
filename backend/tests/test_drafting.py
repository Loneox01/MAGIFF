import json
import unittest
from types import SimpleNamespace

import httpx

from drafting.agent import DraftAgentService
from drafting.board import (
    DraftContextBuilder,
    pick_coordinates,
    snake_pick_number,
)
from drafting.models import DraftCandidate, DraftPick
from integrations.sleeper import SleeperDraftClient
from tools.base import ToolExecutionResult


def _candidates(count: int = 100) -> list[DraftCandidate]:
    positions = ("WR", "RB", "QB", "TE")
    return [
        DraftCandidate(
            player_id=f"internal-{index}",
            external_id=f"sleeper-{index}",
            display_name=f"Player {index}",
            position=positions[(index - 1) % len(positions)],
            team=f"T{index % 32:02d}",
            overall_rank=float(index),
            position_rank=(index - 1) // 4 + 1,
            best_rank=float(max(1, index - 2)),
            worst_rank=float(index + 3),
            rank_sd=2.1,
        )
        for index in range(1, count + 1)
    ]


class FakeBoardRepository:
    def __init__(self, candidates=None):
        self.candidates = candidates or _candidates()

    def load_candidates(self, **_kwargs):
        return "2026-08-30", list(self.candidates), "FantasyPros", "consensus-cheatsheets"


def _usage(input_tokens=20, output_tokens=5, cached_tokens=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


class DraftBoardTests(unittest.TestCase):
    def test_snake_math_reverses_even_rounds(self):
        self.assertEqual(snake_pick_number(1, 10, 12), 10)
        self.assertEqual(snake_pick_number(2, 10, 12), 15)
        self.assertEqual(pick_coordinates(15, 12), (2, 10))

    def test_build_excludes_picks_and_derives_roster_state(self):
        picks = [
            DraftPick(1, 1, 1, 1, "sleeper-1"),
            DraftPick(2, 1, 2, 2, "sleeper-2"),
        ]
        context = DraftContextBuilder(FakeBoardRepository()).build(
            season=2026,
            scoring_format="ppr",
            league_format="redraft_1qb",
            teams=4,
            rounds=10,
            draft_slot=3,
            roster_id=3,
            picks=picks,
        )

        self.assertTrue(context.on_clock)
        self.assertEqual(context.current_pick, 3)
        self.assertEqual(context.picks_until_turn, 0)
        available_ids = {
            candidate.external_id for candidate in context.available_candidates
        }
        self.assertNotIn("sleeper-1", available_ids)
        self.assertNotIn("sleeper-2", available_ids)
        self.assertEqual(context.roster_counts["RB"], 0)

    def test_simulation_stops_immediately_before_selected_turn(self):
        context = DraftContextBuilder(FakeBoardRepository()).simulate(
            season=2026,
            scoring_format="ppr",
            league_format="redraft_1qb",
            teams=12,
            rounds=16,
            draft_slot=10,
            target_round=5,
        )

        self.assertEqual(context.current_pick, 58)
        self.assertEqual(context.current_round, 5)
        self.assertTrue(context.on_clock)
        self.assertEqual(len(context.my_roster), 4)
        self.assertEqual(
            [pick.pick_no for pick in context.my_roster],
            [10, 15, 34, 39],
        )


class SleeperDraftClientTests(unittest.TestCase):
    def test_snapshot_normalizes_user_slot_and_pick_metadata(self):
        requested_urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_urls.append(request.url)
            if request.url.path.endswith("/picks"):
                return httpx.Response(
                    200,
                    json=[
                        {
                            "pick_no": 1,
                            "round": 1,
                            "draft_slot": 1,
                            "roster_id": 7,
                            "player_id": "4046",
                            "metadata": {
                                "first_name": "Josh",
                                "last_name": "Allen",
                                "position": "QB",
                                "team": "BUF",
                            },
                        }
                    ],
                )
            return httpx.Response(
                200,
                json={
                    "draft_order": {"user-1": 1},
                    "slot_to_roster_id": {"1": 7},
                    "settings": {"teams": 10, "rounds": 15},
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = SleeperDraftClient(client=http_client)
        draft, picks, slot, roster_id = client.snapshot(
            draft_id="draft-1",
            user_id="user-1",
        )

        self.assertEqual(draft["settings"]["teams"], 10)
        self.assertEqual((slot, roster_id), (1, 7))
        self.assertEqual(picks[0].display_name, "Josh Allen")
        self.assertEqual(picks[0].external_player_id, "4046")
        self.assertEqual(len(requested_urls), 2)
        self.assertTrue(
            all("magiff_refresh" in url.params for url in requested_urls)
        )

    def test_mock_pick_uses_draft_slot_when_roster_id_is_null(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/picks"):
                return httpx.Response(
                    200,
                    json=[
                        {
                            "pick_no": 7,
                            "round": 1,
                            "draft_slot": 7,
                            "roster_id": None,
                            "player_id": "9493",
                            "metadata": {
                                "first_name": "Puka",
                                "last_name": "Nacua",
                                "position": "WR",
                                "team": "LAR",
                            },
                        }
                    ],
                )
            return httpx.Response(
                200,
                json={
                    "draft_order": {},
                    "slot_to_roster_id": {},
                    "settings": {"teams": 10, "rounds": 15},
                },
            )

        client = SleeperDraftClient(
            client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        _, picks, slot, roster_id = client.snapshot(
            draft_id="mock-1",
            draft_slot=7,
        )

        self.assertEqual((slot, roster_id), (7, 7))
        self.assertEqual(picks[0].roster_id, 7)


class DraftAgentTests(unittest.TestCase):
    def test_report_only_loop_returns_answer_and_telemetry(self):
        tool_call = SimpleNamespace(
            type="function_call",
            name="search_reports",
            call_id="call-1",
            arguments=json.dumps(
                {"query": "Player 58 current injury status", "limit": 3}
            ),
        )
        responses = iter(
            [
                SimpleNamespace(
                    output=[tool_call],
                    output_text="",
                    usage=_usage(100, 20, 40),
                ),
                SimpleNamespace(
                    output=[],
                    output_text="Draft Player 58. Backups: Player 59, Player 60.",
                    usage=_usage(140, 30, 60),
                ),
            ]
        )
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return next(responses)

        report_queries = []

        def report_search(query, limit, source_question=None):
            report_queries.append((query, limit, source_question))
            return ToolExecutionResult(
                output={"status": "partial", "reports": []},
                input_tokens=30,
                cached_input_tokens=10,
                output_tokens=8,
                estimated_cost_usd=0.001,
                details={"status": "partial"},
            )

        context = DraftContextBuilder(FakeBoardRepository()).simulate(
            season=2026,
            scoring_format="ppr",
            league_format="redraft_1qb",
            teams=12,
            rounds=16,
            draft_slot=10,
            target_round=5,
        )
        service = DraftAgentService(
            client=SimpleNamespace(
                responses=SimpleNamespace(create=create)
            ),
            report_search=report_search,
            model="gpt-5.6-terra",
        )

        result = service.run(context, "Who should I draft?")

        self.assertEqual(result.tool_rounds, 1)
        self.assertIn("Draft Player 58", result.answer)
        self.assertEqual(len(report_queries), 1)
        self.assertEqual(report_queries[0][2], "Who should I draft?")
        self.assertEqual(requests[0]["tools"][0]["name"], "search_reports")
        self.assertNotIn("web_search", str(requests[0]["tools"]))
        self.assertEqual(result.usage.input_tokens, 270)
        self.assertEqual(result.usage.cached_input_tokens, 110)
        self.assertEqual(result.usage.output_tokens, 58)


if __name__ == "__main__":
    unittest.main()
