import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag.documents import parse_report
from rag.executor import QueryPlanExecutor
from pydantic import ValidationError
import polars as pl

from processing.normalization.team_codes import normalize_team_codes
from rag.planner import (
    NumericFilter,
    PlayerResolutionBasis,
    PlayerSelector,
    PositionFilter,
    PositionGroupFilter,
    QueryPlan,
    QueryPlanner,
    TeamCodeFilter,
    TeamSelector,
)
from rag.resolver import (
    EntityResolver,
    ResolutionResult,
    ResolvedEntity,
    SelectorResolution,
)
from rag.store import LocalRAGStore
from rag.router import (
    EscalationRouter,
    PlayerIdentityDecision,
    PlayerIdentityResponse,
)


REPORT = """---
id: "alpha-update"
title: "Alpha returns to practice"
source: "Test Wire"
url: "https://example.com/alpha"
author: null
published_at: "2026-08-09"
fetched_at: "2026-08-09"
players: ["Alpha Runner"]
teams: ["TST"]
season: 2026
document_type: "injury_update"
storyline: "alpha_health"
content_mode: "source_summary"
---

# Summary

Alpha Runner returned to practice after recovering from a hamstring injury.

# Key facts

- Alpha was a full participant.

# Source note

Boilerplate that should not be indexed.
"""


class RagTests(unittest.TestCase):
    def test_team_code_normalization(self) -> None:
        frame = pl.DataFrame(
            {
                "latest_team": ["AZ", "SEA", None],
                "opponent_team": ["SF", "AZ", "NYJ"],
                "display_name": ["One", "Two", "Three"],
            }
        )

        normalized = normalize_team_codes(frame)

        self.assertEqual(
            normalized.get_column("latest_team").to_list(),
            ["ARI", "SEA", None],
        )
        self.assertEqual(
            normalized.get_column("opponent_team").to_list(),
            ["SF", "ARI", "NYJ"],
        )

    def test_parse_and_keyword_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            document = parse_report(report_path)

            self.assertEqual(document.players, ("Alpha Runner",))
            self.assertNotIn("Boilerplate", document.body)

            store = LocalRAGStore(root / "index.sqlite3")
            result = store.build_index([document])
            hits = store.keyword_search("hamstring full participant")

            self.assertEqual(result.document_count, 1)
            self.assertEqual(result.embedded_count, 0)
            self.assertEqual(hits[0].document.id, "alpha-update")

    def test_cosine_similarity(self) -> None:
        score = LocalRAGStore._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        opposite = LocalRAGStore._cosine_similarity([1.0, 0.0], [-1.0, 0.0])

        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(opposite, -1.0)

    def test_vector_search_reuses_cached_query_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            document = parse_report(report_path)
            store = LocalRAGStore(root / "index.sqlite3")

            with patch(
                "rag.store.embed_texts",
                side_effect=[[[1.0, 0.0]], [[1.0, 0.0]]],
            ) as mocked_embed:
                result = store.build_index([document], with_embeddings=True)
                first_hits = store.vector_search("healthy runner")
                second_hits = store.vector_search("healthy runner")

            self.assertEqual(result.generated_embedding_count, 1)
            self.assertEqual(first_hits[0].document.id, "alpha-update")
            self.assertEqual(second_hits[0].document.id, "alpha-update")
            self.assertEqual(mocked_embed.call_count, 2)

    def test_hybrid_search_accepts_separate_planned_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            document = parse_report(report_path)
            store = LocalRAGStore(root / "index.sqlite3")

            with patch(
                "rag.store.embed_texts",
                side_effect=[[[1.0, 0.0]], [[1.0, 0.0]]],
            ) as mocked_embed:
                store.build_index([document], with_embeddings=True)
                hits = store.hybrid_search(
                    "words absent from the report",
                    keyword_query="hamstring participant",
                    vector_query="healthy runner returning to practice",
                )

            self.assertEqual(hits[0].document.id, "alpha-update")
            self.assertEqual(
                mocked_embed.call_args_list[1].args[0],
                ["healthy runner returning to practice"],
            )

    def test_query_planner_caches_structured_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = QueryPlan(
                semantic_query="current Alpha Runner hamstring availability",
                keyword_query="Alpha Runner hamstring practice",
                intent="current_status",
                player_mentions=["Alpha Runner"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[
                    PlayerSelector(
                        entity_type="player",
                        reference_text="Alpha Runner",
                        names=["Alpha Runner"],
                        identity_confidence=1.0,
                        resolution_basis=PlayerResolutionBasis.EXACT_NAME,
                        filters=[],
                        semantic_qualifiers=[],
                    )
                ],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="multiple_documents",
            )
            response = SimpleNamespace(
                output_parsed=plan,
                usage=SimpleNamespace(input_tokens=50, output_tokens=20),
            )
            client = SimpleNamespace(
                responses=SimpleNamespace(parse=lambda **_: response)
            )
            planner = QueryPlanner(
                index_path=Path(directory) / "index.sqlite3",
                model="test-planner",
                client=client,
            )

            first = planner.plan(
                "Is Alpha healthy?",
                planning_date=date(2026, 8, 10),
            )

            client.responses.parse = lambda **_: self.fail(
                "Cached plan should avoid a second model call"
            )
            second = planner.plan(
                "Is Alpha healthy?",
                planning_date=date(2026, 8, 10),
            )

            self.assertFalse(first.cached)
            self.assertEqual(first.input_tokens, 50)
            self.assertTrue(second.cached)
            self.assertEqual(second.input_tokens, 0)
            self.assertEqual(second.plan, plan)

    def test_entity_resolver_normalizes_team_and_resolves_attributes(self) -> None:
        class FakeRepository:
            def __init__(self) -> None:
                self.selector = None

            def list_teams(self) -> list[dict]:
                return [
                    {
                        "team_abbr": "PHI",
                        "team_id": "3700",
                        "team_name": "Philadelphia Eagles",
                        "team_nick": "Eagles",
                        "team_conf": "NFC",
                        "team_division": "NFC East",
                    }
                ]

            def resolve_player_selector(self, selector, *, season, week):
                self.selector = selector
                return (
                    [
                        {
                            "player_id": "makai-id",
                            "display_name": "Makai Lemon",
                            "position": "WR",
                            "player_status": {
                                "latest_team": "PHI",
                                "status": "ACT",
                                "years_of_experience": 0,
                            },
                        }
                    ],
                    [],
                    False,
                )

        selector = PlayerSelector(
            entity_type="player",
            reference_text="the Eagles rookie receiver",
            names=[],
            identity_confidence=0.0,
            resolution_basis=PlayerResolutionBasis.NOT_APPLICABLE,
            filters=[
                TeamCodeFilter(field="team", operator="eq", values=["PHI"]),
                PositionFilter(field="position", operator="eq", values=["WR"]),
                NumericFilter(
                    field="rookie_season", operator="eq", values=["2026"]
                ),
            ],
            semantic_qualifiers=["injury outlook"],
        )
        plan = QueryPlan(
            semantic_query="current injury outlook for the Eagles rookie receiver",
            keyword_query="Eagles rookie WR injury",
            intent="current_status",
            player_mentions=[],
            team_mentions=["PHI"],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date="2026-08-10",
            needs_baseline=False,
            evidence_strategy="single_document",
        )
        repository = FakeRepository()

        result = EntityResolver(repository=repository).resolve(plan)

        self.assertEqual(result.players[0].display_name, "Makai Lemon")
        team_filter = next(
            item for item in repository.selector.filters if item.field == "team"
        )
        self.assertEqual(team_filter.values, ["PHI"])
        with tempfile.TemporaryDirectory() as directory:
            routing = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(
                        parse=lambda **_: self.fail(
                            "Player-group selectors must not call Sol"
                        )
                    )
                ),
            ).route(
                "How is the Eagles rookie receiver looking?",
                plan,
                result,
                resolver=EntityResolver(repository=repository),
            )
            self.assertFalse(routing.event.triggered)
            self.assertEqual(routing.event.signals, ())

    def test_executor_links_ids_and_prioritizes_current_report(self) -> None:
        older_report = REPORT.replace(
            'id: "alpha-update"', 'id: "alpha-older"'
        ).replace(
            'published_at: "2026-08-09"', 'published_at: "2026-06-01"'
        ).replace(
            "returned to practice", "began offseason rehabilitation"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            newest_path = root / "newest.md"
            older_path = root / "older.md"
            newest_path.write_text(REPORT, encoding="utf-8")
            older_path.write_text(older_report, encoding="utf-8")
            store = LocalRAGStore(root / "index.sqlite3")
            store.build_index([parse_report(newest_path), parse_report(older_path)])

            selector = PlayerSelector(
                entity_type="player",
                reference_text="Alpha Runner",
                names=["Alpha Runner"],
                identity_confidence=1.0,
                resolution_basis=PlayerResolutionBasis.EXACT_NAME,
                filters=[],
                semantic_qualifiers=["health"],
            )
            entity = ResolvedEntity(
                entity_type="player",
                entity_id="alpha-player-id",
                display_name="Alpha Runner",
                team="TST",
                position="RB",
            )
            resolution = ResolutionResult(
                selectors=[
                    SelectorResolution(
                        selector_index=0,
                        selector=selector,
                        status="resolved",
                        matches=[entity],
                        unresolved_filters=[],
                        semantic_qualifiers=["health"],
                        truncated=False,
                    )
                ]
            )
            plan = QueryPlan(
                semantic_query="current Alpha Runner hamstring availability",
                keyword_query="Alpha Runner hamstring practice",
                intent="current_status",
                player_mentions=["Alpha Runner"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[selector],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date="2026-08-10",
                needs_baseline=False,
                evidence_strategy="multiple_documents",
            )

            result = QueryPlanExecutor(store).execute(
                "Is Alpha healthy?",
                plan,
                resolution,
                mode="keyword",
                limit=2,
                embedding_model="unused",
            )

            self.assertEqual(result.hits[0].document.id, "alpha-update")
            self.assertTrue(
                all(
                    hit.document.player_ids == ("alpha-player-id",)
                    for hit in result.hits
                )
            )

    def test_finite_filter_values_are_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            PositionFilter(
                field="position",
                operator="eq",
                values=["HEAD_COACH"],
            )

        with self.assertRaises(ValidationError):
            TeamCodeFilter(
                field="team",
                operator="eq",
                values=["Seattle"],
            )

    def test_team_selector_rejects_player_filters(self) -> None:
        with self.assertRaises(ValidationError):
            TeamSelector(
                entity_type="team",
                names=["SEA"],
                filters=[
                    PositionGroupFilter(
                        field="position_group",
                        operator="eq",
                        values=["RB"],
                    )
                ],
                semantic_qualifiers=["running back competition"],
            )

    def test_identity_router_corrects_low_confidence_alias_and_caches_sol(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                players = {
                    "Stefon Diggs": {
                        "player_id": "stefon-id",
                        "display_name": "Stefon Diggs",
                        "position": "WR",
                        "player_status": {"latest_team": "NE"},
                    },
                    "Amon-Ra St. Brown": {
                        "player_id": "arsb-id",
                        "display_name": "Amon-Ra St. Brown",
                        "position": "WR",
                        "player_status": {"latest_team": "DET"},
                    },
                }
                rows = [players[name] for name in selector.names if name in players]
                return rows, [], False

        selector = PlayerSelector(
            entity_type="player",
            reference_text="the Sun God",
            names=["Stefon Diggs"],
            identity_confidence=0.40,
            resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
            filters=[],
            semantic_qualifiers=["training-camp performance"],
        )
        plan = QueryPlan(
            semantic_query="Recent camp reports about Stefon Diggs",
            keyword_query="Stefon Diggs training camp",
            intent="current_status",
            player_mentions=["Stefon Diggs"],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="single_document",
        )
        response = SimpleNamespace(
            output_parsed=PlayerIdentityResponse(
                decisions=[
                    PlayerIdentityDecision(
                        selector_index=0,
                        status="resolved",
                        canonical_name="Amon-Ra St. Brown",
                        player_id=None,
                        alternatives=[],
                    )
                ]
            ),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(parse=lambda **_: response)
        )
        resolver = EntityResolver(repository=FakeRepository())
        initial = resolver.resolve(plan)

        with tempfile.TemporaryDirectory() as directory:
            router = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=client,
            )
            first = router.route(
                "How is the Sun God looking in camp?",
                plan,
                initial,
                resolver=resolver,
            )

            self.assertTrue(first.event.triggered)
            self.assertTrue(first.event.api_called)
            self.assertTrue(first.event.impactful)
            self.assertEqual(
                first.event.signals[0].reasons,
                ("player_identity_low_confidence",),
            )
            self.assertEqual(
                first.event.signals[0].database_status,
                "resolved",
            )
            self.assertEqual(first.plan.player_mentions, ["Amon-Ra St. Brown"])
            self.assertEqual(first.resolution.players[0].entity_id, "arsb-id")
            self.assertAlmostEqual(first.event.estimated_cost_usd, 0.0011)

            client.responses.parse = lambda **_: self.fail(
                "Cached identity decision should avoid a second Sol call"
            )
            second = router.route(
                "How is the Sun God looking in camp?",
                plan,
                initial,
                resolver=resolver,
            )
            stats = router.stats()

            self.assertTrue(second.event.cache_hit)
            self.assertFalse(second.event.api_called)
            self.assertEqual(stats["searches_evaluated"], 2)
            self.assertEqual(stats["api_calls"], 1)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["impactful_routes"], 2)
            self.assertEqual(
                stats["routing_signals_by_basis"]["known_alias"]["escalated"],
                2,
            )

    def test_identity_router_skips_grounded_alias_and_cleans_mentions(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                if "Christian McCaffrey" not in selector.names:
                    return [], [], False
                return (
                    [
                        {
                            "player_id": "cmc-id",
                            "display_name": "Christian McCaffrey",
                            "position": "RB",
                            "player_status": {"latest_team": "SF"},
                        }
                    ],
                    [],
                    False,
                )

        selector = PlayerSelector(
            entity_type="player",
            reference_text="CMC",
            names=["Christian McCaffrey"],
            identity_confidence=0.96,
            resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
            filters=[],
            semantic_qualifiers=[],
        )
        plan = QueryPlan(
            semantic_query="latest Christian McCaffrey news",
            keyword_query="Christian McCaffrey latest",
            intent="current_status",
            player_mentions=["CMC"],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="latest",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="single_document",
        )
        resolver = EntityResolver(repository=FakeRepository())
        initial = resolver.resolve(plan)

        with tempfile.TemporaryDirectory() as directory:
            router = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(
                        parse=lambda **_: self.fail("Sol should not be called")
                    )
                ),
            )
            result = router.route(
                "What is the latest Christian McCaffrey news?",
                plan,
                initial,
                resolver=resolver,
            )

            self.assertFalse(result.event.triggered)
            self.assertFalse(result.event.api_called)
            self.assertFalse(result.event.impactful)
            self.assertTrue(result.event.plan_changed)
            self.assertEqual(result.plan.player_mentions, ["Christian McCaffrey"])
            self.assertEqual(len(result.resolution.selectors), 1)
            self.assertEqual(router.stats()["escalations_triggered"], 0)
            self.assertEqual(router.stats()["impactful_routes"], 0)

    def test_contextual_alias_at_threshold_does_not_escalate(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                return (
                    [
                        {
                            "player_id": "smith-id",
                            "display_name": "DeVonta Smith",
                            "position": "WR",
                            "player_status": {"latest_team": "PHI"},
                        }
                    ],
                    [],
                    False,
                )

        selector = PlayerSelector(
            entity_type="player",
            reference_text="Smitty",
            names=["DeVonta Smith"],
            identity_confidence=0.70,
            resolution_basis=PlayerResolutionBasis.CONTEXTUAL_ALIAS,
            filters=[],
            semantic_qualifiers=[],
        )
        plan = QueryPlan(
            semantic_query="latest DeVonta Smith news",
            keyword_query="DeVonta Smith latest",
            intent="current_status",
            player_mentions=["DeVonta Smith"],
            team_mentions=["PHI"],
            negative_focus=[],
            entity_selectors=[selector],
            season=None,
            week=None,
            temporal_mode="latest",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="single_document",
        )
        resolver = EntityResolver(repository=FakeRepository())
        initial = resolver.resolve(plan)

        with tempfile.TemporaryDirectory() as directory:
            result = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(
                        parse=lambda **_: self.fail("Sol should not be called")
                    )
                ),
            ).route(
                "How is Philly's Smitty looking?",
                plan,
                initial,
                resolver=resolver,
            )

        self.assertFalse(result.event.triggered)
        self.assertEqual(result.event.signals[0].reasons, ())

    def test_duplicate_names_are_sent_with_context_and_selected_by_id(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                if selector.names != ["Justin Jefferson"]:
                    return [], [], False
                return (
                    [
                        {
                            "player_id": "vikings-wr-id",
                            "display_name": "Justin Jefferson",
                            "position": "WR",
                            "position_group": "WR",
                            "rookie_season": 2020,
                            "draft_year": 2020,
                            "player_status": {
                                "latest_team": "MIN",
                                "jersey_number": "18",
                                "status": "ACT",
                            },
                        },
                        {
                            "player_id": "browns-lb-id",
                            "display_name": "Justin Jefferson",
                            "position": "LB",
                            "position_group": "LB",
                            "rookie_season": 2026,
                            "draft_year": None,
                            "player_status": {
                                "latest_team": "CLE",
                                "jersey_number": "17",
                                "status": "ACT",
                            },
                        },
                    ],
                    [],
                    False,
                )

        selector = PlayerSelector(
            entity_type="player",
            reference_text="JJettas",
            names=["Justin Jefferson"],
            identity_confidence=0.98,
            resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
            filters=[],
            semantic_qualifiers=["training-camp performance"],
        )
        plan = QueryPlan(
            semantic_query="Justin Jefferson camp outlook",
            keyword_query="Justin Jefferson camp",
            intent="current_status",
            player_mentions=["Justin Jefferson"],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="single_document",
        )
        response = SimpleNamespace(
            output_parsed=PlayerIdentityResponse(
                decisions=[
                    PlayerIdentityDecision(
                        selector_index=0,
                        status="resolved",
                        canonical_name="Justin Jefferson",
                        player_id="vikings-wr-id",
                        alternatives=[],
                    )
                ]
            ),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
        captured = {}

        def parse(**kwargs):
            captured.update(kwargs)
            return response

        resolver = EntityResolver(repository=FakeRepository())
        initial = resolver.resolve(plan)
        with tempfile.TemporaryDirectory() as directory:
            result = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=parse)
                ),
            ).route(
                "How is JJettas looking in camp?",
                plan,
                initial,
                resolver=resolver,
            )

        request = json.loads(captured["input"][1]["content"])
        candidates = request["references"][0]["database_matches"]
        self.assertEqual(
            {candidate["player_id"] for candidate in candidates},
            {"vikings-wr-id", "browns-lb-id"},
        )
        self.assertEqual(candidates[0]["team"], "MIN")
        self.assertEqual(candidates[0]["position"], "WR")
        self.assertEqual(candidates[0]["jersey_number"], "18")
        self.assertEqual(result.resolution.selectors[0].status, "resolved")
        self.assertEqual(result.resolution.players[0].entity_id, "vikings-wr-id")
        self.assertEqual(result.event.decisions[0].player_id, "vikings-wr-id")
        self.assertTrue(result.event.decisions[0].grounded)

    def test_large_fuzzy_candidate_set_is_omitted_from_sol_payload(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                if selector.names == ["DeMario Douglas"]:
                    return (
                        [
                            {
                                "player_id": "demario-id",
                                "display_name": "DeMario Douglas",
                                "position": "WR",
                                "player_status": {"latest_team": "NE"},
                            }
                        ],
                        [],
                        False,
                    )
                if selector.names == ["Douglas"]:
                    return (
                        [
                            {
                                "player_id": f"douglas-{index}",
                                "display_name": (
                                    "DeMario Douglas"
                                    if index == 0
                                    else f"Player Douglas {index}"
                                ),
                                "position": "WR",
                                "player_status": {"latest_team": "NE"},
                            }
                            for index in range(9)
                        ],
                        [],
                        False,
                    )
                return [], [], False

        selector = PlayerSelector(
            entity_type="player",
            reference_text="Pop Douglas",
            names=["Douglas"],
            identity_confidence=0.90,
            resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
            filters=[],
            semantic_qualifiers=[],
        )
        plan = QueryPlan(
            semantic_query="latest Pop Douglas news",
            keyword_query="Pop Douglas latest",
            intent="current_status",
            player_mentions=["Douglas"],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=None,
            week=None,
            temporal_mode="latest",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="single_document",
        )
        response = SimpleNamespace(
            output_parsed=PlayerIdentityResponse(
                decisions=[
                    PlayerIdentityDecision(
                        selector_index=0,
                        status="resolved",
                        canonical_name="DeMario Douglas",
                        player_id=None,
                        alternatives=[],
                    )
                ]
            ),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
        captured = {}

        def parse(**kwargs):
            captured.update(kwargs)
            return response

        resolver = EntityResolver(repository=FakeRepository())
        initial = resolver.resolve(plan)
        with tempfile.TemporaryDirectory() as directory:
            result = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=parse)
                ),
            ).route(
                "What is the latest news about Pop Douglas?",
                plan,
                initial,
                resolver=resolver,
            )

        request = json.loads(captured["input"][1]["content"])
        issue = request["references"][0]
        self.assertEqual(issue["database_match_count"], 9)
        self.assertTrue(issue["database_matches_omitted"])
        self.assertEqual(issue["database_matches"], [])
        self.assertEqual(result.resolution.players[0].entity_id, "demario-id")
        self.assertTrue(result.event.decisions[0].grounded)

    def test_multiple_aliases_receive_independent_decisions_in_one_call(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                players = {
                    "Kenneth Walker III": {
                        "player_id": "walker-id",
                        "display_name": "Kenneth Walker III",
                        "position": "RB",
                        "player_status": {"latest_team": "SEA"},
                    },
                    "George Pickens": {
                        "player_id": "pickens-id",
                        "display_name": "George Pickens",
                        "position": "WR",
                        "player_status": {"latest_team": "DAL"},
                    },
                }
                return (
                    [players[name] for name in selector.names if name in players],
                    [],
                    False,
                )

        selectors = [
            PlayerSelector(
                entity_type="player",
                reference_text="K9",
                names=["K9"],
                identity_confidence=0.20,
                resolution_basis=PlayerResolutionBasis.INFERRED,
                filters=[],
                semantic_qualifiers=[],
            ),
            PlayerSelector(
                entity_type="player",
                reference_text="GP3",
                names=["GP3"],
                identity_confidence=0.35,
                resolution_basis=PlayerResolutionBasis.INFERRED,
                filters=[],
                semantic_qualifiers=[],
            ),
        ]
        plan = QueryPlan(
            semantic_query="latest K9 and GP3 news",
            keyword_query="K9 GP3 latest",
            intent="comparison",
            player_mentions=["K9", "GP3"],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=selectors,
            season=None,
            week=None,
            temporal_mode="latest",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="per_entity",
        )
        response = SimpleNamespace(
            output_parsed=PlayerIdentityResponse(
                decisions=[
                    PlayerIdentityDecision(
                        selector_index=0,
                        status="resolved",
                        canonical_name="Kenneth Walker III",
                        player_id=None,
                        alternatives=[],
                    ),
                    PlayerIdentityDecision(
                        selector_index=1,
                        status="resolved",
                        canonical_name="George Pickens",
                        player_id=None,
                        alternatives=[],
                    ),
                ]
            ),
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=30,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
        captured = {}

        def parse(**kwargs):
            captured.update(kwargs)
            return response

        resolver = EntityResolver(repository=FakeRepository())
        initial = resolver.resolve(plan)
        with tempfile.TemporaryDirectory() as directory:
            result = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=parse)
                ),
            ).route(
                "What is the latest news about K9 and GP3?",
                plan,
                initial,
                resolver=resolver,
            )

        request = json.loads(captured["input"][1]["content"])
        self.assertEqual(len(request["references"]), 2)
        self.assertEqual(len(result.event.decisions), 2)
        self.assertEqual(
            {player.entity_id for player in result.resolution.players},
            {"walker-id", "pickens-id"},
        )

    def test_database_mismatch_escalates_with_full_luna_signal(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                if selector.names == ["Jaxon Smith-Njigba"]:
                    return (
                        [
                            {
                                "player_id": "jsn-id",
                                "display_name": "Jaxon Smith-Njigba",
                                "position": "WR",
                                "player_status": {"latest_team": "SEA"},
                            }
                        ],
                        [],
                        False,
                    )
                return [], ["candidate had no database match"], False

        selector = PlayerSelector(
            entity_type="player",
            reference_text="JSN",
            names=["John Smith"],
            identity_confidence=0.97,
            resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
            filters=[],
            semantic_qualifiers=["latest news"],
        )
        plan = QueryPlan(
            semantic_query="latest news about John Smith",
            keyword_query="John Smith latest news",
            intent="current_status",
            player_mentions=["John Smith"],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="latest",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="single_document",
        )
        response = SimpleNamespace(
            output_parsed=PlayerIdentityResponse(
                decisions=[
                    PlayerIdentityDecision(
                        selector_index=0,
                        status="resolved",
                        canonical_name="Jaxon Smith-Njigba",
                        player_id=None,
                        alternatives=[],
                    )
                ]
            ),
            usage=SimpleNamespace(
                input_tokens=90,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
        captured = {}

        def parse(**kwargs):
            captured.update(kwargs)
            return response

        resolver = EntityResolver(repository=FakeRepository())
        initial = resolver.resolve(plan)
        with tempfile.TemporaryDirectory() as directory:
            router = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=parse)
                ),
            )
            result = router.route(
                "What is the latest news about JSN?",
                plan,
                initial,
                resolver=resolver,
            )

        request = json.loads(captured["input"][1]["content"])
        routed_signal = request["references"][0]
        self.assertEqual(routed_signal["reference_text"], "JSN")
        self.assertEqual(routed_signal["luna_candidates"], ["John Smith"])
        self.assertEqual(routed_signal["identity_confidence"], 0.97)
        self.assertEqual(routed_signal["resolution_basis"], "known_alias")
        self.assertEqual(routed_signal["database_status"], "unresolved")
        self.assertIn(
            "candidate had no database match",
            routed_signal["database_errors"],
        )
        self.assertTrue(result.event.impactful)
        self.assertEqual(result.resolution.players[0].entity_id, "jsn-id")

    def test_unresolved_explicit_identity_does_not_expand_to_all_players(self) -> None:
        class FailIfQueriedRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                raise AssertionError(
                    "An ungrounded identity must not become a broad query"
                )

        selector = PlayerSelector(
            entity_type="player",
            reference_text="Hollywood",
            names=[],
            identity_confidence=0.20,
            resolution_basis=PlayerResolutionBasis.INFERRED,
            filters=[],
            semantic_qualifiers=["latest news"],
        )
        plan = QueryPlan(
            semantic_query="latest news about Hollywood",
            keyword_query="Hollywood latest news",
            intent="current_status",
            player_mentions=[],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="latest",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="single_document",
        )

        result = EntityResolver(repository=FailIfQueriedRepository()).resolve(plan)

        self.assertEqual(result.selectors[0].status, "unresolved")
        self.assertEqual(result.players, [])
        self.assertIn(
            "player identity could not be grounded",
            result.selectors[0].unresolved_filters,
        )


if __name__ == "__main__":
    unittest.main()
