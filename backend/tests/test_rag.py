import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag.documents import load_reports, parse_report
from rag.retrieval.executor import QueryPlanExecutor
from pydantic import ValidationError
import polars as pl

from processing.normalization.team_codes import normalize_team_codes
from rag.planning.planner import (
    ConferenceFilter,
    ContextRequest,
    DivisionFilter,
    FormationFilter,
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
from rag.planning.enrichment import (
    ContextEnrichment,
    LookupExecution,
    StructuredEnrichment,
    StructuredLookupExecutor,
    TargetEnrichment,
)
from rag.planning.lookups import (
    ContextScopePolicy,
    LookupPurpose,
    PlayerSeasonStatsLookup,
    TeamRosterLookup,
)
from rag.planning.resolver import (
    EntityResolver,
    ContextResolution,
    PLAYER_FIELD_SPECS,
    ResolutionResult,
    ResolutionValidationError,
    ResolvedEntity,
    SelectorResolution,
    SupabaseEntityRepository,
    validate_resolution_bounds,
)
from rag.retrieval.reranker import (
    ConditionAlignment,
    EvidenceRelationship,
    EvidenceSufficiency,
    ReportReranker,
    RerankJudgment,
    RerankResponse,
    TemporalRole,
)
from rag.retrieval.store import LocalRAGStore, SearchHit
from rag.planning.router import (
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
    @staticmethod
    def _context_test_repository():
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return [
                    {
                        "team_abbr": "ATL",
                        "team_id": "0200",
                        "team_name": "Atlanta Falcons",
                        "team_nick": "Falcons",
                        "team_conf": "NFC",
                        "team_division": "NFC South",
                    }
                ]

            def resolve_player_selector(self, selector, *, season, week):
                if selector.names:
                    return (
                        [
                            {
                                "player_id": "london-id",
                                "display_name": "Drake London",
                                "position": "WR",
                                "position_group": "WR",
                                "rookie_season": 2022,
                                "draft_year": 2022,
                                "player_status": {
                                    "latest_team": "ATL",
                                    "jersey_number": "5",
                                    "status": "ACT",
                                },
                            }
                        ],
                        [],
                        False,
                    )
                fields = {item.field for item in selector.hard_filters}
                if fields == {"team", "position_group"}:
                    return (
                        [
                            {
                                "player_id": "tua-id",
                                "display_name": "Tua Tagovailoa",
                                "position": "QB",
                                "position_group": "QB",
                                "rookie_season": 2020,
                                "draft_year": 2020,
                                "player_status": {
                                    "latest_team": "ATL",
                                    "jersey_number": "1",
                                    "status": "ACT",
                                },
                            },
                            {
                                "player_id": "penix-id",
                                "display_name": "Michael Penix Jr.",
                                "position": "QB",
                                "position_group": "QB",
                                "rookie_season": 2024,
                                "draft_year": 2024,
                                "player_status": {
                                    "latest_team": "ATL",
                                    "jersey_number": "9",
                                    "status": "ACT",
                                },
                            },
                        ],
                        [],
                        False,
                    )
                return [], [], False

            def resolve_player_teams(self, player_ids, *, season, week):
                return {player_id: "ATL" for player_id in player_ids}

        return FakeRepository()

    @staticmethod
    def _london_context_plan(*, include_context: bool) -> QueryPlan:
        selector = PlayerSelector(
            entity_type="player",
            reference_text="Drake London",
            names=["Drake London"],
            identity_confidence=1.0,
            resolution_basis=PlayerResolutionBasis.EXACT_NAME,
            hard_filters=[],
            soft_filters=[],
            semantic_qualifiers=["quarterback context"],
        )
        contexts = []
        if include_context:
            contexts.append(
                ContextRequest(
                    anchor_selector_index=0,
                    relation="same_team",
                    semantic_query=(
                        "current quarterback situation affecting receiver outlook"
                    ),
                    keyword_query="quarterback competition injury starter",
                    semantic_qualifiers=["quarterback context"],
                    structured_lookups=[
                        TeamRosterLookup(
                            lookup_id="atl-quarterbacks",
                            purpose=LookupPurpose.EXPAND_CANDIDATES,
                            operation="team_roster",
                            season=2026,
                            week=None,
                            position="QB",
                            status=None,
                        )
                    ],
                )
            )
        return QueryPlan(
            semantic_query="current concerns affecting Drake London",
            keyword_query="Drake London current outlook concerns",
            intent="current_status",
            player_mentions=["Drake London"],
            team_mentions=[],
            soft_team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            context_requests=contexts,
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="multiple_documents",
        )

    def test_context_branch_unions_indirect_same_team_evidence(self) -> None:
        direct = REPORT.replace(
            'id: "alpha-update"', 'id: "london-update"'
        ).replace(
            'title: "Alpha returns to practice"',
            'title: "Drake London role remains stable"',
        ).replace(
            'players: ["Alpha Runner"]', 'players: ["Drake London"]'
        ).replace(
            'teams: ["TST"]', 'teams: ["ATL"]'
        ).replace(
            "Alpha Runner returned to practice after recovering from a hamstring injury.",
            "Drake London remained the primary Atlanta receiver in practice.",
        )
        quarterback = REPORT.replace(
            'id: "alpha-update"', 'id: "falcons-quarterbacks"'
        ).replace(
            'title: "Alpha returns to practice"',
            'title: "Tua and Penix compete for Falcons quarterback job"',
        ).replace(
            'players: ["Alpha Runner"]',
            'players: ["Tua Tagovailoa", "Michael Penix Jr."]',
        ).replace(
            'teams: ["TST"]', 'teams: ["ATL"]'
        ).replace(
            "Alpha Runner returned to practice after recovering from a hamstring injury.",
            "Tua Tagovailoa and Michael Penix Jr. split starting quarterback work.",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct_path = root / "london.md"
            quarterback_path = root / "quarterbacks.md"
            direct_path.write_text(direct, encoding="utf-8")
            quarterback_path.write_text(quarterback, encoding="utf-8")
            store = LocalRAGStore(root / "index.sqlite3")
            store.build_index(
                [parse_report(direct_path), parse_report(quarterback_path)]
            )
            plan = self._london_context_plan(include_context=True)
            resolution = EntityResolver(
                repository=self._context_test_repository()
            ).resolve(plan)
            result = QueryPlanExecutor(store).execute(
                "How does Atlanta's quarterback situation affect Drake London?",
                plan,
                resolution,
                mode="keyword",
                limit=5,
                embedding_model="unused",
            )

        ids = {hit.document.id for hit in result.hits}
        self.assertEqual(resolution.contexts[0].teams, ["ATL"])
        self.assertIn("london-update", ids)
        self.assertIn("falcons-quarterbacks", ids)
        quarterback_hit = next(
            hit for hit in result.hits
            if hit.document.id == "falcons-quarterbacks"
        )
        self.assertIn("context:0", quarterback_hit.retrieval_scopes)
        self.assertEqual(result.strategy, "resolved+context")

    def test_direct_question_does_not_expand_to_team_context(self) -> None:
        plan = self._london_context_plan(include_context=False)
        resolution = EntityResolver(
            repository=self._context_test_repository()
        ).resolve(plan)

        self.assertEqual(resolution.contexts, [])

    def test_context_request_requires_bounded_player_group_anchor(self) -> None:
        group = PlayerSelector(
            entity_type="player",
            reference_text="Falcons receivers",
            names=[],
            identity_confidence=0,
            resolution_basis=PlayerResolutionBasis.NOT_APPLICABLE,
            hard_filters=[],
            soft_filters=[],
            semantic_qualifiers=[],
        )
        with self.assertRaises(ValidationError):
            QueryPlan(
                semantic_query="Falcons receiver outlook",
                keyword_query="Falcons receivers",
                intent="current_status",
                player_mentions=[],
                team_mentions=[],
                soft_team_mentions=[],
                negative_focus=[],
                entity_selectors=[group],
                context_requests=[
                    ContextRequest(
                        anchor_selector_index=0,
                        relation="same_team",
                        semantic_query="surrounding team context",
                        keyword_query="team context",
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

        bounded_group = group.model_copy(
            update={
                "hard_filters": [
                    TeamCodeFilter(field="team", operator="eq", values=["ATL"]),
                    PositionGroupFilter(
                        field="position_group",
                        operator="eq",
                        values=["WR"],
                    ),
                ]
            }
        )
        plan = QueryPlan(
            semantic_query="Falcons receiver outlook",
            keyword_query="Falcons receivers",
            intent="current_status",
            player_mentions=[],
            team_mentions=["ATL"],
            soft_team_mentions=[],
            negative_focus=[],
            entity_selectors=[bounded_group],
            context_requests=[
                ContextRequest(
                    anchor_selector_index=0,
                    relation="environment",
                    semantic_query="surrounding team context",
                    keyword_query="team context",
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
        resolution = EntityResolver(
            repository=self._context_test_repository()
        ).resolve(plan)

        self.assertEqual(resolution.contexts[0].status, "resolved")
        self.assertEqual(resolution.contexts[0].teams, ["ATL"])
        self.assertEqual(len(resolution.contexts[0].anchor_entities), 2)

    def test_context_request_resolves_from_team_anchor(self) -> None:
        team = TeamSelector(
            entity_type="team",
            names=["ATL"],
            hard_filters=[],
            soft_filters=[],
            semantic_qualifiers=["offensive environment"],
        )
        plan = QueryPlan(
            semantic_query="Atlanta offense outlook",
            keyword_query="Atlanta offense outlook",
            intent="projection",
            player_mentions=[],
            team_mentions=["ATL"],
            soft_team_mentions=[],
            negative_focus=[],
            entity_selectors=[team],
            context_requests=[
                ContextRequest(
                    anchor_selector_index=0,
                    relation="same_team",
                    semantic_query=(
                        "quarterback availability and passing environment"
                    ),
                    keyword_query="quarterback starter injury competition",
                    semantic_qualifiers=["quarterback context"],
                    structured_lookups=[
                        TeamRosterLookup(
                            lookup_id="atl-quarterbacks",
                            purpose=LookupPurpose.EXPAND_CANDIDATES,
                            operation="team_roster",
                            season=2026,
                            week=None,
                            position="QB",
                            status=None,
                        )
                    ],
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

        resolution = EntityResolver(
            repository=self._context_test_repository()
        ).resolve(plan)

        context = resolution.contexts[0]
        self.assertEqual(context.status, "resolved")
        self.assertEqual(context.teams, ["ATL"])
        self.assertEqual(
            [
                (entity.entity_type, entity.entity_id)
                for entity in context.anchor_entities
            ],
            [("team", "ATL")],
        )
        self.assertEqual(
            context.request.structured_lookups[0].operation,
            "team_roster",
        )

    def test_formation_schema_matches_normalized_depth_chart_side(self) -> None:
        schema = FormationFilter.model_json_schema()

        self.assertEqual(
            PLAYER_FIELD_SPECS["formation"].column,
            "position_group",
        )
        description = schema["properties"]["field"]["description"]
        self.assertIn("Normalized depth-chart side", description)
        self.assertIn("requires QueryPlan.season", description)

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

    def test_load_reports_inherits_base_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports_root = Path(directory)
            base_dir = reports_root / "2026-08-09"
            current_dir = reports_root / "2026-08-13"
            base_dir.mkdir()
            current_dir.mkdir()

            (base_dir / "alpha.md").write_text(REPORT, encoding="utf-8")
            (base_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "document_count": 1,
                        "documents": [{"id": "alpha-update", "path": "alpha.md"}],
                    }
                ),
                encoding="utf-8",
            )

            beta_report = REPORT.replace("alpha-update", "beta-update").replace(
                "Alpha", "Beta"
            )
            (current_dir / "beta.md").write_text(beta_report, encoding="utf-8")
            (current_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "base_snapshot": "2026-08-09",
                        "document_count": 2,
                        "documents": [{"id": "beta-update", "path": "beta.md"}],
                    }
                ),
                encoding="utf-8",
            )

            documents = load_reports(
                snapshot="2026-08-13",
                reports_root=reports_root,
            )

            self.assertEqual(
                [document.id for document in documents],
                ["alpha-update", "beta-update"],
            )

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
                "rag.retrieval.store.embed_texts",
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
                "rag.retrieval.store.embed_texts",
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

    def test_query_planner_pairs_relative_week_with_current_nfl_season(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parsed_plan = QueryPlan(
                semantic_query="A.J. Brown Week 1 matchup",
                keyword_query="A.J. Brown Week 1 matchup",
                intent="yes_no",
                player_mentions=["A.J. Brown"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[
                    PlayerSelector(
                        entity_type="player",
                        reference_text="AJ Brown",
                        names=["A.J. Brown"],
                        identity_confidence=1.0,
                        resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
                        filters=[],
                        semantic_qualifiers=["Week 1 matchup"],
                    )
                ],
                season=None,
                week=1,
                temporal_mode="latest",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="single_document",
            )
            response = SimpleNamespace(
                output_parsed=parsed_plan,
                usage=SimpleNamespace(input_tokens=50, output_tokens=20),
            )
            planner = QueryPlanner(
                index_path=Path(directory) / "index.sqlite3",
                model="test-planner",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=lambda **_: response)
                ),
            )

            result = planner.plan(
                "Does AJ Brown have a good Week 1 matchup?",
                planning_date=date(2026, 8, 20),
                use_cache=False,
            )

            self.assertEqual(result.plan.season, 2026)
            self.assertEqual(result.plan.week, 1)

    def test_query_planner_gives_current_nfl_season_only_to_luna(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = QueryPlan(
                semantic_query="test query",
                keyword_query="test query",
                intent="fact",
                player_mentions=[],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[],
                season=None,
                week=None,
                temporal_mode="none",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="single_document",
            )
            response = SimpleNamespace(
                output_parsed=plan,
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )

            luna_parse = Mock(return_value=response)
            luna_planner = QueryPlanner(
                index_path=Path(directory) / "luna.sqlite3",
                model="gpt-5.6-luna",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=luna_parse)
                ),
            )
            luna_planner.plan(
                "test",
                planning_date=date(2026, 8, 20),
                use_cache=False,
            )

            luna_input = luna_parse.call_args.kwargs["input"][1]["content"]
            self.assertIn("Current date: 2026-08-20", luna_input)
            self.assertIn("Current NFL season: 2026", luna_input)

            terra_parse = Mock(return_value=response)
            terra_planner = QueryPlanner(
                index_path=Path(directory) / "terra.sqlite3",
                model="gpt-5.6-terra",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=terra_parse)
                ),
            )
            terra_planner.plan(
                "test",
                planning_date=date(2026, 8, 20),
                use_cache=False,
            )

            terra_input = terra_parse.call_args.kwargs["input"][1]["content"]
            self.assertIn("Current date: 2026-08-20", terra_input)
            self.assertNotIn("Current NFL season:", terra_input)

    def test_query_planner_retries_schema_incompatible_direct_plan_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selector = PlayerSelector(
                entity_type="player",
                reference_text="AJ Brown",
                names=["A.J. Brown"],
                identity_confidence=1.0,
                resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
                filters=[],
                semantic_qualifiers=["Week 1 flex decision"],
            )
            corrected_plan = QueryPlan(
                semantic_query="A.J. Brown Week 1 flex outlook",
                keyword_query="A.J. Brown Week 1 flex",
                intent="comparison",
                player_mentions=["A.J. Brown"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[selector],
                season=2026,
                week=1,
                temporal_mode="latest",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="per_entity",
            )
            invalid_plan = corrected_plan.model_dump(mode="json")
            invalid_plan["entity_selectors"][0]["structured_lookups"] = [
                {
                    "lookup_id": "week-one-ecr",
                    "purpose": "reranker_context",
                    "operation": "ecr_ranking",
                    "season": 2026,
                    "positions": ["RB", "WR"],
                    "scoring_format": "ppr",
                    "league_format": "redraft_1qb",
                    "snapshot_type": "current",
                    "as_of_date": None,
                    "minimum_overall_rank": None,
                    "maximum_overall_rank": None,
                    "limit": 10,
                }
            ]
            responses = Mock(
                side_effect=[
                    SimpleNamespace(
                        output_parsed=invalid_plan,
                        usage=SimpleNamespace(input_tokens=50, output_tokens=20),
                    ),
                    SimpleNamespace(
                        output_parsed=corrected_plan,
                        usage=SimpleNamespace(input_tokens=55, output_tokens=21),
                    ),
                ]
            )
            planner = QueryPlanner(
                index_path=Path(directory) / "index.sqlite3",
                model="gpt-5.6-luna",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=responses)
                ),
            )

            result = planner.plan(
                "Who should I start Week 1, AJ Brown or ETN?",
                planning_date=date(2026, 8, 20),
                use_cache=False,
            )

            self.assertEqual(responses.call_count, 2)
            self.assertEqual(result.plan, corrected_plan)
            self.assertEqual(result.attempts, 2)
            self.assertTrue(result.retried)
            self.assertIsNotNone(result.retry_reason)
            retry_input = responses.call_args_list[1].kwargs["input"]
            self.assertIn(
                "Correction required",
                retry_input[1]["content"],
            )
            self.assertIn(
                "Specific-player target lookups",
                retry_input[1]["content"],
            )

    def test_query_planner_does_not_apply_issue_specific_scope_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            player_group = PlayerSelector(
                entity_type="player",
                reference_text="running back",
                names=[],
                identity_confidence=0,
                resolution_basis=PlayerResolutionBasis.NOT_APPLICABLE,
                filters=[
                    PositionGroupFilter(
                        field="position_group",
                        operator="eq",
                        values=["RB"],
                    )
                ],
                semantic_qualifiers=["lead the backfield"],
            )
            team_selector = TeamSelector(
                entity_type="team",
                names=["LAC"],
                filters=[],
                semantic_qualifiers=[],
            )
            initial_plan = QueryPlan(
                semantic_query="Identify the Chargers lead running back.",
                keyword_query="Chargers lead running back",
                intent="projection",
                player_mentions=[],
                team_mentions=["LAC"],
                negative_focus=[],
                entity_selectors=[team_selector, player_group],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="single_document",
            )
            responses = Mock(
                return_value=SimpleNamespace(
                    output_parsed=initial_plan,
                    usage=SimpleNamespace(input_tokens=50, output_tokens=20),
                )
            )
            planner = QueryPlanner(
                index_path=Path(directory) / "index.sqlite3",
                model="test-planner",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=responses)
                ),
            )

            result = planner.plan(
                "Which Chargers running back will lead the backfield?",
                planning_date=date(2026, 8, 13),
            )

            self.assertEqual(responses.call_count, 1)
            self.assertEqual(result.input_tokens, 50)
            self.assertEqual(result.output_tokens, 20)
            self.assertEqual(result.plan, initial_plan)

    def test_truncated_player_group_stops_before_retrieval(self) -> None:
        group = PlayerSelector(
            entity_type="player",
            reference_text="running backs",
            names=[],
            identity_confidence=0,
            resolution_basis=PlayerResolutionBasis.NOT_APPLICABLE,
            filters=[
                PositionGroupFilter(
                    field="position_group",
                    operator="eq",
                    values=["RB"],
                )
            ],
            semantic_qualifiers=[],
        )
        resolution = ResolutionResult(
            selectors=[
                SelectorResolution(
                    selector_index=0,
                    selector=group,
                    status="multiple",
                    matches=[],
                    unresolved_filters=[],
                    semantic_qualifiers=[],
                    truncated=True,
                )
            ]
        )

        with self.assertRaises(ResolutionValidationError) as raised:
            validate_resolution_bounds(resolution)

        self.assertEqual(raised.exception.code, "unbounded_player_group")

    def test_current_team_player_group_uses_current_membership(self) -> None:
        class TrackingRepository(SupabaseEntityRepository):
            def __init__(self) -> None:
                self.current_filters = None
                self.base_filters = None
                self.candidate_ids = None

            def _current_team_candidate_ids(self, filters, *, season):
                self.current_filters = filters
                return {"current-rb-id"}, []

            def _advanced_candidate_ids(self, filters, *, season, week):
                return None, []

            def _base_player_rows(self, names, filters, candidate_ids):
                self.base_filters = filters
                self.candidate_ids = candidate_ids
                return [
                    {
                        "player_id": "current-rb-id",
                        "display_name": "Current Runner",
                        "position": "RB",
                        "position_group": "RB",
                        "rookie_season": 2025,
                        "draft_year": 2025,
                        "player_status": {
                            "latest_team": "LAC",
                            "jersey_number": "1",
                            "status": "ACT",
                        },
                    }
                ]

        repository = TrackingRepository()
        selector = PlayerSelector(
            entity_type="player",
            reference_text="Chargers running backs",
            names=[],
            identity_confidence=0,
            resolution_basis=PlayerResolutionBasis.NOT_APPLICABLE,
            filters=[
                TeamCodeFilter(field="team", operator="eq", values=["LAC"]),
                PositionGroupFilter(
                    field="position_group",
                    operator="eq",
                    values=["RB"],
                ),
            ],
            semantic_qualifiers=[],
        )

        rows, unresolved, truncated = repository.resolve_player_selector(
            selector,
            season=None,
            week=None,
        )

        self.assertEqual(unresolved, [])
        self.assertFalse(truncated)
        self.assertEqual(rows[0]["display_name"], "Current Runner")
        self.assertEqual(repository.candidate_ids, {"current-rb-id"})
        self.assertEqual(
            [item.field for item in repository.current_filters],
            ["team"],
        )
        self.assertEqual(
            [item.field for item in repository.base_filters],
            ["position_group"],
        )

    def test_current_division_deduplicates_historical_franchise_codes(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return [
                    {
                        "team_abbr": code,
                        "team_id": code,
                        "team_name": name,
                        "team_nick": name.split()[-1],
                        "team_conf": "AFC",
                        "team_division": "AFC West",
                    }
                    for code, name in [
                        ("DEN", "Denver Broncos"),
                        ("KC", "Kansas City Chiefs"),
                        ("LAC", "Los Angeles Chargers"),
                        ("LV", "Las Vegas Raiders"),
                        ("OAK", "Oakland Raiders"),
                        ("SD", "San Diego Chargers"),
                    ]
                ]

            def resolve_player_selector(self, selector, *, season, week):
                return [], [], False

        selector = TeamSelector(
            entity_type="team",
            names=[],
            filters=[
                ConferenceFilter(
                    field="conference",
                    operator="eq",
                    values=["AFC"],
                ),
                DivisionFilter(
                    field="division",
                    operator="eq",
                    values=["AFC West"],
                ),
            ],
            semantic_qualifiers=[],
        )
        plan = QueryPlan(
            semantic_query="AFC West teams",
            keyword_query="AFC West",
            intent="fact",
            player_mentions=[],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="per_entity",
        )

        result = EntityResolver(repository=FakeRepository()).resolve(plan)

        self.assertEqual(
            [team.entity_id for team in result.teams],
            ["DEN", "KC", "LAC", "LV"],
        )
        self.assertEqual(
            [team.display_name for team in result.teams],
            [
                "Denver Broncos",
                "Kansas City Chiefs",
                "Los Angeles Chargers",
                "Las Vegas Raiders",
            ],
        )

    def test_player_group_division_scope_expands_to_current_teams(self) -> None:
        class FakeRepository:
            def __init__(self) -> None:
                self.selector = None

            def list_teams(self) -> list[dict]:
                return [
                    {
                        "team_abbr": code,
                        "team_id": code,
                        "team_name": name,
                        "team_nick": name.split()[-1],
                        "team_conf": "AFC",
                        "team_division": "AFC West",
                    }
                    for code, name in [
                        ("DEN", "Denver Broncos"),
                        ("KC", "Kansas City Chiefs"),
                        ("LAC", "Los Angeles Chargers"),
                        ("LV", "Las Vegas Raiders"),
                        ("OAK", "Oakland Raiders"),
                        ("SD", "San Diego Chargers"),
                    ]
                ]

            def resolve_player_selector(self, selector, *, season, week):
                self.selector = selector
                return [], [], False

        repository = FakeRepository()
        selector = PlayerSelector(
            entity_type="player",
            reference_text="AFC West running backs",
            names=[],
            identity_confidence=0,
            resolution_basis=PlayerResolutionBasis.NOT_APPLICABLE,
            filters=[
                ConferenceFilter(
                    field="conference",
                    operator="eq",
                    values=["AFC"],
                ),
                DivisionFilter(
                    field="division",
                    operator="eq",
                    values=["AFC West"],
                ),
                PositionGroupFilter(
                    field="position_group",
                    operator="eq",
                    values=["RB"],
                ),
            ],
            semantic_qualifiers=["unresolved workload"],
        )
        plan = QueryPlan(
            semantic_query="unresolved AFC West running back workloads",
            keyword_query="AFC West RB workload",
            intent="current_status",
            player_mentions=[],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="per_entity",
        )

        EntityResolver(repository=repository).resolve(plan)

        filters = {item.field: item for item in repository.selector.filters}
        self.assertEqual(filters["team"].values, ["DEN", "KC", "LAC", "LV"])
        self.assertEqual(filters["position_group"].values, ["RB"])

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

    def test_soft_filters_and_team_mentions_never_constrain_resolution(self) -> None:
        class FakeRepository:
            def __init__(self) -> None:
                self.selector = None

            def list_teams(self) -> list[dict]:
                self.fail_if_called = True
                return []

            def resolve_player_selector(self, selector, *, season, week):
                self.selector = selector
                return (
                    [
                        {
                            "player_id": "tua-id",
                            "display_name": "Tua Tagovailoa",
                            "position": "QB",
                            "position_group": "QB",
                            "player_status": {
                                "latest_team": "ATL",
                                "status": "ACT",
                            },
                        }
                    ],
                    [],
                    False,
                )

        selector = PlayerSelector(
            entity_type="player",
            reference_text="Tua Tagovailoa",
            names=["Tua Tagovailoa"],
            identity_confidence=1.0,
            resolution_basis=PlayerResolutionBasis.EXACT_NAME,
            hard_filters=[],
            soft_filters=[
                TeamCodeFilter(field="team", operator="eq", values=["MIA"])
            ],
            semantic_qualifiers=["current status"],
        )
        plan = QueryPlan(
            semantic_query="latest Tua Tagovailoa status",
            keyword_query="Tua Tagovailoa latest",
            intent="current_status",
            player_mentions=["Tua Tagovailoa"],
            team_mentions=[],
            soft_team_mentions=["MIA"],
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
        repository = FakeRepository()

        resolution = EntityResolver(repository=repository).resolve(plan)

        self.assertEqual(resolution.players[0].entity_id, "tua-id")
        self.assertEqual(resolution.players[0].team, "ATL")
        self.assertEqual(repository.selector.hard_filters, [])
        self.assertEqual(repository.selector.soft_filters[0].values, ["MIA"])
        self.assertEqual(resolution.teams, [])
        self.assertEqual(
            QueryPlanExecutor._resolved_team_filters(resolution),
            [],
        )

    def test_literal_full_name_is_looked_up_with_model_candidate(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return [
                    {
                        "team_abbr": "BUF",
                        "team_id": "BUF",
                        "team_name": "Buffalo Bills",
                        "team_nick": "Bills",
                        "team_conf": "AFC",
                        "team_division": "AFC East",
                    }
                ]

            def resolve_player_selector(self, selector, *, season, week):
                self.lookup_names = selector.names
                self.hard_filters = selector.hard_filters
                rows = []
                if "DJ Moore" in selector.names:
                    rows.append(
                        {
                            "player_id": "buffalo-wr-id",
                            "display_name": "DJ Moore",
                            "position": "WR",
                            "position_group": "WR",
                            "player_status": {
                                "latest_team": "BUF",
                                "status": "ACT",
                            },
                        }
                    )
                # The hard Buffalo relationship excludes the distinct former
                # Carolina defensive back represented by the punctuated name.
                return rows, [], False

        selector = PlayerSelector(
            entity_type="player",
            reference_text="DJ Moore",
            names=["D.J. Moore"],
            identity_confidence=1.0,
            resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
            hard_filters=[
                TeamCodeFilter(field="team", operator="eq", values=["BUF"])
            ],
            soft_filters=[],
            semantic_qualifiers=["receiving outlook"],
        )
        plan = QueryPlan(
            semantic_query="DJ Moore Buffalo receiving outlook",
            keyword_query="DJ Moore Buffalo",
            intent="current_status",
            player_mentions=["D.J. Moore"],
            team_mentions=["BUF"],
            soft_team_mentions=[],
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
        repository = FakeRepository()

        resolution = EntityResolver(repository=repository).resolve(plan)

        self.assertEqual(repository.lookup_names, ["DJ Moore", "D.J. Moore"])
        self.assertEqual(repository.hard_filters[0].values, ["BUF"])
        self.assertEqual(resolution.players[0].entity_id, "buffalo-wr-id")
        self.assertEqual(resolution.players[0].display_name, "DJ Moore")

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

    def test_suffix_variant_candidates_are_sent_through_existing_escalation(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                if selector.names in (["travis etienne"], ["Travis Etienne"]):
                    return (
                        [
                            {
                                "player_id": "etienne-id",
                                "display_name": "Travis Etienne",
                                "position": "RB",
                                "position_group": "RB",
                                "rookie_season": 2021,
                                "draft_year": 2021,
                                "player_status": {
                                    "latest_team": "JAX",
                                    "jersey_number": "1",
                                    "status": "ACT",
                                },
                            }
                        ],
                        [],
                        False,
                    )
                return [], [], False

        selector = PlayerSelector(
            entity_type="player",
            reference_text="etn",
            names=["Travis Etienne Jr."],
            identity_confidence=0.98,
            resolution_basis=PlayerResolutionBasis.KNOWN_ALIAS,
            filters=[],
            semantic_qualifiers=["Week 1 flex comparison"],
        )
        plan = QueryPlan(
            semantic_query="Travis Etienne Jr. Week 1 outlook",
            keyword_query="Travis Etienne Jr. Week 1",
            intent="comparison",
            player_mentions=["Travis Etienne Jr."],
            team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            season=2026,
            week=1,
            temporal_mode="none",
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
                        canonical_name="Travis Etienne",
                        player_id="etienne-id",
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
        self.assertEqual(initial.selectors[0].status, "unresolved")

        with tempfile.TemporaryDirectory() as directory:
            result = EscalationRouter(
                index_path=Path(directory) / "index.sqlite3",
                model="test-sol",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=parse)
                ),
            ).route(
                "Who should I start, A.J. Brown or ETN?",
                plan,
                initial,
                resolver=resolver,
            )

        request = json.loads(captured["input"][1]["content"])
        issue = request["references"][0]
        self.assertEqual(issue["database_status"], "unresolved")
        self.assertEqual(issue["database_match_method"], "suffix_normalized")
        self.assertEqual(issue["database_match_count"], 1)
        self.assertEqual(
            issue["database_matches"][0]["display_name"],
            "Travis Etienne",
        )
        self.assertEqual(
            issue["database_matches"][0]["player_id"],
            "etienne-id",
        )
        self.assertTrue(result.event.triggered)
        self.assertTrue(result.event.decisions[0].grounded)
        self.assertEqual(result.resolution.selectors[0].status, "resolved")
        self.assertEqual(result.resolution.players[0].entity_id, "etienne-id")
        self.assertEqual(result.plan.player_mentions, ["Travis Etienne"])

    def test_duplicate_names_are_sent_with_context_and_selected_by_id(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return [
                    {
                        "team_abbr": "MIN",
                        "team_id": "1600",
                        "team_name": "Minnesota Vikings",
                        "team_nick": "Vikings",
                        "team_conf": "NFC",
                        "team_division": "NFC North",
                    }
                ]

            def resolve_player_selector(self, selector, *, season, week):
                if not selector.names:
                    return (
                        [
                            {
                                "player_id": "vikings-qb-id",
                                "display_name": "Example Vikings QB",
                                "position": "QB",
                                "position_group": "QB",
                                "rookie_season": 2024,
                                "draft_year": 2024,
                                "player_status": {
                                    "latest_team": "MIN",
                                    "jersey_number": "9",
                                    "status": "ACT",
                                },
                            }
                        ],
                        [],
                        False,
                    )
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
            context_requests=[
                ContextRequest(
                    anchor_selector_index=0,
                    relation="same_team",
                    semantic_query="Minnesota quarterback situation",
                    keyword_query="Minnesota quarterback starter",
                    semantic_qualifiers=["quarterback context"],
                    structured_lookups=[
                        TeamRosterLookup(
                            lookup_id="min-quarterbacks",
                            purpose=LookupPurpose.EXPAND_CANDIDATES,
                            operation="team_roster",
                            season=2026,
                            week=None,
                            position="QB",
                            status=None,
                        )
                    ],
                )
            ],
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
        self.assertEqual(result.resolution.contexts[0].status, "resolved")
        self.assertEqual(result.resolution.contexts[0].teams, ["MIN"])
        self.assertEqual(
            result.resolution.contexts[0].anchor_entities[0].entity_id,
            "vikings-wr-id",
        )
        self.assertEqual(
            result.plan.context_requests[0].structured_lookups[0].lookup_id,
            "min-quarterbacks",
        )
        self.assertEqual(result.event.decisions[0].player_id, "vikings-wr-id")
        self.assertTrue(result.event.decisions[0].grounded)

    def test_large_fuzzy_candidate_set_is_omitted_from_sol_payload(self) -> None:
        class FakeRepository:
            def list_teams(self) -> list[dict]:
                return []

            def resolve_player_selector(self, selector, *, season, week):
                if "DeMario Douglas" in selector.names:
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
                if "Douglas" in selector.names:
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

    def test_reranker_batches_candidates_reorders_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            original = parse_report(report_path)
            newer_context = replace(
                original,
                id="newer-context",
                title="General team notes",
                published_at="2026-08-10",
                body="The team held a normal practice without an Alpha update.",
            )
            direct = replace(
                original,
                id="direct-update",
                title="Alpha cleared",
                published_at="2026-08-09",
                body="Alpha Runner was cleared and returned as a full participant.",
            )
            hits = [
                SearchHit(newer_context, 0.04, "hybrid", 1, 1),
                SearchHit(direct, 0.03, "hybrid", 2, 2),
            ]
            plan = QueryPlan(
                semantic_query="Alpha Runner current health",
                keyword_query="Alpha Runner cleared practice",
                intent="current_status",
                player_mentions=["Alpha Runner"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="single_document",
            )
            resolution = ResolutionResult(selectors=[])
            response = SimpleNamespace(
                output_parsed=RerankResponse(
                    judgments=[
                        RerankJudgment(
                            document_id="newer-context",
                            relevance_score=25,
                            relationship=EvidenceRelationship.SUPPORTING_CONTEXT,
                            temporal_role=TemporalRole.CURRENT,
                            condition_alignment=ConditionAlignment.MIXED,
                            redundant_with=None,
                            reason="New but does not report Alpha's status.",
                        ),
                        RerankJudgment(
                            document_id="direct-update",
                            relevance_score=96,
                            relationship=EvidenceRelationship.DIRECT,
                            temporal_role=TemporalRole.CURRENT,
                            condition_alignment=ConditionAlignment.SUPPORTS,
                            redundant_with=None,
                            reason="Directly states Alpha's practice clearance.",
                        ),
                    ],
                    evidence_sufficiency=EvidenceSufficiency.STRONG,
                    sufficiency_reason="One report directly answers the question.",
                ),
                usage=SimpleNamespace(
                    input_tokens=200,
                    output_tokens=60,
                    input_tokens_details=SimpleNamespace(cached_tokens=20),
                ),
            )
            client = SimpleNamespace(
                responses=SimpleNamespace(parse=lambda **_: response)
            )
            reranker = ReportReranker(
                index_path=root / "index.sqlite3",
                model="test-luna",
                client=client,
            )

            first = reranker.rerank(
                "Is Alpha healthy?",
                plan,
                resolution,
                hits,
                limit=2,
            )
            self.assertEqual(first.hits[0].document.id, "direct-update")
            self.assertTrue(first.ranking_changed)
            self.assertEqual(first.input_tokens, 200)
            self.assertEqual(first.cached_input_tokens, 20)
            self.assertIsNone(first.error)
            with closing(sqlite3.connect(root / "index.sqlite3")) as connection:
                event_judgments = json.loads(
                    connection.execute(
                        "SELECT judgments_json FROM rerank_events ORDER BY id DESC"
                    ).fetchone()[0]
                )
            alignment_by_id = {
                item["document_id"]: item["condition_alignment"]
                for item in event_judgments
            }
            self.assertEqual(
                alignment_by_id,
                {
                    "direct-update": "supports",
                    "newer-context": "mixed",
                },
            )

            client.responses.parse = lambda **_: self.fail(
                "Cached reranking should avoid a second model call"
            )
            second = reranker.rerank(
                "Is Alpha healthy?",
                plan,
                resolution,
                hits,
                limit=2,
            )
            stats = reranker.stats()

            self.assertTrue(second.cached)
            self.assertFalse(second.api_called)
            self.assertEqual(second.hits[0].document.id, "direct-update")
            self.assertEqual(stats["executions"], 2)
            self.assertEqual(stats["api_calls"], 1)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(
                stats["condition_alignment"],
                {"mixed": 2, "supports": 2},
            )

    def test_reranker_rejects_invalid_ids_and_preserves_retrieval_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            document = parse_report(report_path)
            hits = [SearchHit(document, 1.0, "keyword", 1, None)]
            plan = QueryPlan(
                semantic_query="Alpha health",
                keyword_query="Alpha health",
                intent="current_status",
                player_mentions=["Alpha Runner"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="single_document",
            )
            invalid = SimpleNamespace(
                output_parsed=RerankResponse(
                    judgments=[
                        RerankJudgment(
                            document_id="invented-id",
                            relevance_score=99,
                            relationship=EvidenceRelationship.DIRECT,
                            temporal_role=TemporalRole.CURRENT,
                            condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                            redundant_with=None,
                            reason="Invalid candidate.",
                        )
                    ],
                    evidence_sufficiency=EvidenceSufficiency.STRONG,
                    sufficiency_reason="Invalid response for testing.",
                ),
                usage=None,
            )
            reranker = ReportReranker(
                index_path=root / "index.sqlite3",
                model="test-luna",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=lambda **_: invalid)
                ),
            )

            result = reranker.rerank(
                "Is Alpha healthy?",
                plan,
                ResolutionResult(selectors=[]),
                hits,
            )

            self.assertEqual(result.hits, hits)
            self.assertIsNotNone(result.error)
            self.assertIn("exactly the supplied document IDs", result.error)
            self.assertEqual(reranker.stats()["failed"], 1)

    def test_reranker_deterministically_preserves_timeline_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            template = parse_report(report_path)
            baseline = replace(
                template,
                id="baseline",
                published_at="2026-07-01",
                body="Alpha began camp limited by a hamstring injury.",
            )
            intermediate = replace(
                template,
                id="intermediate",
                published_at="2026-07-20",
                body="Alpha continued individual rehabilitation work.",
            )
            current = replace(
                template,
                id="current",
                published_at="2026-08-09",
                body="Alpha returned as a full participant.",
            )
            hits = [
                SearchHit(intermediate, 0.04, "hybrid", 1, 1),
                SearchHit(current, 0.03, "hybrid", 2, 2),
                SearchHit(baseline, 0.02, "hybrid", 3, 3),
            ]
            judgments = [
                RerankJudgment(
                    document_id="intermediate",
                    relevance_score=99,
                    relationship=EvidenceRelationship.DIRECT,
                    temporal_role=TemporalRole.INTERMEDIATE,
                    condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                    redundant_with=None,
                    reason="Strong middle update.",
                ),
                RerankJudgment(
                    document_id="current",
                    relevance_score=80,
                    relationship=EvidenceRelationship.DIRECT,
                    temporal_role=TemporalRole.CURRENT,
                    condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                    redundant_with=None,
                    reason="Establishes the current state.",
                ),
                RerankJudgment(
                    document_id="baseline",
                    relevance_score=70,
                    relationship=EvidenceRelationship.DIRECT,
                    temporal_role=TemporalRole.BASELINE,
                    condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                    redundant_with=None,
                    reason="Establishes the earlier state.",
                ),
            ]
            response = SimpleNamespace(
                output_parsed=RerankResponse(
                    judgments=judgments,
                    evidence_sufficiency=EvidenceSufficiency.STRONG,
                    sufficiency_reason="Both endpoints are present.",
                ),
                usage=None,
            )
            plan = QueryPlan(
                semantic_query="How Alpha's health changed",
                keyword_query="Alpha health timeline",
                intent="timeline",
                player_mentions=["Alpha Runner"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[],
                season=2026,
                week=None,
                temporal_mode="timeline",
                start_date="2026-07-01",
                end_date="2026-08-09",
                needs_baseline=True,
                evidence_strategy="timeline",
            )
            reranker = ReportReranker(
                index_path=root / "index.sqlite3",
                model="test-luna",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=lambda **_: response)
                ),
            )

            result = reranker.rerank(
                "How did Alpha's health change?",
                plan,
                ResolutionResult(selectors=[]),
                hits,
                limit=3,
            )

            self.assertEqual(
                [hit.document.id for hit in result.hits],
                ["baseline", "intermediate", "current"],
            )
            self.assertEqual(
                [
                    item.final_rank
                    for item in result.ranked_candidates
                    if item.final_rank is not None
                ],
                [1, 2, 3],
            )

    def test_reranker_excludes_redundant_evidence_when_alternative_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            template = parse_report(report_path)
            primary = replace(template, id="primary")
            duplicate = replace(template, id="duplicate")
            distinct = replace(
                template,
                id="distinct",
                body="The coach separately confirmed Alpha's expected workload.",
            )
            hits = [
                SearchHit(primary, 0.04, "hybrid", 1, 1),
                SearchHit(duplicate, 0.03, "hybrid", 2, 2),
                SearchHit(distinct, 0.02, "hybrid", 3, 3),
            ]
            response = SimpleNamespace(
                output_parsed=RerankResponse(
                    judgments=[
                        RerankJudgment(
                            document_id="primary",
                            relevance_score=95,
                            relationship=EvidenceRelationship.DIRECT,
                            temporal_role=TemporalRole.NOT_APPLICABLE,
                            condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                            redundant_with=None,
                            reason="Best direct report.",
                        ),
                        RerankJudgment(
                            document_id="duplicate",
                            relevance_score=90,
                            relationship=EvidenceRelationship.DIRECT,
                            temporal_role=TemporalRole.NOT_APPLICABLE,
                            condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                            redundant_with="primary",
                            reason="Repeats the primary report.",
                        ),
                        RerankJudgment(
                            document_id="distinct",
                            relevance_score=75,
                            relationship=EvidenceRelationship.SUPPORTING_CONTEXT,
                            temporal_role=TemporalRole.NOT_APPLICABLE,
                            condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                            redundant_with=None,
                            reason="Adds separate workload evidence.",
                        ),
                    ],
                    evidence_sufficiency=EvidenceSufficiency.STRONG,
                    sufficiency_reason="Direct and corroborating evidence exists.",
                ),
                usage=None,
            )
            plan = QueryPlan(
                semantic_query="Alpha role",
                keyword_query="Alpha role workload",
                intent="current_status",
                player_mentions=["Alpha Runner"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[],
                season=2026,
                week=None,
                temporal_mode="none",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="multiple_documents",
            )
            reranker = ReportReranker(
                index_path=root / "index.sqlite3",
                model="test-luna",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=lambda **_: response)
                ),
            )

            result = reranker.rerank(
                "What is Alpha's role?",
                plan,
                ResolutionResult(selectors=[]),
                hits,
                limit=2,
            )

            self.assertEqual(
                [hit.document.id for hit in result.hits],
                ["primary", "distinct"],
            )

    def test_reranker_never_fills_output_with_irrelevant_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "alpha.md"
            report_path.write_text(REPORT, encoding="utf-8")
            template = parse_report(report_path)
            direct = replace(template, id="direct")
            context = replace(
                template,
                id="context",
                body="The team described the surrounding workload context.",
            )
            irrelevant = replace(
                template,
                id="irrelevant",
                body="An unrelated team discussed a separate player.",
            )
            hits = [
                SearchHit(irrelevant, 0.04, "hybrid", 1, 1),
                SearchHit(direct, 0.03, "hybrid", 2, 2),
                SearchHit(context, 0.02, "hybrid", 3, 3),
            ]
            response = SimpleNamespace(
                output_parsed=RerankResponse(
                    judgments=[
                        RerankJudgment(
                            document_id="irrelevant",
                            relevance_score=100,
                            relationship=EvidenceRelationship.IRRELEVANT,
                            temporal_role=TemporalRole.CURRENT,
                            condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                            redundant_with=None,
                            reason="Does not address the requested subject.",
                        ),
                        RerankJudgment(
                            document_id="direct",
                            relevance_score=70,
                            relationship=EvidenceRelationship.DIRECT,
                            temporal_role=TemporalRole.CURRENT,
                            condition_alignment=ConditionAlignment.SUPPORTS,
                            redundant_with=None,
                            reason="Directly answers the question.",
                        ),
                        RerankJudgment(
                            document_id="context",
                            relevance_score=55,
                            relationship=EvidenceRelationship.SUPPORTING_CONTEXT,
                            temporal_role=TemporalRole.CURRENT,
                            condition_alignment=ConditionAlignment.NOT_APPLICABLE,
                            redundant_with=None,
                            reason="Provides material surrounding context.",
                        ),
                    ],
                    evidence_sufficiency=EvidenceSufficiency.STRONG,
                    sufficiency_reason="Direct and contextual evidence exists.",
                ),
                usage=None,
            )
            plan = QueryPlan(
                semantic_query="Alpha role",
                keyword_query="Alpha role",
                intent="current_status",
                player_mentions=["Alpha Runner"],
                team_mentions=[],
                negative_focus=[],
                entity_selectors=[],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="multiple_documents",
            )
            reranker = ReportReranker(
                index_path=root / "index.sqlite3",
                model="test-luna",
                client=SimpleNamespace(
                    responses=SimpleNamespace(parse=lambda **_: response)
                ),
            )

            result = reranker.rerank(
                "What is Alpha's role?",
                plan,
                ResolutionResult(selectors=[]),
                hits,
                limit=3,
            )

        self.assertEqual(
            [hit.document.id for hit in result.hits],
            ["direct", "context"],
        )
        irrelevant_result = next(
            item
            for item in result.ranked_candidates
            if item.hit.document.id == "irrelevant"
        )
        self.assertIsNone(irrelevant_result.final_rank)


if __name__ == "__main__":
    unittest.main()
