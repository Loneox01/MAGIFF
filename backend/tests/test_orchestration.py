import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from orchestration.registry import tool_names_for_route
from orchestration.router import (
    Capability,
    FreshnessRequirement,
    RequestIntent,
    RequestRoute,
    RequestRouter,
    StructuredDomain,
)
from rag.documents import ReportDocument
from rag.pipeline import ReportRetrievalPipeline, ReportSearchStatus
from rag.planning.context_planner import ContextPlan, ContextPlanResult
from rag.planning.planner import QueryPlan, QueryPlanResult
from rag.planning.enrichment import StructuredEnrichment
from rag.planning.resolver import ResolutionResult
from rag.retrieval.executor import ExecutionResult
from rag.retrieval.reranker import (
    ConditionAlignment,
    EvidenceRelationship,
    RankedCandidate,
    RerankJudgment,
    RerankResult,
    TemporalRole,
)
from rag.retrieval.store import SearchHit


def _report_plan() -> QueryPlan:
    return QueryPlan(
        semantic_query="current Seattle running back role",
        keyword_query="Seattle running back role",
        intent="current_status",
        player_mentions=[],
        team_mentions=["SEA"],
        soft_team_mentions=[],
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


def _report_hit() -> SearchHit:
    document = ReportDocument(
        id="seattle-backfield",
        title="Seattle updates its running back rotation",
        source="Test Sports",
        url="https://example.com/seattle-backfield",
        author="Reporter",
        published_at="2026-08-15T12:00:00-04:00",
        fetched_at="2026-08-15T13:00:00-04:00",
        players=("Example Runner",),
        teams=("SEA",),
        season=2026,
        document_type="practice_report",
        storyline="backfield_competition",
        content_mode="source_summary",
        body="Example Runner took the first rep and remained in the lead role.",
        source_path=Path("/tmp/seattle-backfield.md"),
    )
    return SearchHit(document=document, score=0.03, method="hybrid")


class OrchestrationTests(unittest.TestCase):
    def test_request_router_uses_luna_route_and_local_cache(self) -> None:
        route = RequestRoute(
            request_summary="Compare 2025 production with current role reports.",
            intent=RequestIntent.COMPARISON,
            freshness=FreshnessRequirement.CURRENT,
            capabilities=[Capability.STRUCTURED_DATA, Capability.REPORTS],
            structured_domains=[StructuredDomain.PLAYER_STATS],
            rationale="The request combines statistics and current narrative evidence.",
        )
        response = SimpleNamespace(
            output_parsed=route,
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=30,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(parse=lambda **_: response)
        )

        with tempfile.TemporaryDirectory() as directory:
            router = RequestRouter(
                index_path=Path(directory) / "router.sqlite3",
                model="test-luna",
                client=client,
            )
            first = router.route(
                "Compare his stats and current role.",
                routing_date=date(2026, 8, 16),
            )
            self.assertFalse(first.cached)
            self.assertEqual(first.cached_input_tokens, 20)

            client.responses.parse = lambda **_: self.fail(
                "The second identical route should use SQLite cache"
            )
            second = router.route(
                "Compare his stats and current role.",
                routing_date=date(2026, 8, 16),
            )
            self.assertTrue(second.cached)
            self.assertEqual(second.route, route)
            self.assertEqual(router.stats()["cache_hits"], 1)

    def test_registry_exposes_only_routed_tool_groups(self) -> None:
        reports_only = RequestRoute(
            request_summary="Find the latest injury report.",
            intent=RequestIntent.NEWS,
            freshness=FreshnessRequirement.CURRENT,
            capabilities=[Capability.REPORTS],
            structured_domains=[],
            rationale="This is a narrative news request.",
        )
        self.assertEqual(tool_names_for_route(reports_only), ["search_reports"])

        mixed = RequestRoute(
            request_summary="Compare player efficiency with ECR.",
            intent=RequestIntent.COMPARISON,
            freshness=FreshnessRequirement.HISTORICAL,
            capabilities=[Capability.STRUCTURED_DATA],
            structured_domains=[
                StructuredDomain.PLAYER_STATS,
                StructuredDomain.ECR,
            ],
            rationale="Both structured domains are required.",
        )
        names = tool_names_for_route(mixed)
        self.assertIn("rank_players_by_formula", names)
        self.assertIn("rank_players_by_ecr", names)
        self.assertNotIn("rank_teams_by_formula", names)
        self.assertNotIn("search_reports", names)
        self.assertEqual(names.count("find_players"), 1)

    def test_report_pipeline_returns_partial_evidence_and_suppresses_weak(self) -> None:
        plan = _report_plan()
        plan_result = QueryPlanResult(
            plan=plan,
            model="test-planner",
            cached=True,
            input_tokens=0,
            output_tokens=0,
        )
        resolution = ResolutionResult(selectors=[])
        event = SimpleNamespace(
            model="test-escalation",
            triggered=False,
            cache_hit=False,
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            impactful=False,
        )
        hit = _report_hit()
        judgment = RerankJudgment(
            document_id=hit.document.id,
            relevance_score=94,
            relationship=EvidenceRelationship.DIRECT,
            temporal_role=TemporalRole.CURRENT,
            condition_alignment=ConditionAlignment.NOT_APPLICABLE,
            redundant_with=None,
            reason="Directly describes the current role.",
        )
        ranked = RankedCandidate(
            hit=hit,
            judgment=judgment,
            original_rank=1,
            final_rank=1,
            adjusted_score=102,
        )

        planner = SimpleNamespace(plan=lambda *_args, **_kwargs: plan_result)
        context_plan = ContextPlan(
            context_needed=False,
            rationale="The direct plan is sufficient for this fixture.",
            context_requests=[],
        )
        context_planner = SimpleNamespace(
            expand=lambda *_args, **_kwargs: ContextPlanResult(
                context_plan=context_plan,
                plan=plan,
                model="test-context-planner",
                cached=True,
                input_tokens=0,
                cached_input_tokens=0,
                output_tokens=0,
            )
        )
        resolver = SimpleNamespace(resolve=lambda _plan: resolution)
        identity_router = SimpleNamespace(
            route=lambda *_args, **_kwargs: SimpleNamespace(
                plan=plan,
                resolution=resolution,
                event=event,
            )
        )
        execution = ExecutionResult(
            hits=[hit],
            resolution=resolution,
            keyword_query=plan.keyword_query,
            vector_query=plan.semantic_query,
            linked_document_entities=0,
            strategy="resolved",
        )
        calls = {}
        enrichment = StructuredEnrichment()
        lookup_executor = SimpleNamespace(
            execute=lambda received_plan, received_resolution: (
                calls.update(
                    {
                        "lookup_plan": received_plan,
                        "lookup_resolution": received_resolution,
                    }
                )
                or enrichment
            )
        )

        def execute(*_args, **kwargs):
            calls["candidate_limit"] = kwargs["limit"]
            calls["executor_enrichment"] = kwargs["enrichment"]
            return execution

        executor = SimpleNamespace(execute=execute)

        def result_for(sufficiency: str) -> RerankResult:
            return RerankResult(
                hits=[hit],
                ranked_candidates=(ranked,),
                model="test-reranker",
                cached=False,
                api_called=True,
                input_tokens=50,
                cached_input_tokens=0,
                output_tokens=10,
                estimated_cost_usd=None,
                latency_ms=10,
                candidate_count=1,
                ranking_changed=False,
                evidence_sufficiency=sufficiency,
                sufficiency_reason=f"Evidence is {sufficiency}.",
                error=None,
            )

        def rerank(*_args, **kwargs):
            calls["result_limit"] = kwargs["limit"]
            calls["reranker_enrichment"] = kwargs["enrichment"]
            return result_for("partial")

        reranker = SimpleNamespace(rerank=rerank)
        pipeline = ReportRetrievalPipeline(
            store=SimpleNamespace(index_path=Path("/tmp/test-index.sqlite3")),
            planner=planner,
            context_planner=context_planner,
            resolver=resolver,
            identity_router=identity_router,
            lookup_executor=lookup_executor,
            executor=executor,
            reranker=reranker,
            candidate_limit=20,
        )

        partial = pipeline.search("Who leads Seattle's backfield?")
        self.assertEqual(partial.status, ReportSearchStatus.PARTIAL)
        self.assertEqual(len(partial.reports), 1)
        self.assertEqual(partial.reports[0].url, hit.document.url)
        self.assertEqual(calls["candidate_limit"], 20)
        self.assertEqual(calls["result_limit"], 5)
        self.assertIs(calls["lookup_plan"], plan)
        self.assertIs(calls["lookup_resolution"], resolution)
        self.assertIs(calls["executor_enrichment"], enrichment)
        self.assertIs(calls["reranker_enrichment"], enrichment)

        reranker.rerank = lambda *_args, **_kwargs: result_for("weak")
        weak = pipeline.search("Who leads Seattle's backfield?")
        self.assertEqual(weak.status, ReportSearchStatus.NO_EVIDENCE)
        self.assertEqual(weak.reports, ())


if __name__ == "__main__":
    unittest.main()
