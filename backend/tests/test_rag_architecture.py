import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from rag.documents import parse_report
from rag.planning.context_planner import (
    ContextPlan,
    ContextPlanner,
    merge_context_plan,
)
from rag.planning.enrichment import (
    ContextEnrichment,
    LookupExecution,
    StructuredEnrichment,
    StructuredLookupExecutor,
    StructuredToolGateway,
    TargetEnrichment,
)
from rag.planning.lookups import (
    ContextScopePolicy,
    PlayerSeasonStatsLookup,
    TeamDepthChartLookup,
    TeamRosterLookup,
    TeamScheduleLookup,
)
from rag.planning.planner import (
    ContextRequest,
    DirectQueryPlan,
    PlayerSelector,
    PositionGroupFilter,
    QueryPlan,
    TeamSelector,
)
from rag.planning.resolver import (
    ContextResolution,
    ResolutionResult,
    ResolvedEntity,
    SelectorResolution,
)
from rag.retrieval.executor import QueryPlanExecutor
from rag.retrieval.store import LocalRAGStore
from tools import nfl as nfl_tools


REPORT = """---
id: "{document_id}"
title: "{title}"
source: "Test Wire"
url: "https://example.com/{document_id}"
author: null
published_at: "2026-08-09"
fetched_at: "2026-08-09"
players: ["{player}"]
teams: ["{team}"]
season: 2026
document_type: "news"
storyline: "architecture_test"
content_mode: "source_summary"
---

# Summary

{body}
"""


def report(
    document_id: str,
    title: str,
    player: str,
    team: str,
    body: str,
) -> str:
    return REPORT.format(
        document_id=document_id,
        title=title,
        player=player,
        team=team,
        body=body,
    )


def player_selector(name: str) -> PlayerSelector:
    return PlayerSelector(
        entity_type="player",
        reference_text=name,
        names=[name],
        identity_confidence=1,
        resolution_basis="exact_name",
        hard_filters=[],
        soft_filters=[],
        semantic_qualifiers=[],
    )


def plan_with_context(
    *,
    scope_policy: ContextScopePolicy = ContextScopePolicy.ANCHOR_TEAMS,
) -> QueryPlan:
    selector = player_selector("Drake London")
    lookup = TeamRosterLookup(
        lookup_id="related-quarterbacks",
        purpose="expand_candidates",
        operation="team_roster",
        season=2026,
        week=None,
        position="QB",
        status=None,
    )
    return QueryPlan(
        semantic_query="quarterback context affecting Drake London",
        keyword_query="Drake London quarterback context",
        intent="projection",
        player_mentions=["Drake London"],
        team_mentions=[],
        soft_team_mentions=[],
        negative_focus=[],
        entity_selectors=[selector],
        context_requests=[
            ContextRequest(
                anchor_selector_index=0,
                relation="dependency",
                semantic_query="quarterback competition and availability",
                keyword_query="quarterback competition availability",
                semantic_qualifiers=["passing environment"],
                scope_policy=scope_policy,
                structured_lookups=[lookup],
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


def london_resolution(plan: QueryPlan) -> tuple[ResolutionResult, ResolvedEntity]:
    london = ResolvedEntity(
        entity_type="player",
        entity_id="london-id",
        display_name="Drake London",
        team="ATL",
        position="WR",
    )
    return (
        ResolutionResult(
            selectors=[
                SelectorResolution(
                    selector_index=0,
                    selector=plan.entity_selectors[0],
                    status="resolved",
                    matches=[london],
                    unresolved_filters=[],
                    semantic_qualifiers=[],
                    truncated=False,
                )
            ],
            contexts=[
                ContextResolution(
                    request_index=0,
                    request=plan.context_requests[0],
                    status="resolved",
                    anchor_entities=[london],
                    teams=["ATL"],
                    unresolved=[],
                    truncated=False,
                )
            ],
        ),
        london,
    )


class RagArchitectureTests(unittest.TestCase):
    def test_context_merge_downgrades_unbacked_lookup_scopes(self) -> None:
        direct_plan = plan_with_context().model_copy(
            update={"context_requests": []}
        )
        context_plan = ContextPlan(
            context_needed=True,
            rationale="Indirect evidence may affect the comparison.",
            context_requests=[
                ContextRequest(
                    anchor_selector_index=0,
                    relation="comparison",
                    semantic_query="related comparison evidence",
                    keyword_query="related comparison evidence",
                    semantic_qualifiers=[],
                    scope_policy=ContextScopePolicy.LOOKUP_ENTITIES,
                    structured_lookups=[],
                ),
                ContextRequest(
                    anchor_selector_index=0,
                    relation="environment",
                    semantic_query="team environment evidence",
                    keyword_query="team environment evidence",
                    semantic_qualifiers=[],
                    scope_policy=ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS,
                    structured_lookups=[],
                ),
            ],
        )

        merged = merge_context_plan(direct_plan, context_plan)

        self.assertEqual(
            merged.context_requests[0].scope_policy,
            ContextScopePolicy.SEMANTIC_ONLY,
        )
        self.assertEqual(
            merged.context_requests[1].scope_policy,
            ContextScopePolicy.ANCHOR_TEAMS,
        )

    def test_context_merge_preserves_lookup_backed_scope(self) -> None:
        direct_plan = plan_with_context().model_copy(
            update={"context_requests": []}
        )
        context_plan = ContextPlan(
            context_needed=True,
            rationale="A schedule lookup grounds the opponent scope.",
            context_requests=[
                ContextRequest(
                    anchor_selector_index=0,
                    relation="matchup",
                    semantic_query="opponent matchup evidence",
                    keyword_query="opponent matchup evidence",
                    semantic_qualifiers=[],
                    scope_policy=ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS,
                    structured_lookups=[
                        TeamScheduleLookup(
                            lookup_id="week-one-opponent",
                            operation="team_schedule",
                            purpose="resolve_relationship",
                            season=2026,
                            week=1,
                        )
                    ],
                )
            ],
        )

        merged = merge_context_plan(direct_plan, context_plan)

        self.assertEqual(
            merged.context_requests[0].scope_policy,
            ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS,
        )

    def test_query_plan_schema_uses_supported_structured_output_composition(
        self,
    ) -> None:
        schema = to_strict_json_schema(QueryPlan)
        unsupported_composition = {
            "allOf",
            "dependentRequired",
            "dependentSchemas",
            "else",
            "if",
            "not",
            "oneOf",
            "then",
        }
        unsupported_paths: list[str] = []

        def visit(value, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    child_path = (*path, key)
                    if key in unsupported_composition:
                        unsupported_paths.append("/".join(child_path))
                    visit(child, child_path)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(schema)

        self.assertEqual(schema.get("type"), "object")
        self.assertEqual(unsupported_paths, [])
        self.assertIn(
            "anyOf",
            schema["$defs"]["PlayerSelector"]["properties"]
            ["structured_lookups"]["items"],
        )

    def test_staged_planner_schemas_are_supported_and_disjoint(self) -> None:
        direct_schema = to_strict_json_schema(DirectQueryPlan)
        context_schema = to_strict_json_schema(ContextPlan)

        self.assertNotIn("context_requests", direct_schema["properties"])
        self.assertNotIn("entity_selectors", context_schema["properties"])

        for schema in (direct_schema, context_schema):
            serialized = str(schema)
            self.assertNotIn("'oneOf'", serialized)
            self.assertNotIn("'allOf'", serialized)

    def test_context_planner_merges_grounded_second_stage_and_caches(self) -> None:
        direct_plan = QueryPlan(
            semantic_query="Drake London current outlook",
            keyword_query="Drake London outlook",
            intent="projection",
            player_mentions=["Drake London"],
            team_mentions=[],
            soft_team_mentions=["ATL"],
            negative_focus=[],
            entity_selectors=[player_selector("Drake London")],
            context_requests=[],
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="multiple_documents",
        )
        resolution, _ = london_resolution(
            direct_plan.model_copy(
                update={
                    "context_requests": [
                        ContextRequest(
                            anchor_selector_index=0,
                            relation="environment",
                            semantic_query="placeholder",
                            keyword_query="placeholder",
                            semantic_qualifiers=[],
                            scope_policy="anchor_teams",
                            structured_lookups=[],
                        )
                    ]
                }
            )
        )
        # The second stage receives direct resolution only.
        resolution = resolution.model_copy(update={"contexts": []})
        context_plan = ContextPlan(
            context_needed=True,
            rationale="The projection depends on related team evidence.",
            context_requests=[
                ContextRequest(
                    anchor_selector_index=0,
                    relation="environment",
                    semantic_query="Atlanta passing environment affecting outlook",
                    keyword_query="Atlanta passing environment outlook",
                    semantic_qualifiers=["material team context"],
                    scope_policy="anchor_and_lookup_teams",
                    structured_lookups=[
                        TeamScheduleLookup(
                            lookup_id="week-one-opponent",
                            operation="team_schedule",
                            purpose="resolve_relationship",
                            season=2026,
                            week=1,
                        )
                    ],
                )
            ],
        )
        response = SimpleNamespace(
            output_parsed=context_plan,
            usage=SimpleNamespace(
                input_tokens=80,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )
        parse = Mock(return_value=response)

        with tempfile.TemporaryDirectory() as directory:
            planner = ContextPlanner(
                index_path=Path(directory) / "index.sqlite3",
                model="test-context",
                client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
            )
            first = planner.expand(
                "How does the environment affect Drake London?",
                direct_plan,
                resolution,
                planning_date=date(2026, 8, 20),
            )
            second = planner.expand(
                "How does the environment affect Drake London?",
                direct_plan,
                resolution,
                planning_date=date(2026, 8, 20),
            )

        self.assertEqual(parse.call_count, 1)
        self.assertIs(parse.call_args.kwargs["text_format"], ContextPlan)
        self.assertEqual(len(first.plan.context_requests), 1)
        self.assertEqual(first.cached_input_tokens, 10)
        self.assertTrue(second.cached)

    def test_context_planner_retries_once_for_incompatible_context(self) -> None:
        base_plan = plan_with_context()
        resolution, _ = london_resolution(base_plan)
        direct_plan = base_plan.model_copy(update={"context_requests": []})
        resolution = resolution.model_copy(update={"contexts": []})

        def context_plan(season: int) -> ContextPlan:
            return ContextPlan(
                context_needed=True,
                rationale="The matchup requires a grounded opponent branch.",
                context_requests=[
                    ContextRequest(
                        anchor_selector_index=0,
                        relation="matchup",
                        semantic_query="opponent defense and matchup context",
                        keyword_query="opponent defense matchup",
                        semantic_qualifiers=["week one opponent"],
                        scope_policy="anchor_teams",
                        structured_lookups=[
                            TeamScheduleLookup(
                                lookup_id="week-one-opponent",
                                operation="team_schedule",
                                purpose="resolve_relationship",
                                season=season,
                                week=1,
                            )
                        ],
                    )
                ],
            )

        responses = [
            SimpleNamespace(
                output_parsed=context_plan(2025),
                usage=SimpleNamespace(
                    input_tokens=80,
                    output_tokens=20,
                    input_tokens_details=SimpleNamespace(cached_tokens=10),
                ),
            ),
            SimpleNamespace(
                output_parsed=context_plan(2026),
                usage=SimpleNamespace(
                    input_tokens=90,
                    output_tokens=25,
                    input_tokens_details=SimpleNamespace(cached_tokens=5),
                ),
            ),
        ]
        parse = Mock(side_effect=responses)

        with tempfile.TemporaryDirectory() as directory:
            planner = ContextPlanner(
                index_path=Path(directory) / "index.sqlite3",
                model="test-context",
                client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
            )
            result = planner.expand(
                "Does Drake London have a good Week 1 matchup?",
                direct_plan,
                resolution,
                planning_date=date(2026, 8, 20),
                use_cache=False,
            )

        self.assertEqual(parse.call_count, 2)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.retried)
        self.assertIn("seasons must match", result.retry_reason)
        self.assertEqual(result.input_tokens, 170)
        self.assertEqual(result.cached_input_tokens, 15)
        self.assertEqual(result.output_tokens, 45)
        lookup = result.plan.context_requests[0].structured_lookups[0]
        self.assertEqual(lookup.season, 2026)

        correction_payload = json.loads(
            parse.call_args_list[1].kwargs["input"][1]["content"]
        )
        self.assertIn("correction", correction_payload)
        self.assertEqual(
            correction_payload["correction"]["previous_context_plan"]
            ["context_requests"][0]["structured_lookups"][0]["season"],
            2025,
        )

    def test_context_planner_does_not_retry_transport_failures(self) -> None:
        base_plan = plan_with_context()
        resolution, _ = london_resolution(base_plan)
        direct_plan = base_plan.model_copy(update={"context_requests": []})
        resolution = resolution.model_copy(update={"contexts": []})
        parse = Mock(side_effect=ConnectionError("temporary transport failure"))

        with tempfile.TemporaryDirectory() as directory:
            planner = ContextPlanner(
                index_path=Path(directory) / "index.sqlite3",
                model="test-context",
                client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "temporary transport failure",
            ):
                planner.expand(
                    "How does the environment affect Drake London?",
                    direct_plan,
                    resolution,
                    planning_date=date(2026, 8, 20),
                    use_cache=False,
                )

        self.assertEqual(parse.call_count, 1)

    def test_player_group_names_do_not_expand_the_search_query(self) -> None:
        selector = PlayerSelector(
            entity_type="player",
            reference_text="running backs",
            names=[],
            identity_confidence=0,
            resolution_basis="not_applicable",
            hard_filters=[
                PositionGroupFilter(
                    field="position_group",
                    operator="eq",
                    values=["RB"],
                )
            ],
            soft_filters=[],
            semantic_qualifiers=["backfield usage"],
        )
        plan = QueryPlan(
            semantic_query="running back usage",
            keyword_query="running back usage",
            intent="comparison",
            player_mentions=[],
            team_mentions=[],
            soft_team_mentions=[],
            negative_focus=[],
            entity_selectors=[selector],
            context_requests=[],
            season=2026,
            week=None,
            temporal_mode="none",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="per_entity",
        )
        players = [
            ResolvedEntity(
                entity_type="player",
                entity_id=f"runner-{index}",
                display_name=name,
                team="SEA",
                position="RB",
            )
            for index, name in enumerate(
                ["First Resolved Runner", "Second Resolved Runner"]
            )
        ]
        resolution = ResolutionResult(
            selectors=[
                SelectorResolution(
                    selector_index=0,
                    selector=selector,
                    status="multiple",
                    matches=players,
                    unresolved_filters=[],
                    semantic_qualifiers=[],
                    truncated=False,
                )
            ]
        )
        calls = []

        class RecordingStore:
            def link_player_entities(self, _players):
                return 0

            def search(self, _query, **kwargs):
                calls.append(kwargs)
                return []

        QueryPlanExecutor(RecordingStore()).execute(
            "Compare the running backs",
            plan,
            resolution,
            mode="keyword",
            limit=5,
            embedding_model="unused",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["keyword_query"], "running back usage")
        self.assertEqual(calls[0]["vector_query"], "running back usage")

    def test_independent_targets_are_a_union_not_a_cross_entity_and(self) -> None:
        london_selector = player_selector("Drake London")
        seattle_selector = TeamSelector(
            entity_type="team",
            names=["SEA"],
            hard_filters=[],
            soft_filters=[],
            semantic_qualifiers=[],
        )
        plan = QueryPlan(
            semantic_query="Compare Drake London and the Seattle offense",
            keyword_query="Drake London Seattle offense camp",
            intent="comparison",
            player_mentions=["Drake London"],
            team_mentions=["SEA"],
            soft_team_mentions=[],
            negative_focus=[],
            entity_selectors=[london_selector, seattle_selector],
            context_requests=[],
            season=2026,
            week=None,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="per_entity",
        )
        london = ResolvedEntity(
            entity_type="player",
            entity_id="london-id",
            display_name="Drake London",
            team="ATL",
            position="WR",
        )
        seattle = ResolvedEntity(
            entity_type="team",
            entity_id="SEA",
            display_name="Seattle Seahawks",
            team="SEA",
            position=None,
        )
        resolution = ResolutionResult(
            selectors=[
                SelectorResolution(
                    selector_index=0,
                    selector=london_selector,
                    status="resolved",
                    matches=[london],
                    unresolved_filters=[],
                    semantic_qualifiers=[],
                    truncated=False,
                ),
                SelectorResolution(
                    selector_index=1,
                    selector=seattle_selector,
                    status="resolved",
                    matches=[seattle],
                    unresolved_filters=[],
                    semantic_qualifiers=[],
                    truncated=False,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            london_path = root / "london.md"
            seattle_path = root / "seattle.md"
            london_path.write_text(
                report(
                    "london-target",
                    "Drake London camp update",
                    "Drake London",
                    "ATL",
                    "Drake London remained the primary receiver in camp.",
                ),
                encoding="utf-8",
            )
            seattle_path.write_text(
                report(
                    "seattle-target",
                    "Seattle offense camp update",
                    "Seattle Runner",
                    "SEA",
                    "Seattle changed its offensive workload during camp.",
                ),
                encoding="utf-8",
            )
            store = LocalRAGStore(root / "index.sqlite3")
            store.build_index(
                [parse_report(london_path), parse_report(seattle_path)]
            )
            result = QueryPlanExecutor(store).execute(
                "Compare Drake London and the Seattle offense",
                plan,
                resolution,
                mode="keyword",
                limit=5,
                embedding_model="unused",
            )

        self.assertEqual(
            {hit.document.id for hit in result.hits},
            {"london-target", "seattle-target"},
        )
        self.assertEqual(result.branch_candidates["target:0"], 1)
        self.assertEqual(result.branch_candidates["target:1"], 1)

    def test_lookup_entity_scope_is_local_to_the_context_branch(self) -> None:
        plan = plan_with_context(
            scope_policy=ContextScopePolicy.LOOKUP_ENTITIES
        )
        resolution, _london = london_resolution(plan)
        penix = ResolvedEntity(
            entity_type="player",
            entity_id="penix-id",
            display_name="Michael Penix Jr.",
            team="ATL",
            position="QB",
        )
        enrichment = StructuredEnrichment(
            contexts=[
                ContextEnrichment(
                    request_index=0,
                    lookups=[
                        LookupExecution(
                            lookup_id="related-quarterbacks",
                            operation="team_roster",
                            purpose="expand_candidates",
                            status="resolved",
                            entities=[penix],
                            teams=["ATL"],
                            query_terms=["Michael Penix Jr."],
                            facts=[],
                            error=None,
                        )
                    ],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = [
                report(
                    "london-direct",
                    "Drake London outlook update",
                    "Drake London",
                    "ATL",
                    "Drake London's role remained stable.",
                ),
                report(
                    "penix-context",
                    "Michael Penix quarterback competition update",
                    "Michael Penix Jr.",
                    "ATL",
                    "Michael Penix remained in the quarterback competition.",
                ),
                report(
                    "adjacent-context",
                    "Quarterback competition affects Atlanta practice",
                    "Bijan Robinson",
                    "ATL",
                    "Atlanta's quarterback competition continued in practice.",
                ),
            ]
            paths = []
            for index, document in enumerate(documents):
                path = root / f"{index}.md"
                path.write_text(document, encoding="utf-8")
                paths.append(path)
            store = LocalRAGStore(root / "index.sqlite3")
            store.build_index([parse_report(path) for path in paths])
            result = QueryPlanExecutor(store).execute(
                "How does the quarterback situation affect Drake London?",
                plan,
                resolution,
                mode="keyword",
                limit=5,
                embedding_model="unused",
                enrichment=enrichment,
            )

        ids = {hit.document.id for hit in result.hits}
        self.assertIn("london-direct", ids)
        self.assertIn("penix-context", ids)
        self.assertNotIn("adjacent-context", ids)

    def test_resolved_relationship_automatically_expands_context_scope(
        self,
    ) -> None:
        selector = player_selector("Drake London")
        relationship_lookup = TeamScheduleLookup(
            lookup_id="week-one-opponent",
            operation="team_schedule",
            purpose="resolve_relationship",
            season=2026,
            week=1,
        )
        plan = QueryPlan(
            semantic_query="Drake London week one matchup",
            keyword_query="Drake London week one matchup",
            intent="projection",
            player_mentions=["Drake London"],
            team_mentions=[],
            soft_team_mentions=["ATL"],
            negative_focus=[],
            entity_selectors=[selector],
            context_requests=[
                ContextRequest(
                    anchor_selector_index=0,
                    relation="matchup",
                    semantic_query="opponent secondary injury context",
                    keyword_query="opponent secondary injury",
                    semantic_qualifiers=["opposing defense"],
                    # Even this restrictive model choice must not discard a
                    # team discovered by a relationship lookup.
                    scope_policy="anchor_teams",
                    structured_lookups=[relationship_lookup],
                )
            ],
            season=2026,
            week=1,
            temporal_mode="current",
            start_date=None,
            end_date=None,
            needs_baseline=False,
            evidence_strategy="multiple_documents",
        )
        resolution, _ = london_resolution(plan)
        enrichment = StructuredEnrichment(
            contexts=[
                ContextEnrichment(
                    request_index=0,
                    lookups=[
                        LookupExecution(
                            lookup_id="week-one-opponent",
                            operation="team_schedule",
                            purpose="resolve_relationship",
                            status="resolved",
                            entities=[],
                            teams=["ATL", "SEA"],
                            query_terms=["ATL", "SEA"],
                            facts=[],
                            error=None,
                        )
                    ],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "opponent.md"
            path.write_text(
                report(
                    "seattle-secondary",
                    "Seattle opponent secondary injury update",
                    "Seattle Defender",
                    "SEA",
                    "Seattle's opponent secondary had a significant injury.",
                ),
                encoding="utf-8",
            )
            store = LocalRAGStore(root / "index.sqlite3")
            store.build_index([parse_report(path)])
            result = QueryPlanExecutor(store).execute(
                "Does Drake London have a good Week 1 matchup?",
                plan,
                resolution,
                mode="keyword",
                limit=5,
                embedding_model="unused",
                enrichment=enrichment,
            )

        self.assertEqual(
            [hit.document.id for hit in result.hits],
            ["seattle-secondary"],
        )
        self.assertEqual(result.branch_candidates["context:0"], 1)

    def test_failed_required_lookup_does_not_broaden_context(self) -> None:
        plan = plan_with_context()
        required_lookup = plan.context_requests[0].structured_lookups[0].model_copy(
            update={"purpose": "resolve_relationship"}
        )
        plan = plan.model_copy(
            update={
                "context_requests": [
                    plan.context_requests[0].model_copy(
                        update={"structured_lookups": [required_lookup]}
                    )
                ]
            }
        )
        resolution, _london = london_resolution(plan)
        enrichment = StructuredEnrichment(
            contexts=[
                ContextEnrichment(
                    request_index=0,
                    lookups=[
                        LookupExecution(
                            lookup_id="related-quarterbacks",
                            operation="team_roster",
                            purpose="resolve_relationship",
                            status="empty",
                            entities=[],
                            teams=[],
                            query_terms=[],
                            facts=[],
                            error=None,
                        )
                    ],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "adjacent.md"
            path.write_text(
                report(
                    "adjacent-quarterback",
                    "Atlanta quarterback competition update",
                    "Other Player",
                    "ATL",
                    "The Atlanta quarterback competition continued.",
                ),
                encoding="utf-8",
            )
            store = LocalRAGStore(root / "index.sqlite3")
            store.build_index([parse_report(path)])
            result = QueryPlanExecutor(store).execute(
                "How does the quarterback situation affect Drake London?",
                plan,
                resolution,
                mode="keyword",
                limit=5,
                embedding_model="unused",
                enrichment=enrichment,
            )
            missing_result = QueryPlanExecutor(store).execute(
                "How does the quarterback situation affect Drake London?",
                plan,
                resolution,
                mode="keyword",
                limit=5,
                embedding_model="unused",
                enrichment=StructuredEnrichment(),
            )

        self.assertEqual(result.hits, [])
        self.assertEqual(result.branch_candidates["context:0"], 0)
        self.assertEqual(missing_result.hits, [])
        self.assertEqual(missing_result.branch_candidates["context:0"], 0)

    def test_optional_candidate_lookup_does_not_block_grounded_team_context(
        self,
    ) -> None:
        plan = plan_with_context()
        resolution, _london = london_resolution(plan)
        enrichment = StructuredEnrichment(
            contexts=[
                ContextEnrichment(
                    request_index=0,
                    lookups=[
                        LookupExecution(
                            lookup_id="related-quarterbacks",
                            operation="team_roster",
                            purpose="expand_candidates",
                            status="empty",
                            entities=[],
                            teams=[],
                            query_terms=[],
                            facts=[],
                            error=None,
                        )
                    ],
                )
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "team-context.md"
            path.write_text(
                report(
                    "team-context",
                    "Atlanta quarterback competition update",
                    "Other Player",
                    "ATL",
                    "The Atlanta quarterback competition continued.",
                ),
                encoding="utf-8",
            )
            store = LocalRAGStore(root / "index.sqlite3")
            store.build_index([parse_report(path)])
            result = QueryPlanExecutor(store).execute(
                "How does the quarterback situation affect Drake London?",
                plan,
                resolution,
                mode="keyword",
                limit=5,
                embedding_model="unused",
                enrichment=enrichment,
            )

        self.assertEqual(
            [hit.document.id for hit in result.hits],
            ["team-context"],
        )
        self.assertEqual(result.branch_candidates["context:0"], 1)

    def test_lookup_executor_receives_grounded_anchor_not_model_ids(self) -> None:
        calls = []

        class RecordingGateway:
            def execute(self, lookup, *, players, teams):
                calls.append((lookup.lookup_id, players, teams))
                return LookupExecution(
                    lookup_id=lookup.lookup_id,
                    operation=lookup.operation,
                    purpose=lookup.purpose,
                    status="empty",
                    entities=[],
                    teams=[],
                    query_terms=[],
                    facts=[],
                    error=None,
                )

        plan = plan_with_context()
        resolution, _london = london_resolution(plan)
        enrichment = StructuredLookupExecutor(
            gateway=RecordingGateway()
        ).execute(plan, resolution)

        self.assertEqual(len(enrichment.contexts[0].lookups), 1)
        self.assertEqual(calls[0][0], "related-quarterbacks")
        self.assertEqual(
            [player.entity_id for player in calls[0][1]],
            ["london-id"],
        )
        self.assertEqual(calls[0][2], ["ATL"])

    def test_current_depth_chart_falls_back_to_latest_snapshot(self) -> None:
        lookup = TeamDepthChartLookup(
            lookup_id="week-one-depth-chart",
            operation="team_depth_chart",
            purpose="reranker_context",
            season=2026,
            week=1,
            position="RB",
        )

        def depth_chart(_team, _season, week, _position):
            if week == 1:
                return []
            return [
                {
                    "player_id": "runner-id",
                    "player_name": "Current Runner",
                    "team": "SEA",
                    "season": 2026,
                    "week": None,
                    "position": "RB",
                }
            ]

        with patch(
            "rag.planning.enrichment.nfl_tools.get_team_depth_chart",
            side_effect=depth_chart,
        ) as get_depth_chart, patch(
            "rag.planning.enrichment.repository.get_player_names",
            return_value={"runner-id": "Current Runner"},
        ):
            execution = StructuredToolGateway(
                current_date=date(2026, 8, 20)
            ).execute(lookup, players=[], teams=["SEA"])

        self.assertEqual(get_depth_chart.call_count, 2)
        self.assertEqual(get_depth_chart.call_args_list[0].args[2], 1)
        self.assertIsNone(get_depth_chart.call_args_list[1].args[2])
        self.assertEqual(execution.status, "resolved")
        self.assertTrue(execution.fallback_used)
        self.assertIn("current snapshot", execution.fallback_reason)
        self.assertIn("SEA", execution.teams)
        self.assertIn(
            "runner-id",
            [entity.entity_id for entity in execution.entities],
        )

    def test_current_roster_falls_back_to_player_status_snapshot(self) -> None:
        current_rows = [
            {
                "player_id": "runner-id",
                "player_name": "Current Runner",
                "season": 2026,
                "week": None,
                "game_type": None,
                "team": "SEA",
                "position": "RB",
                "depth_chart_position": None,
                "jersey_number": "30",
                "status": "ACT",
                "status_description_abbr": None,
                "years_exp": 2,
            }
        ]
        with patch(
            "tools.nfl.repository.get_team_roster",
            return_value=[],
        ) as weekly_roster, patch(
            "tools.nfl.repository.get_current_team_roster",
            return_value=current_rows,
        ) as current_roster:
            result = nfl_tools.get_team_roster(
                "SEA",
                2026,
                1,
                "RB",
                None,
                current_date=date(2026, 8, 20),
            )

        weekly_roster.assert_called_once_with("SEA", 2026, 1, "RB", None)
        current_roster.assert_called_once_with("SEA", 2026, "RB", None)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["source_snapshot"], "current_player_status")
        self.assertEqual(result["results"], current_rows)
        self.assertIn("preseason-sized", result["fallback_reason"])

    def test_historical_roster_does_not_use_current_status_snapshot(self) -> None:
        with patch(
            "tools.nfl.repository.get_team_roster",
            return_value=[],
        ), patch(
            "tools.nfl.repository.get_current_team_roster",
        ) as current_roster:
            result = nfl_tools.get_team_roster(
                "SEA",
                2025,
                1,
                "RB",
                None,
                current_date=date(2026, 8, 20),
            )

        current_roster.assert_not_called()
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["results"], [])

    def test_structured_roster_lookup_records_fallback_provenance(self) -> None:
        lookup = TeamRosterLookup(
            lookup_id="current-roster",
            operation="team_roster",
            purpose="expand_candidates",
            season=2026,
            week=1,
            position="RB",
            status=None,
        )
        tool_result = {
            "results": [
                {
                    "player_id": "runner-id",
                    "player_name": "Current Runner",
                    "season": 2026,
                    "week": None,
                    "team": "SEA",
                    "position": "RB",
                    "status": "ACT",
                }
            ],
            "source_snapshot": "current_player_status",
            "fallback_used": True,
            "fallback_reason": "Used current status snapshot.",
        }
        with patch(
            "rag.planning.enrichment.nfl_tools.get_team_roster",
            return_value=tool_result,
        ) as get_roster, patch(
            "rag.planning.enrichment.repository.get_player_names",
            return_value={"runner-id": "Current Runner"},
        ):
            execution = StructuredToolGateway(
                current_date=date(2026, 8, 20)
            ).execute(lookup, players=[], teams=["SEA"])

        self.assertEqual(
            get_roster.call_args.kwargs["current_date"],
            date(2026, 8, 20),
        )
        self.assertEqual(execution.status, "resolved")
        self.assertTrue(execution.fallback_used)
        self.assertEqual(
            execution.fallback_reason,
            "Used current status snapshot.",
        )

    def test_historical_depth_chart_does_not_use_current_snapshot(self) -> None:
        lookup = TeamDepthChartLookup(
            lookup_id="historical-depth-chart",
            operation="team_depth_chart",
            purpose="reranker_context",
            season=2025,
            week=1,
            position="RB",
        )

        with patch(
            "rag.planning.enrichment.nfl_tools.get_team_depth_chart",
            return_value=[],
        ) as get_depth_chart:
            execution = StructuredToolGateway(
                current_date=date(2026, 8, 20)
            ).execute(lookup, players=[], teams=["SEA"])

        self.assertEqual(get_depth_chart.call_count, 1)
        self.assertEqual(execution.status, "empty")
        self.assertFalse(execution.fallback_used)

    def test_lookup_vocabulary_and_scope_contract_are_validated(self) -> None:
        with self.assertRaises(ValidationError):
            PlayerSeasonStatsLookup(
                lookup_id="bad-field",
                purpose="reranker_context",
                operation="player_season_stats",
                season=2025,
                season_type="REG",
                fields=["not_a_stored_stat"],
            )

        selector = player_selector("Drake London")
        invalid_target = selector.model_copy(
            update={
                "structured_lookups": [
                    TeamRosterLookup(
                        lookup_id="target-roster",
                        purpose="enrich_query",
                        operation="team_roster",
                        season=2026,
                        week=None,
                        position="QB",
                        status=None,
                    )
                ]
            }
        )
        with self.assertRaises(ValidationError):
            QueryPlan(
                semantic_query="player team environment",
                keyword_query="player team environment",
                intent="projection",
                player_mentions=["Drake London"],
                team_mentions=[],
                soft_team_mentions=[],
                negative_focus=[],
                entity_selectors=[invalid_target],
                context_requests=[],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="multiple_documents",
            )

        relationship_target = selector.model_copy(
            update={
                "structured_lookups": [
                    PlayerSeasonStatsLookup(
                        lookup_id="target-relationship",
                        purpose="resolve_relationship",
                        operation="player_season_stats",
                        season=2025,
                        season_type="REG",
                        fields=["fantasy_points_ppr"],
                    )
                ]
            }
        )
        with self.assertRaises(ValidationError):
            QueryPlan(
                semantic_query="player report context",
                keyword_query="player report context",
                intent="projection",
                player_mentions=["Drake London"],
                team_mentions=[],
                soft_team_mentions=[],
                negative_focus=[],
                entity_selectors=[relationship_target],
                context_requests=[],
                season=2026,
                week=None,
                temporal_mode="current",
                start_date=None,
                end_date=None,
                needs_baseline=False,
                evidence_strategy="multiple_documents",
            )

        with self.assertRaises(ValidationError):
            QueryPlan(
                semantic_query="related evidence",
                keyword_query="related evidence",
                intent="projection",
                player_mentions=["Drake London"],
                team_mentions=[],
                soft_team_mentions=[],
                negative_focus=[],
                entity_selectors=[selector],
                context_requests=[
                    ContextRequest(
                        anchor_selector_index=0,
                        relation="dependency",
                        semantic_query="related evidence",
                        keyword_query="related evidence",
                        semantic_qualifiers=[],
                        scope_policy="lookup_entities",
                        structured_lookups=[
                            PlayerSeasonStatsLookup(
                                lookup_id="facts-only",
                                purpose="reranker_context",
                                operation="player_season_stats",
                                season=2025,
                                season_type="REG",
                                fields=["fantasy_points_ppr"],
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

    def test_reranker_context_lookup_does_not_change_retrieval_terms(self) -> None:
        enrichment = TargetEnrichment(
            selector_index=0,
            lookups=[
                LookupExecution(
                    lookup_id="season-facts",
                    operation="player_season_stats",
                    purpose="reranker_context",
                    status="resolved",
                    entities=[],
                    teams=[],
                    query_terms=["must not enter retrieval"],
                    facts=[{"fantasy_points_ppr": 200}],
                    error=None,
                )
            ],
        )

        self.assertEqual(enrichment.query_terms, [])


if __name__ == "__main__":
    unittest.main()
