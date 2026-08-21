"""One model-facing boundary around the complete report retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_CONTEXT_PLANNER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_RERANK_CANDIDATES,
)
from .planning.context_planner import ContextPlanner
from .planning.planner import QueryPlanner
from .planning.enrichment import StructuredEnrichment, StructuredLookupExecutor
from .planning.lookups import ContextScopePolicy, LookupPurpose
from .planning.resolver import (
    EntityResolver,
    ResolutionResult,
    validate_resolution_bounds,
)
from .planning.router import EscalationRouter
from .retrieval.executor import QueryPlanExecutor
from .retrieval.reranker import MAX_RERANK_CANDIDATES, ReportReranker
from .retrieval.store import LocalRAGStore, SearchHit


class ReportSearchStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True)
class ReportEvidence:
    title: str
    source: str
    published_at: str
    url: str
    excerpt: str
    players: tuple[str, ...]
    teams: tuple[str, ...]
    relationship: str
    relevance_score: int
    relevance_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "source": self.source,
            "published_at": self.published_at,
            "url": self.url,
            "excerpt": self.excerpt,
            "players": list(self.players),
            "teams": list(self.teams),
            "relationship": self.relationship,
            "relevance_reason": self.relevance_reason,
        }


@dataclass(frozen=True)
class ReportSearchResult:
    status: ReportSearchStatus
    evidence_sufficiency: str
    sufficiency_reason: str
    unresolved_constraints: tuple[str, ...]
    reports: tuple[ReportEvidence, ...]
    telemetry: dict[str, Any]

    def agent_output(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "evidence_sufficiency": self.evidence_sufficiency,
            "sufficiency_reason": self.sufficiency_reason,
            "unresolved_constraints": list(self.unresolved_constraints),
            "reports": [report.as_dict() for report in self.reports],
        }


def _unresolved_constraints(
    resolution: ResolutionResult,
    enrichment: StructuredEnrichment,
) -> tuple[str, ...]:
    issues: list[str] = []
    for item in resolution.selectors:
        selector = item.selector
        reference = getattr(selector, "reference_text", "") or ", ".join(
            selector.names
        )
        is_specific_player = selector.entity_type == "player" and bool(
            selector.names
        )
        if item.status == "unresolved" and (reference or is_specific_player):
            issues.append(
                f"Unresolved {selector.entity_type}: {reference or 'unknown'}"
            )
        elif item.status == "multiple" and is_specific_player:
            issues.append(f"Ambiguous player: {reference}")
        issues.extend(item.unresolved_filters)
    for context in resolution.contexts:
        issues.extend(
            f"Context {context.request_index}: {message}"
            for message in context.unresolved
        )
        context_enrichment = enrichment.context(context.request_index)
        if context_enrichment is None:
            continue
        for lookup in context_enrichment.lookups:
            if (
                lookup.status != "resolved"
                and lookup.purpose == LookupPurpose.RESOLVE_RELATIONSHIP
            ):
                issues.append(
                    f"Context {context.request_index}: required structured "
                    f"lookup {lookup.lookup_id} returned {lookup.status}"
                )
        if (
            context.request.scope_policy == ContextScopePolicy.LOOKUP_ENTITIES
            and not context_enrichment.scope_entities
        ):
            issues.append(
                f"Context {context.request_index}: lookup scope resolved no entities"
            )
        if (
            context.request.scope_policy
            == ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS
            and not [
                team
                for team in context_enrichment.scope_teams
                if team not in context.teams
            ]
        ):
            issues.append(
                f"Context {context.request_index}: lookup resolved no related team"
            )
    issues.extend(f"Structured lookup: {error}" for error in enrichment.errors)
    return tuple(dict.fromkeys(issues))


def _judgment_value(value: object) -> str:
    return str(getattr(value, "value", value))


class ReportRetrievalPipeline:
    """Run planning, grounding, hybrid retrieval, reranking, and gating.

    Storage is injected behind ``store`` so the current local SQLite index can
    later be replaced by a Supabase/pgvector implementation without changing the
    model-facing ``search_reports`` tool.
    """

    def __init__(
        self,
        *,
        store: LocalRAGStore | None = None,
        planner: QueryPlanner | None = None,
        context_planner: ContextPlanner | None = None,
        resolver: EntityResolver | None = None,
        identity_router: EscalationRouter | None = None,
        lookup_executor: StructuredLookupExecutor | None = None,
        executor: QueryPlanExecutor | None = None,
        reranker: ReportReranker | None = None,
        cache_path: Path = DEFAULT_INDEX_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        candidate_limit: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        self.store = store or LocalRAGStore(cache_path)
        self.planner = planner or QueryPlanner(index_path=cache_path)
        self.context_planner = context_planner or ContextPlanner(
            index_path=cache_path,
            model=DEFAULT_CONTEXT_PLANNER_MODEL,
        )
        self.resolver = resolver or EntityResolver()
        self.identity_router = identity_router or EscalationRouter(
            index_path=cache_path
        )
        self.lookup_executor = lookup_executor or StructuredLookupExecutor()
        self.executor = executor or QueryPlanExecutor(self.store)
        self.reranker = reranker or ReportReranker(index_path=cache_path)
        self.embedding_model = embedding_model
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        use_cache: bool = True,
    ) -> ReportSearchResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Report search query must not be empty")
        if not 1 <= limit <= 5:
            raise ValueError("Report search limit must be between 1 and 5")
        if self.candidate_limit < limit:
            raise ValueError("Report candidate limit must be at least result limit")
        if self.candidate_limit > MAX_RERANK_CANDIDATES:
            raise ValueError(
                f"Report candidate limit cannot exceed {MAX_RERANK_CANDIDATES}"
            )

        plan_result = self.planner.plan(
            normalized_query,
            use_cache=use_cache,
        )
        active_plan = plan_result.plan
        resolution = self.resolver.resolve(active_plan)
        validate_resolution_bounds(resolution)

        routing = self.identity_router.route(
            normalized_query,
            active_plan,
            resolution,
            resolver=self.resolver,
            use_cache=use_cache,
        )
        active_plan = routing.plan
        resolution = routing.resolution
        validate_resolution_bounds(resolution)

        context_result = self.context_planner.expand(
            normalized_query,
            active_plan,
            resolution,
            use_cache=use_cache,
        )
        active_plan = context_result.plan
        # Terra may add context branches but cannot change direct selectors.
        # Resolve the merged plan so those new branch anchors and scopes are
        # grounded before any structured lookup or report retrieval executes.
        resolution = self.resolver.resolve(active_plan)
        validate_resolution_bounds(resolution)
        enrichment = self.lookup_executor.execute(active_plan, resolution)

        execution = self.executor.execute(
            normalized_query,
            active_plan,
            resolution,
            mode="hybrid",
            limit=self.candidate_limit,
            embedding_model=self.embedding_model,
            enrichment=enrichment,
        )
        rerank = self.reranker.rerank(
            normalized_query,
            active_plan,
            resolution,
            execution.hits,
            limit=limit,
            use_cache=use_cache,
            enrichment=enrichment,
        )

        judgment_by_document = {
            item.hit.document.id: item.judgment
            for item in rerank.ranked_candidates
        }
        evidence: list[ReportEvidence] = []
        for hit in rerank.hits:
            judgment = judgment_by_document.get(hit.document.id)
            if judgment is None:
                continue
            relationship = _judgment_value(judgment.relationship)
            if relationship == "irrelevant":
                continue
            evidence.append(self._evidence(hit, judgment))

        sufficiency = _judgment_value(rerank.evidence_sufficiency)
        if rerank.error is not None:
            status = ReportSearchStatus.NO_EVIDENCE
            evidence = []
            sufficiency_reason = (
                "Report reranking failed, so unvalidated retrieval results were "
                "not supplied to the answering agent."
            )
        elif sufficiency == "weak" or not evidence:
            status = ReportSearchStatus.NO_EVIDENCE
            evidence = []
            sufficiency_reason = rerank.sufficiency_reason
        elif sufficiency == "partial":
            status = ReportSearchStatus.PARTIAL
            sufficiency_reason = rerank.sufficiency_reason
        else:
            status = ReportSearchStatus.READY
            sufficiency_reason = rerank.sufficiency_reason

        escalation = routing.event
        telemetry = {
            "planner": {
                "model": plan_result.model,
                "cached": plan_result.cached,
                "input_tokens": plan_result.input_tokens,
                "cached_input_tokens": plan_result.cached_input_tokens,
                "output_tokens": plan_result.output_tokens,
                "attempts": plan_result.attempts,
                "retried": plan_result.retried,
                "retry_reason": plan_result.retry_reason,
            },
            "context_planner": {
                "model": context_result.model,
                "cached": context_result.cached,
                "context_needed": context_result.context_plan.context_needed,
                "rationale": context_result.context_plan.rationale,
                "branches": len(context_result.context_plan.context_requests),
                "input_tokens": context_result.input_tokens,
                "cached_input_tokens": context_result.cached_input_tokens,
                "output_tokens": context_result.output_tokens,
                "attempts": context_result.attempts,
                "retried": context_result.retried,
                "retry_reason": context_result.retry_reason,
            },
            "identity": {
                "model": escalation.model,
                "triggered": escalation.triggered,
                "cached": escalation.cache_hit,
                "input_tokens": escalation.input_tokens,
                "cached_input_tokens": escalation.cached_input_tokens,
                "output_tokens": escalation.output_tokens,
                "impactful": escalation.impactful,
            },
            "retrieval": {
                "mode": "hybrid",
                "strategy": execution.strategy,
                "candidates": len(execution.hits),
                "branch_candidates": execution.branch_candidates,
                "linked_document_entities": execution.linked_document_entities,
            },
            "structured_enrichment": {
                "lookups": sum(
                    len(group.lookups)
                    for group in [*enrichment.targets, *enrichment.contexts]
                ),
                "resolved": sum(
                    lookup.status == "resolved"
                    for group in [*enrichment.targets, *enrichment.contexts]
                    for lookup in group.lookups
                ),
                "empty": sum(
                    lookup.status == "empty"
                    for group in [*enrichment.targets, *enrichment.contexts]
                    for lookup in group.lookups
                ),
                "errors": enrichment.errors,
                "results": [
                    {
                        "lookup_id": lookup.lookup_id,
                        "operation": lookup.operation,
                        "purpose": lookup.purpose,
                        "status": lookup.status,
                        "entities": len(lookup.entities),
                        "teams": lookup.teams,
                        "fallback_used": lookup.fallback_used,
                        "fallback_reason": lookup.fallback_reason,
                    }
                    for group in [*enrichment.targets, *enrichment.contexts]
                    for lookup in group.lookups
                ],
            },
            "reranker": {
                "model": rerank.model,
                "cached": rerank.cached,
                "api_called": rerank.api_called,
                "input_tokens": rerank.input_tokens,
                "cached_input_tokens": rerank.cached_input_tokens,
                "output_tokens": rerank.output_tokens,
                "latency_ms": rerank.latency_ms,
                "ranking_changed": rerank.ranking_changed,
                "error": rerank.error,
            },
        }
        return ReportSearchResult(
            status=status,
            evidence_sufficiency=sufficiency,
            sufficiency_reason=sufficiency_reason,
            unresolved_constraints=_unresolved_constraints(
                resolution,
                enrichment,
            ),
            reports=tuple(evidence),
            telemetry=telemetry,
        )

    @staticmethod
    def _evidence(hit: SearchHit, judgment) -> ReportEvidence:
        snippet = hit.document.snippet
        excerpt = f"{snippet[:700]}{'...' if len(snippet) > 700 else ''}"
        return ReportEvidence(
            title=hit.document.title,
            source=hit.document.source,
            published_at=hit.document.published_at,
            url=hit.document.url,
            excerpt=excerpt,
            players=hit.document.players,
            teams=hit.document.teams,
            relationship=_judgment_value(judgment.relationship),
            relevance_score=judgment.relevance_score,
            relevance_reason=judgment.reason,
        )
