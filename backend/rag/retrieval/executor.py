"""Apply resolved query plans to local report retrieval deterministically."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ..planning.enrichment import StructuredEnrichment
from ..planning.lookups import ContextScopePolicy, LookupPurpose
from ..planning.planner import QueryPlan
from ..planning.resolver import ResolutionResult, ResolvedEntity, SelectorResolution
from .store import LocalRAGStore, SearchHit


@dataclass(frozen=True)
class ExecutionResult:
    hits: list[SearchHit]
    resolution: ResolutionResult
    keyword_query: str
    vector_query: str
    linked_document_entities: int
    strategy: str
    branch_candidates: dict[str, int] = field(default_factory=dict)
    enrichment: StructuredEnrichment = field(default_factory=StructuredEnrichment)


def _append_terms(query: str, terms: list[str]) -> str:
    result = query.strip()
    normalized = result.casefold()
    for term in terms:
        if term.casefold() in normalized:
            continue
        result = f"{result} {term}".strip()
        normalized = result.casefold()
    return result


def _deduplicate_hits(hits: list[SearchHit]) -> list[SearchHit]:
    unique: dict[str, SearchHit] = {}
    for hit in hits:
        existing = unique.get(hit.document.id)
        if existing is None:
            unique[hit.document.id] = hit
            continue
        scopes = tuple(
            dict.fromkeys([*existing.retrieval_scopes, *hit.retrieval_scopes])
        )
        winner = hit if hit.score > existing.score else existing
        unique[hit.document.id] = replace(winner, retrieval_scopes=scopes)
    return list(unique.values())


def _tag_hits(hits: list[SearchHit], scope: str) -> list[SearchHit]:
    return [
        replace(
            hit,
            retrieval_scopes=tuple(
                dict.fromkeys([*hit.retrieval_scopes, scope])
            ),
        )
        for hit in hits
    ]


def _merge_branches(
    branches: list[tuple[str, list[SearchHit]]],
    *,
    limit: int,
) -> list[SearchHit]:
    """Preserve branch coverage, then fill the pool by retrieval score."""
    tagged = [(name, _tag_hits(hits, name)) for name, hits in branches]
    selected: list[SearchHit] = []
    for _, hits in tagged:
        if hits:
            selected.append(hits[0])
    remaining = sorted(
        (hit for _, hits in tagged for hit in hits[1:]),
        key=lambda hit: hit.score,
        reverse=True,
    )
    return _deduplicate_hits([*selected, *remaining])[:limit]


def _newest_first(hits: list[SearchHit]) -> list[SearchHit]:
    return sorted(
        hits,
        key=lambda hit: (
            LocalRAGStore._published_date(hit.document.published_at),
            hit.score,
        ),
        reverse=True,
    )


class QueryPlanExecutor:
    def __init__(self, store: LocalRAGStore) -> None:
        self.store = store

    @staticmethod
    def _resolved_team_filters(resolution: ResolutionResult) -> list[str]:
        teams = [team.entity_id for team in resolution.teams]
        for selector_result in resolution.selectors:
            # Only prompt-grounded hard filters may constrain report lookup.
            # Soft filters remain in the plan for diagnostics/reranking.
            for item in selector_result.selector.hard_filters:
                if item.field == "team":
                    teams.extend(item.values)
        return list(dict.fromkeys(teams))

    @staticmethod
    def _selector_filters(
        selector_result: SelectorResolution,
    ) -> dict[str, object]:
        """Build one target's filters without flattening unrelated selectors."""
        players = [
            match.entity_id
            for match in selector_result.matches
            if match.entity_type == "player"
        ]
        teams = [
            match.entity_id
            for match in selector_result.matches
            if match.entity_type == "team"
        ]
        for item in selector_result.selector.hard_filters:
            if item.field == "team":
                teams.extend(str(value) for value in item.values)

        filters: dict[str, object] = {}
        if players:
            filters["player_ids"] = list(dict.fromkeys(players))
        if teams:
            filters["teams"] = list(dict.fromkeys(teams))
        return filters

    @staticmethod
    def _merge_branch_filters(
        base: dict[str, object],
        branch: dict[str, object],
    ) -> dict[str, object] | None:
        """Combine explicit manual filters with one branch's local scope."""
        merged = dict(base)
        for key, value in branch.items():
            if key not in merged:
                merged[key] = value
                continue
            if key not in {"player_ids", "players", "teams"}:
                if merged[key] != value:
                    return None
                continue
            existing = set(merged[key])
            selected = existing.intersection(value)
            if not selected:
                return None
            merged[key] = sorted(selected)
        return merged

    def _search_once(
        self,
        original_query: str,
        *,
        keyword_query: str,
        vector_query: str,
        mode: str,
        limit: int,
        embedding_model: str,
        filters: dict[str, object],
    ) -> list[SearchHit]:
        search_filters = dict(filters)
        if mode != "keyword":
            search_filters["embedding_model"] = embedding_model
        return self.store.search(
            original_query,
            mode=mode,
            limit=limit,
            keyword_query=keyword_query,
            vector_query=vector_query,
            **search_filters,
        )

    def _search_temporal(
        self,
        original_query: str,
        plan: QueryPlan,
        *,
        keyword_query: str,
        vector_query: str,
        mode: str,
        limit: int,
        embedding_model: str,
        filters: dict[str, object],
    ) -> list[SearchHit]:
        pool_size = max(limit * 5, 20)
        temporal_filters = dict(filters)

        if plan.temporal_mode == "after" and plan.start_date:
            temporal_filters["published_after"] = plan.start_date
            if plan.end_date:
                temporal_filters["published_to"] = plan.end_date
            newer = self._search_once(
                original_query,
                keyword_query=keyword_query,
                vector_query=vector_query,
                mode=mode,
                limit=pool_size,
                embedding_model=embedding_model,
                filters=temporal_filters,
            )
            newer = _newest_first(newer)
            if not plan.needs_baseline:
                return newer[:limit]

            baseline_filters = dict(filters)
            baseline_filters["published_to"] = plan.start_date
            baseline = self._search_once(
                original_query,
                keyword_query=keyword_query,
                vector_query=vector_query,
                mode=mode,
                limit=pool_size,
                embedding_model=embedding_model,
                filters=baseline_filters,
            )
            return _deduplicate_hits([*newer, *_newest_first(baseline)[:1]])[:limit]

        if plan.temporal_mode == "before":
            boundary = plan.end_date or plan.start_date
            if boundary:
                temporal_filters["published_before"] = boundary
        elif plan.temporal_mode == "between":
            if plan.start_date:
                temporal_filters["published_from"] = plan.start_date
            if plan.end_date:
                temporal_filters["published_to"] = plan.end_date
        else:
            if plan.start_date and plan.temporal_mode == "timeline":
                temporal_filters["published_from"] = plan.start_date
            if plan.end_date:
                temporal_filters["published_to"] = plan.end_date

        hits = self._search_once(
            original_query,
            keyword_query=keyword_query,
            vector_query=vector_query,
            mode=mode,
            limit=pool_size,
            embedding_model=embedding_model,
            filters=temporal_filters,
        )
        if plan.temporal_mode in {"latest", "current", "before"}:
            hits = _newest_first(hits)
        return hits[:limit]

    @staticmethod
    def _negative_focus_filter(
        hits: list[SearchHit],
        plan: QueryPlan,
        positive_players: list[ResolvedEntity],
    ) -> list[SearchHit]:
        if not plan.negative_focus:
            return hits

        negative = {name.casefold() for name in plan.negative_focus}
        positive = {player.display_name.casefold() for player in positive_players}
        preferred: list[SearchHit] = []
        penalized: list[SearchHit] = []
        for hit in hits:
            names = {name.casefold() for name in hit.document.players}
            has_negative = bool(names.intersection(negative))
            has_positive = bool(names.intersection(positive))
            if has_negative and not has_positive:
                penalized.append(hit)
            else:
                preferred.append(hit)
        return [*preferred, *penalized]

    def execute(
        self,
        original_query: str,
        plan: QueryPlan,
        resolution: ResolutionResult,
        *,
        mode: str,
        limit: int,
        embedding_model: str,
        manual_filters: dict[str, object] | None = None,
        enrichment: StructuredEnrichment | None = None,
    ) -> ExecutionResult:
        active_enrichment = enrichment or StructuredEnrichment()
        players = resolution.players
        active_contexts = [
            context
            for context in resolution.contexts
            if context.status == "resolved"
        ]
        enrichment_players = [
            entity
            for group in [*active_enrichment.targets, *active_enrichment.contexts]
            for lookup in group.lookups
            for entity in lookup.entities
            if entity.entity_type == "player"
        ]
        linkable_players = [
            *players,
            *enrichment_players,
        ]
        linked = (
            self.store.link_player_entities(linkable_players)
            if linkable_players
            else 0
        )

        base_filters = {
            key: value
            for key, value in (manual_filters or {}).items()
            if value is not None
        }
        if plan.season is not None and "season" not in base_filters:
            base_filters["season"] = plan.season

        target_results = [
            item
            for item in resolution.selectors
            if item.matches and item.status in {"resolved", "multiple"}
        ]
        total_branches = max(1, len(target_results) + len(active_contexts))
        branch_limit = max(3, math.ceil(limit / total_branches) + 2)
        branches: list[tuple[str, list[SearchHit]]] = []
        branch_candidates: dict[str, int] = {}

        for selector_result in target_results:
            scope = f"target:{selector_result.selector_index}"
            local_filters = self._selector_filters(selector_result)
            filters = self._merge_branch_filters(base_filters, local_filters)
            if filters is None:
                branch_candidates[scope] = 0
                continue
            target_enrichment = active_enrichment.target(
                selector_result.selector_index
            )
            lookup_terms = (
                target_enrichment.query_terms
                if target_enrichment is not None
                else []
            )
            is_player_group = (
                selector_result.selector.entity_type == "player"
                and not selector_result.selector.names
            )
            entity_terms = (
                []
                if is_player_group
                else [match.display_name for match in selector_result.matches]
            )
            team_terms = [
                match.entity_id
                for match in selector_result.matches
                if match.entity_type == "team"
            ]
            keyword_query = _append_terms(
                plan.keyword_query,
                [*entity_terms, *team_terms, *lookup_terms],
            )
            vector_query = _append_terms(
                plan.semantic_query,
                [*entity_terms, *lookup_terms],
            )
            hits = self._search_temporal(
                original_query,
                plan,
                keyword_query=keyword_query,
                vector_query=vector_query,
                mode=mode,
                limit=branch_limit,
                embedding_model=embedding_model,
                filters=filters,
            )
            hits = self._negative_focus_filter(hits, plan, players)
            branches.append((scope, hits))
            branch_candidates[scope] = len(hits)

        if not target_results and not plan.entity_selectors:
            scope = "target:semantic"
            hits = self._search_temporal(
                original_query,
                plan,
                keyword_query=plan.keyword_query,
                vector_query=plan.semantic_query,
                mode=mode,
                limit=branch_limit,
                embedding_model=embedding_model,
                filters=base_filters,
            )
            hits = self._negative_focus_filter(hits, plan, players)
            branches.append((scope, hits))
            branch_candidates[scope] = len(hits)

        for context in active_contexts:
            scope = f"context:{context.request_index}"
            context_enrichment = active_enrichment.context(
                context.request_index
            )
            required_relationship_ids = {
                lookup.lookup_id
                for lookup in context.request.structured_lookups
                if lookup.purpose == LookupPurpose.RESOLVE_RELATIONSHIP
            }
            relationship_lookups = {
                lookup.lookup_id: lookup
                for lookup in (
                    context_enrichment.lookups
                    if context_enrichment is not None
                    else []
                )
                if lookup.purpose == LookupPurpose.RESOLVE_RELATIONSHIP
            }
            if any(
                lookup_id not in relationship_lookups
                or relationship_lookups[lookup_id].status != "resolved"
                for lookup_id in required_relationship_ids
            ):
                # A declared relationship did not ground. Never turn that
                # failure into a broader search.
                branch_candidates[scope] = 0
                continue
            lookup_entities = (
                context_enrichment.scope_entities
                if context_enrichment is not None
                else []
            )
            lookup_teams = (
                context_enrichment.scope_teams
                if context_enrichment is not None
                else []
            )
            relationship_teams = list(
                dict.fromkeys(
                    team
                    for lookup in (
                        context_enrichment.lookups
                        if context_enrichment is not None
                        else []
                    )
                    if lookup.status == "resolved"
                    and lookup.purpose == LookupPurpose.RESOLVE_RELATIONSHIP
                    for team in lookup.teams
                )
            )
            new_relationship_teams = [
                team for team in relationship_teams if team not in context.teams
            ]
            scope_filters: dict[str, object] = {}
            policy = context.request.scope_policy
            if new_relationship_teams:
                # A resolved relationship is executable provenance, not merely
                # query vocabulary. Always consume newly grounded teams so a
                # model-selected anchor-only policy cannot discard an opponent
                # or counterparty that the lookup was requested to discover.
                scope_filters["teams"] = list(
                    dict.fromkeys(
                        [*context.teams, *new_relationship_teams]
                    )
                )
            elif policy == ContextScopePolicy.ANCHOR_TEAMS:
                if context.teams:
                    scope_filters["teams"] = context.teams
            elif policy == ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS:
                expanded_teams = [
                    team for team in lookup_teams if team not in context.teams
                ]
                if not expanded_teams:
                    branch_candidates[scope] = 0
                    continue
                teams = list(dict.fromkeys([*context.teams, *expanded_teams]))
                if teams:
                    scope_filters["teams"] = teams
            elif policy == ContextScopePolicy.LOOKUP_ENTITIES:
                selected_players = [
                    entity.entity_id
                    for entity in lookup_entities
                    if entity.entity_type == "player"
                ]
                selected_teams = [
                    entity.entity_id
                    for entity in lookup_entities
                    if entity.entity_type == "team"
                ]
                if selected_players:
                    scope_filters["player_ids"] = selected_players
                elif selected_teams:
                    scope_filters["teams"] = selected_teams
                if not scope_filters:
                    branch_candidates[scope] = 0
                    continue
            context_filters = self._merge_branch_filters(
                base_filters,
                scope_filters,
            )
            if context_filters is None:
                branch_candidates[scope] = 0
                continue

            lookup_terms = (
                context_enrichment.query_terms
                if context_enrichment is not None
                else []
            )
            context_keyword = _append_terms(
                context.request.keyword_query,
                [*lookup_terms, *context.teams],
            )
            context_vector = _append_terms(
                context.request.semantic_query,
                lookup_terms,
            )
            context_hits = self._search_temporal(
                original_query,
                plan,
                keyword_query=context_keyword,
                vector_query=context_vector,
                mode=mode,
                limit=branch_limit,
                embedding_model=embedding_model,
                filters=context_filters,
            )
            context_hits = self._negative_focus_filter(
                context_hits,
                plan,
                players,
            )
            branches.append((scope, context_hits))
            branch_candidates[scope] = len(context_hits)

        if branches:
            hits = _merge_branches(branches, limit=limit)
            has_target = any(name.startswith("target:") for name, _ in branches)
            has_context = any(name.startswith("context:") for name, _ in branches)
            if has_target and has_context:
                strategy = "resolved+context"
            elif has_target:
                strategy = "resolved"
            else:
                strategy = "context"
        else:
            hits = []
            strategy = "unresolved"

        names = [player.display_name for player in players]
        teams = self._resolved_team_filters(resolution)
        keyword_query = _append_terms(plan.keyword_query, [*names, *teams])
        vector_query = _append_terms(plan.semantic_query, names)

        return ExecutionResult(
            hits=hits,
            resolution=resolution,
            keyword_query=keyword_query,
            vector_query=vector_query,
            linked_document_entities=linked,
            strategy=strategy,
            branch_candidates=branch_candidates,
            enrichment=active_enrichment,
        )
