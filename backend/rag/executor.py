"""Apply resolved query plans to local report retrieval deterministically."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .planner import QueryPlan
from .resolver import ResolutionResult, ResolvedEntity
from .store import LocalRAGStore, SearchHit


@dataclass(frozen=True)
class ExecutionResult:
    hits: list[SearchHit]
    resolution: ResolutionResult
    keyword_query: str
    vector_query: str
    linked_document_entities: int
    strategy: str


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
        if existing is None or hit.score > existing.score:
            unique[hit.document.id] = hit
    return list(unique.values())


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
            for item in selector_result.selector.filters:
                if item.field == "team":
                    teams.extend(item.values)
        return list(dict.fromkeys(teams))

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
    ) -> ExecutionResult:
        players = resolution.players
        teams = self._resolved_team_filters(resolution)
        linked = self.store.link_player_entities(players) if players else 0

        names = [player.display_name for player in players]
        keyword_query = _append_terms(plan.keyword_query, [*names, *teams])
        vector_query = _append_terms(plan.semantic_query, names)

        base_filters = {
            key: value
            for key, value in (manual_filters or {}).items()
            if value is not None
        }
        if plan.season is not None and "season" not in base_filters:
            base_filters["season"] = plan.season
        if teams and "team" not in base_filters:
            base_filters["teams"] = teams

        per_entity: list[ResolvedEntity] = []
        if plan.evidence_strategy == "per_entity":
            if len(players) > 1:
                per_entity = players
            elif len(resolution.teams) > 1:
                per_entity = resolution.teams

        if per_entity:
            per_limit = max(2, math.ceil(limit / len(per_entity)) + 1)
            selected: list[SearchHit] = []
            for entity in per_entity:
                entity_filters = dict(base_filters)
                entity_keyword = keyword_query
                entity_vector = vector_query
                if entity.entity_type == "player":
                    entity_filters["player_ids"] = [entity.entity_id]
                    entity_keyword = _append_terms(plan.keyword_query, [entity.display_name])
                    entity_vector = _append_terms(plan.semantic_query, [entity.display_name])
                else:
                    entity_filters["teams"] = [entity.entity_id]
                selected.extend(
                    self._search_temporal(
                        original_query,
                        plan,
                        keyword_query=entity_keyword,
                        vector_query=entity_vector,
                        mode=mode,
                        limit=per_limit,
                        embedding_model=embedding_model,
                        filters=entity_filters,
                    )
                )
            hits = _deduplicate_hits(selected)
            if plan.temporal_mode in {"latest", "current"}:
                hits = _newest_first(hits)
            strategy = "per_entity"
        else:
            if players and "player" not in base_filters:
                base_filters["player_ids"] = [player.entity_id for player in players]
            hits = self._search_temporal(
                original_query,
                plan,
                keyword_query=keyword_query,
                vector_query=vector_query,
                mode=mode,
                limit=limit,
                embedding_model=embedding_model,
                filters=base_filters,
            )
            strategy = "resolved"

        hits = self._negative_focus_filter(hits, plan, players)[:limit]
        return ExecutionResult(
            hits=hits,
            resolution=resolution,
            keyword_query=keyword_query,
            vector_query=vector_query,
            linked_document_entities=linked,
            strategy=strategy,
        )
