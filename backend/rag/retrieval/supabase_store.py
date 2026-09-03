"""Supabase/pgvector implementation of the report retrieval store."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from supabase import Client

from database.client import get_supabase_client

from ..config import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_PATH
from ..documents import ReportDocument
from .embeddings import embed_texts
from .store import LocalRAGStore, SearchHit


EMBEDDING_DIMENSIONS = 1536
MAX_QUERY_EMBEDDING_CACHE = 256
EmbeddingFunction = Callable[..., list[list[float]]]


class SupabaseRAGStore(LocalRAGStore):
    """Search report chunks through service-role-only Supabase RPCs.

    The subclass relationship preserves the existing executor interface while
    every persistence and retrieval operation in this class is remote. The
    inherited SQLite implementation is never initialized or called.
    """

    def __init__(
        self,
        *,
        client: Client | None = None,
        embedder: EmbeddingFunction = embed_texts,
        cache_path: Path = DEFAULT_INDEX_PATH,
    ) -> None:
        self.client = client or get_supabase_client()
        self.embedder = embedder
        # Planner, identity, and reranker caches are process-local. Keeping this
        # attribute maintains the pipeline's existing cache contract.
        self.index_path = Path(cache_path)
        self._query_embeddings: dict[tuple[str, str], list[float]] = {}

    @staticmethod
    def _filter_params(filters: dict[str, object]) -> dict[str, object]:
        player_names: list[str] = []
        if filters.get("player"):
            player_names.append(str(filters["player"]))
        if filters.get("players"):
            player_names.extend(str(value) for value in filters["players"])

        teams: list[str] = []
        if filters.get("team"):
            teams.append(str(filters["team"]))
        if filters.get("teams"):
            teams.extend(str(value) for value in filters["teams"])

        player_ids = [
            str(value) for value in (filters.get("player_ids") or [])
        ]
        return {
            "filter_season": filters.get("season"),
            "filter_teams": list(dict.fromkeys(teams)) or None,
            "filter_player_ids": list(dict.fromkeys(player_ids)) or None,
            "filter_player_names": list(dict.fromkeys(player_names)) or None,
            "filter_source": filters.get("source"),
            "filter_document_type": filters.get("document_type"),
            "filter_storyline": filters.get("storyline"),
            "published_after": filters.get("published_after"),
            "published_before": filters.get("published_before"),
            "published_from": filters.get("published_from"),
            "published_to": filters.get("published_to"),
        }

    @staticmethod
    def _row_to_document(row: dict[str, Any]) -> ReportDocument:
        report_id = str(row["report_id"])
        season = row.get("season")
        return ReportDocument(
            id=report_id,
            title=str(row["title"]),
            source=str(row["source"]),
            url=str(row["source_url"]),
            author=None if row.get("author") is None else str(row["author"]),
            published_at=str(row["published_at"]),
            fetched_at=str(row["fetched_at"]),
            players=tuple(str(value) for value in (row.get("player_names") or [])),
            teams=tuple(str(value) for value in (row.get("teams") or [])),
            season=None if season is None else int(season),
            document_type=str(row["document_type"]),
            storyline=str(row.get("storyline") or ""),
            content_mode=str(row["content_mode"]),
            body=str(row["content"]),
            source_path=Path("supabase") / report_id,
            player_ids=tuple(
                str(value) for value in (row.get("player_ids") or [])
            ),
        )

    def _query_embedding(self, query: str, model: str) -> list[float]:
        key = (query.strip(), model)
        cached = self._query_embeddings.get(key)
        if cached is not None:
            return cached
        if not model.startswith("text-embedding-3-"):
            raise ValueError(
                "Supabase report vectors use 1536 dimensions; configure a "
                "text-embedding-3 model so that dimension can be requested."
            )
        values = self.embedder(
            [query],
            model=model,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        if len(values) != 1 or len(values[0]) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Embedding API returned an unexpected query-vector shape"
            )
        if len(self._query_embeddings) >= MAX_QUERY_EMBEDDING_CACHE:
            self._query_embeddings.pop(next(iter(self._query_embeddings)))
        self._query_embeddings[key] = values[0]
        return values[0]

    def link_player_entities(self, players: list[object]) -> int:
        """Ingestion owns report-player links; retrieval never mutates them."""
        return 0

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        **filters: object,
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Search query must contain at least one word or number")
        params = {
            # Match the local FTS contract: discard common query glue and OR
            # the remaining terms. Passing a planner-expanded sentence directly
            # to websearch_to_tsquery would require nearly every term and can
            # silently remove the keyword half of hybrid retrieval.
            "query_text": self._fts_query(query),
            "match_count": min(max(limit, 1), 100),
            **self._filter_params(filters),
        }
        response = self.client.rpc("search_report_chunks_v2", params).execute()
        return [
            SearchHit(
                document=self._row_to_document(row),
                score=float(row["keyword_rank"]),
                method="keyword",
                keyword_rank=rank,
            )
            for rank, row in enumerate(response.data or [], start=1)
        ]

    def vector_search(
        self,
        query: str,
        limit: int = 5,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        **filters: object,
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Search query must contain at least one word or number")
        embedding = self._query_embedding(query, embedding_model)
        params = {
            "query_embedding": embedding,
            "match_threshold": 0,
            "match_count": min(max(limit, 1), 100),
            "filter_embedding_model": embedding_model,
            **self._filter_params(filters),
        }
        response = self.client.rpc("match_report_chunks_v2", params).execute()
        return [
            SearchHit(
                document=self._row_to_document(row),
                score=float(row["similarity"]),
                method="vector",
                vector_rank=rank,
            )
            for rank, row in enumerate(response.data or [], start=1)
        ]

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        *,
        keyword_query: str | None = None,
        vector_query: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        **filters: object,
    ) -> list[SearchHit]:
        pool_size = max(limit * 4, 20)
        keyword_hits = self.keyword_search(
            keyword_query or query,
            limit=pool_size,
            **filters,
        )
        vector_hits = self.vector_search(
            vector_query or query,
            limit=pool_size,
            embedding_model=embedding_model,
            **filters,
        )

        documents: dict[str, ReportDocument] = {}
        scores: dict[str, float] = {}
        keyword_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}
        for rank, hit in enumerate(keyword_hits, start=1):
            report_id = hit.document.id
            documents[report_id] = hit.document
            scores[report_id] = scores.get(report_id, 0.0) + 1 / (60 + rank)
            keyword_ranks[report_id] = rank
        for rank, hit in enumerate(vector_hits, start=1):
            report_id = hit.document.id
            documents[report_id] = hit.document
            scores[report_id] = scores.get(report_id, 0.0) + 1 / (60 + rank)
            vector_ranks[report_id] = rank

        ranked_ids = sorted(
            documents,
            key=lambda report_id: (
                scores[report_id],
                documents[report_id].published_at,
            ),
            reverse=True,
        )
        return [
            SearchHit(
                document=documents[report_id],
                score=scores[report_id],
                method="hybrid",
                keyword_rank=keyword_ranks.get(report_id),
                vector_rank=vector_ranks.get(report_id),
            )
            for report_id in ranked_ids[:limit]
        ]

    def search(
        self,
        query: str,
        mode: str = "keyword",
        limit: int = 5,
        keyword_query: str | None = None,
        vector_query: str | None = None,
        **filters: object,
    ) -> list[SearchHit]:
        if mode == "keyword":
            return self.keyword_search(
                keyword_query or query,
                limit=limit,
                **filters,
            )
        if mode == "vector":
            return self.vector_search(
                vector_query or query,
                limit=limit,
                **filters,
            )
        if mode == "hybrid":
            return self.hybrid_search(
                query,
                limit=limit,
                keyword_query=keyword_query,
                vector_query=vector_query,
                **filters,
            )
        raise ValueError(f"Unsupported search mode: {mode}")

    def status(self) -> dict[str, object]:
        response = self.client.rpc("report_store_status", {}).execute()
        value = response.data or {}
        return dict(value)
