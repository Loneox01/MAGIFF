import json
import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from ..config import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_PATH
from ..documents import ReportDocument
from .embeddings import embed_texts


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "latest",
    "me",
    "most",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "will",
    "with",
}


@dataclass(frozen=True)
class SearchHit:
    document: ReportDocument
    score: float
    method: str
    keyword_rank: int | None = None
    vector_rank: int | None = None
    retrieval_scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexBuildResult:
    document_count: int
    embedded_count: int
    generated_embedding_count: int
    index_path: Path


class LocalRAGStore:
    """Small SQLite keyword index plus local cosine vector search."""

    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH) -> None:
        self.index_path = Path(index_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.index_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                author TEXT,
                published_at TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                players_json TEXT NOT NULL,
                player_ids_json TEXT NOT NULL DEFAULT '[]',
                teams_json TEXT NOT NULL,
                season INTEGER NOT NULL,
                document_type TEXT NOT NULL,
                storyline TEXT NOT NULL,
                content_mode TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_path TEXT NOT NULL,
                embedding_json TEXT,
                embedding_model TEXT
            );

            CREATE TABLE IF NOT EXISTS document_player_links (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                player_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                PRIMARY KEY (document_id, player_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                document_id UNINDEXED,
                title,
                content,
                players,
                teams,
                storyline,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS query_embeddings (
                query_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                PRIMARY KEY (query_hash, model)
            );
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(documents)")
        }
        if "player_ids_json" not in columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN player_ids_json TEXT NOT NULL DEFAULT '[]'"
            )

    def build_index(
        self,
        documents: list[ReportDocument],
        with_embeddings: bool = False,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_batch_size: int = 64,
    ) -> IndexBuildResult:
        if not documents:
            raise ValueError("Cannot build an index with no report documents")

        with self._connect() as connection:
            self._initialize(connection)
            existing = {
                row["id"]: row
                for row in connection.execute(
                    """
                    SELECT id, content_hash, player_ids_json, embedding_json,
                           embedding_model
                    FROM documents
                    """
                )
            }

            current_ids: list[str] = []
            for document in documents:
                current_ids.append(document.id)
                cached = existing.get(document.id)
                content_unchanged = (
                    cached is not None
                    and cached["content_hash"] == document.content_hash
                )
                cached_embedding = (
                    cached["embedding_json"] if content_unchanged else None
                )
                cached_model = cached["embedding_model"] if content_unchanged else None
                cached_player_ids = (
                    json.loads(cached["player_ids_json"])
                    if content_unchanged and cached["player_ids_json"]
                    else []
                )
                player_ids = list(document.player_ids) or cached_player_ids
                if cached is not None and not content_unchanged:
                    connection.execute(
                        "DELETE FROM document_player_links WHERE document_id = ?",
                        (document.id,),
                    )

                connection.execute(
                    """
                    INSERT INTO documents (
                        id, title, source, url, author, published_at, fetched_at,
                        players_json, player_ids_json, teams_json, season,
                        document_type, storyline,
                        content_mode, content, content_hash, source_path,
                        embedding_json, embedding_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        source = excluded.source,
                        url = excluded.url,
                        author = excluded.author,
                        published_at = excluded.published_at,
                        fetched_at = excluded.fetched_at,
                        players_json = excluded.players_json,
                        player_ids_json = excluded.player_ids_json,
                        teams_json = excluded.teams_json,
                        season = excluded.season,
                        document_type = excluded.document_type,
                        storyline = excluded.storyline,
                        content_mode = excluded.content_mode,
                        content = excluded.content,
                        content_hash = excluded.content_hash,
                        source_path = excluded.source_path,
                        embedding_json = excluded.embedding_json,
                        embedding_model = excluded.embedding_model
                    """,
                    (
                        document.id,
                        document.title,
                        document.source,
                        document.url,
                        document.author,
                        document.published_at,
                        document.fetched_at,
                        json.dumps(document.players),
                        json.dumps(player_ids),
                        json.dumps(document.teams),
                        document.season,
                        document.document_type,
                        document.storyline,
                        document.content_mode,
                        document.body,
                        document.content_hash,
                        str(document.source_path),
                        cached_embedding,
                        cached_model,
                    ),
                )

            placeholders = ",".join("?" for _ in current_ids)
            connection.execute(
                f"DELETE FROM documents WHERE id NOT IN ({placeholders})",
                current_ids,
            )

            connection.execute("DELETE FROM documents_fts")
            connection.executemany(
                """
                INSERT INTO documents_fts (
                    document_id, title, content, players, teams, storyline
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document.id,
                        document.title,
                        document.body,
                        " ".join(document.players),
                        " ".join(document.teams),
                        document.storyline.replace("_", " "),
                    )
                    for document in documents
                ],
            )

            generated_count = 0
            if with_embeddings:
                missing = list(
                    connection.execute(
                        """
                        SELECT id, title, source, published_at, players_json,
                               teams_json, document_type, storyline, content
                        FROM documents
                        WHERE embedding_json IS NULL OR embedding_model != ?
                        ORDER BY id
                        """,
                        (embedding_model,),
                    )
                )

                for start in range(0, len(missing), embedding_batch_size):
                    batch = missing[start : start + embedding_batch_size]
                    texts = [self._embedding_text_from_row(row) for row in batch]
                    embeddings = embed_texts(texts, model=embedding_model)
                    if len(embeddings) != len(batch):
                        raise RuntimeError("Embedding API returned an unexpected row count")

                    connection.executemany(
                        """
                        UPDATE documents
                        SET embedding_json = ?, embedding_model = ?
                        WHERE id = ?
                        """,
                        [
                            (json.dumps(embedding), embedding_model, row["id"])
                            for row, embedding in zip(batch, embeddings, strict=True)
                        ],
                    )
                    generated_count += len(batch)
                    connection.commit()

            embedded_count = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE embedding_json IS NOT NULL"
            ).fetchone()[0]

        return IndexBuildResult(
            document_count=len(documents),
            embedded_count=embedded_count,
            generated_embedding_count=generated_count,
            index_path=self.index_path,
        )

    @staticmethod
    def _embedding_text_from_row(row: sqlite3.Row) -> str:
        players = ", ".join(json.loads(row["players_json"]))
        teams = ", ".join(json.loads(row["teams_json"]))
        metadata = [
            f"Title: {row['title']}",
            f"Source: {row['source']}",
            f"Published: {row['published_at']}",
            f"Players: {players}",
            f"Teams: {teams}",
            f"Document type: {row['document_type']}",
            f"Storyline: {row['storyline'].replace('_', ' ')}",
        ]
        return "\n".join(metadata) + "\n\n" + row["content"]

    @staticmethod
    def _fts_query(query: str) -> str:
        all_terms = re.findall(r"[A-Za-z0-9]+", query.casefold())
        terms = [term for term in all_terms if term not in STOP_WORDS]
        if not terms:
            terms = all_terms
        if not terms:
            raise ValueError("Search query must contain at least one word or number")
        return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> ReportDocument:
        return ReportDocument(
            id=row["id"],
            title=row["title"],
            source=row["source"],
            url=row["url"],
            author=row["author"],
            published_at=row["published_at"],
            fetched_at=row["fetched_at"],
            players=tuple(json.loads(row["players_json"])),
            teams=tuple(json.loads(row["teams_json"])),
            season=row["season"],
            document_type=row["document_type"],
            storyline=row["storyline"],
            content_mode=row["content_mode"],
            body=row["content"],
            source_path=Path(row["source_path"]),
            player_ids=tuple(json.loads(row["player_ids_json"])),
        )

    @staticmethod
    def _published_date(value: str) -> date:
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date()
        except ValueError:
            return date.fromisoformat(value[:10])

    @staticmethod
    def _matches_filters(
        document: ReportDocument,
        player: str | None,
        players: tuple[str, ...] | list[str] | None,
        player_ids: tuple[str, ...] | list[str] | None,
        team: str | None,
        teams: tuple[str, ...] | list[str] | None,
        source: str | None,
        season: int | None,
        document_type: str | None,
        storyline: str | None,
        published_after: str | None,
        published_before: str | None,
        published_from: str | None,
        published_to: str | None,
    ) -> bool:
        if player and not any(
            player.casefold() in candidate.casefold() for candidate in document.players
        ):
            return False
        if players:
            wanted_players = {item.casefold() for item in players}
            if not wanted_players.intersection(
                candidate.casefold() for candidate in document.players
            ):
                return False
        if player_ids and not set(player_ids).intersection(document.player_ids):
            return False
        if team and team.casefold() not in {item.casefold() for item in document.teams}:
            return False
        if teams:
            wanted_teams = {item.casefold() for item in teams}
            if not wanted_teams.intersection(item.casefold() for item in document.teams):
                return False
        if source and source.casefold() not in document.source.casefold():
            return False
        if season is not None and season != document.season:
            return False
        if document_type and document_type.casefold() != document.document_type.casefold():
            return False
        if storyline and storyline.casefold() != document.storyline.casefold():
            return False
        document_date = LocalRAGStore._published_date(document.published_at)
        if published_after and document_date <= date.fromisoformat(published_after):
            return False
        if published_before and document_date >= date.fromisoformat(published_before):
            return False
        if published_from and document_date < date.fromisoformat(published_from):
            return False
        if published_to and document_date > date.fromisoformat(published_to):
            return False
        return True

    def link_player_entities(self, players: list[object]) -> int:
        """Persist canonical IDs for report metadata names already in the index."""
        linked = 0
        with self._connect() as connection:
            self._initialize(connection)
            rows = list(
                connection.execute(
                    "SELECT id, players_json, player_ids_json FROM documents"
                )
            )
            for row in rows:
                document_names = {
                    name.casefold() for name in json.loads(row["players_json"])
                }
                player_ids = set(json.loads(row["player_ids_json"]))
                changed = False
                for player_entity in players:
                    player_id = str(getattr(player_entity, "entity_id"))
                    display_name = str(getattr(player_entity, "display_name"))
                    if display_name.casefold() not in document_names:
                        continue
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO document_player_links (
                            document_id, player_id, display_name
                        ) VALUES (?, ?, ?)
                        """,
                        (row["id"], player_id, display_name),
                    )
                    if player_id not in player_ids:
                        player_ids.add(player_id)
                        changed = True
                        linked += 1
                if changed:
                    connection.execute(
                        "UPDATE documents SET player_ids_json = ? WHERE id = ?",
                        (json.dumps(sorted(player_ids)), row["id"]),
                    )
        return linked

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        *,
        player: str | None = None,
        players: tuple[str, ...] | list[str] | None = None,
        player_ids: tuple[str, ...] | list[str] | None = None,
        team: str | None = None,
        teams: tuple[str, ...] | list[str] | None = None,
        source: str | None = None,
        season: int | None = None,
        document_type: str | None = None,
        storyline: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        published_from: str | None = None,
        published_to: str | None = None,
    ) -> list[SearchHit]:
        with self._connect() as connection:
            self._initialize(connection)
            rows = connection.execute(
                """
                SELECT documents.*,
                       bm25(documents_fts, 0.0, 5.0, 1.0, 2.5, 1.5, 1.0) AS rank
                FROM documents_fts
                JOIN documents ON documents.id = documents_fts.document_id
                WHERE documents_fts MATCH ?
                ORDER BY rank ASC
                LIMIT 250
                """,
                (self._fts_query(query),),
            )

            hits: list[SearchHit] = []
            for row in rows:
                document = self._row_to_document(row)
                if not self._matches_filters(
                    document,
                    player,
                    players,
                    player_ids,
                    team,
                    teams,
                    source,
                    season,
                    document_type,
                    storyline,
                    published_after,
                    published_before,
                    published_from,
                    published_to,
                ):
                    continue
                hits.append(
                    SearchHit(
                        document=document,
                        score=-float(row["rank"]),
                        method="keyword",
                        keyword_rank=len(hits) + 1,
                    )
                )
                if len(hits) >= limit:
                    break

        return hits

    def _query_embedding(self, query: str, model: str) -> list[float]:
        import hashlib

        query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        with self._connect() as connection:
            self._initialize(connection)
            cached = connection.execute(
                """
                SELECT embedding_json FROM query_embeddings
                WHERE query_hash = ? AND model = ?
                """,
                (query_hash, model),
            ).fetchone()
            if cached:
                return list(json.loads(cached["embedding_json"]))

            embedding = embed_texts([query], model=model)[0]
            connection.execute(
                """
                INSERT OR REPLACE INTO query_embeddings (
                    query_hash, model, embedding_json
                ) VALUES (?, ?, ?)
                """,
                (query_hash, model, json.dumps(embedding)),
            )
            connection.commit()
            return embedding

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Cannot compare embeddings with different dimensions")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right, strict=True))
        return dot_product / (left_norm * right_norm)

    def vector_search(
        self,
        query: str,
        limit: int = 5,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        player: str | None = None,
        players: tuple[str, ...] | list[str] | None = None,
        player_ids: tuple[str, ...] | list[str] | None = None,
        team: str | None = None,
        teams: tuple[str, ...] | list[str] | None = None,
        source: str | None = None,
        season: int | None = None,
        document_type: str | None = None,
        storyline: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        published_from: str | None = None,
        published_to: str | None = None,
    ) -> list[SearchHit]:
        with self._connect() as connection:
            self._initialize(connection)
            rows = list(
                connection.execute(
                    """
                    SELECT * FROM documents
                    WHERE embedding_json IS NOT NULL AND embedding_model = ?
                    """,
                    (embedding_model,),
                )
            )

        if not rows:
            raise RuntimeError(
                "No compatible document embeddings are indexed. Run "
                "`python -m rag.cli index --with-embeddings` first."
            )

        eligible: list[tuple[sqlite3.Row, ReportDocument]] = []
        for row in rows:
            document = self._row_to_document(row)
            if self._matches_filters(
                document,
                player,
                players,
                player_ids,
                team,
                teams,
                source,
                season,
                document_type,
                storyline,
                published_after,
                published_before,
                published_from,
                published_to,
            ):
                eligible.append((row, document))

        if not eligible:
            return []

        # Confirm that a compatible document index exists before paying to embed
        # a new query, and apply metadata filters before that call too.
        query_embedding = self._query_embedding(query, embedding_model)

        scored: list[tuple[ReportDocument, float]] = []
        for row, document in eligible:
            embedding = list(json.loads(row["embedding_json"]))
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append((document, score))

        scored.sort(key=lambda item: (item[1], item[0].published_at), reverse=True)
        return [
            SearchHit(
                document=document,
                score=score,
                method="vector",
                vector_rank=rank,
            )
            for rank, (document, score) in enumerate(scored[:limit], start=1)
        ]

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        *,
        keyword_query: str | None = None,
        vector_query: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        player: str | None = None,
        players: tuple[str, ...] | list[str] | None = None,
        player_ids: tuple[str, ...] | list[str] | None = None,
        team: str | None = None,
        teams: tuple[str, ...] | list[str] | None = None,
        source: str | None = None,
        season: int | None = None,
        document_type: str | None = None,
        storyline: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        published_from: str | None = None,
        published_to: str | None = None,
    ) -> list[SearchHit]:
        pool_size = max(limit * 4, 20)
        filters = {
            "player": player,
            "players": players,
            "player_ids": player_ids,
            "team": team,
            "teams": teams,
            "source": source,
            "season": season,
            "document_type": document_type,
            "storyline": storyline,
            "published_after": published_after,
            "published_before": published_before,
            "published_from": published_from,
            "published_to": published_to,
        }
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

        # Reciprocal-rank fusion avoids pretending BM25 and cosine scores share a
        # comparable numeric scale.
        for rank, hit in enumerate(keyword_hits, start=1):
            documents[hit.document.id] = hit.document
            scores[hit.document.id] = scores.get(hit.document.id, 0.0) + 1 / (60 + rank)
            keyword_ranks[hit.document.id] = rank

        for rank, hit in enumerate(vector_hits, start=1):
            documents[hit.document.id] = hit.document
            scores[hit.document.id] = scores.get(hit.document.id, 0.0) + 1 / (60 + rank)
            vector_ranks[hit.document.id] = rank

        ranked_ids = sorted(
            documents,
            key=lambda document_id: (
                scores[document_id],
                documents[document_id].published_at,
            ),
            reverse=True,
        )
        return [
            SearchHit(
                document=documents[document_id],
                score=scores[document_id],
                method="hybrid",
                keyword_rank=keyword_ranks.get(document_id),
                vector_rank=vector_ranks.get(document_id),
            )
            for document_id in ranked_ids[:limit]
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
        if not self.index_path.exists():
            return {
                "index_path": str(self.index_path),
                "document_count": 0,
                "embedded_count": 0,
                "embedding_models": [],
            }

        with self._connect() as connection:
            self._initialize(connection)
            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            embedded_count = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE embedding_json IS NOT NULL"
            ).fetchone()[0]
            models = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT embedding_model FROM documents
                    WHERE embedding_model IS NOT NULL
                    ORDER BY embedding_model
                    """
                )
            ]

        return {
            "index_path": str(self.index_path),
            "document_count": document_count,
            "embedded_count": embedded_count,
            "embedding_models": models,
        }
