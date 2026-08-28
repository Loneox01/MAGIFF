"""One-time migration of the inherited local Markdown report snapshot.

This loader is intentionally separate from ``load_reports``. Continuous and
backfill FantasyPros ingestion owns provider envelopes under
``data/raw/reports/sources``; this migration owns the earlier curated Markdown
snapshots under dated directories. Existing Supabase source URLs are skipped so
the migration cannot duplicate or overwrite provider-managed reports.

Run from ``backend/``::

    python -m database.load_local_reports --dry-run
    python -m database.load_local_reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import polars as pl
from supabase import Client

from rag.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_PATH
from rag.documents import ReportDocument, load_reports, resolve_snapshot

from .client import get_supabase_client
from .load_reports import EMBEDDING_DIMENSIONS, PreparedReport


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
DEFAULT_PLAYERS_PATH = PROCESSED_DIR / "reference" / "players.parquet"
DEFAULT_STATUS_PATH = PROCESSED_DIR / "current" / "player_status.parquet"
DEFAULT_LOG_PATH = PROCESSED_DIR / "reports" / "local_snapshot_load_latest_run.json"
DATABASE_PAGE_SIZE = 1000


@dataclass(frozen=True)
class LocalSnapshotLoadResult:
    snapshot: str
    discovered: int
    skipped_existing_urls: int
    prepared: int
    uploaded: int
    new_versions: int
    player_links: int
    chunks: int
    reused_local_embeddings: int
    dry_run: bool


@dataclass(frozen=True)
class PlayerCandidate:
    player_id: str
    display_name: str
    position: str | None
    last_season: int | None


def _normalized_url(value: str) -> str:
    parts = urlsplit(value.strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, "")
    )


def _timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _existing_report_urls(client: Client) -> set[str]:
    values: set[str] = set()
    offset = 0
    while True:
        rows = (
            client.table("reports")
            .select("source_url")
            .range(offset, offset + DATABASE_PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        values.update(
            _normalized_url(str(row["source_url"]))
            for row in rows
            if row.get("source_url")
        )
        if len(rows) < DATABASE_PAGE_SIZE:
            break
        offset += DATABASE_PAGE_SIZE
    return values


def _local_player_links(index_path: Path) -> dict[tuple[str, str], str]:
    if not index_path.exists():
        return {}
    connection = sqlite3.connect(index_path)
    try:
        rows = connection.execute(
            """
            select document_id, display_name, player_id
            from document_player_links
            """
        )
        return {
            (str(document_id), str(display_name).casefold()): str(player_id)
            for document_id, display_name, player_id in rows
        }
    finally:
        connection.close()


def _local_embeddings(
    index_path: Path,
) -> dict[str, tuple[str, str, list[float]]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Local RAG index not found: {index_path}")
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            select id, content_hash, embedding_model, embedding_json
            from documents
            where embedding_json is not null
            """
        )
        result: dict[str, tuple[str, str, list[float]]] = {}
        for row in rows:
            vector = [float(value) for value in json.loads(row["embedding_json"])]
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Local embedding for {row['id']} has {len(vector)} dimensions; "
                    f"expected {EMBEDDING_DIMENSIONS}"
                )
            result[str(row["id"])] = (
                str(row["content_hash"]),
                str(row["embedding_model"]),
                vector,
            )
        return result
    finally:
        connection.close()


class LocalPlayerCatalog:
    def __init__(
        self,
        players_path: Path = DEFAULT_PLAYERS_PATH,
        status_path: Path = DEFAULT_STATUS_PATH,
        index_path: Path = DEFAULT_INDEX_PATH,
    ) -> None:
        players = pl.read_parquet(players_path).select(
            "player_id", "display_name", "position", "last_season"
        )
        self.by_name: dict[str, list[PlayerCandidate]] = defaultdict(list)
        for row in players.iter_rows(named=True):
            candidate = PlayerCandidate(
                player_id=str(row["player_id"]),
                display_name=str(row["display_name"]),
                position=None if row["position"] is None else str(row["position"]),
                last_season=row["last_season"],
            )
            self.by_name[candidate.display_name.casefold()].append(candidate)

        self.active_ids: set[str] = set()
        if status_path.exists():
            statuses = pl.read_parquet(status_path).select("player_id", "status")
            self.active_ids = {
                str(row["player_id"])
                for row in statuses.iter_rows(named=True)
                if str(row.get("status") or "").upper() in {"ACT", "RES", "PUP"}
            }
        self.document_links = _local_player_links(index_path)

    def resolve(self, document: ReportDocument, display_name: str) -> PlayerCandidate:
        candidates = self.by_name.get(display_name.casefold(), [])
        if not candidates:
            raise ValueError(
                f"No processed player matches {display_name!r} in {document.id}"
            )
        if len(candidates) == 1:
            return candidates[0]

        linked_id = self.document_links.get((document.id, display_name.casefold()))
        linked = [item for item in candidates if item.player_id == linked_id]
        if len(linked) == 1:
            return linked[0]

        active = [item for item in candidates if item.player_id in self.active_ids]
        if len(active) == 1:
            return active[0]

        latest_season = max(item.last_season or -1 for item in candidates)
        latest = [item for item in candidates if (item.last_season or -1) == latest_season]
        if len(latest) == 1:
            return latest[0]

        details = ", ".join(
            f"{item.player_id} ({item.position}, {item.last_season})"
            for item in candidates
        )
        raise ValueError(
            f"Ambiguous processed player {display_name!r} in {document.id}: {details}"
        )


def _source_hash(document: ReportDocument) -> str:
    return hashlib.sha256(document.source_path.read_bytes()).hexdigest()


def _prepare_document(
    document: ReportDocument,
    *,
    snapshot_dir: Path,
    catalog: LocalPlayerCatalog,
    embeddings: dict[str, tuple[str, str, list[float]]],
    processed_at: str,
) -> PreparedReport:
    source_hash = _source_hash(document)
    fetched_at = _timestamp(document.fetched_at)
    player_rows: list[dict[str, object]] = []
    player_ids: list[str] = []
    for index, display_name in enumerate(document.players):
        player = catalog.resolve(document, display_name)
        UUID(player.player_id)
        player_ids.append(player.player_id)
        player_rows.append(
            {
                "player_id": player.player_id,
                "reference_text": display_name,
                "identity_confidence": 1.0,
                "resolution_basis": "exact_name",
                "mention_role": "primary_subject" if index == 0 else "contextual",
                "resolution_source": "local_processed_player_catalog",
            }
        )

    embedded = embeddings.get(document.id)
    if embedded is None:
        raise ValueError(f"No reusable local embedding exists for {document.id}")
    embedded_hash, embedding_model, vector = embedded
    if embedded_hash != document.content_hash:
        raise ValueError(f"Local document and embedding hashes differ for {document.id}")
    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Unexpected local embedding model for {document.id}: {embedding_model}"
        )

    relative_path = str(document.source_path.relative_to(BACKEND_DIR))
    raw_payload = {
        "format": "markdown_frontmatter",
        "snapshot": snapshot_dir.name,
        "source_path": relative_path,
        "content": document.source_path.read_text(encoding="utf-8"),
    }
    normalized_payload = {
        "id": document.id,
        "title": document.title,
        "source": document.source,
        "url": document.url,
        "author": document.author,
        "published_at": document.published_at,
        "fetched_at": document.fetched_at,
        "players": list(document.players),
        "player_ids": player_ids,
        "teams": list(document.teams),
        "season": document.season,
        "document_type": document.document_type,
        "storyline": document.storyline,
        "content_mode": document.content_mode,
        "body": document.body,
    }
    report = {
        "report_id": document.id,
        "provider": "local_seed",
        "external_id": document.id,
        "source": document.source,
        "source_url": document.url,
        "title": document.title,
        "author": document.author,
        "language": "en",
        "published_at": _timestamp(document.published_at),
        "source_updated_at": None,
        "first_seen_at": fetched_at,
        "last_seen_at": fetched_at,
        "event_start_at": None,
        "event_end_at": None,
        "event_time_confidence": None,
        "season": document.season,
        "document_type": document.document_type,
        "document_type_confidence": None,
        "storyline": document.storyline,
        "content_mode": document.content_mode,
        "source_team_id": None,
        "teams": list(document.teams),
        "source_categories": [],
        "body": document.body,
        "source_content_hash": source_hash,
        "content_hash": document.content_hash,
        "metadata": {
            "migration": "local_report_snapshot_v1",
            "snapshot": snapshot_dir.name,
            "source_path": relative_path,
        },
        "is_active": True,
        "retracted_at": None,
    }
    version = {
        "report_id": document.id,
        "source_content_hash": source_hash,
        "content_hash": document.content_hash,
        "fetched_at": fetched_at,
        "processed_at": processed_at,
        "metadata_model": None,
        "metadata_prompt_version": None,
        "normalizer_version": "local-snapshot-v1",
        "raw_payload": raw_payload,
        "raw_storage_path": None,
        "normalized_payload": normalized_payload,
    }
    chunk = {
        "chunk_id": f"{document.id}:0",
        "report_id": document.id,
        "chunk_index": 0,
        "heading": document.title,
        "content": document.body,
        "embedding_text": document.embedding_text,
        "content_hash": document.content_hash,
        "token_count": None,
        "chunk_metadata": {
            "provider": "local_seed",
            "players": list(document.players),
            "player_ids": player_ids,
            "teams": list(document.teams),
            "document_type": document.document_type,
            "snapshot": snapshot_dir.name,
        },
        "embedding": vector,
        "embedding_model": embedding_model,
        "embedded_at": processed_at,
    }
    return PreparedReport(
        path=document.source_path,
        report=report,
        version=version,
        players=player_rows,
        chunks=[chunk],
    )


def _write_log(result: LocalSnapshotLoadResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workflow": "load_local_report_snapshot",
        "completed_at": datetime.now(UTC).isoformat(),
        **asdict(result),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_local_report_snapshot(
    *,
    snapshot: str | Path | None = None,
    dry_run: bool = False,
    client: Client | None = None,
    index_path: Path = DEFAULT_INDEX_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> LocalSnapshotLoadResult:
    snapshot_dir = resolve_snapshot(snapshot)
    documents = load_reports(snapshot=snapshot_dir)
    supabase_client = client or get_supabase_client()
    existing_urls = _existing_report_urls(supabase_client)
    missing_documents = [
        document
        for document in documents
        if _normalized_url(document.url) not in existing_urls
    ]
    catalog = LocalPlayerCatalog(index_path=index_path)
    embeddings = _local_embeddings(index_path)
    processed_at = datetime.now(UTC).isoformat()
    prepared = [
        _prepare_document(
            document,
            snapshot_dir=snapshot_dir,
            catalog=catalog,
            embeddings=embeddings,
            processed_at=processed_at,
        )
        for document in missing_documents
    ]
    player_links = sum(len(item.players) for item in prepared)
    chunks = sum(len(item.chunks) for item in prepared)
    print(
        f"Snapshot {snapshot_dir.name}: discovered {len(documents)}, skipped "
        f"{len(documents) - len(missing_documents)} existing URLs, prepared "
        f"{len(prepared)} reports with {player_links} player links and {chunks} chunks."
    )
    print(f"Embeddings: reusing {chunks} local {DEFAULT_EMBEDDING_MODEL} vectors.")

    uploaded = 0
    new_versions = 0
    if dry_run:
        print("Dry run complete; no database writes were made.")
    else:
        for number, item in enumerate(prepared, start=1):
            response = supabase_client.rpc(
                "upsert_report_document",
                {
                    "p_report": item.report,
                    "p_version": item.version,
                    "p_players": item.players,
                    "p_chunks": item.chunks,
                },
            ).execute()
            value = response.data or {}
            uploaded += 1
            new_versions += int(value.get("version_inserted") is True)
            print(
                f"local reports: uploaded {number}/{len(prepared)} "
                f"({item.report['report_id']}, "
                f"new version={value.get('version_inserted', 'unknown')})"
            )

    result = LocalSnapshotLoadResult(
        snapshot=snapshot_dir.name,
        discovered=len(documents),
        skipped_existing_urls=len(documents) - len(missing_documents),
        prepared=len(prepared),
        uploaded=uploaded,
        new_versions=new_versions,
        player_links=player_links,
        chunks=chunks,
        reused_local_embeddings=chunks,
        dry_run=dry_run,
    )
    _write_log(result, log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        help="Snapshot date/path; defaults to the latest inherited snapshot.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = load_local_report_snapshot(
        snapshot=args.snapshot,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
