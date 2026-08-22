"""Load processed report documents into Supabase atomically.

Examples from the backend directory:
    python -m database.load_reports --dry-run
    python -m database.load_reports --source fantasypros

The loader calls ``upsert_report_document`` once per report. PostgreSQL then
updates the report, immutable version, current player links, and current chunks
inside one transaction. Embeddings are generated only for new or changed
chunks; an unchanged database chunk keeps its existing vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from supabase import Client

if __package__ == "backend.database":
    from backend.rag.config import DEFAULT_EMBEDDING_MODEL
    from backend.rag.retrieval.embeddings import embed_texts
else:
    from rag.config import DEFAULT_EMBEDDING_MODEL
    from rag.retrieval.embeddings import embed_texts

from .client import get_supabase_client


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROCESSED_REPORTS_DIR = BACKEND_DIR / "data" / "processed" / "reports"
DEFAULT_DOCUMENTS_DIR = PROCESSED_REPORTS_DIR / "documents"
DEFAULT_RAW_REPORTS_DIR = BACKEND_DIR / "data" / "raw" / "reports" / "sources"
DEFAULT_LOG_PATH = PROCESSED_REPORTS_DIR / "load_latest_run.json"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_DATABASE_BATCH_SIZE = 100
ALLOWED_RESOLUTION_BASES = {
    "provider_id",
    "exact_name",
    "known_alias",
    "contextual_alias",
    "inferred",
}
ALLOWED_MENTION_ROLES = {
    "primary_subject",
    "materially_affected",
    "contextual",
}

EmbeddingFunction = Callable[..., list[list[float]]]


@dataclass
class PreparedReport:
    path: Path
    report: dict[str, object]
    version: dict[str, object]
    players: list[dict[str, object]]
    chunks: list[dict[str, object]]


@dataclass(frozen=True)
class ReportLoadResult:
    discovered: int
    uploaded: int
    versions_prepared: int
    new_versions: int
    player_links: int
    chunks: int
    generated_embeddings: int
    reused_embeddings: int
    unembedded_chunks: int
    dry_run: bool


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _required_text(value: dict[str, Any], field: str, path: Path) -> str:
    result = str(value.get(field) or "").strip()
    if not result:
        raise ValueError(f"Missing required {field!r} in {path}")
    return result


def _string_list(value: object, field: str, path: Path) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field!r} must be a list in {path}")
    return [str(item) for item in value]


def _validated_timestamp(value: object, field: str, path: Path) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"Missing required {field!r} in {path}")
    try:
        datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid {field!r} timestamp in {path}: {result}") from error
    return result


def _embedding_text(document: dict[str, Any]) -> str:
    players = ", ".join(str(value) for value in document["players"])
    teams = ", ".join(str(value) for value in document["teams"])
    storyline = str(document.get("storyline") or "").replace("_", " ")
    metadata = [
        f"Title: {document['title']}",
        f"Source: {document['source']}",
        f"Published: {document['published_at']}",
        f"Players: {players}",
        f"Teams: {teams}",
        f"Document type: {document['document_type']}",
        f"Storyline: {storyline}",
    ]
    return "\n".join(metadata) + "\n\n" + str(document["body"])


def _chunk_hash(embedding_text: str) -> str:
    return hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()


def _raw_path(
    raw_reports_dir: Path,
    provider: str,
    external_id: str,
) -> Path:
    return raw_reports_dir / provider / "items" / f"{external_id}.json"


def prepare_report(
    path: Path,
    *,
    raw_reports_dir: Path = DEFAULT_RAW_REPORTS_DIR,
) -> PreparedReport:
    """Validate one normalized document and build its four RPC payloads."""
    document = _read_object(path)
    report_id = _required_text(document, "id", path)
    provider = _required_text(document, "provider", path)
    external_id = _required_text(document, "external_id", path)
    if report_id != f"{provider}:{external_id}":
        raise ValueError(
            f"Report id {report_id!r} does not match provider/external id in {path}"
        )

    raw_path = _raw_path(raw_reports_dir, provider, external_id)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw source envelope for {report_id} was not found: {raw_path}"
        )
    raw_envelope = _read_object(raw_path)
    if raw_envelope.get("provider") != provider:
        raise ValueError(f"Raw and normalized providers differ for {report_id}")
    if str(raw_envelope.get("external_id")) != external_id:
        raise ValueError(f"Raw and normalized external IDs differ for {report_id}")
    source_content_hash = _required_text(document, "source_content_hash", path)
    if raw_envelope.get("content_hash") != source_content_hash:
        raise ValueError(f"Raw and normalized source hashes differ for {report_id}")

    published_at = _validated_timestamp(
        document.get("published_at"), "published_at", path
    )
    fetched_at = _validated_timestamp(document.get("fetched_at"), "fetched_at", path)
    players = _string_list(document.get("players"), "players", path)
    teams = _string_list(document.get("teams"), "teams", path)
    categories = _string_list(
        document.get("source_categories"), "source_categories", path
    )
    metadata_processing = document.get("metadata_processing")
    if not isinstance(metadata_processing, dict):
        raise ValueError(f"metadata_processing must be an object in {path}")
    processed_at = _validated_timestamp(
        metadata_processing.get("processed_at"), "metadata_processing.processed_at", path
    )
    document_type_confidence = document.get("document_type_confidence")
    if (
        document_type_confidence is not None
        and (
            not isinstance(document_type_confidence, (int, float))
            or not 0 <= document_type_confidence <= 1
        )
    ):
        raise ValueError(f"Invalid document_type_confidence in {path}")

    report = {
        "report_id": report_id,
        "provider": provider,
        "external_id": external_id,
        "source": _required_text(document, "source", path),
        "source_url": _required_text(document, "url", path),
        "title": _required_text(document, "title", path),
        "author": document.get("author"),
        "language": "en",
        "published_at": published_at,
        "source_updated_at": None,
        "first_seen_at": fetched_at,
        "last_seen_at": fetched_at,
        "event_start_at": None,
        "event_end_at": None,
        "event_time_confidence": None,
        "season": document.get("season"),
        "document_type": _required_text(document, "document_type", path),
        "document_type_confidence": document_type_confidence,
        "storyline": document.get("storyline"),
        "content_mode": _required_text(document, "content_mode", path),
        "source_team_id": document.get("source_team_id"),
        "teams": teams,
        "source_categories": categories,
        "body": _required_text(document, "body", path),
        "source_content_hash": source_content_hash,
        "content_hash": _required_text(document, "content_hash", path),
        "metadata": {"metadata_processing": metadata_processing},
        "is_active": True,
        "retracted_at": None,
    }
    version = {
        "report_id": report_id,
        "source_content_hash": report["source_content_hash"],
        "content_hash": report["content_hash"],
        "fetched_at": fetched_at,
        "processed_at": processed_at,
        "metadata_model": metadata_processing.get("model"),
        "metadata_prompt_version": metadata_processing.get("prompt_version"),
        "normalizer_version": _required_text(
            metadata_processing, "normalizer_version", path
        ),
        "raw_payload": raw_envelope,
        "raw_storage_path": None,
        "normalized_payload": document,
    }

    raw_entities = document.get("player_entities")
    if not isinstance(raw_entities, list):
        raise ValueError(f"player_entities must be a list in {path}")
    player_rows: list[dict[str, object]] = []
    seen_player_ids: set[str] = set()
    for entity in raw_entities:
        if not isinstance(entity, dict):
            raise ValueError(f"Every player entity must be an object in {path}")
        player_id = _required_text(entity, "player_id", path)
        try:
            UUID(player_id)
        except ValueError as error:
            raise ValueError(f"Invalid player UUID {player_id!r} in {path}") from error
        if player_id in seen_player_ids:
            raise ValueError(f"Duplicate player UUID {player_id!r} in {path}")
        seen_player_ids.add(player_id)
        confidence = entity.get("identity_confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid identity confidence for {player_id} in {path}")
        resolution_basis = _required_text(entity, "resolution_basis", path)
        if resolution_basis not in ALLOWED_RESOLUTION_BASES:
            raise ValueError(
                f"Invalid resolution basis {resolution_basis!r} in {path}"
            )
        mention_role = _required_text(entity, "mention_role", path)
        if mention_role not in ALLOWED_MENTION_ROLES:
            raise ValueError(f"Invalid mention role {mention_role!r} in {path}")
        player_rows.append(
            {
                "player_id": player_id,
                "reference_text": _required_text(entity, "reference_text", path),
                "identity_confidence": confidence,
                "resolution_basis": resolution_basis,
                "mention_role": mention_role,
                "resolution_source": _required_text(
                    entity, "resolution_source", path
                ),
            }
        )

    listed_player_ids = _string_list(document.get("player_ids"), "player_ids", path)
    if listed_player_ids != [row["player_id"] for row in player_rows]:
        raise ValueError(f"player_ids and player_entities disagree in {path}")
    if players != [str(entity.get("display_name")) for entity in raw_entities]:
        raise ValueError(f"players and player_entities disagree in {path}")

    embedding_text = _embedding_text(document)
    chunk = {
        "chunk_id": f"{report_id}:0",
        "chunk_index": 0,
        "heading": document["title"],
        "content": document["body"],
        "embedding_text": embedding_text,
        "content_hash": _chunk_hash(embedding_text),
        "token_count": None,
        "chunk_metadata": {
            "provider": provider,
            "external_id": external_id,
            "players": players,
            "player_ids": listed_player_ids,
            "teams": teams,
            "document_type": document["document_type"],
        },
        "embedding": None,
        "embedding_model": None,
        "embedded_at": None,
    }
    return PreparedReport(
        path=path,
        report=report,
        version=version,
        players=player_rows,
        chunks=[chunk],
    )


def discover_reports(documents_dir: Path, source: str) -> list[Path]:
    source_dir = documents_dir / source
    if not source_dir.exists():
        raise FileNotFoundError(f"Processed report source not found: {source_dir}")
    paths = sorted(source_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No processed reports found in: {source_dir}")
    return paths


def _batches(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _existing_embedding_keys(
    client: Client,
    chunks: list[dict[str, object]],
) -> set[tuple[str, str, str]]:
    """Return chunks whose current database vector can be safely reused."""
    rows: list[dict[str, Any]] = []
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    for batch in _batches(chunk_ids, DEFAULT_DATABASE_BATCH_SIZE):
        response = (
            client.table("report_chunks")
            .select("chunk_id,content_hash,embedding_model")
            .in_("chunk_id", list(batch))
            .execute()
        )
        rows.extend(response.data or [])
    return {
        (
            str(row["chunk_id"]),
            str(row["content_hash"]),
            str(row["embedding_model"]),
        )
        for row in rows
        if row.get("embedding_model")
    }


def _attach_embeddings(
    prepared: list[PreparedReport],
    *,
    client: Client,
    embedding_model: str,
    embedding_batch_size: int,
    embedder: EmbeddingFunction,
    with_embeddings: bool,
) -> tuple[int, int, int]:
    chunks = [chunk for item in prepared for chunk in item.chunks]
    if not with_embeddings:
        return 0, 0, len(chunks)
    if not embedding_model.startswith("text-embedding-3-"):
        raise ValueError(
            "The report_chunks schema requires 1536 dimensions; configure a "
            "text-embedding-3 model so the loader can request that exact size."
        )

    existing = _existing_embedding_keys(client, chunks)
    missing = [
        chunk
        for chunk in chunks
        if (
            str(chunk["chunk_id"]),
            str(chunk["content_hash"]),
            embedding_model,
        )
        not in existing
    ]
    embedded_at = datetime.now(UTC).isoformat()
    for batch in _batches(missing, embedding_batch_size):
        vectors = embedder(
            [str(chunk["embedding_text"]) for chunk in batch],
            model=embedding_model,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding API returned an unexpected row count")
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Embedding for {chunk['chunk_id']} has {len(vector)} dimensions; "
                    f"expected {EMBEDDING_DIMENSIONS}"
                )
            chunk["embedding"] = vector
            chunk["embedding_model"] = embedding_model
            chunk["embedded_at"] = embedded_at
    return len(missing), len(chunks) - len(missing), 0


def _write_log(result: ReportLoadResult, path: Path) -> None:
    value = {
        "workflow": "load_report_documents",
        "completed_at": datetime.now(UTC).isoformat(),
        **asdict(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_report_data(
    *,
    source: str = "fantasypros",
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR,
    raw_reports_dir: Path = DEFAULT_RAW_REPORTS_DIR,
    dry_run: bool = False,
    with_embeddings: bool = True,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    client: Client | None = None,
    embedder: EmbeddingFunction = embed_texts,
    log_path: Path = DEFAULT_LOG_PATH,
) -> ReportLoadResult:
    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be positive")
    paths = discover_reports(documents_dir, source)
    prepared = [
        prepare_report(path, raw_reports_dir=raw_reports_dir) for path in paths
    ]
    player_links = sum(len(item.players) for item in prepared)
    chunk_count = sum(len(item.chunks) for item in prepared)
    print(
        f"Prepared {len(prepared)} reports, {player_links} player links, "
        f"and {chunk_count} chunks from {source}."
    )

    if dry_run:
        result = ReportLoadResult(
            discovered=len(prepared),
            uploaded=0,
            versions_prepared=len(prepared),
            new_versions=0,
            player_links=player_links,
            chunks=chunk_count,
            generated_embeddings=0,
            reused_embeddings=0,
            unembedded_chunks=chunk_count,
            dry_run=True,
        )
        print("Dry run complete; no API calls or database writes were made.")
        _write_log(result, log_path)
        return result

    supabase_client = client or get_supabase_client()
    generated, reused, unembedded = _attach_embeddings(
        prepared,
        client=supabase_client,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        embedder=embedder,
        with_embeddings=with_embeddings,
    )
    print(
        f"Embeddings: generated {generated}, reused {reused}, "
        f"left keyword-only {unembedded}."
    )

    uploaded = 0
    new_versions = 0
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
        uploaded += 1
        result_value = response.data or {}
        new_versions += int(result_value.get("version_inserted") is True)
        print(
            f"reports: uploaded {number}/{len(prepared)} "
            f"({item.report['report_id']}, "
            f"new version={result_value.get('version_inserted', 'unknown')})"
        )

    result = ReportLoadResult(
        discovered=len(prepared),
        uploaded=uploaded,
        versions_prepared=len(prepared),
        new_versions=new_versions,
        player_links=player_links,
        chunks=chunk_count,
        generated_embeddings=generated,
        reused_embeddings=reused,
        unembedded_chunks=unembedded,
        dry_run=False,
    )
    _write_log(result, log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["fantasypros"], default="fantasypros")
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--raw-reports-dir", type=Path, default=DEFAULT_RAW_REPORTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Upload keyword-searchable chunks without generating missing vectors.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    args = parser.parse_args()
    result = load_report_data(
        source=args.source,
        documents_dir=args.documents_dir,
        raw_reports_dir=args.raw_reports_dir,
        dry_run=args.dry_run,
        with_embeddings=not args.skip_embeddings,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
