"""Continuously refresh the FantasyPros report corpus in Supabase.

One invocation performs at most one FantasyPros API request, filters unchanged
provider items before any model work, extracts metadata for only new/changed
items, generates only missing embeddings, and transactionally loads each report.

Run from ``backend/``:

    python -m jobs.refresh_reports
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client

from database.client import get_supabase_client
from database.load_reports import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    ReportLoadResult,
    load_report_data,
)
from ingestion.reports.fantasypros import content_hash, fetch_news, ingest_payload
from processing.reports.fantasypros import (
    DEFAULT_BATCH_SIZE,
    ProcessingResult,
    SupabasePlayerCatalog,
    process_reports,
)
from rag.config import DEFAULT_EMBEDDING_MODEL


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
PROVIDER = "fantasypros"
PROVIDER_DAILY_REQUEST_CAP = 50
DEFAULT_DAILY_REQUEST_BUDGET = 40
DEFAULT_REPORT_LIMIT = 20
DEFAULT_LEASE_SECONDS = 1800

FetchFunction = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class FeedDiff:
    received: int
    eligible: int
    valid: int
    new_items: list[dict[str, Any]]
    changed_items: list[dict[str, Any]]
    unchanged_items: list[dict[str, Any]]
    invalid_items: int
    date_filtered_items: int
    oldest_published_at: str | None
    newest_published_at: str | None
    feed_window_saturated: bool
    possible_coverage_gap: bool

    @property
    def pending_items(self) -> list[dict[str, Any]]:
        return [*self.new_items, *self.changed_items]


@dataclass(frozen=True)
class RefreshResult:
    run_id: str
    status: str
    reason: str | None
    requested_reports: int
    requests_used_last_24_hours: int
    daily_request_budget: int
    provider_items_received: int
    eligible_reports: int
    date_filtered_reports: int
    new_reports: int
    changed_reports: int
    unchanged_reports: int
    failed_reports: int
    metadata_input_tokens: int
    metadata_cached_input_tokens: int
    metadata_output_tokens: int
    generated_embeddings: int
    reused_embeddings: int
    oldest_published_at: str | None
    newest_published_at: str | None
    feed_window_saturated: bool
    possible_coverage_gap: bool


def _batches(values: list[str], size: int = 100) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _published_at(item: dict[str, Any]) -> datetime | None:
    value = item.get("created")
    if value is None:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


def _filter_payload_by_date(
    payload: dict[str, Any],
    *,
    published_from: date | None,
    published_to: date | None,
) -> tuple[dict[str, Any], int, int]:
    """Return an inclusive publication-date slice and source row counts."""
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("FantasyPros response did not contain an items list")
    if published_from is None and published_to is None:
        return payload, len(raw_items), 0

    eligible: list[object] = []
    excluded = 0
    for item in raw_items:
        published = _published_at(item) if isinstance(item, dict) else None
        if published is None:
            excluded += 1
            continue
        published_date = published.date()
        if published_from is not None and published_date < published_from:
            excluded += 1
            continue
        if published_to is not None and published_date > published_to:
            excluded += 1
            continue
        eligible.append(item)
    return {**payload, "items": eligible}, len(raw_items), excluded


def _existing_hashes(
    client: Client,
    external_ids: list[str],
) -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    for batch in _batches(sorted(set(external_ids))):
        response = (
            client.table("reports")
            .select("external_id,source_content_hash")
            .eq("provider", PROVIDER)
            .in_("external_id", batch)
            .execute()
        )
        rows.extend(response.data or [])
    return {
        str(row["external_id"]): str(row["source_content_hash"])
        for row in rows
        if row.get("external_id") is not None
        and row.get("source_content_hash") is not None
    }


def classify_feed(
    payload: dict[str, Any],
    existing_hashes: dict[str, str],
    *,
    requested_limit: int,
    source_received: int | None = None,
    date_filtered_items: int = 0,
) -> FeedDiff:
    """Classify one provider page before model or embedding work."""
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("FantasyPros response did not contain an items list")

    valid: list[dict[str, Any]] = []
    invalid = 0
    seen_external_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict) or item.get("id") is None:
            invalid += 1
            continue
        external_id = str(item["id"])
        if external_id in seen_external_ids:
            invalid += 1
            continue
        seen_external_ids.add(external_id)
        valid.append(item)

    new_items: list[dict[str, Any]] = []
    changed_items: list[dict[str, Any]] = []
    unchanged_items: list[dict[str, Any]] = []
    for item in valid:
        external_id = str(item["id"])
        previous_hash = existing_hashes.get(external_id)
        current_hash = content_hash(item)
        if previous_hash is None:
            new_items.append(item)
        elif previous_hash != current_hash:
            changed_items.append(item)
        else:
            unchanged_items.append(item)

    publication_times = [
        value for item in valid if (value := _published_at(item)) is not None
    ]
    provider_received = (
        len(raw_items) if source_received is None else source_received
    )
    # Some FantasyPros API plans silently return a smaller window than the
    # requested limit. A descending feed only proves continuity when at least
    # one returned external ID was already stored. If the entire observable
    # window is new, treat it as exhausted even when its row count is below the
    # requested limit; older unseen reports may already have fallen behind it.
    has_stored_overlap = bool(changed_items or unchanged_items)
    observable_window_exhausted = bool(valid) and not has_stored_overlap
    saturated = (
        provider_received >= requested_limit or observable_window_exhausted
    )
    possible_gap = observable_window_exhausted
    return FeedDiff(
        received=provider_received,
        eligible=len(raw_items),
        valid=len(valid),
        new_items=new_items,
        changed_items=changed_items,
        unchanged_items=unchanged_items,
        invalid_items=invalid,
        date_filtered_items=date_filtered_items,
        oldest_published_at=(
            min(publication_times).isoformat() if publication_times else None
        ),
        newest_published_at=(
            max(publication_times).isoformat() if publication_times else None
        ),
        feed_window_saturated=saturated,
        possible_coverage_gap=possible_gap,
    )


def _reserve_run(
    client: Client,
    *,
    trigger: str,
    requested_reports: int,
    daily_request_budget: int,
    lease_seconds: int,
) -> dict[str, Any]:
    response = client.rpc(
        "reserve_report_ingestion_run",
        {
            "p_provider": PROVIDER,
            "p_trigger": trigger,
            "p_requested_reports": requested_reports,
            "p_daily_request_budget": daily_request_budget,
            "p_lease_seconds": lease_seconds,
        },
    ).execute()
    if not isinstance(response.data, dict):
        raise RuntimeError("Ingestion reservation RPC returned an invalid response")
    return response.data


def _finish_run(
    client: Client,
    *,
    run_id: str,
    status: str,
    metrics: dict[str, object],
    error: str | None = None,
) -> None:
    client.rpc(
        "finish_report_ingestion_run",
        {
            "p_run_id": run_id,
            "p_status": status,
            "p_metrics": metrics,
            "p_error": error,
        },
    ).execute()


def _mark_existing_reports_seen(
    client: Client,
    external_ids: list[str],
    *,
    fetched_at: str,
) -> None:
    for batch in _batches(sorted(set(external_ids))):
        (
            client.table("reports")
            .update({"last_seen_at": fetched_at})
            .eq("provider", PROVIDER)
            .in_("external_id", batch)
            .execute()
        )


def _metrics(
    diff: FeedDiff | None,
    processing: ProcessingResult | None,
    loading: ReportLoadResult | None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        **(extra or {}),
        "provider_items_received": diff.received if diff else 0,
        "eligible_reports": diff.eligible if diff else 0,
        "date_filtered_reports": diff.date_filtered_items if diff else 0,
        "new_reports": len(diff.new_items) if diff else 0,
        "changed_reports": len(diff.changed_items) if diff else 0,
        "unchanged_reports": len(diff.unchanged_items) if diff else 0,
        "failed_reports": (
            (diff.invalid_items if diff else 0)
            + (processing.failed if processing else 0)
        ),
        "unresolved_player_mentions": (
            processing.unresolved_player_mentions if processing else 0
        ),
        "metadata_input_tokens": processing.input_tokens if processing else 0,
        "metadata_cached_input_tokens": (
            processing.cached_input_tokens if processing else 0
        ),
        "metadata_output_tokens": processing.output_tokens if processing else 0,
        "generated_embeddings": loading.generated_embeddings if loading else 0,
        "reused_embeddings": loading.reused_embeddings if loading else 0,
        "oldest_published_at": diff.oldest_published_at if diff else None,
        "newest_published_at": diff.newest_published_at if diff else None,
        "feed_window_saturated": diff.feed_window_saturated if diff else False,
        "possible_coverage_gap": diff.possible_coverage_gap if diff else False,
        "normalized_reports": (
            processing.inserted + processing.updated if processing else 0
        ),
        "uploaded_reports": loading.uploaded if loading else 0,
    }


def refresh_reports(
    *,
    api_key: str,
    report_limit: int = DEFAULT_REPORT_LIMIT,
    daily_request_budget: int = DEFAULT_DAILY_REQUEST_BUDGET,
    trigger: str = "manual",
    category: str | None = None,
    fpid: str | int | None = None,
    published_from: date | None = None,
    published_to: date | None = None,
    metadata_model: str = "gpt-5.6-luna",
    metadata_batch_size: int = DEFAULT_BATCH_SIZE,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    with_embeddings: bool = True,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    client: Client | None = None,
    openai_client: OpenAI | None = None,
    fetcher: FetchFunction = fetch_news,
    run_metadata: dict[str, object] | None = None,
    emit_result: bool = True,
) -> RefreshResult:
    if report_limit < 1:
        raise ValueError("report_limit must be positive")
    if not 1 <= daily_request_budget <= PROVIDER_DAILY_REQUEST_CAP:
        raise ValueError(
            "daily_request_budget must be between 1 and the 50-request provider cap"
        )
    if metadata_batch_size < 1:
        raise ValueError("metadata_batch_size must be positive")
    if lease_seconds < 60 or lease_seconds > 3600:
        raise ValueError("lease_seconds must be between 60 and 3600")
    if published_from and published_to and published_to < published_from:
        raise ValueError("published_to cannot be earlier than published_from")

    supabase_client = client or get_supabase_client()
    reservation = _reserve_run(
        supabase_client,
        trigger=trigger,
        requested_reports=report_limit,
        daily_request_budget=daily_request_budget,
        lease_seconds=lease_seconds,
    )
    run_id = str(reservation.get("run_id") or "")
    requests_used_last_24_hours = int(
        reservation.get("requests_used_last_24_hours") or 0
    )
    if not reservation.get("acquired"):
        result = RefreshResult(
            run_id=run_id,
            status="skipped",
            reason=str(reservation.get("reason") or "reservation_not_acquired"),
            requested_reports=report_limit,
            requests_used_last_24_hours=requests_used_last_24_hours,
            daily_request_budget=daily_request_budget,
            provider_items_received=0,
            eligible_reports=0,
            date_filtered_reports=0,
            new_reports=0,
            changed_reports=0,
            unchanged_reports=0,
            failed_reports=0,
            metadata_input_tokens=0,
            metadata_cached_input_tokens=0,
            metadata_output_tokens=0,
            generated_embeddings=0,
            reused_embeddings=0,
            oldest_published_at=None,
            newest_published_at=None,
            feed_window_saturated=False,
            possible_coverage_gap=False,
        )
        if emit_result:
            print(json.dumps(asdict(result), indent=2))
        return result

    diff: FeedDiff | None = None
    processing: ProcessingResult | None = None
    loading: ReportLoadResult | None = None
    try:
        # Deliberately no automatic provider retry: every attempt can consume
        # FantasyPros quota and must first receive its own ledger reservation.
        payload = fetcher(
            api_key,
            limit=report_limit,
            category=category,
            fpid=fpid,
            order_by="created",
        )
        payload, source_received, date_filtered_items = _filter_payload_by_date(
            payload,
            published_from=published_from,
            published_to=published_to,
        )
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("FantasyPros response did not contain an items list")
        external_ids = [
            str(item["id"])
            for item in raw_items
            if isinstance(item, dict) and item.get("id") is not None
        ]
        existing = _existing_hashes(supabase_client, external_ids)
        diff = classify_feed(
            payload,
            existing,
            requested_limit=report_limit,
            source_received=source_received,
            date_filtered_items=date_filtered_items,
        )
        fetched_at = datetime.now(UTC)
        _mark_existing_reports_seen(
            supabase_client,
            [external_id for external_id in external_ids if external_id in existing],
            fetched_at=fetched_at.isoformat(),
        )

        if diff.pending_items:
            with tempfile.TemporaryDirectory(prefix="magiff-report-refresh-") as directory:
                root = Path(directory)
                raw_reports_dir = root / "raw" / "sources"
                raw_source_dir = raw_reports_dir / PROVIDER
                documents_dir = root / "processed" / "documents"
                document_source_dir = documents_dir / PROVIDER

                ingest_payload(
                    {**payload, "items": diff.pending_items},
                    output_dir=raw_source_dir,
                    category=category,
                    requested_limit=report_limit,
                    fetched_at=fetched_at,
                )
                catalog = SupabasePlayerCatalog.from_external_ids(
                    supabase_client,
                    [
                        str(item["player_id"])
                        for item in diff.pending_items
                        if item.get("player_id") is not None
                    ],
                )
                processing = process_reports(
                    raw_dir=raw_source_dir,
                    output_dir=document_source_dir,
                    model=metadata_model,
                    batch_size=metadata_batch_size,
                    client=openai_client,
                    catalog=catalog,
                )
                if processing.inserted + processing.updated:
                    loading = load_report_data(
                        source=PROVIDER,
                        documents_dir=documents_dir,
                        raw_reports_dir=raw_reports_dir,
                        with_embeddings=with_embeddings,
                        embedding_model=embedding_model,
                        embedding_batch_size=embedding_batch_size,
                        client=supabase_client,
                        log_path=root / "load_latest_run.json",
                    )

        request_metadata = {
            **(run_metadata or {}),
            "request_category": category,
            "request_fpid": None if fpid is None else str(fpid),
            "published_from": (
                published_from.isoformat() if published_from else None
            ),
            "published_to": published_to.isoformat() if published_to else None,
        }
        metrics = _metrics(diff, processing, loading, request_metadata)
        status = "partial" if int(metrics["failed_reports"]) else "succeeded"
        _finish_run(
            supabase_client,
            run_id=run_id,
            status=status,
            metrics=metrics,
        )
        result = RefreshResult(
            run_id=run_id,
            status=status,
            reason=None,
            requested_reports=report_limit,
            requests_used_last_24_hours=requests_used_last_24_hours,
            daily_request_budget=daily_request_budget,
            provider_items_received=int(metrics["provider_items_received"]),
            eligible_reports=int(metrics["eligible_reports"]),
            date_filtered_reports=int(metrics["date_filtered_reports"]),
            new_reports=int(metrics["new_reports"]),
            changed_reports=int(metrics["changed_reports"]),
            unchanged_reports=int(metrics["unchanged_reports"]),
            failed_reports=int(metrics["failed_reports"]),
            metadata_input_tokens=int(metrics["metadata_input_tokens"]),
            metadata_cached_input_tokens=int(
                metrics["metadata_cached_input_tokens"]
            ),
            metadata_output_tokens=int(metrics["metadata_output_tokens"]),
            generated_embeddings=int(metrics["generated_embeddings"]),
            reused_embeddings=int(metrics["reused_embeddings"]),
            oldest_published_at=diff.oldest_published_at,
            newest_published_at=diff.newest_published_at,
            feed_window_saturated=diff.feed_window_saturated,
            possible_coverage_gap=diff.possible_coverage_gap,
        )
        if emit_result:
            print(json.dumps(asdict(result), indent=2))
        return result
    except Exception as error:
        failure_metrics = _metrics(
            diff,
            processing,
            loading,
            {
                **(run_metadata or {}),
                "request_category": category,
                "request_fpid": None if fpid is None else str(fpid),
                "published_from": (
                    published_from.isoformat() if published_from else None
                ),
                "published_to": published_to.isoformat() if published_to else None,
            },
        )
        try:
            _finish_run(
                supabase_client,
                run_id=run_id,
                status="failed",
                metrics=failure_metrics,
                error=f"{type(error).__name__}: {error}"[:2000],
            )
        except Exception as finish_error:
            error.add_note(f"Additionally failed to close ingestion run: {finish_error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_REPORT_LIMIT)
    parser.add_argument(
        "--daily-request-budget",
        type=int,
        default=DEFAULT_DAILY_REQUEST_BUDGET,
    )
    parser.add_argument("--trigger", default="manual")
    parser.add_argument(
        "--category",
        choices=["injury", "recap", "transaction", "rumor", "breaking"],
    )
    parser.add_argument("--fpid", help="Filter by FantasyPros player ID.")
    parser.add_argument(
        "--published-from",
        type=date.fromisoformat,
        help="Keep reports published on or after YYYY-MM-DD.",
    )
    parser.add_argument(
        "--published-to",
        type=date.fromisoformat,
        help="Keep reports published on or before YYYY-MM-DD.",
    )
    parser.add_argument("--metadata-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FANTASYPROS_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set FANTASYPROS_API_KEY in the project-root .env file")
    metadata_model = os.getenv("OPENAI_REPORT_METADATA_MODEL", "gpt-5.6-luna")
    refresh_reports(
        api_key=api_key,
        report_limit=args.limit,
        daily_request_budget=args.daily_request_budget,
        trigger=args.trigger,
        category=args.category,
        fpid=args.fpid,
        published_from=args.published_from,
        published_to=args.published_to,
        metadata_model=metadata_model,
        metadata_batch_size=args.metadata_batch_size,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        with_embeddings=not args.skip_embeddings,
    )


if __name__ == "__main__":
    main()
