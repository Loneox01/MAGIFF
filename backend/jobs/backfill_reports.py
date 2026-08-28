"""Incrementally backfill historical FantasyPros NFL player news.

The public news API does not expose a documented page cursor. This job uses
its supported per-player filter instead: current ECR, recent production, and
active skill-position players provide a deterministic priority queue. Each
invocation processes only a small number of player feeds, while every request
shares the same Supabase rolling quota ledger as ``jobs.refresh_reports``.

Run from ``backend/``::

    python -m jobs.backfill_reports --plan
    python -m jobs.backfill_reports --max-requests 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client

from database.client import get_supabase_client
from jobs.refresh_reports import (
    DEFAULT_DAILY_REQUEST_BUDGET,
    RefreshResult,
    refresh_reports,
)
from rag.config import DEFAULT_EMBEDDING_MODEL


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
PROVIDER = "fantasypros"
BACKFILL_TRIGGER = "backfill_player_news"
DEFAULT_SINCE = date(2026, 1, 1)
DEFAULT_TARGET_REPORTS = 200
DEFAULT_MAX_REQUESTS = 2
DEFAULT_PER_PLAYER_LIMIT = 100
DEFAULT_CANDIDATE_LIMIT = 300
DEFAULT_REQUEST_DELAY_SECONDS = 1.1
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


@dataclass(frozen=True)
class BackfillCandidate:
    fantasypros_id: str
    player_id: str
    display_name: str
    priority_source: str


@dataclass(frozen=True)
class BackfillProgress:
    new_reports: int
    changed_reports: int
    completed_fantasypros_ids: frozenset[str]
    ecr_snapshot_date: date | None


@dataclass(frozen=True)
class BackfillDefinition:
    ecr_snapshot_date: date
    candidates: tuple[BackfillCandidate, ...]


@dataclass(frozen=True)
class BackfillResult:
    backfill_id: str
    ecr_snapshot_date: str
    status: str
    reason: str | None
    cutoff_from: str
    cutoff_to: str
    target_new_reports: int
    new_reports_before_run: int
    new_reports_after_run: int
    remaining_target: int
    provider_requests_made: int
    max_provider_requests: int
    pending_candidates: int
    processed_players: list[dict[str, object]]


CandidateLoader = Callable[
    [Client, int, str, str, int, date],
    list[BackfillCandidate],
]
DefinitionLoader = Callable[..., BackfillDefinition]


def _batches(values: list[str], size: int = 100) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _latest_ecr_player_ids(
    client: Client,
    *,
    season: int,
    scoring_format: str,
    league_format: str,
    limit: int,
    snapshot_date: date,
) -> list[str]:
    rows = (
        client.table("player_ecr")
        .select("player_id,overall_rank")
        .eq("season", season)
        .eq("scoring_format", scoring_format)
        .eq("league_format", league_format)
        .eq("snapshot_type", "current")
        .eq("scrape_date", snapshot_date.isoformat())
        .order("overall_rank")
        .limit(limit)
        .execute()
        .data
        or []
    )
    return [str(row["player_id"]) for row in rows if row.get("player_id")]


def _select_ecr_snapshot_date(
    client: Client,
    *,
    season: int,
    scoring_format: str,
    league_format: str,
    on_or_before: date | None,
) -> date | None:
    query = (
        client.table("player_ecr")
        .select("scrape_date")
        .eq("season", season)
        .eq("scoring_format", scoring_format)
        .eq("league_format", league_format)
        .eq("snapshot_type", "current")
    )
    if on_or_before is not None:
        query = query.lte("scrape_date", on_or_before.isoformat())
    rows = query.order("scrape_date", desc=True).limit(1).execute().data or []
    return date.fromisoformat(str(rows[0]["scrape_date"])) if rows else None


def _recent_producer_player_ids(
    client: Client,
    *,
    season: int,
    limit: int,
) -> list[str]:
    rows = (
        client.table("player_season_stats")
        .select("player_id,fantasy_points_ppr")
        .eq("season", season - 1)
        .eq("season_type", "REG")
        .in_("position", SKILL_POSITIONS)
        .order("fantasy_points_ppr", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    return [str(row["player_id"]) for row in rows if row.get("player_id")]


def _active_skill_player_ids(client: Client, *, limit: int) -> list[str]:
    rows = (
        client.table("player_status")
        .select("player_id")
        .eq("status", "ACT")
        .in_("position", SKILL_POSITIONS)
        .order("player_id")
        .limit(limit)
        .execute()
        .data
        or []
    )
    return [str(row["player_id"]) for row in rows if row.get("player_id")]


def _fantasypros_ids(
    client: Client,
    player_ids: list[str],
) -> dict[str, str]:
    rows: list[dict] = []
    for batch in _batches(player_ids):
        response = (
            client.table("player_external_ids")
            .select("player_id,external_id")
            .eq("provider", PROVIDER)
            .in_("player_id", batch)
            .execute()
        )
        rows.extend(response.data or [])
    return {
        str(row["player_id"]): str(row["external_id"])
        for row in rows
        if row.get("player_id") and row.get("external_id")
    }


def _player_names(client: Client, player_ids: list[str]) -> dict[str, str]:
    rows: list[dict] = []
    for batch in _batches(player_ids):
        response = (
            client.table("players")
            .select("player_id,display_name")
            .in_("player_id", batch)
            .execute()
        )
        rows.extend(response.data or [])
    return {
        str(row["player_id"]): str(row["display_name"])
        for row in rows
        if row.get("player_id") and row.get("display_name")
    }


def load_backfill_candidates(
    client: Client,
    season: int,
    scoring_format: str,
    league_format: str,
    limit: int,
    ecr_snapshot_date: date,
) -> list[BackfillCandidate]:
    """Build a fantasy-relevance-first queue from one pinned ECR snapshot."""
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    source_by_player: dict[str, str] = {}
    ordered_ids: list[str] = []

    ranked = _latest_ecr_player_ids(
        client,
        season=season,
        scoring_format=scoring_format,
        league_format=league_format,
        limit=limit * 2,
        snapshot_date=ecr_snapshot_date,
    )
    producers = _recent_producer_player_ids(
        client,
        season=season,
        limit=limit * 2,
    )
    active = _active_skill_player_ids(client, limit=limit * 3)
    for source, player_ids in (
        ("current_ecr", ranked),
        ("prior_season_production", producers),
        ("active_skill_player", active),
    ):
        for player_id in player_ids:
            if player_id not in source_by_player:
                source_by_player[player_id] = source
                ordered_ids.append(player_id)

    external_ids = _fantasypros_ids(client, ordered_ids)
    eligible_ids = [
        player_id for player_id in ordered_ids if player_id in external_ids
    ][:limit]
    names = _player_names(client, eligible_ids)
    return [
        BackfillCandidate(
            fantasypros_id=external_ids[player_id],
            player_id=player_id,
            display_name=names.get(player_id, player_id),
            priority_source=source_by_player[player_id],
        )
        for player_id in eligible_ids
    ]


def load_backfill_progress(
    client: Client,
    *,
    backfill_id: str,
) -> BackfillProgress:
    rows = (
        client.table("report_ingestion_runs")
        .select("status,new_reports,changed_reports,metadata,started_at")
        .eq("provider", PROVIDER)
        .eq("trigger", BACKFILL_TRIGGER)
        .in_("status", ["succeeded", "partial"])
        .order("started_at")
        .limit(2000)
        .execute()
        .data
        or []
    )
    new_reports = 0
    changed_reports = 0
    completed: set[str] = set()
    pinned_ecr_dates: set[date] = set()
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("backfill_id") != backfill_id:
            continue
        pinned_ecr = metadata.get("ecr_snapshot_date")
        if pinned_ecr:
            pinned_ecr_dates.add(date.fromisoformat(str(pinned_ecr)))
        new_reports += int(row.get("new_reports") or 0)
        changed_reports += int(row.get("changed_reports") or 0)
        fantasypros_id = metadata.get("request_fpid") or metadata.get(
            "candidate_fantasypros_id"
        )
        if row.get("status") == "succeeded" and fantasypros_id is not None:
            completed.add(str(fantasypros_id))
    if len(pinned_ecr_dates) > 1:
        values = ", ".join(sorted(value.isoformat() for value in pinned_ecr_dates))
        raise RuntimeError(f"Backfill {backfill_id} has conflicting ECR pins: {values}")
    return BackfillProgress(
        new_reports=new_reports,
        changed_reports=changed_reports,
        completed_fantasypros_ids=frozenset(completed),
        ecr_snapshot_date=next(iter(pinned_ecr_dates), None),
    )


def load_or_create_backfill_definition(
    client: Client,
    *,
    backfill_id: str,
    season: int,
    scoring_format: str,
    league_format: str,
    candidate_limit: int,
    requested_ecr_snapshot_date: date | None,
    progress: BackfillProgress,
    candidate_loader: CandidateLoader,
) -> BackfillDefinition:
    """Recover or durably materialize one immutable candidate queue."""
    rows = (
        client.table("report_backfills")
        .select(
            "backfill_id,provider,season,scoring_format,league_format,"
            "ecr_snapshot_date,candidate_limit,candidates"
        )
        .eq("backfill_id", backfill_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if rows:
        row = rows[0]
        expected = {
            "provider": PROVIDER,
            "season": season,
            "scoring_format": scoring_format,
            "league_format": league_format,
            "candidate_limit": candidate_limit,
        }
        mismatches = {
            field: (row.get(field), value)
            for field, value in expected.items()
            if row.get(field) != value
        }
        if mismatches:
            raise ValueError(
                f"Backfill {backfill_id} parameters differ from its stored "
                f"definition: {mismatches}"
            )
        pinned = date.fromisoformat(str(row["ecr_snapshot_date"]))
        if requested_ecr_snapshot_date and requested_ecr_snapshot_date != pinned:
            raise ValueError(
                f"Backfill {backfill_id} is pinned to ECR {pinned}; "
                f"received {requested_ecr_snapshot_date}"
            )
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list):
            raise RuntimeError(f"Backfill {backfill_id} has an invalid candidate queue")
        return BackfillDefinition(
            ecr_snapshot_date=pinned,
            candidates=tuple(BackfillCandidate(**item) for item in raw_candidates),
        )

    pinned = requested_ecr_snapshot_date or progress.ecr_snapshot_date
    if pinned is None and progress.completed_fantasypros_ids:
        raise RuntimeError(
            f"Legacy backfill {backfill_id} has no durable ECR pin. Supply "
            "--ecr-snapshot-date once to materialize its original queue."
        )
    if pinned is None:
        pinned = _select_ecr_snapshot_date(
            client,
            season=season,
            scoring_format=scoring_format,
            league_format=league_format,
            on_or_before=None,
        )
    if pinned is None:
        raise RuntimeError(
            "No current ECR snapshot is available for the requested backfill format"
        )
    candidates = candidate_loader(
        client,
        season,
        scoring_format,
        league_format,
        candidate_limit,
        pinned,
    )
    payload = {
        "backfill_id": backfill_id,
        "provider": PROVIDER,
        "season": season,
        "scoring_format": scoring_format,
        "league_format": league_format,
        "ecr_snapshot_date": pinned.isoformat(),
        "candidate_limit": candidate_limit,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    client.table("report_backfills").insert(payload).execute()
    return BackfillDefinition(
        ecr_snapshot_date=pinned,
        candidates=tuple(candidates),
    )


def default_backfill_id(cutoff_from: date) -> str:
    return f"fantasypros-player-news-{cutoff_from.isoformat()}-v1"


def _result(
    *,
    backfill_id: str,
    ecr_snapshot_date: date,
    status: str,
    reason: str | None,
    cutoff_from: date,
    cutoff_to: date,
    target_new_reports: int,
    new_reports_before_run: int,
    new_reports_after_run: int,
    provider_requests_made: int,
    max_provider_requests: int,
    pending_candidates: int,
    processed_players: list[dict[str, object]],
) -> BackfillResult:
    return BackfillResult(
        backfill_id=backfill_id,
        ecr_snapshot_date=ecr_snapshot_date.isoformat(),
        status=status,
        reason=reason,
        cutoff_from=cutoff_from.isoformat(),
        cutoff_to=cutoff_to.isoformat(),
        target_new_reports=target_new_reports,
        new_reports_before_run=new_reports_before_run,
        new_reports_after_run=new_reports_after_run,
        remaining_target=max(0, target_new_reports - new_reports_after_run),
        provider_requests_made=provider_requests_made,
        max_provider_requests=max_provider_requests,
        pending_candidates=pending_candidates,
        processed_players=processed_players,
    )


def backfill_reports(
    *,
    api_key: str,
    cutoff_from: date = DEFAULT_SINCE,
    cutoff_to: date | None = None,
    season: int = 2026,
    target_new_reports: int = DEFAULT_TARGET_REPORTS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    per_player_limit: int = DEFAULT_PER_PLAYER_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    daily_request_budget: int = DEFAULT_DAILY_REQUEST_BUDGET,
    scoring_format: str = "ppr",
    league_format: str = "redraft_1qb",
    ecr_snapshot_date: date | None = None,
    backfill_id: str | None = None,
    metadata_model: str = "gpt-5.6-luna",
    metadata_batch_size: int = 10,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = 100,
    with_embeddings: bool = True,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    plan_only: bool = False,
    client: Client | None = None,
    openai_client: OpenAI | None = None,
    candidate_loader: CandidateLoader = load_backfill_candidates,
    definition_loader: DefinitionLoader = load_or_create_backfill_definition,
    refresh_function: Callable[..., RefreshResult] = refresh_reports,
) -> BackfillResult:
    """Process a resumable, bounded slice of the historical player queue."""
    end = cutoff_to or datetime.now(UTC).date()
    identity = backfill_id or default_backfill_id(cutoff_from)
    if end < cutoff_from:
        raise ValueError("cutoff_to cannot be earlier than cutoff_from")
    if target_new_reports < 1:
        raise ValueError("target_new_reports must be positive")
    if not 1 <= max_requests <= 10:
        raise ValueError("max_requests must be between 1 and 10")
    if not 1 <= per_player_limit <= 100:
        raise ValueError("per_player_limit must be between 1 and 100")
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if not plan_only and not api_key.strip():
        raise ValueError("FantasyPros API key cannot be empty")

    supabase_client = client or get_supabase_client()
    progress = load_backfill_progress(supabase_client, backfill_id=identity)
    definition = definition_loader(
        supabase_client,
        backfill_id=identity,
        season=season,
        scoring_format=scoring_format,
        league_format=league_format,
        candidate_limit=candidate_limit,
        requested_ecr_snapshot_date=ecr_snapshot_date,
        progress=progress,
        candidate_loader=candidate_loader,
    )
    pinned_ecr_date = definition.ecr_snapshot_date
    candidates = list(definition.candidates)
    pending = [
        candidate
        for candidate in candidates
        if candidate.fantasypros_id not in progress.completed_fantasypros_ids
    ]
    before = progress.new_reports
    if before >= target_new_reports:
        return _result(
            backfill_id=identity,
            ecr_snapshot_date=pinned_ecr_date,
            status="complete",
            reason="target_reached",
            cutoff_from=cutoff_from,
            cutoff_to=end,
            target_new_reports=target_new_reports,
            new_reports_before_run=before,
            new_reports_after_run=before,
            provider_requests_made=0,
            max_provider_requests=max_requests,
            pending_candidates=len(pending),
            processed_players=[],
        )

    if plan_only:
        preview = [
            {
                "fantasypros_id": candidate.fantasypros_id,
                "player_id": candidate.player_id,
                "display_name": candidate.display_name,
                "priority_source": candidate.priority_source,
            }
            for candidate in pending[:max_requests]
        ]
        return _result(
            backfill_id=identity,
            ecr_snapshot_date=pinned_ecr_date,
            status="planned",
            reason=None if preview else "candidate_queue_exhausted",
            cutoff_from=cutoff_from,
            cutoff_to=end,
            target_new_reports=target_new_reports,
            new_reports_before_run=before,
            new_reports_after_run=before,
            provider_requests_made=0,
            max_provider_requests=max_requests,
            pending_candidates=len(pending),
            processed_players=preview,
        )

    processed: list[dict[str, object]] = []
    requests_made = 0
    completed_this_run = 0
    added = before
    reason: str | None = None
    for candidate in pending[:max_requests]:
        refresh = refresh_function(
            api_key=api_key,
            report_limit=per_player_limit,
            daily_request_budget=daily_request_budget,
            trigger=BACKFILL_TRIGGER,
            fpid=candidate.fantasypros_id,
            published_from=cutoff_from,
            published_to=end,
            metadata_model=metadata_model,
            metadata_batch_size=metadata_batch_size,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            with_embeddings=with_embeddings,
            client=supabase_client,
            openai_client=openai_client,
            run_metadata={
                "backfill_id": identity,
                "backfill_kind": "player_news",
                "candidate_fantasypros_id": candidate.fantasypros_id,
                "candidate_player_id": candidate.player_id,
                "candidate_display_name": candidate.display_name,
                "candidate_priority_source": candidate.priority_source,
                "ecr_snapshot_date": pinned_ecr_date.isoformat(),
                "target_new_reports": target_new_reports,
            },
            emit_result=False,
        )
        if refresh.status == "skipped":
            reason = refresh.reason
            processed.append(
                {
                    "fantasypros_id": candidate.fantasypros_id,
                    "display_name": candidate.display_name,
                    "status": refresh.status,
                    "reason": refresh.reason,
                    "new_reports": 0,
                }
            )
            break

        requests_made += 1
        if refresh.status == "succeeded":
            completed_this_run += 1
        added += refresh.new_reports
        processed.append(
            {
                "fantasypros_id": candidate.fantasypros_id,
                "display_name": candidate.display_name,
                "priority_source": candidate.priority_source,
                "status": refresh.status,
                "provider_items_received": refresh.provider_items_received,
                "eligible_reports": refresh.eligible_reports,
                "date_filtered_reports": refresh.date_filtered_reports,
                "new_reports": refresh.new_reports,
                "changed_reports": refresh.changed_reports,
                "unchanged_reports": refresh.unchanged_reports,
                "possible_coverage_gap": refresh.possible_coverage_gap,
            }
        )
        if added >= target_new_reports:
            reason = "target_reached"
            break
        if request_delay_seconds and requests_made < max_requests:
            time.sleep(request_delay_seconds)

    remaining_pending = max(0, len(pending) - completed_this_run)
    if added >= target_new_reports:
        status = "complete"
    elif reason in {"daily_budget_exhausted", "overlap"}:
        status = "paused"
    elif remaining_pending == 0:
        status = "exhausted"
        reason = "candidate_queue_exhausted"
    else:
        status = "in_progress"
        reason = reason or "request_chunk_complete"
    return _result(
        backfill_id=identity,
        ecr_snapshot_date=pinned_ecr_date,
        status=status,
        reason=reason,
        cutoff_from=cutoff_from,
        cutoff_to=end,
        target_new_reports=target_new_reports,
        new_reports_before_run=before,
        new_reports_after_run=added,
        provider_requests_made=requests_made,
        max_provider_requests=max_requests,
        pending_candidates=remaining_pending,
        processed_players=processed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="cutoff_from",
        type=date.fromisoformat,
        default=DEFAULT_SINCE,
    )
    parser.add_argument("--to", dest="cutoff_to", type=date.fromisoformat)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--target-reports", type=int, default=DEFAULT_TARGET_REPORTS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument(
        "--per-player-limit",
        type=int,
        default=DEFAULT_PER_PLAYER_LIMIT,
    )
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument(
        "--daily-request-budget",
        type=int,
        default=DEFAULT_DAILY_REQUEST_BUDGET,
    )
    parser.add_argument("--scoring-format", default="ppr")
    parser.add_argument("--league-format", default="redraft_1qb")
    parser.add_argument(
        "--ecr-snapshot-date",
        type=date.fromisoformat,
        help=(
            "Pin a new backfill to this current-ECR snapshot date. Existing "
            "backfills automatically recover their stored/original pin."
        ),
    )
    parser.add_argument("--backfill-id")
    parser.add_argument("--metadata-batch-size", type=int, default=10)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=100)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Preview the next player-feed chunk without calling FantasyPros.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FANTASYPROS_API_KEY", "")
    if not args.plan and not api_key:
        raise RuntimeError("Set FANTASYPROS_API_KEY in the project-root .env file")
    metadata_model = os.getenv("OPENAI_REPORT_METADATA_MODEL", "gpt-5.6-luna")
    result = backfill_reports(
        api_key=api_key,
        cutoff_from=args.cutoff_from,
        cutoff_to=args.cutoff_to,
        season=args.season,
        target_new_reports=args.target_reports,
        max_requests=args.max_requests,
        per_player_limit=args.per_player_limit,
        candidate_limit=args.candidate_limit,
        daily_request_budget=args.daily_request_budget,
        scoring_format=args.scoring_format,
        league_format=args.league_format,
        ecr_snapshot_date=args.ecr_snapshot_date,
        backfill_id=args.backfill_id,
        metadata_model=metadata_model,
        metadata_batch_size=args.metadata_batch_size,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        with_embeddings=not args.skip_embeddings,
        request_delay_seconds=args.request_delay_seconds,
        plan_only=args.plan,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
