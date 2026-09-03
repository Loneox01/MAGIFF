"""Idempotent automatic lineup-review orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from integrations.discord_lineups import format_lineup_review

from .agent import LineupAgentService
from .context import LineupContextBuilder
from .models import LineupChange, LineupContext
from .reviews import (
    AutomaticLineupReview,
    ReviewOutcome,
    ReviewTrigger,
    health_snapshot,
    health_snapshot_hash,
    next_kickoff_slate,
    review_is_due,
    review_question,
    split_deadline_changes,
)


class LineupReviewRepository(Protocol):
    def latest_automation_review(self, **kwargs) -> dict[str, Any] | None: ...
    def claim(self, **kwargs) -> str | None: ...
    def finish(self, review_id: str, **kwargs) -> None: ...
    def mark_notification(self, review_id: str, **kwargs) -> None: ...
    def pending_notifications(self, limit: int = 5) -> list[dict[str, Any]]: ...


class LineupReviewNotifier(Protocol):
    def send(self, content: str) -> str: ...


@dataclass(frozen=True)
class LineupAutomationResult:
    status: str
    reason: str | None
    review: AutomaticLineupReview | None


def _change_payload(change: LineupChange) -> dict[str, Any]:
    return {
        "slot_id": change.slot_id,
        "outgoing_player_id": change.outgoing_player_id,
        "outgoing_player": change.outgoing_player,
        "incoming_player_id": change.incoming_player_id,
        "incoming_player": change.incoming_player,
    }


def _review_key(
    context: LineupContext,
    *,
    kickoff: datetime | None,
    trigger: ReviewTrigger,
    snapshot_hash: str,
) -> str:
    prefix = (
        f"{context.league.league_id}:{context.league.managed_roster_id}:"
        f"{context.league.season}:{context.week}:"
        f"{kickoff.isoformat() if kickoff else 'schedule-error'}"
    )
    if trigger == ReviewTrigger.EMERGENCY:
        return f"{prefix}:emergency:{snapshot_hash}"
    if trigger == ReviewTrigger.E2E:
        return f"{prefix}:e2e:{uuid.uuid4()}"
    return f"{prefix}:scheduled"


class AutomaticLineupReviewService:
    def __init__(
        self,
        *,
        context_builder: LineupContextBuilder | None = None,
        lineup_agent: LineupAgentService | None = None,
        repository: LineupReviewRepository | None = None,
        notifier: LineupReviewNotifier | None = None,
        owner_discord_user_id: str | None = None,
    ) -> None:
        self.context_builder = context_builder or LineupContextBuilder()
        self.lineup_agent = lineup_agent or LineupAgentService()
        self.repository = repository
        self.notifier = notifier
        self.owner_discord_user_id = owner_discord_user_id

    def _send(self, review_id: str | None, message: str) -> None:
        if self.notifier is None:
            return
        try:
            message_id = self.notifier.send(message)
        except Exception as error:
            if review_id is not None and self.repository is not None:
                self.repository.mark_notification(
                    review_id,
                    status="failed",
                    error=str(error),
                )
            raise
        if review_id is not None and self.repository is not None:
            self.repository.mark_notification(
                review_id,
                status="sent",
                message_id=message_id,
            )

    def flush_pending_notifications(self) -> int:
        if self.repository is None or self.notifier is None:
            return 0
        delivered = 0
        for row in self.repository.pending_notifications():
            review_id = str(row["review_id"])
            message = str(row.get("notification_message") or "").strip()
            if not message:
                continue
            try:
                self._send(review_id, message)
            except Exception:
                continue
            delivered += 1
        return delivered

    def _persist_claim(
        self,
        *,
        context: LineupContext,
        slate,
        trigger: ReviewTrigger,
        lead_minutes: int,
        snapshot: dict[str, Any],
        persist: bool,
        notify: bool,
    ) -> str | None:
        if not persist:
            return None
        if self.repository is None:
            raise RuntimeError("A lineup-review repository is required to persist")
        key = _review_key(
            context,
            kickoff=slate.kickoff_at if slate else None,
            trigger=trigger,
            snapshot_hash=health_snapshot_hash(snapshot),
        )
        return self.repository.claim(
            review_key=key,
            league_id=context.league.league_id,
            roster_id=context.league.managed_roster_id,
            season=context.league.season,
            week=context.week,
            slate_kickoff=(slate.kickoff_at if slate else None),
            trigger=trigger.value,
            lead_minutes=lead_minutes,
            slate_players=(slate.agent_view() if slate else {}),
            health_snapshot=snapshot,
            context_snapshot=context.agent_payload(),
            notification_enabled=notify,
        )

    def _finish(
        self,
        review: AutomaticLineupReview,
        *,
        persist: bool,
        notify: bool,
    ) -> str:
        if not self.owner_discord_user_id:
            raise RuntimeError("DISCORD_OWNER_USER_ID is required for lineup messages")
        message = format_lineup_review(
            review,
            owner_discord_user_id=self.owner_discord_user_id,
        )
        if persist and review.review_id is not None:
            if self.repository is None:
                raise RuntimeError("A lineup-review repository is required to finish")
            result = review.agent_result
            self.repository.finish(
                review.review_id,
                status=("failed" if review.error else "succeeded"),
                outcome=review.outcome.value,
                analysis=(
                    result.analysis.model_dump(mode="json") if result else None
                ),
                changes=[
                    _change_payload(change) for change in review.immediate_changes
                ],
                provisional_changes=[
                    _change_payload(change) for change in review.provisional_changes
                ],
                warnings=(list(result.analysis.warnings) if result else []),
                model=(result.model if result else None),
                latency_seconds=(result.latency_seconds if result else None),
                input_tokens=(result.usage.input_tokens if result else 0),
                cached_input_tokens=(
                    result.usage.cached_input_tokens if result else 0
                ),
                output_tokens=(result.usage.output_tokens if result else 0),
                estimated_cost_usd=(result.estimated_cost_usd if result else None),
                error=review.error,
                notification_message=(message if notify else None),
            )
        if notify:
            self._send(review.review_id, message)
        return message

    def run(
        self,
        *,
        league_id: str,
        user_reference: str,
        week: int | None = None,
        as_of: datetime | None = None,
        lead_minutes: int = 75,
        force_next: bool = False,
        persist: bool = True,
        notify: bool = True,
    ) -> LineupAutomationResult:
        selected_time = as_of or datetime.now(UTC)
        if selected_time.tzinfo is None:
            raise ValueError("as_of must include a timezone")
        selected_time = selected_time.astimezone(UTC)
        if notify and (self.notifier is None or not self.owner_discord_user_id):
            raise RuntimeError(
                "Discord notifier and DISCORD_OWNER_USER_ID are required to notify"
            )
        if persist and self.repository is None:
            raise RuntimeError("A lineup-review repository is required to persist")

        if persist and notify:
            self.flush_pending_notifications()

        context = self.context_builder.build(
            league_id=league_id,
            user_reference=user_reference,
            week=week,
            as_of=selected_time,
        )
        snapshot = health_snapshot(context)
        slate = next_kickoff_slate(context, as_of=selected_time)

        if context.schedule_error:
            trigger = ReviewTrigger.E2E if force_next else ReviewTrigger.SCHEDULED
            review_id = self._persist_claim(
                context=context,
                slate=None,
                trigger=trigger,
                lead_minutes=lead_minutes,
                snapshot=snapshot,
                persist=persist,
                notify=notify,
            )
            if persist and review_id is None:
                return LineupAutomationResult("skipped", "already_reviewed", None)
            review = AutomaticLineupReview(
                review_id=review_id,
                outcome=ReviewOutcome.REVIEW_FAILED,
                trigger=trigger,
                context=context,
                slate=None,
                agent_result=None,
                immediate_changes=(),
                provisional_changes=(),
                error=f"Kickoff schedule unavailable: {context.schedule_error}",
            )
            self._finish(review, persist=persist, notify=notify)
            return LineupAutomationResult("completed", None, review)

        if context.lineup_fully_locked:
            return LineupAutomationResult("skipped", "lineup_fully_locked", None)
        if slate is None:
            return LineupAutomationResult("skipped", "no_upcoming_slate", None)

        previous = None
        if persist and self.repository is not None and not force_next:
            previous = self.repository.latest_automation_review(
                league_id=context.league.league_id,
                roster_id=context.league.managed_roster_id,
                season=context.league.season,
                week=context.week,
                slate_kickoff=slate.kickoff_at,
            )

        trigger = ReviewTrigger.E2E if force_next else ReviewTrigger.SCHEDULED
        if previous is not None:
            if previous.get("health_snapshot") == snapshot:
                return LineupAutomationResult("skipped", "already_reviewed", None)
            trigger = ReviewTrigger.EMERGENCY
        elif not force_next and not review_is_due(
            slate,
            as_of=selected_time,
            lead_minutes=lead_minutes,
        ):
            return LineupAutomationResult("skipped", "before_review_window", None)

        review_id = self._persist_claim(
            context=context,
            slate=slate,
            trigger=trigger,
            lead_minutes=lead_minutes,
            snapshot=snapshot,
            persist=persist,
            notify=notify,
        )
        if persist and review_id is None:
            return LineupAutomationResult("skipped", "duplicate_claim", None)

        if context.projection_error:
            review = AutomaticLineupReview(
                review_id=review_id,
                outcome=ReviewOutcome.REVIEW_FAILED,
                trigger=trigger,
                context=context,
                slate=slate,
                agent_result=None,
                immediate_changes=(),
                provisional_changes=(),
                error=(
                    "Sleeper weekly projections were unavailable at the lineup "
                    f"deadline: {context.projection_error}"
                ),
            )
            self._finish(review, persist=persist, notify=notify)
            return LineupAutomationResult("completed", None, review)

        try:
            agent_result = self.lineup_agent.run(
                context,
                review_question(slate),
            )
            immediate, provisional = split_deadline_changes(agent_result, slate)
            outcome = (
                ReviewOutcome.EMERGENCY_UPDATE
                if trigger == ReviewTrigger.EMERGENCY
                else (
                    ReviewOutcome.CHANGE_RECOMMENDED
                    if immediate
                    else ReviewOutcome.NO_CHANGE
                )
            )
            review = AutomaticLineupReview(
                review_id=review_id,
                outcome=outcome,
                trigger=trigger,
                context=context,
                slate=slate,
                agent_result=agent_result,
                immediate_changes=immediate,
                provisional_changes=provisional,
            )
        except Exception as error:
            review = AutomaticLineupReview(
                review_id=review_id,
                outcome=ReviewOutcome.REVIEW_FAILED,
                trigger=trigger,
                context=context,
                slate=slate,
                agent_result=None,
                immediate_changes=(),
                provisional_changes=(),
                error=str(error),
            )
        self._finish(review, persist=persist, notify=notify)
        return LineupAutomationResult("completed", None, review)
