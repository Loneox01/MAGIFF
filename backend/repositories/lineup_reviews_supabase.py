"""Supabase persistence for automatic lineup-review runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from database.client import get_supabase_client


def _is_duplicate(error: Exception) -> bool:
    return (
        str(getattr(error, "code", "")) == "23505"
        or "23505" in str(error)
        or "duplicate key" in str(error).casefold()
    )


class SupabaseLineupReviewRepository:
    def __init__(self, client=None) -> None:
        self.client = client or get_supabase_client()

    def latest_automation_review(
        self,
        *,
        league_id: str,
        roster_id: int,
        season: int,
        week: int,
        slate_kickoff: datetime,
    ) -> dict[str, Any] | None:
        rows = (
            self.client.table("lineup_review_runs")
            .select("*")
            .eq("league_id", league_id)
            .eq("roster_id", roster_id)
            .eq("season", season)
            .eq("week", week)
            .eq("slate_kickoff", slate_kickoff.isoformat())
            .in_("trigger", ["scheduled", "emergency"])
            .order("started_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return dict(rows[0]) if rows else None

    def claim(
        self,
        *,
        review_key: str,
        league_id: str,
        roster_id: int,
        season: int,
        week: int,
        slate_kickoff: datetime | None,
        trigger: str,
        lead_minutes: int,
        slate_players: dict[str, Any],
        health_snapshot: dict[str, Any],
        context_snapshot: dict[str, Any],
        notification_enabled: bool,
    ) -> str | None:
        try:
            rows = (
                self.client.table("lineup_review_runs")
                .insert(
                    {
                        "review_key": review_key,
                        "league_id": league_id,
                        "roster_id": roster_id,
                        "season": season,
                        "week": week,
                        "slate_kickoff": (
                            slate_kickoff.isoformat() if slate_kickoff else None
                        ),
                        "trigger": trigger,
                        "status": "running",
                        "lead_minutes": lead_minutes,
                        "slate_players": slate_players,
                        "health_snapshot": health_snapshot,
                        "context_snapshot": context_snapshot,
                        "notification_status": (
                            "pending" if notification_enabled else "skipped"
                        ),
                    }
                )
                .execute()
                .data
            )
        except Exception as error:
            if _is_duplicate(error):
                return None
            raise
        if not rows:
            raise RuntimeError("Supabase did not return the claimed lineup review")
        return str(rows[0]["review_id"])

    def finish(
        self,
        review_id: str,
        *,
        status: str,
        outcome: str,
        analysis: dict[str, Any] | None,
        changes: list[dict[str, Any]],
        provisional_changes: list[dict[str, Any]],
        warnings: list[str],
        model: str | None,
        latency_seconds: float | None,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float | None,
        error: str | None,
        notification_message: str | None,
    ) -> None:
        payload = {
            "status": status,
            "outcome": outcome,
            "analysis": analysis,
            "changes": changes,
            "provisional_changes": provisional_changes,
            "warnings": warnings,
            "model": model,
            "latency_seconds": latency_seconds,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "error": error,
            "notification_message": notification_message,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        if notification_message is None:
            payload["notification_status"] = "skipped"
        rows = (
            self.client.table("lineup_review_runs")
            .update(payload)
            .eq("review_id", review_id)
            .eq("status", "running")
            .execute()
            .data
        )
        if not rows:
            raise RuntimeError(f"Running lineup review {review_id} was not found")

    def mark_notification(
        self,
        review_id: str,
        *,
        status: str,
        message_id: str | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "notification_status": status,
            "notification_error": error,
            "discord_message_id": message_id,
        }
        if status == "sent":
            payload["notified_at"] = datetime.now(UTC).isoformat()
        self.client.table("lineup_review_runs").update(payload).eq(
            "review_id", review_id
        ).execute()

    def pending_notifications(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(
            (
                self.client.table("lineup_review_runs")
                .select("review_id,notification_message")
                .in_("notification_status", ["pending", "failed"])
                .not_.is_("notification_message", "null")
                .order("started_at")
                .limit(limit)
                .execute()
                .data
            )
            or []
        )
