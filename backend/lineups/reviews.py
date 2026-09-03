"""Deterministic kickoff-slate selection for automatic lineup reviews."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from .agent import LineupRunResult
from .models import LineupChange, LineupContext, LineupPlayer


class ReviewOutcome(StrEnum):
    NO_CHANGE = "no_change"
    CHANGE_RECOMMENDED = "change_recommended"
    REVIEW_FAILED = "review_failed"
    EMERGENCY_UPDATE = "emergency_update"


class ReviewTrigger(StrEnum):
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"
    E2E = "e2e"


@dataclass(frozen=True)
class KickoffSlate:
    kickoff_at: datetime
    players: tuple[LineupPlayer, ...]

    @property
    def player_ids(self) -> frozenset[str]:
        return frozenset(player.sleeper_player_id for player in self.players)

    @property
    def starters(self) -> tuple[LineupPlayer, ...]:
        return tuple(
            player for player in self.players if player.roster_group == "starter"
        )

    @property
    def bench(self) -> tuple[LineupPlayer, ...]:
        return tuple(
            player for player in self.players if player.roster_group == "bench"
        )

    def agent_view(self) -> dict[str, Any]:
        def player_view(player: LineupPlayer) -> dict[str, Any]:
            return {
                "sleeper_player_id": player.sleeper_player_id,
                "name": player.display_name,
                "position": player.position,
                "team": player.team,
                "current_slot_id": player.current_slot_id,
                "projected_points": player.projected_points,
                "injury_designation": player.injury_code,
            }

        return {
            "kickoff_at": self.kickoff_at.isoformat(),
            "starters": [player_view(player) for player in self.starters],
            "bench": [player_view(player) for player in self.bench],
        }


@dataclass(frozen=True)
class AutomaticLineupReview:
    review_id: str | None
    outcome: ReviewOutcome
    trigger: ReviewTrigger
    context: LineupContext
    slate: KickoffSlate | None
    agent_result: LineupRunResult | None
    immediate_changes: tuple[LineupChange, ...]
    provisional_changes: tuple[LineupChange, ...]
    error: str | None = None


def kickoff_slates(
    context: LineupContext,
    *,
    as_of: datetime | None = None,
) -> tuple[KickoffSlate, ...]:
    """Group every unlocked starter/bench player by exact kickoff."""
    selected_time = (as_of or context.as_of).astimezone(UTC)
    grouped: dict[datetime, list[LineupPlayer]] = {}
    for player in context.players:
        if player.roster_group not in {"starter", "bench"}:
            continue
        if player.kickoff_at is None or player.kickoff_at <= selected_time:
            continue
        grouped.setdefault(player.kickoff_at, []).append(player)
    return tuple(
        KickoffSlate(
            kickoff_at=kickoff,
            players=tuple(
                sorted(
                    players,
                    key=lambda player: (
                        player.roster_group != "starter",
                        player.current_slot_id or "",
                        player.display_name,
                    ),
                )
            ),
        )
        for kickoff, players in sorted(grouped.items())
    )


def next_kickoff_slate(
    context: LineupContext,
    *,
    as_of: datetime | None = None,
) -> KickoffSlate | None:
    values = kickoff_slates(context, as_of=as_of)
    return values[0] if values else None


def review_is_due(
    slate: KickoffSlate,
    *,
    as_of: datetime,
    lead_minutes: int = 75,
) -> bool:
    if lead_minutes < 1:
        raise ValueError("lead_minutes must be positive")
    selected_time = as_of.astimezone(UTC)
    return slate.kickoff_at - timedelta(minutes=lead_minutes) <= selected_time < slate.kickoff_at


def health_snapshot(context: LineupContext) -> dict[str, dict[str, Any]]:
    """Return only stable fields capable of triggering an emergency re-review."""
    return {
        player.sleeper_player_id: {
            "name": player.display_name,
            "designation": player.injury_code,
            "roster_status": player.roster_status,
            "unavailable": player.unavailable,
        }
        for player in sorted(
            context.players,
            key=lambda player: player.sleeper_player_id,
        )
        if player.roster_group in {"starter", "bench"}
    }


def health_snapshot_hash(snapshot: dict[str, dict[str, Any]]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_deadline_changes(
    result: LineupRunResult,
    slate: KickoffSlate,
) -> tuple[tuple[LineupChange, ...], tuple[LineupChange, ...]]:
    immediate = []
    provisional = []
    for change in result.validated.changes:
        destination = (
            immediate
            if change.outgoing_player_id in slate.player_ids
            or change.incoming_player_id in slate.player_ids
            else provisional
        )
        destination.append(change)
    return tuple(immediate), tuple(provisional)


def review_question(slate: KickoffSlate) -> str:
    return (
        "Perform an automatic lineup deadline review. Decide only whether a "
        "lineup change must be made before the supplied kickoff slate locks. "
        "Evaluate every starter and bench player in that slate together, the "
        "remaining unlocked roster, injury evidence, projections, and safe "
        "later-game fallbacks. Preserve locked players. Later-slot choices are "
        "provisional and will be reviewed again at their own deadline. Return "
        "one best legal full lineup; do not discuss waivers or trades.\n\n"
        f"Upcoming kickoff slate:\n{json.dumps(slate.agent_view(), separators=(',', ':'))}"
    )
