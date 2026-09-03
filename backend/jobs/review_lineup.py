"""Run one due or forced next-slate automatic lineup review.

Run from ``backend/``. The scheduled form performs no model work until the next
roster kickoff reaches the configured lead time. ``--e2e-next`` deliberately
forces one complete review of the next slate for deployment testing.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from integrations.discord_lineups import DiscordLineupNotifier
from lineups.automation import AutomaticLineupReviewService
from lineups.context import LineupContextBuilder
from lineups.reviews import kickoff_slates
from repositories.lineup_reviews_supabase import SupabaseLineupReviewRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected an ISO-8601 timestamp with a timezone"
        ) from error
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamp must include a timezone")
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", default=os.getenv("SLEEPER_LEAGUE_ID"))
    parser.add_argument(
        "--user",
        default=os.getenv("SLEEPER_USERNAME") or os.getenv("SLEEPER_USER_ID"),
    )
    parser.add_argument("--week", type=int)
    parser.add_argument("--lead-minutes", type=int, default=75)
    parser.add_argument(
        "--at",
        type=_datetime,
        help="Evaluate at an explicit timezone-aware instant (testing only).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print upcoming kickoff slates without model, database, or Discord writes.",
    )
    parser.add_argument(
        "--e2e-next",
        action="store_true",
        help="Force one complete persisted and notified review of the next slate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the review without persistence or Discord notification.",
    )
    parser.add_argument("--no-notify", action="store_true")
    return parser


def _required(parser: argparse.ArgumentParser, value: str | None, message: str) -> str:
    if not value:
        parser.error(message)
    return value


def _plan_payload(args, *, as_of: datetime) -> dict:
    context = LineupContextBuilder().build(
        league_id=args.league_id,
        user_reference=args.user,
        week=args.week,
        as_of=as_of,
    )
    return {
        "status": "planned",
        "as_of": as_of.isoformat(),
        "league_id": context.league.league_id,
        "roster_id": context.league.managed_roster_id,
        "season": context.league.season,
        "week": context.week,
        "lineup_fully_locked": context.lineup_fully_locked,
        "projection_error": context.projection_error,
        "schedule_error": context.schedule_error,
        "slates": [
            {
                **slate.agent_view(),
                "review_at": (
                    slate.kickoff_at - timedelta(minutes=args.lead_minutes)
                ).isoformat(),
            }
            for slate in kickoff_slates(context, as_of=as_of)
        ],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.league_id = _required(
        parser,
        args.league_id,
        "--league-id or SLEEPER_LEAGUE_ID is required",
    )
    args.user = _required(
        parser,
        args.user,
        "--user, SLEEPER_USERNAME, or SLEEPER_USER_ID is required",
    )
    if not 1 <= args.lead_minutes <= 180:
        parser.error("--lead-minutes must be between 1 and 180")
    as_of = args.at or datetime.now(UTC)

    if args.plan:
        print(json.dumps(_plan_payload(args, as_of=as_of), indent=2))
        return

    persist = not args.dry_run
    notify = not args.dry_run and not args.no_notify
    repository = SupabaseLineupReviewRepository() if persist else None
    notifier = None
    owner_id = os.getenv("DISCORD_OWNER_USER_ID")
    if notify:
        token = _required(
            parser,
            os.getenv("DISCORD_BOT_TOKEN"),
            "DISCORD_BOT_TOKEN is required unless --dry-run or --no-notify is used",
        )
        channel_id = _required(
            parser,
            os.getenv("DISCORD_LINEUP_CHANNEL_ID"),
            "DISCORD_LINEUP_CHANNEL_ID is required unless notification is disabled",
        )
        owner_id = _required(
            parser,
            owner_id,
            "DISCORD_OWNER_USER_ID is required unless notification is disabled",
        )
        notifier = DiscordLineupNotifier(token, channel_id)
    elif not owner_id:
        owner_id = "0"

    service = AutomaticLineupReviewService(
        repository=repository,
        notifier=notifier,
        owner_discord_user_id=owner_id,
    )
    result = service.run(
        league_id=args.league_id,
        user_reference=args.user,
        week=args.week,
        as_of=as_of,
        lead_minutes=args.lead_minutes,
        force_next=args.e2e_next,
        persist=persist,
        notify=notify,
    )
    review = result.review
    payload = {
        "status": result.status,
        "reason": result.reason,
        "review_id": review.review_id if review else None,
        "outcome": review.outcome.value if review else None,
        "trigger": review.trigger.value if review else None,
        "week": review.context.week if review else args.week,
        "slate_kickoff": (
            review.slate.kickoff_at.isoformat()
            if review and review.slate
            else None
        ),
        "slate_players": (
            review.slate.agent_view() if review and review.slate else None
        ),
        "immediate_changes": (
            [
                {
                    "slot_id": change.slot_id,
                    "out": change.outgoing_player,
                    "in": change.incoming_player,
                }
                for change in review.immediate_changes
            ]
            if review
            else []
        ),
        "provisional_changes": (
            [
                {
                    "slot_id": change.slot_id,
                    "out": change.outgoing_player,
                    "in": change.incoming_player,
                }
                for change in review.provisional_changes
            ]
            if review
            else []
        ),
        "error": review.error if review else None,
        "latency_seconds": (
            review.agent_result.latency_seconds
            if review and review.agent_result
            else None
        ),
        "estimated_cost_usd": (
            review.agent_result.estimated_cost_usd
            if review and review.agent_result
            else None
        ),
    }
    print(json.dumps(payload, indent=2))
    if review and review.outcome.value == "review_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
