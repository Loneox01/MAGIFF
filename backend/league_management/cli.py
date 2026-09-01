"""Inspect the deterministic league context without invoking an LLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .context import LeagueContextBuilder
from .models import LeagueContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league-id",
        default=os.getenv("SLEEPER_LEAGUE_ID"),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("SLEEPER_USERNAME") or os.getenv("SLEEPER_USER_ID"),
        help="Sleeper username or immutable user ID.",
    )
    parser.add_argument("--week", type=int)
    parser.add_argument("--available-limit", type=int, default=40)
    parser.add_argument("--trending-hours", type=int, default=24)
    parser.add_argument("--trending-limit", type=int, default=25)
    parser.add_argument("--ecr-as-of-date")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json",
        action="store_true",
        help="Print the full provider-neutral debug context.",
    )
    output.add_argument(
        "--agent-json",
        action="store_true",
        help="Print the compact future policy-agent payload.",
    )
    return parser


def _player_names(context: LeagueContext) -> str:
    return ", ".join(
        player.display_name for player in context.managed_roster.bench
    ) or "empty"


def _print_context(context: LeagueContext) -> None:
    roster = context.managed_roster
    waiver_budget = context.waiver_settings.get("waiver_budget", 0)
    remaining_budget = max(0, waiver_budget - roster.waiver_budget_used)
    print(
        f"{context.league_name} | {context.season} {context.status} | "
        f"week {context.current_week} | {context.total_rosters} teams"
    )
    print(
        f"Managed team: {context.managed_user_name} | roster "
        f"{context.managed_roster_id} | waiver priority "
        f"{roster.waiver_position or '-'} | budget {remaining_budget}/{waiver_budget}"
    )
    print("Starters:")
    for assignment in roster.starters:
        name = assignment.player.display_name if assignment.player else "EMPTY"
        detail = (
            f" ({assignment.player.position}, {assignment.player.team or '-'})"
            if assignment.player
            else ""
        )
        print(f"  {assignment.slot:<10} {name}{detail}")
    print(f"Bench: {_player_names(context)}")
    if roster.reserve:
        print("Reserve: " + ", ".join(player.display_name for player in roster.reserve))
    if roster.taxi:
        print("Taxi: " + ", ".join(player.display_name for player in roster.taxi))

    if context.matchup is None:
        print("Matchup: no current matchup row")
    else:
        opponent = context.matchup.opponent_name or (
            f"roster {context.matchup.opponent_roster_id}"
            if context.matchup.opponent_roster_id is not None
            else "not assigned"
        )
        print(f"Matchup: vs {opponent}")

    print(
        f"ECR snapshot: {context.ecr_snapshot_date or '-'} | top available:"
    )
    for candidate in context.available_candidates[:15]:
        print(
            f"  {candidate.overall_rank:>5.1f}  "
            f"{candidate.display_name:<24} "
            f"{candidate.position}{candidate.position_rank} "
            f"{candidate.team or '-'}"
        )
    print(
        f"Current-week transactions: {len(context.transactions)} | "
        f"unmapped Sleeper IDs: {len(context.unmapped_sleeper_player_ids)}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.league_id:
        parser.error("--league-id or SLEEPER_LEAGUE_ID is required")
    if not args.user:
        parser.error("--user, SLEEPER_USERNAME, or SLEEPER_USER_ID is required")

    context = LeagueContextBuilder().build(
        league_id=args.league_id,
        user_reference=args.user,
        week=args.week,
        available_limit=args.available_limit,
        trending_lookback_hours=args.trending_hours,
        trending_limit=args.trending_limit,
        ecr_as_of_date=args.ecr_as_of_date,
    )
    if args.json:
        print(json.dumps(context.debug_payload(), indent=2, default=str))
    elif args.agent_json:
        print(json.dumps(context.agent_payload(), indent=2, default=str))
    else:
        _print_context(context)


if __name__ == "__main__":
    main()
