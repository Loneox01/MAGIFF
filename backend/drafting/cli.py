"""Replayable local harness for the read-only draft advisor."""

from __future__ import annotations

import argparse
import json
from datetime import date

from integrations.sleeper import SleeperDraftClient

from .agent import DraftAgentService
from .board import DraftContextBuilder, roster_requirements_from_settings
from .models import DraftContext


def _add_ecr_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--league-format", default="redraft_1qb")
    parser.add_argument("--snapshot-type", default="current")
    parser.add_argument("--as-of-date")
    parser.add_argument("--shortlist-size", type=int, default=28)
    parser.add_argument(
        "--question",
        default="Who should I draft here and why?",
    )
    parser.add_argument(
        "--board-only",
        action="store_true",
        help="Print deterministic state without making an OpenAI request.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full provider-neutral draft context as JSON.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser(
        "simulate",
        help="Replay a chalk-ECR snake draft at a selected turn.",
    )
    simulate.add_argument("--season", type=int, default=date.today().year)
    simulate.add_argument("--teams", type=int, default=12)
    simulate.add_argument("--rounds", type=int, default=16)
    simulate.add_argument("--draft-slot", type=int, default=6)
    simulate.add_argument("--round", type=int, default=5, dest="target_round")
    _add_ecr_arguments(simulate)

    live = subparsers.add_parser(
        "live",
        help="Read one current Sleeper draft snapshot.",
    )
    live.add_argument("--draft-id", required=True)
    identity = live.add_mutually_exclusive_group(required=True)
    identity.add_argument("--user-id")
    identity.add_argument("--draft-slot", type=int)
    live.add_argument("--season", type=int)
    _add_ecr_arguments(live)
    return parser


def _build_context(args: argparse.Namespace) -> DraftContext:
    builder = DraftContextBuilder()
    if args.command == "simulate":
        return builder.simulate(
            season=args.season,
            scoring_format=args.scoring,
            league_format=args.league_format,
            teams=args.teams,
            rounds=args.rounds,
            draft_slot=args.draft_slot,
            target_round=args.target_round,
            snapshot_type=args.snapshot_type,
            as_of_date=args.as_of_date,
            shortlist_size=args.shortlist_size,
        )

    draft, picks, draft_slot, roster_id = SleeperDraftClient().snapshot(
        draft_id=args.draft_id,
        user_id=args.user_id,
        draft_slot=args.draft_slot,
    )
    settings = draft.get("settings") or {}
    season = args.season or int(draft.get("season") or date.today().year)
    return builder.build(
        season=season,
        scoring_format=args.scoring,
        league_format=args.league_format,
        teams=int(settings.get("teams") or 12),
        rounds=int(settings.get("rounds") or 16),
        draft_slot=draft_slot,
        roster_id=roster_id,
        picks=picks,
        roster_requirements=roster_requirements_from_settings(settings),
        draft_status=str(draft.get("status") or "unknown"),
        snapshot_type=args.snapshot_type,
        as_of_date=args.as_of_date,
        shortlist_size=args.shortlist_size,
    )


def _print_board(context: DraftContext, *, full_json: bool) -> None:
    if full_json:
        print(json.dumps(context.debug_payload(), indent=2, default=str))
        return
    turn = "ON CLOCK" if context.on_clock else (
        f"{context.picks_until_turn} picks away"
        if context.picks_until_turn is not None
        else "no remaining turn"
    )
    print(
        f"{context.teams}-team {context.scoring_format} | slot "
        f"{context.draft_slot} | pick {context.current_pick} | {turn}"
    )
    print(
        "Roster: "
        + (
            ", ".join(
                f"{pick.display_name} ({pick.position})"
                for pick in context.my_roster
            )
            or "empty"
        )
    )
    print(
        "Open starters: "
        + ", ".join(
            f"{position}={count}"
            for position, count in context.open_starter_slots.items()
            if count
        )
    )
    print(
        f"ECR snapshot: {context.ecr_snapshot_date} | next turn gap: "
        f"{context.picks_between_turns}"
    )
    print("Available shortlist:")
    for candidate in context.available_candidates:
        print(
            f"  {candidate.overall_rank:>5.1f}  "
            f"{candidate.display_name:<24} {candidate.position}" 
            f"{candidate.position_rank} {candidate.team or '-'}"
        )


def main() -> None:
    args = build_parser().parse_args()
    context = _build_context(args)
    _print_board(context, full_json=args.json)
    if args.board_only:
        return

    result = DraftAgentService().run(context, args.question)
    print("\nMAGIFF DRAFT ADVISOR\n")
    print(result.answer)
    print(
        f"\nLatency: {result.latency_seconds:.2f}s | report searches: "
        f"{sum(call.succeeded for call in result.report_calls)} | tokens: "
        f"{result.usage.input_tokens} input "
        f"({result.usage.cached_input_tokens} cached), "
        f"{result.usage.output_tokens} output"
    )
    if result.estimated_cost_usd is not None:
        print(f"Estimated cost: ${result.estimated_cost_usd:.6f}")


if __name__ == "__main__":
    main()
