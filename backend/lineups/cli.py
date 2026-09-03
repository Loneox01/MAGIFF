"""Build a live Sleeper lineup snapshot and run the read-only advisor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import LineupAgentService
from .context import LineupContextBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="Set my best legal lineup for this week.",
    )
    parser.add_argument("--league-id", default=os.getenv("SLEEPER_LEAGUE_ID"))
    parser.add_argument(
        "--user",
        default=os.getenv("SLEEPER_USERNAME") or os.getenv("SLEEPER_USER_ID"),
    )
    parser.add_argument("--week", type=int)
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Print the verified lineup packet without invoking a model.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete structured result and telemetry as JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.league_id:
        parser.error("--league-id or SLEEPER_LEAGUE_ID is required")
    if not args.user:
        parser.error("--user, SLEEPER_USERNAME, or SLEEPER_USER_ID is required")

    context = LineupContextBuilder().build(
        league_id=args.league_id,
        user_reference=args.user,
        week=args.week,
    )
    if args.context_only:
        print(json.dumps(context.agent_payload(), indent=2, default=str))
        return

    result = LineupAgentService().run(context, args.question)
    payload = {
        "analysis": result.analysis.model_dump(mode="json"),
        "recommended_projected_total": result.validated.projected_total,
        "changes": [
            {
                "slot_id": change.slot_id,
                "out_id": change.outgoing_player_id,
                "out": change.outgoing_player,
                "in_id": change.incoming_player_id,
                "in": change.incoming_player,
            }
            for change in result.validated.changes
        ],
        "model": result.model,
        "latency_seconds": result.latency_seconds,
        "tool_rounds": result.tool_rounds,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "estimated_cost_usd": result.estimated_cost_usd,
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.arguments,
                "succeeded": call.succeeded,
                "error": call.error,
                "report_status": call.report_status,
                "automatic": call.automatic,
            }
            for call in result.tool_calls
        ],
        "news_evidence": result.news_evidence,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    print(json.dumps(result.analysis.model_dump(mode="json"), indent=2))
    print(f"\nProjected total: {result.validated.projected_total:.2f}")
    print("Changes:")
    if result.validated.changes:
        for change in result.validated.changes:
            print(
                f"  {change.slot_id}: {change.outgoing_player or 'empty'} -> "
                f"{change.incoming_player or 'empty'}"
            )
    else:
        print("  None")
    print(
        f"Model: {result.model} | latency {result.latency_seconds:.2f}s | "
        f"tool rounds {result.tool_rounds}"
    )
    print(
        f"Tokens: {result.usage.input_tokens:,} input "
        f"({result.usage.cached_input_tokens:,} cached), "
        f"{result.usage.output_tokens:,} output"
    )
    if result.estimated_cost_usd is not None:
        print(f"Estimated text-token cost: ${result.estimated_cost_usd:.6f}")


if __name__ == "__main__":
    main()
