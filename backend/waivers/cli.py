"""Build a live league snapshot and run the read-only waiver advisor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import WaiverAgentService
from .context import WaiverContextBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="Review my waiver options and recommend any worthwhile moves.",
    )
    parser.add_argument(
        "--league-id",
        default=os.getenv("SLEEPER_LEAGUE_ID"),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("SLEEPER_USERNAME") or os.getenv("SLEEPER_USER_ID"),
    )
    parser.add_argument("--week", type=int)
    parser.add_argument("--top-default-count", type=int, default=12)
    parser.add_argument("--ecr-as-of-date")
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Print the compact default packet without invoking the model.",
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

    context = WaiverContextBuilder().build(
        league_id=args.league_id,
        user_reference=args.user,
        week=args.week,
        top_default_count=args.top_default_count,
        ecr_as_of_date=args.ecr_as_of_date,
    )
    if args.context_only:
        print(json.dumps(context.agent_payload(), indent=2, default=str))
        return

    result = WaiverAgentService().run(context, args.question)
    if args.json:
        print(
            json.dumps(
                {
                    "analysis": result.analysis.model_dump(mode="json"),
                    "preliminary": result.preliminary.model_dump(mode="json"),
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
                },
                indent=2,
                default=str,
            )
        )
        return

    print(json.dumps(result.analysis.model_dump(mode="json"), indent=2))
    print(
        f"\nModel: {result.model} | latency {result.latency_seconds:.2f}s | "
        f"tool rounds {result.tool_rounds}"
    )
    print(
        f"Tokens: {result.usage.input_tokens:,} input "
        f"({result.usage.cached_input_tokens:,} cached), "
        f"{result.usage.output_tokens:,} output"
    )
    if result.estimated_cost_usd is not None:
        print(f"Estimated text-token cost: ${result.estimated_cost_usd:.6f}")
    print("Tool calls:")
    for call in result.tool_calls:
        automatic = " automatic" if call.automatic else ""
        status = "ok" if call.succeeded else f"error: {call.error}"
        print(f"  {call.name}{automatic}: {status} {call.arguments}")


if __name__ == "__main__":
    main()
