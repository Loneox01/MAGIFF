"""Interactive terminal adapter for the shared fantasy agent service."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from model_costs import estimate_text_token_cost_usd
from prompts import WEB_ONLY_BENCHMARK_INSTRUCTIONS
from services.agent import (
    DEFAULT_AGENT_MODEL,
    AgentRunResult,
    AgentService,
    TokenUsage,
    ToolCallTelemetry,
)


@dataclass(frozen=True)
class WebOnlyRunResult:
    answer: str
    model: str
    latency_seconds: float
    usage: TokenUsage
    estimated_cost_usd: float | None
    web_search_calls: int


def _print_route(result: AgentRunResult) -> None:
    route = result.route
    if route.fallback_used:
        print(f"\nRouter: fallback to all local capabilities | {route.error}")
    else:
        cache = "cache hit" if route.cached else "cache miss"
        print(
            f"\nRouter: {route.model} | {cache} | "
            f"input tokens {route.usage.input_tokens} "
            f"({route.usage.cached_input_tokens} cached) | "
            f"output tokens {route.usage.output_tokens}"
        )
    print(f"  Capabilities: {', '.join(route.capabilities)}")
    print(
        "  Structured domains: "
        f"{', '.join(route.structured_domains) or 'none'}"
    )
    print(f"  Reason: {route.rationale}")


def _print_tool(call: ToolCallTelemetry) -> None:
    print(f"\nTool: {call.name}({call.arguments})")
    if not call.succeeded:
        print(f"  Tool error: {call.error}")
        return

    details = call.report_pipeline
    if details is None:
        return
    planner = details["planner"]
    context = details["context_planner"]
    identity = details["identity"]
    retrieval = details["retrieval"]
    enrichment = details["structured_enrichment"]
    reranker = details["reranker"]
    planner_cache = "cache hit" if planner["cached"] else "API call"
    context_cache = "cache hit" if context["cached"] else "API call"
    reranker_status = (
        "cache hit"
        if reranker["cached"]
        else ("API call" if reranker["api_called"] else "no call")
    )
    branches = ", ".join(
        f"{name}={count}"
        for name, count in retrieval["branch_candidates"].items()
    ) or "direct"
    print(
        "  Report pipeline: "
        f"{details['status']} / {details['evidence_sufficiency']} | "
        f"planner {planner_cache} "
        f"(retry {'yes' if planner['retried'] else 'no'}) | "
        f"identity escalation "
        f"{'yes' if identity['triggered'] else 'no'} | "
        f"context {context_cache} "
        f"({'yes' if context['context_needed'] else 'no'}, "
        f"{context['branches']} branches, "
        f"retry {'yes' if context['retried'] else 'no'}) | "
        f"hybrid candidates {retrieval['candidates']} | "
        f"branches {branches} | "
        f"structured lookups {enrichment['resolved']} resolved, "
        f"{enrichment['empty']} empty, "
        f"{enrichment['fallbacks']} fallback | "
        f"reranker {reranker_status}"
    )


_service = AgentService()


def run_agent(prompt: str) -> AgentRunResult:
    """Run one request without depending on terminal input/output."""
    return _service.run(prompt)


def run_web_only(
    prompt: str,
    *,
    client: Any | None = None,
    model: str = DEFAULT_AGENT_MODEL,
) -> WebOnlyRunResult:
    """Run the temporary benchmark baseline with only hosted web search."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("Web-only prompt must not be empty")

    started_at = time.perf_counter()
    response = (client or OpenAI()).responses.create(
        model=model,
        instructions=WEB_ONLY_BENCHMARK_INSTRUCTIONS,
        tools=[{"type": "web_search"}],
        input=normalized_prompt,
    )
    latency_seconds = time.perf_counter() - started_at

    raw_usage = getattr(response, "usage", None)
    input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
    input_details = getattr(raw_usage, "input_tokens_details", None)
    cached_input_tokens = min(
        input_tokens,
        int(getattr(input_details, "cached_tokens", 0) or 0),
    )
    usage = TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )
    return WebOnlyRunResult(
        answer=str(getattr(response, "output_text", "") or ""),
        model=model,
        latency_seconds=latency_seconds,
        usage=usage,
        estimated_cost_usd=estimate_text_token_cost_usd(
            model=model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        ),
        web_search_calls=sum(
            getattr(item, "type", None) == "web_search_call"
            for item in (getattr(response, "output", None) or [])
        ),
    )


def _print_usage(result: AgentRunResult | WebOnlyRunResult) -> None:
    print(f"\nTotal latency: {result.latency_seconds:.2f}s")
    print(
        f"Tokens: {result.usage.input_tokens:,} input "
        f"({result.usage.cached_input_tokens:,} cached), "
        f"{result.usage.output_tokens:,} output"
    )
    if result.estimated_cost_usd is None:
        print("Estimated text-token cost: unavailable for configured model")
    else:
        print(f"Estimated text-token cost: ${result.estimated_cost_usd:.6f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MAGIFF terminal agent")
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="temporary benchmark mode with only OpenAI web search enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = "web-search-only baseline" if args.web_only else "MAGIFF"
    print(f"Fantasy agent ready ({mode}). Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        try:
            result = run_web_only(prompt) if args.web_only else run_agent(prompt)
        except Exception as error:
            print(f"\nError: {error}\n")
            continue

        if isinstance(result, WebOnlyRunResult):
            print(
                f"\nMode: web search only | {result.model} | "
                f"web searches {result.web_search_calls}"
            )
        else:
            _print_route(result)
            for call in result.tool_calls:
                _print_tool(call)
            if result.web_search_calls:
                print(
                    f"\nWeb search: {result.web_search_calls} hosted "
                    f"{'call' if result.web_search_calls == 1 else 'calls'}"
                )
        print(f"\nAgent: {result.answer}")
        _print_usage(result)
        print()


if __name__ == "__main__":
    main()
