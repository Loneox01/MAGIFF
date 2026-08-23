"""Interactive terminal adapter for the shared fantasy agent service."""

import time

from services.agent import AgentRunResult, AgentService, ToolCallTelemetry


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


def main() -> None:
    print("Fantasy agent ready. Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        started_at = time.perf_counter()
        try:
            result = run_agent(prompt)
        except Exception as error:
            print(f"\nError: {error}\n")
            continue

        _print_route(result)
        for call in result.tool_calls:
            _print_tool(call)
        elapsed = time.perf_counter() - started_at
        print(f"\nAgent: {result.answer}")
        print(f"\nLatency: {elapsed:.2f}s")
        print(f"Input tokens: {result.usage.input_tokens}")
        print(f"Cached input tokens: {result.usage.cached_input_tokens}")
        print(f"Output tokens: {result.usage.output_tokens}")
        print()


if __name__ == "__main__":
    main()
