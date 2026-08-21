import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from orchestration.registry import TOOL_HANDLERS, tool_schemas_for_route
from orchestration.router import (
    Capability,
    FreshnessRequirement,
    RequestIntent,
    RequestRoute,
    RequestRouteResult,
    RequestRouter,
    StructuredDomain,
)
from prompts import build_system_prompt
from tools.base import ToolExecutionResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = OpenAI()

# Future web fallback, deliberately unavailable to the agent for now. Before
# enabling it, add a web capability to the request router and update the system
# prompt with an explicit local-evidence-first fallback policy.
# WEB_SEARCH_TOOL = {
#     "type": "web_search",
#     "filters": {
#         "allowed_domains": [
#             "nfl.com",
#             "espn.com",
#             "cbssports.com",
#             "profootballtalk.nbcsports.com",
#         ]
#     },
# }


def _fallback_route(prompt: str) -> RequestRoute:
    """Keep the agent usable if the cheap router request itself fails."""
    return RequestRoute(
        request_summary=prompt,
        intent=RequestIntent.OTHER,
        freshness=FreshnessRequirement.UNSPECIFIED,
        capabilities=[Capability.STRUCTURED_DATA, Capability.REPORTS],
        structured_domains=list(StructuredDomain),
        rationale="Router failed, so all local capabilities are available.",
    )


def _print_route(
    result: RequestRouteResult | None,
    route: RequestRoute,
    error: Exception | None,
) -> None:
    if result is None:
        print(f"\nRouter: fallback to all local capabilities | {error}")
    else:
        cache = "cache hit" if result.cached else "cache miss"
        print(
            f"\nRouter: {result.model} | {cache} | "
            f"input tokens {result.input_tokens} "
            f"({result.cached_input_tokens} cached) | "
            f"output tokens {result.output_tokens}"
        )
    capabilities = ", ".join(item.value for item in route.capabilities)
    domains = ", ".join(item.value for item in route.structured_domains) or "none"
    print(f"  Capabilities: {capabilities}")
    print(f"  Structured domains: {domains}")
    print(f"  Reason: {route.rationale}")


def _print_tool_telemetry(result: ToolExecutionResult) -> None:
    if result.details.get("component") != "report_pipeline":
        return
    planner = result.details["planner"]
    context_planner = result.details["context_planner"]
    identity = result.details["identity"]
    retrieval = result.details["retrieval"]
    enrichment = result.details.get("structured_enrichment", {})
    reranker = result.details["reranker"]
    planner_cache = "cache hit" if planner["cached"] else "cache miss"
    context_cache = (
        "cache hit" if context_planner["cached"] else "API call"
    )
    reranker_status = (
        "cache hit"
        if reranker["cached"]
        else ("API call" if reranker["api_called"] else "no call")
    )
    branches = ", ".join(
        f"{name}={count}"
        for name, count in retrieval.get("branch_candidates", {}).items()
    ) or "direct"
    lookup_results = enrichment.get("results", [])
    lookup_summary = ", ".join(
        f"{item['lookup_id']}={item['status']}"
        f"{'(fallback)' if item.get('fallback_used') else ''}"
        for item in lookup_results
    ) or "none"
    print(
        "  Report pipeline: "
        f"{result.details['status']} / "
        f"{result.details['evidence_sufficiency']} | "
        f"planner {planner_cache} "
        f"(retry {'yes' if planner.get('retried') else 'no'}) | "
        f"identity escalation "
        f"{'yes' if identity['triggered'] else 'no'} | "
        f"context {context_cache} "
        f"({'yes' if context_planner['context_needed'] else 'no'}, "
        f"{context_planner['branches']} branches, "
        f"retry {'yes' if context_planner.get('retried') else 'no'}) | "
        f"hybrid candidates {retrieval['candidates']} | "
        f"branches {branches} | "
        f"structured lookups {lookup_summary} | "
        f"reranker {reranker_status}"
    )


def run_agent(prompt: str):
    """Run model/tool turns until the model returns a final answer."""
    route_result = None
    route_error = None
    try:
        route_result = RequestRouter().route(prompt)
        route = route_result.route
    except Exception as error:
        route_error = error
        route = _fallback_route(prompt)

    _print_route(route_result, route, route_error)
    tools = tool_schemas_for_route(route)
    input_items = [{"role": "user", "content": prompt}]
    total_input_tokens = route_result.input_tokens if route_result else 0
    total_output_tokens = route_result.output_tokens if route_result else 0

    for _ in range(8):
        response = client.responses.create(
            model="gpt-5.6-terra",
            instructions=build_system_prompt(route),
            tools=tools,
            input=input_items,
        )

        if response.usage:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        # Preserve function calls and any reasoning items for the next model turn.
        input_items += response.output
        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if not tool_calls:
            return response, total_input_tokens, total_output_tokens

        for call in tool_calls:
            print(f"\nTool: {call.name}({call.arguments})")

            try:
                arguments = json.loads(call.arguments)
                handler = TOOL_HANDLERS[call.name]
                result = handler(**arguments)
                if isinstance(result, ToolExecutionResult):
                    total_input_tokens += result.input_tokens
                    total_output_tokens += result.output_tokens
                    _print_tool_telemetry(result)
                    result = result.output
            except Exception as error:
                result = {"error": str(error)}

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, default=str),
                }
            )

    raise RuntimeError("Tool-call limit reached before a final response")


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
            response, input_tokens, output_tokens = run_agent(prompt)
        except Exception as error:
            print(f"\nError: {error}\n")
            continue

        elapsed = time.perf_counter() - started_at

        print(f"\nAgent: {response.output_text}")
        print(f"\nLatency: {elapsed:.2f}s")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print()


if __name__ == "__main__":
    main()
