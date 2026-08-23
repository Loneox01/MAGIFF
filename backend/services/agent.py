"""Reusable model-and-tool loop shared by the CLI and HTTP API."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.6-terra")

# Future web fallback, deliberately unavailable to the agent for now. Before
# enabling it, add a web capability to the request router and update the system
# prompt with an explicit maintained-evidence-first fallback policy.
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


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RouteTelemetry:
    model: str | None
    cached: bool
    fallback_used: bool
    request_summary: str
    intent: str
    freshness: str
    capabilities: tuple[str, ...]
    structured_domains: tuple[str, ...]
    rationale: str
    error: str | None
    usage: TokenUsage


@dataclass(frozen=True)
class ToolCallTelemetry:
    name: str
    arguments: dict[str, Any]
    succeeded: bool
    error: str | None
    report_pipeline: dict[str, Any] | None


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    model: str
    latency_seconds: float
    tool_rounds: int
    usage: TokenUsage
    route: RouteTelemetry
    tool_calls: tuple[ToolCallTelemetry, ...]


def _fallback_route(prompt: str) -> RequestRoute:
    """Keep the agent usable if the inexpensive router request fails."""
    return RequestRoute(
        request_summary=prompt,
        intent=RequestIntent.OTHER,
        freshness=FreshnessRequirement.UNSPECIFIED,
        capabilities=[Capability.STRUCTURED_DATA, Capability.REPORTS],
        structured_domains=list(StructuredDomain),
        rationale="Router failed, so all local capabilities are available.",
    )


def _cached_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "input_tokens_details", None)
    if details is None:
        return 0
    return int(getattr(details, "cached_tokens", 0) or 0)


def _report_pipeline_summary(details: Mapping[str, Any]) -> dict[str, Any] | None:
    """Keep API telemetry useful without returning full internal artifacts."""
    if details.get("component") != "report_pipeline":
        return None

    planner = details.get("planner", {})
    context = details.get("context_planner", {})
    identity = details.get("identity", {})
    retrieval = details.get("retrieval", {})
    enrichment = details.get("structured_enrichment", {})
    reranker = details.get("reranker", {})

    return {
        "status": details.get("status"),
        "evidence_sufficiency": details.get("evidence_sufficiency"),
        "planner": {
            "model": planner.get("model"),
            "cached": bool(planner.get("cached")),
            "retried": bool(planner.get("retried")),
        },
        "context_planner": {
            "model": context.get("model"),
            "cached": bool(context.get("cached")),
            "context_needed": bool(context.get("context_needed")),
            "branches": int(context.get("branches", 0) or 0),
            "retried": bool(context.get("retried")),
        },
        "identity": {
            "model": identity.get("model"),
            "triggered": bool(identity.get("triggered")),
            "cached": bool(identity.get("cached")),
            "impactful": bool(identity.get("impactful")),
        },
        "retrieval": {
            "strategy": retrieval.get("strategy"),
            "candidates": int(retrieval.get("candidates", 0) or 0),
            "branch_candidates": retrieval.get("branch_candidates", {}),
            "linked_document_entities": int(
                retrieval.get("linked_document_entities", 0) or 0
            ),
        },
        "structured_enrichment": {
            "lookups": int(enrichment.get("lookups", 0) or 0),
            "resolved": int(enrichment.get("resolved", 0) or 0),
            "empty": int(enrichment.get("empty", 0) or 0),
            "errors": len(enrichment.get("errors", []) or []),
            "fallbacks": sum(
                bool(item.get("fallback_used"))
                for item in enrichment.get("results", []) or []
            ),
        },
        "reranker": {
            "model": reranker.get("model"),
            "cached": bool(reranker.get("cached")),
            "api_called": bool(reranker.get("api_called")),
            "ranking_changed": bool(reranker.get("ranking_changed")),
            "latency_ms": int(reranker.get("latency_ms", 0) or 0),
        },
    }


class AgentService:
    """Execute one complete request through routing, tools, and final response."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        router_factory: Callable[[], RequestRouter] = RequestRouter,
        tool_handlers: Mapping[str, Callable[..., Any]] | None = None,
        tool_schema_builder: Callable[[RequestRoute], list[dict]] = (
            tool_schemas_for_route
        ),
        model: str = DEFAULT_AGENT_MODEL,
        max_tool_rounds: int = 8,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self._client = client
        self._client_lock = threading.Lock()
        self.router_factory = router_factory
        self.tool_handlers = dict(
            TOOL_HANDLERS if tool_handlers is None else tool_handlers
        )
        self.tool_schema_builder = tool_schema_builder
        self.model = model
        self.max_tool_rounds = max_tool_rounds

    @property
    def client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = OpenAI()
        return self._client

    def _route(self, prompt: str) -> tuple[RequestRoute, RouteTelemetry]:
        route_result: RequestRouteResult | None = None
        route_error: Exception | None = None
        try:
            route_result = self.router_factory().route(prompt)
            route = route_result.route
        except Exception as error:
            route_error = error
            route = _fallback_route(prompt)
            LOGGER.warning(
                "Request routing failed; using all local capabilities",
                exc_info=True,
            )

        usage = TokenUsage(
            input_tokens=route_result.input_tokens if route_result else 0,
            cached_input_tokens=(
                route_result.cached_input_tokens if route_result else 0
            ),
            output_tokens=route_result.output_tokens if route_result else 0,
        )
        telemetry = RouteTelemetry(
            model=route_result.model if route_result else None,
            cached=bool(route_result and route_result.cached),
            fallback_used=route_result is None,
            request_summary=route.request_summary,
            intent=route.intent.value,
            freshness=route.freshness.value,
            capabilities=tuple(item.value for item in route.capabilities),
            structured_domains=tuple(
                item.value for item in route.structured_domains
            ),
            rationale=route.rationale,
            error=str(route_error) if route_error else None,
            usage=usage,
        )
        return route, telemetry

    def run(self, prompt: str) -> AgentRunResult:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("Agent prompt must not be empty")

        started_at = time.perf_counter()
        route, route_telemetry = self._route(normalized_prompt)
        tools = self.tool_schema_builder(route)
        input_items: list[Any] = [
            {"role": "user", "content": normalized_prompt}
        ]
        total_input_tokens = route_telemetry.usage.input_tokens
        total_cached_input_tokens = route_telemetry.usage.cached_input_tokens
        total_output_tokens = route_telemetry.usage.output_tokens
        tool_telemetry: list[ToolCallTelemetry] = []

        for round_index in range(1, self.max_tool_rounds + 1):
            response = self.client.responses.create(
                model=self.model,
                instructions=build_system_prompt(route),
                tools=tools,
                input=input_items,
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                total_input_tokens += input_tokens
                total_cached_input_tokens += min(
                    _cached_tokens(usage), input_tokens
                )
                total_output_tokens += int(
                    getattr(usage, "output_tokens", 0) or 0
                )

            response_output = list(getattr(response, "output", []) or [])
            input_items.extend(response_output)
            tool_calls = [
                item
                for item in response_output
                if getattr(item, "type", None) == "function_call"
            ]

            if not tool_calls:
                return AgentRunResult(
                    answer=str(getattr(response, "output_text", "") or ""),
                    model=self.model,
                    latency_seconds=time.perf_counter() - started_at,
                    tool_rounds=round_index - 1,
                    usage=TokenUsage(
                        input_tokens=total_input_tokens,
                        cached_input_tokens=total_cached_input_tokens,
                        output_tokens=total_output_tokens,
                    ),
                    route=route_telemetry,
                    tool_calls=tuple(tool_telemetry),
                )

            for call in tool_calls:
                name = str(call.name)
                arguments: dict[str, Any] = {}
                succeeded = False
                error_message: str | None = None
                report_pipeline: dict[str, Any] | None = None
                try:
                    decoded_arguments = json.loads(call.arguments)
                    if not isinstance(decoded_arguments, dict):
                        raise TypeError("Tool arguments must be a JSON object")
                    arguments = decoded_arguments
                    handler = self.tool_handlers[name]
                    result = handler(**arguments)
                    if isinstance(result, ToolExecutionResult):
                        total_input_tokens += result.input_tokens
                        total_cached_input_tokens += min(
                            result.cached_input_tokens,
                            result.input_tokens,
                        )
                        total_output_tokens += result.output_tokens
                        report_pipeline = _report_pipeline_summary(result.details)
                        result = result.output
                    succeeded = True
                except Exception as error:
                    error_message = str(error)
                    result = {"error": error_message}
                    LOGGER.warning(
                        "Agent tool call failed: %s",
                        name,
                        exc_info=True,
                    )

                tool_telemetry.append(
                    ToolCallTelemetry(
                        name=name,
                        arguments=arguments,
                        succeeded=succeeded,
                        error=error_message,
                        report_pipeline=report_pipeline,
                    )
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, default=str),
                    }
                )

        raise RuntimeError("Tool-call limit reached before a final response")
