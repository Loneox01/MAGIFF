"""Reusable model-and-tool loop shared by the CLI and HTTP API."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from database.client import clear_supabase_client, is_transient_supabase_error
from model_costs import estimate_text_token_cost_usd
from orchestration.player_references import (
    PlayerReferenceAdapter,
    PlayerReferenceError,
    ResolvedPlayerReference,
)
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


def _configured_parallel_tool_limit() -> int:
    raw_value = os.getenv("MAGIFF_MAX_PARALLEL_TOOLS", "6")
    try:
        value = int(raw_value)
    except ValueError:
        LOGGER.warning(
            "Ignoring invalid MAGIFF_MAX_PARALLEL_TOOLS=%r; using 6",
            raw_value,
        )
        return 6
    if not 1 <= value <= 16:
        LOGGER.warning(
            "MAGIFF_MAX_PARALLEL_TOOLS must be 1-16; using 6 instead of %s",
            value,
        )
        return 6
    return value


DEFAULT_MAX_PARALLEL_TOOLS = _configured_parallel_tool_limit()

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
    estimated_cost_usd: float | None = None


@dataclass(frozen=True)
class _PreparedToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    handler_arguments: dict[str, Any] | None
    preparation_error: Exception | None = None


@dataclass(frozen=True)
class _ToolCallOutcome:
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: Any
    succeeded: bool
    error: str | None
    report_pipeline: dict[str, Any] | None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = 0.0


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
            "temporal_policy": retrieval.get("temporal_policy", {}),
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
            "attempts": int(reranker.get("attempts", 0) or 0),
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
        player_reference_adapter: PlayerReferenceAdapter | None = None,
        model: str = DEFAULT_AGENT_MODEL,
        max_tool_rounds: int = 8,
        max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if not 1 <= max_parallel_tools <= 16:
            raise ValueError("max_parallel_tools must be between 1 and 16")
        self._client = client
        self._client_lock = threading.Lock()
        self.router_factory = router_factory
        self.tool_handlers = dict(
            TOOL_HANDLERS if tool_handlers is None else tool_handlers
        )
        self.tool_schema_builder = tool_schema_builder
        self.player_reference_adapter = (
            player_reference_adapter or PlayerReferenceAdapter()
        )
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.max_parallel_tools = max_parallel_tools
        # Report search owns a cached pipeline with mutable request telemetry.
        # It may overlap structured reads, but two report searches must not use
        # that singleton simultaneously.
        self._report_tool_lock = threading.Lock()

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

    def _prepare_tool_calls(
        self,
        tool_calls: list[Any],
        *,
        source_question: str | None = None,
        player_cache: dict[
            str, ResolvedPlayerReference | PlayerReferenceError
        ],
    ) -> list[_PreparedToolCall]:
        decoded: list[tuple[Any, str, dict[str, Any], Exception | None]] = []
        references: list[str] = []

        for call in tool_calls:
            name = str(call.name)
            arguments: dict[str, Any] = {}
            error: Exception | None = None
            try:
                parsed = json.loads(call.arguments)
                if not isinstance(parsed, dict):
                    raise TypeError("Tool arguments must be a JSON object")
                arguments = parsed
                if name not in self.tool_handlers:
                    raise KeyError(f"Unknown tool: {name}")
                references.extend(
                    self.player_reference_adapter.references_for(
                        name, arguments
                    )
                )
            except Exception as caught:
                error = caught
            decoded.append((call, name, arguments, error))

        self.player_reference_adapter.resolve_many(
            references,
            cache=player_cache,
            max_workers=self.max_parallel_tools,
        )

        prepared: list[_PreparedToolCall] = []
        for call, name, arguments, error in decoded:
            handler_arguments = None
            if error is None:
                try:
                    handler_arguments = self.player_reference_adapter.adapt(
                        name,
                        arguments,
                        cache=player_cache,
                    )
                    if name == "search_reports" and source_question:
                        handler_arguments["source_question"] = source_question
                except Exception as caught:
                    error = caught
            prepared.append(
                _PreparedToolCall(
                    call_id=str(call.call_id),
                    name=name,
                    arguments=arguments,
                    handler_arguments=handler_arguments,
                    preparation_error=error,
                )
            )
        return prepared

    def _execute_tool_call(
        self,
        call: _PreparedToolCall,
    ) -> _ToolCallOutcome:
        if call.preparation_error is not None:
            error = call.preparation_error
            result = (
                error.as_tool_output()
                if isinstance(error, PlayerReferenceError)
                else {"error": str(error)}
            )
            return _ToolCallOutcome(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                result=result,
                succeeded=False,
                error=str(error),
                report_pipeline=None,
            )

        try:
            handler = self.tool_handlers[call.name]

            def execute_handler() -> Any:
                if call.name == "search_reports":
                    with self._report_tool_lock:
                        return handler(**(call.handler_arguments or {}))
                return handler(**(call.handler_arguments or {}))

            try:
                result = execute_handler()
            except Exception as error:
                if not is_transient_supabase_error(error):
                    raise
                LOGGER.info(
                    "Retrying transient tool failure once: %s (%s)",
                    call.name,
                    error,
                )
                clear_supabase_client()
                time.sleep(0.1)
                result = execute_handler()

            if isinstance(result, ToolExecutionResult):
                report_pipeline = _report_pipeline_summary(result.details)
                return _ToolCallOutcome(
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                    result=result.output,
                    succeeded=True,
                    error=None,
                    report_pipeline=report_pipeline,
                    input_tokens=result.input_tokens,
                    cached_input_tokens=min(
                        result.cached_input_tokens,
                        result.input_tokens,
                    ),
                    output_tokens=result.output_tokens,
                    estimated_cost_usd=result.estimated_cost_usd,
                )

            return _ToolCallOutcome(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                result=result,
                succeeded=True,
                error=None,
                report_pipeline=None,
            )
        except Exception as error:
            LOGGER.warning(
                "Agent tool call failed: %s",
                call.name,
                exc_info=True,
            )
            return _ToolCallOutcome(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                result={"error": str(error)},
                succeeded=False,
                error=str(error),
                report_pipeline=None,
            )

    def _execute_tool_batch(
        self,
        calls: list[_PreparedToolCall],
    ) -> list[_ToolCallOutcome]:
        if len(calls) == 1 or self.max_parallel_tools == 1:
            return [self._execute_tool_call(call) for call in calls]

        LOGGER.info(
            "Executing %d independent tool calls with up to %d workers",
            len(calls),
            self.max_parallel_tools,
        )
        outcomes: list[_ToolCallOutcome | None] = [None] * len(calls)
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_tools, len(calls)),
            thread_name_prefix="agent-tool",
        ) as executor:
            futures = {
                executor.submit(self._execute_tool_call, call): index
                for index, call in enumerate(calls)
            }
            for future in as_completed(futures):
                outcomes[futures[future]] = future.result()
        return [outcome for outcome in outcomes if outcome is not None]

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
        estimated_cost_usd = 0.0
        cost_is_complete = True
        if route_telemetry.model is not None:
            route_cost = estimate_text_token_cost_usd(
                model=route_telemetry.model,
                input_tokens=route_telemetry.usage.input_tokens,
                cached_input_tokens=route_telemetry.usage.cached_input_tokens,
                output_tokens=route_telemetry.usage.output_tokens,
            )
            if route_cost is None and (
                route_telemetry.usage.input_tokens
                or route_telemetry.usage.output_tokens
            ):
                cost_is_complete = False
            else:
                estimated_cost_usd += route_cost or 0.0
        tool_telemetry: list[ToolCallTelemetry] = []
        player_reference_cache: dict[
            str, ResolvedPlayerReference | PlayerReferenceError
        ] = {}

        for round_index in range(1, self.max_tool_rounds + 1):
            response = self.client.responses.create(
                model=self.model,
                instructions=build_system_prompt(route),
                tools=tools,
                parallel_tool_calls=True,
                input=input_items,
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                cached_input_tokens = min(_cached_tokens(usage), input_tokens)
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                total_input_tokens += input_tokens
                total_cached_input_tokens += cached_input_tokens
                total_output_tokens += output_tokens
                response_cost = estimate_text_token_cost_usd(
                    model=self.model,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    output_tokens=output_tokens,
                )
                if response_cost is None and (input_tokens or output_tokens):
                    cost_is_complete = False
                else:
                    estimated_cost_usd += response_cost or 0.0

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
                    estimated_cost_usd=(
                        estimated_cost_usd if cost_is_complete else None
                    ),
                )

            prepared_calls = self._prepare_tool_calls(
                tool_calls,
                source_question=prompt,
                player_cache=player_reference_cache,
            )
            outcomes = self._execute_tool_batch(prepared_calls)
            for outcome in outcomes:
                total_input_tokens += outcome.input_tokens
                total_cached_input_tokens += outcome.cached_input_tokens
                total_output_tokens += outcome.output_tokens
                if outcome.estimated_cost_usd is None and (
                    outcome.input_tokens or outcome.output_tokens
                ):
                    cost_is_complete = False
                else:
                    estimated_cost_usd += outcome.estimated_cost_usd or 0.0
                tool_telemetry.append(
                    ToolCallTelemetry(
                        name=outcome.name,
                        arguments=outcome.arguments,
                        succeeded=outcome.succeeded,
                        error=outcome.error,
                        report_pipeline=outcome.report_pipeline,
                    )
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": outcome.call_id,
                        "output": json.dumps(outcome.result, default=str),
                    }
                )

        raise RuntimeError("Tool-call limit reached before a final response")
