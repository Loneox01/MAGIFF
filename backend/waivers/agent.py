"""Two-stage, read-only waiver advisor with mandatory news verification."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from model_costs import estimate_text_token_cost_usd
from prompts import WAIVER_AGENT_INSTRUCTIONS, WAIVER_FINALIZATION_INSTRUCTIONS
from services.news import NewsOutcome
from tools.base import ToolExecutionResult
from tools.reports import SEARCH_REPORTS_TOOL, search_reports

from .models import (
    PreliminaryWaiverAnalysis,
    WaiverAnalysis,
    WaiverContext,
)
from .tools import WaiverToolbox


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_WAIVER_MODEL = os.getenv(
    "OPENAI_WAIVER_MODEL",
    os.getenv("OPENAI_AGENT_MODEL", "gpt-5.6-terra"),
)


@dataclass(frozen=True)
class WaiverTokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class WaiverToolCall:
    name: str
    arguments: dict[str, Any]
    succeeded: bool
    error: str | None
    report_status: str | None = None
    automatic: bool = False


@dataclass(frozen=True)
class WaiverRunResult:
    analysis: WaiverAnalysis
    preliminary: PreliminaryWaiverAnalysis
    model: str
    latency_seconds: float
    tool_rounds: int
    usage: WaiverTokenUsage
    estimated_cost_usd: float | None
    tool_calls: tuple[WaiverToolCall, ...]
    news_evidence: dict[str, Any]


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _cached_tokens(usage: Any) -> int:
    details = _value(usage, "input_tokens_details")
    return int(_value(details, "cached_tokens", 0) or 0)


def _name_key(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)


class _UsageAccumulator:
    def __init__(self, model: str) -> None:
        self.model = model
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.cost_complete = True

    def add_response(self, response: Any) -> None:
        usage = _value(response, "usage")
        if usage is None:
            return
        input_tokens = int(_value(usage, "input_tokens", 0) or 0)
        cached_tokens = min(_cached_tokens(usage), input_tokens)
        output_tokens = int(_value(usage, "output_tokens", 0) or 0)
        self.input_tokens += input_tokens
        self.cached_input_tokens += cached_tokens
        self.output_tokens += output_tokens
        estimated = estimate_text_token_cost_usd(
            model=self.model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
        )
        if estimated is None and (input_tokens or output_tokens):
            self.cost_complete = False
        else:
            self.cost += estimated or 0.0

    def add_tool_result(self, result: ToolExecutionResult) -> None:
        self.input_tokens += result.input_tokens
        self.cached_input_tokens += min(
            result.cached_input_tokens,
            result.input_tokens,
        )
        self.output_tokens += result.output_tokens
        if result.estimated_cost_usd is None and (
            result.input_tokens or result.output_tokens
        ):
            self.cost_complete = False
        else:
            self.cost += result.estimated_cost_usd or 0.0


class WaiverAgentService:
    """Explore a waiver pool, verify finalist news, and return typed advice."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = DEFAULT_WAIVER_MODEL,
        max_discovery_rounds: int = 5,
        max_report_searches: int = 2,
        report_search=search_reports,
    ) -> None:
        if not 1 <= max_discovery_rounds <= 8:
            raise ValueError("max_discovery_rounds must be between 1 and 8")
        if not 0 <= max_report_searches <= 4:
            raise ValueError("max_report_searches must be between 0 and 4")
        self._client = client
        self.model = model
        self.max_discovery_rounds = max_discovery_rounds
        self.max_report_searches = max_report_searches
        self.report_search = report_search

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def _run_preliminary(
        self,
        *,
        context: WaiverContext,
        toolbox: WaiverToolbox,
        question: str,
        usage: _UsageAccumulator,
        telemetry: list[WaiverToolCall],
    ) -> tuple[PreliminaryWaiverAnalysis, list[Any], int]:
        input_items: list[Any] = [
            {
                "role": "user",
                "content": (
                    f"Waiver question:\n{question}\n\n"
                    "Verified waiver snapshot:\n"
                    f"{json.dumps(context.agent_payload(), separators=(',', ':'))}"
                ),
            }
        ]
        tools = [*toolbox.schemas, SEARCH_REPORTS_TOOL]
        handlers = toolbox.handlers
        cache: dict[tuple[str, str], Any] = {}
        report_searches = 0

        for round_index in range(1, self.max_discovery_rounds + 1):
            response = self.client.responses.parse(
                model=self.model,
                instructions=WAIVER_AGENT_INSTRUCTIONS,
                tools=tools,
                parallel_tool_calls=True,
                input=input_items,
                text_format=PreliminaryWaiverAnalysis,
            )
            usage.add_response(response)
            response_output = list(_value(response, "output", []) or [])
            calls = [
                item
                for item in response_output
                if _value(item, "type") == "function_call"
            ]
            if not calls:
                parsed = _value(response, "output_parsed")
                if parsed is None:
                    raise RuntimeError(
                        "Waiver advisor returned no preliminary structured output"
                    )
                preliminary = PreliminaryWaiverAnalysis.model_validate(parsed)
                if len(preliminary.shortlist) > 5:
                    raise RuntimeError("Waiver preliminary shortlist exceeded five")
                # Parsed response objects contain SDK-only Pydantic metadata
                # that should not be serialized back into a later request.
                input_items.append(
                    {
                        "role": "assistant",
                        "content": preliminary.model_dump_json(),
                    }
                )
                return preliminary, input_items, round_index - 1

            input_items.extend(response_output)

            for call in calls:
                name = str(_value(call, "name", ""))
                call_id = str(_value(call, "call_id", ""))
                arguments: dict[str, Any] = {}
                report_status = None
                try:
                    decoded = json.loads(str(_value(call, "arguments", "{}")))
                    if not isinstance(decoded, dict):
                        raise ValueError("tool arguments must be an object")
                    arguments = decoded
                    cache_key = (
                        name,
                        json.dumps(arguments, sort_keys=True, separators=(",", ":")),
                    )
                    if cache_key in cache:
                        output = cache[cache_key]
                    elif name == "search_reports":
                        if report_searches >= self.max_report_searches:
                            raise RuntimeError(
                                "Waiver report-search budget exhausted; continue from "
                                "the evidence already returned."
                            )
                        report_searches += 1
                        result = self.report_search(
                            str(arguments["query"]),
                            int(arguments["limit"]),
                            source_question=question,
                        )
                        usage.add_tool_result(result)
                        output = result.output
                        report_status = str(result.details.get("status") or "") or None
                        cache[cache_key] = output
                    else:
                        handler = handlers.get(name)
                        if handler is None:
                            raise ValueError(f"unsupported waiver tool: {name}")
                        output = handler(**arguments)
                        cache[cache_key] = output
                    telemetry.append(
                        WaiverToolCall(
                            name=name,
                            arguments=arguments,
                            succeeded=True,
                            error=None,
                            report_status=report_status,
                        )
                    )
                except Exception as error:
                    output = {"error": str(error)}
                    telemetry.append(
                        WaiverToolCall(
                            name=name,
                            arguments=arguments,
                            succeeded=False,
                            error=str(error),
                            report_status=report_status,
                        )
                    )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(output, default=str),
                    }
                )
        raise RuntimeError("Waiver advisor exceeded its discovery-round limit")

    @staticmethod
    def _verification_names(
        preliminary: PreliminaryWaiverAnalysis,
    ) -> list[str]:
        values: list[str] = []
        for move in preliminary.shortlist:
            values.append(move.add_player)
            if move.drop_player:
                values.append(move.drop_player)
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    def _verify_news(
        self,
        *,
        toolbox: WaiverToolbox,
        preliminary: PreliminaryWaiverAnalysis,
        telemetry: list[WaiverToolCall],
    ) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        for name in self._verification_names(preliminary):
            arguments = toolbox.news_arguments_for(name, limit=3)
            try:
                output = toolbox.get_recent_news(**arguments)
                succeeded = output.get("status") not in {
                    NewsOutcome.PLAYER_NOT_FOUND.value,
                    NewsOutcome.PLAYER_AMBIGUOUS.value,
                }
                error = None if succeeded else str(output.get("status"))
            except Exception as caught:
                output = {"status": "error", "error": str(caught), "reports": []}
                succeeded = False
                error = str(caught)
            evidence[name] = output
            telemetry.append(
                WaiverToolCall(
                    name="get_recent_news",
                    arguments=arguments,
                    succeeded=succeeded,
                    error=error,
                    automatic=True,
                )
            )
        return evidence

    @staticmethod
    def _validate_final_names(
        preliminary: PreliminaryWaiverAnalysis,
        analysis: WaiverAnalysis,
    ) -> None:
        verified = {
            _name_key(name)
            for name in WaiverAgentService._verification_names(preliminary)
        }
        for recommendation in analysis.recommendations:
            for field, name in (
                ("add_player", recommendation.add_player),
                ("drop_player", recommendation.drop_player),
            ):
                if name is not None and _name_key(name) not in verified:
                    raise RuntimeError(
                        f"Final waiver {field} {name!r} bypassed preliminary news verification"
                    )

    def run(
        self,
        context: WaiverContext,
        question: str = "Review my waiver options and recommend any worthwhile moves.",
        *,
        toolbox: WaiverToolbox | None = None,
    ) -> WaiverRunResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not context.available_players:
            raise ValueError("waiver context has no available market players")

        started_at = time.perf_counter()
        active_toolbox = toolbox or WaiverToolbox(context)
        usage = _UsageAccumulator(self.model)
        telemetry: list[WaiverToolCall] = []
        preliminary, input_items, tool_rounds = self._run_preliminary(
            context=context,
            toolbox=active_toolbox,
            question=normalized_question,
            usage=usage,
            telemetry=telemetry,
        )
        news_evidence = self._verify_news(
            toolbox=active_toolbox,
            preliminary=preliminary,
            telemetry=telemetry,
        )
        input_items.append(
            {
                "role": "user",
                "content": (
                    "Automatic newest-first maintained-news verification for every "
                    "preliminary add and drop follows. Finalize the analysis using "
                    "only verified preliminary names.\n"
                    f"{json.dumps(news_evidence, separators=(',', ':'), default=str)}"
                ),
            }
        )
        final_response = self.client.responses.parse(
            model=self.model,
            instructions=WAIVER_FINALIZATION_INSTRUCTIONS,
            input=input_items,
            text_format=WaiverAnalysis,
        )
        usage.add_response(final_response)
        parsed = _value(final_response, "output_parsed")
        if parsed is None:
            raise RuntimeError("Waiver advisor returned no final structured output")
        analysis = WaiverAnalysis.model_validate(parsed)
        if len(analysis.recommendations) > 5:
            raise RuntimeError("Waiver analysis exceeded five recommendations")
        self._validate_final_names(preliminary, analysis)

        return WaiverRunResult(
            analysis=analysis,
            preliminary=preliminary,
            model=self.model,
            latency_seconds=time.perf_counter() - started_at,
            tool_rounds=tool_rounds,
            usage=WaiverTokenUsage(
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
            ),
            estimated_cost_usd=usage.cost if usage.cost_complete else None,
            tool_calls=tuple(telemetry),
            news_evidence=news_evidence,
        )
