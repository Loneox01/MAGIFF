"""A small report-only model loop dedicated to one draft decision."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from model_costs import estimate_text_token_cost_usd
from prompts import DRAFT_AGENT_INSTRUCTIONS
from tools.base import ToolExecutionResult
from tools.reports import SEARCH_REPORTS_TOOL, search_reports

from .models import DraftContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_DRAFT_MODEL = os.getenv(
    "OPENAI_DRAFT_MODEL",
    os.getenv("OPENAI_AGENT_MODEL", "gpt-5.6-terra"),
)


@dataclass(frozen=True)
class DraftTokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class DraftReportCall:
    query: str
    limit: int
    succeeded: bool
    status: str | None
    error: str | None


@dataclass(frozen=True)
class DraftRunResult:
    answer: str
    model: str
    latency_seconds: float
    tool_rounds: int
    usage: DraftTokenUsage
    estimated_cost_usd: float | None
    report_calls: tuple[DraftReportCall, ...]


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _cached_tokens(usage: Any) -> int:
    details = _value(usage, "input_tokens_details")
    return int(_value(details, "cached_tokens", 0) or 0)


def _answer_text(response: Any) -> str:
    direct = str(_value(response, "output_text", "") or "").strip()
    if direct:
        return direct
    blocks = []
    for item in list(_value(response, "output", []) or []):
        if _value(item, "type") != "message":
            continue
        for block in list(_value(item, "content", []) or []):
            if _value(block, "type") == "output_text":
                text = str(_value(block, "text", "") or "").strip()
                if text:
                    blocks.append(text)
    return "\n\n".join(blocks)


class DraftAgentService:
    """Answer from one immutable draft snapshot with bounded report research."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        report_search: Callable[..., ToolExecutionResult] = search_reports,
        model: str = DEFAULT_DRAFT_MODEL,
        max_report_searches: int = 2,
        max_rounds: int = 4,
    ) -> None:
        if not 0 <= max_report_searches <= 4:
            raise ValueError("max_report_searches must be between 0 and 4")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self._client = client
        self.report_search = report_search
        self.model = model
        self.max_report_searches = max_report_searches
        self.max_rounds = max_rounds

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def run(
        self,
        context: DraftContext,
        question: str = "Who should I draft here and why?",
    ) -> DraftRunResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not context.available_candidates:
            raise ValueError("draft context has no available candidates")

        started_at = time.perf_counter()
        input_items: list[Any] = [
            {
                "role": "user",
                "content": (
                    f"Draft question:\n{normalized_question}\n\n"
                    "Verified draft snapshot:\n"
                    f"{json.dumps(context.agent_payload(), separators=(',', ':'))}"
                ),
            }
        ]
        total_input = 0
        total_cached = 0
        total_output = 0
        estimated_cost = 0.0
        cost_complete = True
        report_calls: list[DraftReportCall] = []
        report_cache: dict[tuple[str, int], ToolExecutionResult] = {}
        searches_used = 0

        for round_index in range(1, self.max_rounds + 1):
            response = self.client.responses.create(
                model=self.model,
                instructions=DRAFT_AGENT_INSTRUCTIONS,
                tools=(
                    [SEARCH_REPORTS_TOOL]
                    if self.max_report_searches > 0
                    else []
                ),
                parallel_tool_calls=True,
                input=input_items,
            )
            usage = _value(response, "usage")
            if usage is not None:
                input_tokens = int(_value(usage, "input_tokens", 0) or 0)
                cached_tokens = min(_cached_tokens(usage), input_tokens)
                output_tokens = int(_value(usage, "output_tokens", 0) or 0)
                total_input += input_tokens
                total_cached += cached_tokens
                total_output += output_tokens
                cost = estimate_text_token_cost_usd(
                    model=self.model,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_tokens,
                    output_tokens=output_tokens,
                )
                if cost is None and (input_tokens or output_tokens):
                    cost_complete = False
                else:
                    estimated_cost += cost or 0.0

            response_output = list(_value(response, "output", []) or [])
            input_items.extend(response_output)
            tool_calls = [
                item
                for item in response_output
                if _value(item, "type") == "function_call"
            ]
            if not tool_calls:
                answer = _answer_text(response)
                if not answer:
                    raise RuntimeError("Draft agent returned no answer")
                return DraftRunResult(
                    answer=answer,
                    model=self.model,
                    latency_seconds=time.perf_counter() - started_at,
                    tool_rounds=round_index - 1,
                    usage=DraftTokenUsage(
                        input_tokens=total_input,
                        cached_input_tokens=total_cached,
                        output_tokens=total_output,
                    ),
                    estimated_cost_usd=(
                        estimated_cost if cost_complete else None
                    ),
                    report_calls=tuple(report_calls),
                )

            for call in tool_calls:
                call_id = str(_value(call, "call_id", ""))
                query = ""
                limit = 0
                try:
                    if _value(call, "name") != "search_reports":
                        raise ValueError("The draft advisor only supports search_reports")
                    arguments = json.loads(str(_value(call, "arguments", "{}")))
                    query = str(arguments["query"]).strip()
                    limit = int(arguments["limit"])
                    if not query or not 1 <= limit <= 5:
                        raise ValueError("query and limit must satisfy the tool schema")
                    key = (query, limit)
                    result = report_cache.get(key)
                    if result is None:
                        if searches_used >= self.max_report_searches:
                            raise RuntimeError(
                                "Draft report-search budget exhausted; answer from "
                                "the verified snapshot and evidence already returned."
                            )
                        searches_used += 1
                        result = self.report_search(
                            query,
                            limit,
                            source_question=normalized_question,
                        )
                        report_cache[key] = result
                        total_input += result.input_tokens
                        total_cached += min(
                            result.cached_input_tokens,
                            result.input_tokens,
                        )
                        total_output += result.output_tokens
                        if result.estimated_cost_usd is None and (
                            result.input_tokens or result.output_tokens
                        ):
                            cost_complete = False
                        else:
                            estimated_cost += result.estimated_cost_usd or 0.0
                    status = str(result.details.get("status") or "") or None
                    report_calls.append(
                        DraftReportCall(
                            query=query,
                            limit=limit,
                            succeeded=True,
                            status=status,
                            error=None,
                        )
                    )
                    output = result.output
                except Exception as error:
                    report_calls.append(
                        DraftReportCall(
                            query=query,
                            limit=limit,
                            succeeded=False,
                            status=None,
                            error=str(error),
                        )
                    )
                    output = {"error": str(error)}
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(output, default=str),
                    }
                )

        raise RuntimeError("Draft agent reached its bounded tool-call limit")
