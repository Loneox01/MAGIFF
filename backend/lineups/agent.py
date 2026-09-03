"""Two-stage, read-only weekly lineup advisor with news verification."""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from model_costs import estimate_text_token_cost_usd
from prompts import LINEUP_AGENT_INSTRUCTIONS, LINEUP_FINALIZATION_INSTRUCTIONS
from services.news import NewsOutcome
from tools.base import ToolExecutionResult
from tools.reports import SEARCH_REPORTS_TOOL, search_reports

from .models import (
    LineupAnalysis,
    LineupChange,
    LineupContext,
    PreliminaryLineupPlan,
    ProposedStarter,
    ValidatedLineup,
)
from .tools import LineupToolbox


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_LINEUP_MODEL = os.getenv(
    "OPENAI_LINEUP_MODEL",
    os.getenv("OPENAI_AGENT_MODEL", "gpt-5.6-terra"),
)


@dataclass(frozen=True)
class LineupTokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LineupToolCall:
    name: str
    arguments: dict[str, Any]
    succeeded: bool
    error: str | None
    report_status: str | None = None
    automatic: bool = False


@dataclass(frozen=True)
class LineupRunResult:
    analysis: LineupAnalysis
    preliminary: PreliminaryLineupPlan
    validated: ValidatedLineup
    model: str
    latency_seconds: float
    tool_rounds: int
    usage: LineupTokenUsage
    estimated_cost_usd: float | None
    tool_calls: tuple[LineupToolCall, ...]
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


def validate_lineup(
    context: LineupContext,
    starters: list[ProposedStarter],
) -> ValidatedLineup:
    expected = {slot.slot_id: slot for slot in context.slots}
    provided = {starter.slot_id: starter for starter in starters}
    if len(provided) != len(starters):
        raise RuntimeError("Lineup contains duplicate slot IDs")
    if set(provided) != set(expected):
        missing = sorted(set(expected) - set(provided))
        extra = sorted(set(provided) - set(expected))
        raise RuntimeError(
            f"Lineup slot mismatch; missing={missing}, unsupported={extra}"
        )

    players = context.player_by_id
    used: set[str] = set()
    projected_total = 0.0
    changes = []
    for slot in context.slots:
        proposed = provided[slot.slot_id]
        player = None
        if proposed.sleeper_player_id is not None:
            player = players.get(proposed.sleeper_player_id)
            if player is None:
                raise RuntimeError(
                    f"Lineup player {proposed.sleeper_player_id!r} is not on the roster"
                )
            if proposed.sleeper_player_id in used:
                raise RuntimeError(
                    f"Lineup uses {player.display_name!r} more than once"
                )
            if proposed.player_name is None or _name_key(proposed.player_name) != _name_key(
                player.display_name
            ):
                raise RuntimeError(
                    f"Lineup name does not match Sleeper ID {player.sleeper_player_id}"
                )
            if not player.can_fill(slot):
                raise RuntimeError(
                    f"{player.display_name} cannot enter the lineup from "
                    f"{player.roster_group} with designation {player.injury_code} "
                    f"and locked={player.is_locked}"
                )
            used.add(player.sleeper_player_id)
            projected_total += float(player.projected_points or 0)
        else:
            fillable = any(
                candidate.sleeper_player_id not in used
                and candidate.can_fill(slot)
                for candidate in context.players
            )
            if fillable:
                raise RuntimeError(f"Lineup left fillable slot {slot.slot_id} empty")

        if proposed.sleeper_player_id != slot.current_player_id:
            outgoing = players.get(slot.current_player_id or "")
            changes.append(
                LineupChange(
                    slot_id=slot.slot_id,
                    outgoing_player_id=(
                        outgoing.sleeper_player_id if outgoing else None
                    ),
                    outgoing_player=(outgoing.display_name if outgoing else None),
                    incoming_player_id=(
                        player.sleeper_player_id if player else None
                    ),
                    incoming_player=(player.display_name if player else None),
                )
            )
    return ValidatedLineup(
        projected_total=round(projected_total, 2),
        changes=tuple(changes),
    )


class LineupAgentService:
    """Research and validate one legal weekly lineup recommendation."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = DEFAULT_LINEUP_MODEL,
        max_discovery_rounds: int = 4,
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
        context: LineupContext,
        toolbox: LineupToolbox,
        question: str,
        usage: _UsageAccumulator,
        telemetry: list[LineupToolCall],
    ) -> tuple[PreliminaryLineupPlan, list[Any], int]:
        input_items: list[Any] = [
            {
                "role": "user",
                "content": (
                    f"Lineup question:\n{question}\n\n"
                    "Verified lineup snapshot:\n"
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
                instructions=LINEUP_AGENT_INSTRUCTIONS,
                tools=tools,
                parallel_tool_calls=True,
                input=input_items,
                text_format=PreliminaryLineupPlan,
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
                    raise RuntimeError("Lineup advisor returned no preliminary plan")
                preliminary = PreliminaryLineupPlan.model_validate(parsed)
                if preliminary.week != context.week:
                    raise RuntimeError("Preliminary lineup used the wrong week")
                validate_lineup(context, preliminary.starters)
                input_items.append(
                    {
                        "role": "assistant",
                        "content": preliminary.model_dump_json(),
                    }
                )
                return preliminary, input_items, round_index - 1

            input_items.extend(response_output)
            prepared = []
            for call in calls:
                name = str(_value(call, "name", ""))
                call_id = str(_value(call, "call_id", ""))
                try:
                    arguments = json.loads(str(_value(call, "arguments", "{}")))
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                    cache_key = (
                        name,
                        json.dumps(arguments, sort_keys=True, separators=(",", ":")),
                    )
                    cached = cache.get(cache_key)
                    if cached is not None:
                        prepared.append((call_id, name, arguments, cache_key, cached, None))
                        continue
                    if name == "search_reports":
                        if report_searches >= self.max_report_searches:
                            raise RuntimeError(
                                "Lineup report-search budget exhausted; continue from returned evidence."
                            )
                        report_searches += 1
                    elif name not in handlers:
                        raise ValueError(f"unsupported lineup tool: {name}")
                    prepared.append((call_id, name, arguments, cache_key, None, None))
                except Exception as error:
                    prepared.append((call_id, name, {}, None, None, error))

            def execute(item):
                call_id, name, arguments, cache_key, cached, prior_error = item
                if prior_error is not None:
                    raise prior_error
                if cached is not None:
                    return cached, None
                if name == "search_reports":
                    result = self.report_search(
                        str(arguments["query"]),
                        int(arguments["limit"]),
                        source_question=question,
                    )
                    return result.output, result
                return handlers[name](**arguments), None

            with ThreadPoolExecutor(
                max_workers=max(1, min(8, len(prepared)))
            ) as executor:
                futures = [executor.submit(execute, item) for item in prepared]
                for item, future in zip(prepared, futures, strict=True):
                    call_id, name, arguments, cache_key, _, _ = item
                    report_status = None
                    try:
                        output, tool_result = future.result()
                        if cache_key is not None:
                            cache[cache_key] = output
                        if tool_result is not None:
                            usage.add_tool_result(tool_result)
                            report_status = (
                                str(tool_result.details.get("status") or "") or None
                            )
                        telemetry.append(
                            LineupToolCall(
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
                            LineupToolCall(
                                name=name,
                                arguments=arguments,
                                succeeded=False,
                                error=str(error),
                            )
                        )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(output, default=str),
                        }
                    )
        raise RuntimeError("Lineup advisor exceeded its discovery-round limit")

    @staticmethod
    def _verification_ids(
        context: LineupContext,
        preliminary: PreliminaryLineupPlan,
    ) -> list[str]:
        verified = context.player_by_id
        requested = [
            player_id
            for player_id in preliminary.news_check_player_ids
            if player_id in verified
        ]
        proposed_by_slot = {
            starter.slot_id: starter.sleeper_player_id
            for starter in preliminary.starters
        }
        for slot in context.slots:
            proposed = proposed_by_slot.get(slot.slot_id)
            if proposed != slot.current_player_id:
                if proposed:
                    requested.append(proposed)
                if slot.current_player_id:
                    requested.append(slot.current_player_id)
        requested.extend(
            player.sleeper_player_id
            for player in context.players
            if player.injury_code is not None
        )
        return list(dict.fromkeys(requested))[:12]

    def _verify_news(
        self,
        *,
        context: LineupContext,
        toolbox: LineupToolbox,
        preliminary: PreliminaryLineupPlan,
        telemetry: list[LineupToolCall],
    ) -> dict[str, Any]:
        player_ids = self._verification_ids(context, preliminary)

        def fetch(player_id: str) -> tuple[str, dict[str, Any], bool, str | None]:
            try:
                output = toolbox.automatic_news(player_id, limit=3)
                succeeded = output.get("status") not in {
                    NewsOutcome.PLAYER_NOT_FOUND.value,
                    NewsOutcome.PLAYER_AMBIGUOUS.value,
                }
                return player_id, output, succeeded, (
                    None if succeeded else str(output.get("status"))
                )
            except Exception as error:
                return (
                    player_id,
                    {"status": "error", "error": str(error), "reports": []},
                    False,
                    str(error),
                )

        evidence: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(6, len(player_ids)))) as executor:
            results = list(executor.map(fetch, player_ids)) if player_ids else []
        for player_id, output, succeeded, error in results:
            evidence[player_id] = output
            player = context.player_by_id[player_id]
            telemetry.append(
                LineupToolCall(
                    name="get_recent_news",
                    arguments={"player_ref": player.display_name, "limit": 3},
                    succeeded=succeeded,
                    error=error,
                    automatic=True,
                )
            )
        return evidence

    @staticmethod
    def _validate_close_calls(context: LineupContext, analysis: LineupAnalysis) -> None:
        player_ids = set(context.player_by_id)
        for close_call in analysis.close_calls:
            if close_call.selected_player_id not in player_ids:
                raise RuntimeError("Close-call selected player is not on the roster")
            if close_call.alternative_player_id not in player_ids:
                raise RuntimeError("Close-call alternative is not on the roster")
            if close_call.selected_player_id == close_call.alternative_player_id:
                raise RuntimeError("Close-call players must be different")

    def run(
        self,
        context: LineupContext,
        question: str = "Set my best legal lineup for this week.",
        *,
        toolbox: LineupToolbox | None = None,
    ) -> LineupRunResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not context.players or not context.slots:
            raise ValueError("lineup context has no roster or starter slots")

        started_at = time.perf_counter()
        active_toolbox = toolbox or LineupToolbox(context)
        usage = _UsageAccumulator(self.model)
        telemetry: list[LineupToolCall] = []
        preliminary, input_items, tool_rounds = self._run_preliminary(
            context=context,
            toolbox=active_toolbox,
            question=normalized_question,
            usage=usage,
            telemetry=telemetry,
        )
        news_evidence = self._verify_news(
            context=context,
            toolbox=active_toolbox,
            preliminary=preliminary,
            telemetry=telemetry,
        )
        input_items.append(
            {
                "role": "user",
                "content": (
                    "Automatic newest-first maintained-news checks for changed, "
                    "close, and health-designated roster players follow. Finalize "
                    "one legal lineup from the verified roster only.\n"
                    f"{json.dumps(news_evidence, separators=(',', ':'), default=str)}"
                ),
            }
        )
        final_response = self.client.responses.parse(
            model=self.model,
            instructions=LINEUP_FINALIZATION_INSTRUCTIONS,
            input=input_items,
            text_format=LineupAnalysis,
        )
        usage.add_response(final_response)
        parsed = _value(final_response, "output_parsed")
        if parsed is None:
            raise RuntimeError("Lineup advisor returned no final structured output")
        analysis = LineupAnalysis.model_validate(parsed)
        if analysis.week != context.week:
            raise RuntimeError("Final lineup used the wrong week")
        validated = validate_lineup(context, analysis.starters)
        self._validate_close_calls(context, analysis)

        return LineupRunResult(
            analysis=analysis,
            preliminary=preliminary,
            validated=validated,
            model=self.model,
            latency_seconds=time.perf_counter() - started_at,
            tool_rounds=tool_rounds,
            usage=LineupTokenUsage(
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
            ),
            estimated_cost_usd=usage.cost if usage.cost_complete else None,
            tool_calls=tuple(telemetry),
            news_evidence=news_evidence,
        )
