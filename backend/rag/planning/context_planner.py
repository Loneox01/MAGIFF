"""Second-stage planning for indirect report evidence.

Luna's direct planner owns literal query interpretation and target selection.
This module gives Terra only the narrower job of identifying material evidence
that may omit those targets, then merges the result into the stable QueryPlan
contract consumed by resolution, retrieval, and reranking.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from prompts import CONTEXT_REPORT_PLANNER_INSTRUCTIONS

from ..config import DEFAULT_CONTEXT_PLANNER_MODEL, DEFAULT_INDEX_PATH
from .lookups import ContextScopePolicy, LookupPurpose
from .planner import ContextRequest, QueryPlan
from .resolver import ResolutionResult


CONTEXT_PLANNER_PROMPT_VERSION = "2"


class _CorrectableContextPlanError(ValueError):
    """A Terra response exists but cannot satisfy the context-plan contract."""


class ContextPlannerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextPlan(ContextPlannerModel):
    context_needed: bool
    rationale: str = Field(
        min_length=1,
        max_length=320,
        description=(
            "Concise explanation of why indirect evidence is required or why "
            "the direct plan is sufficient."
        ),
    )
    context_requests: list[ContextRequest] = Field(
        max_length=3,
        description=(
            "Indirect retrieval branches only. Never repeat direct targets."
        ),
    )

    @model_validator(mode="after")
    def _validate_need_matches_requests(self):
        if self.context_needed and not self.context_requests:
            raise ValueError(
                "context_needed requires at least one context request"
            )
        if not self.context_needed and self.context_requests:
            raise ValueError(
                "context requests must be empty when context_needed is false"
            )
        return self


@dataclass(frozen=True)
class ContextPlanResult:
    context_plan: ContextPlan
    plan: QueryPlan
    model: str
    cached: bool
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    attempts: int = 1
    retried: bool = False
    retry_reason: str | None = None


def _compact_resolution(
    resolution: ResolutionResult,
    *,
    selector_count: int,
) -> dict[str, object]:
    selectors: list[dict[str, object]] = []
    for item in resolution.selectors:
        # The resolver may add compatibility selectors for uncovered mention
        # fields. Context anchors must reference the actual direct-plan array,
        # so those internal extras are intentionally hidden from Terra.
        if item.selector_index >= selector_count:
            continue
        selectors.append(
            {
                "selector_index": item.selector_index,
                "selector": item.selector.model_dump(mode="json"),
                "status": item.status,
                "matches": [
                    match.model_dump(mode="json") for match in item.matches[:20]
                ],
                "unresolved_filters": item.unresolved_filters,
                "truncated": item.truncated,
            }
        )
    return {"selectors": selectors}


def merge_context_plan(
    direct_plan: QueryPlan,
    context_plan: ContextPlan,
) -> QueryPlan:
    """Normalize safe scope fallbacks, then validate both model outputs."""
    context_requests: list[dict[str, object]] = []
    for request in context_plan.context_requests:
        scope_policy = request.scope_policy
        has_scope_lookup = any(
            lookup.purpose
            in {
                LookupPurpose.RESOLVE_RELATIONSHIP,
                LookupPurpose.EXPAND_CANDIDATES,
            }
            for lookup in request.structured_lookups
        )
        if not has_scope_lookup:
            if scope_policy == ContextScopePolicy.LOOKUP_ENTITIES:
                scope_policy = ContextScopePolicy.SEMANTIC_ONLY
            elif scope_policy == ContextScopePolicy.ANCHOR_AND_LOOKUP_TEAMS:
                scope_policy = ContextScopePolicy.ANCHOR_TEAMS

        payload = request.model_dump(mode="json")
        payload["scope_policy"] = scope_policy.value
        context_requests.append(payload)

    return QueryPlan.model_validate(
        {
            **direct_plan.model_dump(mode="json"),
            "context_requests": context_requests,
        }
    )


class ContextPlanner:
    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        model: str = DEFAULT_CONTEXT_PLANNER_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.model = model
        self.client = client

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.index_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS context_plans (
                        context_hash TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        planning_date TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (context_hash, model, prompt_version)
                    )
                    """
                )
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _context_hash(
        query: str,
        planning_date: date,
        direct_plan: QueryPlan,
        resolution: ResolutionResult,
    ) -> str:
        cache_input = json.dumps(
            {
                "planning_date": planning_date.isoformat(),
                "query": query.strip(),
                "direct_plan": direct_plan.model_dump(
                    mode="json",
                    exclude={"context_requests"},
                ),
                "resolution": _compact_resolution(
                    resolution,
                    selector_count=len(direct_plan.entity_selectors),
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()

    def _cached_plan(self, context_hash: str) -> ContextPlan | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT plan_json
                FROM context_plans
                WHERE context_hash = ? AND model = ? AND prompt_version = ?
                """,
                (
                    context_hash,
                    self.model,
                    CONTEXT_PLANNER_PROMPT_VERSION,
                ),
            ).fetchone()
        if row is None:
            return None
        return ContextPlan.model_validate_json(row["plan_json"])

    def _cache_plan(
        self,
        context_hash: str,
        planning_date: date,
        context_plan: ContextPlan,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO context_plans (
                    context_hash, model, prompt_version, planning_date, plan_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    context_hash,
                    self.model,
                    CONTEXT_PLANNER_PROMPT_VERSION,
                    planning_date.isoformat(),
                    context_plan.model_dump_json(),
                ),
            )
            connection.commit()

    def _request_plan(
        self,
        client: OpenAI,
        query: str,
        current_date: date,
        direct_plan: QueryPlan,
        resolution: ResolutionResult,
        *,
        correction_error: str | None = None,
        previous_context_plan: ContextPlan | None = None,
    ) -> tuple[ContextPlan, int, int, int]:
        payload = {
            "current_date": current_date.isoformat(),
            "question": query,
            "direct_plan": direct_plan.model_dump(
                mode="json",
                exclude={"context_requests"},
            ),
            "grounded_resolution": _compact_resolution(
                resolution,
                selector_count=len(direct_plan.entity_selectors),
            ),
        }
        if correction_error is not None:
            payload["correction"] = {
                "instruction": (
                    "The prior context output was malformed or incompatible "
                    "with the fixed direct plan. Correct only the context "
                    "output using the validation feedback below."
                ),
                "validation_error": correction_error[:2400],
                "previous_context_plan": (
                    previous_context_plan.model_dump(mode="json")
                    if previous_context_plan is not None
                    else None
                ),
            }
        response = client.responses.parse(
            model=self.model,
            reasoning={"effort": "none"},
            input=[
                {
                    "role": "system",
                    "content": CONTEXT_REPORT_PLANNER_INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, separators=(",", ":")),
                },
            ],
            text_format=ContextPlan,
        )
        context_plan = response.output_parsed
        if context_plan is None:
            raise _CorrectableContextPlanError(
                "Context planner returned no structured plan"
            )

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        cached_tokens = 0
        if usage is not None:
            details = getattr(usage, "input_tokens_details", None)
            cached_tokens = getattr(details, "cached_tokens", 0) or 0
        output_tokens = usage.output_tokens if usage else 0
        return context_plan, input_tokens, cached_tokens, output_tokens

    def expand(
        self,
        query: str,
        direct_plan: QueryPlan,
        resolution: ResolutionResult,
        *,
        planning_date: date | None = None,
        use_cache: bool = True,
    ) -> ContextPlanResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Context planner query must not be empty")
        if direct_plan.context_requests:
            raise ValueError(
                "Context planner requires a direct plan without context requests"
            )

        current_date = planning_date or date.today()
        context_hash = self._context_hash(
            normalized_query,
            current_date,
            direct_plan,
            resolution,
        )
        if use_cache:
            cached_plan = self._cached_plan(context_hash)
            if cached_plan is not None:
                try:
                    combined = merge_context_plan(direct_plan, cached_plan)
                except ValidationError as error:
                    retry_reason = str(error)
                    previous_context_plan = cached_plan
                else:
                    return ContextPlanResult(
                        context_plan=cached_plan,
                        plan=combined,
                        model=self.model,
                        cached=True,
                        input_tokens=0,
                        cached_input_tokens=0,
                        output_tokens=0,
                        attempts=0,
                    )
            else:
                retry_reason = None
                previous_context_plan = None
        else:
            retry_reason = None
            previous_context_plan = None

        input_tokens = 0
        cached_tokens = 0
        output_tokens = 0
        attempts = 0
        try:
            client = self.client or OpenAI()
            if retry_reason is None:
                attempts += 1
                try:
                    (
                        context_plan,
                        request_input_tokens,
                        request_cached_tokens,
                        request_output_tokens,
                    ) = self._request_plan(
                        client,
                        normalized_query,
                        current_date,
                        direct_plan,
                        resolution,
                    )
                except (ValidationError, _CorrectableContextPlanError) as error:
                    retry_reason = str(error)
                    previous_context_plan = None
                else:
                    input_tokens += request_input_tokens
                    cached_tokens += request_cached_tokens
                    output_tokens += request_output_tokens
                    try:
                        combined = merge_context_plan(
                            direct_plan,
                            context_plan,
                        )
                    except ValidationError as error:
                        retry_reason = str(error)
                        previous_context_plan = context_plan

            if retry_reason is not None:
                attempts += 1
                (
                    context_plan,
                    request_input_tokens,
                    request_cached_tokens,
                    request_output_tokens,
                ) = self._request_plan(
                    client,
                    normalized_query,
                    current_date,
                    direct_plan,
                    resolution,
                    correction_error=retry_reason,
                    previous_context_plan=previous_context_plan,
                )
                input_tokens += request_input_tokens
                cached_tokens += request_cached_tokens
                output_tokens += request_output_tokens
                combined = merge_context_plan(direct_plan, context_plan)
        except Exception as error:
            raise RuntimeError(f"Context planner request failed: {error}") from error

        self._cache_plan(context_hash, current_date, context_plan)
        return ContextPlanResult(
            context_plan=context_plan,
            plan=combined,
            model=self.model,
            cached=False,
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            attempts=attempts,
            retried=retry_reason is not None,
            retry_reason=retry_reason,
        )


__all__ = [
    "ContextPlan",
    "ContextPlanResult",
    "ContextPlanner",
    "merge_context_plan",
]
