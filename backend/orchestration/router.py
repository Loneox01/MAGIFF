"""Cheap, universal request planning before capability-specific execution."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from prompts import REQUEST_ROUTER_INSTRUCTIONS

from .config import DEFAULT_ROUTER_INDEX_PATH, DEFAULT_ROUTER_MODEL


ROUTER_PROMPT_VERSION = "2"


class RouterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Capability(StrEnum):
    STRUCTURED_DATA = "structured_data"
    REPORTS = "reports"
    WEB_SEARCH = "web_search"


class StructuredDomain(StrEnum):
    PLAYER_LOOKUP = "player_lookup"
    PLAYER_STATS = "player_stats"
    TEAM_STATS = "team_stats"
    SCHEDULES = "schedules"
    ROSTERS_DEPTH_CHARTS = "rosters_depth_charts"
    ECR = "ecr"


class RequestIntent(StrEnum):
    LOOKUP = "lookup"
    RANKING = "ranking"
    COMPARISON = "comparison"
    NEWS = "news"
    TIMELINE = "timeline"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


class FreshnessRequirement(StrEnum):
    HISTORICAL = "historical"
    CURRENT = "current"
    LIVE = "live"
    UNSPECIFIED = "unspecified"


class RequestRoute(RouterModel):
    request_summary: str = Field(
        description="Concise statement of the complete information need."
    )
    intent: RequestIntent
    freshness: FreshnessRequirement
    capabilities: list[Capability] = Field(min_length=1)
    structured_domains: list[StructuredDomain]
    rationale: str = Field(description="One short explanation of the route.")

    @model_validator(mode="after")
    def validate_domains(self) -> "RequestRoute":
        uses_structured = Capability.STRUCTURED_DATA in self.capabilities
        if uses_structured and not self.structured_domains:
            raise ValueError(
                "structured_data requires at least one structured domain"
            )
        if not uses_structured and self.structured_domains:
            raise ValueError(
                "structured domains require the structured_data capability"
            )
        return self


@dataclass(frozen=True)
class RequestRouteResult:
    route: RequestRoute
    model: str
    cached: bool
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class RequestRouter:
    """Plan one request with Luna and cache the result for the current date."""

    def __init__(
        self,
        index_path: Path = DEFAULT_ROUTER_INDEX_PATH,
        model: str = DEFAULT_ROUTER_MODEL,
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
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS request_routes (
                        request_hash TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        routing_date TEXT NOT NULL,
                        route_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (request_hash, model, prompt_version)
                    );

                    CREATE TABLE IF NOT EXISTS request_route_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_hash TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        cache_hit INTEGER NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        cached_input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        capabilities_json TEXT NOT NULL,
                        structured_domains_json TEXT NOT NULL,
                        error TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _request_hash(query: str, routing_date: date) -> str:
        value = f"{routing_date.isoformat()}\n{query.strip()}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _cached_route(self, request_hash: str) -> RequestRoute | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT route_json FROM request_routes
                WHERE request_hash = ? AND model = ? AND prompt_version = ?
                """,
                (request_hash, self.model, ROUTER_PROMPT_VERSION),
            ).fetchone()
        if row is None:
            return None
        return RequestRoute.model_validate_json(row["route_json"])

    def _cache_route(
        self,
        request_hash: str,
        routing_date: date,
        route: RequestRoute,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO request_routes (
                    request_hash, model, prompt_version, routing_date, route_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_hash,
                    self.model,
                    ROUTER_PROMPT_VERSION,
                    routing_date.isoformat(),
                    route.model_dump_json(),
                ),
            )

    def _log_event(
        self,
        request_hash: str,
        *,
        cached: bool,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        route: RequestRoute | None,
        error: str | None = None,
    ) -> None:
        capabilities = (
            [item.value for item in route.capabilities] if route else []
        )
        domains = (
            [item.value for item in route.structured_domains] if route else []
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_route_events (
                    request_hash, model, prompt_version, cache_hit, input_tokens,
                    cached_input_tokens, output_tokens, capabilities_json,
                    structured_domains_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_hash,
                    self.model,
                    ROUTER_PROMPT_VERSION,
                    int(cached),
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                    json.dumps(capabilities, separators=(",", ":")),
                    json.dumps(domains, separators=(",", ":")),
                    error,
                ),
            )

    def route(
        self,
        query: str,
        *,
        routing_date: date | None = None,
        use_cache: bool = True,
    ) -> RequestRouteResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Router query must not be empty")

        current_date = routing_date or date.today()
        request_hash = self._request_hash(normalized_query, current_date)
        if use_cache:
            cached_route = self._cached_route(request_hash)
            if cached_route is not None:
                self._log_event(
                    request_hash,
                    cached=True,
                    input_tokens=0,
                    cached_input_tokens=0,
                    output_tokens=0,
                    route=cached_route,
                )
                return RequestRouteResult(
                    route=cached_route,
                    model=self.model,
                    cached=True,
                    input_tokens=0,
                    cached_input_tokens=0,
                    output_tokens=0,
                )

        input_tokens = 0
        cached_input_tokens = 0
        output_tokens = 0
        try:
            client = self.client or OpenAI()
            response = client.responses.parse(
                model=self.model,
                reasoning={"effort": "none"},
                input=[
                    {"role": "system", "content": REQUEST_ROUTER_INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": (
                            f"Current date: {current_date.isoformat()}\n"
                            f"Request: {normalized_query}"
                        ),
                    },
                ],
                text_format=RequestRoute,
            )
            route = response.output_parsed
            if route is None:
                raise RuntimeError("Request router returned no structured route")
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            details = getattr(usage, "input_tokens_details", None)
            cached_input_tokens = (
                getattr(details, "cached_tokens", 0) or 0
                if details is not None
                else 0
            )
            cached_input_tokens = min(cached_input_tokens, input_tokens)
        except Exception as error:
            self._log_event(
                request_hash,
                cached=False,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                route=None,
                error=str(error),
            )
            raise RuntimeError(f"Request router failed: {error}") from error

        self._cache_route(request_hash, current_date, route)
        self._log_event(
            request_hash,
            cached=False,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            route=route,
        )
        return RequestRouteResult(
            route=route,
            model=self.model,
            cached=False,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )

    def stats(self) -> dict[str, object]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT cache_hit, input_tokens, cached_input_tokens,
                       output_tokens, capabilities_json,
                       structured_domains_json, error
                FROM request_route_events
                """
            ).fetchall()

        capability_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        for row in rows:
            for capability in json.loads(row["capabilities_json"]):
                capability_counts[capability] = (
                    capability_counts.get(capability, 0) + 1
                )
            for domain in json.loads(row["structured_domains_json"]):
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return {
            "executions": len(rows),
            "cache_hits": sum(bool(row["cache_hit"]) for row in rows),
            "failed": sum(bool(row["error"]) for row in rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "cached_input_tokens": sum(
                row["cached_input_tokens"] for row in rows
            ),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "capabilities": capability_counts,
            "structured_domains": domain_counts,
        }
