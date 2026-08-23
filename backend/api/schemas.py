"""Validated request and response contracts for the agent API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentQueryRequest(ApiModel):
    prompt: str = Field(min_length=1, max_length=4_000)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt must not be blank")
        return normalized


class TokenUsageResponse(ApiModel):
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class RouteTelemetryResponse(ApiModel):
    model: str | None
    cached: bool
    fallback_used: bool
    request_summary: str
    intent: str
    freshness: str
    capabilities: list[str]
    structured_domains: list[str]
    rationale: str
    error: str | None
    usage: TokenUsageResponse


class ToolCallTelemetryResponse(ApiModel):
    name: str
    arguments: dict[str, Any]
    succeeded: bool
    error: str | None
    report_pipeline: dict[str, Any] | None


class AgentQueryResponse(ApiModel):
    request_id: str
    answer: str
    model: str
    latency_seconds: float = Field(ge=0)
    tool_rounds: int = Field(ge=0)
    usage: TokenUsageResponse
    route: RouteTelemetryResponse
    tool_calls: list[ToolCallTelemetryResponse]


class HealthResponse(ApiModel):
    status: str
    service: str


class ReadinessResponse(ApiModel):
    status: str
    dependencies: dict[str, bool]
