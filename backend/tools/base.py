"""Shared result envelope for tools with internal model usage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolExecutionResult:
    """Separate model-facing output from local observability metadata."""

    output: Any
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
