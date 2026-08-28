"""Current OpenAI text-token price estimates used by local telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# Verified against the official OpenAI model pages on 2026-08-24. Sol's rate is
# promotional through at least 2026-11-21. Keep this small table centralized so
# a price change cannot silently create several conflicting estimates.
PRICING_VERIFIED_ON = "2026-08-24"


@dataclass(frozen=True)
class TextTokenRates:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


TEXT_TOKEN_RATES = {
    "gpt-5.6-luna": TextTokenRates(0.20, 0.02, 1.20),
    "gpt-5.6-terra": TextTokenRates(2.00, 0.20, 12.00),
    "gpt-5.6-sol": TextTokenRates(4.00, 0.40, 20.00),
}


def _pricing_key(model: str) -> str | None:
    normalized = model.strip().lower()
    if normalized == "gpt-5.6":
        return "gpt-5.6-sol"
    for name in TEXT_TOKEN_RATES:
        if normalized == name or normalized.startswith(f"{name}-"):
            return name
    return None


def estimate_text_token_cost_usd(
    *,
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate one model call from standard, cached-read, and output tokens.

    Cache-write tokens and hosted-tool fees are intentionally excluded because
    the current aggregate pipeline telemetry does not retain those values.
    """
    key = _pricing_key(model)
    if key is None:
        return None
    rates = TEXT_TOKEN_RATES[key]
    safe_input = max(0, int(input_tokens))
    safe_cached = min(safe_input, max(0, int(cached_input_tokens)))
    safe_output = max(0, int(output_tokens))
    uncached_input = safe_input - safe_cached
    return (
        uncached_input * rates.input_per_million
        + safe_cached * rates.cached_input_per_million
        + safe_output * rates.output_per_million
    ) / 1_000_000


def estimate_component_costs_usd(
    components: Iterable[dict[str, object]],
) -> float | None:
    """Sum model-specific telemetry, returning None if a used model is unknown."""
    total = 0.0
    for component in components:
        input_tokens = int(component.get("input_tokens", 0) or 0)
        cached_input_tokens = int(
            component.get("cached_input_tokens", 0) or 0
        )
        output_tokens = int(component.get("output_tokens", 0) or 0)
        if input_tokens == 0 and output_tokens == 0:
            continue
        model = str(component.get("model") or "")
        estimate = estimate_text_token_cost_usd(
            model=model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
        if estimate is None:
            return None
        total += estimate
    return total
