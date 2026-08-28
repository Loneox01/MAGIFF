"""Deterministic authorization for hard report publication-date filters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .planner import QueryPlan, TemporalBasis


@dataclass(frozen=True)
class ReportTemporalPolicy:
    """Separate model-suggested time preferences from exclusionary filters."""

    hard_start_date: str | None = None
    hard_end_date: str | None = None
    hard_filter_applied: bool = False
    reason: str = "No explicit closed publication-date range was authorized."


def _date_is_mentioned(question: str, boundary: date) -> bool:
    """Recognize common literal renderings of one exact calendar date."""
    month = boundary.strftime("%B")
    month_short = boundary.strftime("%b")
    day = boundary.day
    year = boundary.year
    suffix = "th" if 10 < day % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(
        day % 10,
        "th",
    )
    patterns = (
        rf"\b{year}-{boundary.month:02d}-{day:02d}\b",
        rf"\b0?{boundary.month}/0?{day}/(?:{year}|{year % 100:02d})\b",
        rf"\b(?:{month}|{month_short})\s+{day}(?:{suffix})?"
        rf"(?:\s*,?\s*{year})?\b",
        rf"\b{day}(?:{suffix})?\s+(?:{month}|{month_short})"
        rf"(?:\s*,?\s*{year})?\b",
    )
    return any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in patterns)


def report_temporal_policy(
    plan: QueryPlan,
    source_question: str,
) -> ReportTemporalPolicy:
    """Authorize only a user-stated, closed, verifiable date range.

    Every other temporal signal remains in ``QueryPlan`` for semantic retrieval,
    recency adjustment, reranking, and timeline ordering, but cannot exclude a
    report from the candidate pool.
    """
    if plan.temporal_mode != "between":
        return ReportTemporalPolicy(
            reason="Temporal mode is not a closed between range."
        )
    if plan.temporal_basis != TemporalBasis.EXPLICIT_USER:
        return ReportTemporalPolicy(
            reason="The closed range was not labeled explicit_user."
        )
    if not plan.start_date or not plan.end_date:
        return ReportTemporalPolicy(
            reason="The explicit range is missing a start or end date."
        )
    try:
        start = date.fromisoformat(plan.start_date)
        end = date.fromisoformat(plan.end_date)
    except ValueError:
        return ReportTemporalPolicy(reason="The explicit range contains invalid dates.")
    if start > end:
        return ReportTemporalPolicy(reason="The explicit range starts after it ends.")
    if not _date_is_mentioned(source_question, start):
        return ReportTemporalPolicy(
            reason="The planned start date is absent from the original question."
        )
    if start != end and not _date_is_mentioned(source_question, end):
        return ReportTemporalPolicy(
            reason="The planned end date is absent from the original question."
        )
    return ReportTemporalPolicy(
        hard_start_date=start.isoformat(),
        hard_end_date=end.isoformat(),
        hard_filter_applied=True,
        reason="Verified an explicit closed range against the original question.",
    )
