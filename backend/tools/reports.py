"""Model-facing adapter for the report retrieval pipeline."""

from functools import lru_cache

from model_costs import estimate_component_costs_usd
from rag.pipeline import ReportRetrievalPipeline

from .base import ToolExecutionResult


SEARCH_REPORTS_TOOL = {
    "type": "function",
    "name": "search_reports",
    "description": (
        "Search the maintained fantasy-football report corpus for injuries, "
        "news, transactions, roles, practice observations, timelines, or other "
        "narrative evidence. The tool performs query planning, metadata "
        "grounding, hybrid retrieval, and reranking internally. Do not use it "
        "for statistics or rankings already covered by structured NFL tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The original user question when it is entirely about "
                    "reports; otherwise a self-contained narrative subquestion "
                    "preserving literal entities, comparisons, and time scope."
                ),
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Maximum number of report records to return.",
            },
        },
        "required": ["query", "limit"],
        "additionalProperties": False,
    },
    "strict": True,
}


@lru_cache(maxsize=1)
def _pipeline() -> ReportRetrievalPipeline:
    return ReportRetrievalPipeline()


def search_reports(
    query: str,
    limit: int = 5,
    *,
    source_question: str | None = None,
) -> ToolExecutionResult:
    result = _pipeline().search(
        query,
        source_question=source_question,
        limit=limit,
    )
    planner = result.telemetry["planner"]
    context_planner = result.telemetry["context_planner"]
    identity = result.telemetry["identity"]
    reranker = result.telemetry["reranker"]
    model_components = (planner, context_planner, identity, reranker)
    return ToolExecutionResult(
        output=result.agent_output(),
        input_tokens=(
            planner["input_tokens"]
            + context_planner["input_tokens"]
            + identity["input_tokens"]
            + reranker["input_tokens"]
        ),
        cached_input_tokens=(
            planner["cached_input_tokens"]
            + context_planner["cached_input_tokens"]
            + identity["cached_input_tokens"]
            + reranker["cached_input_tokens"]
        ),
        output_tokens=(
            planner["output_tokens"]
            + context_planner["output_tokens"]
            + identity["output_tokens"]
            + reranker["output_tokens"]
        ),
        estimated_cost_usd=estimate_component_costs_usd(model_components),
        details={
            "component": "report_pipeline",
            "status": result.status.value,
            "evidence_sufficiency": result.evidence_sufficiency,
            **result.telemetry,
        },
    )


REPORT_TOOL_HANDLERS = {"search_reports": search_reports}
