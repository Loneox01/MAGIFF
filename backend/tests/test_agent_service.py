import json
import unittest
from types import SimpleNamespace

from orchestration.router import (
    Capability,
    FreshnessRequirement,
    RequestIntent,
    RequestRoute,
    RequestRouteResult,
    StructuredDomain,
)
from services.agent import AgentService
from tools.base import ToolExecutionResult


def _route_result() -> RequestRouteResult:
    route = RequestRoute(
        request_summary="Look up one player value.",
        intent=RequestIntent.LOOKUP,
        freshness=FreshnessRequirement.HISTORICAL,
        capabilities=[Capability.STRUCTURED_DATA],
        structured_domains=[StructuredDomain.PLAYER_STATS],
        rationale="Structured statistics answer the request.",
    )
    return RequestRouteResult(
        route=route,
        model="test-router",
        cached=False,
        input_tokens=20,
        cached_input_tokens=5,
        output_tokens=4,
    )


def _usage(input_tokens: int, output_tokens: int, cached: int = 0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


class AgentServiceTests(unittest.TestCase):
    def test_returns_final_response_with_combined_usage(self) -> None:
        response = SimpleNamespace(
            output=[],
            output_text="The answer is 42.",
            usage=_usage(100, 12, 25),
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_: response)
        )
        router = SimpleNamespace(route=lambda _: _route_result())
        service = AgentService(
            client=client,
            router_factory=lambda: router,
            tool_schema_builder=lambda _: [],
            model="test-agent",
        )

        result = service.run("What is the answer?")

        self.assertEqual(result.answer, "The answer is 42.")
        self.assertEqual(result.usage.input_tokens, 120)
        self.assertEqual(result.usage.cached_input_tokens, 30)
        self.assertEqual(result.usage.output_tokens, 16)
        self.assertEqual(result.tool_rounds, 0)
        self.assertFalse(result.route.fallback_used)

    def test_executes_tool_and_preserves_report_telemetry(self) -> None:
        tool_call = SimpleNamespace(
            type="function_call",
            name="test_tool",
            arguments=json.dumps({"value": 7}),
            call_id="call-1",
        )
        responses = iter(
            [
                SimpleNamespace(
                    output=[tool_call],
                    output_text="",
                    usage=_usage(80, 6),
                ),
                SimpleNamespace(
                    output=[],
                    output_text="Seven.",
                    usage=_usage(90, 8, 30),
                ),
            ]
        )
        captured_inputs = []

        def create(**kwargs):
            captured_inputs.append(list(kwargs["input"]))
            return next(responses)

        details = {
            "component": "report_pipeline",
            "status": "ready",
            "evidence_sufficiency": "sufficient",
            "planner": {"model": "planner", "cached": False},
            "context_planner": {
                "model": "context",
                "cached": False,
                "context_needed": True,
                "branches": 1,
            },
            "identity": {"model": "identity", "triggered": False},
            "retrieval": {
                "strategy": "resolved",
                "candidates": 4,
                "branch_candidates": {"direct": 2, "context_1": 2},
            },
            "structured_enrichment": {
                "lookups": 1,
                "resolved": 1,
                "empty": 0,
                "errors": [],
                "results": [{"fallback_used": False}],
            },
            "reranker": {
                "model": "reranker",
                "api_called": True,
                "latency_ms": 10,
            },
        }
        tool = lambda value: ToolExecutionResult(  # noqa: E731
            output={"value": value},
            input_tokens=40,
            cached_input_tokens=10,
            output_tokens=5,
            details=details,
        )
        router = SimpleNamespace(route=lambda _: _route_result())
        service = AgentService(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            router_factory=lambda: router,
            tool_handlers={"test_tool": tool},
            tool_schema_builder=lambda _: [],
        )

        result = service.run("Use the tool.")

        self.assertEqual(result.answer, "Seven.")
        self.assertEqual(result.tool_rounds, 1)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertTrue(result.tool_calls[0].succeeded)
        self.assertEqual(
            result.tool_calls[0].report_pipeline["retrieval"]["candidates"],
            4,
        )
        self.assertEqual(result.usage.input_tokens, 230)
        self.assertEqual(result.usage.cached_input_tokens, 45)
        self.assertEqual(result.usage.output_tokens, 23)
        self.assertEqual(
            captured_inputs[1][-1]["type"], "function_call_output"
        )

    def test_router_failure_falls_back_to_all_local_capabilities(self) -> None:
        def fail_route(_):
            raise RuntimeError("router unavailable")

        response = SimpleNamespace(
            output=[],
            output_text="Fallback answer.",
            usage=_usage(10, 2),
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_: response)
        )
        router = SimpleNamespace(route=fail_route)
        captured_routes = []
        service = AgentService(
            client=client,
            router_factory=lambda: router,
            tool_schema_builder=lambda route: captured_routes.append(route) or [],
        )

        result = service.run("Keep working.")

        self.assertTrue(result.route.fallback_used)
        self.assertIn(Capability.REPORTS.value, result.route.capabilities)
        self.assertEqual(len(captured_routes[0].structured_domains), 6)


if __name__ == "__main__":
    unittest.main()
