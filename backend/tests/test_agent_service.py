import json
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

from orchestration.router import (
    Capability,
    FreshnessRequirement,
    RequestIntent,
    RequestRoute,
    RequestRouteResult,
    StructuredDomain,
)
from orchestration.player_references import (
    PlayerReferenceAdapter,
    PlayerReferenceResolver,
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


def _reports_route_result() -> RequestRouteResult:
    return RequestRouteResult(
        route=RequestRoute(
            request_summary="Find current role news.",
            intent=RequestIntent.NEWS,
            freshness=FreshnessRequirement.CURRENT,
            capabilities=[Capability.REPORTS],
            structured_domains=[],
            rationale="Current narrative evidence is required.",
        ),
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
    def test_routed_web_search_is_available_and_citations_are_clickable(
        self,
    ) -> None:
        text = "Minnesota named its Week 1 starter."
        citation = SimpleNamespace(
            type="url_citation",
            url="https://www.nfl.com/example",
            title="NFL announcement",
            start_index=0,
            end_index=len(text),
        )
        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="web_search_call"),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text=text,
                            annotations=[citation],
                        )
                    ],
                ),
            ],
            output_text=text,
            usage=_usage(100, 12),
        )
        captured = []
        route_result = replace(
            _reports_route_result(),
            route=_reports_route_result().route.model_copy(
                update={"capabilities": [Capability.WEB_SEARCH]},
            ),
        )
        service = AgentService(
            client=SimpleNamespace(
                responses=SimpleNamespace(
                    create=lambda **kwargs: captured.append(kwargs) or response
                )
            ),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: route_result
            ),
            tool_schema_builder=lambda _: [],
        )

        result = service.run("Search the web for the latest announcement.")

        self.assertEqual(captured[0]["tools"][0]["type"], "web_search")
        self.assertEqual(captured[0]["max_tool_calls"], 2)
        self.assertEqual(result.web_search_calls, 1)
        self.assertIn(
            "[NFL announcement](https://www.nfl.com/example)",
            result.answer,
        )
        self.assertEqual(result.web_sources[0].title, "NFL announcement")

    def test_weak_report_evidence_forces_one_web_fallback_round(self) -> None:
        report_call = SimpleNamespace(
            type="function_call",
            name="search_reports",
            arguments=json.dumps({"query": "Latest role update", "limit": 5}),
            call_id="reports-call",
        )
        text = "A current public report clarified the role."
        citation = SimpleNamespace(
            type="url_citation",
            url="https://www.espn.com/example",
            title="Current report",
            start_index=0,
            end_index=len(text),
        )
        responses = iter(
            [
                SimpleNamespace(
                    output=[report_call],
                    output_text="",
                    usage=_usage(20, 4),
                ),
                SimpleNamespace(
                    output=[
                        SimpleNamespace(type="web_search_call"),
                        SimpleNamespace(
                            type="message",
                            content=[
                                SimpleNamespace(
                                    type="output_text",
                                    text=text,
                                    annotations=[citation],
                                )
                            ],
                        ),
                    ],
                    output_text=text,
                    usage=_usage(30, 6),
                ),
            ]
        )
        captured = []

        def create(**kwargs):
            captured.append(kwargs)
            return next(responses)

        def search_reports(query, limit, source_question=None):
            return ToolExecutionResult(
                output={"status": "no_evidence", "reports": []},
                details={
                    "component": "report_pipeline",
                    "status": "no_evidence",
                    "evidence_sufficiency": "weak",
                },
            )

        service = AgentService(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: _reports_route_result()
            ),
            tool_handlers={"search_reports": search_reports},
            tool_schema_builder=lambda _: [
                {"type": "function", "name": "search_reports"}
            ],
        )

        result = service.run("What is the latest role update?")

        self.assertEqual(captured[0]["tools"][0]["type"], "function")
        self.assertNotIn("max_tool_calls", captured[0])
        self.assertEqual(captured[1]["tools"], [
            {"type": "web_search", "search_context_size": "low"}
        ])
        self.assertEqual(captured[1]["tool_choice"], "required")
        self.assertIn("Web fallback state: enabled", captured[1]["instructions"])
        self.assertEqual(result.web_search_calls, 1)
        self.assertEqual(result.tool_rounds, 1)

    def test_routed_web_need_retries_once_when_model_skips_search(self) -> None:
        responses = iter(
            [
                SimpleNamespace(
                    output=[],
                    output_text="Unsupported first answer.",
                    usage=_usage(10, 2),
                ),
                SimpleNamespace(
                    output=[SimpleNamespace(type="web_search_call")],
                    output_text="Web-grounded answer.",
                    usage=_usage(10, 2),
                ),
            ]
        )
        captured = []
        route_result = replace(
            _reports_route_result(),
            route=_reports_route_result().route.model_copy(
                update={"capabilities": [Capability.WEB_SEARCH]},
            ),
        )

        def create(**kwargs):
            captured.append(kwargs)
            return next(responses)

        service = AgentService(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: route_result
            ),
            tool_schema_builder=lambda _: [],
        )

        result = service.run("Search for a live update.")

        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[1]["tools"], [
            {"type": "web_search", "search_context_size": "low"}
        ])
        self.assertEqual(captured[1]["tool_choice"], "required")
        self.assertEqual(result.answer, "Web-grounded answer.")
        self.assertEqual(result.web_search_calls, 1)

    def test_estimates_router_and_agent_models_at_separate_rates(self) -> None:
        response = SimpleNamespace(
            output=[],
            output_text="Done.",
            usage=_usage(1_000, 100, 200),
        )
        route_result = replace(_route_result(), model="gpt-5.6-luna")
        service = AgentService(
            client=SimpleNamespace(
                responses=SimpleNamespace(create=lambda **_: response)
            ),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: route_result
            ),
            tool_schema_builder=lambda _: [],
            model="gpt-5.6-terra",
        )

        result = service.run("Test mixed pricing.")

        self.assertAlmostEqual(result.estimated_cost_usd, 0.0028479)

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

    def test_parallel_tool_calls_execute_concurrently_and_keep_call_order(
        self,
    ) -> None:
        calls = [
            SimpleNamespace(
                type="function_call",
                name="first_tool",
                arguments=json.dumps({"value": "first"}),
                call_id="call-first",
            ),
            SimpleNamespace(
                type="function_call",
                name="second_tool",
                arguments=json.dumps({"value": "second"}),
                call_id="call-second",
            ),
        ]
        responses = iter(
            [
                SimpleNamespace(
                    output=calls,
                    output_text="",
                    usage=_usage(10, 2),
                ),
                SimpleNamespace(
                    output=[],
                    output_text="Done.",
                    usage=_usage(10, 2),
                ),
            ]
        )
        create_arguments = []
        barrier = threading.Barrier(2)
        release_first = threading.Event()

        def create(**kwargs):
            create_arguments.append(kwargs)
            return next(responses)

        def first_tool(value):
            barrier.wait(timeout=1)
            release_first.wait(timeout=1)
            return {"value": value}

        def second_tool(value):
            barrier.wait(timeout=1)
            release_first.set()
            return {"value": value}

        service = AgentService(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: _route_result()
            ),
            tool_handlers={
                "first_tool": first_tool,
                "second_tool": second_tool,
            },
            tool_schema_builder=lambda _: [],
            max_parallel_tools=2,
        )

        result = service.run("Run both tools.")

        self.assertEqual(result.answer, "Done.")
        self.assertTrue(all(call.succeeded for call in result.tool_calls))
        self.assertTrue(create_arguments[0]["parallel_tool_calls"])
        outputs = [
            item
            for item in create_arguments[1]["input"]
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
        ]
        self.assertEqual(
            [item["call_id"] for item in outputs],
            ["call-first", "call-second"],
        )

    def test_transient_database_tool_failure_retries_without_model_round(self) -> None:
        tool_call = SimpleNamespace(
            type="function_call",
            name="read_tool",
            arguments=json.dumps({"value": "ok"}),
            call_id="read-call",
        )
        responses = iter(
            [
                SimpleNamespace(
                    output=[tool_call],
                    output_text="",
                    usage=_usage(10, 2),
                ),
                SimpleNamespace(
                    output=[],
                    output_text="Done.",
                    usage=_usage(10, 2),
                ),
            ]
        )
        attempts = 0

        def read_tool(value):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("[Errno 35] Resource temporarily unavailable")
            return {"value": value}

        service = AgentService(
            client=SimpleNamespace(
                responses=SimpleNamespace(create=lambda **_: next(responses))
            ),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: _route_result()
            ),
            tool_handlers={"read_tool": read_tool},
            tool_schema_builder=lambda _: [],
        )

        result = service.run("Read once.")

        self.assertEqual(attempts, 2)
        self.assertEqual(result.tool_rounds, 1)
        self.assertTrue(result.tool_calls[0].succeeded)

    def test_player_reference_resolves_once_per_request_for_multiple_tools(
        self,
    ) -> None:
        player_id = "11111111-1111-5111-8111-111111111111"
        find_calls = []

        def find_players(name):
            find_calls.append(name)
            return [
                {
                    "player_id": player_id,
                    "display_name": "A.J. Brown",
                    "position": "WR",
                    "latest_team": "PHI",
                    "status": "ACT",
                }
            ]

        tool_calls = [
            SimpleNamespace(
                type="function_call",
                name="get_player_season_stats",
                arguments=json.dumps(
                    {
                        "player_ref": "A.J. Brown",
                        "season": 2025,
                        "season_type": "REG",
                        "fields": ["fantasy_points_ppr"],
                    }
                ),
                call_id="stats",
            ),
            SimpleNamespace(
                type="function_call",
                name="get_player_ecr",
                arguments=json.dumps(
                    {
                        "player_ref": "A.J. Brown",
                        "season": 2026,
                        "scoring_format": "ppr",
                        "league_format": "redraft_1qb",
                        "snapshot_type": "current",
                        "as_of_date": None,
                    }
                ),
                call_id="ecr",
            ),
        ]
        responses = iter(
            [
                SimpleNamespace(
                    output=tool_calls,
                    output_text="",
                    usage=_usage(10, 2),
                ),
                SimpleNamespace(
                    output=[], output_text="Done.", usage=_usage(10, 2)
                ),
            ]
        )
        received_ids = []

        def handler(player_id, **_kwargs):
            received_ids.append(player_id)
            return {"player_id": player_id}

        service = AgentService(
            client=SimpleNamespace(
                responses=SimpleNamespace(create=lambda **_: next(responses))
            ),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: _route_result()
            ),
            tool_handlers={
                "get_player_season_stats": handler,
                "get_player_ecr": handler,
            },
            tool_schema_builder=lambda _: [],
            player_reference_adapter=PlayerReferenceAdapter(
                PlayerReferenceResolver(find_players=find_players)
            ),
        )

        result = service.run("Compare AJ Brown's stats and ECR.")

        self.assertEqual(find_calls, ["A.J. Brown"])
        self.assertEqual(received_ids, [player_id, player_id])
        self.assertTrue(all(call.succeeded for call in result.tool_calls))

    def test_ambiguous_player_reference_returns_candidates_without_calling_tool(
        self,
    ) -> None:
        candidates = [
            {
                "player_id": "11111111-1111-5111-8111-111111111111",
                "display_name": "Josh Allen",
                "position": "QB",
                "latest_team": "BUF",
                "status": "ACT",
            },
            {
                "player_id": "22222222-2222-5222-8222-222222222222",
                "display_name": "Josh Allen",
                "position": "DE",
                "latest_team": "JAX",
                "status": "ACT",
            },
        ]
        call = SimpleNamespace(
            type="function_call",
            name="get_player_season_stats",
            arguments=json.dumps(
                {
                    "player_ref": "Josh Allen",
                    "season": 2025,
                    "season_type": "REG",
                    "fields": None,
                }
            ),
            call_id="ambiguous",
        )
        captured = []
        responses = iter(
            [
                SimpleNamespace(
                    output=[call], output_text="", usage=_usage(10, 2)
                ),
                SimpleNamespace(
                    output=[], output_text="Clarify.", usage=_usage(10, 2)
                ),
            ]
        )

        def create(**kwargs):
            captured.append(kwargs)
            return next(responses)

        handler_called = False

        def handler(**_kwargs):
            nonlocal handler_called
            handler_called = True
            return {}

        service = AgentService(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            router_factory=lambda: SimpleNamespace(
                route=lambda _: _route_result()
            ),
            tool_handlers={"get_player_season_stats": handler},
            tool_schema_builder=lambda _: [],
            player_reference_adapter=PlayerReferenceAdapter(
                PlayerReferenceResolver(find_players=lambda _: candidates)
            ),
        )

        result = service.run("Get Josh Allen's stats.")

        self.assertFalse(handler_called)
        self.assertFalse(result.tool_calls[0].succeeded)
        output = json.loads(captured[1]["input"][-1]["output"])
        self.assertEqual(output["error"]["code"], "ambiguous_player_ref")
        self.assertEqual(len(output["error"]["candidates"]), 2)

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
