import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.app import create_app
from api.config import ApiSettings
from services.agent import (
    AgentRunResult,
    RouteTelemetry,
    TokenUsage,
    ToolCallTelemetry,
)


def _settings(**overrides) -> ApiSettings:
    values = {
        "environment": "test",
        "api_key": "test-api-key",
        "openai_api_key": "test-openai-key",
        "supabase_url": "https://example.supabase.co",
        "supabase_secret_key": "test-supabase-key",
        "cors_origins_raw": "http://localhost:5173",
    }
    values.update(overrides)
    return ApiSettings(**values)


def _agent_result() -> AgentRunResult:
    usage = TokenUsage(
        input_tokens=120,
        cached_input_tokens=20,
        output_tokens=30,
    )
    route = RouteTelemetry(
        model="test-router",
        cached=False,
        fallback_used=False,
        request_summary="Answer a test question.",
        intent="lookup",
        freshness="historical",
        capabilities=("structured_data",),
        structured_domains=("player_stats",),
        rationale="Use structured data.",
        error=None,
        usage=TokenUsage(20, 5, 4),
    )
    tool = ToolCallTelemetry(
        name="get_player_season_stats",
        arguments={"season": 2025},
        succeeded=True,
        error=None,
        report_pipeline=None,
    )
    return AgentRunResult(
        answer="Test answer.",
        model="test-agent",
        latency_seconds=0.2,
        tool_rounds=1,
        usage=usage,
        route=route,
        tool_calls=(tool,),
    )


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SimpleNamespace(run=lambda _prompt: _agent_result())
        self.client = TestClient(
            create_app(settings=_settings(), agent_service=self.service)
        )

    def test_health_and_readiness_do_not_require_authentication(self) -> None:
        health = self.client.get("/health")
        ready = self.client.get("/ready")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")

    def test_agent_endpoint_requires_bearer_token(self) -> None:
        response = self.client.post(
            "/v1/agent/query",
            json={"prompt": "Who led the league?"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_agent_endpoint_returns_structured_result_and_request_id(self) -> None:
        response = self.client.post(
            "/v1/agent/query",
            headers={"Authorization": "Bearer test-api-key"},
            json={"prompt": "  Who led the league?  "},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "Test answer.")
        self.assertEqual(body["usage"]["input_tokens"], 120)
        self.assertEqual(body["tool_calls"][0]["name"], "get_player_season_stats")
        self.assertEqual(body["request_id"], response.headers["x-request-id"])

    def test_blank_prompt_is_rejected_before_agent_execution(self) -> None:
        response = self.client.post(
            "/v1/agent/query",
            headers={"Authorization": "Bearer test-api-key"},
            json={"prompt": "   "},
        )

        self.assertEqual(response.status_code, 422)

    def test_production_requires_all_runtime_secrets(self) -> None:
        settings = _settings(
            environment="production",
            openai_api_key=None,
        )
        with self.assertRaisesRegex(RuntimeError, "openai"):
            create_app(settings=settings, agent_service=self.service)


if __name__ == "__main__":
    unittest.main()
