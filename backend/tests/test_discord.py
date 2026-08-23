import json
import unittest
from types import SimpleNamespace

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import ApiSettings
from integrations.discord import RecentInteractionIds, split_discord_message
from jobs.register_discord_commands import register_guild_command


APPLICATION_ID = "123456789012345678"
GUILD_ID = "987654321098765432"


class FakeAgentService:
    def __init__(self, answer: str = "Discord test answer.") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def run(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(
            answer=self.answer,
            model="test-agent",
            latency_seconds=0.25,
            usage=SimpleNamespace(
                input_tokens=100,
                cached_input_tokens=20,
                output_tokens=30,
            ),
            tool_calls=(),
        )


class FakeDiscordWebhookClient:
    def __init__(self) -> None:
        self.edits: list[dict[str, str]] = []
        self.followups: list[dict[str, str]] = []

    def edit_original(self, **values: str) -> None:
        self.edits.append(values)

    def create_followup(self, **values: str) -> None:
        self.followups.append(values)


class DiscordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        settings = ApiSettings(
            _env_file=None,
            environment="test",
            api_key="test-api-key",
            openai_api_key="test-openai-key",
            supabase_url="https://example.supabase.co",
            supabase_secret_key="test-supabase-key",
            discord_application_id=APPLICATION_ID,
            discord_public_key=public_key.hex(),
            discord_guild_id=GUILD_ID,
        )
        self.agent = FakeAgentService()
        self.webhook = FakeDiscordWebhookClient()
        self.client = TestClient(
            create_app(
                settings=settings,
                agent_service=self.agent,
                discord_webhook_client=self.webhook,
                discord_interaction_ids=RecentInteractionIds(),
            )
        )

    def signed_post(self, payload: dict, *, valid: bool = True):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        timestamp = "1787461200"
        signature = self.private_key.sign(timestamp.encode("utf-8") + body)
        if not valid:
            signature = bytes([signature[0] ^ 1]) + signature[1:]
        return self.client.post(
            "/v1/discord/interactions",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature-Ed25519": signature.hex(),
                "X-Signature-Timestamp": timestamp,
            },
        )

    def command_payload(self, *, interaction_id: str = "interaction-1") -> dict:
        return {
            "id": interaction_id,
            "application_id": APPLICATION_ID,
            "guild_id": GUILD_ID,
            "token": "interaction-token",
            "type": 2,
            "data": {
                "name": "ask",
                "options": [
                    {
                        "name": "question",
                        "type": 3,
                        "value": "Who should I start?",
                    }
                ],
            },
        }

    def test_rejects_missing_or_invalid_signature(self) -> None:
        unsigned = self.client.post(
            "/v1/discord/interactions",
            json=self.command_payload(),
        )
        invalid = self.signed_post(self.command_payload(), valid=False)

        self.assertEqual(unsigned.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(self.agent.prompts, [])

    def test_responds_to_discord_ping(self) -> None:
        response = self.signed_post(
            {"application_id": APPLICATION_ID, "type": 1}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"type": 1})

    def test_shows_question_then_edits_original_response(self) -> None:
        response = self.signed_post(self.command_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], 4)
        self.assertEqual(
            response.json()["data"]["content"],
            "**Question**\n> Who should I start?\n\n*MAGIFF is thinking…*",
        )
        self.assertEqual(
            response.json()["data"]["allowed_mentions"],
            {"parse": []},
        )
        self.assertEqual(self.agent.prompts, ["Who should I start?"])
        self.assertEqual(len(self.webhook.edits), 1)
        self.assertEqual(
            self.webhook.edits[0]["content"],
            (
                "**Question**\n> Who should I start?\n\n"
                "**MAGIFF**\nDiscord test answer."
            ),
        )
        self.assertEqual(self.webhook.followups, [])

    def test_wrong_guild_is_rejected_without_running_agent(self) -> None:
        payload = self.command_payload()
        payload["guild_id"] = "111111111111111111"

        response = self.signed_post(payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], 4)
        self.assertEqual(response.json()["data"]["flags"], 64)
        self.assertEqual(self.agent.prompts, [])

    def test_duplicate_interaction_does_not_run_agent_twice(self) -> None:
        payload = self.command_payload(interaction_id="same-interaction")

        first = self.signed_post(payload)
        second = self.signed_post(payload)

        self.assertEqual(first.json()["type"], 4)
        self.assertEqual(second.json()["type"], 4)
        self.assertEqual(self.agent.prompts, ["Who should I start?"])

    def test_long_answers_are_split_with_a_bounded_message_count(self) -> None:
        messages = split_discord_message("word " * 5_000)

        self.assertEqual(len(messages), 5)
        self.assertTrue(all(len(message) <= 2_000 for message in messages))
        self.assertTrue(messages[-1].endswith("… response truncated"))

    def test_partial_discord_configuration_is_rejected(self) -> None:
        settings = ApiSettings(
            _env_file=None,
            environment="test",
            discord_application_id=APPLICATION_ID,
            discord_public_key=None,
            discord_guild_id=None,
        )

        with self.assertRaisesRegex(RuntimeError, "Discord requires"):
            create_app(settings=settings, agent_service=self.agent)


class DiscordCommandRegistrationTests(unittest.TestCase):
    def test_registers_private_guild_command(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("Authorization")
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"id": "command-id", "name": "ask", "version": "1"},
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = register_guild_command(
                application_id=APPLICATION_ID,
                guild_id=GUILD_ID,
                bot_token="private-bot-token",
                client=client,
            )

        self.assertEqual(result["name"], "ask")
        self.assertEqual(captured["authorization"], "Bot private-bot-token")
        self.assertEqual(captured["payload"]["name"], "ask")
        self.assertEqual(captured["payload"]["options"][0]["name"], "question")
        self.assertIn(f"/guilds/{GUILD_ID}/commands", captured["url"])


if __name__ == "__main__":
    unittest.main()
