import json
import random
import unittest
from types import SimpleNamespace

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import ApiSettings
from integrations.discord import (
    DiscordBotClient,
    RecentInteractionIds,
    split_discord_message,
)
from integrations.discord_news import (
    DiscordNewsRunner,
    NewsCompletion,
    extract_news_query,
    format_news_result,
)
from integrations.discord_stats import (
    extract_stats_query,
    stats_autocomplete_choices,
)
from jobs.register_discord_commands import (
    GAME_COMMAND,
    NEWS_COMMAND,
    STATS_COMMAND,
    TEST_COMMANDS,
    UAI_COMMANDS,
    register_guild_command,
)
from services.news import (
    NewsDetail,
    NewsOutcome,
    NewsQuery,
    NewsReport,
    NewsResult,
    PlayerCandidate,
)
from services.stats import StatsOutcome, StatsResult
from services.roster_game import RosterGameService
from tests.test_roster_game_service import FakeRosterGameRepository


APPLICATION_ID = "123456789012345678"
GUILD_ID = "987654321098765432"
UAI_GUILD_ID = "876543210987654321"


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
        self.edits: list[dict[str, object]] = []
        self.followups: list[dict[str, object]] = []

    def edit_original(self, **values: object) -> None:
        self.edits.append(values)

    def create_followup(self, **values: object) -> None:
        self.followups.append(values)


class FakeNewsService:
    def __init__(self) -> None:
        self.queries: list[NewsQuery] = []

    def latest(self, query: NewsQuery) -> NewsResult:
        self.queries.append(query)
        return NewsResult(
            outcome=NewsOutcome.SUCCESS,
            query=query,
            resolved_player=(
                PlayerCandidate(
                    "gainwell-id", "Kenny Gainwell", "RB", "TB", "ACT"
                )
                if query.player
                else None
            ),
            reports=(
                NewsReport(
                    report_id="report-1",
                    title="Gainwell role update",
                    source="FantasyPros",
                    source_url="https://example.com/report-1",
                    author=None,
                    published_at="2026-08-23T12:00:00+00:00",
                    players=("Kenny Gainwell",),
                    teams=("TB",),
                    document_type="role_update",
                    storyline=None,
                    content_mode="provider_news",
                    body="Gainwell worked with the first-team offense.",
                ),
            ),
        )


class FailingNewsService:
    def latest(self, query: NewsQuery) -> NewsResult:
        raise RuntimeError("database unavailable")


class FakeStatsService:
    def __init__(self) -> None:
        self.queries = []

    def execute(self, query):
        self.queries.append(query)
        return StatsResult(
            outcome=StatsOutcome.SUCCESS,
            query=query,
            season=2025,
            rows=(
                {
                    "rank": 1,
                    "display_name": "A.J. Brown",
                    "position": "WR",
                    "team": "PHI",
                    "metric_value": 10.0,
                    "inputs": {"receiving_yards": 1500, "targets": 150},
                },
            ),
            formula="receiving_yards / targets",
        )


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
            discord_test_guild_id=GUILD_ID,
            discord_uai_guild_id=UAI_GUILD_ID,
            discord_uai_enabled=True,
        )
        self.agent = FakeAgentService()
        self.news = FakeNewsService()
        self.stats = FakeStatsService()
        self.game_repository = FakeRosterGameRepository()
        self.game = RosterGameService(
            self.game_repository,
            rng=random.Random(11),
        )
        self.webhook = FakeDiscordWebhookClient()
        self.client = TestClient(
            create_app(
                settings=settings,
                agent_service=self.agent,
                news_service=self.news,
                stats_service=self.stats,
                roster_game_service=self.game,
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

    def news_payload(self, *, interaction_id: str = "news-interaction") -> dict:
        return {
            "id": interaction_id,
            "application_id": APPLICATION_ID,
            "guild_id": GUILD_ID,
            "token": "news-interaction-token",
            "type": 2,
            "data": {
                "name": "news",
                "options": [
                    {"name": "player", "type": 3, "value": "Kenny Gainwell"},
                    {"name": "count", "type": 4, "value": 3},
                    {"name": "detail", "type": 3, "value": "summary"},
                ],
            },
        }

    def stats_payload(self, *, interaction_id: str = "stats-interaction") -> dict:
        return {
            "id": interaction_id,
            "application_id": APPLICATION_ID,
            "guild_id": GUILD_ID,
            "token": "stats-interaction-token",
            "type": 2,
            "data": {
                "name": "stats",
                "options": [
                    {
                        "name": "leaders",
                        "type": 1,
                        "options": [
                            {
                                "name": "formula",
                                "type": 3,
                                "value": "receiving_yards / targets",
                            },
                            {"name": "position", "type": 3, "value": "WR"},
                        ],
                    }
                ],
            },
        }

    def game_payload(self, *, interaction_id: str = "game-interaction") -> dict:
        return {
            "id": interaction_id,
            "application_id": APPLICATION_ID,
            "guild_id": GUILD_ID,
            "token": "game-interaction-token",
            "type": 2,
            "member": {
                "nick": "Roster Tester",
                "user": {
                    "id": "222222222222222222",
                    "username": "tester",
                },
            },
            "data": {
                "name": "game",
                "options": [
                    {
                        "name": "challenge",
                        "type": 1,
                        "options": [],
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

    def test_bot_client_sends_one_message_and_only_allows_explicit_user_ping(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"id": "discord-message-1"})

        bot = DiscordBotClient(
            "private-token",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        message_id = bot.send_channel_message(
            channel_id="123456789012345678",
            content="<@222222222222222222>\n## REVIEW FAILED\nTry again.",
        )

        self.assertEqual(message_id, "discord-message-1")
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0].headers["Authorization"],
            "Bot private-token",
        )
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["allowed_mentions"]["parse"], [])
        self.assertEqual(
            payload["allowed_mentions"]["users"],
            ["222222222222222222"],
        )
        self.assertEqual(payload["flags"], 4)

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
            "## Question\n> Who should I start?\n\n*MAGIFF is thinking…*",
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
                "## Question\n> Who should I start?\n\n"
                "## MAGIFF\nDiscord test answer."
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

    def test_uai_profile_allows_news_and_stats(self) -> None:
        news_payload = self.news_payload(interaction_id="uai-news")
        news_payload["guild_id"] = UAI_GUILD_ID
        stats_payload = self.stats_payload(interaction_id="uai-stats")
        stats_payload["guild_id"] = UAI_GUILD_ID

        news_response = self.signed_post(news_payload)
        stats_response = self.signed_post(stats_payload)

        self.assertEqual(news_response.json()["type"], 4)
        self.assertEqual(stats_response.json()["type"], 4)
        self.assertEqual(len(self.news.queries), 1)
        self.assertEqual(len(self.stats.queries), 1)

    def test_uai_profile_rejects_ask_even_if_command_remains_visible(self) -> None:
        payload = self.command_payload(interaction_id="uai-ask")
        payload["guild_id"] = UAI_GUILD_ID

        response = self.signed_post(payload)

        self.assertIn("only enabled", response.json()["data"]["content"])
        self.assertEqual(self.agent.prompts, [])

    def test_game_starts_hidden_with_team_art_and_buttons(self) -> None:
        response = self.signed_post(self.game_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], 4)
        data = response.json()["data"]
        self.assertIn("17-0 Challenge", data["content"])
        self.assertIn("Season Total", data["content"])
        self.assertIn(
            "Player and points hidden",
            data["embeds"][0]["description"],
        )
        self.assertNotIn("season PPR", data["embeds"][0]["description"])
        embed = data["embeds"][0]
        self.assertTrue(embed["thumbnail"]["url"])
        self.assertEqual(
            [field["name"] for field in embed["fields"]],
            ["Roster", "\u200b", "Rerolls"],
        )
        self.assertTrue(embed["fields"][0]["inline"])
        self.assertTrue(embed["fields"][1]["inline"])
        self.assertNotIn("footer", embed)
        buttons = data["components"][0]["components"]
        self.assertEqual(
            [button["label"] for button in buttons],
            [
                "Reroll Team",
                "Reroll Position",
                "Lock & Spin Next",
                "Forfeit Run",
            ],
        )

    def test_game_reveal_mode_and_component_advance(self) -> None:
        payload = self.game_payload(interaction_id="revealed-game")
        payload["data"]["options"][0]["options"] = [
            {"name": "season", "type": 4, "value": 2025},
            {"name": "reveal", "type": 5, "value": True},
        ]
        started = self.signed_post(payload).json()
        lock_button = started["data"]["components"][0]["components"][2]
        component_payload = {
            "id": "lock-component",
            "application_id": APPLICATION_ID,
            "guild_id": GUILD_ID,
            "token": "component-token",
            "type": 3,
            "member": payload["member"],
            "data": {
                "component_type": 2,
                "custom_id": lock_button["custom_id"],
            },
        }

        advanced = self.signed_post(component_payload)

        self.assertIn(
            "season PPR",
            started["data"]["embeds"][0]["description"],
        )
        self.assertEqual(advanced.json()["type"], 7)
        self.assertIn("17-0 Challenge", advanced.json()["data"]["content"])

    def test_game_supports_ppg_scoring_mode(self) -> None:
        payload = self.game_payload(interaction_id="ppg-game")
        payload["data"]["options"][0]["options"] = [
            {"name": "scoring", "type": 3, "value": "ppg"},
            {"name": "reveal", "type": 5, "value": True},
        ]

        response = self.signed_post(payload)

        data = response.json()["data"]
        self.assertIn("PPR PPG", data["content"])
        self.assertIn("PPR PPG", data["embeds"][0]["description"])
        state = next(iter(self.game_repository.games.values()))
        self.assertEqual(state.scoring_mode.value, "ppg")

    def test_game_forfeit_ends_run_and_removes_buttons(self) -> None:
        started = self.signed_post(self.game_payload()).json()
        forfeit_button = started["data"]["components"][0]["components"][3]
        payload = self.game_payload(interaction_id="forfeit-component")
        payload["type"] = 3
        payload["data"] = {
            "component_type": 2,
            "custom_id": forfeit_button["custom_id"],
        }

        response = self.signed_post(payload)

        self.assertEqual(response.json()["type"], 7)
        data = response.json()["data"]
        self.assertIn("Run Forfeited", data["embeds"][0]["title"])
        self.assertEqual(data["components"], [])

    def test_game_buttons_reject_a_different_user_ephemerally(self) -> None:
        started = self.signed_post(self.game_payload()).json()
        custom_id = started["data"]["components"][0]["components"][2]["custom_id"]
        payload = self.game_payload(interaction_id="other-user-click")
        payload["type"] = 3
        payload["member"]["user"]["id"] = "333333333333333333"
        payload["data"] = {"component_type": 2, "custom_id": custom_id}

        response = self.signed_post(payload)

        self.assertEqual(response.json()["type"], 4)
        self.assertEqual(response.json()["data"]["flags"], 64)
        self.assertIn("Only the player", response.json()["data"]["content"])

    def test_uai_profile_allows_game_and_component_actions(self) -> None:
        payload = self.game_payload(interaction_id="uai-game")
        payload["guild_id"] = UAI_GUILD_ID

        started = self.signed_post(payload).json()
        lock_button = started["data"]["components"][0]["components"][2]
        payload["id"] = "uai-game-lock"
        payload["type"] = 3
        payload["data"] = {
            "component_type": 2,
            "custom_id": lock_button["custom_id"],
        }
        advanced = self.signed_post(payload)

        self.assertEqual(started["type"], 4)
        self.assertEqual(advanced.json()["type"], 7)
        self.assertEqual(len(self.game_repository.games), 1)

    def test_disabled_uai_profile_rejects_commands(self) -> None:
        self.client.app.state.settings.discord_uai_enabled = False
        payload = self.news_payload(interaction_id="disabled-uai-news")
        payload["guild_id"] = UAI_GUILD_ID

        response = self.signed_post(payload)

        self.assertIn("temporarily disabled", response.json()["data"]["content"])
        self.assertEqual(self.news.queries, [])

    def test_duplicate_interaction_does_not_run_agent_twice(self) -> None:
        payload = self.command_payload(interaction_id="same-interaction")

        first = self.signed_post(payload)
        second = self.signed_post(payload)

        self.assertEqual(first.json()["type"], 4)
        self.assertEqual(second.json()["type"], 4)
        self.assertEqual(self.agent.prompts, ["Who should I start?"])

    def test_news_uses_slash_options_and_edits_original_response(self) -> None:
        response = self.signed_post(self.news_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], 4)
        self.assertEqual(response.json()["data"]["flags"], 4)
        self.assertIn("latest 3", response.json()["data"]["content"])
        self.assertEqual(len(self.news.queries), 1)
        self.assertEqual(self.news.queries[0].player, "Kenny Gainwell")
        self.assertEqual(self.news.queries[0].count, 3)
        self.assertEqual(self.news.queries[0].detail, NewsDetail.SUMMARY)
        self.assertEqual(len(self.webhook.edits), 1)
        self.assertIn("Gainwell role update", self.webhook.edits[0]["content"])
        self.assertIn(
            "Gainwell worked with the first-team offense",
            self.webhook.edits[0]["content"],
        )
        self.assertTrue(self.webhook.edits[0]["suppress_embeds"])

    def test_news_can_enable_link_previews(self) -> None:
        payload = self.news_payload(interaction_id="preview-news")
        payload["data"]["options"].append(
            {"name": "previews", "type": 5, "value": True}
        )

        response = self.signed_post(payload)

        self.assertNotIn("flags", response.json()["data"])
        self.assertTrue(self.news.queries[0].previews)
        self.assertFalse(self.webhook.edits[0]["suppress_embeds"])

    def test_news_rejects_invalid_options_without_querying_service(self) -> None:
        payload = self.news_payload()
        payload["data"]["options"] = [
            {"name": "detail", "type": 3, "value": "everything"}
        ]

        response = self.signed_post(payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("headlines, summary, or full", response.json()["data"]["content"])
        self.assertEqual(self.news.queries, [])

    def test_news_parser_defaults_are_bounded(self) -> None:
        payload = self.news_payload()
        payload["data"]["options"] = []

        query = extract_news_query(payload)

        self.assertEqual(query, NewsQuery())

    def test_news_ambiguity_format_explains_retry(self) -> None:
        query = NewsQuery(player="Justin Jefferson")
        message = format_news_result(
            NewsResult(
                outcome=NewsOutcome.PLAYER_AMBIGUOUS,
                query=query,
                player_candidates=(
                    PlayerCandidate("one", "Justin Jefferson", "WR", "MIN", "ACT"),
                    PlayerCandidate("two", "Justin Jefferson", "DB", "ATL", "ACT"),
                ),
            )
        )

        self.assertIn("matched multiple", message)
        self.assertIn("team:MIN", message)

    def test_news_failure_punts_with_retry_and_request_id(self) -> None:
        runner = DiscordNewsRunner(
            application_id=APPLICATION_ID,
            news_service=FailingNewsService(),
            webhook_client=self.webhook,
        )

        with self.assertLogs("integrations.discord_news", level="ERROR"):
            runner.complete(
                NewsCompletion(
                    interaction_id="failed-news",
                    interaction_token="failed-token",
                    query=NewsQuery(),
                    request_id="request-123",
                )
            )

        self.assertEqual(len(self.webhook.edits), 1)
        self.assertIn("Retry in a moment", self.webhook.edits[0]["content"])
        self.assertIn("request-123", self.webhook.edits[0]["content"])

    def test_stats_runs_deterministically_and_edits_response(self) -> None:
        response = self.signed_post(self.stats_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], 4)
        self.assertIn("player leaders", response.json()["data"]["content"])
        self.assertEqual(len(self.stats.queries), 1)
        self.assertEqual(self.stats.queries[0].position, "WR")
        self.assertIn("A.J. Brown", self.webhook.edits[0]["content"])
        self.assertIn("10", self.webhook.edits[0]["content"])

    def test_stats_formula_autocomplete_composes_current_expression(self) -> None:
        payload = self.stats_payload(interaction_id="stats-autocomplete")
        payload["type"] = 4
        payload["data"]["options"][0]["options"] = [
            {
                "name": "formula",
                "type": 3,
                "value": "(receptions + receiv",
                "focused": True,
            }
        ]

        response = self.signed_post(payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], 8)
        values = [choice["value"] for choice in response.json()["data"]["choices"]]
        self.assertIn("(receptions + receiving_yards", values)
        self.assertEqual(self.stats.queries, [])

    def test_stats_parser_understands_team_subcommand(self) -> None:
        payload = self.stats_payload()
        payload["data"]["options"] = [
            {
                "name": "team",
                "type": 1,
                "options": [
                    {"name": "team", "type": 3, "value": "Eagles"},
                    {"name": "perspective", "type": 3, "value": "defense"},
                    {"name": "formula", "type": 3, "value": "points_allowed / games"},
                ],
            }
        ]

        query = extract_stats_query(payload)

        self.assertEqual(query.team, "Eagles")
        self.assertEqual(query.perspective, "defense")

    def test_stats_parser_normalizes_discord_whole_number_threshold(self) -> None:
        payload = self.stats_payload()
        payload["data"]["options"][0]["options"].extend(
            [
                {"name": "minimum_field", "type": 3, "value": "carries"},
                {"name": "minimum_value", "type": 10, "value": 100.0},
            ]
        )

        query = extract_stats_query(payload)

        self.assertEqual(query.minimum_value, 100)
        self.assertIsInstance(query.minimum_value, int)

    def test_stats_autocomplete_uses_defensive_catalog(self) -> None:
        payload = self.stats_payload()
        payload["type"] = 4
        payload["data"]["options"] = [
            {
                "name": "team-leaders",
                "type": 1,
                "options": [
                    {"name": "perspective", "type": 3, "value": "defense"},
                    {"name": "formula", "type": 3, "value": "passing_yards", "focused": True},
                ],
            }
        ]

        choices = stats_autocomplete_choices(payload)

        self.assertTrue(any(choice["value"] == "passing_yards_allowed" for choice in choices))

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
            discord_test_guild_id=None,
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

    def test_registers_news_command_with_structured_options(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"id": "news-id", "name": "news", "version": "1"},
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = register_guild_command(
                application_id=APPLICATION_ID,
                guild_id=GUILD_ID,
                bot_token="private-bot-token",
                command=NEWS_COMMAND,
                client=client,
            )

        self.assertEqual(result["name"], "news")
        option_names = {
            option["name"] for option in captured["payload"]["options"]
        }
        self.assertEqual(
            option_names, {"player", "team", "count", "detail", "previews"}
        )

    def test_stats_command_uses_subcommands_and_native_autocomplete(self) -> None:
        subcommands = {option["name"]: option for option in STATS_COMMAND["options"]}

        self.assertEqual(
            set(subcommands),
            {"player", "leaders", "team", "team-leaders", "fields"},
        )
        leader_options = {
            option["name"]: option for option in subcommands["leaders"]["options"]
        }
        self.assertTrue(leader_options["formula"]["autocomplete"])
        self.assertTrue(leader_options["minimum_field"]["autocomplete"])
        self.assertNotIn("choices", leader_options["formula"])

    def test_game_command_has_scoring_and_reveal_options(self) -> None:
        self.assertEqual(GAME_COMMAND["options"][0]["name"], "challenge")
        options = {
            option["name"]: option
            for option in GAME_COMMAND["options"][0]["options"]
        }
        self.assertEqual(set(options), {"season", "scoring", "reveal"})
        self.assertFalse(options["season"]["required"])
        self.assertFalse(options["scoring"]["required"])
        self.assertFalse(options["reveal"]["required"])
        self.assertEqual(
            {choice["value"] for choice in options["scoring"]["choices"]},
            {"season_total", "ppg"},
        )

    def test_test_and_uai_command_buckets_are_intentionally_different(self) -> None:
        self.assertEqual(
            {command["name"] for command in TEST_COMMANDS},
            {"ask", "news", "stats", "game"},
        )
        self.assertEqual(
            {command["name"] for command in UAI_COMMANDS},
            {"news", "stats", "game"},
        )


if __name__ == "__main__":
    unittest.main()
