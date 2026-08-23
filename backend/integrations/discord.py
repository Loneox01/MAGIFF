"""Discord interaction verification and deferred response delivery."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from services.agent import AgentService


LOGGER = logging.getLogger(__name__)
DISCORD_API_BASE = "https://discord.com/api/v10"
DISCORD_MESSAGE_LIMIT = 2_000
MAX_DISCORD_MESSAGES = 5

PING_INTERACTION = 1
APPLICATION_COMMAND_INTERACTION = 2
PONG_RESPONSE = 1
CHANNEL_MESSAGE_RESPONSE = 4
DEFERRED_CHANNEL_MESSAGE_RESPONSE = 5
EPHEMERAL_MESSAGE_FLAG = 1 << 6


class DiscordRequestVerifier:
    """Verify Discord's Ed25519 signature over timestamp plus raw body."""

    def __init__(self, public_key_hex: str) -> None:
        try:
            public_key = bytes.fromhex(public_key_hex.strip())
            self._key = Ed25519PublicKey.from_public_bytes(public_key)
        except (TypeError, ValueError) as error:
            raise ValueError("DISCORD_PUBLIC_KEY must be a valid Ed25519 hex key") from error

    def verify(
        self,
        *,
        signature_hex: str | None,
        timestamp: str | None,
        body: bytes,
    ) -> bool:
        if not signature_hex or not timestamp:
            return False
        try:
            signature = bytes.fromhex(signature_hex)
            self._key.verify(signature, timestamp.encode("utf-8") + body)
        except (InvalidSignature, ValueError):
            return False
        return True


class RecentInteractionIds:
    """Process-local protection against Discord retrying the same command."""

    def __init__(self, ttl_seconds: int = 15 * 60) -> None:
        self.ttl_seconds = ttl_seconds
        self._claimed: dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, interaction_id: str) -> bool:
        now = time.monotonic()
        with self._lock:
            expired = [
                value
                for value, claimed_at in self._claimed.items()
                if now - claimed_at >= self.ttl_seconds
            ]
            for value in expired:
                self._claimed.pop(value, None)
            if interaction_id in self._claimed:
                return False
            self._claimed[interaction_id] = now
            return True


def split_discord_message(
    content: str,
    *,
    limit: int = DISCORD_MESSAGE_LIMIT,
    max_messages: int = MAX_DISCORD_MESSAGES,
) -> list[str]:
    """Split a model response at readable boundaries within Discord limits."""
    normalized = content.strip() or "MAGIFF returned an empty response."
    pieces: list[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= limit:
            pieces.append(remaining)
            break
        window = remaining[: limit + 1]
        candidates = [
            window.rfind("\n\n", 0, limit + 1),
            window.rfind("\n", 0, limit + 1),
            window.rfind(" ", 0, limit + 1),
        ]
        split_at = max(candidates)
        if split_at < limit // 2:
            split_at = limit
        piece = remaining[:split_at].rstrip()
        pieces.append(piece)
        remaining = remaining[split_at:].lstrip()

    if len(pieces) > max_messages:
        marker = "\n\n… response truncated"
        pieces = pieces[:max_messages]
        pieces[-1] = pieces[-1][: limit - len(marker)].rstrip() + marker
    return pieces


def format_discord_question(prompt: str) -> str:
    """Render user text as a Discord blockquote without enabling mentions."""
    lines = prompt.strip().splitlines() or [prompt.strip()]
    quoted = "\n".join(f"> {line}" if line else ">" for line in lines)
    return f"**Question**\n{quoted}"


def format_discord_thinking(prompt: str) -> str:
    return f"{format_discord_question(prompt)}\n\n*MAGIFF is thinking…*"


def format_discord_answer(prompt: str, answer: str) -> str:
    normalized_answer = answer.strip() or "MAGIFF returned an empty response."
    return f"{format_discord_question(prompt)}\n\n**MAGIFF**\n{normalized_answer}"


def extract_ask_prompt(payload: dict[str, Any]) -> str | None:
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("name") != "ask":
        return None
    options = data.get("options")
    if not isinstance(options, list):
        return None
    for option in options:
        if (
            isinstance(option, dict)
            and option.get("name") == "question"
            and isinstance(option.get("value"), str)
        ):
            value = option["value"].strip()
            return value or None
    return None


class DiscordWebhookClient:
    """Edit and extend a deferred interaction through Discord webhooks."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 20,
    ) -> None:
        self.client = client or httpx.Client(timeout=timeout_seconds)

    @staticmethod
    def _message(content: str) -> dict[str, object]:
        return {
            "content": content,
            "allowed_mentions": {"parse": []},
        }

    def edit_original(
        self,
        *,
        application_id: str,
        interaction_token: str,
        content: str,
    ) -> None:
        response = self.client.patch(
            (
                f"{DISCORD_API_BASE}/webhooks/{application_id}/"
                f"{interaction_token}/messages/@original"
            ),
            json=self._message(content),
        )
        response.raise_for_status()

    def create_followup(
        self,
        *,
        application_id: str,
        interaction_token: str,
        content: str,
    ) -> None:
        response = self.client.post(
            f"{DISCORD_API_BASE}/webhooks/{application_id}/{interaction_token}",
            json=self._message(content),
        )
        response.raise_for_status()


@dataclass(frozen=True)
class DiscordCompletion:
    interaction_id: str
    interaction_token: str
    prompt: str
    request_id: str


class DiscordInteractionRunner:
    """Run MAGIFF after deferral and publish the eventual Discord response."""

    def __init__(
        self,
        *,
        application_id: str,
        agent_service: AgentService,
        webhook_client: DiscordWebhookClient | None = None,
    ) -> None:
        self.application_id = application_id
        self.agent_service = agent_service
        self.webhook_client = webhook_client or DiscordWebhookClient()

    def complete(self, completion: DiscordCompletion) -> None:
        try:
            result = self.agent_service.run(completion.prompt)
            messages = split_discord_message(
                format_discord_answer(completion.prompt, result.answer)
            )
            self.webhook_client.edit_original(
                application_id=self.application_id,
                interaction_token=completion.interaction_token,
                content=messages[0],
            )
            for message in messages[1:]:
                self.webhook_client.create_followup(
                    application_id=self.application_id,
                    interaction_token=completion.interaction_token,
                    content=message,
                )
            LOGGER.info(
                json.dumps(
                    {
                        "event": "discord_agent_request_complete",
                        "request_id": completion.request_id,
                        "interaction_id": completion.interaction_id,
                        "model": result.model,
                        "latency_seconds": round(result.latency_seconds, 3),
                        "input_tokens": result.usage.input_tokens,
                        "cached_input_tokens": result.usage.cached_input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "tool_calls": len(result.tool_calls),
                        "messages": len(messages),
                    },
                    separators=(",", ":"),
                )
            )
        except Exception:
            LOGGER.exception(
                "Discord agent request failed request_id=%s interaction_id=%s",
                completion.request_id,
                completion.interaction_id,
            )
            try:
                self.webhook_client.edit_original(
                    application_id=self.application_id,
                    interaction_token=completion.interaction_token,
                    content=format_discord_answer(
                        completion.prompt,
                        (
                            "I couldn't complete that request. Please try again "
                            f"later. Request ID: `{completion.request_id}`"
                        ),
                    ),
                )
            except Exception:
                LOGGER.exception(
                    "Discord failure response could not be delivered request_id=%s",
                    completion.request_id,
                )
