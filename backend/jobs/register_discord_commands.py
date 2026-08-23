"""Register MAGIFF's private guild-scoped Discord slash commands."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISCORD_API_BASE = "https://discord.com/api/v10"

ASK_COMMAND = {
    "name": "ask",
    "type": 1,
    "description": "Ask MAGIFF an NFL fantasy-football question",
    "options": [
        {
            "name": "question",
            "description": "The fantasy-football question for MAGIFF",
            "type": 3,
            "required": True,
            "min_length": 1,
            "max_length": 1_500,
        }
    ],
}

NEWS_COMMAND = {
    "name": "news",
    "type": 1,
    "description": "Get the latest reports stored by MAGIFF",
    "options": [
        {
            "name": "player",
            "description": "Optional full or partial player name",
            "type": 3,
            "required": False,
            "min_length": 1,
            "max_length": 100,
        },
        {
            "name": "team",
            "description": "Optional team abbreviation or official name",
            "type": 3,
            "required": False,
            "min_length": 2,
            "max_length": 40,
        },
        {
            "name": "count",
            "description": "Number of recent reports to return (default 5)",
            "type": 4,
            "required": False,
            "min_value": 1,
            "max_value": 10,
        },
        {
            "name": "detail",
            "description": "How much stored report text to display",
            "type": 3,
            "required": False,
            "choices": [
                {"name": "Headlines", "value": "headlines"},
                {"name": "Summaries", "value": "summary"},
                {"name": "Full stored text", "value": "full"},
            ],
        },
    ],
}

DISCORD_COMMANDS = (ASK_COMMAND, NEWS_COMMAND)


def register_guild_command(
    *,
    application_id: str,
    guild_id: str,
    bot_token: str,
    command: dict[str, object] = ASK_COMMAND,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    if not application_id.isdigit() or not guild_id.isdigit():
        raise ValueError("Discord application and guild IDs must be numeric")
    if not bot_token.strip():
        raise ValueError("Discord bot token cannot be empty")
    http_client = client or httpx.Client(timeout=30)
    response = http_client.post(
        (
            f"{DISCORD_API_BASE}/applications/{application_id}/guilds/"
            f"{guild_id}/commands"
        ),
        headers={"Authorization": f"Bot {bot_token}"},
        json=command,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = " ".join(response.text.split())[:500]
        raise RuntimeError(
            f"Discord command registration failed with HTTP "
            f"{response.status_code}: {detail}"
        ) from error
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Discord returned an invalid command response")
    return payload


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    application_id = os.getenv("DISCORD_APPLICATION_ID", "").strip()
    guild_id = os.getenv("DISCORD_GUILD_ID", "").strip()
    bot_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    missing = [
        name
        for name, value in {
            "DISCORD_APPLICATION_ID": application_id,
            "DISCORD_GUILD_ID": guild_id,
            "DISCORD_BOT_TOKEN": bot_token,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError("Missing Discord configuration: " + ", ".join(missing))

    commands = [
        register_guild_command(
            application_id=application_id,
            guild_id=guild_id,
            bot_token=bot_token,
            command=command,
        )
        for command in DISCORD_COMMANDS
    ]
    print(
        json.dumps(
            {
                "status": "registered",
                "guild_id": guild_id,
                "commands": [
                    {
                        "command_id": command.get("id"),
                        "command_name": command.get("name"),
                        "version": command.get("version"),
                    }
                    for command in commands
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
