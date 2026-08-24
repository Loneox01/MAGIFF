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
            "description": "Number of recent reports to return (default 3)",
            "type": 4,
            "required": False,
            "min_value": 1,
            "max_value": 10,
        },
        {
            "name": "detail",
            "description": "Display headlines, summaries, or full text (default headlines)",
            "type": 3,
            "required": False,
            "choices": [
                {"name": "Headlines", "value": "headlines"},
                {"name": "Summaries", "value": "summary"},
                {"name": "Full stored text", "value": "full"},
            ],
        },
        {
            "name": "previews",
            "description": "Show Discord's large link preview cards (default false)",
            "type": 5,
            "required": False,
        },
    ],
}

SEASON_TYPE_OPTION = {
    "name": "season_type",
    "description": "Regular season (default) or postseason",
    "type": 3,
    "required": False,
    "choices": [
        {"name": "Regular season", "value": "REG"},
        {"name": "Postseason", "value": "POST"},
    ],
}
PERSPECTIVE_OPTION = {
    "name": "perspective",
    "description": "Team offense (default) or defense allowed",
    "type": 3,
    "required": False,
    "choices": [
        {"name": "Offense", "value": "offense"},
        {"name": "Defense allowed", "value": "defense"},
    ],
}
DIRECTION_OPTION = {
    "name": "direction",
    "description": "Highest values first (default) or lowest first",
    "type": 3,
    "required": False,
    "choices": [
        {"name": "Highest", "value": "desc"},
        {"name": "Lowest", "value": "asc"},
    ],
}
COUNT_OPTION = {
    "name": "count",
    "description": "Number of rows to return (default 5)",
    "type": 4,
    "required": False,
    "min_value": 1,
    "max_value": 10,
}

STATS_COMMAND = {
    "name": "stats",
    "type": 1,
    "description": "Look up or rank NFL stats with safe custom formulas",
    "options": [
        {
            "name": "player",
            "description": "Show one player's season or weekly statistics",
            "type": 1,
            "options": [
                {"name": "player", "description": "Full or partial player name", "type": 3, "required": True, "min_length": 1, "max_length": 100},
                {"name": "season", "description": "NFL season; defaults to latest stored season", "type": 4, "required": False, "min_value": 1999, "max_value": 2100},
                {"name": "week", "description": "Optional week; switches to weekly fields", "type": 4, "required": False, "min_value": 1, "max_value": 22},
                {"name": "view", "description": "Preset shown when formula is omitted", "type": 3, "required": False, "choices": [{"name": value.title(), "value": value} for value in ("summary", "fantasy", "passing", "rushing", "receiving", "usage")]},
                {"name": "formula", "description": "Optional formula; use autocomplete for fields", "type": 3, "required": False, "autocomplete": True, "max_length": 100},
                SEASON_TYPE_OPTION,
            ],
        },
        {
            "name": "leaders",
            "description": "Rank players by a custom season formula",
            "type": 1,
            "options": [
                {"name": "formula", "description": "Formula assembled from autocomplete fields", "type": 3, "required": True, "autocomplete": True, "max_length": 100},
                {"name": "season", "description": "NFL season; defaults to latest stored season", "type": 4, "required": False, "min_value": 1999, "max_value": 2100},
                {"name": "position", "description": "Optional position filter", "type": 3, "required": False, "choices": [{"name": value, "value": value} for value in ("QB", "RB", "WR", "TE", "K")]},
                {"name": "minimum_field", "description": "Optional eligibility field", "type": 3, "required": False, "autocomplete": True, "max_length": 100},
                {"name": "minimum_value", "description": "Required minimum for minimum_field", "type": 10, "required": False, "min_value": 0},
                DIRECTION_OPTION,
                COUNT_OPTION,
                SEASON_TYPE_OPTION,
            ],
        },
        {
            "name": "team",
            "description": "Show one team's offense or defense statistics",
            "type": 1,
            "options": [
                {"name": "team", "description": "Team abbreviation or official name", "type": 3, "required": True, "min_length": 2, "max_length": 40},
                {"name": "season", "description": "NFL season; defaults to latest stored season", "type": 4, "required": False, "min_value": 1999, "max_value": 2100},
                {"name": "week", "description": "Optional single week", "type": 4, "required": False, "min_value": 1, "max_value": 22},
                PERSPECTIVE_OPTION,
                {"name": "view", "description": "Preset shown when formula is omitted", "type": 3, "required": False, "choices": [{"name": value.title(), "value": value} for value in ("summary", "passing", "rushing")]},
                {"name": "formula", "description": "Optional formula; use autocomplete for fields", "type": 3, "required": False, "autocomplete": True, "max_length": 100},
                SEASON_TYPE_OPTION,
            ],
        },
        {
            "name": "team-leaders",
            "description": "Rank teams by a custom offense or defense formula",
            "type": 1,
            "options": [
                {"name": "formula", "description": "Formula assembled from autocomplete fields", "type": 3, "required": True, "autocomplete": True, "max_length": 100},
                {"name": "season", "description": "NFL season; defaults to latest stored season", "type": 4, "required": False, "min_value": 1999, "max_value": 2100},
                PERSPECTIVE_OPTION,
                {"name": "minimum_games", "description": "Minimum games played", "type": 4, "required": False, "min_value": 1, "max_value": 25},
                DIRECTION_OPTION,
                COUNT_OPTION,
                SEASON_TYPE_OPTION,
            ],
        },
        {
            "name": "fields",
            "description": "Browse valid formula fields",
            "type": 1,
            "options": [
                {"name": "scope", "description": "Formula field catalog", "type": 3, "required": True, "choices": [{"name": name, "value": value} for name, value in (("Player season", "player-season"), ("Player weekly", "player-weekly"), ("Team offense", "team-offense"), ("Team defense allowed", "team-defense"))]},
                {"name": "search", "description": "Optional text contained in the field name", "type": 3, "required": False, "max_length": 50},
                {"name": "count", "description": "Number of fields to show", "type": 4, "required": False, "min_value": 1, "max_value": 25},
            ],
        },
    ],
}

GAME_COMMAND = {
    "name": "game",
    "type": 1,
    "description": "Play the MAGIFF 17-0 Challenge",
    "options": [
        {
            "name": "challenge",
            "description": "Build a seven-player roster and chase a 17-0 record",
            "type": 1,
            "options": [
                {
                    "name": "season",
                    "description": "Completed NFL season; defaults to latest stored season",
                    "type": 4,
                    "required": False,
                    "min_value": 1999,
                    "max_value": 2100,
                },
                {
                    "name": "scoring",
                    "description": "Season PPR total (default) or PPR points per game",
                    "type": 3,
                    "required": False,
                    "choices": [
                        {"name": "Season total", "value": "season_total"},
                        {"name": "PPR points per game", "value": "ppg"},
                    ],
                },
                {
                    "name": "reveal",
                    "description": "Show each player and score during rolls (default false)",
                    "type": 5,
                    "required": False,
                },
            ],
        }
    ],
}

TEST_COMMANDS = (ASK_COMMAND, NEWS_COMMAND, STATS_COMMAND, GAME_COMMAND)
UAI_COMMANDS = (NEWS_COMMAND, STATS_COMMAND, GAME_COMMAND)
# Backwards-compatible name for callers that expect the full command catalog.
DISCORD_COMMANDS = TEST_COMMANDS


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
    test_guild_id = os.getenv("DISCORD_TEST_GUILD_ID", "").strip()
    uai_guild_id = os.getenv("DISCORD_UAI_GUILD_ID", "").strip()
    bot_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    missing = [
        name
        for name, value in {
            "DISCORD_APPLICATION_ID": application_id,
            "DISCORD_TEST_GUILD_ID": test_guild_id,
            "DISCORD_BOT_TOKEN": bot_token,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError("Missing Discord configuration: " + ", ".join(missing))
    if uai_guild_id and uai_guild_id == test_guild_id:
        raise RuntimeError("Discord test and UAI guild IDs must be different")

    profiles = [("test", test_guild_id, TEST_COMMANDS)]
    if uai_guild_id:
        profiles.append(("uai", uai_guild_id, UAI_COMMANDS))

    registrations = []
    for profile, guild_id, profile_commands in profiles:
        commands = [
            register_guild_command(
                application_id=application_id,
                guild_id=guild_id,
                bot_token=bot_token,
                command=command,
            )
            for command in profile_commands
        ]
        registrations.append(
            {
                "profile": profile,
                "guild_id": guild_id,
                "commands": [
                    {
                        "command_id": command.get("id"),
                        "command_name": command.get("name"),
                        "version": command.get("version"),
                    }
                    for command in commands
                ],
            }
        )
    print(
        json.dumps(
            {
                "status": "registered",
                "profiles": registrations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
