"""Discord UI and payload parsing for the 17-0 Challenge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.roster_game import (
    GameAction,
    GameOutcome,
    GameResult,
    GameScoringMode,
    GameStatus,
    ROSTER_SLOTS,
    RosterGameService,
    RosterGameState,
)


GAME_CUSTOM_ID_PREFIX = "magiff_game"


@dataclass(frozen=True)
class GameStartQuery:
    season: int | None = None
    scoring_mode: GameScoringMode = GameScoringMode.SEASON_TOTAL
    reveal_during_roll: bool = False


@dataclass(frozen=True)
class GameComponent:
    game_id: str
    expected_version: int
    action: GameAction


def extract_game_start(payload: dict[str, Any]) -> GameStartQuery:
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("name") != "game":
        raise ValueError("Interaction is not the /game command")
    options = data.get("options", [])
    if not isinstance(options, list) or len(options) != 1:
        raise ValueError("Use `/game challenge` to start the 17-0 Challenge.")
    subcommand = options[0]
    if not isinstance(subcommand, dict) or subcommand.get("name") != "challenge":
        raise ValueError("Use `/game challenge` to start the 17-0 Challenge.")
    raw_values = subcommand.get("options", [])
    if not isinstance(raw_values, list):
        raise ValueError("Discord supplied invalid game options.")
    values: dict[str, object] = {}
    for option in raw_values:
        if not isinstance(option, dict) or option.get("name") not in {
            "season",
            "scoring",
            "reveal",
        }:
            raise ValueError("Discord supplied an unsupported game option.")
        values[str(option["name"])] = option.get("value")

    season = values.get("season")
    scoring = values.get("scoring", GameScoringMode.SEASON_TOTAL.value)
    reveal = values.get("reveal", False)
    if season is not None and (isinstance(season, bool) or not isinstance(season, int)):
        raise ValueError("season must be a four-digit NFL season.")
    if season is not None and not 1999 <= season <= 2100:
        raise ValueError("season must be between 1999 and 2100.")
    try:
        scoring_mode = GameScoringMode(str(scoring))
    except ValueError as error:
        raise ValueError("scoring must be season totals or PPR PPG.") from error
    if not isinstance(reveal, bool):
        raise ValueError("reveal must be true or false.")
    return GameStartQuery(season, scoring_mode, reveal)


def extract_game_component(payload: dict[str, Any]) -> GameComponent:
    data = payload.get("data")
    custom_id = data.get("custom_id") if isinstance(data, dict) else None
    if not isinstance(custom_id, str):
        raise ValueError("This game button is invalid.")
    parts = custom_id.split(":")
    if len(parts) != 4 or parts[0] != GAME_CUSTOM_ID_PREFIX:
        raise ValueError("This game button is invalid.")
    game_id, raw_version, raw_action = parts[1:]
    try:
        version = int(raw_version)
        action = GameAction(raw_action)
    except (ValueError, TypeError) as error:
        raise ValueError("This game button is invalid.") from error
    if version < 0 or len(game_id) != 36:
        raise ValueError("This game button is invalid.")
    return GameComponent(game_id, version, action)


def discord_user(payload: dict[str, Any]) -> tuple[str, str | None]:
    member = payload.get("member")
    user = member.get("user") if isinstance(member, dict) else payload.get("user")
    if not isinstance(user, dict) or not str(user.get("id", "")).isdigit():
        raise ValueError("Discord user identity is missing.")
    display_name = None
    if isinstance(member, dict) and isinstance(member.get("nick"), str):
        display_name = member["nick"].strip() or None
    if display_name is None:
        for field in ("global_name", "username"):
            if isinstance(user.get(field), str) and user[field].strip():
                display_name = user[field].strip()
                break
    return str(user["id"]), display_name


def _slot_value(state: RosterGameState, slot) -> str:
    pick = next((value for value in state.picks if value.roster_slot == slot), None)
    if pick is None:
        return "-"
    if state.reveal_during_roll or state.status == GameStatus.COMPLETED:
        return (
            f"{pick.player.display_name} ({pick.player.team}) - "
            f"{pick.player.fantasy_points_ppr:.1f} "
            f"{'PPR PPG' if state.scoring_mode == GameScoringMode.PPG else 'PPR'}"
        )
    if state.status == GameStatus.ABANDONED:
        return (
            f"{pick.player.display_name} ({pick.player.team}) - "
            f"{pick.player.fantasy_points_ppr:.1f} "
            f"{'PPR PPG' if state.scoring_mode == GameScoringMode.PPG else 'PPR'}"
        )
    return f"{pick.player.team} 🔒"


def _roster_lines(state: RosterGameState) -> list[str]:
    return [
        f"**{slot.value}** - {_slot_value(state, slot)}"
        for slot in ROSTER_SLOTS
    ]


def _roster_fields(state: RosterGameState) -> list[dict[str, object]]:
    if state.status == GameStatus.ACTIVE and not state.reveal_during_roll:
        columns = (
            (ROSTER_SLOTS[0], ROSTER_SLOTS[1], ROSTER_SLOTS[2], ROSTER_SLOTS[6]),
            (ROSTER_SLOTS[5], ROSTER_SLOTS[3], ROSTER_SLOTS[4]),
        )
        return [
            {
                "name": "Roster" if index == 0 else "\u200b",
                "value": "\n".join(
                    f"**{slot.value}** - {_slot_value(state, slot)}"
                    for slot in slots
                ),
                "inline": True,
            }
            for index, slots in enumerate(columns)
        ]
    return [
        {
            "name": "Roster",
            "value": "\n".join(_roster_lines(state)),
            "inline": False,
        }
    ]


def _embed(state: RosterGameState) -> list[dict[str, object]]:
    lines: list[str] = []
    if state.status == GameStatus.ACTIVE:
        assert state.pending is not None
        player = state.pending.player
        position_label = (
            f"FLEX ({player.position})"
            if state.pending.roster_slot.value == "FLEX"
            else player.position
        )
        title = (
            f"Roll {len(state.picks) + 1}/{len(ROSTER_SLOTS)} - "
            f"{player.team} - {position_label}"
        )
        if state.reveal_during_roll:
            metric = (
                "PPR PPG"
                if state.scoring_mode == GameScoringMode.PPG
                else "season PPR"
            )
            lines.append(
                f"**{player.display_name}** - "
                f"{player.fantasy_points_ppr:.1f} {metric}"
            )
        else:
            lines.append("Player and points hidden until the final roster.")
    elif state.status == GameStatus.COMPLETED:
        player = None
        metric = (
            "PPR PPG"
            if state.scoring_mode == GameScoringMode.PPG
            else "PPR"
        )
        title = (
            f"Final - {state.total_points:,.1f} {metric} - "
            f"{state.wins}-{state.losses}"
        )
    else:
        player = None
        title = f"Run Forfeited - {len(state.picks)}/{len(ROSTER_SLOTS)} picks"
        metric = (
            "PPR PPG"
            if state.scoring_mode == GameScoringMode.PPG
            else "PPR"
        )
        lines.append(f"Saved subtotal: **{state.total_points:,.1f} {metric}**")

    embed: dict[str, object] = {
        "title": title,
        "description": "\n".join(lines),
        "fields": _roster_fields(state),
    }
    if state.status == GameStatus.ACTIVE:
        embed["fields"].append(
            {
                "name": "Rerolls",
                "value": (
                    "Team: "
                    f"**{'used' if state.team_reroll_used else 'available'}** | "
                    "Position: "
                    f"**{'used' if state.position_reroll_used else 'available'}**"
                ),
                "inline": False,
            }
        )
    if player is not None and player.team_logo_url:
        embed["thumbnail"] = {"url": player.team_logo_url}
    if player is not None and player.team_color:
        value = player.team_color.strip().lstrip("#")
        try:
            embed["color"] = int(value, 16)
        except ValueError:
            pass
    return [embed]


def _custom_id(state: RosterGameState, action: GameAction) -> str:
    return (
        f"{GAME_CUSTOM_ID_PREFIX}:{state.game_id}:{state.version}:{action.value}"
    )


def _components(
    state: RosterGameState,
    game_service: RosterGameService,
) -> list[dict[str, object]]:
    if state.status != GameStatus.ACTIVE:
        return []
    return [
        {
            "type": 1,
            "components": [
                {
                    "type": 2,
                    "style": 2,
                    "label": "Reroll Team",
                    "custom_id": _custom_id(state, GameAction.REROLL_TEAM),
                    "disabled": not game_service.can_reroll_team(state),
                },
                {
                    "type": 2,
                    "style": 2,
                    "label": "Reroll Position",
                    "custom_id": _custom_id(
                        state,
                        GameAction.REROLL_POSITION,
                    ),
                    "disabled": not game_service.can_reroll_position(state),
                },
                {
                    "type": 2,
                    "style": 1,
                    "label": (
                        "Lock & Reveal Final"
                        if len(state.picks) == len(ROSTER_SLOTS) - 1
                        else "Lock & Spin Next"
                    ),
                    "custom_id": _custom_id(state, GameAction.LOCK),
                },
                {
                    "type": 2,
                    "style": 4,
                    "label": "Forfeit Run",
                    "custom_id": _custom_id(state, GameAction.FORFEIT),
                },
            ],
        }
    ]


def format_game_state(
    state: RosterGameState,
    game_service: RosterGameService,
    *,
    note: str | None = None,
) -> dict[str, object]:
    embed = _embed(state)[0]
    if note:
        description = str(embed.get("description") or "")
        embed["description"] = f"**{note}**\n{description}"
    return {
        "content": (
            f"## 17-0 Challenge - {state.season} - "
            f"{'PPR PPG' if state.scoring_mode == GameScoringMode.PPG else 'Season Total'}"
        ),
        "embeds": [embed],
        "components": _components(state, game_service),
        "allowed_mentions": {"parse": []},
    }


def format_game_error(result: GameResult) -> str:
    mapping = {
        GameOutcome.NOT_FOUND: (
            "That challenge no longer exists. Start another with `/game challenge`."
        ),
        GameOutcome.NOT_OWNER: "Only the player who started this game can use its buttons.",
        GameOutcome.ALREADY_COMPLETE: "That challenge is already finished.",
        GameOutcome.REROLL_USED: result.note or "That reroll has already been used.",
        GameOutcome.REROLL_UNAVAILABLE: result.note or "That reroll is unavailable.",
        GameOutcome.SEASON_UNAVAILABLE: result.note or "That season is unavailable.",
    }
    return "## 17-0 Challenge\n" + mapping.get(
        result.outcome,
        result.note or "The game could not be updated. Retry in a moment.",
    )
