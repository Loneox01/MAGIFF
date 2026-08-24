"""Discord UI and payload parsing for roster roulette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.roster_game import (
    GameAction,
    GameOutcome,
    GameResult,
    GameStatus,
    ROSTER_SLOTS,
    RosterGameService,
    RosterGameState,
)


GAME_CUSTOM_ID_PREFIX = "magiff_game"


@dataclass(frozen=True)
class GameStartQuery:
    season: int | None = None
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
        raise ValueError("Use `/game roster` to start roster roulette.")
    subcommand = options[0]
    if not isinstance(subcommand, dict) or subcommand.get("name") != "roster":
        raise ValueError("Use `/game roster` to start roster roulette.")
    raw_values = subcommand.get("options", [])
    if not isinstance(raw_values, list):
        raise ValueError("Discord supplied invalid game options.")
    values: dict[str, object] = {}
    for option in raw_values:
        if not isinstance(option, dict) or option.get("name") not in {
            "season",
            "reveal",
        }:
            raise ValueError("Discord supplied an unsupported game option.")
        values[str(option["name"])] = option.get("value")

    season = values.get("season")
    reveal = values.get("reveal", False)
    if season is not None and (isinstance(season, bool) or not isinstance(season, int)):
        raise ValueError("season must be a four-digit NFL season.")
    if season is not None and not 1999 <= season <= 2100:
        raise ValueError("season must be between 1999 and 2100.")
    if not isinstance(reveal, bool):
        raise ValueError("reveal must be true or false.")
    return GameStartQuery(season, reveal)


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


def _slot_label(state: RosterGameState, slot) -> str:
    pick = next((value for value in state.picks if value.roster_slot == slot), None)
    if pick is None:
        return f"- **{slot.value}:** —"
    if state.reveal_during_roll or state.status == GameStatus.COMPLETED:
        return (
            f"- **{slot.value}:** {pick.player.display_name} "
            f"({pick.player.team}) — {pick.player.fantasy_points_ppr:.1f} PPR"
        )
    return f"- **{slot.value}:** {pick.player.team} — locked"


def _current_roll(state: RosterGameState) -> list[str]:
    assert state.pending is not None
    pending = state.pending
    position = pending.player.position
    label = (
        f"{pending.roster_slot.value} ({position})"
        if pending.roster_slot.value == "FLEX"
        else position
    )
    lines = [
        f"### Roll {len(state.picks) + 1} of {len(ROSTER_SLOTS)}",
        f"**{pending.player.team} · {label}**",
    ]
    if state.reveal_during_roll:
        lines.append(
            f"**{pending.player.display_name}** — "
            f"{pending.player.fantasy_points_ppr:.1f} season PPR points"
        )
    else:
        lines.append("Player and points are hidden until the final roster.")
    return lines


def _embed(state: RosterGameState) -> list[dict[str, object]]:
    if state.pending is None:
        return []
    player = state.pending.player
    embed: dict[str, object] = {
        "title": f"{player.team} — {player.team_name}",
    }
    if player.team_logo_url:
        embed["thumbnail"] = {"url": player.team_logo_url}
    if player.team_color:
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
            ],
        }
    ]


def format_game_state(
    state: RosterGameState,
    game_service: RosterGameService,
    *,
    note: str | None = None,
) -> dict[str, object]:
    lines = [f"## Roster Roulette · {state.season}"]
    if note:
        lines.append(f"*{note}*")
    if state.status == GameStatus.ACTIVE:
        lines.extend(["", *_current_roll(state)])
    else:
        lines.extend(
            [
                "",
                "### Final Result",
                f"**{state.total_points:,.1f} PPR points**",
                f"## {state.wins}–{state.losses}",
            ]
        )
    lines.extend(["", "### Roster"])
    lines.extend(_slot_label(state, slot) for slot in ROSTER_SLOTS)
    if state.status == GameStatus.ACTIVE:
        lines.extend(
            [
                "",
                (
                    "Team reroll: **used**"
                    if state.team_reroll_used
                    else "Team reroll: **available**"
                ),
                (
                    "Position reroll: **used**"
                    if state.position_reroll_used
                    else "Position reroll: **available**"
                ),
            ]
        )
    return {
        "content": "\n".join(lines),
        "embeds": _embed(state),
        "components": _components(state, game_service),
        "allowed_mentions": {"parse": []},
    }


def format_game_error(result: GameResult) -> str:
    mapping = {
        GameOutcome.NOT_FOUND: (
            "That roster game no longer exists. Start another with `/game roster`."
        ),
        GameOutcome.NOT_OWNER: "Only the player who started this game can use its buttons.",
        GameOutcome.ALREADY_COMPLETE: "That roster game is already complete.",
        GameOutcome.REROLL_USED: result.note or "That reroll has already been used.",
        GameOutcome.REROLL_UNAVAILABLE: result.note or "That reroll is unavailable.",
        GameOutcome.SEASON_UNAVAILABLE: result.note or "That season is unavailable.",
    }
    return "## Roster Roulette\n" + mapping.get(
        result.outcome,
        result.note or "The game could not be updated. Retry in a moment.",
    )
