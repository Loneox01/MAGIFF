"""Discord parsing, autocomplete, formatting, and delivery for /stats."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from services.news import PlayerCandidate, TeamCandidate
from services.stats import (
    DEFAULT_STATS_COUNT,
    PlayerLeadersQuery,
    PlayerStatsQuery,
    StatsFieldsQuery,
    StatsOutcome,
    StatsQuery,
    StatsResult,
    StatsScope,
    StatsService,
    TeamLeadersQuery,
    TeamStatsQuery,
    formula_fields,
)

from .discord import DiscordWebhookClient, split_discord_message


LOGGER = logging.getLogger(__name__)
FORMULA_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def _subcommand(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("name") != "stats":
        raise ValueError("Interaction is not the /stats command")
    options = data.get("options")
    if not isinstance(options, list) or len(options) != 1 or not isinstance(options[0], dict):
        raise ValueError("Choose one /stats subcommand")
    command = options[0]
    name = command.get("name")
    nested = command.get("options", [])
    if name not in {"player", "leaders", "team", "team-leaders", "fields"} or not isinstance(nested, list):
        raise ValueError("Discord supplied an invalid /stats subcommand")
    return str(name), nested


def _values(options: list[dict[str, Any]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("name"), str):
            raise ValueError("Discord supplied an invalid /stats option")
        values[option["name"]] = option.get("value")
    return values


def _string(values: dict[str, object], name: str, default: str | None = None) -> str | None:
    value = values.get(name, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _integer(values: dict[str, object], name: str) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(values: dict[str, object], name: str) -> float | None:
    value = values.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def extract_stats_query(payload: dict[str, Any]) -> StatsQuery:
    name, options = _subcommand(payload)
    values = _values(options)
    season = _integer(values, "season")
    season_type = _string(values, "season_type", "REG") or "REG"
    if name == "player":
        player = _string(values, "player")
        if player is None:
            raise ValueError("player is required")
        return PlayerStatsQuery(
            player=player,
            season=season,
            week=_integer(values, "week"),
            season_type=season_type,
            view=_string(values, "view", "summary") or "summary",
            formula=_string(values, "formula"),
        )
    if name == "leaders":
        formula = _string(values, "formula")
        if formula is None:
            raise ValueError("formula is required")
        return PlayerLeadersQuery(
            formula=formula,
            season=season,
            season_type=season_type,
            position=_string(values, "position"),
            minimum_field=_string(values, "minimum_field"),
            minimum_value=_number(values, "minimum_value"),
            sort_direction=_string(values, "direction", "desc") or "desc",
            count=_integer(values, "count") or DEFAULT_STATS_COUNT,
        )
    if name == "team":
        team = _string(values, "team")
        if team is None:
            raise ValueError("team is required")
        return TeamStatsQuery(
            team=team,
            season=season,
            week=_integer(values, "week"),
            season_type=season_type,
            perspective=_string(values, "perspective", "offense") or "offense",
            view=_string(values, "view", "summary") or "summary",
            formula=_string(values, "formula"),
        )
    if name == "team-leaders":
        formula = _string(values, "formula")
        if formula is None:
            raise ValueError("formula is required")
        return TeamLeadersQuery(
            formula=formula,
            season=season,
            season_type=season_type,
            perspective=_string(values, "perspective", "offense") or "offense",
            minimum_games=_integer(values, "minimum_games"),
            sort_direction=_string(values, "direction", "desc") or "desc",
            count=_integer(values, "count") or DEFAULT_STATS_COUNT,
        )
    scope = _string(values, "scope")
    if scope is None:
        raise ValueError("scope is required")
    try:
        selected_scope = StatsScope(scope)
    except ValueError as error:
        raise ValueError("invalid stats field scope") from error
    return StatsFieldsQuery(
        scope=selected_scope,
        search=_string(values, "search"),
        count=_integer(values, "count") or 25,
    )


def stats_autocomplete_choices(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Compose a formula by replacing only its currently focused field token."""
    try:
        name, options = _subcommand(payload)
    except ValueError:
        return []
    focused = next((option for option in options if option.get("focused") is True), None)
    if not isinstance(focused, dict):
        return []
    option_name = focused.get("name")
    value = focused.get("value", "")
    if option_name not in {"formula", "minimum_field"} or not isinstance(value, str):
        return []

    values = _values(options)
    if name == "player" and values.get("week") is not None:
        scope = StatsScope.PLAYER_WEEKLY
    elif name in {"player", "leaders"}:
        scope = StatsScope.PLAYER_SEASON
    else:
        scope = StatsScope.TEAM_DEFENSE if values.get("perspective") == "defense" else StatsScope.TEAM_OFFENSE
    fields = formula_fields(scope)

    if option_name == "minimum_field":
        prefix = ""
        needle = value.casefold()
    else:
        match = FORMULA_TOKEN.search(value)
        prefix = value[: match.start()] if match else value
        needle = match.group(0).casefold() if match else ""
    ranked = sorted(fields, key=lambda field: (not field.casefold().startswith(needle), field))
    choices = []
    for field in ranked:
        if needle and needle not in field.casefold():
            continue
        completed = field if option_name == "minimum_field" else prefix + field
        if len(completed) <= 100:
            choices.append({"name": field, "value": completed})
        if len(choices) == 25:
            break
    return choices


def _scope(query: StatsQuery) -> str:
    if isinstance(query, PlayerStatsQuery):
        return f"player: {query.player}" + (f" · week {query.week}" if query.week else "")
    if isinstance(query, PlayerLeadersQuery):
        return f"player leaders · `{query.formula}`"
    if isinstance(query, TeamStatsQuery):
        return f"team: {query.team} · {query.perspective}" + (f" · week {query.week}" if query.week else "")
    if isinstance(query, TeamLeadersQuery):
        return f"team leaders · {query.perspective} · `{query.formula}`"
    return f"fields: {query.scope.value}"


def format_stats_pending(query: StatsQuery) -> str:
    return f"## Stats\n> {_scope(query)}\n\n*Reading structured data…*"


def _human(field: str) -> str:
    return field.replace("_", " ").title()


def _value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _candidate(candidate: PlayerCandidate) -> str:
    context = " · ".join(value for value in (candidate.team, candidate.position) if value)
    return f"- **{candidate.display_name}**" + (f" — {context}" if context else "")


def _team_candidate(candidate: TeamCandidate) -> str:
    return f"- **{candidate.code}** — {candidate.name}"


def format_stats_result(result: StatsResult) -> str:
    query = result.query
    if result.outcome == StatsOutcome.PLAYER_NOT_FOUND:
        return f"## Player not found\nI couldn't match `{getattr(query, 'player', '')}`. Retry with a fuller canonical name."
    if result.outcome == StatsOutcome.PLAYER_AMBIGUOUS:
        return "\n".join(["## Player is ambiguous", "Retry with one exact full name:", *map(_candidate, result.player_candidates)])
    if result.outcome in {StatsOutcome.TEAM_NOT_FOUND, StatsOutcome.TEAM_AMBIGUOUS}:
        lines = ["## Team not resolved", f"I couldn't resolve `{getattr(query, 'team', '')}` to one current NFL team."]
        if result.team_candidates:
            lines.extend(["", "Possible retries:", *map(_team_candidate, result.team_candidates)])
        return "\n".join(lines)
    if result.outcome == StatsOutcome.INVALID_FORMULA:
        return f"## Invalid formula\n`{result.error or 'That expression is not supported.'}`\n\nUse autocomplete to insert valid fields; operators are `+`, `-`, `*`, `/`, and parentheses."
    if result.outcome == StatsOutcome.ZERO_DENOMINATOR:
        return "## Undefined result\nThe formula divided by zero for this row. Try a different scope or denominator."
    if result.outcome == StatsOutcome.NO_STATS:
        return f"## No matching stats\nNo stored structured rows matched **{_scope(query)}**. Try an earlier season or remove the week/filter."

    if isinstance(query, StatsFieldsQuery):
        fields = [str(row["field"]) for row in result.rows]
        listing = ", ".join(f"`{field}`" for field in fields)
        return f"## Available Fields — {query.scope.value}\n" + (listing if fields else "No fields matched.")

    lines = ["## Stats"]
    if result.resolved_player:
        lines.append(f"**{result.resolved_player.display_name}** · {result.season}" + (f" Week {query.week}" if isinstance(query, PlayerStatsQuery) and query.week else ""))
    elif result.resolved_team:
        lines.append(f"**{result.resolved_team.name} ({result.resolved_team.code})** · {result.season}" + (f" Week {query.week}" if isinstance(query, TeamStatsQuery) and query.week else ""))
    else:
        lines.append(f"**{result.season}**" + (f" · `{result.formula}`" if result.formula else ""))
    if result.resolution_note:
        lines.append(result.resolution_note)

    rankings = isinstance(query, (PlayerLeadersQuery, TeamLeadersQuery))
    for index, row in enumerate(result.rows, start=1):
        if rankings:
            label = row.get("display_name") or row.get("team_name") or row.get("team") or "Unknown"
            context = " · ".join(str(value) for value in (row.get("team"), row.get("position")) if value)
            lines.append(f"{index}. **{label}** — **{_value(row.get('metric_value'))}**" + (f" ({context})" if context else ""))
            inputs = row.get("inputs") or {}
            if inputs:
                lines.append("   " + " · ".join(f"{_human(key)}: {_value(value)}" for key, value in inputs.items()))
        elif result.formula:
            lines.append(f"**{result.formula} = {_value(row.get('metric_value'))}**")
            lines.append(" · ".join(f"{_human(key)}: {_value(value)}" for key, value in (row.get("inputs") or {}).items()))
        else:
            lines.extend(f"- **{_human(key)}:** {_value(value)}" for key, value in row.items() if value is not None)
    return "\n".join(lines)


@dataclass(frozen=True)
class StatsCompletion:
    interaction_id: str
    interaction_token: str
    query: StatsQuery
    request_id: str


class DiscordStatsRunner:
    def __init__(self, *, application_id: str, stats_service: StatsService, webhook_client: DiscordWebhookClient | None = None) -> None:
        self.application_id = application_id
        self.stats_service = stats_service
        self.webhook_client = webhook_client or DiscordWebhookClient()

    def complete(self, completion: StatsCompletion) -> None:
        started_at = time.perf_counter()
        try:
            result = self.stats_service.execute(completion.query)
            messages = split_discord_message(format_stats_result(result))
            self.webhook_client.edit_original(application_id=self.application_id, interaction_token=completion.interaction_token, content=messages[0])
            for message in messages[1:]:
                self.webhook_client.create_followup(application_id=self.application_id, interaction_token=completion.interaction_token, content=message)
            LOGGER.info(json.dumps({"event": "discord_stats_request_complete", "request_id": completion.request_id, "interaction_id": completion.interaction_id, "query_type": type(completion.query).__name__, "outcome": result.outcome.value, "season": result.season, "formula": result.formula, "result_count": len(result.rows), "latency_seconds": round(time.perf_counter() - started_at, 3), "messages": len(messages)}, separators=(",", ":")))
        except ValueError as error:
            LOGGER.info(
                "Rejected invalid Discord stats request request_id=%s error=%s",
                completion.request_id,
                error,
            )
            self.webhook_client.edit_original(
                application_id=self.application_id,
                interaction_token=completion.interaction_token,
                content=(
                    "## Invalid stats request\n"
                    f"{error}\n\nRetry using the displayed choices and field "
                    "autocomplete."
                ),
            )
        except Exception:
            LOGGER.exception("Discord stats request failed request_id=%s interaction_id=%s", completion.request_id, completion.interaction_id)
            try:
                self.webhook_client.edit_original(application_id=self.application_id, interaction_token=completion.interaction_token, content="## Stats unavailable\nI couldn't read the structured store. Retry in a moment or simplify the filters.\n\n" f"Request ID: `{completion.request_id}`")
            except Exception:
                LOGGER.exception("Discord stats failure response could not be delivered request_id=%s", completion.request_id)
