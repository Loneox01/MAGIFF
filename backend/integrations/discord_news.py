"""Discord parsing, formatting, and delivery for the deterministic /news command."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from services.news import (
    DEFAULT_NEWS_COUNT,
    NewsDetail,
    NewsOutcome,
    NewsQuery,
    NewsReport,
    NewsResult,
    NewsService,
    PlayerCandidate,
    TeamCandidate,
)

from .discord import DiscordWebhookClient, split_discord_message


LOGGER = logging.getLogger(__name__)


def extract_news_query(payload: dict[str, Any]) -> NewsQuery:
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("name") != "news":
        raise ValueError("Interaction is not the /news command")
    raw_options = data.get("options", [])
    if not isinstance(raw_options, list):
        raise ValueError("Discord supplied invalid /news options")

    options: dict[str, object] = {}
    for item in raw_options:
        if not isinstance(item, dict):
            raise ValueError("Discord supplied an invalid /news option")
        name = item.get("name")
        if name not in {"player", "team", "count", "detail", "previews"}:
            raise ValueError(f"Unsupported /news option: {name}")
        options[str(name)] = item.get("value")

    count = options.get("count", DEFAULT_NEWS_COUNT)
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("count must be an integer between 1 and 10")
    detail_value = options.get("detail", NewsDetail.HEADLINES.value)
    if not isinstance(detail_value, str):
        raise ValueError("detail must be headlines, summary, or full")
    try:
        detail = NewsDetail(detail_value)
    except ValueError as error:
        raise ValueError("detail must be headlines, summary, or full") from error

    player = options.get("player")
    team = options.get("team")
    previews = options.get("previews", False)
    if player is not None and not isinstance(player, str):
        raise ValueError("player must be a name")
    if team is not None and not isinstance(team, str):
        raise ValueError("team must be an NFL abbreviation or official name")
    if not isinstance(previews, bool):
        raise ValueError("previews must be true or false")
    return NewsQuery(
        count=count,
        detail=detail,
        player=player.strip() if player else None,
        team=team.strip() if team else None,
        previews=previews,
    )


def _request_scope(query: NewsQuery) -> str:
    values = [f"latest {query.count}", f"detail: {query.detail.value}"]
    if query.player:
        values.append(f"player: {query.player}")
    if query.team:
        values.append(f"team: {query.team}")
    if query.previews:
        values.append("previews: on")
    return " · ".join(values)


def format_news_pending(query: NewsQuery) -> str:
    return f"## News\n> {_request_scope(query)}\n\n*Fetching stored reports…*"


def _clean_inline(value: str) -> str:
    return " ".join(value.split())


def _link_label(value: str) -> str:
    return _clean_inline(value).replace("[", "").replace("]", "")


def _summary(value: str, limit: int = 420) -> str:
    normalized = _clean_inline(value)
    if len(normalized) <= limit:
        return normalized
    shortened = normalized[: limit - 1].rsplit(" ", 1)[0].rstrip(".,;:")
    return shortened + "…"


def _published_date(value: str) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else value


def _report_heading(index: int, report: NewsReport) -> str:
    return f"### {index}. [{_link_label(report.title)}]({report.source_url})"


def _report_metadata(report: NewsReport) -> str:
    values = [report.source, _published_date(report.published_at)]
    if report.players:
        values.append(", ".join(report.players[:4]))
    elif report.teams:
        values.append(", ".join(report.teams[:4]))
    return " · ".join(values)


def _format_success(result: NewsResult) -> str:
    lines = ["## Latest News"]
    filters = []
    if result.resolved_player:
        filters.append(f"player: **{result.resolved_player.display_name}**")
    if result.resolved_team:
        filters.append(f"team: **{result.resolved_team.code}**")
    if filters:
        lines.append("Filtered by " + " · ".join(filters))
    if result.resolution_note:
        lines.append(result.resolution_note)
    if result.full_view_capped:
        lines.append(
            "Full view is capped at three reports; use `detail:summary` for more."
        )

    for index, report in enumerate(result.reports, start=1):
        if result.query.detail == NewsDetail.HEADLINES:
            lines.extend(
                [
                    "",
                    (
                        f"**{index}.** [{_link_label(report.title)}]"
                        f"({report.source_url})"
                    ),
                    _report_metadata(report),
                ]
            )
            continue
        lines.extend(
            [
                "",
                _report_heading(index, report),
                _report_metadata(report),
                (
                    report.body.strip()
                    if result.query.detail == NewsDetail.FULL
                    else _summary(report.body)
                ),
            ]
        )
    return "\n".join(lines)


def _candidate_line(candidate: PlayerCandidate) -> str:
    attributes = [value for value in (candidate.team, candidate.position) if value]
    suffix = f" — {' · '.join(attributes)}" if attributes else ""
    return f"- **{candidate.display_name}**{suffix}"


def _team_line(candidate: TeamCandidate) -> str:
    return f"- **{candidate.code}** — {candidate.name}"


def _format_punt(result: NewsResult) -> str:
    query = result.query
    if result.outcome == NewsOutcome.PLAYER_NOT_FOUND:
        value = query.player or "that player"
        surname = value.split()[-1] if value.split() else value
        return (
            "## Player not found\n"
            f"I couldn't match `{value}` to a stored player. No news was returned.\n\n"
            "Check the spelling or retry with a canonical or partial name, for "
            f"example `/news player:{surname}`."
        )
    if result.outcome == NewsOutcome.PLAYER_AMBIGUOUS:
        lines = [
            "## Player is ambiguous",
            f"`{query.player}` matched multiple stored players:",
            *(_candidate_line(value) for value in result.player_candidates),
            "",
            "Retry with the full name and, when useful, a team filter—for example "
            "`/news player:Full Name team:MIN`.",
        ]
        return "\n".join(lines)
    if result.outcome in {
        NewsOutcome.TEAM_NOT_FOUND,
        NewsOutcome.TEAM_AMBIGUOUS,
    }:
        heading = (
            "Team is ambiguous"
            if result.outcome == NewsOutcome.TEAM_AMBIGUOUS
            else "Team not found"
        )
        lines = [
            f"## {heading}",
            f"I couldn't resolve `{query.team}` to one current NFL team.",
        ]
        if result.team_candidates:
            lines.extend(
                ["", "Possible retries:", *map(_team_line, result.team_candidates)]
            )
        lines.extend(
            ["", "Retry using the team abbreviation, such as `/news team:TB`."]
        )
        return "\n".join(lines)
    if result.outcome == NewsOutcome.NO_REPORTS:
        filters = []
        if result.resolved_player:
            filters.append(result.resolved_player.display_name)
        if result.resolved_team:
            filters.append(result.resolved_team.code)
        scope = " + ".join(filters) or "those filters"
        return (
            "## No matching news\n"
            f"No active stored reports matched **{scope}**.\n\n"
            "Try removing one filter or run `/news count:3` to inspect the latest "
            "unfiltered reports."
        )
    raise ValueError(f"Unsupported news outcome: {result.outcome}")


def format_news_result(result: NewsResult) -> str:
    if result.outcome == NewsOutcome.SUCCESS:
        return _format_success(result)
    return _format_punt(result)


@dataclass(frozen=True)
class NewsCompletion:
    interaction_id: str
    interaction_token: str
    query: NewsQuery
    request_id: str


class DiscordNewsRunner:
    def __init__(
        self,
        *,
        application_id: str,
        news_service: NewsService,
        webhook_client: DiscordWebhookClient | None = None,
    ) -> None:
        self.application_id = application_id
        self.news_service = news_service
        self.webhook_client = webhook_client or DiscordWebhookClient()

    def complete(self, completion: NewsCompletion) -> None:
        started_at = time.perf_counter()
        try:
            result = self.news_service.latest(completion.query)
            messages = split_discord_message(format_news_result(result))
            self.webhook_client.edit_original(
                application_id=self.application_id,
                interaction_token=completion.interaction_token,
                content=messages[0],
                suppress_embeds=not completion.query.previews,
            )
            for message in messages[1:]:
                self.webhook_client.create_followup(
                    application_id=self.application_id,
                    interaction_token=completion.interaction_token,
                    content=message,
                    suppress_embeds=not completion.query.previews,
                )
            LOGGER.info(
                json.dumps(
                    {
                        "event": "discord_news_request_complete",
                        "request_id": completion.request_id,
                        "interaction_id": completion.interaction_id,
                        "outcome": result.outcome.value,
                        "requested_count": completion.query.count,
                        "result_count": len(result.reports),
                        "detail": completion.query.detail.value,
                        "player_filter": result.resolved_player.display_name
                        if result.resolved_player
                        else completion.query.player,
                        "team_filter": result.resolved_team.code
                        if result.resolved_team
                        else completion.query.team,
                        "latency_seconds": round(
                            time.perf_counter() - started_at, 3
                        ),
                        "messages": len(messages),
                    },
                    separators=(",", ":"),
                )
            )
        except Exception:
            LOGGER.exception(
                "Discord news request failed request_id=%s interaction_id=%s",
                completion.request_id,
                completion.interaction_id,
            )
            try:
                self.webhook_client.edit_original(
                    application_id=self.application_id,
                    interaction_token=completion.interaction_token,
                    content=(
                        "## News unavailable\n"
                        "I couldn't read the maintained report store. Retry in a "
                        "moment. If it keeps failing, check `/ready`.\n\n"
                        f"Request ID: `{completion.request_id}`"
                    ),
                )
            except Exception:
                LOGGER.exception(
                    "Discord news failure response could not be delivered "
                    "request_id=%s",
                    completion.request_id,
                )
