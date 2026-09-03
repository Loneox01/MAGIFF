"""Bounded deterministic news tools for one weekly lineup snapshot."""

from __future__ import annotations

import re
from typing import Any

from services.news import NewsDetail, NewsOutcome, NewsQuery, NewsService

from .models import LineupContext, LineupPlayer


GET_LINEUP_NEWS_TOOL = {
    "type": "function",
    "name": "get_recent_news",
    "description": (
        "Return newest-first maintained news for one player on this roster. "
        "Use it for injuries, practice participation, role changes, or another "
        "current issue that could materially affect a start/sit decision."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "player_ref": {
                "type": "string",
                "description": "Exact or unambiguous rostered player name.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["player_ref", "limit"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _name_key(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)


def _bounded_body(value: str, limit: int = 700) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


class LineupToolbox:
    def __init__(
        self,
        context: LineupContext,
        *,
        news: NewsService | None = None,
    ) -> None:
        self.context = context
        self.news = news or NewsService()

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [GET_LINEUP_NEWS_TOOL]

    @property
    def handlers(self) -> dict[str, Any]:
        return {"get_recent_news": self.get_recent_news}

    def resolve_player(self, player_ref: str) -> tuple[LineupPlayer | None, list[LineupPlayer]]:
        normalized = player_ref.strip()
        if not normalized:
            raise ValueError("player_ref cannot be blank")
        direct = self.context.player_by_id.get(normalized)
        if direct is not None:
            return direct, []
        key = _name_key(normalized)
        exact = [
            player
            for player in self.context.players
            if _name_key(player.display_name) == key
        ]
        partial = [
            player
            for player in self.context.players
            if key in _name_key(player.display_name)
        ]
        matches = exact or partial
        if len(matches) == 1:
            return matches[0], []
        return None, matches[:5]

    @staticmethod
    def _news_query(player: LineupPlayer, limit: int) -> NewsQuery:
        if player.position == "DEF" and player.team:
            return NewsQuery(team=player.team, count=limit, detail=NewsDetail.SUMMARY)
        return NewsQuery(
            player=player.display_name,
            count=limit,
            detail=NewsDetail.SUMMARY,
        )

    def get_recent_news(self, player_ref: str, limit: int) -> dict[str, Any]:
        if not 1 <= limit <= 5:
            raise ValueError("limit must be between 1 and 5")
        player, candidates = self.resolve_player(player_ref)
        if player is None:
            return {
                "status": "ambiguous" if candidates else "not_on_roster",
                "player_ref": player_ref,
                "candidates": [player.agent_view() for player in candidates],
                "retry": (
                    "Retry with one candidate's exact name."
                    if candidates
                    else "Use a player name from the verified lineup snapshot."
                ),
            }
        result = self.news.latest(self._news_query(player, limit))
        return {
            "status": result.outcome.value,
            "player": player.agent_view(),
            "reports": [
                {
                    "title": report.title,
                    "source": report.source,
                    "source_url": report.source_url,
                    "published_at": report.published_at,
                    "players": list(report.players),
                    "teams": list(report.teams),
                    "body": _bounded_body(report.body),
                }
                for report in result.reports
            ],
            "note": result.resolution_note,
        }

    def automatic_news(self, sleeper_player_id: str, limit: int = 3) -> dict[str, Any]:
        player = self.context.player_by_id.get(sleeper_player_id)
        if player is None:
            return {
                "status": NewsOutcome.PLAYER_NOT_FOUND.value,
                "error": "Player is not present in the verified roster snapshot.",
                "reports": [],
            }
        return self.get_recent_news(player.display_name, limit)
