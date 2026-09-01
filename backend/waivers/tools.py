"""Bounded discovery tools over one immutable waiver snapshot."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rag.planning.schema_values import TeamCode
from repositories.league_supabase import SupabaseWaiverPlayerRepository
from services.news import NewsDetail, NewsOutcome, NewsQuery, NewsService
from tools.nfl import get_player_season_stats, get_team_depth_chart

from .models import WaiverCandidate, WaiverContext


WAIVER_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
WAIVER_SORTS = ["fantasycalc_value", "fantasycalc_trend_30_day", "ecr"]
CURRENT_TEAM_CODES = [
    value.value
    for value in TeamCode
    if value.value not in {"OAK", "SD", "STL"}
]


def _nullable_enum(values: list[str], description: str) -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "string", "enum": values},
            {"type": "null"},
        ],
        "description": description,
    }


RANK_AVAILABLE_PLAYERS_TOOL = {
    "type": "function",
    "name": "rank_available_players",
    "description": (
        "Return a small ranked slice of players confirmed available in this "
        "league. Filter by position, team, or both for needs, handcuffs, and "
        "depth-chart alternatives. This searches the full loaded market pool; "
        "the default waiver snapshot intentionally does not expose that pool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "position": _nullable_enum(
                WAIVER_POSITIONS,
                "Fantasy position, or null for every position.",
            ),
            "team": _nullable_enum(
                CURRENT_TEAM_CODES,
                "Canonical NFL team abbreviation, or null for every team.",
            ),
            "sort_by": {
                "type": "string",
                "enum": WAIVER_SORTS,
                "description": "Market value, 30-day market trend, or ECR.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["position", "team", "sort_by", "limit"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_AVAILABLE_PLAYER_TOOL = {
    "type": "function",
    "name": "get_available_player",
    "description": (
        "Look up one named player in this league's available pool and return "
        "market, ECR, prior-season production, and current depth-chart context. "
        "Use this to investigate a sleeper or stash identified outside the top lists."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "player_ref": {
                "type": "string",
                "description": "Player name to verify against league availability.",
            }
        },
        "required": ["player_ref"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_RECENT_WAIVER_NEWS_TOOL = {
    "type": "function",
    "name": "get_recent_news",
    "description": (
        "Get the newest maintained reports deterministically, ordered newest "
        "first, for a player, team, or player constrained to a team. Use this "
        "before deeper report search when checking current waiver news."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "player_ref": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Player name, or null for team-only news.",
            },
            "team": _nullable_enum(
                CURRENT_TEAM_CODES,
                "Canonical team abbreviation, or null for player/global news.",
            ),
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": ["player_ref", "team", "limit"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _bounded_body(value: str, limit: int = 700) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


class WaiverToolbox:
    """Tool handlers scoped to one league snapshot and availability set."""

    def __init__(
        self,
        context: WaiverContext,
        *,
        news: NewsService | None = None,
        season_stats: Callable[..., dict] = get_player_season_stats,
        depth_chart: Callable[..., list[dict]] = get_team_depth_chart,
        player_search: Any | None = None,
    ) -> None:
        self.context = context
        self.news = news or NewsService()
        self.season_stats = season_stats
        self.depth_chart = depth_chart
        self.player_search = player_search or SupabaseWaiverPlayerRepository()

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            RANK_AVAILABLE_PLAYERS_TOOL,
            GET_AVAILABLE_PLAYER_TOOL,
            GET_RECENT_WAIVER_NEWS_TOOL,
        ]

    @property
    def handlers(self) -> dict[str, Callable[..., Any]]:
        return {
            "rank_available_players": self.rank_available_players,
            "get_available_player": self.get_available_player,
            "get_recent_news": self.get_recent_news,
        }

    def rank_available_players(
        self,
        position: str | None,
        team: str | None,
        sort_by: str,
        limit: int,
    ) -> dict[str, Any]:
        if position is not None and position not in WAIVER_POSITIONS:
            raise ValueError(f"Unsupported position: {position}")
        if team is not None and team not in CURRENT_TEAM_CODES:
            raise ValueError(f"Unsupported team: {team}")
        if sort_by not in WAIVER_SORTS:
            raise ValueError(f"Unsupported sort_by: {sort_by}")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")

        values = [
            item
            for item in self.context.available_players
            if (position is None or item.position == position)
            and (team is None or item.team == team)
        ]
        if team is not None and position is not None:
            known = {item.sleeper_player_id for item in values}
            rostered = self._rostered_sleeper_ids()
            for profile in self.player_search.team_position_players(
                team=team,
                position=position,
                season=self.context.league.season,
            ):
                sleeper_id = str(profile.get("sleeper_player_id") or "")
                if not sleeper_id or sleeper_id in known or sleeper_id in rostered:
                    continue
                values.append(self._profile_candidate(profile))
                known.add(sleeper_id)
        if sort_by == "fantasycalc_value":
            values.sort(
                key=lambda item: (
                    item.fantasycalc_value is None,
                    -(item.fantasycalc_value or 0),
                    item.display_name,
                )
            )
        elif sort_by == "fantasycalc_trend_30_day":
            values.sort(
                key=lambda item: (
                    item.fantasycalc_trend_30_day is None,
                    -(item.fantasycalc_trend_30_day or 0),
                    -(item.fantasycalc_value or 0),
                    item.display_name,
                )
            )
        else:
            values.sort(
                key=lambda item: (
                    item.ecr is None,
                    item.ecr if item.ecr is not None else float("inf"),
                    -(item.fantasycalc_value or 0),
                    item.display_name,
                )
            )
        selected = values[:limit]
        return {
            "filters": {"position": position, "team": team},
            "sort_by": sort_by,
            "returned": len(selected),
            "players": [item.agent_view() for item in selected],
        }

    def _rostered_sleeper_ids(self) -> set[str]:
        return {
            player.sleeper_player_id
            for roster in (
                self.context.league.managed_roster,
                *self.context.league.other_rosters,
            )
            for player in roster.all_players
        }

    @staticmethod
    def _profile_candidate(profile: dict[str, Any]) -> WaiverCandidate:
        return WaiverCandidate(
            sleeper_player_id=str(profile["sleeper_player_id"]),
            player_id=(
                str(profile["player_id"]) if profile.get("player_id") else None
            ),
            display_name=str(profile.get("display_name") or "Unknown player"),
            position=str(profile.get("position") or "").upper(),
            team=(
                {"AZ": "ARI", "LAR": "LA"}.get(
                    str(profile["team"]).upper(),
                    str(profile["team"]).upper(),
                )
                if profile.get("team")
                else None
            ),
            roster_status=profile.get("roster_status"),
            fantasycalc_value=None,
            fantasycalc_overall_rank=None,
            fantasycalc_position_rank=None,
            fantasycalc_trend_30_day=None,
            roster_percent=None,
            trade_frequency=None,
            ecr=None,
            ecr_position_rank=None,
        )

    def _resolve_available(
        self,
        player_ref: str,
    ) -> tuple[WaiverCandidate | None, list[WaiverCandidate]]:
        query = player_ref.strip()
        if not query:
            raise ValueError("player_ref cannot be blank")
        key = _name_key(query)
        exact = [
            item
            for item in self.context.available_players
            if _name_key(item.display_name) == key
        ]
        if len(exact) == 1:
            return exact[0], []
        partial = [
            item
            for item in self.context.available_players
            if key in _name_key(item.display_name)
        ]
        if len(partial) == 1:
            return partial[0], []
        if exact or partial:
            return None, (exact or partial)[:5]

        rostered = self._rostered_sleeper_ids()
        deep = []
        for profile in self.player_search.find_players(query):
            sleeper_id = str(profile.get("sleeper_player_id") or "")
            if not sleeper_id or sleeper_id in rostered:
                continue
            deep.append(self._profile_candidate(profile))
        if len(deep) == 1:
            return deep[0], []
        return None, deep[:5]

    def get_available_player(self, player_ref: str) -> dict[str, Any]:
        player, candidates = self._resolve_available(player_ref)
        if player is None:
            return {
                "status": "ambiguous" if candidates else "not_available",
                "player_ref": player_ref,
                "candidates": [item.agent_view() for item in candidates],
                "retry": (
                    "Retry with one candidate's exact name."
                    if candidates
                    else "Check another name or search a ranked available-player slice."
                ),
            }

        prior_stats: dict[str, Any] | None = None
        depth_context: list[dict[str, Any]] = []
        errors = []
        if player.player_id is not None:
            try:
                prior_stats = self.season_stats(
                    player.player_id,
                    self.context.league.season - 1,
                    "REG",
                    None,
                )
            except Exception as error:
                errors.append(f"Prior-season stats unavailable: {error}")
        if player.team and player.position in {"QB", "RB", "WR", "TE"}:
            try:
                rows = self.depth_chart(
                    player.team,
                    self.context.league.season,
                    None,
                    player.position,
                )
                depth_context = [
                    row
                    for row in rows
                    if (
                        player.player_id is not None
                        and str(row.get("player_id")) == player.player_id
                    )
                    or _name_key(str(row.get("player_name") or ""))
                    == _name_key(player.display_name)
                ][:5]
            except Exception as error:
                errors.append(f"Depth chart unavailable: {error}")
        return {
            "status": "available",
            "player": player.agent_view(),
            "prior_season": self.context.league.season - 1,
            "prior_season_stats": prior_stats,
            "current_depth_chart_entries": depth_context,
            "warnings": errors,
        }

    def get_recent_news(
        self,
        player_ref: str | None,
        team: str | None,
        limit: int,
    ) -> dict[str, Any]:
        if player_ref is None and team is None:
            raise ValueError("Provide player_ref, team, or both")
        result = self.news.latest(
            NewsQuery(
                count=limit,
                detail=NewsDetail.SUMMARY,
                player=player_ref,
                team=team,
                previews=False,
            )
        )
        return {
            "status": result.outcome.value,
            "requested_player": player_ref,
            "requested_team": team,
            "resolved_player": (
                result.resolved_player.display_name
                if result.resolved_player is not None
                else None
            ),
            "resolved_team": (
                result.resolved_team.code
                if result.resolved_team is not None
                else None
            ),
            "resolution_note": result.resolution_note,
            "player_candidates": [
                {
                    "name": item.display_name,
                    "position": item.position,
                    "team": item.team,
                }
                for item in result.player_candidates
            ],
            "reports": [
                {
                    "title": report.title,
                    "source": report.source,
                    "published_at": report.published_at,
                    "source_url": report.source_url,
                    "players": list(report.players),
                    "teams": list(report.teams),
                    "summary": _bounded_body(report.body),
                }
                for report in result.reports
            ],
            "freshness_note": (
                "Newest maintained reports are returned without imposing a hidden date cutoff."
                if result.outcome == NewsOutcome.SUCCESS
                else "No maintained report was found for this resolved scope."
            ),
        }
