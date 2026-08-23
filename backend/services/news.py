"""Deterministic recent-news service used by non-agent transports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from database.client import get_supabase_client
from repositories import nfl_supabase


DEFAULT_NEWS_COUNT = 5
MAX_NEWS_COUNT = 10
MAX_FULL_NEWS_COUNT = 3
OPTIONAL_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
CURRENT_FRANCHISE_CODES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TEAM_CODE_ALIASES = {"AZ": "ARI"}


class NewsDetail(StrEnum):
    HEADLINES = "headlines"
    SUMMARY = "summary"
    FULL = "full"


class NewsOutcome(StrEnum):
    SUCCESS = "success"
    PLAYER_NOT_FOUND = "player_not_found"
    PLAYER_AMBIGUOUS = "player_ambiguous"
    TEAM_NOT_FOUND = "team_not_found"
    TEAM_AMBIGUOUS = "team_ambiguous"
    NO_REPORTS = "no_reports"


@dataclass(frozen=True)
class NewsQuery:
    count: int = DEFAULT_NEWS_COUNT
    detail: NewsDetail = NewsDetail.SUMMARY
    player: str | None = None
    team: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or not 1 <= self.count <= MAX_NEWS_COUNT
        ):
            raise ValueError(
                f"news count must be between 1 and {MAX_NEWS_COUNT}"
            )
        if not isinstance(self.detail, NewsDetail):
            raise ValueError("news detail must be headlines, summary, or full")
        if self.player is not None and not self.player.strip():
            raise ValueError("player cannot be blank")
        if self.player is not None and len(self.player) > 100:
            raise ValueError("player cannot exceed 100 characters")
        if self.team is not None and not self.team.strip():
            raise ValueError("team cannot be blank")
        if self.team is not None and len(self.team) > 40:
            raise ValueError("team cannot exceed 40 characters")


@dataclass(frozen=True)
class PlayerCandidate:
    player_id: str
    display_name: str
    position: str | None
    team: str | None
    status: str | None


@dataclass(frozen=True)
class TeamCandidate:
    code: str
    name: str
    nickname: str


@dataclass(frozen=True)
class NewsReport:
    report_id: str
    title: str
    source: str
    source_url: str
    author: str | None
    published_at: str
    players: tuple[str, ...]
    teams: tuple[str, ...]
    document_type: str
    storyline: str | None
    content_mode: str
    body: str


@dataclass(frozen=True)
class NewsResult:
    outcome: NewsOutcome
    query: NewsQuery
    reports: tuple[NewsReport, ...] = ()
    resolved_player: PlayerCandidate | None = None
    resolved_team: TeamCandidate | None = None
    player_candidates: tuple[PlayerCandidate, ...] = ()
    team_candidates: tuple[TeamCandidate, ...] = ()
    resolution_note: str | None = None
    full_view_capped: bool = False


class NewsRepository(Protocol):
    def find_players(self, name: str) -> list[PlayerCandidate]: ...

    def list_teams(self) -> list[TeamCandidate]: ...

    def recent_reports(
        self,
        *,
        count: int,
        player_id: str | None,
        team: str | None,
    ) -> list[NewsReport]: ...


class SupabaseNewsRepository:
    """Bounded Supabase reads for the direct news command."""

    def __init__(self, client=None) -> None:
        self.client = client or get_supabase_client()
        self._teams: list[TeamCandidate] | None = None

    def find_players(self, name: str) -> list[PlayerCandidate]:
        return [
            PlayerCandidate(
                player_id=str(row["player_id"]),
                display_name=str(row["display_name"]),
                position=(
                    None if row.get("position") is None else str(row["position"])
                ),
                team=(
                    None
                    if row.get("latest_team") is None
                    else str(row["latest_team"])
                ),
                status=(
                    None if row.get("status") is None else str(row["status"])
                ),
            )
            for row in nfl_supabase.find_players(name)
        ]

    def list_teams(self) -> list[TeamCandidate]:
        if self._teams is None:
            rows = (
                self.client.table("teams")
                .select("team_abbr,team_name,team_nick")
                .limit(100)
                .execute()
                .data
            )
            self._teams = [
                TeamCandidate(
                    code=str(row["team_abbr"]),
                    name=str(row["team_name"]),
                    nickname=str(row["team_nick"]),
                )
                for row in rows
            ]
        return self._teams

    def recent_reports(
        self,
        *,
        count: int,
        player_id: str | None,
        team: str | None,
    ) -> list[NewsReport]:
        rows = (
            self.client.rpc(
                "get_recent_reports",
                {
                    "match_count": count,
                    "filter_player_id": player_id,
                    "filter_team": team,
                },
            )
            .execute()
            .data
            or []
        )
        return [
            NewsReport(
                report_id=str(row["report_id"]),
                title=str(row["title"]),
                source=str(row["source"]),
                source_url=str(row["source_url"]),
                author=None if row.get("author") is None else str(row["author"]),
                published_at=str(row["published_at"]),
                players=tuple(
                    str(value) for value in (row.get("player_names") or [])
                ),
                teams=tuple(str(value) for value in (row.get("teams") or [])),
                document_type=str(row["document_type"]),
                storyline=(
                    None
                    if row.get("storyline") is None
                    else str(row["storyline"])
                ),
                content_mode=str(row["content_mode"]),
                body=str(row["body"]),
            )
            for row in rows
        ]


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _surname(value: str) -> str | None:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    if tokens and tokens[-1] in OPTIONAL_NAME_SUFFIXES:
        tokens.pop()
    return tokens[-1] if len(tokens) >= 2 else None


def _deduplicate_players(values: list[PlayerCandidate]) -> list[PlayerCandidate]:
    return list({value.player_id: value for value in values}.values())


class NewsService:
    def __init__(self, repository: NewsRepository | None = None) -> None:
        self.repository = repository or SupabaseNewsRepository()

    def _resolve_team(
        self, value: str
    ) -> tuple[TeamCandidate | None, tuple[TeamCandidate, ...], bool]:
        teams = self.repository.list_teams()
        by_code = {team.code: team for team in teams}
        aliases: dict[str, list[TeamCandidate]] = {}

        for team in teams:
            city = team.name.removesuffix(team.nickname).strip()
            for alias in (team.code, team.name, team.nickname, city):
                aliases.setdefault(_normalized(alias), []).append(team)

        raw_code = value.strip().upper()
        raw_code = TEAM_CODE_ALIASES.get(raw_code, raw_code)
        raw_code = CURRENT_FRANCHISE_CODES.get(raw_code, raw_code)
        if raw_code in by_code:
            return by_code[raw_code], (), False

        matches_by_code: dict[str, TeamCandidate] = {}
        for team in aliases.get(_normalized(value), []):
            canonical_code = CURRENT_FRANCHISE_CODES.get(team.code, team.code)
            matches_by_code[canonical_code] = by_code.get(canonical_code, team)
        matches = list(matches_by_code.values())
        if len(matches) == 1:
            canonical = CURRENT_FRANCHISE_CODES.get(matches[0].code, matches[0].code)
            return by_code.get(canonical, matches[0]), (), False
        if len(matches) > 1:
            return None, tuple(matches), True

        needle = _normalized(value)
        suggestion_distances: dict[str, tuple[int, TeamCandidate]] = {}
        for team in teams:
            canonical_code = CURRENT_FRANCHISE_CODES.get(team.code, team.code)
            canonical_team = by_code.get(canonical_code, team)
            distance = min(
                _edit_distance(needle, _normalized(team.code)),
                _edit_distance(needle, _normalized(team.name)),
                _edit_distance(needle, _normalized(team.nickname)),
            )
            current = suggestion_distances.get(canonical_code)
            if current is None or distance < current[0]:
                suggestion_distances[canonical_code] = (distance, canonical_team)
        ranked_suggestions = sorted(
            suggestion_distances.values(), key=lambda item: item[0]
        )
        threshold = max(2, len(needle) // 3)
        suggestions = [
            team
            for distance, team in ranked_suggestions
            if distance <= threshold
        ][:3]
        return None, tuple(suggestions), False

    def _resolve_player(
        self,
        value: str,
        team: TeamCandidate | None,
    ) -> tuple[
        PlayerCandidate | None,
        tuple[PlayerCandidate, ...],
        str | None,
    ]:
        candidates = _deduplicate_players(self.repository.find_players(value))
        if not candidates:
            surname = _surname(value)
            if surname:
                candidates = _deduplicate_players(
                    self.repository.find_players(surname)
                )

        if len(candidates) > 1 and team is not None:
            team_matches = [
                candidate for candidate in candidates if candidate.team == team.code
            ]
            if len(team_matches) == 1:
                candidates = team_matches

        if len(candidates) == 1:
            candidate = candidates[0]
            note = None
            if _normalized(candidate.display_name) != _normalized(value):
                note = (
                    f"Interpreted `{value.strip()}` as "
                    f"**{candidate.display_name}**."
                )
            return candidate, (), note
        return None, tuple(candidates[:5]), None

    def latest(self, query: NewsQuery) -> NewsResult:
        resolved_team = None
        if query.team:
            resolved_team, team_candidates, team_ambiguous = self._resolve_team(
                query.team
            )
            if resolved_team is None:
                outcome = (
                    NewsOutcome.TEAM_AMBIGUOUS
                    if team_ambiguous
                    else NewsOutcome.TEAM_NOT_FOUND
                )
                return NewsResult(
                    outcome=outcome,
                    query=query,
                    team_candidates=team_candidates,
                )

        resolved_player = None
        resolution_note = None
        if query.player:
            resolved_player, player_candidates, resolution_note = (
                self._resolve_player(query.player, resolved_team)
            )
            if resolved_player is None:
                return NewsResult(
                    outcome=(
                        NewsOutcome.PLAYER_AMBIGUOUS
                        if player_candidates
                        else NewsOutcome.PLAYER_NOT_FOUND
                    ),
                    query=query,
                    resolved_team=resolved_team,
                    player_candidates=player_candidates,
                )

        effective_count = query.count
        full_view_capped = False
        if query.detail == NewsDetail.FULL and query.count > MAX_FULL_NEWS_COUNT:
            effective_count = MAX_FULL_NEWS_COUNT
            full_view_capped = True

        reports = self.repository.recent_reports(
            count=effective_count,
            player_id=(
                resolved_player.player_id if resolved_player is not None else None
            ),
            team=resolved_team.code if resolved_team is not None else None,
        )
        if not reports:
            return NewsResult(
                outcome=NewsOutcome.NO_REPORTS,
                query=query,
                resolved_player=resolved_player,
                resolved_team=resolved_team,
                resolution_note=resolution_note,
                full_view_capped=full_view_capped,
            )
        return NewsResult(
            outcome=NewsOutcome.SUCCESS,
            query=query,
            reports=tuple(reports),
            resolved_player=resolved_player,
            resolved_team=resolved_team,
            resolution_note=resolution_note,
            full_view_capped=full_view_capped,
        )


def _edit_distance(left: str, right: str) -> int:
    """Small dependency-free Levenshtein distance for team retry suggestions."""
    if not left:
        return len(right)
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
