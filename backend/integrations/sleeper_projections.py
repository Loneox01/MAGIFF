"""Read-only adapter for Sleeper's weekly projection payloads.

Sleeper's documented public API does not currently list this endpoint, so the
adapter is intentionally isolated from the stable league client. A projection
failure must never prevent the rest of a league or waiver snapshot from loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import httpx


SLEEPER_PROJECTIONS_BASE_URL = "https://api.sleeper.app/projections"
PROJECTION_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}
SEASON_TYPES = {"pre", "regular", "post"}


class SleeperProjectionError(RuntimeError):
    pass


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SleeperWeeklyProjection:
    sleeper_player_id: str
    display_name: str
    position: str
    team: str | None
    opponent: str | None
    season: int
    week: int
    season_type: str
    game_date: str | None
    game_id: str | None
    updated_at: int | None
    company: str | None
    stats: dict[str, float]
    game_status: str | None = None
    injury_status: str | None = None
    injury_body_part: str | None = None
    injury_notes: str | None = None
    injury_start_date: str | None = None
    news_updated_at: int | None = None

    def projected_points(self, scoring_settings: dict[str, float]) -> float | None:
        """Calculate points using the league's own Sleeper scoring keys."""
        matched = [
            self.stats[key] * float(weight)
            for key, weight in scoring_settings.items()
            if key in self.stats and isinstance(weight, (int, float))
        ]
        if matched:
            return round(sum(matched), 2)

        receptions = float(scoring_settings.get("rec", 0))
        fallback = (
            "pts_ppr"
            if receptions >= 0.75
            else ("pts_half_ppr" if receptions >= 0.25 else "pts_std")
        )
        value = self.stats.get(fallback)
        return None if value is None else round(value, 2)


class SleeperProjectionClient:
    """Fetch and normalize one bounded NFL weekly projection set."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        base_url: str = SLEEPER_PROJECTIONS_BASE_URL,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self.base_url = base_url.rstrip("/")

    def weekly_projections(
        self,
        *,
        season: int,
        week: int,
        season_type: str = "regular",
        positions: Iterable[str] | None = None,
    ) -> list[SleeperWeeklyProjection]:
        if season < 2017:
            raise ValueError("season must be 2017 or later")
        if not 1 <= week <= 22:
            raise ValueError("week must be between 1 and 22")
        if season_type not in SEASON_TYPES:
            raise ValueError("season_type must be pre, regular, or post")

        selected_positions = list(
            dict.fromkeys(str(value).upper() for value in (positions or []))
        )
        invalid = set(selected_positions) - PROJECTION_POSITIONS
        if invalid:
            raise ValueError(
                "Unsupported projection positions: " + ", ".join(sorted(invalid))
            )
        params: list[tuple[str, str]] = [("season_type", season_type)]
        params.extend(("position[]", value) for value in selected_positions)
        path = f"/nfl/{season}/{week}"
        try:
            response = self._client.get(
                f"{self.base_url}{path}",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise SleeperProjectionError(
                f"Sleeper projection request failed for {path}: {error}"
            ) from error
        if not isinstance(payload, list):
            raise SleeperProjectionError("Sleeper returned an invalid projection payload")

        values: list[SleeperWeeklyProjection] = []
        for item in payload:
            if not isinstance(item, dict) or item.get("player_id") is None:
                continue
            player = item.get("player") or {}
            raw_stats = item.get("stats") or {}
            if not isinstance(player, dict) or not isinstance(raw_stats, dict):
                continue
            position = str(player.get("position") or "").upper()
            if not position:
                fantasy_positions = player.get("fantasy_positions") or []
                if fantasy_positions:
                    position = str(fantasy_positions[0]).upper()
            if selected_positions and position not in selected_positions:
                continue
            first = str(player.get("first_name") or "").strip()
            last = str(player.get("last_name") or "").strip()
            display_name = " ".join(value for value in (first, last) if value)
            if position == "DEF" and display_name:
                display_name = f"{display_name} D/ST"
            sleeper_id = str(item["player_id"])
            stats = {
                str(key): number
                for key, value in raw_stats.items()
                if (number := _number(value)) is not None
            }
            updated = item.get("updated_at") or item.get("last_modified")
            news_updated = player.get("news_updated")
            values.append(
                SleeperWeeklyProjection(
                    sleeper_player_id=sleeper_id,
                    display_name=display_name or sleeper_id,
                    position=position,
                    team=(
                        str(item["team"]).upper()
                        if item.get("team") is not None
                        else None
                    ),
                    opponent=(
                        str(item["opponent"]).upper()
                        if item.get("opponent") is not None
                        else None
                    ),
                    season=int(item.get("season") or season),
                    week=int(item.get("week") or week),
                    season_type=str(item.get("season_type") or season_type),
                    game_date=(
                        str(item["date"]) if item.get("date") is not None else None
                    ),
                    game_id=(
                        str(item["game_id"])
                        if item.get("game_id") is not None
                        else None
                    ),
                    updated_at=(int(updated) if updated is not None else None),
                    company=(
                        str(item["company"])
                        if item.get("company") is not None
                        else None
                    ),
                    stats=stats,
                    game_status=(
                        str(item["status"])
                        if item.get("status") is not None
                        else None
                    ),
                    injury_status=(
                        str(player["injury_status"])
                        if player.get("injury_status") is not None
                        else None
                    ),
                    injury_body_part=(
                        str(player["injury_body_part"])
                        if player.get("injury_body_part") is not None
                        else None
                    ),
                    injury_notes=(
                        str(player["injury_notes"])
                        if player.get("injury_notes") is not None
                        else None
                    ),
                    injury_start_date=(
                        str(player["injury_start_date"])
                        if player.get("injury_start_date") is not None
                        else None
                    ),
                    news_updated_at=(
                        int(news_updated) if news_updated is not None else None
                    ),
                )
            )
        return values
