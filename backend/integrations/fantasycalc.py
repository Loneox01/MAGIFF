"""Best-effort read-only adapter for FantasyCalc redraft market values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


FANTASYCALC_API_URL = "https://api.fantasycalc.com/values/current"


class FantasyCalcApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class FantasyCalcValue:
    fantasycalc_player_id: str
    sleeper_player_id: str | None
    display_name: str
    position: str
    team: str | None
    value: int
    overall_rank: int
    position_rank: int
    trend_30_day: int
    roster_percent: float | None
    trade_frequency: float | None

    def public_view(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "position": self.position,
            "team": self.team,
            "fantasycalc_value": self.value,
            "fantasycalc_overall_rank": self.overall_rank,
            "fantasycalc_position_rank": self.position_rank,
            "fantasycalc_trend_30_day": self.trend_30_day,
            "roster_percent": self.roster_percent,
            "trade_frequency": self.trade_frequency,
        }


def _required_int(value: object, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise FantasyCalcApiError(
            f"FantasyCalc returned an invalid {field}: {value!r}"
        ) from error


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class FantasyCalcClient:
    """Fetch current market values without coupling callers to provider JSON."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        endpoint: str = FANTASYCALC_API_URL,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self.endpoint = endpoint

    def current_redraft_values(
        self,
        *,
        teams: int,
        quarterback_slots: int,
        ppr: float,
    ) -> list[FantasyCalcValue]:
        if not 2 <= teams <= 32:
            raise ValueError("teams must be between 2 and 32")
        if quarterback_slots not in {1, 2}:
            raise ValueError("quarterback_slots must be 1 or 2")
        if ppr not in {0.0, 0.5, 1.0}:
            raise ValueError("ppr must be 0, 0.5, or 1")

        try:
            response = self._client.get(
                self.endpoint,
                params={
                    "isDynasty": "false",
                    "numQbs": quarterback_slots,
                    "numTeams": teams,
                    "ppr": ppr,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise FantasyCalcApiError(
                f"FantasyCalc request failed: {error}"
            ) from error
        if not isinstance(payload, list):
            raise FantasyCalcApiError("FantasyCalc returned a non-list payload")

        values: list[FantasyCalcValue] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            player = raw.get("player")
            if not isinstance(player, dict):
                continue
            name = str(player.get("name") or "").strip()
            position = str(player.get("position") or "").strip().upper()
            if not name or not position:
                continue
            sleeper_id = str(player.get("sleeperId") or "").strip() or None
            team = str(player.get("maybeTeam") or "").strip().upper() or None
            values.append(
                FantasyCalcValue(
                    fantasycalc_player_id=str(player.get("id") or ""),
                    sleeper_player_id=sleeper_id,
                    display_name=name,
                    position=position,
                    team=team,
                    value=_required_int(raw.get("value"), "value"),
                    overall_rank=_required_int(
                        raw.get("overallRank"), "overallRank"
                    ),
                    position_rank=_required_int(
                        raw.get("positionRank"), "positionRank"
                    ),
                    trend_30_day=_required_int(
                        raw.get("trend30Day") or 0, "trend30Day"
                    ),
                    roster_percent=_optional_float(raw.get("maybeRosterPercent")),
                    trade_frequency=_optional_float(raw.get("maybeTradeFrequency")),
                )
            )
        values.sort(key=lambda item: (item.overall_rank, item.display_name))
        return values
