"""Read-only adapters for Sleeper's public draft and league endpoints."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from drafting.models import DraftPick


SLEEPER_API_BASE_URL = "https://api.sleeper.app/v1"


class SleeperApiError(RuntimeError):
    pass


class SleeperApiClient:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        base_url: str = SLEEPER_API_BASE_URL,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self.base_url = base_url.rstrip("/")

    def _get(
        self,
        path: str,
        *,
        fresh: bool = False,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            # Sleeper's CDN advertises a long shared-cache lifetime with stale
            # revalidation. During a live draft that can make the first read
            # show the previous pick and only refresh the second read. A unique
            # ignored query value bypasses that shared cache without changing
            # the API resource.
            query_params = dict(params or {})
            if fresh:
                query_params["magiff_refresh"] = time.time_ns()
            response = self._client.get(
                f"{self.base_url}{path}",
                params=query_params or None,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise SleeperApiError(f"Sleeper request failed for {path}: {error}") from error


def _required_reference(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _positive_int(value: object, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


class SleeperDraftClient(SleeperApiClient):
    def get_draft(self, draft_id: str) -> dict[str, Any]:
        normalized = _required_reference(draft_id, "draft_id")
        payload = self._get(f"/draft/{normalized}", fresh=True)
        if not isinstance(payload, dict):
            raise SleeperApiError("Sleeper returned an invalid draft payload")
        return payload

    def get_draft_picks(self, draft_id: str) -> list[dict[str, Any]]:
        normalized = _required_reference(draft_id, "draft_id")
        payload = self._get(f"/draft/{normalized}/picks", fresh=True)
        if not isinstance(payload, list):
            raise SleeperApiError("Sleeper returned an invalid draft-picks payload")
        return [item for item in payload if isinstance(item, dict)]

    def snapshot(
        self,
        *,
        draft_id: str,
        user_id: str | None = None,
        draft_slot: int | None = None,
    ) -> tuple[dict[str, Any], list[DraftPick], int, int | None]:
        draft = self.get_draft(draft_id)
        raw_picks = self.get_draft_picks(draft_id)
        order = draft.get("draft_order") or {}
        slot_to_roster = draft.get("slot_to_roster_id") or {}
        if draft_slot is None:
            if not user_id:
                raise ValueError("Provide user_id or draft_slot")
            raw_slot = order.get(str(user_id))
            if raw_slot is None:
                raise ValueError("The user is not present in this draft order")
            draft_slot = int(raw_slot)
        roster_value = slot_to_roster.get(str(draft_slot))
        roster_id = int(roster_value) if roster_value is not None else draft_slot
        picks = [_normalize_pick(item) for item in raw_picks]
        return draft, picks, draft_slot, roster_id


class SleeperLeagueClient(SleeperApiClient):
    """Fetch one consistent read-only view of a Sleeper league."""

    def get_user(self, user_reference: str) -> dict[str, Any]:
        normalized = _required_reference(user_reference, "user_reference")
        payload = self._get(f"/user/{normalized}")
        if not isinstance(payload, dict) or not payload.get("user_id"):
            raise SleeperApiError(
                f"Sleeper user {user_reference!r} was not found"
            )
        return payload

    def get_league(self, league_id: str) -> dict[str, Any]:
        normalized = _required_reference(league_id, "league_id")
        payload = self._get(f"/league/{normalized}", fresh=True)
        if not isinstance(payload, dict) or not payload.get("league_id"):
            raise SleeperApiError(f"Sleeper league {league_id!r} was not found")
        return payload

    def get_league_users(self, league_id: str) -> list[dict[str, Any]]:
        normalized = _required_reference(league_id, "league_id")
        return self._get_list(f"/league/{normalized}/users")

    def get_league_rosters(self, league_id: str) -> list[dict[str, Any]]:
        normalized = _required_reference(league_id, "league_id")
        return self._get_list(f"/league/{normalized}/rosters", fresh=True)

    def get_matchups(self, league_id: str, week: int) -> list[dict[str, Any]]:
        normalized = _required_reference(league_id, "league_id")
        normalized_week = _positive_int(week, "week")
        return self._get_list(
            f"/league/{normalized}/matchups/{normalized_week}",
            fresh=True,
        )

    def get_transactions(
        self,
        league_id: str,
        week: int,
    ) -> list[dict[str, Any]]:
        normalized = _required_reference(league_id, "league_id")
        normalized_week = _positive_int(week, "week")
        return self._get_list(
            f"/league/{normalized}/transactions/{normalized_week}",
            fresh=True,
        )

    def get_nfl_state(self) -> dict[str, Any]:
        payload = self._get("/state/nfl", fresh=True)
        if not isinstance(payload, dict):
            raise SleeperApiError("Sleeper returned an invalid NFL-state payload")
        return payload

    def get_trending_players(
        self,
        trend_type: str,
        *,
        lookback_hours: int = 24,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        if trend_type not in {"add", "drop"}:
            raise ValueError("trend_type must be add or drop")
        normalized_hours = _positive_int(lookback_hours, "lookback_hours")
        normalized_limit = _positive_int(limit, "limit")
        if normalized_limit > 100:
            raise ValueError("limit must not exceed 100")
        payload = self._get(
            f"/players/nfl/trending/{trend_type}",
            fresh=True,
            params={
                "lookback_hours": normalized_hours,
                "limit": normalized_limit,
            },
        )
        if not isinstance(payload, list):
            raise SleeperApiError("Sleeper returned an invalid trending payload")
        return [item for item in payload if isinstance(item, dict)]

    def snapshot(
        self,
        *,
        league_id: str,
        user_reference: str,
        week: int | None = None,
        trending_lookback_hours: int = 24,
        trending_limit: int = 25,
    ) -> dict[str, Any]:
        """Fetch independent league resources concurrently in two bounded waves."""
        normalized_league = _required_reference(league_id, "league_id")
        normalized_user = _required_reference(user_reference, "user_reference")

        with ThreadPoolExecutor(max_workers=5) as executor:
            user_future = executor.submit(self.get_user, normalized_user)
            league_future = executor.submit(self.get_league, normalized_league)
            users_future = executor.submit(
                self.get_league_users,
                normalized_league,
            )
            rosters_future = executor.submit(
                self.get_league_rosters,
                normalized_league,
            )
            state_future = executor.submit(self.get_nfl_state)
            user = user_future.result()
            league = league_future.result()
            users = users_future.result()
            rosters = rosters_future.result()
            nfl_state = state_future.result()

        selected_week = _positive_int(
            week if week is not None else nfl_state.get("week") or 1,
            "week",
        )
        with ThreadPoolExecutor(max_workers=4) as executor:
            matchups_future = executor.submit(
                self.get_matchups,
                normalized_league,
                selected_week,
            )
            transactions_future = executor.submit(
                self.get_transactions,
                normalized_league,
                selected_week,
            )
            adds_future = executor.submit(
                self.get_trending_players,
                "add",
                lookback_hours=trending_lookback_hours,
                limit=trending_limit,
            )
            drops_future = executor.submit(
                self.get_trending_players,
                "drop",
                lookback_hours=trending_lookback_hours,
                limit=trending_limit,
            )
            matchups = matchups_future.result()
            transactions = transactions_future.result()
            trending_adds = adds_future.result()
            trending_drops = drops_future.result()

        return {
            "user": user,
            "league": league,
            "users": users,
            "rosters": rosters,
            "nfl_state": nfl_state,
            "week": selected_week,
            "matchups": matchups,
            "transactions": transactions,
            "trending_adds": trending_adds,
            "trending_drops": trending_drops,
        }

    def _get_list(self, path: str, *, fresh: bool = False) -> list[dict[str, Any]]:
        payload = self._get(path, fresh=fresh)
        if not isinstance(payload, list):
            raise SleeperApiError(f"Sleeper returned an invalid list for {path}")
        return [item for item in payload if isinstance(item, dict)]


def _normalize_pick(payload: dict[str, Any]) -> DraftPick:
    metadata = payload.get("metadata") or {}
    first = str(metadata.get("first_name") or "").strip()
    last = str(metadata.get("last_name") or "").strip()
    display_name = " ".join(value for value in (first, last) if value) or None
    draft_slot = int(payload["draft_slot"])
    # Sleeper's standalone mock draftboards leave roster_id null. In that
    # format, the claimed draft column is the only stable roster owner.
    roster_value = payload.get("roster_id")
    return DraftPick(
        pick_no=int(payload["pick_no"]),
        round=int(payload["round"]),
        draft_slot=draft_slot,
        roster_id=int(roster_value) if roster_value is not None else draft_slot,
        external_player_id=str(payload["player_id"]),
        display_name=display_name,
        position=metadata.get("position"),
        team=metadata.get("team"),
    )
