"""Deterministic construction of a read-only in-season league snapshot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import zip_longest
from typing import Any, Protocol

from drafting.board import DraftBoardRepository, SupabaseDraftBoardRepository
from drafting.models import DraftCandidate
from integrations.sleeper import SleeperLeagueClient
from repositories.league_supabase import SupabaseLeaguePlayerRepository

from .models import (
    AvailableCandidate,
    LeagueContext,
    LeaguePlayer,
    LeagueRoster,
    LeagueTransaction,
    LineupAssignment,
    ManagedMatchup,
    TransactionChange,
    TrendingPlayer,
)


WAIVER_SETTING_FIELDS = (
    "waiver_budget",
    "waiver_type",
    "waiver_clear_days",
    "waiver_day_of_week",
    "daily_waivers",
    "daily_waivers_hour",
    "waiver_bid_min",
    "faab_suggestions",
)
TRADE_SETTING_FIELDS = (
    "disable_trades",
    "trade_deadline",
    "trade_review_days",
    "pick_trading",
)
NON_STARTER_SLOTS = {"BN", "IR", "TAXI"}


class LeaguePlayerRepository(Protocol):
    def resolve_players(
        self,
        sleeper_player_ids: list[str],
    ) -> dict[str, dict]: ...


class LeagueSnapshotSource(Protocol):
    def snapshot(
        self,
        *,
        league_id: str,
        user_reference: str,
        week: int | None = None,
        trending_lookback_hours: int = 24,
        trending_limit: int = 25,
    ) -> dict[str, Any]: ...


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _points(settings: dict[str, Any]) -> float:
    return _float(settings.get("fpts")) + _float(settings.get("fpts_decimal")) / 100


def _timestamp(value: object) -> str | None:
    milliseconds = _optional_int(value)
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(
        milliseconds / 1000,
        tz=timezone.utc,
    ).isoformat()


def _scoring_format(scoring_settings: dict[str, Any]) -> str:
    receptions = _float(scoring_settings.get("rec"))
    if receptions >= 0.75:
        return "ppr"
    if receptions >= 0.25:
        return "half_ppr"
    return "standard"


def _league_format(roster_positions: list[str]) -> str:
    normalized = [str(slot).upper() for slot in roster_positions]
    return (
        "redraft_superflex"
        if "SUPER_FLEX" in normalized or normalized.count("QB") > 1
        else "redraft_1qb"
    )


def _external_ids(snapshot: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for roster in snapshot.get("rosters") or []:
        for field in ("players", "starters", "reserve", "taxi"):
            values.extend(str(item) for item in roster.get(field) or [])
    for matchup in snapshot.get("matchups") or []:
        for field in ("players", "starters"):
            values.extend(str(item) for item in matchup.get(field) or [])
    for transaction in snapshot.get("transactions") or []:
        for field in ("adds", "drops"):
            values.extend(str(item) for item in (transaction.get(field) or {}))
    for field in ("trending_adds", "trending_drops"):
        values.extend(
            str(item.get("player_id"))
            for item in snapshot.get(field) or []
            if item.get("player_id") is not None
        )
    return [value for value in dict.fromkeys(values) if value and value != "0"]


def _player(
    sleeper_player_id: str,
    profiles: dict[str, dict],
) -> LeaguePlayer | None:
    normalized = str(sleeper_player_id)
    if not normalized or normalized == "0":
        return None
    profile = profiles.get(normalized)
    if profile is None:
        return LeaguePlayer(
            sleeper_player_id=normalized,
            player_id=None,
            display_name=f"Unmapped Sleeper player {normalized}",
            position=None,
            team=None,
        )
    return LeaguePlayer(
        sleeper_player_id=normalized,
        player_id=profile.get("player_id"),
        display_name=str(profile.get("display_name") or normalized),
        position=profile.get("position"),
        team=profile.get("team"),
        roster_status=profile.get("roster_status"),
    )


def _players(
    values: list[object] | None,
    profiles: dict[str, dict],
) -> tuple[LeaguePlayer, ...]:
    resolved = []
    for value in values or []:
        player = _player(str(value), profiles)
        if player is not None:
            resolved.append(player)
    return tuple(resolved)


def _normalize_roster(
    raw: dict[str, Any],
    *,
    owner_names: dict[str, str],
    roster_positions: list[str],
    profiles: dict[str, dict],
) -> LeagueRoster:
    starter_slots = [
        str(slot).upper()
        for slot in roster_positions
        if str(slot).upper() not in NON_STARTER_SLOTS
    ]
    starter_ids = [str(value) for value in raw.get("starters") or []]
    starters = []
    for slot, player_id in zip_longest(
        starter_slots,
        starter_ids,
        fillvalue=None,
    ):
        if slot is None:
            slot = "STARTER"
        starters.append(
            LineupAssignment(
                slot=str(slot),
                player=(
                    _player(player_id, profiles)
                    if player_id is not None
                    else None
                ),
            )
        )

    starter_set = {value for value in starter_ids if value != "0"}
    reserve_values = [str(value) for value in raw.get("reserve") or []]
    taxi_values = [str(value) for value in raw.get("taxi") or []]
    reserve_ids = set(reserve_values)
    taxi_ids = set(taxi_values)
    bench_ids = [
        str(value)
        for value in raw.get("players") or []
        if str(value) not in starter_set | reserve_ids | taxi_ids
    ]
    settings = raw.get("settings") or {}
    owner_id = str(raw.get("owner_id")) if raw.get("owner_id") else None
    return LeagueRoster(
        roster_id=_int(raw.get("roster_id")),
        owner_id=owner_id,
        owner_name=owner_names.get(owner_id or ""),
        starters=tuple(starters),
        bench=_players(bench_ids, profiles),
        reserve=_players(reserve_values, profiles),
        taxi=_players(taxi_values, profiles),
        wins=_int(settings.get("wins")),
        losses=_int(settings.get("losses")),
        ties=_int(settings.get("ties")),
        points_for=_points(settings),
        waiver_position=_optional_int(settings.get("waiver_position")),
        waiver_budget_used=_int(settings.get("waiver_budget_used")),
    )


def _normalize_matchup(
    rows: list[dict[str, Any]],
    *,
    week: int,
    managed_roster: LeagueRoster,
    roster_by_id: dict[int, LeagueRoster],
) -> ManagedMatchup | None:
    managed = next(
        (
            row
            for row in rows
            if _int(row.get("roster_id"), -1) == managed_roster.roster_id
        ),
        None,
    )
    if managed is None:
        return None
    matchup_id = _optional_int(managed.get("matchup_id"))
    opponent = next(
        (
            row
            for row in rows
            if matchup_id is not None
            and _optional_int(row.get("matchup_id")) == matchup_id
            and _int(row.get("roster_id"), -1) != managed_roster.roster_id
        ),
        None,
    )
    opponent_id = _optional_int(opponent.get("roster_id")) if opponent else None
    opponent_roster = roster_by_id.get(opponent_id or -1)
    return ManagedMatchup(
        week=week,
        matchup_id=matchup_id,
        roster_id=managed_roster.roster_id,
        opponent_roster_id=opponent_id,
        opponent_name=opponent_roster.owner_name if opponent_roster else None,
        points=_float(managed.get("points")),
        opponent_points=_float(opponent.get("points")) if opponent else None,
    )


def _normalize_transactions(
    rows: list[dict[str, Any]],
    *,
    week: int,
    profiles: dict[str, dict],
) -> tuple[LeagueTransaction, ...]:
    values = []
    for raw in sorted(
        rows,
        key=lambda row: _int(row.get("created")),
        reverse=True,
    ):
        changes = []
        for action, field in (("add", "adds"), ("drop", "drops")):
            for player_id, roster_id in (raw.get(field) or {}).items():
                player = _player(str(player_id), profiles)
                if player is not None:
                    changes.append(
                        TransactionChange(
                            action=action,
                            roster_id=_int(roster_id),
                            player=player,
                        )
                    )
        settings = raw.get("settings") or {}
        values.append(
            LeagueTransaction(
                transaction_id=str(raw.get("transaction_id") or ""),
                transaction_type=str(raw.get("type") or "unknown"),
                status=str(raw.get("status") or "unknown"),
                week=_int(raw.get("leg"), week),
                created_at=_timestamp(raw.get("created")),
                roster_ids=tuple(
                    _int(value) for value in raw.get("roster_ids") or []
                ),
                changes=tuple(changes),
                waiver_bid=_optional_int(settings.get("waiver_bid")),
                draft_picks=tuple(
                    dict(value)
                    for value in raw.get("draft_picks") or []
                    if isinstance(value, dict)
                ),
            )
        )
    return tuple(values)


def _normalize_trends(
    rows: list[dict[str, Any]],
    *,
    trend_type: str,
    profiles: dict[str, dict],
) -> tuple[TrendingPlayer, ...]:
    values = []
    for raw in rows:
        player = _player(str(raw.get("player_id") or ""), profiles)
        if player is not None:
            values.append(
                TrendingPlayer(
                    trend_type=trend_type,
                    count=_int(raw.get("count")),
                    player=player,
                )
            )
    return tuple(values)


def _available_candidates(
    candidates: list[DraftCandidate],
    *,
    rostered_sleeper_ids: set[str],
    limit: int,
) -> tuple[AvailableCandidate, ...]:
    values = []
    for candidate in candidates:
        if (
            not candidate.external_id
            or candidate.external_id in rostered_sleeper_ids
        ):
            continue
        values.append(
            AvailableCandidate(
                player_id=candidate.player_id,
                sleeper_player_id=candidate.external_id,
                display_name=candidate.display_name,
                position=candidate.position,
                team=candidate.team,
                overall_rank=candidate.overall_rank,
                position_rank=candidate.position_rank,
                best_rank=candidate.best_rank,
                worst_rank=candidate.worst_rank,
                rank_sd=candidate.rank_sd,
            )
        )
        if len(values) == limit:
            break
    return tuple(values)


class LeagueContextBuilder:
    """Combine public league state with MAGIFF's stored identity and ECR data."""

    def __init__(
        self,
        *,
        sleeper: LeagueSnapshotSource | None = None,
        players: LeaguePlayerRepository | None = None,
        market: DraftBoardRepository | None = None,
    ) -> None:
        self.sleeper = sleeper or SleeperLeagueClient()
        self.players = players or SupabaseLeaguePlayerRepository()
        self.market = market or SupabaseDraftBoardRepository()

    def build(
        self,
        *,
        league_id: str,
        user_reference: str,
        week: int | None = None,
        available_limit: int = 40,
        trending_lookback_hours: int = 24,
        trending_limit: int = 25,
        ecr_as_of_date: str | None = None,
        include_market: bool = True,
    ) -> LeagueContext:
        if not 10 <= available_limit <= 100:
            raise ValueError("available_limit must be between 10 and 100")
        snapshot = self.sleeper.snapshot(
            league_id=league_id,
            user_reference=user_reference,
            week=week,
            trending_lookback_hours=trending_lookback_hours,
            trending_limit=trending_limit,
        )
        user = snapshot["user"]
        user_id = str(user["user_id"])
        league = snapshot["league"]
        raw_users = snapshot.get("users") or []
        member = next(
            (item for item in raw_users if str(item.get("user_id")) == user_id),
            None,
        )
        if member is None:
            raise ValueError(
                f"Sleeper user {user_reference!r} is not a member of league {league_id}"
            )

        raw_rosters = snapshot.get("rosters") or []
        managed_rows = [
            roster
            for roster in raw_rosters
            if str(roster.get("owner_id")) == user_id
            or user_id in {str(value) for value in roster.get("co_owners") or []}
        ]
        if len(managed_rows) != 1:
            raise ValueError(
                f"Expected one roster for Sleeper user {user_id}; found "
                f"{len(managed_rows)}"
            )

        season = _int(league.get("season"))
        roster_positions = [
            str(value).upper()
            for value in league.get("roster_positions") or []
        ]
        scoring_settings = dict(league.get("scoring_settings") or {})
        detected_scoring = _scoring_format(scoring_settings)
        ecr_scoring = "ppr"
        ecr_league = _league_format(roster_positions)
        external_ids = _external_ids(snapshot)

        market_error = None
        with ThreadPoolExecutor(max_workers=2 if include_market else 1) as executor:
            profiles_future = executor.submit(
                self.players.resolve_players,
                external_ids,
            )
            market_future = (
                executor.submit(
                    self.market.load_candidates,
                    season=season,
                    scoring_format=ecr_scoring,
                    league_format=ecr_league,
                    snapshot_type="current",
                    as_of_date=ecr_as_of_date,
                )
                if include_market
                else None
            )
            profiles = profiles_future.result()
            try:
                if market_future is None:
                    selected_date, market, source, ranking_page = None, [], None, None
                else:
                    selected_date, market, source, ranking_page = market_future.result()
            except RuntimeError as error:
                selected_date, market, source, ranking_page = None, [], None, None
                market_error = str(error)

        owner_names = {
            str(item.get("user_id")): str(
                item.get("display_name") or item.get("user_id")
            )
            for item in raw_users
            if item.get("user_id") is not None
        }
        rosters = tuple(
            sorted(
                (
                    _normalize_roster(
                        row,
                        owner_names=owner_names,
                        roster_positions=roster_positions,
                        profiles=profiles,
                    )
                    for row in raw_rosters
                ),
                key=lambda value: value.roster_id,
            )
        )
        managed_roster_id = _int(managed_rows[0].get("roster_id"))
        managed_roster = next(
            roster for roster in rosters if roster.roster_id == managed_roster_id
        )
        roster_by_id = {roster.roster_id: roster for roster in rosters}
        current_week = _int(snapshot.get("week"), 1)
        matchup = _normalize_matchup(
            snapshot.get("matchups") or [],
            week=current_week,
            managed_roster=managed_roster,
            roster_by_id=roster_by_id,
        )
        rostered_ids = {
            str(player_id)
            for roster in raw_rosters
            for player_id in roster.get("players") or []
            if str(player_id) != "0"
        }
        unmapped = tuple(
            sorted(value for value in external_ids if value not in profiles)
        )
        settings = dict(league.get("settings") or {})
        notes = [
            "Sleeper's official API is read-only; this snapshot cannot submit "
            "lineups, claims, drops, or trades.",
            "Sleeper projections are not exposed by the documented public API "
            "and are not included.",
            "Available candidates are the highest current ECR players absent "
            "from every league roster, not the complete free-agent pool.",
        ]
        if detected_scoring != ecr_scoring:
            notes.append(
                f"The league scoring profile is {detected_scoring}, while the "
                "stored redraft market reference is PPR ECR."
            )
        if market_error:
            notes.append(f"Current ECR candidates were unavailable: {market_error}")
        if unmapped:
            notes.append(
                f"{len(unmapped)} Sleeper player IDs were not mapped to MAGIFF "
                "identities; placeholders are preserved rather than guessed."
            )

        nfl_state = snapshot.get("nfl_state") or {}
        return LeagueContext(
            league_id=str(league.get("league_id") or league_id),
            league_name=str(league.get("name") or league_id),
            season=season,
            status=str(league.get("status") or "unknown"),
            current_week=current_week,
            season_type=str(
                nfl_state.get("season_type")
                or league.get("season_type")
                or "unknown"
            ),
            season_start_date=(
                str(nfl_state["season_start_date"])
                if nfl_state.get("season_start_date")
                else None
            ),
            managed_user_id=user_id,
            managed_user_name=str(
                member.get("display_name")
                or user.get("display_name")
                or user_reference
            ),
            managed_roster_id=managed_roster_id,
            total_rosters=_int(league.get("total_rosters"), len(rosters)),
            roster_positions=tuple(roster_positions),
            scoring_settings={
                str(key): _float(value)
                for key, value in scoring_settings.items()
                if isinstance(value, (int, float))
            },
            waiver_settings={
                field: _int(settings.get(field)) for field in WAIVER_SETTING_FIELDS
            },
            trade_settings={
                field: _int(settings.get(field)) for field in TRADE_SETTING_FIELDS
            },
            managed_roster=managed_roster,
            other_rosters=tuple(
                roster for roster in rosters if roster.roster_id != managed_roster_id
            ),
            matchup=matchup,
            transactions=_normalize_transactions(
                snapshot.get("transactions") or [],
                week=current_week,
                profiles=profiles,
            ),
            trending_adds=_normalize_trends(
                snapshot.get("trending_adds") or [],
                trend_type="add",
                profiles=profiles,
            ),
            trending_drops=_normalize_trends(
                snapshot.get("trending_drops") or [],
                trend_type="drop",
                profiles=profiles,
            ),
            available_candidates=_available_candidates(
                market,
                rostered_sleeper_ids=rostered_ids,
                limit=available_limit,
            ),
            ecr_snapshot_date=selected_date,
            ecr_scoring_format=ecr_scoring,
            ecr_league_format=ecr_league,
            ecr_source=source,
            ecr_ranking_page=ranking_page,
            unmapped_sleeper_player_ids=unmapped,
            notes=tuple(notes),
        )
