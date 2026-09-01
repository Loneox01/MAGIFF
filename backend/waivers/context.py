"""Build a compact waiver snapshot over the searchable full market pool."""

from __future__ import annotations

from typing import Protocol

from integrations.fantasycalc import (
    FantasyCalcApiError,
    FantasyCalcClient,
    FantasyCalcValue,
)
from league_management.context import LeagueContextBuilder, LeaguePlayerRepository
from repositories.league_supabase import SupabaseLeaguePlayerRepository

from .models import WaiverCandidate, WaiverContext


TEAM_ALIASES = {"AZ": "ARI", "LAR": "LA"}


class FantasyCalcSource(Protocol):
    def current_redraft_values(
        self,
        *,
        teams: int,
        quarterback_slots: int,
        ppr: float,
    ) -> list[FantasyCalcValue]: ...


def _team(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return TEAM_ALIASES.get(normalized, normalized) or None


def _ppr_value(scoring_settings: dict[str, float]) -> float:
    receptions = float(scoring_settings.get("rec", 0))
    if receptions >= 0.75:
        return 1.0
    if receptions >= 0.25:
        return 0.5
    return 0.0


def _quarterback_slots(roster_positions: tuple[str, ...]) -> int:
    return (
        2
        if "SUPER_FLEX" in roster_positions or roster_positions.count("QB") > 1
        else 1
    )


class WaiverContextBuilder:
    """Join live league availability, ECR, identities, and market values."""

    def __init__(
        self,
        *,
        league_builder: LeagueContextBuilder | None = None,
        players: LeaguePlayerRepository | None = None,
        fantasycalc: FantasyCalcSource | None = None,
    ) -> None:
        self.players = players or SupabaseLeaguePlayerRepository()
        self.league_builder = league_builder or LeagueContextBuilder(
            players=self.players
        )
        self.fantasycalc = fantasycalc or FantasyCalcClient()

    def build(
        self,
        *,
        league_id: str,
        user_reference: str,
        week: int | None = None,
        top_default_count: int = 12,
        trending_lookback_hours: int = 24,
        trending_limit: int = 25,
        ecr_as_of_date: str | None = None,
    ) -> WaiverContext:
        if not 5 <= top_default_count <= 20:
            raise ValueError("top_default_count must be between 5 and 20")
        league = self.league_builder.build(
            league_id=league_id,
            user_reference=user_reference,
            week=week,
            available_limit=100,
            trending_lookback_hours=trending_lookback_hours,
            trending_limit=trending_limit,
            ecr_as_of_date=ecr_as_of_date,
        )
        market_error = None
        try:
            values = self.fantasycalc.current_redraft_values(
                teams=league.total_rosters,
                quarterback_slots=_quarterback_slots(league.roster_positions),
                ppr=_ppr_value(league.scoring_settings),
            )
        except FantasyCalcApiError as error:
            values = []
            market_error = str(error)
        sleeper_ids = [
            item.sleeper_player_id
            for item in values
            if item.sleeper_player_id is not None
        ]
        profiles = self.players.resolve_players(sleeper_ids)
        ecr_by_sleeper = {
            item.sleeper_player_id: item for item in league.available_candidates
        }
        rostered_ids = {
            player.sleeper_player_id
            for roster in (league.managed_roster, *league.other_rosters)
            for player in roster.all_players
        }
        managed_ids = {
            player.sleeper_player_id
            for player in league.managed_roster.all_players
        }

        candidates = []
        for item in values:
            sleeper_id = item.sleeper_player_id
            if sleeper_id is None:
                continue
            profile = profiles.get(sleeper_id, {})
            ecr = ecr_by_sleeper.get(sleeper_id)
            candidates.append(
                WaiverCandidate(
                    sleeper_player_id=sleeper_id,
                    player_id=profile.get("player_id"),
                    display_name=str(
                        profile.get("display_name") or item.display_name
                    ),
                    position=str(
                        profile.get("position") or item.position
                    ).upper(),
                    team=_team(profile.get("team") or item.team),
                    roster_status=profile.get("roster_status"),
                    fantasycalc_value=item.value,
                    fantasycalc_overall_rank=item.overall_rank,
                    fantasycalc_position_rank=item.position_rank,
                    fantasycalc_trend_30_day=item.trend_30_day,
                    roster_percent=item.roster_percent,
                    trade_frequency=item.trade_frequency,
                    ecr=ecr.overall_rank if ecr else None,
                    ecr_position_rank=ecr.position_rank if ecr else None,
                )
            )

        known_ids = {item.sleeper_player_id for item in candidates}
        if market_error:
            for ecr in league.available_candidates:
                if ecr.sleeper_player_id in known_ids:
                    continue
                candidates.append(
                    WaiverCandidate(
                        sleeper_player_id=ecr.sleeper_player_id,
                        player_id=ecr.player_id,
                        display_name=ecr.display_name,
                        position=ecr.position,
                        team=_team(ecr.team),
                        roster_status=None,
                        fantasycalc_value=None,
                        fantasycalc_overall_rank=None,
                        fantasycalc_position_rank=None,
                        fantasycalc_trend_30_day=None,
                        roster_percent=None,
                        trade_frequency=None,
                        ecr=ecr.overall_rank,
                        ecr_position_rank=ecr.position_rank,
                    )
                )
                known_ids.add(ecr.sleeper_player_id)
        for player in league.managed_roster.all_players:
            if player.sleeper_player_id in known_ids:
                continue
            candidates.append(
                WaiverCandidate(
                    sleeper_player_id=player.sleeper_player_id,
                    player_id=player.player_id,
                    display_name=player.display_name,
                    position=str(player.position or "UNKNOWN").upper(),
                    team=_team(player.team),
                    roster_status=player.roster_status,
                    fantasycalc_value=None,
                    fantasycalc_overall_rank=None,
                    fantasycalc_position_rank=None,
                    fantasycalc_trend_30_day=None,
                    roster_percent=None,
                    trade_frequency=None,
                    ecr=None,
                    ecr_position_rank=None,
                )
            )
            known_ids.add(player.sleeper_player_id)

        available = tuple(
            sorted(
                (
                    item
                    for item in candidates
                    if item.sleeper_player_id not in rostered_ids
                ),
                key=lambda item: (
                    item.fantasycalc_value is None,
                    -(item.fantasycalc_value or 0),
                    item.ecr is None,
                    item.ecr if item.ecr is not None else float("inf"),
                    item.display_name,
                ),
            )
        )
        managed = tuple(
            sorted(
                (
                    item
                    for item in candidates
                    if item.sleeper_player_id in managed_ids
                ),
                key=lambda item: (
                    item.position,
                    item.fantasycalc_value is None,
                    -(item.fantasycalc_value or 0),
                    item.display_name,
                ),
            )
        )
        return WaiverContext(
            league=league,
            available_players=available,
            managed_players=managed,
            top_default_count=top_default_count,
            market_error=market_error,
        )
