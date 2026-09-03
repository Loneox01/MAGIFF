"""Typed state and structured outputs for the waiver advisor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from league_management.models import LeagueContext


class WaiverModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WaiverAction(StrEnum):
    SUBMIT_CLAIM = "submit_claim"
    ADD_FREE_AGENT = "add_free_agent"
    WATCH = "watch"
    PASS = "pass"
    NO_ACTION = "no_action"


class CandidateRole(StrEnum):
    IMMEDIATE_STARTER = "immediate_starter"
    SHORT_TERM_STARTER = "short_term_starter"
    STREAMER = "streamer"
    DEPTH_PIECE = "depth_piece"
    HANDCUFF = "handcuff"
    UPSIDE_STASH = "upside_stash"
    INJURY_STASH = "injury_stash"
    SPECULATIVE_ADD = "speculative_add"
    NOT_APPLICABLE = "not_applicable"


class TimeHorizon(StrEnum):
    THIS_WEEK = "this_week"
    SHORT_TERM = "short_term"
    REST_OF_SEASON = "rest_of_season"
    LONG_TERM = "long_term"
    NOT_APPLICABLE = "not_applicable"


class TeamNeed(StrEnum):
    IMMEDIATE_STARTER = "immediate_starter"
    INJURY_REPLACEMENT = "injury_replacement"
    BYE_COVERAGE = "bye_coverage"
    POSITIONAL_DEPTH = "positional_depth"
    UPSIDE = "upside"
    HANDCUFF_PROTECTION = "handcuff_protection"
    ROSTER_FLEXIBILITY = "roster_flexibility"
    NO_CLEAR_NEED = "no_clear_need"


class RecommendationPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_APPLICABLE = "not_applicable"


class PreliminaryWaiverMove(WaiverModel):
    add_player: str
    drop_player: str | None
    candidate_role: CandidateRole
    time_horizon: TimeHorizon
    rationale: str


class PreliminaryWaiverAnalysis(WaiverModel):
    team_needs: list[TeamNeed]
    shortlist: list[PreliminaryWaiverMove] = Field(max_length=5)
    preliminary_strategy: str
    no_action_is_plausible: bool


class WaiverRecommendation(WaiverModel):
    action: WaiverAction
    add_player: str | None
    drop_player: str | None
    candidate_role: CandidateRole
    priority: RecommendationPriority
    time_horizon: TimeHorizon
    faab_bid: int | None = Field(default=None, ge=0)
    immediate_lineup_impact: str
    long_term_value: str
    add_over_drop: str
    evidence_summary: str
    risks: list[str]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_action_players(self) -> "WaiverRecommendation":
        transactional = self.action in {
            WaiverAction.SUBMIT_CLAIM,
            WaiverAction.ADD_FREE_AGENT,
        }
        if transactional and not self.add_player:
            raise ValueError("A claim or add requires add_player")
        if self.action == WaiverAction.NO_ACTION and self.add_player is not None:
            raise ValueError("no_action cannot include add_player")
        return self


class WaiverAnalysis(WaiverModel):
    team_needs: list[TeamNeed]
    recommendations: list[WaiverRecommendation] = Field(max_length=5)
    overall_strategy: str
    no_action_reason: str | None


@dataclass(frozen=True)
class WaiverCandidate:
    sleeper_player_id: str
    player_id: str | None
    display_name: str
    position: str
    team: str | None
    roster_status: str | None
    fantasycalc_value: int | None
    fantasycalc_overall_rank: int | None
    fantasycalc_position_rank: int | None
    fantasycalc_trend_30_day: int | None
    roster_percent: float | None
    trade_frequency: float | None
    ecr: float | None = None
    ecr_position_rank: int | None = None
    projection_week: int | None = None
    projected_points: float | None = None
    projection_opponent: str | None = None
    projection_game_date: str | None = None
    projection_updated_at: int | None = None
    projection_source: str | None = None

    def agent_view(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "position": self.position,
            "team": self.team,
            "roster_status": self.roster_status,
            "fantasycalc": {
                "value": self.fantasycalc_value,
                "overall_rank": self.fantasycalc_overall_rank,
                "position_rank": self.fantasycalc_position_rank,
                "trend_30_day": self.fantasycalc_trend_30_day,
                "roster_percent": self.roster_percent,
                "trade_frequency": self.trade_frequency,
            },
            "ecr": self.ecr,
            "ecr_position_rank": self.ecr_position_rank,
            "weekly_projection": {
                "week": self.projection_week,
                "points": self.projected_points,
                "opponent": self.projection_opponent,
                "game_date": self.projection_game_date,
                "updated_at": self.projection_updated_at,
                "source": self.projection_source,
            },
        }


@dataclass(frozen=True)
class WaiverContext:
    league: LeagueContext
    available_players: tuple[WaiverCandidate, ...]
    managed_players: tuple[WaiverCandidate, ...]
    top_default_count: int
    market_error: str | None = None
    projection_error: str | None = None

    def agent_payload(self) -> dict[str, Any]:
        league_payload = self.league.agent_payload()
        rostered_ids = {
            player.sleeper_player_id
            for roster in (
                self.league.managed_roster,
                *self.league.other_rosters,
            )
            for player in roster.all_players
        }
        available_trending_adds = [
            item.agent_view()
            for item in self.league.trending_adds
            if item.player.sleeper_player_id not in rostered_ids
        ][:10]
        available_trending_drops = [
            item.agent_view()
            for item in self.league.trending_drops
            if item.player.sleeper_player_id not in rostered_ids
        ][:5]
        limitations = [
            note
            for note in league_payload["limitations"]
            if not note.startswith("Available candidates are the highest current ECR")
            and not note.startswith("Sleeper projections are not exposed")
        ]
        receptions = float(self.league.scoring_settings.get("rec", 0))
        scoring_format = (
            "ppr"
            if receptions >= 0.75
            else ("half_ppr" if receptions >= 0.25 else "standard")
        )
        top_available_by_position = {}
        for position, count in (("QB", 2), ("RB", 2), ("WR", 2), ("TE", 2), ("K", 1)):
            values = sorted(
                (
                    candidate
                    for candidate in self.available_players
                    if candidate.position == position
                    and candidate.projected_points is not None
                ),
                key=lambda candidate: (
                    -(candidate.projected_points or 0),
                    candidate.display_name,
                ),
            )[:count]
            top_available_by_position[position] = [
                candidate.agent_view() for candidate in values
            ]
        current_defenses = [
            candidate.agent_view()
            for candidate in self.managed_players
            if candidate.position == "DEF"
        ]
        available_defenses = sorted(
            (
                candidate
                for candidate in self.available_players
                if candidate.position == "DEF"
                and candidate.projected_points is not None
            ),
            key=lambda candidate: (
                -(candidate.projected_points or 0),
                candidate.display_name,
            ),
        )[:3]
        return {
            "league": {
                "league_id": self.league.league_id,
                "name": self.league.league_name,
                "season": self.league.season,
                "status": self.league.status,
                "current_week": self.league.current_week,
                "season_type": self.league.season_type,
                "total_rosters": self.league.total_rosters,
                "roster_positions": list(self.league.roster_positions),
                "scoring_format": scoring_format,
                "points_per_reception": self.league.scoring_settings.get("rec", 0),
                "passing_touchdown_points": self.league.scoring_settings.get(
                    "pass_td", 0
                ),
                "waiver_settings": self.league.waiver_settings,
            },
            "managed_team": league_payload["managed_team"],
            "matchup": league_payload["matchup"],
            "current_week_transactions": league_payload[
                "current_week_transactions"
            ][:10],
            "available_trending_adds": available_trending_adds,
            "available_trending_drops": available_trending_drops,
            "top_available_candidates": [
                candidate.agent_view()
                for candidate in self.available_players[: self.top_default_count]
            ],
            "managed_roster_market": [
                candidate.agent_view() for candidate in self.managed_players
            ],
            "weekly_projections": {
                "week": self.league.current_week,
                "source": "Sleeper projection feed",
                "scoring": "calculated from projected stats using this league's Sleeper scoring settings",
                "error": self.projection_error,
                "top_available_by_position": top_available_by_position,
                "defense_streaming": {
                    "current": current_defenses,
                    "top_available": [
                        candidate.agent_view() for candidate in available_defenses
                    ],
                },
            },
            "market": {
                "source": "FantasyCalc",
                "scope": "current redraft values configured to league size, QB format, and PPR scoring",
                "default_sort": (
                    "ecr" if self.market_error else "fantasycalc_value"
                ),
                "error": self.market_error,
            },
            "ecr": league_payload["ecr"],
            "limitations": [
                *limitations,
                "The default packet exposes only top market candidates; use the waiver discovery tools for filtered or named searches.",
                "FantasyCalc is a market signal, not a projection or real-time news source.",
                "Sleeper weekly projections are estimates rather than guaranteed outcomes.",
            ],
        }
