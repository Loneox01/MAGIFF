"""Typed context and structured outputs for weekly lineup decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from league_management.models import LeagueContext


class LineupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProposedStarter(LineupModel):
    slot_id: str
    sleeper_player_id: str | None
    player_name: str | None

    @model_validator(mode="after")
    def validate_player_pair(self) -> "ProposedStarter":
        if (self.sleeper_player_id is None) != (self.player_name is None):
            raise ValueError(
                "sleeper_player_id and player_name must both be supplied or null"
            )
        return self


class PreliminaryLineupPlan(LineupModel):
    week: int = Field(ge=1, le=22)
    starters: list[ProposedStarter]
    news_check_player_ids: list[str] = Field(max_length=10)
    preliminary_strategy: str


class RecommendedStarter(ProposedStarter):
    rationale: str
    confidence: DecisionConfidence


class LineupCloseCall(LineupModel):
    selected_player_id: str
    alternative_player_id: str
    rationale: str


class LineupAnalysis(LineupModel):
    week: int = Field(ge=1, le=22)
    starters: list[RecommendedStarter]
    close_calls: list[LineupCloseCall] = Field(max_length=5)
    overall_strategy: str
    warnings: list[str]


INJURY_STATUS_CODES = {
    "questionable": "Q",
    "doubtful": "D",
    "out": "O",
    "injured reserve": "IR",
    "ir": "IR",
    "physically unable to perform": "PUP",
    "pup": "PUP",
    "suspended": "SUSP",
    "susp": "SUSP",
    "non-football injury": "NFI",
    "nfi": "NFI",
    "inactive": "INACTIVE",
}
UNAVAILABLE_INJURY_CODES = {"O", "IR", "PUP", "SUSP", "NFI", "INACTIVE"}


def injury_status_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return INJURY_STATUS_CODES.get(normalized.casefold(), normalized.upper())


@dataclass(frozen=True)
class LineupSlot:
    slot_id: str
    slot_type: str
    current_player_id: str | None

    def agent_view(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "slot_type": self.slot_type,
            "current_player_id": self.current_player_id,
        }


@dataclass(frozen=True)
class LineupPlayer:
    sleeper_player_id: str
    player_id: str | None
    display_name: str
    position: str
    team: str | None
    roster_status: str | None
    roster_group: str
    current_slot_id: str | None
    eligible_slot_types: tuple[str, ...]
    projection_week: int
    projected_points: float | None
    opponent: str | None
    game_id: str | None
    game_date: str | None
    kickoff_at: datetime | None
    game_status: str | None
    is_locked: bool
    projection_updated_at: int | None
    projection_source: str | None
    injury_status: str | None
    injury_body_part: str | None
    injury_notes: str | None
    injury_start_date: str | None
    injury_news_updated_at: int | None

    @property
    def injury_code(self) -> str | None:
        return injury_status_code(self.injury_status)

    @property
    def unavailable(self) -> bool:
        return self.injury_code in UNAVAILABLE_INJURY_CODES

    @property
    def can_enter_lineup(self) -> bool:
        return (
            self.roster_group in {"starter", "bench"}
            and not self.unavailable
            and not self.is_locked
        )

    def can_fill(self, slot: "LineupSlot") -> bool:
        """Return whether this player can legally occupy the slot now."""
        if slot.slot_type not in self.eligible_slot_types:
            return False
        if self.is_locked:
            return (
                self.roster_group == "starter"
                and self.current_slot_id == slot.slot_id
            )
        return self.roster_group in {"starter", "bench"} and not self.unavailable

    def agent_view(self) -> dict[str, Any]:
        return {
            "sleeper_player_id": self.sleeper_player_id,
            "name": self.display_name,
            "position": self.position,
            "team": self.team,
            "roster_group": self.roster_group,
            "current_slot_id": self.current_slot_id,
            "eligible_slot_types": list(self.eligible_slot_types),
            "roster_status": self.roster_status,
            "weekly_projection": {
                "week": self.projection_week,
                "points": self.projected_points,
                "opponent": self.opponent,
                "game_id": self.game_id,
                "game_date": self.game_date,
                "kickoff_at": (
                    self.kickoff_at.isoformat() if self.kickoff_at else None
                ),
                "game_status": self.game_status,
                "locked": self.is_locked,
                "updated_at": self.projection_updated_at,
                "source": self.projection_source,
            },
            "health": {
                "designation": self.injury_code,
                "raw_designation": self.injury_status,
                "body_part": self.injury_body_part,
                "notes": self.injury_notes,
                "injury_start_date": self.injury_start_date,
                "news_updated_at": self.injury_news_updated_at,
                "unavailable": self.unavailable,
            },
        }


@dataclass(frozen=True)
class BaselineAssignment:
    slot_id: str
    slot_type: str
    sleeper_player_id: str | None
    player_name: str | None
    projected_points: float | None

    def agent_view(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "slot_type": self.slot_type,
            "sleeper_player_id": self.sleeper_player_id,
            "player_name": self.player_name,
            "projected_points": self.projected_points,
        }


@dataclass(frozen=True)
class LineupContext:
    league: LeagueContext
    week: int
    as_of: datetime
    slots: tuple[LineupSlot, ...]
    players: tuple[LineupPlayer, ...]
    projection_baseline: tuple[BaselineAssignment, ...]
    current_projected_total: float
    baseline_projected_total: float
    opponent_current_projected_total: float | None
    opponent_starters: tuple[dict[str, Any], ...]
    projection_error: str | None = None
    schedule_error: str | None = None

    @property
    def player_by_id(self) -> dict[str, LineupPlayer]:
        return {player.sleeper_player_id: player for player in self.players}

    @property
    def lineup_fully_locked(self) -> bool:
        players = self.player_by_id
        return bool(self.slots) and all(
            slot.current_player_id is not None
            and players.get(slot.current_player_id) is not None
            and players[slot.current_player_id].is_locked
            for slot in self.slots
        )

    def agent_payload(self) -> dict[str, Any]:
        starters = [
            player.agent_view()
            for player in self.players
            if player.roster_group == "starter"
        ]
        bench = [
            player.agent_view()
            for player in self.players
            if player.roster_group == "bench"
        ]
        unavailable = [
            player.agent_view()
            for player in self.players
            if player.roster_group in {"reserve", "taxi"}
        ]
        return {
            "league": {
                "league_id": self.league.league_id,
                "name": self.league.league_name,
                "season": self.league.season,
                "week": self.week,
                "as_of": self.as_of.isoformat(),
                "scoring_settings": self.league.scoring_settings,
                "roster_positions": list(self.league.roster_positions),
            },
            "matchup": {
                "opponent_name": (
                    self.league.matchup.opponent_name
                    if self.league.matchup
                    else None
                ),
                "current_lineup_projected_total": self.current_projected_total,
                "opponent_current_lineup_projected_total": (
                    self.opponent_current_projected_total
                ),
                "opponent_starters": list(self.opponent_starters),
            },
            "starter_slots": [slot.agent_view() for slot in self.slots],
            "current_starters": starters,
            "bench": bench,
            "reserve_and_taxi": unavailable,
            "projection_only_baseline": {
                "projected_total": self.baseline_projected_total,
                "assignments": [
                    assignment.agent_view()
                    for assignment in self.projection_baseline
                ],
                "warning": (
                    "This is a deterministic maximum of available Sleeper point "
                    "projections under legal slots, not the final recommendation."
                ),
            },
            "projection_error": self.projection_error,
            "schedule_error": self.schedule_error,
            "lineup_fully_locked": self.lineup_fully_locked,
            "limitations": [
                "This workflow is read-only and cannot submit lineup changes.",
                "Sleeper projections are estimates, not guarantees of role, health, or outcomes.",
                "Q/D/O and related designations come from Sleeper's projection payload; reports provide additional practice and injury context.",
                "Kickoff locks are derived from the stored NFL schedule and enforced by code.",
                "Never claim that a lineup recommendation was submitted to Sleeper.",
            ],
        }


@dataclass(frozen=True)
class LineupChange:
    slot_id: str
    outgoing_player_id: str | None
    outgoing_player: str | None
    incoming_player_id: str | None
    incoming_player: str | None


@dataclass(frozen=True)
class ValidatedLineup:
    projected_total: float
    changes: tuple[LineupChange, ...]
