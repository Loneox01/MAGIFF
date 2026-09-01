"""Typed, provider-neutral state for the dedicated draft advisor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DraftCandidate:
    player_id: str
    external_id: str | None
    display_name: str
    position: str
    team: str | None
    overall_rank: float
    position_rank: int
    best_rank: float | None = None
    worst_rank: float | None = None
    rank_sd: float | None = None
    rank_delta: float | None = None

    def agent_view(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "position": self.position,
            "team": self.team,
            "ecr": self.overall_rank,
            "position_rank": self.position_rank,
            "best_rank": self.best_rank,
            "worst_rank": self.worst_rank,
            "rank_sd": self.rank_sd,
            "rank_delta": self.rank_delta,
        }


@dataclass(frozen=True)
class DraftPick:
    pick_no: int
    round: int
    draft_slot: int
    roster_id: int | None
    external_player_id: str
    display_name: str | None = None
    position: str | None = None
    team: str | None = None


@dataclass(frozen=True)
class DraftContext:
    season: int
    scoring_format: str
    league_format: str
    teams: int
    rounds: int
    draft_slot: int
    roster_id: int | None
    draft_status: str
    current_pick: int
    current_round: int
    on_clock: bool
    next_pick: int | None
    picks_until_turn: int | None
    following_pick: int | None
    picks_between_turns: int | None
    roster_requirements: dict[str, int]
    roster_counts: dict[str, int]
    open_starter_slots: dict[str, int]
    my_roster: tuple[DraftPick, ...]
    available_candidates: tuple[DraftCandidate, ...]
    ecr_snapshot_date: str
    ecr_source: str | None
    ecr_ranking_page: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def agent_payload(self) -> dict[str, Any]:
        """Return a compact model-facing snapshot without provider IDs."""
        return {
            "season": self.season,
            "scoring_format": self.scoring_format,
            "league_format": self.league_format,
            "league": {"teams": self.teams, "rounds": self.rounds},
            "turn": {
                "draft_slot": self.draft_slot,
                "current_pick": self.current_pick,
                "current_round": self.current_round,
                "on_clock": self.on_clock,
                "next_pick": self.next_pick,
                "picks_until_turn": self.picks_until_turn,
                "following_pick": self.following_pick,
                "picks_between_turns": self.picks_between_turns,
            },
            "roster_requirements": self.roster_requirements,
            "roster_counts": self.roster_counts,
            "open_starter_slots": self.open_starter_slots,
            "my_roster": [
                {
                    "name": pick.display_name,
                    "position": pick.position,
                    "team": pick.team,
                    "round": pick.round,
                    "pick": pick.pick_no,
                }
                for pick in self.my_roster
            ],
            "available_candidates": [
                candidate.agent_view()
                for candidate in self.available_candidates
            ],
            "ecr": {
                "snapshot_date": self.ecr_snapshot_date,
                "source": self.ecr_source,
                "ranking_page": self.ecr_ranking_page,
            },
            "notes": list(self.notes),
        }

    def debug_payload(self) -> dict[str, Any]:
        return asdict(self)
