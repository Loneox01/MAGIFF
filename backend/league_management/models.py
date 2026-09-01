"""Provider-neutral, immutable state for in-season fantasy management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LeaguePlayer:
    sleeper_player_id: str
    player_id: str | None
    display_name: str
    position: str | None
    team: str | None
    roster_status: str | None = None

    def agent_view(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "position": self.position,
            "team": self.team,
            "roster_status": self.roster_status,
        }


@dataclass(frozen=True)
class LineupAssignment:
    slot: str
    player: LeaguePlayer | None

    def agent_view(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "player": self.player.agent_view() if self.player else None,
        }


@dataclass(frozen=True)
class LeagueRoster:
    roster_id: int
    owner_id: str | None
    owner_name: str | None
    starters: tuple[LineupAssignment, ...]
    bench: tuple[LeaguePlayer, ...]
    reserve: tuple[LeaguePlayer, ...]
    taxi: tuple[LeaguePlayer, ...]
    wins: int
    losses: int
    ties: int
    points_for: float
    waiver_position: int | None
    waiver_budget_used: int

    @property
    def all_players(self) -> tuple[LeaguePlayer, ...]:
        values: dict[str, LeaguePlayer] = {}
        for assignment in self.starters:
            if assignment.player is not None:
                values[assignment.player.sleeper_player_id] = assignment.player
        for group in (self.bench, self.reserve, self.taxi):
            for player in group:
                values[player.sleeper_player_id] = player
        return tuple(values.values())

    def agent_view(self) -> dict[str, Any]:
        return {
            "roster_id": self.roster_id,
            "owner": self.owner_name,
            "record": {
                "wins": self.wins,
                "losses": self.losses,
                "ties": self.ties,
                "points_for": self.points_for,
            },
            "waiver_position": self.waiver_position,
            "waiver_budget_used": self.waiver_budget_used,
            "starters": [assignment.agent_view() for assignment in self.starters],
            "bench": [player.agent_view() for player in self.bench],
            "reserve": [player.agent_view() for player in self.reserve],
            "taxi": [player.agent_view() for player in self.taxi],
        }


@dataclass(frozen=True)
class ManagedMatchup:
    week: int
    matchup_id: int | None
    roster_id: int
    opponent_roster_id: int | None
    opponent_name: str | None
    points: float
    opponent_points: float | None

    def agent_view(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransactionChange:
    action: str
    roster_id: int
    player: LeaguePlayer


@dataclass(frozen=True)
class LeagueTransaction:
    transaction_id: str
    transaction_type: str
    status: str
    week: int
    created_at: str | None
    roster_ids: tuple[int, ...]
    changes: tuple[TransactionChange, ...]
    waiver_bid: int | None
    draft_picks: tuple[dict[str, Any], ...]

    def agent_view(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "type": self.transaction_type,
            "status": self.status,
            "week": self.week,
            "created_at": self.created_at,
            "roster_ids": list(self.roster_ids),
            "changes": [
                {
                    "action": change.action,
                    "roster_id": change.roster_id,
                    "player": change.player.agent_view(),
                }
                for change in self.changes
            ],
            "waiver_bid": self.waiver_bid,
            "draft_picks": list(self.draft_picks),
        }


@dataclass(frozen=True)
class TrendingPlayer:
    trend_type: str
    count: int
    player: LeaguePlayer

    def agent_view(self) -> dict[str, Any]:
        return {
            "trend_type": self.trend_type,
            "count": self.count,
            "player": self.player.agent_view(),
        }


@dataclass(frozen=True)
class AvailableCandidate:
    player_id: str
    sleeper_player_id: str
    display_name: str
    position: str
    team: str | None
    overall_rank: float
    position_rank: int
    best_rank: float | None = None
    worst_rank: float | None = None
    rank_sd: float | None = None

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
        }


@dataclass(frozen=True)
class LeagueContext:
    league_id: str
    league_name: str
    season: int
    status: str
    current_week: int
    season_type: str
    season_start_date: str | None
    managed_user_id: str
    managed_user_name: str
    managed_roster_id: int
    total_rosters: int
    roster_positions: tuple[str, ...]
    scoring_settings: dict[str, float]
    waiver_settings: dict[str, int]
    trade_settings: dict[str, int]
    managed_roster: LeagueRoster
    other_rosters: tuple[LeagueRoster, ...]
    matchup: ManagedMatchup | None
    transactions: tuple[LeagueTransaction, ...]
    trending_adds: tuple[TrendingPlayer, ...]
    trending_drops: tuple[TrendingPlayer, ...]
    available_candidates: tuple[AvailableCandidate, ...]
    ecr_snapshot_date: str | None
    ecr_scoring_format: str
    ecr_league_format: str
    ecr_source: str | None
    ecr_ranking_page: str | None
    unmapped_sleeper_player_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def agent_payload(self) -> dict[str, Any]:
        """Return a compact snapshot suitable for future policy-specific agents."""
        waiver_budget = int(self.waiver_settings.get("waiver_budget", 0))
        return {
            "league": {
                "league_id": self.league_id,
                "name": self.league_name,
                "season": self.season,
                "status": self.status,
                "current_week": self.current_week,
                "season_type": self.season_type,
                "season_start_date": self.season_start_date,
                "total_rosters": self.total_rosters,
                "roster_positions": list(self.roster_positions),
                "scoring_settings": self.scoring_settings,
                "waiver_settings": self.waiver_settings,
                "trade_settings": self.trade_settings,
            },
            "managed_team": {
                **self.managed_roster.agent_view(),
                "waiver_budget_remaining": max(
                    0,
                    waiver_budget - self.managed_roster.waiver_budget_used,
                ),
            },
            "matchup": self.matchup.agent_view() if self.matchup else None,
            "other_rosters": [roster.agent_view() for roster in self.other_rosters],
            "current_week_transactions": [
                transaction.agent_view() for transaction in self.transactions
            ],
            "trending_adds": [item.agent_view() for item in self.trending_adds],
            "trending_drops": [item.agent_view() for item in self.trending_drops],
            "available_ecr_candidates": [
                candidate.agent_view() for candidate in self.available_candidates
            ],
            "ecr": {
                "snapshot_date": self.ecr_snapshot_date,
                "scoring_format": self.ecr_scoring_format,
                "league_format": self.ecr_league_format,
                "source": self.ecr_source,
                "ranking_page": self.ecr_ranking_page,
                "source_roster_assumption": "3 starting WR slots",
            },
            "limitations": list(self.notes),
        }

    def debug_payload(self) -> dict[str, Any]:
        return asdict(self)
