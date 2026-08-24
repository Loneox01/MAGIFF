"""Deterministic 17-0 Challenge game used by Discord clients."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, Sequence


class RosterSlot(StrEnum):
    QB = "QB"
    RB1 = "RB1"
    RB2 = "RB2"
    WR1 = "WR1"
    WR2 = "WR2"
    TE = "TE"
    FLEX = "FLEX"


ROSTER_SLOTS = (
    RosterSlot.QB,
    RosterSlot.RB1,
    RosterSlot.RB2,
    RosterSlot.WR1,
    RosterSlot.WR2,
    RosterSlot.TE,
    RosterSlot.FLEX,
)
FLEX_POSITIONS = ("RB", "WR", "TE")
BASE_SLOTS_BY_POSITION = {
    "RB": (RosterSlot.RB1, RosterSlot.RB2),
    "WR": (RosterSlot.WR1, RosterSlot.WR2),
    "TE": (RosterSlot.TE,),
}

# Each index is a win total. The closer anchor wins; ties favor the lower
# record. Two 50-point steps pad each tail and the middle uses 100-point steps.
RECORD_SCORE_ANCHORS = (
    800.0,
    850.0,
    900.0,
    1_000.0,
    1_100.0,
    1_200.0,
    1_300.0,
    1_400.0,
    1_500.0,
    1_600.0,
    1_700.0,
    1_800.0,
    1_900.0,
    2_000.0,
    2_100.0,
    2_200.0,
    2_250.0,
    2_300.0,
)


class GameStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GameAction(StrEnum):
    LOCK = "lock"
    REROLL_TEAM = "reroll_team"
    REROLL_POSITION = "reroll_position"
    FORFEIT = "forfeit"


class GameOutcome(StrEnum):
    READY = "ready"
    UPDATED = "updated"
    COMPLETED = "completed"
    FORFEITED = "forfeited"
    STALE = "stale"
    NOT_OWNER = "not_owner"
    NOT_FOUND = "not_found"
    ALREADY_COMPLETE = "already_complete"
    REROLL_USED = "reroll_used"
    REROLL_UNAVAILABLE = "reroll_unavailable"
    SEASON_UNAVAILABLE = "season_unavailable"


@dataclass(frozen=True)
class PlayerPoolEntry:
    player_id: str
    display_name: str
    team: str
    position: str
    fantasy_points_ppr: float
    team_name: str
    team_logo_url: str | None = None
    team_color: str | None = None


@dataclass(frozen=True)
class RolledPlayer:
    roster_slot: RosterSlot
    player: PlayerPoolEntry


@dataclass(frozen=True)
class GamePick:
    pick_number: int
    roster_slot: RosterSlot
    player: PlayerPoolEntry


@dataclass(frozen=True)
class RosterGameState:
    game_id: str
    app_user_id: str
    discord_user_id: str
    discord_guild_id: str
    season: int
    reveal_during_roll: bool
    status: GameStatus
    pending: RolledPlayer | None
    picks: tuple[GamePick, ...] = ()
    team_reroll_used: bool = False
    position_reroll_used: bool = False
    total_points: float | None = None
    wins: int | None = None
    losses: int | None = None
    version: int = 0


@dataclass(frozen=True)
class GameResult:
    outcome: GameOutcome
    state: RosterGameState | None = None
    note: str | None = None


class RosterGameRepository(Protocol):
    def latest_completed_season(self) -> int | None: ...

    def season_is_complete(self, season: int) -> bool: ...

    def player_pool(self, season: int) -> list[PlayerPoolEntry]: ...

    def ensure_user(self, discord_user_id: str, display_name: str | None) -> str: ...

    def create_game(self, state: RosterGameState) -> None: ...

    def get_game(self, game_id: str) -> RosterGameState | None: ...

    def save_transition(
        self,
        state: RosterGameState,
        *,
        expected_version: int,
        new_pick: GamePick | None,
        interaction_id: str,
        action: GameAction,
    ) -> bool: ...


def wins_for_score(total_points: float) -> int:
    """Map a roster total to the nearest padded 0-17 through 17-0 anchor."""
    return min(
        range(len(RECORD_SCORE_ANCHORS)),
        key=lambda wins: (abs(total_points - RECORD_SCORE_ANCHORS[wins]), wins),
    )


def position_choices(slot: RosterSlot) -> tuple[str, ...]:
    if slot == RosterSlot.QB:
        return ("QB",)
    if slot in {RosterSlot.RB1, RosterSlot.RB2}:
        return ("RB",)
    if slot in {RosterSlot.WR1, RosterSlot.WR2}:
        return ("WR",)
    if slot == RosterSlot.TE:
        return ("TE",)
    return FLEX_POSITIONS


class RosterGameService:
    """Create and advance games without model calls or client-side state."""

    def __init__(
        self,
        repository: RosterGameRepository | None = None,
        *,
        rng: random.Random | random.SystemRandom | None = None,
    ) -> None:
        if repository is None:
            from repositories.roster_game_supabase import (
                SupabaseRosterGameRepository,
            )

            repository = SupabaseRosterGameRepository()
        self.repository = repository
        self.rng = rng or random.SystemRandom()
        self._pool_cache: dict[int, tuple[PlayerPoolEntry, ...]] = {}

    def start(
        self,
        *,
        discord_user_id: str,
        display_name: str | None,
        discord_guild_id: str,
        season: int | None = None,
        reveal_during_roll: bool = False,
    ) -> GameResult:
        selected_season = season or self.repository.latest_completed_season()
        if selected_season is None:
            return GameResult(
                GameOutcome.SEASON_UNAVAILABLE,
                note="No completed regular-season player data is stored.",
            )
        if season is not None and not self.repository.season_is_complete(selected_season):
            return GameResult(
                GameOutcome.SEASON_UNAVAILABLE,
                note=(
                    f"The {selected_season} regular season is not complete in "
                    "the stored weekly data."
                ),
            )
        pool = self._pool(selected_season)
        team_positions: dict[str, set[str]] = {}
        for entry in pool:
            team_positions.setdefault(entry.team, set()).add(entry.position)
        if (
            len(team_positions) != 32
            or any(
                positions != {"QB", "RB", "WR", "TE"}
                for positions in team_positions.values()
            )
        ):
            return GameResult(
                GameOutcome.SEASON_UNAVAILABLE,
                note=(
                    f"The {selected_season} regular-season pool does not contain "
                    "all 32 teams and four fantasy positions."
                ),
            )
        pending = self._new_roll(pool, (), ())
        if pending is None:
            return GameResult(
                GameOutcome.SEASON_UNAVAILABLE,
                note=f"The {selected_season} game pool is incomplete.",
            )
        app_user_id = self.repository.ensure_user(discord_user_id, display_name)
        state = RosterGameState(
            game_id=str(uuid.uuid4()),
            app_user_id=app_user_id,
            discord_user_id=discord_user_id,
            discord_guild_id=discord_guild_id,
            season=selected_season,
            reveal_during_roll=reveal_during_roll,
            status=GameStatus.ACTIVE,
            pending=pending,
        )
        self.repository.create_game(state)
        return GameResult(GameOutcome.READY, state)

    def act(
        self,
        *,
        game_id: str,
        expected_version: int,
        discord_user_id: str,
        interaction_id: str,
        action: GameAction,
    ) -> GameResult:
        state = self.repository.get_game(game_id)
        if state is None:
            return GameResult(GameOutcome.NOT_FOUND)
        if state.discord_user_id != discord_user_id:
            return GameResult(GameOutcome.NOT_OWNER, state)
        if state.version != expected_version:
            return GameResult(
                GameOutcome.STALE,
                state,
                "That button belonged to an older roll; showing the current one.",
            )
        if state.status != GameStatus.ACTIVE or state.pending is None:
            return GameResult(GameOutcome.ALREADY_COMPLETE, state)

        pool = self._pool(state.season)
        if action == GameAction.REROLL_TEAM:
            result = self._reroll_team(state, pool)
            new_pick = None
        elif action == GameAction.REROLL_POSITION:
            result = self._reroll_position(state, pool)
            new_pick = None
        elif action == GameAction.FORFEIT:
            result = self._forfeit(state)
            new_pick = None
        else:
            result, new_pick = self._lock(state, pool)

        if result.outcome not in {
            GameOutcome.UPDATED,
            GameOutcome.COMPLETED,
            GameOutcome.FORFEITED,
        }:
            return result
        assert result.state is not None
        saved = self.repository.save_transition(
            result.state,
            expected_version=expected_version,
            new_pick=new_pick,
            interaction_id=interaction_id,
            action=action,
        )
        if not saved:
            current = self.repository.get_game(game_id)
            return GameResult(
                GameOutcome.STALE,
                current,
                "Another click advanced this game first; showing the current state.",
            )
        return result

    def can_reroll_team(self, state: RosterGameState) -> bool:
        if state.team_reroll_used or state.pending is None:
            return False
        return bool(self._team_alternatives(state, self._pool(state.season)))

    def can_reroll_position(self, state: RosterGameState) -> bool:
        if state.position_reroll_used or state.pending is None:
            return False
        return bool(self._position_alternatives(state, self._pool(state.season)))

    def _pool(self, season: int) -> tuple[PlayerPoolEntry, ...]:
        if season not in self._pool_cache:
            self._pool_cache[season] = tuple(self.repository.player_pool(season))
        return self._pool_cache[season]

    def _new_roll(
        self,
        pool: Sequence[PlayerPoolEntry],
        picks: Sequence[GamePick],
        excluded_slots: Sequence[RosterSlot],
    ) -> RolledPlayer | None:
        used_teams = {pick.player.team for pick in picks}
        used_players = {pick.player.player_id for pick in picks}
        combinations = [
            (slot, position)
            for slot, position in self._available_slot_positions(
                picks,
                excluded_slots,
            )
            if any(
                entry.position == position
                and entry.team not in used_teams
                and entry.player_id not in used_players
                for entry in pool
            )
        ]
        if not combinations:
            return None

        # Choose an open roster slot first so FLEX is not three times as likely
        # merely because it accepts three positions.
        open_slots = list(dict.fromkeys(slot for slot, _ in combinations))
        slot = self.rng.choice(open_slots)
        valid_positions = [
            position for candidate_slot, position in combinations
            if candidate_slot == slot
        ]
        position = self.rng.choice(valid_positions)
        candidates = [
            entry for entry in pool
            if entry.position == position
            and entry.team not in used_teams
            and entry.player_id not in used_players
        ]
        return RolledPlayer(slot, self.rng.choice(candidates))

    @staticmethod
    def _available_slot_positions(
        picks: Sequence[GamePick],
        excluded_slots: Sequence[RosterSlot] = (),
    ) -> list[tuple[RosterSlot, str]]:
        used_slots = {pick.roster_slot for pick in picks} | set(excluded_slots)
        combinations: list[tuple[RosterSlot, str]] = []
        for slot in ROSTER_SLOTS:
            if slot in used_slots:
                continue
            if slot != RosterSlot.FLEX:
                combinations.extend(
                    (slot, position) for position in position_choices(slot)
                )
                continue
            # FLEX becomes eligible for a position only after every base slot
            # for that position is occupied. This prevents an early FLEX roll
            # from bypassing an available RB, WR, or TE slot.
            for position in FLEX_POSITIONS:
                if all(
                    base_slot in used_slots
                    for base_slot in BASE_SLOTS_BY_POSITION[position]
                ):
                    combinations.append((slot, position))
        return combinations

    def _team_alternatives(
        self,
        state: RosterGameState,
        pool: Sequence[PlayerPoolEntry],
    ) -> list[PlayerPoolEntry]:
        assert state.pending is not None
        used_teams = {pick.player.team for pick in state.picks}
        used_players = {pick.player.player_id for pick in state.picks}
        return [
            entry for entry in pool
            if entry.position == state.pending.player.position
            and entry.team != state.pending.player.team
            and entry.team not in used_teams
            and entry.player_id not in used_players
        ]

    def _position_alternatives(
        self,
        state: RosterGameState,
        pool: Sequence[PlayerPoolEntry],
    ) -> list[RolledPlayer]:
        assert state.pending is not None
        used_players = {pick.player.player_id for pick in state.picks}
        current_position = state.pending.player.position
        entries = {
            entry.position: entry for entry in pool
            if entry.team == state.pending.player.team
            and entry.player_id not in used_players
        }
        return [
            RolledPlayer(slot, entries[position])
            for slot, position in self._available_slot_positions(state.picks)
            if position != current_position and position in entries
        ]

    @staticmethod
    def _forfeit(state: RosterGameState) -> GameResult:
        total = round(
            sum(pick.player.fantasy_points_ppr for pick in state.picks),
            2,
        )
        forfeited = replace(
            state,
            status=GameStatus.ABANDONED,
            pending=None,
            total_points=total,
            version=state.version + 1,
        )
        return GameResult(
            GameOutcome.FORFEITED,
            forfeited,
        )

    def _reroll_team(
        self,
        state: RosterGameState,
        pool: Sequence[PlayerPoolEntry],
    ) -> GameResult:
        if state.team_reroll_used:
            return GameResult(GameOutcome.REROLL_USED, state, "Team reroll already used.")
        alternatives = self._team_alternatives(state, pool)
        if not alternatives:
            return GameResult(
                GameOutcome.REROLL_UNAVAILABLE,
                state,
                "No legal unused team remains for this position.",
            )
        assert state.pending is not None
        updated = replace(
            state,
            pending=RolledPlayer(
                state.pending.roster_slot,
                self.rng.choice(alternatives),
            ),
            team_reroll_used=True,
            version=state.version + 1,
        )
        return GameResult(GameOutcome.UPDATED, updated, "Team reroll used.")

    def _reroll_position(
        self,
        state: RosterGameState,
        pool: Sequence[PlayerPoolEntry],
    ) -> GameResult:
        if state.position_reroll_used:
            return GameResult(
                GameOutcome.REROLL_USED,
                state,
                "Position reroll already used.",
            )
        alternatives = self._position_alternatives(state, pool)
        if not alternatives:
            return GameResult(
                GameOutcome.REROLL_UNAVAILABLE,
                state,
                "No different legal position remains for this team.",
            )
        updated = replace(
            state,
            pending=self.rng.choice(alternatives),
            position_reroll_used=True,
            version=state.version + 1,
        )
        return GameResult(GameOutcome.UPDATED, updated, "Position reroll used.")

    def _lock(
        self,
        state: RosterGameState,
        pool: Sequence[PlayerPoolEntry],
    ) -> tuple[GameResult, GamePick]:
        assert state.pending is not None
        pick = GamePick(
            pick_number=len(state.picks) + 1,
            roster_slot=state.pending.roster_slot,
            player=state.pending.player,
        )
        picks = (*state.picks, pick)
        if len(picks) == len(ROSTER_SLOTS):
            total = round(sum(value.player.fantasy_points_ppr for value in picks), 2)
            wins = wins_for_score(total)
            completed = replace(
                state,
                status=GameStatus.COMPLETED,
                pending=None,
                picks=picks,
                total_points=total,
                wins=wins,
                losses=17 - wins,
                version=state.version + 1,
            )
            return GameResult(GameOutcome.COMPLETED, completed), pick

        pending = self._new_roll(pool, picks, ())
        if pending is None:
            raise RuntimeError("No legal roll remains for an incomplete game")
        updated = replace(
            state,
            picks=picks,
            pending=pending,
            version=state.version + 1,
        )
        return GameResult(GameOutcome.UPDATED, updated), pick
