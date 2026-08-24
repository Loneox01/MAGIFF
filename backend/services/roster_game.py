"""Deterministic 17-0 Challenge game used by Discord clients."""

from __future__ import annotations

import random
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from math import floor
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

ZERO_WIN_RANGE_FRACTION = 0.10
UNDEFEATED_RANGE_FRACTION = 0.85
MIDDLE_RECORD_COUNT = 16
SEASON_TOTAL_SCORE_INCREMENT = 25.0
PPG_SCORE_INCREMENT = 1.0


class GameScoringMode(StrEnum):
    SEASON_TOTAL = "season_total"
    PPG = "ppg"


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
class RecordScale:
    """Season-specific hard score boundaries for records from 0-17 to 17-0."""

    legal_minimum: float
    legal_maximum: float
    # Each value is the minimum score for the next win total. There are 17
    # boundaries: 0 -> 1 through 16 -> 17.
    win_boundaries: tuple[float, ...]

    @property
    def zero_win_cutoff(self) -> float:
        return self.win_boundaries[0]

    @property
    def undefeated_cutoff(self) -> float:
        return self.win_boundaries[-1]


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
    scoring_mode: GameScoringMode
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

    def player_pool(
        self,
        season: int,
        scoring_mode: GameScoringMode,
    ) -> list[PlayerPoolEntry]: ...

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


def _round_to_increment(value: float, increment: float) -> float:
    """Round a positive fantasy score to the nearest configured increment."""
    return round(floor(value / increment + 0.5) * increment, 2)


def legal_score_extrema(
    pool: Sequence[PlayerPoolEntry],
) -> tuple[float, float]:
    """Return the minimum and maximum scores of any fully legal roster.

    The dynamic program enforces every game constraint that can affect the
    final roster: all seven slots, no repeated team, and no repeated player.
    Only player IDs that occur more than once in the team-position pool need
    to be represented in the state mask.
    """
    duplicate_player_ids = {
        player_id
        for player_id, count in Counter(
            entry.player_id for entry in pool
        ).items()
        if count > 1
    }
    duplicate_bits = {
        player_id: 1 << index
        for index, player_id in enumerate(sorted(duplicate_player_ids))
    }
    entries_by_team: dict[str, list[PlayerPoolEntry]] = {}
    for entry in pool:
        entries_by_team.setdefault(entry.team, []).append(entry)

    # (filled slot mask, used duplicate-player mask) -> (minimum, maximum)
    states: dict[tuple[int, int], tuple[float, float]] = {(0, 0): (0.0, 0.0)}
    for team in sorted(entries_by_team):
        next_states = dict(states)
        for (slot_mask, player_mask), (minimum, maximum) in states.items():
            for entry in entries_by_team[team]:
                player_bit = duplicate_bits.get(entry.player_id, 0)
                if player_bit and player_mask & player_bit:
                    continue
                for slot_index, slot in enumerate(ROSTER_SLOTS):
                    slot_bit = 1 << slot_index
                    if slot_mask & slot_bit:
                        continue
                    if entry.position not in position_choices(slot):
                        continue
                    key = (slot_mask | slot_bit, player_mask | player_bit)
                    candidate = (
                        minimum + entry.fantasy_points_ppr,
                        maximum + entry.fantasy_points_ppr,
                    )
                    existing = next_states.get(key)
                    if existing is None:
                        next_states[key] = candidate
                    else:
                        next_states[key] = (
                            min(existing[0], candidate[0]),
                            max(existing[1], candidate[1]),
                        )
        states = next_states

    complete_slot_mask = (1 << len(ROSTER_SLOTS)) - 1
    complete_scores = [
        extrema
        for (slot_mask, _), extrema in states.items()
        if slot_mask == complete_slot_mask
    ]
    if not complete_scores:
        raise ValueError("The player pool cannot construct a legal full roster")
    return (
        round(min(score[0] for score in complete_scores), 2),
        round(max(score[1] for score in complete_scores), 2),
    )


def build_record_scale(
    pool: Sequence[PlayerPoolEntry],
    scoring_mode: GameScoringMode,
) -> RecordScale:
    """Build rounded record boundaries from a season's attainable score span."""
    legal_minimum, legal_maximum = legal_score_extrema(pool)
    score_range = legal_maximum - legal_minimum
    if score_range <= 0:
        raise ValueError("The legal roster score range must be positive")

    increment = (
        PPG_SCORE_INCREMENT
        if scoring_mode == GameScoringMode.PPG
        else SEASON_TOTAL_SCORE_INCREMENT
    )
    low = _round_to_increment(
        legal_minimum + ZERO_WIN_RANGE_FRACTION * score_range,
        increment,
    )
    high = _round_to_increment(
        legal_minimum + UNDEFEATED_RANGE_FRACTION * score_range,
        increment,
    )
    if high <= low:
        raise ValueError("The rounded record score range is too narrow")

    win_boundaries = tuple(
        _round_to_increment(
            low + index * (high - low) / MIDDLE_RECORD_COUNT,
            increment,
        )
        for index in range(MIDDLE_RECORD_COUNT + 1)
    )
    if any(
        current <= previous
        for previous, current in zip(win_boundaries, win_boundaries[1:])
    ):
        raise ValueError("Rounded record boundaries must be strictly increasing")
    return RecordScale(
        legal_minimum=legal_minimum,
        legal_maximum=legal_maximum,
        win_boundaries=win_boundaries,
    )


def wins_for_score(total_points: float, scale: RecordScale) -> int:
    """Map a score to a record using explicit season-specific boundaries."""
    for wins, boundary in enumerate(scale.win_boundaries):
        if total_points < boundary:
            return wins
    return 17


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
        self._pool_cache: dict[
            tuple[int, GameScoringMode],
            tuple[PlayerPoolEntry, ...],
        ] = {}
        self._record_scale_cache: dict[
            tuple[int, GameScoringMode],
            RecordScale,
        ] = {}

    def start(
        self,
        *,
        discord_user_id: str,
        display_name: str | None,
        discord_guild_id: str,
        season: int | None = None,
        scoring_mode: GameScoringMode = GameScoringMode.SEASON_TOTAL,
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
        pool = self._pool(selected_season, scoring_mode)
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
        try:
            self._record_scale(selected_season, scoring_mode, pool)
        except ValueError as error:
            return GameResult(
                GameOutcome.SEASON_UNAVAILABLE,
                note=f"The {selected_season} game pool cannot be scored: {error}.",
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
            scoring_mode=scoring_mode,
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

        pool = self._pool(state.season, state.scoring_mode)
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
        return bool(
            self._team_alternatives(
                state,
                self._pool(state.season, state.scoring_mode),
            )
        )

    def can_reroll_position(self, state: RosterGameState) -> bool:
        if state.position_reroll_used or state.pending is None:
            return False
        return bool(
            self._position_alternatives(
                state,
                self._pool(state.season, state.scoring_mode),
            )
        )

    def _pool(
        self,
        season: int,
        scoring_mode: GameScoringMode,
    ) -> tuple[PlayerPoolEntry, ...]:
        key = (season, scoring_mode)
        if key not in self._pool_cache:
            self._pool_cache[key] = tuple(
                self.repository.player_pool(season, scoring_mode)
            )
        return self._pool_cache[key]

    def _record_scale(
        self,
        season: int,
        scoring_mode: GameScoringMode,
        pool: Sequence[PlayerPoolEntry] | None = None,
    ) -> RecordScale:
        key = (season, scoring_mode)
        if key not in self._record_scale_cache:
            selected_pool = pool or self._pool(season, scoring_mode)
            self._record_scale_cache[key] = build_record_scale(
                selected_pool,
                scoring_mode,
            )
        return self._record_scale_cache[key]

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
            wins = wins_for_score(
                total,
                self._record_scale(state.season, state.scoring_mode, pool),
            )
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
