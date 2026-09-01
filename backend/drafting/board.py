"""Deterministic ECR board construction and snake-draft state math."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Protocol

from repositories import nfl_supabase
from tools.ecr_analytics import add_ecr_position_ranks

from .models import DraftCandidate, DraftContext, DraftPick


SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_REQUIREMENTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "SUPER_FLEX": 0,
}


class DraftBoardRepository(Protocol):
    def load_candidates(
        self,
        *,
        season: int,
        scoring_format: str,
        league_format: str,
        snapshot_type: str,
        as_of_date: str | None,
    ) -> tuple[str, list[DraftCandidate], str | None, str | None]: ...


class SupabaseDraftBoardRepository:
    """Load the latest ECR board and provider identities from Supabase."""

    def load_candidates(
        self,
        *,
        season: int,
        scoring_format: str,
        league_format: str,
        snapshot_type: str,
        as_of_date: str | None,
    ) -> tuple[str, list[DraftCandidate], str | None, str | None]:
        selected_date, stored_rows = nfl_supabase.get_ecr_rows(
            season,
            scoring_format,
            league_format,
            snapshot_type,
            as_of_date,
        )
        if selected_date is None or not stored_rows:
            raise RuntimeError(
                "No matching ECR snapshot is stored. Refresh ECR or choose a "
                "stored season/scoring/league format."
            )

        rows = add_ecr_position_ranks([dict(row) for row in stored_rows])
        player_ids = [str(row["player_id"]) for row in rows]
        names = nfl_supabase.get_player_names(player_ids)
        sleeper_ids = nfl_supabase.get_player_external_ids(
            player_ids,
            "sleeper",
        )
        candidates = []
        for row in rows:
            player_id = str(row["player_id"])
            position = str(row.get("position") or "").upper()
            overall_rank = row.get("overall_rank")
            if (
                position not in SUPPORTED_POSITIONS
                or overall_rank is None
                or player_id not in names
            ):
                continue
            candidates.append(
                DraftCandidate(
                    player_id=player_id,
                    external_id=sleeper_ids.get(player_id),
                    display_name=names[player_id],
                    position=position,
                    team=row.get("team"),
                    overall_rank=float(overall_rank),
                    position_rank=int(row["position_rank"]),
                    best_rank=_optional_float(row.get("best_rank")),
                    worst_rank=_optional_float(row.get("worst_rank")),
                    rank_sd=_optional_float(row.get("rank_sd")),
                    rank_delta=_optional_float(row.get("rank_delta")),
                )
            )
        candidates.sort(key=lambda item: (item.overall_rank, item.player_id))
        first = stored_rows[0]
        return (
            str(selected_date),
            candidates,
            first.get("source"),
            first.get("ranking_page"),
        )


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def snake_pick_number(round_number: int, draft_slot: int, teams: int) -> int:
    if round_number < 1:
        raise ValueError("round_number must be positive")
    if not 1 <= draft_slot <= teams:
        raise ValueError("draft_slot must be between 1 and teams")
    offset = draft_slot if round_number % 2 else teams - draft_slot + 1
    return (round_number - 1) * teams + offset


def pick_coordinates(pick_no: int, teams: int) -> tuple[int, int]:
    if pick_no < 1:
        raise ValueError("pick_no must be positive")
    round_number = (pick_no - 1) // teams + 1
    offset = (pick_no - 1) % teams + 1
    draft_slot = offset if round_number % 2 else teams - offset + 1
    return round_number, draft_slot


def next_slot_picks(
    *,
    current_pick: int,
    draft_slot: int,
    teams: int,
    rounds: int,
) -> tuple[int | None, int | None]:
    slot_picks = [
        snake_pick_number(round_number, draft_slot, teams)
        for round_number in range(1, rounds + 1)
    ]
    future = [pick for pick in slot_picks if pick >= current_pick]
    return (
        future[0] if future else None,
        future[1] if len(future) > 1 else None,
    )


def roster_requirements_from_settings(
    settings: dict | None,
) -> dict[str, int]:
    settings = settings or {}
    field_map = {
        "QB": "slots_qb",
        "RB": "slots_rb",
        "WR": "slots_wr",
        "TE": "slots_te",
        "FLEX": "slots_flex",
        "SUPER_FLEX": "slots_super_flex",
    }
    requirements = {}
    for position, field in field_map.items():
        fallback = DEFAULT_REQUIREMENTS[position]
        raw = settings.get(field, fallback)
        try:
            requirements[position] = max(0, int(raw))
        except (TypeError, ValueError):
            requirements[position] = fallback
    return requirements


def _open_starter_slots(
    counts: Counter[str],
    requirements: dict[str, int],
) -> dict[str, int]:
    open_slots = {
        position: max(0, requirements[position] - counts[position])
        for position in SUPPORTED_POSITIONS
    }
    excess = {
        position: max(0, counts[position] - requirements[position])
        for position in SUPPORTED_POSITIONS
    }
    flex_used = min(
        requirements["FLEX"],
        sum(excess[position] for position in ("RB", "WR", "TE")),
    )
    open_slots["FLEX"] = max(0, requirements["FLEX"] - flex_used)

    remaining_excess = sum(excess.values()) - flex_used
    super_flex_used = min(requirements["SUPER_FLEX"], remaining_excess)
    open_slots["SUPER_FLEX"] = max(
        0,
        requirements["SUPER_FLEX"] - super_flex_used,
    )
    return open_slots


def _candidate_key(candidate: DraftCandidate) -> str:
    return " ".join(candidate.display_name.lower().replace(".", "").split())


class DraftContextBuilder:
    def __init__(
        self,
        repository: DraftBoardRepository | None = None,
    ) -> None:
        self.repository = repository or SupabaseDraftBoardRepository()

    def build(
        self,
        *,
        season: int,
        scoring_format: str,
        league_format: str,
        teams: int,
        rounds: int,
        draft_slot: int,
        roster_id: int | None,
        picks: list[DraftPick],
        roster_requirements: dict[str, int] | None = None,
        draft_status: str = "drafting",
        snapshot_type: str = "current",
        as_of_date: str | None = None,
        shortlist_size: int = 28,
        _loaded_board: tuple[
            str,
            list[DraftCandidate],
            str | None,
            str | None,
        ]
        | None = None,
    ) -> DraftContext:
        if teams < 2:
            raise ValueError("teams must be at least 2")
        if rounds < 1:
            raise ValueError("rounds must be positive")
        if shortlist_size < 8:
            raise ValueError("shortlist_size must be at least 8")
        if not 1 <= draft_slot <= teams:
            raise ValueError("draft_slot must be between 1 and teams")

        selected_date, board, source, ranking_page = (
            _loaded_board
            if _loaded_board is not None
            else self.repository.load_candidates(
                season=season,
                scoring_format=scoring_format,
                league_format=league_format,
                snapshot_type=snapshot_type,
                as_of_date=as_of_date,
            )
        )
        by_external = {
            candidate.external_id: candidate
            for candidate in board
            if candidate.external_id
        }
        by_name = {_candidate_key(candidate): candidate for candidate in board}

        drafted_player_ids: set[str] = set()
        hydrated_picks = []
        for pick in sorted(picks, key=lambda item: item.pick_no):
            candidate = by_external.get(pick.external_player_id)
            if candidate is None and pick.display_name:
                normalized = " ".join(
                    pick.display_name.lower().replace(".", "").split()
                )
                candidate = by_name.get(normalized)
            if candidate is not None:
                drafted_player_ids.add(candidate.player_id)
                hydrated_picks.append(
                    replace(
                        pick,
                        display_name=pick.display_name or candidate.display_name,
                        position=pick.position or candidate.position,
                        team=pick.team or candidate.team,
                    )
                )
            else:
                hydrated_picks.append(pick)

        target_roster = roster_id if roster_id is not None else draft_slot
        my_roster = tuple(
            pick for pick in hydrated_picks if pick.roster_id == target_roster
        )
        counts: Counter[str] = Counter(
            pick.position for pick in my_roster if pick.position
        )
        requirements = dict(roster_requirements or DEFAULT_REQUIREMENTS)
        for key, fallback in DEFAULT_REQUIREMENTS.items():
            requirements.setdefault(key, fallback)

        available = [
            candidate
            for candidate in board
            if candidate.player_id not in drafted_player_ids
        ]
        shortlist = _balanced_shortlist(available, shortlist_size)
        current_pick = len(hydrated_picks) + 1
        current_round = min(rounds, (current_pick - 1) // teams + 1)
        next_pick, following_pick = next_slot_picks(
            current_pick=current_pick,
            draft_slot=draft_slot,
            teams=teams,
            rounds=rounds,
        )
        picks_until_turn = (
            next_pick - current_pick if next_pick is not None else None
        )
        notes = (
            "Available-player filtering uses provider ID first and a normalized "
            "name fallback for unmapped historical picks.",
            "The shortlist contains top overall ECR plus positional coverage; "
            "it is not the entire remaining player pool.",
        )
        return DraftContext(
            season=season,
            scoring_format=scoring_format,
            league_format=league_format,
            teams=teams,
            rounds=rounds,
            draft_slot=draft_slot,
            roster_id=roster_id,
            draft_status=draft_status,
            current_pick=current_pick,
            current_round=current_round,
            on_clock=next_pick == current_pick,
            next_pick=next_pick,
            picks_until_turn=picks_until_turn,
            following_pick=following_pick,
            picks_between_turns=(
                following_pick - next_pick - 1
                if next_pick is not None and following_pick is not None
                else None
            ),
            roster_requirements=requirements,
            roster_counts={
                position: counts[position] for position in SUPPORTED_POSITIONS
            },
            open_starter_slots=_open_starter_slots(counts, requirements),
            my_roster=my_roster,
            available_candidates=tuple(shortlist),
            ecr_snapshot_date=selected_date,
            ecr_source=source,
            ecr_ranking_page=ranking_page,
            notes=notes,
        )

    def simulate(
        self,
        *,
        season: int,
        scoring_format: str,
        league_format: str,
        teams: int,
        rounds: int,
        draft_slot: int,
        target_round: int,
        roster_requirements: dict[str, int] | None = None,
        snapshot_type: str = "current",
        as_of_date: str | None = None,
        shortlist_size: int = 28,
    ) -> DraftContext:
        """Create a reproducible chalk-ECR board immediately before one pick."""
        if not 1 <= target_round <= rounds:
            raise ValueError("target_round must be between 1 and rounds")
        loaded_board = self.repository.load_candidates(
                season=season,
                scoring_format=scoring_format,
                league_format=league_format,
                snapshot_type=snapshot_type,
                as_of_date=as_of_date,
        )
        _, board, _, _ = loaded_board
        target_pick = snake_pick_number(target_round, draft_slot, teams)
        if len(board) < target_pick - 1:
            raise RuntimeError("The stored ECR board is too small for this simulation")
        simulated_picks = []
        for pick_no, candidate in enumerate(board[: target_pick - 1], start=1):
            round_number, slot = pick_coordinates(pick_no, teams)
            simulated_picks.append(
                DraftPick(
                    pick_no=pick_no,
                    round=round_number,
                    draft_slot=slot,
                    roster_id=slot,
                    external_player_id=(
                        candidate.external_id or f"internal:{candidate.player_id}"
                    ),
                    display_name=candidate.display_name,
                    position=candidate.position,
                    team=candidate.team,
                )
            )
        return self.build(
            season=season,
            scoring_format=scoring_format,
            league_format=league_format,
            teams=teams,
            rounds=rounds,
            draft_slot=draft_slot,
            roster_id=draft_slot,
            picks=simulated_picks,
            roster_requirements=roster_requirements,
            draft_status="mock",
            snapshot_type=snapshot_type,
            as_of_date=as_of_date,
            shortlist_size=shortlist_size,
            _loaded_board=loaded_board,
        )


def _balanced_shortlist(
    available: list[DraftCandidate],
    limit: int,
) -> list[DraftCandidate]:
    selected: dict[str, DraftCandidate] = {
        candidate.player_id: candidate for candidate in available[: max(8, limit - 8)]
    }
    for position in SUPPORTED_POSITIONS:
        added = 0
        for candidate in available:
            if candidate.position != position:
                continue
            selected.setdefault(candidate.player_id, candidate)
            added += 1
            if added == 3:
                break
    return sorted(
        selected.values(),
        key=lambda item: (item.overall_rank, item.player_id),
    )[:limit]
