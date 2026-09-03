"""Build a read-only weekly lineup snapshot and projection-only baseline."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from functools import lru_cache
from typing import Protocol
from zoneinfo import ZoneInfo

from integrations.sleeper_projections import (
    SleeperProjectionClient,
    SleeperProjectionError,
    SleeperWeeklyProjection,
)
from league_management.context import LeagueContextBuilder
from league_management.models import LeaguePlayer, LineupAssignment
from repositories import nfl_supabase

from .models import (
    BaselineAssignment,
    LineupContext,
    LineupPlayer,
    LineupSlot,
)


TEAM_ALIASES = {"AZ": "ARI", "LAR": "LA"}
FLEX_ELIGIBILITY = {
    "FLEX": {"RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "IDP_FLEX": {"DL", "LB", "DB"},
}


class LineupProjectionSource(Protocol):
    def weekly_projections(
        self,
        *,
        season: int,
        week: int,
        season_type: str,
        positions: tuple[str, ...],
    ) -> list[SleeperWeeklyProjection]: ...


class LineupScheduleSource(Protocol):
    def week_games(self, *, season: int, week: int) -> list[dict]: ...


class SupabaseLineupScheduleSource:
    def week_games(self, *, season: int, week: int) -> list[dict]:
        return nfl_supabase.get_week_games(season, week)


NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")
LOCKED_GAME_STATUSES = {
    "in progress",
    "in_progress",
    "live",
    "complete",
    "completed",
    "final",
    "post",
}


def _team(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return TEAM_ALIASES.get(normalized, normalized) or None


def _position(value: str | None) -> str:
    normalized = str(value or "UNKNOWN").upper()
    return "DEF" if normalized in {"DST", "D/ST"} else normalized


def _kickoff_at(game: dict) -> datetime:
    raw_day = game.get("gameday")
    raw_time = game.get("gametime")
    if raw_day is None or raw_time is None:
        raise ValueError("Schedule row is missing gameday or gametime")
    game_day = raw_day if isinstance(raw_day, date) else date.fromisoformat(str(raw_day))
    if isinstance(raw_time, time):
        game_time = raw_time
    else:
        value = str(raw_time).strip().upper().replace(" ", "")
        parsed = None
        for pattern in ("%H:%M", "%H:%M:%S", "%I:%M%p"):
            try:
                parsed = datetime.strptime(value, pattern).time()
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"Unsupported NFL game time {raw_time!r}")
        game_time = parsed
    return datetime.combine(
        game_day,
        game_time,
        tzinfo=NFL_SCHEDULE_TIMEZONE,
    ).astimezone(UTC)


def _schedule_by_team(
    games: list[dict],
) -> dict[str, tuple[str | None, datetime]]:
    values: dict[str, tuple[str | None, datetime]] = {}
    for game in games:
        kickoff = _kickoff_at(game)
        game_id = str(game["game_id"]) if game.get("game_id") else None
        for field in ("home_team", "away_team"):
            team = _team(str(game.get(field) or ""))
            if team:
                if team in values:
                    raise ValueError(f"Multiple games found for {team} in one week")
                values[team] = (game_id, kickoff)
    return values


def _slot_ids(assignments: tuple[LineupAssignment, ...]) -> tuple[LineupSlot, ...]:
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for assignment in assignments:
        slot_type = str(assignment.slot).upper()
        totals[slot_type] = totals.get(slot_type, 0) + 1

    slots = []
    for assignment in assignments:
        slot_type = str(assignment.slot).upper()
        counts[slot_type] = counts.get(slot_type, 0) + 1
        slot_id = (
            f"{slot_type}{counts[slot_type]}"
            if totals[slot_type] > 1
            else slot_type
        )
        slots.append(
            LineupSlot(
                slot_id=slot_id,
                slot_type=slot_type,
                current_player_id=(
                    assignment.player.sleeper_player_id
                    if assignment.player is not None
                    else None
                ),
            )
        )
    return tuple(slots)


def _eligible_slot_types(
    position: str,
    slots: tuple[LineupSlot, ...],
) -> tuple[str, ...]:
    available_types = {slot.slot_type for slot in slots}
    values = []
    if position in available_types:
        values.append(position)
    for slot_type, positions in FLEX_ELIGIBILITY.items():
        if slot_type in available_types and position in positions:
            values.append(slot_type)
    return tuple(values)


def _lineup_player(
    player: LeaguePlayer,
    *,
    roster_group: str,
    current_slot_id: str | None,
    slots: tuple[LineupSlot, ...],
    week: int,
    projection: SleeperWeeklyProjection | None,
    scoring_settings: dict[str, float],
    schedule: tuple[str | None, datetime] | None,
    as_of: datetime,
) -> LineupPlayer:
    position = _position(player.position or (projection.position if projection else None))
    return LineupPlayer(
        sleeper_player_id=player.sleeper_player_id,
        player_id=player.player_id,
        display_name=player.display_name,
        position=position,
        team=_team(player.team or (projection.team if projection else None)),
        roster_status=player.roster_status,
        roster_group=roster_group,
        current_slot_id=current_slot_id,
        eligible_slot_types=_eligible_slot_types(position, slots),
        projection_week=week,
        projected_points=(
            projection.projected_points(scoring_settings)
            if projection is not None
            else None
        ),
        opponent=_team(projection.opponent) if projection else None,
        game_id=(schedule[0] if schedule else (projection.game_id if projection else None)),
        game_date=projection.game_date if projection else None,
        kickoff_at=schedule[1] if schedule else None,
        game_status=projection.game_status if projection else None,
        is_locked=(
            (schedule is not None and schedule[1] <= as_of)
            or (
                projection is not None
                and str(projection.game_status or "").strip().casefold()
                in LOCKED_GAME_STATUSES
            )
        ),
        projection_updated_at=projection.updated_at if projection else None,
        projection_source=(projection.company or "Sleeper") if projection else None,
        injury_status=projection.injury_status if projection else None,
        injury_body_part=projection.injury_body_part if projection else None,
        injury_notes=projection.injury_notes if projection else None,
        injury_start_date=projection.injury_start_date if projection else None,
        injury_news_updated_at=projection.news_updated_at if projection else None,
    )


def optimize_projected_lineup(
    slots: tuple[LineupSlot, ...],
    players: tuple[LineupPlayer, ...],
) -> tuple[BaselineAssignment, ...]:
    """Maximize filled legal slots first and projected points second."""
    candidates = tuple(
        sorted(
            (
                player
                for player in players
                if player.roster_group in {"starter", "bench"}
            ),
            key=lambda player: player.sleeper_player_id,
        )
    )
    ordered_slots = tuple(
        sorted(
            slots,
            key=lambda slot: (
                sum(
                    player.can_fill(slot)
                    for player in candidates
                ),
                slot.slot_id,
            ),
        )
    )

    @lru_cache(maxsize=None)
    def solve(index: int, used_mask: int) -> tuple[int, float, tuple[int | None, ...]]:
        if index == len(ordered_slots):
            return 0, 0.0, ()
        slot = ordered_slots[index]
        next_filled, next_points, next_assignments = solve(index + 1, used_mask)
        best = (next_filled, next_points, (None, *next_assignments))
        for player_index, player in enumerate(candidates):
            bit = 1 << player_index
            if used_mask & bit or not player.can_fill(slot):
                continue
            filled, points, assignments = solve(index + 1, used_mask | bit)
            option = (
                filled + 1,
                points + float(player.projected_points or 0),
                (player_index, *assignments),
            )
            if option[:2] > best[:2]:
                best = option
        return best

    _, _, selected = solve(0, 0)
    by_slot = {
        slot.slot_id: (candidates[player_index] if player_index is not None else None)
        for slot, player_index in zip(ordered_slots, selected, strict=True)
    }
    return tuple(
        BaselineAssignment(
            slot_id=slot.slot_id,
            slot_type=slot.slot_type,
            sleeper_player_id=(
                by_slot[slot.slot_id].sleeper_player_id
                if by_slot[slot.slot_id]
                else None
            ),
            player_name=(
                by_slot[slot.slot_id].display_name
                if by_slot[slot.slot_id]
                else None
            ),
            projected_points=(
                by_slot[slot.slot_id].projected_points
                if by_slot[slot.slot_id]
                else None
            ),
        )
        for slot in slots
    )


class LineupContextBuilder:
    """Join one league roster with live weekly projections and health fields."""

    def __init__(
        self,
        *,
        league_builder: LeagueContextBuilder | None = None,
        projections: LineupProjectionSource | None = None,
        schedule: LineupScheduleSource | None = None,
    ) -> None:
        self.league_builder = league_builder or LeagueContextBuilder()
        self.projections = projections or SleeperProjectionClient()
        self.schedule = schedule or SupabaseLineupScheduleSource()

    def build(
        self,
        *,
        league_id: str,
        user_reference: str,
        week: int | None = None,
        as_of: datetime | None = None,
    ) -> LineupContext:
        selected_as_of = as_of or datetime.now(UTC)
        if selected_as_of.tzinfo is None:
            raise ValueError("as_of must include a timezone")
        selected_as_of = selected_as_of.astimezone(UTC)
        league = self.league_builder.build(
            league_id=league_id,
            user_reference=user_reference,
            week=week,
            available_limit=10,
            include_market=False,
        )
        selected_week = week or league.current_week
        schedule_error = None
        schedule_by_team: dict[str, tuple[str | None, datetime]] = {}
        try:
            week_games = self.schedule.week_games(
                season=league.season,
                week=selected_week,
            )
            if not week_games:
                raise ValueError(
                    f"No stored games found for {league.season} Week {selected_week}"
                )
            schedule_by_team = _schedule_by_team(week_games)
        except Exception as error:
            schedule_error = str(error)
        projection_error = None
        try:
            projections = self.projections.weekly_projections(
                season=league.season,
                week=selected_week,
                season_type=("post" if league.season_type == "post" else "regular"),
                positions=("QB", "RB", "WR", "TE", "K", "DEF"),
            )
        except SleeperProjectionError as error:
            projections = []
            projection_error = str(error)
        projection_by_id = {
            projection.sleeper_player_id: projection
            for projection in projections
        }

        slots = _slot_ids(league.managed_roster.starters)
        slot_by_player = {
            slot.current_player_id: slot.slot_id
            for slot in slots
            if slot.current_player_id is not None
        }
        player_rows: list[LineupPlayer] = []
        seen: set[str] = set()

        def append_group(values: tuple[LeaguePlayer, ...], group: str) -> None:
            for player in values:
                if player.sleeper_player_id in seen:
                    continue
                seen.add(player.sleeper_player_id)
                player_rows.append(
                    _lineup_player(
                        player,
                        roster_group=group,
                        current_slot_id=slot_by_player.get(player.sleeper_player_id),
                        slots=slots,
                        week=selected_week,
                        projection=projection_by_id.get(player.sleeper_player_id),
                        scoring_settings=league.scoring_settings,
                        schedule=schedule_by_team.get(_team(player.team)),
                        as_of=selected_as_of,
                    )
                )

        append_group(
            tuple(
                assignment.player
                for assignment in league.managed_roster.starters
                if assignment.player is not None
            ),
            "starter",
        )
        append_group(league.managed_roster.bench, "bench")
        append_group(league.managed_roster.reserve, "reserve")
        append_group(league.managed_roster.taxi, "taxi")
        players = tuple(player_rows)
        baseline = optimize_projected_lineup(slots, players)

        current_total = round(
            sum(
                player.projected_points or 0
                for player in players
                if player.roster_group == "starter"
            ),
            2,
        )
        baseline_total = round(
            sum(assignment.projected_points or 0 for assignment in baseline),
            2,
        )

        opponent_rows: list[dict[str, object]] = []
        opponent_total: float | None = None
        if league.matchup and league.matchup.opponent_roster_id is not None:
            opponent = next(
                (
                    roster
                    for roster in league.other_rosters
                    if roster.roster_id == league.matchup.opponent_roster_id
                ),
                None,
            )
            if opponent is not None:
                points = 0.0
                found_projection = False
                for assignment in opponent.starters:
                    if assignment.player is None:
                        continue
                    projection = projection_by_id.get(
                        assignment.player.sleeper_player_id
                    )
                    projected = (
                        projection.projected_points(league.scoring_settings)
                        if projection
                        else None
                    )
                    if projected is not None:
                        found_projection = True
                        points += projected
                    opponent_rows.append(
                        {
                            "slot": assignment.slot,
                            "name": assignment.player.display_name,
                            "position": assignment.player.position,
                            "team": _team(assignment.player.team),
                            "projected_points": projected,
                            "injury_designation": (
                                projection.injury_status if projection else None
                            ),
                        }
                    )
                if found_projection:
                    opponent_total = round(points, 2)

        return LineupContext(
            league=league,
            week=selected_week,
            as_of=selected_as_of,
            slots=slots,
            players=players,
            projection_baseline=baseline,
            current_projected_total=current_total,
            baseline_projected_total=baseline_total,
            opponent_current_projected_total=opponent_total,
            opponent_starters=tuple(opponent_rows),
            projection_error=projection_error,
            schedule_error=schedule_error,
        )
