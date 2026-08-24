"""Supabase persistence for the deterministic roster game."""

from __future__ import annotations

from database.client import get_supabase_client
from services.roster_game import (
    GameAction,
    GamePick,
    GameStatus,
    PlayerPoolEntry,
    RolledPlayer,
    RosterGameState,
    RosterSlot,
)


def _player_payload(player: PlayerPoolEntry) -> dict[str, object]:
    return {
        "player_id": player.player_id,
        "display_name": player.display_name,
        "team": player.team,
        "position": player.position,
        "fantasy_points_ppr": player.fantasy_points_ppr,
        "team_name": player.team_name,
        "team_logo_url": player.team_logo_url,
        "team_color": player.team_color,
    }


def _player_from_payload(payload: dict) -> PlayerPoolEntry:
    return PlayerPoolEntry(
        player_id=str(payload["player_id"]),
        display_name=str(payload["display_name"]),
        team=str(payload["team"]),
        position=str(payload["position"]),
        fantasy_points_ppr=float(payload["fantasy_points_ppr"]),
        team_name=str(payload.get("team_name") or payload["team"]),
        team_logo_url=(
            None
            if payload.get("team_logo_url") is None
            else str(payload["team_logo_url"])
        ),
        team_color=(
            None if payload.get("team_color") is None else str(payload["team_color"])
        ),
    )


def _rolled_payload(rolled: RolledPlayer | None) -> dict[str, object] | None:
    if rolled is None:
        return None
    return {
        "roster_slot": rolled.roster_slot.value,
        "player": _player_payload(rolled.player),
    }


def _rolled_from_payload(payload: dict | None) -> RolledPlayer | None:
    if not payload:
        return None
    return RolledPlayer(
        RosterSlot(str(payload["roster_slot"])),
        _player_from_payload(dict(payload["player"])),
    )


def _pick_payload(pick: GamePick) -> dict[str, object]:
    return {
        "pick_number": pick.pick_number,
        "roster_slot": pick.roster_slot.value,
        **_player_payload(pick.player),
    }


class SupabaseRosterGameRepository:
    def __init__(self, client=None) -> None:
        self.client = client or get_supabase_client()

    def latest_completed_season(self) -> int | None:
        rows = (
            self.client.table("player_weekly_stats")
            .select("season")
            .eq("season_type", "REG")
            .eq("week", 18)
            .order("season", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return int(rows[0]["season"]) if rows else None

    def season_is_complete(self, season: int) -> bool:
        rows = (
            self.client.table("player_weekly_stats")
            .select("game_id")
            .eq("season", season)
            .eq("season_type", "REG")
            .eq("week", 18)
            .limit(1)
            .execute()
            .data
        )
        return bool(rows)

    def player_pool(self, season: int) -> list[PlayerPoolEntry]:
        rows = self.client.rpc(
            "get_roster_game_pool",
            {"target_season": season},
        ).execute().data
        return [
            PlayerPoolEntry(
                player_id=str(row["player_id"]),
                display_name=str(row["display_name"]),
                team=str(row["team"]),
                position=str(row["player_position"]),
                fantasy_points_ppr=float(row["fantasy_points_ppr"]),
                team_name=str(row.get("team_name") or row["team"]),
                team_logo_url=(
                    None
                    if row.get("team_logo_url") is None
                    else str(row["team_logo_url"])
                ),
                team_color=(
                    None
                    if row.get("team_color") is None
                    else str(row["team_color"])
                ),
            )
            for row in rows
        ]

    def ensure_user(self, discord_user_id: str, display_name: str | None) -> str:
        app_user_id = self.client.rpc(
            "ensure_app_user_identity",
            {
                "p_provider": "discord",
                "p_provider_user_id": discord_user_id,
                "p_display_name": display_name,
            },
        ).execute().data
        if not app_user_id:
            raise RuntimeError("Supabase did not return an application user ID")
        return str(app_user_id)

    def create_game(self, state: RosterGameState) -> None:
        self.client.table("roster_game_sessions").insert(
            {
                "game_id": state.game_id,
                "app_user_id": state.app_user_id,
                "discord_user_id": state.discord_user_id,
                "discord_guild_id": state.discord_guild_id,
                "season": state.season,
                "reveal_during_roll": state.reveal_during_roll,
                "status": state.status.value,
                "pending_pick": _rolled_payload(state.pending),
                "team_reroll_used": state.team_reroll_used,
                "position_reroll_used": state.position_reroll_used,
                "version": state.version,
            }
        ).execute()

    def get_game(self, game_id: str) -> RosterGameState | None:
        rows = (
            self.client.table("roster_game_sessions")
            .select("*")
            .eq("game_id", game_id)
            .limit(1)
            .execute()
            .data
        )
        if not rows:
            return None
        row = rows[0]
        pick_rows = (
            self.client.table("roster_game_picks")
            .select("*")
            .eq("game_id", game_id)
            .order("pick_number")
            .execute()
            .data
        )
        picks = tuple(
            GamePick(
                pick_number=int(pick["pick_number"]),
                roster_slot=RosterSlot(str(pick["roster_slot"])),
                player=_player_from_payload(pick),
            )
            for pick in pick_rows
        )
        return RosterGameState(
            game_id=str(row["game_id"]),
            app_user_id=str(row["app_user_id"]),
            discord_user_id=str(row["discord_user_id"]),
            discord_guild_id=str(row["discord_guild_id"]),
            season=int(row["season"]),
            reveal_during_roll=bool(row["reveal_during_roll"]),
            status=GameStatus(str(row["status"])),
            pending=_rolled_from_payload(row.get("pending_pick")),
            picks=picks,
            team_reroll_used=bool(row["team_reroll_used"]),
            position_reroll_used=bool(row["position_reroll_used"]),
            total_points=(
                None
                if row.get("total_points") is None
                else float(row["total_points"])
            ),
            wins=None if row.get("wins") is None else int(row["wins"]),
            losses=None if row.get("losses") is None else int(row["losses"]),
            version=int(row["version"]),
        )

    def save_transition(
        self,
        state: RosterGameState,
        *,
        expected_version: int,
        new_pick: GamePick | None,
        interaction_id: str,
        action: GameAction,
    ) -> bool:
        payload = self.client.rpc(
            "apply_roster_game_transition",
            {
                "p_game_id": state.game_id,
                "p_app_user_id": state.app_user_id,
                "p_expected_version": expected_version,
                "p_interaction_id": interaction_id,
                "p_action": action.value,
                "p_pending_pick": _rolled_payload(state.pending),
                "p_team_reroll_used": state.team_reroll_used,
                "p_position_reroll_used": state.position_reroll_used,
                "p_status": state.status.value,
                "p_total_points": state.total_points,
                "p_wins": state.wins,
                "p_losses": state.losses,
                "p_new_pick": None if new_pick is None else _pick_payload(new_pick),
            },
        ).execute().data
        return bool(payload)
