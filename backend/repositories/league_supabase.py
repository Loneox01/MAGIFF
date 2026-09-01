"""Supabase identity enrichment for public fantasy-league snapshots."""

from __future__ import annotations

from database.client import get_supabase_client
from repositories import nfl_supabase


class SupabaseLeaguePlayerRepository:
    """Resolve Sleeper identifiers without treating unknown IDs as identities."""

    def resolve_players(
        self,
        sleeper_player_ids: list[str],
    ) -> dict[str, dict]:
        external_ids = [
            value
            for value in dict.fromkeys(str(item) for item in sleeper_player_ids)
            if value and value != "0"
        ]
        if not external_ids:
            return {}

        client = get_supabase_client()
        links = (
            client.table("player_external_ids")
            .select("player_id,external_id")
            .eq("provider", "sleeper")
            .in_("external_id", external_ids)
            .execute()
            .data
        )
        internal_ids = list(
            dict.fromkeys(
                str(row["player_id"])
                for row in links
                if row.get("player_id")
            )
        )
        profiles = []
        statuses = []
        if internal_ids:
            profiles = (
                client.table("players")
                .select("player_id,display_name,position,position_group")
                .in_("player_id", internal_ids)
                .execute()
                .data
            )
            statuses = (
                client.table("player_status")
                .select("player_id,latest_team,status")
                .in_("player_id", internal_ids)
                .execute()
                .data
            )

        profile_by_id = {
            str(row["player_id"]): row for row in profiles if row.get("player_id")
        }
        status_by_id = {
            str(row["player_id"]): row for row in statuses if row.get("player_id")
        }
        resolved: dict[str, dict] = {}
        for link in links:
            player_id = str(link.get("player_id") or "")
            external_id = str(link.get("external_id") or "")
            profile = profile_by_id.get(player_id)
            if not external_id or profile is None:
                continue
            status = status_by_id.get(player_id, {})
            resolved[external_id] = {
                "sleeper_player_id": external_id,
                "player_id": player_id,
                "display_name": profile.get("display_name") or external_id,
                "position": profile.get("position") or profile.get("position_group"),
                "team": status.get("latest_team"),
                "roster_status": status.get("status"),
            }

        defense_codes = [
            value
            for value in external_ids
            if value not in resolved
            and value.isalpha()
            and value == value.upper()
            and 2 <= len(value) <= 4
        ]
        if defense_codes:
            teams = (
                client.table("teams")
                .select("team_abbr,team_name")
                .in_("team_abbr", defense_codes)
                .execute()
                .data
            )
            for team in teams:
                abbreviation = str(team.get("team_abbr") or "")
                if not abbreviation:
                    continue
                team_name = str(team.get("team_name") or abbreviation)
                resolved[abbreviation] = {
                    "sleeper_player_id": abbreviation,
                    "player_id": None,
                    "display_name": f"{team_name} D/ST",
                    "position": "DEF",
                    "team": abbreviation,
                    "roster_status": "ACT",
                }
        return resolved


class SupabaseWaiverPlayerRepository:
    """Resolve deep named and team-position candidates beyond market leaders."""

    @staticmethod
    def _with_sleeper_ids(rows: list[dict]) -> list[dict]:
        player_ids = [
            str(row["player_id"]) for row in rows if row.get("player_id")
        ]
        sleeper_ids = nfl_supabase.get_player_external_ids(
            player_ids,
            "sleeper",
        )
        return [
            {
                "player_id": str(row["player_id"]),
                "sleeper_player_id": sleeper_ids.get(str(row["player_id"])),
                "display_name": row.get("display_name") or row.get("player_name"),
                "position": row.get("position"),
                "team": row.get("latest_team") or row.get("team"),
                "roster_status": row.get("status"),
            }
            for row in rows
            if row.get("player_id")
        ]

    def find_players(self, name: str) -> list[dict]:
        return self._with_sleeper_ids(nfl_supabase.find_players(name))

    def team_position_players(
        self,
        *,
        team: str,
        position: str,
        season: int,
    ) -> list[dict]:
        rows = nfl_supabase.get_current_team_roster(
            team,
            season,
            position,
            None,
        )
        return self._with_sleeper_ids(rows)
