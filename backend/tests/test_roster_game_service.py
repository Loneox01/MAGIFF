import random
import unittest
from dataclasses import replace

from services.roster_game import (
    GameAction,
    GameOutcome,
    GamePick,
    GameScoringMode,
    GameStatus,
    PlayerPoolEntry,
    RecordScale,
    RosterSlot,
    RosterGameService,
    RosterGameState,
    build_record_scale,
    legal_score_extrema,
    wins_for_score,
)


class FakeRosterGameRepository:
    def __init__(self) -> None:
        self.games: dict[str, RosterGameState] = {}
        self.interactions: set[str] = set()
        self.pool = [
            PlayerPoolEntry(
                player_id=f"{team}-{position}",
                display_name=f"{team} {position}",
                team=team,
                position=position,
                fantasy_points_ppr=100 + team_number * 20 + position_number,
                team_name=f"Team {team}",
                team_logo_url=f"https://example.com/{team}.png",
                team_color="112233",
            )
            for team_number in range(32)
            for team in (f"T{team_number:02d}",)
            for position_number, position in enumerate(("QB", "RB", "WR", "TE"))
        ]

    def latest_completed_season(self):
        return 2025

    def player_pool(self, season, scoring_mode):
        return list(self.pool) if season == 2025 else []

    def season_is_complete(self, season):
        return season == 2025

    def ensure_user(self, discord_user_id, display_name):
        return f"user-{discord_user_id}"

    def create_game(self, state):
        self.games[state.game_id] = state

    def get_game(self, game_id):
        return self.games.get(game_id)

    def save_transition(
        self,
        state,
        *,
        expected_version,
        new_pick,
        interaction_id,
        action,
    ):
        current = self.games[state.game_id]
        if current.version != expected_version or interaction_id in self.interactions:
            return False
        self.interactions.add(interaction_id)
        self.games[state.game_id] = state
        return True


class RosterGameServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = FakeRosterGameRepository()
        self.service = RosterGameService(
            self.repository,
            rng=random.Random(7),
        )

    def start(self):
        result = self.service.start(
            discord_user_id="123",
            display_name="Tester",
            discord_guild_id="guild",
        )
        self.assertEqual(result.outcome, GameOutcome.READY)
        self.assertIsNotNone(result.state)
        return result.state

    def test_completes_seven_unique_slots_teams_and_players(self) -> None:
        state = self.start()

        for index in range(7):
            if state.pending.roster_slot == RosterSlot.FLEX:
                position = state.pending.player.position
                occupied_slots = {pick.roster_slot for pick in state.picks}
                required_slots = {
                    "RB": {RosterSlot.RB1, RosterSlot.RB2},
                    "WR": {RosterSlot.WR1, RosterSlot.WR2},
                    "TE": {RosterSlot.TE},
                }[position]
                self.assertTrue(required_slots.issubset(occupied_slots))
            result = self.service.act(
                game_id=state.game_id,
                expected_version=state.version,
                discord_user_id="123",
                interaction_id=f"lock-{index}",
                action=GameAction.LOCK,
            )
            state = result.state
            self.assertIsNotNone(state)

        self.assertEqual(result.outcome, GameOutcome.COMPLETED)
        self.assertEqual(len(state.picks), 7)
        self.assertEqual(len({pick.roster_slot for pick in state.picks}), 7)
        self.assertEqual(len({pick.player.team for pick in state.picks}), 7)
        self.assertEqual(len({pick.player.player_id for pick in state.picks}), 7)
        self.assertEqual(state.wins + state.losses, 17)

    def test_forfeit_abandons_run_and_clears_pending_roll(self) -> None:
        state = self.start()
        locked = self.service.act(
            game_id=state.game_id,
            expected_version=state.version,
            discord_user_id="123",
            interaction_id="lock-before-forfeit",
            action=GameAction.LOCK,
        ).state

        result = self.service.act(
            game_id=locked.game_id,
            expected_version=locked.version,
            discord_user_id="123",
            interaction_id="forfeit",
            action=GameAction.FORFEIT,
        )

        self.assertEqual(result.outcome, GameOutcome.FORFEITED)
        self.assertEqual(result.state.status, GameStatus.ABANDONED)
        self.assertIsNone(result.state.pending)
        self.assertEqual(
            result.state.total_points,
            locked.picks[0].player.fantasy_points_ppr,
        )

    def test_flex_waits_until_the_matching_base_slots_are_full(self) -> None:
        rb_entries = [
            entry for entry in self.repository.pool if entry.position == "RB"
        ]
        rb1 = GamePick(1, RosterSlot.RB1, rb_entries[0])
        rb2 = GamePick(2, RosterSlot.RB2, rb_entries[1])

        empty = self.service._available_slot_positions(())
        one_rb = self.service._available_slot_positions((rb1,))
        two_rbs = self.service._available_slot_positions((rb1, rb2))

        self.assertNotIn((RosterSlot.FLEX, "RB"), empty)
        self.assertNotIn((RosterSlot.FLEX, "RB"), one_rb)
        self.assertIn((RosterSlot.FLEX, "RB"), two_rbs)
        self.assertNotIn((RosterSlot.FLEX, "WR"), two_rbs)
        self.assertNotIn((RosterSlot.FLEX, "TE"), two_rbs)

    def test_each_reroll_is_single_use_and_changes_visible_dimension(self) -> None:
        state = self.start()
        original_team = state.pending.player.team
        original_position = state.pending.player.position

        team_result = self.service.act(
            game_id=state.game_id,
            expected_version=state.version,
            discord_user_id="123",
            interaction_id="team-reroll",
            action=GameAction.REROLL_TEAM,
        )
        state = team_result.state
        self.assertTrue(state.team_reroll_used)
        self.assertNotEqual(state.pending.player.team, original_team)

        position_result = self.service.act(
            game_id=state.game_id,
            expected_version=state.version,
            discord_user_id="123",
            interaction_id="position-reroll",
            action=GameAction.REROLL_POSITION,
        )
        state = position_result.state
        self.assertTrue(state.position_reroll_used)
        self.assertNotEqual(state.pending.player.position, original_position)

        used_result = self.service.act(
            game_id=state.game_id,
            expected_version=state.version,
            discord_user_id="123",
            interaction_id="team-reroll-again",
            action=GameAction.REROLL_TEAM,
        )
        self.assertEqual(used_result.outcome, GameOutcome.REROLL_USED)

    def test_rejects_other_users_and_stale_versions(self) -> None:
        state = self.start()
        unauthorized = self.service.act(
            game_id=state.game_id,
            expected_version=state.version,
            discord_user_id="someone-else",
            interaction_id="unauthorized",
            action=GameAction.LOCK,
        )
        stale_state = replace(state, version=state.version + 1)
        self.repository.games[state.game_id] = stale_state
        stale = self.service.act(
            game_id=state.game_id,
            expected_version=state.version,
            discord_user_id="123",
            interaction_id="stale",
            action=GameAction.LOCK,
        )

        self.assertEqual(unauthorized.outcome, GameOutcome.NOT_OWNER)
        self.assertEqual(stale.outcome, GameOutcome.STALE)

    def test_missing_season_punts_without_creating_game(self) -> None:
        result = self.service.start(
            discord_user_id="123",
            display_name=None,
            discord_guild_id="guild",
            season=2024,
        )

        self.assertEqual(result.outcome, GameOutcome.SEASON_UNAVAILABLE)
        self.assertEqual(self.repository.games, {})

    def test_ppg_mode_is_persisted_on_the_game(self) -> None:
        result = self.service.start(
            discord_user_id="123",
            display_name="Tester",
            discord_guild_id="guild",
            scoring_mode=GameScoringMode.PPG,
        )

        self.assertEqual(result.outcome, GameOutcome.READY)
        self.assertEqual(result.state.scoring_mode, GameScoringMode.PPG)

    def test_legal_extrema_obey_roster_and_unique_team_constraints(self) -> None:
        minimum, maximum = legal_score_extrema(self.repository.pool)

        self.assertEqual(minimum, 1_130)
        self.assertEqual(maximum, 4_632)

    def test_legal_extrema_reject_a_player_repeated_across_teams(self) -> None:
        duplicate_pool = [
            replace(entry, player_id="traded-player")
            if entry.team in {"T30", "T31"}
            else entry
            for entry in self.repository.pool
        ]

        _, maximum = legal_score_extrema(duplicate_pool)

        self.assertEqual(maximum, 4_512)

    def test_season_record_scale_uses_10_and_85_percent_boundaries(self) -> None:
        scale = build_record_scale(
            self.repository.pool,
            GameScoringMode.SEASON_TOTAL,
        )

        self.assertEqual(scale.legal_minimum, 1_130)
        self.assertEqual(scale.legal_maximum, 4_632)
        self.assertEqual(scale.zero_win_cutoff, 1_475)
        self.assertEqual(scale.undefeated_cutoff, 4_100)
        self.assertEqual(len(scale.win_boundaries), 17)
        self.assertTrue(
            all(boundary % 25 == 0 for boundary in scale.win_boundaries)
        )
        self.assertEqual(wins_for_score(1_474.9, scale), 0)
        self.assertEqual(wins_for_score(1_475, scale), 1)
        self.assertEqual(wins_for_score(4_099.9, scale), 16)
        self.assertEqual(wins_for_score(4_100, scale), 17)

    def test_ppg_record_scale_rounds_boundaries_to_whole_points(self) -> None:
        ppg_pool = [
            replace(entry, fantasy_points_ppr=entry.fantasy_points_ppr / 10)
            for entry in self.repository.pool
        ]
        scale = build_record_scale(ppg_pool, GameScoringMode.PPG)

        self.assertEqual(scale.zero_win_cutoff, 148)
        self.assertEqual(scale.undefeated_cutoff, 411)
        self.assertTrue(
            all(boundary.is_integer() for boundary in scale.win_boundaries)
        )

    def test_score_mapping_uses_hard_boundaries(self) -> None:
        scale = RecordScale(
            legal_minimum=650,
            legal_maximum=2_590,
            win_boundaries=tuple(850 + index * 25 for index in range(17)),
        )

        self.assertEqual(wins_for_score(849.9, scale), 0)
        self.assertEqual(wins_for_score(850, scale), 1)
        self.assertEqual(wins_for_score(1_025, scale), 8)
        self.assertEqual(wins_for_score(1_249.9, scale), 16)
        self.assertEqual(wins_for_score(1_250, scale), 17)


if __name__ == "__main__":
    unittest.main()
