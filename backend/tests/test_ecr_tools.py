import unittest
from unittest.mock import patch

from tools import nfl


class PlayerEcrToolTests(unittest.TestCase):
    @patch.object(nfl.repository, "get_player_names")
    @patch.object(nfl.repository, "get_player_ecr_row")
    def test_get_player_ecr_returns_exact_snapshot_row(
        self,
        get_player_ecr_row,
        get_player_names,
    ) -> None:
        get_player_ecr_row.return_value = (
            "2025-08-29",
            {
                "player_id": "player-1",
                "position": "WR",
                "team": "PHI",
                "overall_rank": 14.7,
                "position_rank": 8,
                "best_rank": 10,
                "worst_rank": 21,
                "rank_sd": 2.5,
                "rank_range": 11,
                "rank_delta": -1,
                "source": "FantasyPros",
                "ranking_page": "https://example.com/ecr",
            },
        )
        get_player_names.return_value = {"player-1": "A.J. Brown"}

        result = nfl.get_player_ecr(
            "player-1",
            2025,
            "ppr",
            "redraft_1qb",
            "final_preseason",
            None,
        )

        self.assertTrue(result["found"])
        self.assertEqual(result["scrape_date"], "2025-08-29")
        self.assertEqual(result["player"]["display_name"], "A.J. Brown")
        self.assertEqual(result["player"]["overall_rank"], 14.7)
        self.assertEqual(result["player"]["position_rank"], 8)

    @patch.object(nfl.repository, "get_player_names", return_value={})
    @patch.object(nfl.repository, "get_player_ecr_row", return_value=(None, None))
    def test_get_player_ecr_explains_missing_snapshot(self, *_mocks) -> None:
        result = nfl.get_player_ecr(
            "player-1",
            2023,
            "ppr",
            "redraft_1qb",
            "final_preseason",
            None,
        )

        self.assertFalse(result["found"])
        self.assertIn("No qualifying ECR snapshot", result["reason"])


if __name__ == "__main__":
    unittest.main()
