import unittest

from orchestration.player_references import (
    PlayerReferenceAdapter,
    PlayerReferenceResolver,
)


PLAYER_ID = "11111111-1111-5111-8111-111111111111"


class PlayerReferenceResolverTests(unittest.TestCase):
    def test_uuid_bypasses_database_lookup(self) -> None:
        resolver = PlayerReferenceResolver(
            find_players=lambda _name: self.fail("database should not be called")
        )

        result = resolver.resolve(PLAYER_ID.upper())

        self.assertEqual(result.player_id, PLAYER_ID)
        self.assertEqual(result.resolution_basis, "uuid")

    def test_punctuation_variant_uses_surname_candidates_then_exact_match(
        self,
    ) -> None:
        calls = []

        def find_players(name):
            calls.append(name)
            if name == "AJ Brown":
                return []
            return [
                {
                    "player_id": PLAYER_ID,
                    "display_name": "A.J. Brown",
                    "position": "WR",
                    "latest_team": "PHI",
                    "status": "ACT",
                },
                {
                    "player_id": "22222222-2222-5222-8222-222222222222",
                    "display_name": "Chase Brown",
                    "position": "RB",
                    "latest_team": "CIN",
                    "status": "ACT",
                },
            ]

        result = PlayerReferenceResolver(find_players=find_players).resolve(
            "AJ Brown"
        )

        self.assertEqual(calls, ["AJ Brown", "brown"])
        self.assertEqual(result.player_id, PLAYER_ID)
        self.assertEqual(result.display_name, "A.J. Brown")

    def test_missing_suffix_resolves_unique_canonical_name(self) -> None:
        resolver = PlayerReferenceResolver(
            find_players=lambda _name: [
                {
                    "player_id": PLAYER_ID,
                    "display_name": "Michael Penix Jr.",
                }
            ]
        )

        result = resolver.resolve("Michael Penix")

        self.assertEqual(result.player_id, PLAYER_ID)
        self.assertEqual(result.resolution_basis, "suffix_normalized_name")

    def test_list_adapter_accepts_names_and_uuids_and_deduplicates(self) -> None:
        adapter = PlayerReferenceAdapter(
            PlayerReferenceResolver(
                find_players=lambda _name: [
                    {
                        "player_id": PLAYER_ID,
                        "display_name": "A.J. Brown",
                    }
                ]
            )
        )
        cache = {}
        adapter.resolve_many(
            ["A.J. Brown", PLAYER_ID],
            cache=cache,
            max_workers=2,
        )

        adapted = adapter.adapt(
            "rank_players_by_weekly_threshold",
            {"player_refs": ["A.J. Brown", PLAYER_ID]},
            cache=cache,
        )

        self.assertEqual(adapted["player_ids"], [PLAYER_ID])
        self.assertNotIn("player_refs", adapted)

    def test_adapter_retries_transient_database_resolution_once(self) -> None:
        attempts = 0

        def find_players(_name):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("JWT issued at future (PGRST303)")
            return [
                {
                    "player_id": PLAYER_ID,
                    "display_name": "A.J. Brown",
                }
            ]

        adapter = PlayerReferenceAdapter(
            PlayerReferenceResolver(find_players=find_players)
        )
        cache = {}

        adapter.resolve_many(
            ["A.J. Brown"],
            cache=cache,
            max_workers=1,
        )

        self.assertEqual(attempts, 2)
        self.assertEqual(cache["a.j. brown"].player_id, PLAYER_ID)


if __name__ == "__main__":
    unittest.main()
