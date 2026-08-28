import unittest

from model_costs import (
    estimate_component_costs_usd,
    estimate_text_token_cost_usd,
)


class ModelCostTests(unittest.TestCase):
    def test_terra_cost_separates_cached_and_uncached_input(self) -> None:
        cost = estimate_text_token_cost_usd(
            model="gpt-5.6-terra",
            input_tokens=10_000,
            cached_input_tokens=4_000,
            output_tokens=1_000,
        )

        self.assertAlmostEqual(cost, 0.0248)

    def test_family_alias_and_snapshot_names_are_supported(self) -> None:
        sol = estimate_text_token_cost_usd(
            model="gpt-5.6",
            input_tokens=1_000,
            cached_input_tokens=0,
            output_tokens=100,
        )
        luna = estimate_text_token_cost_usd(
            model="gpt-5.6-luna-2026-08-01",
            input_tokens=1_000,
            cached_input_tokens=0,
            output_tokens=100,
        )

        self.assertAlmostEqual(sol, 0.006)
        self.assertAlmostEqual(luna, 0.00032)

    def test_component_cost_uses_each_models_own_rate(self) -> None:
        cost = estimate_component_costs_usd(
            [
                {
                    "model": "gpt-5.6-luna",
                    "input_tokens": 1_000,
                    "cached_input_tokens": 0,
                    "output_tokens": 100,
                },
                {
                    "model": "gpt-5.6-terra",
                    "input_tokens": 1_000,
                    "cached_input_tokens": 0,
                    "output_tokens": 100,
                },
            ]
        )

        self.assertAlmostEqual(cost, 0.00352)

    def test_unknown_used_model_returns_unavailable(self) -> None:
        self.assertIsNone(
            estimate_component_costs_usd(
                [
                    {
                        "model": "unknown",
                        "input_tokens": 1,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                    }
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
