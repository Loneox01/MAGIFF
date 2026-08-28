import unittest
from types import SimpleNamespace

from main import run_web_only


class WebOnlyMainTests(unittest.TestCase):
    def test_web_only_run_exposes_no_local_tools_and_keeps_usage_logging(self) -> None:
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text="A cited web answer.",
                output=[
                    SimpleNamespace(type="web_search_call"),
                    SimpleNamespace(type="message"),
                ],
                usage=SimpleNamespace(
                    input_tokens=1000,
                    output_tokens=100,
                    input_tokens_details=SimpleNamespace(cached_tokens=200),
                ),
            )

        result = run_web_only(
            "What is the latest NFL news?",
            client=SimpleNamespace(
                responses=SimpleNamespace(create=create)
            ),
            model="gpt-5.6-terra",
        )

        self.assertEqual(captured["tools"], [{"type": "web_search"}])
        self.assertNotIn("tool_choice", captured)
        self.assertEqual(result.answer, "A cited web answer.")
        self.assertEqual(result.web_search_calls, 1)
        self.assertEqual(result.usage.input_tokens, 1000)
        self.assertEqual(result.usage.cached_input_tokens, 200)
        self.assertEqual(result.usage.output_tokens, 100)
        self.assertAlmostEqual(result.estimated_cost_usd, 0.00284)


if __name__ == "__main__":
    unittest.main()
