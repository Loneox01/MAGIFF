import unittest
from datetime import date
from types import SimpleNamespace

from jobs.backfill_reports import (
    BackfillCandidate,
    backfill_reports,
    default_backfill_id,
)
from jobs.refresh_reports import RefreshResult


class FakeQuery:
    def __init__(self, rows):
        self.rows = list(rows)
        self.filters = []

    def select(self, _columns):
        return self

    def eq(self, field, value):
        self.filters.append((field, "eq", value))
        return self

    def in_(self, field, values):
        self.filters.append((field, "in", list(values)))
        return self

    def order(self, _field, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        rows = self.rows
        for field, operator, wanted in self.filters:
            if operator == "eq":
                rows = [row for row in rows if row.get(field) == wanted]
            else:
                rows = [row for row in rows if row.get(field) in wanted]
        return SimpleNamespace(data=rows[: getattr(self, "limit_value", len(rows))])


class FakeClient:
    def __init__(self, runs=None):
        self.runs = runs or []

    def table(self, name):
        if name != "report_ingestion_runs":
            raise AssertionError(f"Unexpected table: {name}")
        return FakeQuery(self.runs)


def candidates(_client, _season, _scoring, _league, _limit):
    return [
        BackfillCandidate("101", "player-1", "First Player", "current_ecr"),
        BackfillCandidate("102", "player-2", "Second Player", "current_ecr"),
        BackfillCandidate(
            "103",
            "player-3",
            "Third Player",
            "prior_season_production",
        ),
    ]


def refresh_result(*, status="succeeded", reason=None, new_reports=3):
    return RefreshResult(
        run_id="run-id",
        status=status,
        reason=reason,
        requested_reports=100,
        requests_used_last_24_hours=8,
        daily_request_budget=40,
        provider_items_received=4,
        eligible_reports=3,
        date_filtered_reports=1,
        new_reports=new_reports,
        changed_reports=0,
        unchanged_reports=0,
        failed_reports=0,
        metadata_input_tokens=100,
        metadata_cached_input_tokens=0,
        metadata_output_tokens=20,
        generated_embeddings=3,
        reused_embeddings=0,
        oldest_published_at="2026-01-10T00:00:00+00:00",
        newest_published_at="2026-08-01T00:00:00+00:00",
        feed_window_saturated=False,
        possible_coverage_gap=False,
    )


class ReportBackfillTests(unittest.TestCase):
    def test_plan_resumes_after_successful_player_feed(self):
        cutoff = date(2026, 1, 1)
        backfill_id = default_backfill_id(cutoff)
        client = FakeClient(
            [
                {
                    "provider": "fantasypros",
                    "trigger": "backfill_player_news",
                    "status": "succeeded",
                    "new_reports": 5,
                    "changed_reports": 0,
                    "metadata": {
                        "backfill_id": backfill_id,
                        "request_fpid": "101",
                    },
                }
            ]
        )

        result = backfill_reports(
            api_key="",
            cutoff_from=cutoff,
            cutoff_to=date(2026, 8, 22),
            max_requests=2,
            plan_only=True,
            client=client,
            candidate_loader=candidates,
        )

        self.assertEqual(result.status, "planned")
        self.assertEqual(result.new_reports_before_run, 5)
        self.assertEqual(
            [row["fantasypros_id"] for row in result.processed_players],
            ["102", "103"],
        )

    def test_live_chunk_never_exceeds_max_requests(self):
        calls = []

        def fake_refresh(**kwargs):
            calls.append(kwargs)
            return refresh_result()

        result = backfill_reports(
            api_key="test-key",
            cutoff_from=date(2026, 1, 1),
            cutoff_to=date(2026, 8, 22),
            target_new_reports=20,
            max_requests=2,
            request_delay_seconds=0,
            client=FakeClient(),
            candidate_loader=candidates,
            refresh_function=fake_refresh,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result.provider_requests_made, 2)
        self.assertEqual(result.new_reports_after_run, 6)
        self.assertEqual(result.status, "in_progress")
        self.assertEqual(calls[0]["fpid"], "101")
        self.assertEqual(calls[0]["published_from"], date(2026, 1, 1))
        self.assertEqual(calls[0]["daily_request_budget"], 40)
        self.assertFalse(calls[0]["emit_result"])

    def test_budget_skip_pauses_without_counting_a_request(self):
        result = backfill_reports(
            api_key="test-key",
            cutoff_from=date(2026, 1, 1),
            cutoff_to=date(2026, 8, 22),
            max_requests=2,
            request_delay_seconds=0,
            client=FakeClient(),
            candidate_loader=candidates,
            refresh_function=lambda **_kwargs: refresh_result(
                status="skipped",
                reason="daily_budget_exhausted",
                new_reports=0,
            ),
        )

        self.assertEqual(result.status, "paused")
        self.assertEqual(result.provider_requests_made, 0)
        self.assertEqual(result.reason, "daily_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
