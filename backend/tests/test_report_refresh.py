import json
import unittest
from datetime import date
from types import SimpleNamespace

from ingestion.reports.fantasypros import content_hash
from jobs.refresh_reports import classify_feed, refresh_reports
from processing.reports.fantasypros import (
    DocumentType,
    ExtractedReportBatch,
    ExtractedReportMetadata,
    SupabasePlayerCatalog,
)


PLAYER_ID = "11111111-1111-5111-8111-111111111111"


def report_item(external_id: int, *, title: str = "Player update") -> dict:
    return {
        "id": external_id,
        "created": "2026-08-21 14:52:01",
        "player_id": 22973,
        "team_id": "ATL",
        "title": title,
        "desc": "Description",
        "impact": "Impact",
        "link": f"https://example.com/{external_id}",
        "categories": ["News"],
    }


class FakeQuery:
    def __init__(self, client, table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self.operation = "select"
        self.values = None
        self.filters = {}

    def select(self, _columns):
        self.operation = "select"
        return self

    def update(self, values):
        self.operation = "update"
        self.values = values
        return self

    def eq(self, field, value):
        self.filters[field] = ("eq", value)
        return self

    def in_(self, field, values):
        self.filters[field] = ("in", list(values))
        return self

    def ilike(self, field, value):
        self.filters[field] = ("ilike", value.strip("%"))
        return self

    def limit(self, _value):
        return self

    def execute(self):
        if self.operation == "update":
            self.client.updates.append(
                (self.table_name, self.values, dict(self.filters))
            )
            return SimpleNamespace(data=[])

        rows = list(self.client.tables.get(self.table_name, []))
        for field, (operator, wanted) in self.filters.items():
            if operator == "eq":
                rows = [row for row in rows if row.get(field) == wanted]
            elif operator == "in":
                rows = [row for row in rows if row.get(field) in wanted]
            elif operator == "ilike":
                rows = [
                    row
                    for row in rows
                    if wanted.casefold() in str(row.get(field, "")).casefold()
                ]
        return SimpleNamespace(data=rows)


class FakeRpc:
    def __init__(self, client, name: str, params: dict) -> None:
        self.client = client
        self.name = name
        self.params = params

    def execute(self):
        self.client.rpc_calls.append((self.name, self.params))
        if self.name == "reserve_report_ingestion_run":
            return SimpleNamespace(
                data={
                    "acquired": True,
                    "run_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "requests_used_last_24_hours": 7,
                    "daily_request_budget": 40,
                }
            )
        if self.name == "upsert_report_document":
            return SimpleNamespace(data={"version_inserted": True})
        return SimpleNamespace(data={"status": self.params["p_status"]})


class FakeClient:
    def __init__(self, tables=None) -> None:
        self.tables = tables or {}
        self.updates = []
        self.rpc_calls = []

    def table(self, name: str):
        return FakeQuery(self, name)

    def rpc(self, name: str, params: dict):
        return FakeRpc(self, name, params)


class FakeResponses:
    def parse(self, **kwargs):
        payload = kwargs["input"][1]["content"]
        external_id = json.loads(payload)["reports"][0]["external_id"]
        return SimpleNamespace(
            output_parsed=ExtractedReportBatch(
                reports=[
                    ExtractedReportMetadata(
                        external_id=external_id,
                        document_type=DocumentType.GENERAL_NEWS,
                        document_type_confidence=0.9,
                        player_mentions=[],
                    )
                ]
            ),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


class ReportRefreshTests(unittest.TestCase):
    def test_classify_feed_flags_a_full_all_new_window(self) -> None:
        items = [report_item(1), report_item(2)]
        diff = classify_feed(
            {"items": items},
            {},
            requested_limit=2,
        )

        self.assertEqual(len(diff.new_items), 2)
        self.assertTrue(diff.feed_window_saturated)
        self.assertTrue(diff.possible_coverage_gap)
        self.assertEqual(diff.oldest_published_at, "2026-08-21T14:52:01+00:00")

    def test_classify_feed_flags_an_all_new_provider_capped_window(self) -> None:
        items = [report_item(index) for index in range(1, 11)]
        diff = classify_feed(
            {"count": 10, "items": items},
            {},
            requested_limit=20,
            source_received=10,
        )

        self.assertEqual(diff.received, 10)
        self.assertEqual(len(diff.new_items), 10)
        self.assertTrue(diff.feed_window_saturated)
        self.assertTrue(diff.possible_coverage_gap)

    def test_classify_feed_accepts_an_underfilled_window_with_overlap(self) -> None:
        old = report_item(1)
        new = report_item(2)
        diff = classify_feed(
            {"count": 2, "items": [new, old]},
            {"1": content_hash(old)},
            requested_limit=20,
            source_received=2,
        )

        self.assertEqual(len(diff.new_items), 1)
        self.assertEqual(len(diff.unchanged_items), 1)
        self.assertFalse(diff.feed_window_saturated)
        self.assertFalse(diff.possible_coverage_gap)

    def test_unchanged_refresh_skips_model_and_embedding_work(self) -> None:
        item = report_item(603602)
        client = FakeClient(
            {
                "reports": [
                    {
                        "provider": "fantasypros",
                        "external_id": "603602",
                        "source_content_hash": content_hash(item),
                    }
                ]
            }
        )

        result = refresh_reports(
            api_key="test-key",
            report_limit=20,
            client=client,
            fetcher=lambda *_args, **_kwargs: {"items": [item]},
        )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.unchanged_reports, 1)
        self.assertEqual(result.metadata_input_tokens, 0)
        self.assertEqual(result.generated_embeddings, 0)
        self.assertEqual(
            [name for name, _ in client.rpc_calls],
            ["reserve_report_ingestion_run", "finish_report_ingestion_run"],
        )
        self.assertEqual(len(client.updates), 1)

    def test_backfill_date_bounds_filter_before_model_work(self) -> None:
        old = report_item(1)
        old["created"] = "2025-12-31 23:59:59"
        eligible = report_item(2)
        eligible["created"] = "2026-01-01 00:00:00"
        future = report_item(3)
        future["created"] = "2026-08-23 00:00:00"
        client = FakeClient(
            {
                "reports": [
                    {
                        "provider": "fantasypros",
                        "external_id": "2",
                        "source_content_hash": content_hash(eligible),
                    }
                ]
            }
        )

        result = refresh_reports(
            api_key="test-key",
            report_limit=100,
            published_from=date(2026, 1, 1),
            published_to=date(2026, 8, 22),
            client=client,
            fetcher=lambda *_args, **_kwargs: {
                "items": [old, eligible, future]
            },
        )

        self.assertEqual(result.provider_items_received, 3)
        self.assertEqual(result.eligible_reports, 1)
        self.assertEqual(result.date_filtered_reports, 2)
        self.assertEqual(result.unchanged_reports, 1)
        self.assertEqual(result.metadata_input_tokens, 0)

    def test_supabase_catalog_batches_provider_ids_and_caches_names(self) -> None:
        client = FakeClient(
            {
                "player_external_ids": [
                    {
                        "provider": "fantasypros",
                        "external_id": "22973",
                        "player_id": PLAYER_ID,
                    }
                ],
                "players": [
                    {
                        "player_id": PLAYER_ID,
                        "display_name": "Michael Penix Jr.",
                        "position": "QB",
                        "position_group": "QB",
                    }
                ],
            }
        )
        catalog = SupabasePlayerCatalog.from_external_ids(client, ["22973"])

        primary = catalog.primary_player("22973")
        first = catalog.name_matches("Michael Penix Jr.")
        second = catalog.name_matches("Michael Penix Jr.")

        self.assertIsNotNone(primary)
        self.assertEqual(primary.player_id, PLAYER_ID)
        self.assertEqual(first, second)
        self.assertEqual(first[0].display_name, "Michael Penix Jr.")

    def test_new_report_runs_the_complete_temporary_pipeline(self) -> None:
        item = report_item(603602)
        client = FakeClient(
            {
                "player_external_ids": [
                    {
                        "provider": "fantasypros",
                        "external_id": "22973",
                        "player_id": PLAYER_ID,
                    }
                ],
                "players": [
                    {
                        "player_id": PLAYER_ID,
                        "display_name": "Michael Penix Jr.",
                        "position": "QB",
                        "position_group": "QB",
                    }
                ],
            }
        )
        openai_client = SimpleNamespace(responses=FakeResponses())

        result = refresh_reports(
            api_key="test-key",
            report_limit=20,
            client=client,
            openai_client=openai_client,
            fetcher=lambda *_args, **_kwargs: {"items": [item]},
            with_embeddings=False,
        )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.new_reports, 1)
        self.assertEqual(result.metadata_input_tokens, 100)
        self.assertIn(
            "upsert_report_document",
            [name for name, _params in client.rpc_calls],
        )

    def test_daily_budget_can_never_exceed_provider_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "50-request provider cap"):
            refresh_reports(
                api_key="test-key",
                daily_request_budget=51,
                client=FakeClient(),
            )


if __name__ == "__main__":
    unittest.main()
