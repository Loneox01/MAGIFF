import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from ingestion.reports.fantasypros import ingest_payload


class ReportIngestionTests(unittest.TestCase):
    def test_fantasypros_raw_ingestion_is_idempotent_and_versioned(self) -> None:
        payload = {
            "sport": "NFL",
            "count": 2,
            "items": [
                {
                    "id": 1001,
                    "created": "2026-08-21 12:00:00",
                    "team_id": "SEA",
                    "title": "Example role update",
                    "link": "https://example.com/news/1001",
                    "desc": "An example fantasy impact summary.",
                },
                {
                    "id": 1002,
                    "created": "2026-08-21 12:05:00",
                    "team_id": "NYJ",
                    "title": "Example injury update",
                    "link": "https://example.com/news/1002",
                    "desc": "An example injury summary.",
                },
            ],
        }
        acquired_at = datetime(2026, 8, 21, 16, tzinfo=UTC)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            first = ingest_payload(
                payload,
                output_dir=output_dir,
                fetched_at=acquired_at,
                requested_limit=25,
            )
            second = ingest_payload(
                payload,
                output_dir=output_dir,
                fetched_at=acquired_at,
                requested_limit=25,
            )

            changed = json.loads(json.dumps(payload))
            changed["items"][0]["desc"] = "A corrected fantasy impact summary."
            third = ingest_payload(
                changed,
                output_dir=output_dir,
                fetched_at=acquired_at,
                requested_limit=25,
            )

            self.assertEqual((first.inserted, first.updated), (2, 0))
            self.assertEqual(second.unchanged, 2)
            self.assertEqual((third.updated, third.unchanged), (1, 1))
            versions = list((output_dir / "versions" / "1001").glob("*.json"))
            self.assertEqual(len(versions), 2)
            latest = json.loads(
                (output_dir / "items" / "1001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                latest["payload"]["desc"],
                "A corrected fantasy impact summary.",
            )
            run = json.loads(
                (output_dir / "latest_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run["updated"], 1)
            self.assertEqual(run["unchanged"], 1)


if __name__ == "__main__":
    unittest.main()
