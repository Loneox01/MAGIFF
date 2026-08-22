import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from processing.reports.fantasypros import (
    DocumentType,
    ExtractedPlayerMention,
    ExtractedReportBatch,
    ExtractedReportMetadata,
    LocalPlayerCatalog,
    MentionRole,
    normalize_report,
    process_reports,
)
from rag.planning.planner import PlayerResolutionBasis


PRIMARY_ID = "11111111-1111-5111-8111-111111111111"
SECONDARY_ID = "22222222-2222-5222-8222-222222222222"


def player_catalog() -> LocalPlayerCatalog:
    players = pl.DataFrame(
        {
            "player_id": [PRIMARY_ID, SECONDARY_ID],
            "display_name": ["Michael Penix Jr.", "Tua Tagovailoa"],
            "position": ["QB", "QB"],
            "position_group": ["QB", "QB"],
        }
    )
    external_ids = pl.DataFrame(
        {
            "player_id": [PRIMARY_ID],
            "provider": ["fantasypros"],
            "external_id": ["22973"],
        }
    )
    return LocalPlayerCatalog(players, external_ids)


def raw_envelope() -> dict:
    return {
        "schema_version": 1,
        "provider": "fantasypros",
        "external_id": "603602",
        "fetched_at": "2026-08-21T19:50:52+00:00",
        "content_hash": "raw-hash",
        "payload": {
            "id": 603602,
            "created": "2026-08-21 14:52:01",
            "author": "Ari Koslow",
            "player_id": 22973,
            "team_id": "ATL",
            "title": "Michael Penix Jr. waiting for clearance",
            "categories": ["Commentary", "News", "Injury"],
            "link": "https://example.com/603602",
            "desc": "<b>Michael Penix Jr.</b> is waiting for clearance.",
            "impact": (
                "Tua Tagovailoa remains the current favorite."
                "<br><a href='https://example.com'>view fantasy impact »</a>"
            ),
        },
    }


def extraction() -> ExtractedReportMetadata:
    return ExtractedReportMetadata(
        external_id="603602",
        document_type=DocumentType.INJURY_UPDATE,
        document_type_confidence=0.98,
        player_mentions=[
            ExtractedPlayerMention(
                reference_text="Michael Penix Jr.",
                canonical_name="Michael Penix Jr.",
                identity_confidence=1.0,
                resolution_basis=PlayerResolutionBasis.EXACT_NAME,
                mention_role=MentionRole.PRIMARY_SUBJECT,
            ),
            ExtractedPlayerMention(
                reference_text="Tua Tagovailoa",
                canonical_name="Tua Tagovailoa",
                identity_confidence=1.0,
                resolution_basis=PlayerResolutionBasis.EXACT_NAME,
                mention_role=MentionRole.MATERIALLY_AFFECTED,
            ),
        ],
    )


class FakeResponses:
    def __init__(self, parsed: ExtractedReportBatch) -> None:
        self.parsed = parsed
        self.calls = 0

    def parse(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            output_parsed=self.parsed,
            usage=SimpleNamespace(
                input_tokens=200,
                output_tokens=80,
                input_tokens_details=SimpleNamespace(cached_tokens=25),
            ),
        )


class ReportProcessingTests(unittest.TestCase):
    def test_normalize_report_maps_primary_and_secondary_players(self) -> None:
        document = normalize_report(
            raw_envelope(),
            extraction(),
            player_catalog(),
            model="gpt-5.6-luna",
            processed_at=datetime(2026, 8, 21, 20, tzinfo=UTC),
        )

        self.assertEqual(document["id"], "fantasypros:603602")
        self.assertEqual(
            document["players"],
            ["Michael Penix Jr.", "Tua Tagovailoa"],
        )
        self.assertEqual(document["player_ids"], [PRIMARY_ID, SECONDARY_ID])
        self.assertEqual(document["teams"], ["ATL"])
        self.assertEqual(document["document_type"], "injury_update")
        self.assertNotIn("<b>", document["body"])
        self.assertNotIn("view fantasy impact", document["body"])
        self.assertEqual(
            document["metadata_processing"]["unresolved_player_mentions"],
            [],
        )

    def test_inferred_player_is_audited_but_never_assigned_an_id(self) -> None:
        weak = extraction().model_copy(
            update={
                "player_mentions": [
                    ExtractedPlayerMention(
                        reference_text="the other quarterback",
                        canonical_name="Tua Tagovailoa",
                        identity_confidence=0.95,
                        resolution_basis=PlayerResolutionBasis.INFERRED,
                        mention_role=MentionRole.CONTEXTUAL,
                    )
                ]
            }
        )
        document = normalize_report(
            raw_envelope(),
            weak,
            player_catalog(),
            model="gpt-5.6-luna",
        )

        self.assertEqual(document["player_ids"], [PRIMARY_ID])
        unresolved = document["metadata_processing"]["unresolved_player_mentions"]
        self.assertEqual(len(unresolved), 1)
        self.assertIn("requires stronger-model escalation", unresolved[0]["reason"])

    def test_unchanged_report_skips_a_second_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            output_dir = root / "processed" / "documents" / "fantasypros"
            items_dir = raw_dir / "items"
            items_dir.mkdir(parents=True)
            (items_dir / "603602.json").write_text(
                json.dumps(raw_envelope()),
                encoding="utf-8",
            )
            responses = FakeResponses(
                ExtractedReportBatch(reports=[extraction()])
            )
            client = SimpleNamespace(responses=responses)

            first = process_reports(
                raw_dir=raw_dir,
                output_dir=output_dir,
                model="gpt-5.6-luna",
                client=client,
                catalog=player_catalog(),
            )
            second = process_reports(
                raw_dir=raw_dir,
                output_dir=output_dir,
                model="gpt-5.6-luna",
                client=client,
                catalog=player_catalog(),
            )

            self.assertEqual((first.inserted, first.unchanged), (1, 0))
            self.assertEqual((second.inserted, second.unchanged), (0, 1))
            self.assertEqual(responses.calls, 1)


if __name__ == "__main__":
    unittest.main()
