import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from database.load_reports import (
    EMBEDDING_DIMENSIONS,
    load_report_data,
    prepare_report,
)


PLAYER_ID = "11111111-1111-5111-8111-111111111111"


def write_fixture(root: Path) -> tuple[Path, Path]:
    documents_dir = root / "processed" / "documents"
    raw_dir = root / "raw" / "sources"
    processed_path = documents_dir / "fantasypros" / "1.json"
    raw_path = raw_dir / "fantasypros" / "items" / "1.json"
    processed_path.parent.mkdir(parents=True)
    raw_path.parent.mkdir(parents=True)
    raw = {
        "provider": "fantasypros",
        "external_id": "1",
        "fetched_at": "2026-08-21T20:00:00+00:00",
        "content_hash": "source-hash",
        "payload": {"id": 1, "title": "Test"},
    }
    document = {
        "id": "fantasypros:1",
        "provider": "fantasypros",
        "external_id": "1",
        "source": "FantasyPros",
        "url": "https://example.com/1",
        "title": "Player returns to practice",
        "author": "Reporter",
        "published_at": "2026-08-21T14:00:00+00:00",
        "fetched_at": "2026-08-21T20:00:00+00:00",
        "players": ["Test Player"],
        "player_ids": [PLAYER_ID],
        "player_entities": [
            {
                "display_name": "Test Player",
                "player_id": PLAYER_ID,
                "reference_text": "Test Player",
                "identity_confidence": 1.0,
                "resolution_basis": "provider_id",
                "mention_role": "primary_subject",
                "resolution_source": "fantasypros_player_id",
            }
        ],
        "teams": ["ATL"],
        "source_team_id": "ATL",
        "season": 2026,
        "document_type": "practice_update",
        "document_type_confidence": 0.95,
        "storyline": None,
        "content_mode": "provider_news",
        "source_categories": ["News"],
        "body": "# News\n\nThe player returned to practice.",
        "source_content_hash": "source-hash",
        "content_hash": "normalized-hash",
        "metadata_processing": {
            "model": "gpt-5.6-luna",
            "prompt_version": "1",
            "normalizer_version": "1",
            "processed_at": "2026-08-21T20:05:00+00:00",
            "unresolved_player_mentions": [],
        },
    }
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    processed_path.write_text(json.dumps(document), encoding="utf-8")
    return documents_dir, raw_dir


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_args):
        return self

    def in_(self, *_args):
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


class FakeRPC:
    def __init__(self, client, params):
        self.client = client
        self.params = params

    def execute(self):
        self.client.calls.append(self.params)
        return SimpleNamespace(data={"version_inserted": True})


class FakeSupabase:
    def __init__(self, existing=None):
        self.existing = existing or []
        self.calls = []

    def table(self, name):
        if name != "report_chunks":
            raise AssertionError(name)
        return FakeQuery(self.existing)

    def rpc(self, name, params):
        if name != "upsert_report_document":
            raise AssertionError(name)
        return FakeRPC(self, params)


class ReportLoaderTests(unittest.TestCase):
    def test_prepare_report_builds_relational_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents_dir, raw_dir = write_fixture(Path(directory))
            prepared = prepare_report(
                documents_dir / "fantasypros" / "1.json",
                raw_reports_dir=raw_dir,
            )

            self.assertEqual(prepared.report["report_id"], "fantasypros:1")
            self.assertEqual(prepared.version["raw_payload"]["external_id"], "1")
            self.assertEqual(prepared.players[0]["player_id"], PLAYER_ID)
            self.assertEqual(prepared.chunks[0]["chunk_id"], "fantasypros:1:0")
            self.assertIn("Players: Test Player", prepared.chunks[0]["embedding_text"])

    def test_dry_run_makes_no_database_or_embedding_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents_dir, raw_dir = write_fixture(Path(directory))

            result = load_report_data(
                documents_dir=documents_dir,
                raw_reports_dir=raw_dir,
                dry_run=True,
                client=None,
                embedder=lambda *_args, **_kwargs: self.fail("embedding called"),
                log_path=Path(directory) / "log.json",
            )

            self.assertEqual(result.discovered, 1)
            self.assertEqual(result.uploaded, 0)
            self.assertEqual(result.player_links, 1)

    def test_upload_embeds_new_chunk_and_calls_transactional_rpc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents_dir, raw_dir = write_fixture(Path(directory))
            client = FakeSupabase()

            result = load_report_data(
                documents_dir=documents_dir,
                raw_reports_dir=raw_dir,
                client=client,
                embedder=lambda texts, **_kwargs: [
                    [0.01] * EMBEDDING_DIMENSIONS for _ in texts
                ],
                log_path=Path(directory) / "log.json",
            )

            self.assertEqual(result.generated_embeddings, 1)
            self.assertEqual(result.uploaded, 1)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(
                len(client.calls[0]["p_chunks"][0]["embedding"]),
                EMBEDDING_DIMENSIONS,
            )

    def test_upload_reuses_unchanged_database_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents_dir, raw_dir = write_fixture(Path(directory))
            prepared = prepare_report(
                documents_dir / "fantasypros" / "1.json",
                raw_reports_dir=raw_dir,
            )
            chunk = prepared.chunks[0]
            client = FakeSupabase(
                [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "content_hash": chunk["content_hash"],
                        "embedding_model": "text-embedding-3-small",
                    }
                ]
            )

            result = load_report_data(
                documents_dir=documents_dir,
                raw_reports_dir=raw_dir,
                client=client,
                embedder=lambda *_args, **_kwargs: self.fail("embedding called"),
                log_path=Path(directory) / "log.json",
            )

            self.assertEqual(result.generated_embeddings, 0)
            self.assertEqual(result.reused_embeddings, 1)
            self.assertIsNone(client.calls[0]["p_chunks"][0]["embedding"])


if __name__ == "__main__":
    unittest.main()
