import unittest
from types import SimpleNamespace

from rag.retrieval.supabase_store import EMBEDDING_DIMENSIONS, SupabaseRAGStore


def report_row(*, report_id: str = "fantasypros:1") -> dict:
    return {
        "chunk_id": f"{report_id}:0",
        "report_id": report_id,
        "title": "Michael Penix Jr. returns",
        "source": "FantasyPros",
        "source_url": "https://example.com/1",
        "author": "Reporter",
        "published_at": "2026-08-21T14:00:00+00:00",
        "fetched_at": "2026-08-21T20:00:00+00:00",
        "player_ids": ["11111111-1111-5111-8111-111111111111"],
        "player_names": ["Michael Penix Jr."],
        "teams": ["ATL"],
        "season": 2026,
        "document_type": "injury_update",
        "storyline": None,
        "content_mode": "provider_news",
        "content": "Penix returned to practice.",
    }


class FakeRPC:
    def __init__(self, client, name, params):
        self.client = client
        self.name = name
        self.params = params

    def execute(self):
        self.client.calls.append((self.name, self.params))
        return SimpleNamespace(data=self.client.responses[self.name])


class FakeSupabase:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def rpc(self, name, params):
        return FakeRPC(self, name, params)


class SupabaseReportStoreTests(unittest.TestCase):
    def test_keyword_search_maps_filters_and_document_metadata(self) -> None:
        row = {**report_row(), "keyword_rank": 0.75}
        client = FakeSupabase({"search_report_chunks_v2": [row]})
        store = SupabaseRAGStore(client=client)

        hits = store.keyword_search(
            "Penix practice",
            limit=5,
            players=["Michael Penix Jr."],
            teams=["ATL"],
            season=2026,
            document_type="injury_update",
            published_from="2026-08-01",
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].document.id, "fantasypros:1")
        self.assertEqual(hits[0].document.players, ("Michael Penix Jr.",))
        name, params = client.calls[0]
        self.assertEqual(name, "search_report_chunks_v2")
        self.assertEqual(params["query_text"], '"penix" OR "practice"')
        self.assertEqual(params["filter_teams"], ["ATL"])
        self.assertEqual(params["filter_player_names"], ["Michael Penix Jr."])
        self.assertEqual(params["published_from"], "2026-08-01")

    def test_vector_search_embeds_once_and_reuses_query_cache(self) -> None:
        row = {**report_row(), "similarity": 0.91}
        client = FakeSupabase({"match_report_chunks_v2": [row]})
        embedding_calls = []

        def embedder(texts, **kwargs):
            embedding_calls.append((texts, kwargs))
            return [[0.01] * EMBEDDING_DIMENSIONS]

        store = SupabaseRAGStore(client=client, embedder=embedder)
        first = store.vector_search("Penix health", player_ids=["player-1"])
        second = store.vector_search("Penix health", player_ids=["player-1"])

        self.assertEqual(first[0].score, 0.91)
        self.assertEqual(second[0].document.title, first[0].document.title)
        self.assertEqual(len(embedding_calls), 1)
        self.assertEqual(embedding_calls[0][1]["dimensions"], 1536)
        name, params = client.calls[0]
        self.assertEqual(name, "match_report_chunks_v2")
        self.assertEqual(params["filter_embedding_model"], "text-embedding-3-small")
        self.assertEqual(params["filter_player_ids"], ["player-1"])

    def test_hybrid_search_uses_reciprocal_rank_fusion(self) -> None:
        first = report_row(report_id="fantasypros:1")
        second = {
            **report_row(report_id="fantasypros:2"),
            "title": "Tua wins starting job",
        }
        client = FakeSupabase(
            {
                "search_report_chunks_v2": [
                    {**first, "keyword_rank": 0.9},
                    {**second, "keyword_rank": 0.8},
                ],
                "match_report_chunks_v2": [
                    {**second, "similarity": 0.95},
                ],
            }
        )
        store = SupabaseRAGStore(
            client=client,
            embedder=lambda _texts, **_kwargs: [
                [0.01] * EMBEDDING_DIMENSIONS
            ],
        )

        hits = store.hybrid_search("Falcons quarterback", limit=2)

        self.assertEqual([hit.document.id for hit in hits], ["fantasypros:2", "fantasypros:1"])
        self.assertEqual(hits[0].keyword_rank, 2)
        self.assertEqual(hits[0].vector_rank, 1)

    def test_status_uses_database_rpc(self) -> None:
        expected = {
            "store": "supabase",
            "document_count": 5,
            "chunk_count": 5,
            "embedded_count": 5,
            "embedding_models": ["text-embedding-3-small"],
        }
        client = FakeSupabase({"report_store_status": expected})

        self.assertEqual(SupabaseRAGStore(client=client).status(), expected)


if __name__ == "__main__":
    unittest.main()
