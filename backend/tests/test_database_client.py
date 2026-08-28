import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from database import client as database_client


class SupabaseClientTests(unittest.TestCase):
    def tearDown(self) -> None:
        database_client.clear_supabase_client()

    def test_client_is_reused_within_thread_and_isolated_across_threads(
        self,
    ) -> None:
        created = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        class FakeClient:
            def identity(self):
                return self

        def create_client(_url, _key):
            client = FakeClient()
            with lock:
                created.append(client)
            return client

        def worker():
            proxy = database_client.get_supabase_client()
            first = proxy.identity()
            barrier.wait(timeout=1)
            second = proxy.identity()
            return first, second

        with (
            patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://example.supabase.co",
                    "SUPABASE_SECRET_KEY": "secret",
                },
            ),
            patch.object(
                database_client,
                "create_client",
                side_effect=create_client,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(lambda _index: worker(), range(2)))

        self.assertIs(results[0][0], results[0][1])
        self.assertIs(results[1][0], results[1][1])
        self.assertIsNot(results[0][0], results[1][0])
        self.assertEqual(len(created), 2)


if __name__ == "__main__":
    unittest.main()
