from dataclasses import dataclass

from .store import LocalRAGStore


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    query: str
    expected_ids: frozenset[str]


CASES = [
    RetrievalCase(
        name="Sadiq setback",
        query="What is the latest update on Kenyon Sadiq's hernia setback?",
        expected_ids=frozenset(
            {"nbc-sports-2026-08-04-kenyon-sadiq-recovery-setback"}
        ),
    ),
    RetrievalCase(
        name="Seattle backfield injury",
        query="Jadarian Price lower body soreness Seattle backfield",
        expected_ids=frozenset(
            {"the-news-tribune-2026-08-08-jadarian-price-lower-body-soreness"}
        ),
    ),
    RetrievalCase(
        name="McConkey availability",
        query="Was Ladd McConkey full go for Chargers training camp?",
        expected_ids=frozenset(
            {"los-angeles-chargers-2026-07-28-ladd-mcconkey-full-go"}
        ),
    ),
    RetrievalCase(
        name="Lemon missed practice",
        query="Makai Lemon missed practice with a hamstring injury",
        expected_ids=frozenset(
            {"nbc-sports-2026-08-08-makai-lemon-misses-practice"}
        ),
    ),
]


def evaluate_retrieval(
    store: LocalRAGStore,
    mode: str = "keyword",
    top_k: int = 3,
) -> tuple[int, list[tuple[RetrievalCase, list[str], bool]]]:
    results: list[tuple[RetrievalCase, list[str], bool]] = []
    passes = 0

    for case in CASES:
        hits = store.search(case.query, mode=mode, limit=top_k)
        ids = [hit.document.id for hit in hits]
        passed = bool(case.expected_ids.intersection(ids))
        passes += int(passed)
        results.append((case, ids, passed))

    return passes, results
