from ..retrieval.store import LocalRAGStore
from .cases import CASES, RetrievalCase


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
