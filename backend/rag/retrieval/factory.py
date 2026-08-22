"""Construct the configured report retrieval store."""

from pathlib import Path

from ..config import DEFAULT_INDEX_PATH, DEFAULT_REPORT_STORE
from .store import LocalRAGStore
from .supabase_store import SupabaseRAGStore


def create_report_store(
    store_name: str | None = None,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> LocalRAGStore:
    selected = (store_name or DEFAULT_REPORT_STORE).strip().lower()
    if selected == "local":
        return LocalRAGStore(index_path=index_path)
    if selected == "supabase":
        return SupabaseRAGStore(cache_path=index_path)
    raise ValueError(
        f"Unsupported report store {selected!r}; expected 'local' or 'supabase'"
    )
