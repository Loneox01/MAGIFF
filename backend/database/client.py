"""Shared Supabase client configuration for the backend."""

import os
from pathlib import Path
import threading
from typing import Any, cast

from dotenv import load_dotenv
from supabase import Client, create_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CLIENTS = threading.local()
_TRANSIENT_ERROR_MARKERS = (
    "resource temporarily unavailable",
    "jwt issued at future",
    "pgrst303",
)


def is_transient_supabase_error(error: Exception) -> bool:
    """Identify read-only Supabase failures that are safe to retry once."""
    message = str(error).casefold()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def _current_thread_client() -> Client:
    client = getattr(_CLIENTS, "client", None)
    if client is not None:
        return client

    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not secret_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY must be set in the root .env"
        )

    client = create_client(url, secret_key)
    _CLIENTS.client = client
    return client


class _ThreadLocalSupabaseClient:
    """Delegate every client operation to the calling thread's client."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_current_thread_client(), name)


_CLIENT_PROXY = _ThreadLocalSupabaseClient()


def get_supabase_client() -> Client:
    """Return a proxy backed by one reusable client per calling thread.

    Agent tools execute concurrently. A process-wide sync client lets multiple
    worker threads read from the same HTTP/2 transport, which can produce
    competing socket reads. Each worker therefore owns its own connection pool
    while still reusing that pool for sequential calls on the same thread. The
    proxy also lets long-lived repositories follow the calling thread instead
    of capturing the client belonging to the thread that created them.
    """
    return cast(Client, _CLIENT_PROXY)


def clear_supabase_client() -> None:
    """Discard the current thread's client, primarily for tests/recovery."""
    if hasattr(_CLIENTS, "client"):
        del _CLIENTS.client
