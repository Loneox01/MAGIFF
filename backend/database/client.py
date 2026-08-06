"""Shared Supabase client configuration for the backend."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Create one client and reuse it for subsequent repository calls."""
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not secret_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY must be set in the root .env"
        )

    return create_client(url, secret_key)
