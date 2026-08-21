"""Configuration for request-level orchestration."""

import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_ROUTER_INDEX_PATH = (
    BACKEND_DIR
    / "data"
    / "processed"
    / "orchestration"
    / "request_router.sqlite3"
)

load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_ROUTER_MODEL = os.getenv("OPENAI_ROUTER_MODEL", "gpt-5.6-luna")
