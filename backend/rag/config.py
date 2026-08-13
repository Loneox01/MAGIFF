import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
REPORTS_ROOT = BACKEND_DIR / "data" / "raw" / "reports"
DEFAULT_INDEX_PATH = (
    BACKEND_DIR / "data" / "processed" / "reports" / "local_rag.sqlite3"
)

load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
DEFAULT_PLANNER_MODEL = os.getenv(
    "OPENAI_PLANNER_MODEL",
    "gpt-5.6-luna",
)
DEFAULT_ESCALATION_MODEL = os.getenv(
    "OPENAI_ESCALATION_MODEL",
    "gpt-5.6-sol",
)

# Defaults reflect the model price when this router was introduced. Environment
# overrides keep cost telemetry useful if pricing or the escalation model changes.
ESCALATION_INPUT_COST_PER_MILLION = float(
    os.getenv("OPENAI_ESCALATION_INPUT_COST_PER_MILLION", "5.00")
)
ESCALATION_CACHED_INPUT_COST_PER_MILLION = float(
    os.getenv("OPENAI_ESCALATION_CACHED_INPUT_COST_PER_MILLION", "0.50")
)
ESCALATION_OUTPUT_COST_PER_MILLION = float(
    os.getenv("OPENAI_ESCALATION_OUTPUT_COST_PER_MILLION", "30.00")
)
