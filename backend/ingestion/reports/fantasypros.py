"""Fetch FantasyPros NFL news and preserve raw items locally.

This module intentionally stops before report summarization and entity linking.
The first live response should be inspected before that processing contract is
finalized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_OUTPUT_DIR = (
    BACKEND_DIR / "data" / "raw" / "reports" / "sources" / "fantasypros"
)
NEWS_URL = "https://api.fantasypros.com/public/v2/json/nfl/news"
ALLOWED_CATEGORIES = {"injury", "recap", "transaction", "rumor", "breaking"}
ALLOWED_ORDER_FIELDS = {"created", "updated"}


@dataclass(frozen=True)
class IngestionResult:
    discovered: int
    inserted: int
    updated: int
    unchanged: int
    failed: int
    output_dir: str


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def content_hash(item: dict[str, Any]) -> str:
    """Return the stable provider-payload hash used for deduplication."""
    canonical = json.dumps(
        item,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(_json_text(value), encoding="utf-8")
    temporary.replace(path)


def fetch_news(
    api_key: str,
    *,
    limit: int = 25,
    category: str | None = None,
    fpid: str | int | None = None,
    order_by: str = "created",
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    """Fetch one FantasyPros NFL news response."""
    if not api_key.strip():
        raise ValueError("FantasyPros API key cannot be empty")
    if limit < 1:
        raise ValueError("limit must be positive")
    if category is not None and category not in ALLOWED_CATEGORIES:
        allowed = ", ".join(sorted(ALLOWED_CATEGORIES))
        raise ValueError(f"category must be one of: {allowed}")
    if fpid is not None and (not str(fpid).isdigit() or int(fpid) < 1):
        raise ValueError("fpid must be a positive FantasyPros player ID")
    if order_by not in ALLOWED_ORDER_FIELDS:
        allowed = ", ".join(sorted(ALLOWED_ORDER_FIELDS))
        raise ValueError(f"order_by must be one of: {allowed}")

    params: dict[str, object] = {"limit": limit, "order_by": order_by}
    if category is not None:
        params["category"] = category
    if fpid is not None:
        params["fpid"] = str(fpid)

    response = httpx.get(
        NEWS_URL,
        headers={"x-api-key": api_key},
        params=params,
        timeout=timeout_seconds,
        follow_redirects=True,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = " ".join(response.text.split())[:500]
        message = (
            f"FantasyPros API returned HTTP {response.status_code}"
            f" for {response.request.url.path}"
        )
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from error
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("FantasyPros response did not contain an items list")
    return payload


def ingest_payload(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    category: str | None = None,
    requested_limit: int | None = None,
    fetched_at: datetime | None = None,
) -> IngestionResult:
    """Idempotently preserve raw feed items and content-addressed versions."""
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("FantasyPros payload must contain an items list")

    acquired_at = (fetched_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    inserted = 0
    updated = 0
    unchanged = 0
    failed = 0

    for raw_item in items:
        if not isinstance(raw_item, dict) or raw_item.get("id") is None:
            failed += 1
            continue

        external_id = str(raw_item["id"])
        item_hash = content_hash(raw_item)
        envelope = {
            "schema_version": 1,
            "provider": "fantasypros",
            "external_id": external_id,
            "fetched_at": acquired_at,
            "content_hash": item_hash,
            "payload": raw_item,
        }

        current_path = output_dir / "items" / f"{external_id}.json"
        version_path = (
            output_dir / "versions" / external_id / f"{item_hash}.json"
        )
        previous_hash = None
        if current_path.exists():
            previous = json.loads(current_path.read_text(encoding="utf-8"))
            previous_hash = previous.get("content_hash")

        if previous_hash == item_hash:
            unchanged += 1
            continue

        _write_json(version_path, envelope)
        _write_json(current_path, envelope)
        if previous_hash is None:
            inserted += 1
        else:
            updated += 1

    result = IngestionResult(
        discovered=len(items),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        failed=failed,
        output_dir=str(output_dir.resolve()),
    )
    response_metadata = {
        key: value for key, value in payload.items() if key != "items"
    }
    _write_json(
        output_dir / "latest_run.json",
        {
            "provider": "fantasypros",
            "dataset": "nfl_news",
            "endpoint": NEWS_URL,
            "fetched_at": acquired_at,
            "category": category,
            "requested_limit": requested_limit,
            "response_metadata": response_metadata,
            **asdict(result),
        },
    )
    return result


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fixture must contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch FantasyPros NFL news into local raw storage."
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--category", choices=sorted(ALLOWED_CATEGORIES))
    parser.add_argument("--fpid", help="Filter news by FantasyPros player ID.")
    parser.add_argument(
        "--order-by",
        choices=sorted(ALLOWED_ORDER_FIELDS),
        default="created",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Read a saved API response instead of calling FantasyPros.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if args.fixture is not None:
        payload = _load_fixture(args.fixture)
    else:
        api_key = os.getenv("FANTASYPROS_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Set FANTASYPROS_API_KEY in the project-root .env file"
            )
        payload = fetch_news(
            api_key,
            limit=args.limit,
            category=args.category,
            fpid=args.fpid,
            order_by=args.order_by,
        )

    result = ingest_payload(
        payload,
        output_dir=args.output_dir,
        category=args.category,
        requested_limit=args.limit,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
