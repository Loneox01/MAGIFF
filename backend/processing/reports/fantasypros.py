"""Normalize raw FantasyPros player news into database-ready report JSON.

The provider's primary player ID is grounded deterministically. A small
structured-output model extracts material secondary player mentions and a
coarse document type; application code then validates every model-proposed
identity against the local processed player catalog before storing an internal
UUID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from supabase import Client

from prompts import REPORT_METADATA_INSTRUCTIONS
from rag.planning.planner import PlayerResolutionBasis

from ..normalization.team_codes import TEAM_CODE_ALIASES


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
RAW_DIR = BACKEND_DIR / "data" / "raw" / "reports" / "sources" / "fantasypros"
DEFAULT_OUTPUT_DIR = (
    BACKEND_DIR / "data" / "processed" / "reports" / "documents" / "fantasypros"
)
PROCESSED_REFERENCE_DIR = BACKEND_DIR / "data" / "processed" / "reference"
METADATA_PROMPT_VERSION = "1"
NORMALIZER_VERSION = "1"
DEFAULT_BATCH_SIZE = 10


class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentType(StrEnum):
    INJURY_UPDATE = "injury_update"
    PRACTICE_UPDATE = "practice_update"
    TRANSACTION = "transaction"
    CONTRACT_UPDATE = "contract_update"
    DEPTH_CHART_UPDATE = "depth_chart_update"
    ROLE_USAGE_UPDATE = "role_usage_update"
    PERFORMANCE_RECAP = "performance_recap"
    PROJECTION_ANALYSIS = "projection_analysis"
    GENERAL_NEWS = "general_news"


class MentionRole(StrEnum):
    PRIMARY_SUBJECT = "primary_subject"
    MATERIALLY_AFFECTED = "materially_affected"
    CONTEXTUAL = "contextual"


class ExtractedPlayerMention(MetadataModel):
    reference_text: str = Field(min_length=1)
    canonical_name: str | None
    identity_confidence: float = Field(ge=0, le=1)
    resolution_basis: PlayerResolutionBasis
    mention_role: MentionRole


class ExtractedReportMetadata(MetadataModel):
    external_id: str = Field(min_length=1)
    document_type: DocumentType
    document_type_confidence: float = Field(ge=0, le=1)
    player_mentions: list[ExtractedPlayerMention]


class ExtractedReportBatch(MetadataModel):
    reports: list[ExtractedReportMetadata] = Field(min_length=1)


@dataclass(frozen=True)
class PlayerRecord:
    player_id: str
    display_name: str
    position: str | None
    position_group: str | None


@dataclass(frozen=True)
class ProcessingResult:
    discovered: int
    inserted: int
    updated: int
    unchanged: int
    failed: int
    unresolved_player_mentions: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    output_dir: str


class PlayerCatalog(Protocol):
    """Minimal identity lookup contract used by report normalization."""

    def primary_player(self, external_id: object) -> PlayerRecord | None: ...

    def name_matches(self, canonical_name: str | None) -> list[PlayerRecord]: ...


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join("".join(self.parts).split())


def _clean_provider_text(value: object) -> str:
    if value is None:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(str(value))
    text = parser.text()
    return re.sub(
        r"\s*view fantasy impact\s*»?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(_json_text(value), encoding="utf-8")
    temporary.replace(path)


def _hash_value(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _nfl_season(published_at: datetime) -> int:
    return published_at.year if published_at.month >= 3 else published_at.year - 1


def _published_at(value: object) -> datetime:
    parsed = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=UTC)


class LocalPlayerCatalog:
    """Exact local identity catalog used after model mention extraction."""

    def __init__(
        self,
        players: pl.DataFrame,
        external_ids: pl.DataFrame,
    ) -> None:
        player_records: dict[str, PlayerRecord] = {}
        names: dict[str, list[PlayerRecord]] = {}
        for row in players.select(
            "player_id", "display_name", "position", "position_group"
        ).iter_rows(named=True):
            if row["display_name"] is None:
                continue
            record = PlayerRecord(
                player_id=str(row["player_id"]),
                display_name=str(row["display_name"]),
                position=None if row["position"] is None else str(row["position"]),
                position_group=(
                    None
                    if row["position_group"] is None
                    else str(row["position_group"])
                ),
            )
            player_records[record.player_id] = record
            names.setdefault(_normalized_name(record.display_name), []).append(record)

        fantasypros: dict[str, PlayerRecord] = {}
        for row in (
            external_ids.filter(pl.col("provider") == "fantasypros")
            .select("external_id", "player_id")
            .iter_rows(named=True)
        ):
            record = player_records.get(str(row["player_id"]))
            if record is not None:
                fantasypros[str(row["external_id"])] = record

        self._by_fantasypros_id = fantasypros
        self._by_name = names

    @classmethod
    def from_processed_data(cls) -> "LocalPlayerCatalog":
        return cls(
            pl.read_parquet(PROCESSED_REFERENCE_DIR / "players.parquet"),
            pl.read_parquet(
                PROCESSED_REFERENCE_DIR / "player_external_ids.parquet"
            ),
        )

    def primary_player(self, external_id: object) -> PlayerRecord | None:
        return self._by_fantasypros_id.get(str(external_id))

    def name_matches(self, canonical_name: str | None) -> list[PlayerRecord]:
        if not canonical_name:
            return []
        return list(self._by_name.get(_normalized_name(canonical_name), []))


class SupabasePlayerCatalog:
    """Bounded player identity catalog for deployed report refresh jobs.

    Provider IDs for the fetched page are loaded in two batched queries. Model-
    proposed secondary names are looked up lazily, normalized locally, and
    cached for the remainder of the run.
    """

    def __init__(
        self,
        client: Client,
        *,
        by_fantasypros_id: dict[str, PlayerRecord] | None = None,
    ) -> None:
        self._client = client
        self._by_fantasypros_id = by_fantasypros_id or {}
        self._by_name: dict[str, list[PlayerRecord]] = {}
        for record in self._by_fantasypros_id.values():
            self._by_name.setdefault(
                _normalized_name(record.display_name), []
            ).append(record)

    @staticmethod
    def _record(row: dict[str, Any]) -> PlayerRecord:
        return PlayerRecord(
            player_id=str(row["player_id"]),
            display_name=str(row["display_name"]),
            position=None if row.get("position") is None else str(row["position"]),
            position_group=(
                None
                if row.get("position_group") is None
                else str(row["position_group"])
            ),
        )

    @classmethod
    def from_external_ids(
        cls,
        client: Client,
        external_ids: list[str],
    ) -> "SupabasePlayerCatalog":
        unique_external_ids = sorted({str(value) for value in external_ids if value})
        if not unique_external_ids:
            return cls(client)

        crosswalk_rows: list[dict[str, Any]] = []
        for batch in _batches(unique_external_ids, 100):
            response = (
                client.table("player_external_ids")
                .select("external_id,player_id")
                .eq("provider", "fantasypros")
                .in_("external_id", batch)
                .execute()
            )
            crosswalk_rows.extend(response.data or [])

        player_ids = sorted(
            {str(row["player_id"]) for row in crosswalk_rows if row.get("player_id")}
        )
        player_rows: list[dict[str, Any]] = []
        for batch in _batches(player_ids, 100):
            response = (
                client.table("players")
                .select("player_id,display_name,position,position_group")
                .in_("player_id", batch)
                .execute()
            )
            player_rows.extend(response.data or [])

        records = {
            str(row["player_id"]): cls._record(row)
            for row in player_rows
            if row.get("player_id") and row.get("display_name")
        }
        by_external_id = {
            str(row["external_id"]): records[str(row["player_id"])]
            for row in crosswalk_rows
            if str(row.get("player_id")) in records
        }
        return cls(client, by_fantasypros_id=by_external_id)

    def primary_player(self, external_id: object) -> PlayerRecord | None:
        return self._by_fantasypros_id.get(str(external_id))

    def name_matches(self, canonical_name: str | None) -> list[PlayerRecord]:
        if not canonical_name:
            return []
        normalized = _normalized_name(canonical_name)
        if normalized in self._by_name:
            return list(self._by_name[normalized])

        response = (
            self._client.table("players")
            .select("player_id,display_name,position,position_group")
            .ilike("display_name", f"%{canonical_name.strip()}%")
            .limit(25)
            .execute()
        )
        matches = [
            self._record(row)
            for row in (response.data or [])
            if row.get("display_name")
            and _normalized_name(str(row["display_name"])) == normalized
        ]
        self._by_name[normalized] = matches
        return list(matches)


def _identity_rejection_reason(
    mention: ExtractedPlayerMention,
    matches: list[PlayerRecord],
) -> str | None:
    if mention.canonical_name is None:
        return "model returned no canonical player candidate"
    if mention.resolution_basis == PlayerResolutionBasis.NOT_APPLICABLE:
        return "not_applicable is invalid for an individual report mention"
    if mention.resolution_basis == PlayerResolutionBasis.INFERRED:
        return "inferred identity requires stronger-model escalation"
    if mention.resolution_basis in {
        PlayerResolutionBasis.KNOWN_ALIAS,
        PlayerResolutionBasis.CONTEXTUAL_ALIAS,
    } and mention.identity_confidence < 0.70:
        return "identity confidence is below the 0.70 basis threshold"
    if len(matches) == 0:
        return "canonical player candidate was not found in the player catalog"
    if len(matches) > 1:
        return "canonical player candidate matched multiple catalog players"
    if (
        mention.resolution_basis == PlayerResolutionBasis.EXACT_NAME
        and _normalized_name(mention.reference_text)
        != _normalized_name(matches[0].display_name)
    ):
        return "exact_name basis does not match the official database name"
    return None


def _resolved_player_dict(
    record: PlayerRecord,
    *,
    reference_text: str,
    identity_confidence: float,
    resolution_basis: str,
    mention_role: str,
    resolution_source: str,
) -> dict[str, object]:
    return {
        "player_id": record.player_id,
        "display_name": record.display_name,
        "position": record.position,
        "position_group": record.position_group,
        "reference_text": reference_text,
        "identity_confidence": identity_confidence,
        "resolution_basis": resolution_basis,
        "mention_role": mention_role,
        "resolution_source": resolution_source,
    }


def _resolve_players(
    raw_item: dict[str, Any],
    extraction: ExtractedReportMetadata,
    catalog: PlayerCatalog,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    resolved: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    primary = catalog.primary_player(raw_item.get("player_id"))
    if primary is not None:
        resolved.append(
            _resolved_player_dict(
                primary,
                reference_text=primary.display_name,
                identity_confidence=1.0,
                resolution_basis="provider_id",
                mention_role=MentionRole.PRIMARY_SUBJECT.value,
                resolution_source="fantasypros_player_id",
            )
        )
        seen_ids.add(primary.player_id)
    elif raw_item.get("player_id") is not None:
        unresolved.append(
            {
                "reference_text": str(raw_item.get("player_id")),
                "canonical_name": None,
                "identity_confidence": 1.0,
                "resolution_basis": "provider_id",
                "mention_role": MentionRole.PRIMARY_SUBJECT.value,
                "reason": "FantasyPros player ID was not present in the player crosswalk",
            }
        )

    for mention in extraction.player_mentions:
        matches = catalog.name_matches(mention.canonical_name)
        rejection = _identity_rejection_reason(mention, matches)
        if rejection is not None:
            unresolved.append(
                {
                    **mention.model_dump(mode="json"),
                    "reason": rejection,
                }
            )
            continue
        record = matches[0]
        if record.player_id in seen_ids:
            continue
        resolved.append(
            _resolved_player_dict(
                record,
                reference_text=mention.reference_text,
                identity_confidence=mention.identity_confidence,
                resolution_basis=mention.resolution_basis.value,
                mention_role=mention.mention_role.value,
                resolution_source="model_candidate_database_match",
            )
        )
        seen_ids.add(record.player_id)
    return resolved, unresolved


def _normalized_teams(raw_item: dict[str, Any]) -> list[str]:
    value = str(raw_item.get("team_id") or "").strip().upper()
    value = TEAM_CODE_ALIASES.get(value, value)
    if not value or value == "FA":
        return []
    return [value]


def normalize_report(
    envelope: dict[str, Any],
    extraction: ExtractedReportMetadata,
    catalog: PlayerCatalog,
    *,
    model: str,
    processed_at: datetime | None = None,
) -> dict[str, object]:
    """Return one normalized report while grounding all internal player IDs."""
    raw_item = envelope.get("payload")
    if not isinstance(raw_item, dict):
        raise ValueError("Raw report envelope did not contain a payload object")
    external_id = str(envelope.get("external_id", raw_item.get("id", "")))
    if extraction.external_id != external_id:
        raise ValueError(
            f"Extraction ID {extraction.external_id} did not match {external_id}"
        )

    published = _published_at(raw_item["created"])
    normalized_at = (processed_at or datetime.now(UTC)).astimezone(UTC)
    description = _clean_provider_text(raw_item.get("desc"))
    impact = _clean_provider_text(raw_item.get("impact"))
    body_parts = []
    if description:
        body_parts.append(f"# News\n\n{description}")
    if impact:
        body_parts.append(f"# Fantasy impact\n\n{impact}")
    body = "\n\n".join(body_parts)

    resolved, unresolved = _resolve_players(raw_item, extraction, catalog)
    teams = _normalized_teams(raw_item)
    core = {
        "id": f"fantasypros:{external_id}",
        "provider": "fantasypros",
        "external_id": external_id,
        "title": _clean_provider_text(raw_item.get("title")),
        "source": "FantasyPros",
        "url": str(raw_item.get("link") or ""),
        "author": raw_item.get("author"),
        "published_at": published.isoformat(),
        "fetched_at": str(envelope.get("fetched_at") or ""),
        "players": [item["display_name"] for item in resolved],
        "player_ids": [item["player_id"] for item in resolved],
        "player_entities": resolved,
        "teams": teams,
        "source_team_id": raw_item.get("team_id"),
        "season": _nfl_season(published),
        "document_type": extraction.document_type.value,
        "document_type_confidence": extraction.document_type_confidence,
        "storyline": None,
        "content_mode": "provider_news",
        "source_categories": list(raw_item.get("categories") or []),
        "body": body,
        "source_content_hash": str(envelope.get("content_hash") or ""),
    }
    core["content_hash"] = _hash_value(
        {
            key: value
            for key, value in core.items()
            if key not in {"fetched_at", "source_content_hash"}
        }
    )
    core["metadata_processing"] = {
        "model": model,
        "prompt_version": METADATA_PROMPT_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "processed_at": normalized_at.isoformat(),
        "model_player_mentions": [
            mention.model_dump(mode="json")
            for mention in extraction.player_mentions
        ],
        "unresolved_player_mentions": unresolved,
    }
    return core


def _load_envelopes(raw_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    items_dir = raw_dir / "items"
    if not items_dir.exists():
        raise FileNotFoundError(f"No raw FantasyPros items found at: {items_dir}")
    records = []
    for path in sorted(items_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Raw report must be a JSON object: {path}")
        records.append((path, value))
    return records


def _output_is_current(path: Path, envelope: dict[str, Any], model: str) -> bool:
    if not path.exists():
        return False
    current = json.loads(path.read_text(encoding="utf-8"))
    metadata = current.get("metadata_processing") or {}
    return (
        current.get("source_content_hash") == envelope.get("content_hash")
        and metadata.get("model") == model
        and metadata.get("prompt_version") == METADATA_PROMPT_VERSION
        and metadata.get("normalizer_version") == NORMALIZER_VERSION
    )


def _model_payload(
    records: list[tuple[Path, dict[str, Any]]],
) -> dict[str, object]:
    reports = []
    for _, envelope in records:
        raw = envelope["payload"]
        reports.append(
            {
                "external_id": str(envelope["external_id"]),
                "source_player_id": raw.get("player_id"),
                "source_team_id": raw.get("team_id"),
                "title": _clean_provider_text(raw.get("title")),
                "description": _clean_provider_text(raw.get("desc")),
                "fantasy_impact": _clean_provider_text(raw.get("impact")),
            }
        )
    return {"reports": reports}


def _extract_batch(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    client: OpenAI,
    model: str,
) -> tuple[dict[str, ExtractedReportMetadata], int, int, int]:
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "none"},
        input=[
            {"role": "system", "content": REPORT_METADATA_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(_model_payload(records), separators=(",", ":")),
            },
        ],
        text_format=ExtractedReportBatch,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("Report metadata extractor returned no structured output")

    expected = [str(envelope["external_id"]) for _, envelope in records]
    returned = [item.external_id for item in parsed.reports]
    if sorted(expected) != sorted(returned) or len(returned) != len(set(returned)):
        raise RuntimeError(
            "Report metadata extractor must return exactly one result per report; "
            f"expected {expected}, got {returned}"
        )
    usage = response.usage
    details = getattr(usage, "input_tokens_details", None) if usage else None
    cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
    return (
        {item.external_id: item for item in parsed.reports},
        usage.input_tokens if usage else 0,
        cached,
        usage.output_tokens if usage else 0,
    )


def _batches(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def process_reports(
    *,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    client: OpenAI | None = None,
    catalog: PlayerCatalog | None = None,
) -> ProcessingResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = _load_envelopes(raw_dir)
    pending = []
    unchanged = 0
    for path, envelope in records:
        output_path = output_dir / f"{envelope['external_id']}.json"
        if not force and _output_is_current(output_path, envelope, model):
            unchanged += 1
        else:
            pending.append((path, envelope))

    inserted = 0
    updated = 0
    failed = 0
    unresolved = 0
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    processed_outputs: list[str] = []
    processing_errors: list[dict[str, str]] = []
    player_catalog = catalog or LocalPlayerCatalog.from_processed_data()
    openai_client = client or (OpenAI() if pending else None)

    for batch in _batches(pending, batch_size):
        assert openai_client is not None
        extracted, batch_input, batch_cached, batch_output = _extract_batch(
            batch,
            client=openai_client,
            model=model,
        )
        input_tokens += batch_input
        cached_input_tokens += batch_cached
        output_tokens += batch_output
        for _, envelope in batch:
            external_id = str(envelope["external_id"])
            output_path = output_dir / f"{external_id}.json"
            existed = output_path.exists()
            try:
                normalized = normalize_report(
                    envelope,
                    extracted[external_id],
                    player_catalog,
                    model=model,
                )
                unresolved += len(
                    normalized["metadata_processing"]["unresolved_player_mentions"]
                )
                _write_json(output_path, normalized)
                processed_outputs.append(str(output_path.resolve()))
                if existed:
                    updated += 1
                else:
                    inserted += 1
            except (KeyError, TypeError, ValueError) as error:
                failed += 1
                processing_errors.append(
                    {"external_id": external_id, "error": str(error)}
                )

    result = ProcessingResult(
        discovered=len(records),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        failed=failed,
        unresolved_player_mentions=unresolved,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        output_dir=str(output_dir.resolve()),
    )
    _write_json(
        output_dir.parent.parent / "latest_run.json",
        {
            "workflow": "fantasypros_report_metadata",
            "processed_at": datetime.now(UTC).isoformat(),
            "model": model,
            "prompt_version": METADATA_PROMPT_VERSION,
            "normalizer_version": NORMALIZER_VERSION,
            "processed_outputs": processed_outputs,
            "processing_errors": processing_errors,
            **asdict(result),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    model = os.getenv("OPENAI_REPORT_METADATA_MODEL", "gpt-5.6-luna")
    result = process_reports(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        model=model,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps(asdict(result), indent=2))
    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
