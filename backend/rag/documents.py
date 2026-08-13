import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import REPORTS_ROOT


REQUIRED_METADATA = {
    "id",
    "title",
    "source",
    "url",
    "published_at",
    "fetched_at",
    "players",
    "teams",
    "season",
    "document_type",
    "storyline",
    "content_mode",
}


@dataclass(frozen=True)
class ReportDocument:
    id: str
    title: str
    source: str
    url: str
    author: str | None
    published_at: str
    fetched_at: str
    players: tuple[str, ...]
    teams: tuple[str, ...]
    season: int
    document_type: str
    storyline: str
    content_mode: str
    body: str
    source_path: Path
    player_ids: tuple[str, ...] = ()

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model and searched by SQLite FTS."""
        metadata = [
            f"Title: {self.title}",
            f"Source: {self.source}",
            f"Published: {self.published_at}",
            f"Players: {', '.join(self.players)}",
            f"Teams: {', '.join(self.teams)}",
            f"Document type: {self.document_type}",
            f"Storyline: {self.storyline.replace('_', ' ')}",
        ]
        return "\n".join(metadata) + "\n\n" + self.body

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embedding_text.encode("utf-8")).hexdigest()

    @property
    def snippet(self) -> str:
        lines = [
            line.removeprefix("- ").strip()
            for line in self.body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return " ".join(lines)


def _parse_metadata(raw_metadata: str, path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for line in raw_metadata.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Invalid frontmatter line in {path}: {line!r}")

        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip()

        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value

        metadata[key.strip()] = value

    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"Missing frontmatter fields in {path}: {missing_fields}")

    return metadata


def parse_report(path: Path) -> ReportDocument:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"Report does not start with frontmatter: {path}")

    try:
        _, raw_metadata, raw_body = text.split("---", 2)
    except ValueError as error:
        raise ValueError(f"Report has incomplete frontmatter: {path}") from error

    metadata = _parse_metadata(raw_metadata, path)

    # The source note is useful in the raw record, but repeated boilerplate weakens
    # retrieval quality and does not belong in the embedding text.
    body = raw_body.strip().partition("\n# Source note")[0].strip()

    return ReportDocument(
        id=str(metadata["id"]),
        title=str(metadata["title"]),
        source=str(metadata["source"]),
        url=str(metadata["url"]),
        author=None if metadata.get("author") is None else str(metadata["author"]),
        published_at=str(metadata["published_at"]),
        fetched_at=str(metadata["fetched_at"]),
        players=tuple(str(player) for player in metadata["players"]),
        teams=tuple(str(team) for team in metadata["teams"]),
        season=int(metadata["season"]),
        document_type=str(metadata["document_type"]),
        storyline=str(metadata["storyline"]),
        content_mode=str(metadata["content_mode"]),
        body=body,
        source_path=path.resolve(),
        player_ids=tuple(str(player_id) for player_id in metadata.get("player_ids", [])),
    )


def resolve_snapshot(
    snapshot: str | Path | None = None,
    reports_root: Path = REPORTS_ROOT,
) -> Path:
    if snapshot is not None:
        candidate = Path(snapshot)
        if not candidate.is_absolute():
            candidate = reports_root / candidate
        candidate = candidate.resolve()
    else:
        candidates = sorted(
            path.resolve()
            for path in reports_root.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )
        if not candidates:
            raise FileNotFoundError(f"No report snapshots found in {reports_root}")
        candidate = candidates[-1]

    if not (candidate / "manifest.json").exists():
        raise FileNotFoundError(f"No manifest.json found in report snapshot {candidate}")

    return candidate


def load_reports(
    snapshot: str | Path | None = None,
    reports_root: Path = REPORTS_ROOT,
) -> list[ReportDocument]:
    snapshot_dir = resolve_snapshot(snapshot=snapshot, reports_root=reports_root)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    documents: list[ReportDocument] = []
    seen_ids: set[str] = set()

    for item in manifest.get("documents", []):
        path = (snapshot_dir / item["path"]).resolve()
        if snapshot_dir not in path.parents:
            raise ValueError(f"Report path escapes snapshot directory: {item['path']}")

        document = parse_report(path)
        if document.id != item["id"]:
            raise ValueError(
                f"Manifest id {item['id']!r} does not match {document.id!r} in {path}"
            )
        if document.id in seen_ids:
            raise ValueError(f"Duplicate report id in manifest: {document.id}")

        seen_ids.add(document.id)
        documents.append(document)

    expected_count = manifest.get("document_count")
    if expected_count is not None and expected_count != len(documents):
        raise ValueError(
            f"Manifest expects {expected_count} documents; loaded {len(documents)}"
        )

    return documents
