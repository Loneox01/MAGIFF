"""Resolve model-facing player references before calling UUID-based tools."""

from __future__ import annotations

import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from database.client import clear_supabase_client, is_transient_supabase_error
from repositories import nfl_supabase as repository


@dataclass(frozen=True)
class PlayerReferenceArgument:
    external_name: str
    internal_name: str
    many: bool = False


PLAYER_REFERENCE_ARGUMENTS: dict[str, PlayerReferenceArgument] = {
    "get_player_weekly_stats": PlayerReferenceArgument(
        "player_ref", "player_id"
    ),
    "get_player_season_stats": PlayerReferenceArgument(
        "player_ref", "player_id"
    ),
    "get_player_snap_counts": PlayerReferenceArgument(
        "player_ref", "player_id"
    ),
    "get_player_ecr": PlayerReferenceArgument("player_ref", "player_id"),
    "rank_players_by_weekly_threshold": PlayerReferenceArgument(
        "player_refs", "player_ids", many=True
    ),
}


@dataclass(frozen=True)
class ResolvedPlayerReference:
    reference: str
    player_id: str
    display_name: str | None
    resolution_basis: str


class PlayerReferenceError(ValueError):
    """A player reference that should be returned to the model, not guessed."""

    def __init__(
        self,
        *,
        code: str,
        reference: str,
        message: str,
        candidates: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reference = reference
        self.candidates = tuple(dict(candidate) for candidate in candidates)

    def as_tool_output(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "player_ref": self.reference,
                "candidates": list(self.candidates),
                "retry": (
                    "Retry with one candidate's player_id as player_ref, or ask "
                    "the user to clarify."
                    if self.code == "ambiguous_player_ref"
                    else "Retry with a fuller canonical player name."
                ),
            }
        }


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalized_name(value: str, *, omit_suffix: bool = False) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode(
        "ascii", "ignore"
    ).decode("ascii")
    # Initials and apostrophes should compare identically with or without
    # punctuation (A.J./AJ, De'Von/Devon); hyphenated names keep a word break.
    ascii_value = ascii_value.replace(".", "").replace("'", "")
    tokens = re.sub(r"[^a-zA-Z0-9]+", " ", ascii_value).casefold().split()
    if omit_suffix and tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _candidate_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate.get(key)
        for key in (
            "player_id",
            "display_name",
            "position",
            "latest_team",
            "status",
        )
    }


class PlayerReferenceResolver:
    """Resolve a canonical name or UUID without letting the model guess IDs."""

    def __init__(
        self,
        find_players: Callable[[str], list[dict[str, Any]]] = (
            repository.find_players
        ),
    ) -> None:
        self.find_players = find_players

    def resolve(self, reference: str) -> ResolvedPlayerReference:
        normalized_reference = reference.strip()
        if not normalized_reference:
            raise PlayerReferenceError(
                code="invalid_player_ref",
                reference=reference,
                message="player_ref cannot be empty.",
            )

        try:
            player_uuid = str(uuid.UUID(normalized_reference))
        except ValueError:
            player_uuid = None
        if player_uuid is not None:
            return ResolvedPlayerReference(
                reference=normalized_reference,
                player_id=player_uuid,
                display_name=None,
                resolution_basis="uuid",
            )

        candidates = self._find_candidates(normalized_reference)
        if not candidates:
            raise PlayerReferenceError(
                code="player_ref_not_found",
                reference=normalized_reference,
                message=(
                    f"No stored player matched {normalized_reference!r}."
                ),
            )

        exact_key = _normalized_name(normalized_reference)
        exact = [
            candidate
            for candidate in candidates
            if _normalized_name(str(candidate.get("display_name") or ""))
            == exact_key
        ]
        selected = self._unique_or_error(normalized_reference, exact)
        basis = "normalized_exact_name"

        if selected is None:
            suffix_key = _normalized_name(
                normalized_reference, omit_suffix=True
            )
            suffix_matches = [
                candidate
                for candidate in candidates
                if _normalized_name(
                    str(candidate.get("display_name") or ""),
                    omit_suffix=True,
                )
                == suffix_key
            ]
            selected = self._unique_or_error(
                normalized_reference, suffix_matches
            )
            basis = "suffix_normalized_name"

        if selected is None and len(candidates) == 1:
            selected = candidates[0]
            basis = "unique_partial_name"

        if selected is None:
            raise PlayerReferenceError(
                code="ambiguous_player_ref",
                reference=normalized_reference,
                message=(
                    f"Multiple stored players matched {normalized_reference!r}."
                ),
                candidates=[_candidate_payload(item) for item in candidates],
            )

        return ResolvedPlayerReference(
            reference=normalized_reference,
            player_id=str(selected["player_id"]),
            display_name=str(selected.get("display_name") or "") or None,
            resolution_basis=basis,
        )

    def _find_candidates(self, reference: str) -> list[dict[str, Any]]:
        candidates = self.find_players(reference)
        if self._contains_normalized_match(reference, candidates):
            return self._deduplicate(candidates)

        fallback = _normalized_name(reference, omit_suffix=True).split()
        if fallback:
            # A surname fallback lets punctuation variants such as initials and
            # hyphenation reach the deterministic normalized-name comparison.
            candidates = [
                *candidates,
                *self.find_players(fallback[-1]),
            ]
        return self._deduplicate(candidates)

    @staticmethod
    def _deduplicate(candidates: Sequence[Mapping[str, Any]]) -> list[dict]:
        by_id: dict[str, dict] = {}
        for candidate in candidates:
            player_id = str(candidate.get("player_id") or "")
            if player_id:
                by_id.setdefault(player_id, dict(candidate))
        return list(by_id.values())

    @staticmethod
    def _contains_normalized_match(
        reference: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> bool:
        exact = _normalized_name(reference)
        suffixless = _normalized_name(reference, omit_suffix=True)
        return any(
            _normalized_name(str(candidate.get("display_name") or "")) == exact
            or _normalized_name(
                str(candidate.get("display_name") or ""), omit_suffix=True
            )
            == suffixless
            for candidate in candidates
        )

    @staticmethod
    def _unique_or_error(
        reference: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        if len(candidates) == 1:
            return dict(candidates[0])
        if len(candidates) > 1:
            raise PlayerReferenceError(
                code="ambiguous_player_ref",
                reference=reference,
                message=f"Multiple stored players matched {reference!r}.",
                candidates=[_candidate_payload(item) for item in candidates],
            )
        return None


class PlayerReferenceAdapter:
    """Translate public player_ref arguments into internal player_id arguments."""

    def __init__(self, resolver: PlayerReferenceResolver | None = None) -> None:
        self.resolver = resolver or PlayerReferenceResolver()

    def resolve_many(
        self,
        references: Sequence[str],
        *,
        cache: dict[str, ResolvedPlayerReference | PlayerReferenceError],
        max_workers: int,
    ) -> None:
        pending: dict[str, str] = {}
        for reference in references:
            key = self.cache_key(reference)
            if key not in cache:
                pending.setdefault(key, reference)
        if not pending:
            return

        def resolve_one(item: tuple[str, str]):
            key, reference = item
            for attempt in range(2):
                try:
                    return key, self.resolver.resolve(reference)
                except PlayerReferenceError as error:
                    return key, error
                except Exception as error:
                    if attempt == 0 and is_transient_supabase_error(error):
                        clear_supabase_client()
                        time.sleep(0.1)
                        continue
                    return key, PlayerReferenceError(
                        code="player_ref_resolution_failed",
                        reference=reference,
                        message=(
                            "Player reference resolution failed for "
                            f"{reference!r}: {error}"
                        ),
                    )

            raise AssertionError("unreachable")

        worker_count = min(max_workers, len(pending))
        if worker_count == 1:
            results = map(resolve_one, pending.items())
        else:
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="player-ref",
            )
            try:
                results = executor.map(resolve_one, pending.items())
                results = list(results)
            finally:
                executor.shutdown(wait=True, cancel_futures=False)
        cache.update(results)

    def adapt(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        cache: Mapping[str, ResolvedPlayerReference | PlayerReferenceError],
    ) -> dict[str, Any]:
        spec = PLAYER_REFERENCE_ARGUMENTS.get(tool_name)
        adapted = dict(arguments)
        if spec is None or spec.external_name not in adapted:
            return adapted

        raw_value = adapted.pop(spec.external_name)
        if spec.many and raw_value is None:
            adapted[spec.internal_name] = None
            return adapted

        references = raw_value if spec.many else [raw_value]
        if not isinstance(references, list) or any(
            not isinstance(item, str) for item in references
        ):
            raise PlayerReferenceError(
                code="invalid_player_ref",
                reference=str(raw_value),
                message=(
                    f"{spec.external_name} must contain player names or UUIDs."
                ),
            )

        player_ids: list[str] = []
        for reference in references:
            resolution = cache[self.cache_key(reference)]
            if isinstance(resolution, PlayerReferenceError):
                raise resolution
            player_ids.append(resolution.player_id)
        adapted[spec.internal_name] = (
            list(dict.fromkeys(player_ids)) if spec.many else player_ids[0]
        )
        return adapted

    @staticmethod
    def references_for(
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> list[str]:
        spec = PLAYER_REFERENCE_ARGUMENTS.get(tool_name)
        if spec is None or spec.external_name not in arguments:
            return []
        raw_value = arguments[spec.external_name]
        if raw_value is None:
            return []
        if spec.many:
            if not isinstance(raw_value, list):
                return []
            return [item for item in raw_value if isinstance(item, str)]
        return [raw_value] if isinstance(raw_value, str) else []

    @staticmethod
    def cache_key(reference: str) -> str:
        return reference.strip().casefold()
