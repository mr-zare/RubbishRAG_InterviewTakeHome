"""Evidence construction, deduplication, and query coverage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .corpus import get_registry
from .models import EntityType, QuerySemanticFrame
from .normalize_fa import normalize_match_text
from .slots import parse_query


@dataclass(slots=True)
class EvidenceSet:
    hits: list[dict[str, Any]]
    covered_requirements: set[str]
    uncovered_requirements: set[str]
    duplicate_doc_ids_removed: int = 0
    exact_text_duplicates_removed: int = 0



@dataclass(frozen=True, slots=True)
class SufficiencyAssessment:
    sufficient: bool
    reasons: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    critical_uncovered: tuple[str, ...] = ()
    citable_doc_ids: tuple[int, ...] = ()


def query_requirements(frame: QuerySemanticFrame) -> set[str]:
    req: set[str] = set()

    if frame.intent.value != "general_info":
        req.add(f"intent:{frame.intent.value}")

    for entity_type in (
        EntityType.BANK,
        EntityType.RANK,
        EntityType.PERSONA,
        EntityType.DEBT,
    ):
        req.update(
            f"{entity_type.value}:{value}"
            for value in frame.values(entity_type)
        )

    for quantity in frame.quantities:
        if quantity.value is not None:
            req.add(
                f"quantity:{quantity.quantity_type.value}:"
                f"{quantity.value}:{quantity.unit or ''}"
            )

    return req


_CRITICAL_REQUIREMENT_PREFIXES = (
    "intent:",
    "bank:",
    "rank:",
    "debt:",
    "quantity:",
)

_HARD_CONFLICT_FIELDS = (
    "bank_reason",
    "rank_reason",
    "debt_reason",
    "quantity_reason",
    "polarity_reason",
)

_HARD_CONFLICT_VALUES = {
    "BANK_CONFLICT",
    "RANK_CONFLICT",
    "DEBT_CONFLICT",
    "QUANTITY_CONFLICT",
    "POLARITY_CONFLICT",
}


def _is_citation_eligible(
    hit: dict[str, Any],
) -> bool:
    """Fail closed when provenance explicitly marks a hit unusable.

    Missing metadata remains backward-compatible with public/synthetic
    contract tests. Production provenance is expected to set these flags
    explicitly.
    """
    if hit.get("citation_eligible") is False:
        return False

    if hit.get("trusted") is False:
        return False

    return True


def _hard_conflicts(
    hits: list[dict[str, Any]],
) -> tuple[str, ...]:
    conflicts: set[str] = set()

    for hit in hits:
        parts = hit.get("local_parts")

        if not isinstance(parts, dict):
            continue

        try:
            doc_id = int(
                hit.get("doc_id", -1)
            )
        except (TypeError, ValueError):
            doc_id = -1

        for field in _HARD_CONFLICT_FIELDS:
            value = parts.get(field)

            if value in _HARD_CONFLICT_VALUES:
                conflicts.add(
                    f"{value}:doc={doc_id}"
                )

    return tuple(sorted(conflicts))


def assess_sufficiency(
    frame: QuerySemanticFrame,
    evidence: EvidenceSet,
) -> SufficiencyAssessment:
    """Hard, deterministic evidence gate.

    Ranking score is intentionally NOT treated as confidence. Sufficiency is
    determined from provenance, requirement coverage, and semantic
    consistency.
    """

    reasons: list[str] = []

    if not evidence.hits:
        reasons.append(
            "no_evidence"
        )

    citable_doc_ids: list[int] = []

    for hit in evidence.hits:
        if not _is_citation_eligible(hit):
            continue

        try:
            doc_id = int(
                hit.get("doc_id", -1)
            )
        except (TypeError, ValueError):
            continue

        if (
            doc_id >= 0
            and doc_id not in citable_doc_ids
        ):
            citable_doc_ids.append(doc_id)

    if evidence.hits and not citable_doc_ids:
        reasons.append(
            "no_trusted_citable_evidence"
        )

    critical_uncovered = tuple(
        sorted(
            requirement
            for requirement
            in evidence.uncovered_requirements
            if requirement.startswith(
                _CRITICAL_REQUIREMENT_PREFIXES
            )
        )
    )

    if critical_uncovered:
        reasons.append(
            "critical_requirements_uncovered"
        )

    conflicts = _hard_conflicts(
        evidence.hits
    )

    if conflicts:
        reasons.append(
            "hard_semantic_conflict"
        )

    return SufficiencyAssessment(
        sufficient=not reasons,
        reasons=tuple(reasons),
        conflicts=conflicts,
        critical_uncovered=critical_uncovered,
        citable_doc_ids=tuple(
            citable_doc_ids
        ),
    )


def _frame_for_hit(hit: dict[str, Any]) -> QuerySemanticFrame:
    # P8 already warms trusted document semantic frames. Reuse them instead
    # of reparsing long FAQ answers repeatedly during evidence selection.
    #
    # Synthetic public contract tests intentionally use supplied hit text.
    if hit.get("candidate_source") != "remote_unverified_test":
        try:
            doc_id = int(hit.get("doc_id", -1))
        except (TypeError, ValueError):
            doc_id = -1

        registry = get_registry()
        cached = registry.document_frames.get(doc_id)

        if cached is not None:
            return cached

    text = " ".join(
        str(hit.get(key) or "")
        for key in ("question", "brief_answer", "text", "keyword")
    )

    return parse_query(text)


def _coverage(
    frame: QuerySemanticFrame,
    hit: dict[str, Any],
) -> set[str]:
    doc_frame = _frame_for_hit(hit)
    covered: set[str] = set()

    if frame.intent == doc_frame.intent:
        covered.add(f"intent:{frame.intent.value}")

    for entity_type in (
        EntityType.BANK,
        EntityType.RANK,
        EntityType.PERSONA,
        EntityType.DEBT,
    ):
        matches = (
            set(frame.values(entity_type))
            & set(doc_frame.values(entity_type))
        )

        for value in matches:
            covered.add(f"{entity_type.value}:{value}")

    for query_quantity in frame.quantities:
        for doc_quantity in doc_frame.quantities:
            if (
                query_quantity.quantity_type
                == doc_quantity.quantity_type
                and query_quantity.value
                == doc_quantity.value
                and (
                    not query_quantity.unit
                    or not doc_quantity.unit
                    or query_quantity.unit == doc_quantity.unit
                )
            ):
                covered.add(
                    f"quantity:{query_quantity.quantity_type.value}:"
                    f"{query_quantity.value}:{query_quantity.unit or ''}"
                )

    return covered


def build_evidence_set(
    query: str,
    ranked_hits: list[dict[str, Any]],
    *,
    max_evidence: int = 3,
) -> EvidenceSet:
    frame = parse_query(query)
    requirements = query_requirements(frame)

    selected: list[dict[str, Any]] = []
    seen_docs: set[int] = set()
    seen_texts: set[str] = set()
    covered: set[str] = set()

    duplicate_docs = 0
    duplicate_texts = 0

    remaining = list(ranked_hits)
    coverage_cache: dict[int, set[str]] = {}

    while remaining and len(selected) < max_evidence:
        # Smallest sufficient evidence set: once all requirements are covered,
        # do not fill the remaining citation slots with redundant documents.
        if (
            selected
            and (
                not requirements
                or requirements.issubset(covered)
            )
        ):
            break

        best_idx: int | None = None
        best_key = None

        for idx, hit in enumerate(remaining):
            doc_id = int(hit.get("doc_id", -1))

            text_key = normalize_match_text(
                str(
                    hit.get("brief_answer")
                    or hit.get("text")
                    or ""
                )
            )

            if doc_id in seen_docs:
                continue

            if text_key and text_key in seen_texts:
                continue

            coverage = coverage_cache.setdefault(
                id(hit),
                _coverage(frame, hit),
            )

            adds = len(coverage - covered)

            # The first candidate is primary evidence.
            # Additional candidates must add new query requirement coverage.
            if selected and adds <= 0:
                continue

            # Provenance precedes ranking score. An untrusted/uncitable
            # candidate must not suppress a lower-ranked trusted source merely
            # because its retrieval score is larger.
            provenance_priority = (
                1
                if _is_citation_eligible(hit)
                else 0
            )

            key = (
                provenance_priority,
                adds if selected else 10_000,
                float(
                    hit.get(
                        "rerank_score",
                        0.0,
                    )
                ),
                -idx,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            break

        hit = remaining.pop(best_idx)
        doc_id = int(hit.get("doc_id", -1))

        text_key = normalize_match_text(
            str(
                hit.get("brief_answer")
                or hit.get("text")
                or ""
            )
        )

        if doc_id in seen_docs:
            duplicate_docs += 1
            continue

        if text_key and text_key in seen_texts:
            duplicate_texts += 1
            continue

        selected.append(hit)
        seen_docs.add(doc_id)

        if text_key:
            seen_texts.add(text_key)

        covered |= coverage_cache.setdefault(
            id(hit),
            _coverage(frame, hit),
        )

    return EvidenceSet(
        hits=selected,
        covered_requirements=covered,
        uncovered_requirements=requirements - covered,
        duplicate_doc_ids_removed=duplicate_docs,
        exact_text_duplicates_removed=duplicate_texts,
    )
