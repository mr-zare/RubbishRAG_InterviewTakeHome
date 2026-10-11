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

            key = (
                adds if selected else 10_000,
                float(hit.get("rerank_score", 0.0)),
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
