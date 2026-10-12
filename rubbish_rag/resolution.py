"""Grounded answer/clarification resolver (P10 core contract)."""
from __future__ import annotations

from typing import Any

from .evidence import (
    assess_sufficiency,
    build_evidence_set,
)
from .models import Intent
from .slots import parse_query


def _clarification_for(frame) -> dict[str, Any] | None:
    missing = set(frame.ambiguity.missing_required_slots)
    if "debt_type" in missing:
        return {
            "status": "clarify",
            "question": "منظورتان کدام نوع بدهی است؟",
            "options": ["بدهی مالیاتی", "چک برگشتی", "بدهی یا معوقه تسهیلات", "محکومیت یا بدهی حقوقی"],
            "missing_slots": ["debt_type"],
            "cites": [],
        }
    if "rank" in missing:
        return {
            "status": "clarify",
            "question": "کدام رتبه اعتباری مدنظر شماست؟",
            "options": ["A1 تا A3", "B1 تا B3", "C1 تا C3", "D1 تا E3"],
            "missing_slots": ["rank"],
            "cites": [],
        }
    return None


def resolve_grounded(query: str, ranked_hits: list[dict[str, Any]]) -> dict[str, Any]:
    frame = parse_query(query)
    clarify = _clarification_for(frame)
    if clarify:
        return clarify

    evidence = build_evidence_set(
        query,
        ranked_hits,
        max_evidence=3,
    )

    assessment = assess_sufficiency(
        frame,
        evidence,
    )

    if not evidence.hits:
        # Public resolver contract only advertises answer|clarify. P11 can map
        # this internal insufficient state to a true abstention later.
        return {
            "status": "clarify",
            "question": "برای پاسخ قابل اتکا، لطفاً جزئیات بیشتری از موضوع یا رخداد اعتباری موردنظر ارائه کنید.",
            "options": ["رتبه و امتیاز", "بدهی مالیاتی", "چک", "تسهیلات و وام"],
            "missing_slots": [],
            "reason": "insufficient_trusted_evidence",
            "cites": [],
        }

    if not assessment.sufficient:
        reasons = set(
            assessment.reasons
        )

        if "hard_semantic_conflict" in reasons:
            question = (
                "شواهد بازیابی‌شده با جزئیات پرسش ناسازگار است؛ "
                "لطفاً اطلاعات کلیدی مانند نوع بدهی، رتبه، بانک یا عدد موردنظر را دوباره مشخص کنید."
            )
        elif "critical_requirements_uncovered" in reasons:
            question = (
                "برای پاسخ قطعی، بخشی از اطلاعات کلیدی پرسش در شواهد موجود پوشش داده نشده است؛ "
                "لطفاً جزئیات را دقیق‌تر مشخص کنید."
            )
        else:
            question = (
                "شواهد قابل اتکای کافی برای پاسخ قطعی پیدا نشد؛ "
                "لطفاً موضوع را دقیق‌تر مشخص کنید."
            )

        return {
            "status": "clarify",
            "question": question,
            "options": [],
            "missing_slots": [],
            "reason": "insufficient_evidence",
            "decision": {
                "sufficient": False,
                "reasons": list(
                    assessment.reasons
                ),
                "critical_uncovered": list(
                    assessment.critical_uncovered
                ),
                "conflicts": list(
                    assessment.conflicts
                ),
                "citable_doc_ids": list(
                    assessment.citable_doc_ids
                ),
            },
            "cites": [],
        }

    primary = evidence.hits[0]
    brief = str(primary.get("brief_answer") or "").strip()
    if not brief:
        brief = str(primary.get("text") or "").strip()
    if len(brief) > 500:
        brief = brief[:500].rsplit(" ", 1)[0].rstrip() + "…"

    cites: list[int] = []
    for hit in evidence.hits:
        if hit.get("citation_eligible", True) is False:
            continue
        doc_id = int(hit.get("doc_id", -1))
        if doc_id >= 0 and doc_id not in cites:
            cites.append(doc_id)

    if not cites:
        return {
            "status": "clarify",
            "question": "شواهد قابل استناد کافی پیدا نشد؛ لطفاً موضوع را دقیق‌تر مشخص کنید.",
            "options": [],
            "reason": "no_citation_eligible_evidence",
            "cites": [],
        }

    return {
        "status": "answer",
        "brief": brief,
        "cites": cites[:3],
        "evidence": {
            "covered_requirements": sorted(
                evidence.covered_requirements
            ),
            "uncovered_requirements": sorted(
                evidence.uncovered_requirements
            ),
            "sufficiency": {
                "sufficient": True,
                "reasons": [],
                "critical_uncovered": [],
                "conflicts": [],
            },
        },
    }
