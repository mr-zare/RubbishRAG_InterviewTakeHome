"""Sentence-aware parent/child chunking (P4)."""
from __future__ import annotations

import re
from typing import Any

from .normalize_fa import normalize_match_text

_SENTENCE_RE = re.compile(r"(?<=[.!؟!?؛])\s+|\n+")
_CLAUSE_RE = re.compile(r"(?<=[،,:؛;])\s+")


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in _SENTENCE_RE.split(text or "") if x.strip()]


def _hard_split(text: str, hard_max: int) -> list[str]:
    if len(text) <= hard_max:
        return [text]
    clauses = [x.strip() for x in _CLAUSE_RE.split(text) if x.strip()]
    if len(clauses) > 1:
        out: list[str] = []
        cur = ""
        for clause in clauses:
            cand = (cur + " " + clause).strip()
            if cur and len(cand) > hard_max:
                out.append(cur)
                cur = clause
            else:
                cur = cand
        if cur:
            out.append(cur)
        return [p for piece in out for p in _hard_split(piece, hard_max) if p]

    # Last resort: word-boundary chunks. Never cut through a token/number-unit pair.
    words = text.split()
    out, cur = [], []
    n = 0
    for word in words:
        add = len(word) + (1 if cur else 0)
        if cur and n + add > hard_max:
            out.append(" ".join(cur))
            cur, n = [word], len(word)
        else:
            cur.append(word)
            n += add
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_record(doc: dict[str, Any], *, soft_target: int = 420, hard_max: int = 700) -> list[dict[str, Any]]:
    parent_id = int(doc["doc_id"])
    answer = (doc.get("Answer") or doc.get("BriefAnswer") or "").strip()
    question = (doc.get("Question") or "").strip()
    category = (doc.get("Category") or "").strip()
    keyword = (doc.get("Keyword") or "").strip()
    if not answer:
        return []

    if len(answer) <= soft_target:
        parts = [answer]
    else:
        atoms: list[str] = []
        for sentence in _sentences(answer) or [answer]:
            atoms.extend(_hard_split(sentence, hard_max))
        parts = []
        current = ""
        for atom in atoms:
            candidate = (current + " " + atom).strip()
            if current and len(candidate) > soft_target:
                parts.append(current)
                current = atom
            else:
                current = candidate
        if current:
            parts.append(current)

    chunks: list[dict[str, Any]] = []
    cursor = 0
    for idx, part in enumerate(parts):
        start = answer.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        cursor = end
        # Keep source text pristine. Contextual views are separate fields.
        lexical_view = " ".join(x for x in (question, keyword, part) if x)
        semantic_view = " ".join(x for x in (category, question, doc.get("BriefAnswer") or "", part) if x)
        chunks.append({
            "doc_id": parent_id,
            "parent_doc_id": parent_id,
            "category": category,
            "keyword": keyword,
            "question": question,
            "text": part,
            "chunk_id": f"{parent_id}:{idx}",
            "source_start": start,
            "source_end": end,
            "lexical_text": normalize_match_text(lexical_view),
            "semantic_text": normalize_match_text(semantic_view),
        })
    return chunks


def chunk_documents(docs: list[dict[str, Any]], *, soft_target: int = 420, hard_max: int = 700) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in docs:
        out.extend(chunk_record(doc, soft_target=soft_target, hard_max=hard_max))
    return out
