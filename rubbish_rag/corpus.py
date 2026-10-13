from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .normalize_fa import normalize_match_text


def _sha256(parts: Iterable[str]) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    source_doc_id: int
    question: str
    category: str
    brief_answer: str
    answer: str
    keyword: str
    source_row_hash: str
    quality_flags: tuple[str, ...] = ()

    @property
    def effective_answer(self) -> str:
        return self.brief_answer or self.answer

    @property
    def retrieval_text(self) -> str:
        return " ".join(x for x in (self.question, self.keyword, self.brief_answer, self.answer) if x).strip()


@dataclass(slots=True)
class TrustedCorpusRegistry:
    records: dict[int, CorpusRecord] = field(default_factory=dict)
    corpus_hash: str = ""
    exact_answer_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    exact_question_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    token_df: dict[str, int] = field(default_factory=dict)
    tokens_by_doc: dict[int, frozenset[str]] = field(default_factory=dict)
    question_tokens: dict[int, frozenset[str]] = field(default_factory=dict)
    document_frames: dict[int, Any] = field(default_factory=dict)
    question_frames: dict[int, Any] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path) -> "TrustedCorpusRegistry":
        p = Path(path)
        raw_bytes = p.read_bytes()
        records: dict[int, CorpusRecord] = {}
        answer_groups: dict[str, list[int]] = {}
        question_groups: dict[str, list[int]] = {}

        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for doc_id, row in enumerate(reader):
                q = (row.get("Question") or "").strip()
                cat = (row.get("Category") or "").strip()
                brief = (row.get("BriefAnswer") or "").strip()
                ans = (row.get("Answer") or "").strip()
                kw = (row.get("Keyword") or "").strip()

                flags: list[str] = []
                if not q: flags.append("MISSING_QUESTION")
                if not brief: flags.append("MISSING_BRIEF")
                if not ans: flags.append("MISSING_ANSWER")
                if not kw: flags.append("EMPTY_KEYWORD")

                row_hash = _sha256((str(doc_id), q, cat, brief, ans, kw))
                records[doc_id] = CorpusRecord(
                    source_doc_id=doc_id,
                    question=q,
                    category=cat,
                    brief_answer=brief,
                    answer=ans,
                    keyword=kw,
                    source_row_hash=row_hash,
                    quality_flags=tuple(flags),
                )

                nq = normalize_match_text(q)
                na = normalize_match_text(brief or ans)
                if nq:
                    question_groups.setdefault(nq, []).append(doc_id)
                if na:
                    answer_groups.setdefault(na, []).append(doc_id)

        return cls(
            records=records,
            corpus_hash=hashlib.sha256(raw_bytes).hexdigest(),
            exact_answer_groups={k: tuple(v) for k, v in answer_groups.items() if len(v) > 1},
            exact_question_groups={k: tuple(v) for k, v in question_groups.items() if len(v) > 1},
        )

    def get(self, doc_id: int) -> CorpusRecord | None:
        return self.records.get(int(doc_id))

    def is_trusted(self, doc_id: int) -> bool:
        return int(doc_id) in self.records

    def rehydrate_hit(self, hit: dict[str, Any]) -> dict[str, Any] | None:
        record = self.get(int(hit.get("doc_id", -1)))
        if record is None:
            return None

        remote_text = str(hit.get("text") or "")
        trusted_text = record.answer or record.brief_answer
        remote_n = normalize_match_text(remote_text)
        trusted_n = normalize_match_text(trusted_text)
        question_n = normalize_match_text(record.question)
        content_matches = bool(
            remote_n
            and (remote_n in trusted_n or trusted_n in remote_n or remote_n in question_n or question_n in remote_n)
        )

        out = dict(hit)
        out.update({
            "doc_id": record.source_doc_id,
            "text": trusted_text,
            "category": record.category,
            "question": record.question,
            "brief_answer": record.brief_answer,
            "keyword": record.keyword,
            "trusted": True,
            "citation_eligible": True,
            "provenance_status": "VERIFIED" if content_matches else "VERIFIED_REHYDRATED",
            "source_row_hash": record.source_row_hash,
            "remote_text_hash": hashlib.sha256(remote_text.encode("utf-8")).hexdigest() if remote_text else "",
        })
        return out


def default_corpus_path() -> Path:
    return Path(__file__).resolve().parent.parent / "corpus" / "test.csv"


_REGISTRY: TrustedCorpusRegistry | None = None


def get_registry(path: str | Path | None = None, *, refresh: bool = False) -> TrustedCorpusRegistry:
    global _REGISTRY
    if path is not None:
        return TrustedCorpusRegistry.from_csv(path)
    if refresh or _REGISTRY is None:
        p = default_corpus_path()
        _REGISTRY = TrustedCorpusRegistry.from_csv(p) if p.exists() else TrustedCorpusRegistry()
    return _REGISTRY
