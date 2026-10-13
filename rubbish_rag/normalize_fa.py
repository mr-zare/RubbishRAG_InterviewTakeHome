from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from .models import NormalizationChange, NormalizationResult


FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
_DIGIT_MAP = {ord(a): b for a, b in zip(FA_DIGITS + AR_DIGITS, EN_DIGITS * 2)}

_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ة": "ه",
    "ۀ": "ه",
})

COLLOQUIAL_MAP = {
    "میشه": "می شود",
    "نمیشه": "نمی شود",
    "چیه": "چیست",
    "چقد": "چقدر",
    "رتبم": "رتبه من",
    "رتبه م": "رتبه من",
    "امتیازم": "امتیاز من",
    "سابقه م": "سابقه من",
    "سابقم": "سابقه من",
    "شرکتم": "شرکت من",
    "چکم": "چک من",
    "وامم": "وام من",
    "قسطام": "اقساط من",
    "قسطامو": "اقساط من را",
    "گزارشم": "گزارش من",
    "گزارشمو": "گزارش من را",
    "مالیاتم": "مالیات من",
    "مالیاتمو": "مالیات من را",
    "میخوام": "می خواهم",
    "نمیخوام": "نمی خواهم",
    "میده": "می دهد",
    "میدن": "می دهند",
}

_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_CONTROL_CHARS = ("\ufeff", "\u200e", "\u200f", "\u2066", "\u2067", "\u2068", "\u2069")
_EXOTIC_SPACES_RE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_WS_RE = re.compile(r"\s+")

_SENTENCE_PUNCT_RE = re.compile(r"[؟?!؛;,.،]+")


def _record(changes: list[NormalizationChange], code: str, before: str, after: str) -> str:
    if before != after:
        changes.append(NormalizationChange(code=code, before=before, after=after))
    return after


def _replace_phrases(text: str, mapping: dict[str, str]) -> str:
    result = text
    for src in sorted(mapping, key=len, reverse=True):
        dst = mapping[src]
        pattern = rf"(?<!\w){re.escape(src)}(?!\w)"
        result = re.sub(pattern, dst, result)
    return result


def normalize_text(text: str | None) -> NormalizationResult:
    original = text or ""
    changes: list[NormalizationChange] = []

    s = unicodedata.normalize("NFC", original)
    if s != original:
        changes.append(NormalizationChange("UNICODE_NFC", original, s))

    before = s
    s = s.translate(_DIGIT_MAP)
    _record(changes, "DIGITS_TO_ASCII", before, s)

    before = s
    s = s.translate(_ARABIC_TO_PERSIAN)
    _record(changes, "ARABIC_TO_PERSIAN", before, s)

    before = s
    s = s.replace("ـ", "")
    _record(changes, "REMOVE_TATWEEL", before, s)

    for ch in _CONTROL_CHARS:
        if ch in s:
            before = s
            s = s.replace(ch, "")
            _record(changes, "REMOVE_BIDI_CONTROL", before, s)

    before = s
    s = _EXOTIC_SPACES_RE.sub(" ", s)
    _record(changes, "NORMALIZE_EXOTIC_SPACE", before, s)

    before = s
    s = re.sub(r"\s*\u200c\s*", "\u200c", s)
    _record(changes, "STANDARDIZE_ZWNJ", before, s)

    before = s
    s = _WS_RE.sub(" ", s).strip()
    _record(changes, "COLLAPSE_WHITESPACE", before, s)

    before = s
    colloquial_work = s.replace("\u200c", " ")
    colloquial_work = _replace_phrases(colloquial_work, COLLOQUIAL_MAP)
    colloquial_work = _WS_RE.sub(" ", colloquial_work).strip()
    s = colloquial_work
    _record(changes, "COLLOQUIAL_CANONICALIZATION", before, s)

    canonical = s

    match = canonical.replace("\u200c", " ")
    match = _DIACRITICS_RE.sub("", match)
    match = _SENTENCE_PUNCT_RE.sub(" ", match)
    match = _WS_RE.sub(" ", match).strip()

    return NormalizationResult(
        original_text=original,
        canonical_text=canonical,
        match_text=match,
        transformations=tuple(changes),
    )


def normalize_match_text(text: str | None) -> str:
    return normalize_text(text).match_text


def normalize_canonical_text(text: str | None) -> str:
    return normalize_text(text).canonical_text


def normalize_fa(text: str | None) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text).translate(_DIGIT_MAP).translate(_ARABIC_TO_PERSIAN)
    s = s.replace("ـ", "").replace("\u200c", " ")
    s = _WS_RE.sub(" ", s).strip()

    legacy_map = dict(COLLOQUIAL_MAP)
    legacy_map["میشه"] = "می‌شود"
    legacy_map["نمیشه"] = "نمی‌شود"
    legacy_map["رتبه م"] = "رتبه من"
    legacy_map["سابقه م"] = "سابقه من"
    s = _replace_phrases(s, legacy_map)
    return _WS_RE.sub(" ", s).strip()


def normalization_is_idempotent(samples: Iterable[str]) -> bool:
    return all(normalize_match_text(normalize_match_text(x)) == normalize_match_text(x) for x in samples)
