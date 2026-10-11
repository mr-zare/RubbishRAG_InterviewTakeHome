"""Typed Persian query understanding for the credit-scoring domain.

`extract_slots()` preserves the original dictionary API. `parse_query()` is the
new typed semantic-frame entry point used by later pipeline phases.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Iterable

from .models import (
    AmbiguityInfo,
    Comparator,
    DebtType,
    EntityMention,
    EntityType,
    Intent,
    Persona,
    Polarity,
    Proposition,
    Quantity,
    QuantityType,
    QuerySemanticFrame,
    Relation,
)
from .normalize_fa import normalize_match_text, normalize_text


# Public compatibility constants.
BANKS = [
    "ملی", "صادرات", "تجارت", "توسعه صادرات", "ملت", "سپه",
    "پاسارگاد", "دیجی پی", "دیجی‌پی", "اسنپ پی", "اسنپ‌پی", "تارا",
    "دیجیکالا", "دیجی‌کالا",
]
RANKS = [f"{letter}{number}" for letter in "ABCDE" for number in "123"]
DEBT_PATTERNS = [
    ("چک", ["چک", "صیادی", "برگشتی", "رفع سوءاثر", "رفع سواثر"]),
    ("مالیاتی", ["مالیات", "جریمه مالیاتی", "امور مالیاتی"]),
    ("تسهیلاتی", ["تسهیلات", "وام", "قسط", "اقساط", "معوق"]),
    ("محکومیت", ["محکومیت", "ورشکستگی", "دادگاه", "قضایی"]),
]

_BANK_REGISTRY: dict[str, tuple[str, ...]] = {
    "BANK_TOSSE_SADERAT": ("بانک توسعه صادرات", "توسعه صادرات"),
    "BANK_MELLI": ("بانک ملی", "ملی"),
    "BANK_SADERAT": ("بانک صادرات", "صادرات"),
    "BANK_TEJARAT": ("بانک تجارت", "تجارت"),
    "BANK_MELLAT": ("بانک ملت", "ملت"),
    "BANK_SEPAH": ("بانک سپه", "سپه"),
    "BANK_PASARGAD": ("بانک پاسارگاد", "پاسارگاد"),
    "BANK_DIGIPAY": ("دیجی پی", "دیجی‌پی"),
    "BANK_SNAPPAY": ("اسنپ پی", "اسنپ‌پی"),
    "BANK_TARA": ("تارا",),
    "BANK_DIGIKALA": ("دیجیکالا", "دیجی کالا", "دیجی‌کالا"),
}
_BANK_CANONICAL = {
    "BANK_TOSSE_SADERAT": "توسعه صادرات",
    "BANK_MELLI": "ملی",
    "BANK_SADERAT": "صادرات",
    "BANK_TEJARAT": "تجارت",
    "BANK_MELLAT": "ملت",
    "BANK_SEPAH": "سپه",
    "BANK_PASARGAD": "پاسارگاد",
    "BANK_DIGIPAY": "دیجی پی",
    "BANK_SNAPPAY": "اسنپ پی",
    "BANK_TARA": "تارا",
    "BANK_DIGIKALA": "دیجیکالا",
}

_PERSONA_REGISTRY: dict[str, tuple[str, ...]] = {
    "PERSONA_LEGAL": ("شخص حقوقی", "اشخاص حقوقی", "کسب و کار", "کسب وکار", "شرکت", "سازمان"),
    "PERSONA_INDIVIDUAL": ("شخص حقیقی", "اشخاص حقیقی", "فرد", "خودم"),
}

_DEBT_REGISTRY: dict[str, tuple[str, tuple[str, ...]]] = {
    "DEBT_CHECK": (DebtType.CHECK.value, ("چک برگشتی", "چک صیادی", "رفع سواثر", "رفع سوءاثر", "برگشتی", "چک")),
    "DEBT_TAX": (DebtType.TAX.value, ("بدهی مالیاتی", "جریمه مالیاتی", "امور مالیاتی", "مالیات")),
    "DEBT_LOAN": (DebtType.LOAN.value, ("اقساط معوق", "بدهی تسهیلاتی", "تسهیلات", "وام", "اقساط", "قسط", "معوق")),
    "DEBT_LEGAL": (DebtType.LEGAL.value, ("محکومیت مالی", "محکومیت", "ورشکستگی", "دادگاه", "قضایی")),
    "DEBT_SUPPLIER": (
        DebtType.SUPPLIER.value,
        (
            "بدهی تامین کنندگان",
            "بدهی تأمین کنندگان",
            "بدهی تامین‌کنندگان",
            "بدهی تأمین‌کنندگان",
            "تامین کنندگان",
            "تأمین کنندگان",
            "تامین‌کنندگان",
            "تأمین‌کنندگان",
            "بدهی به تامین کننده",
            "بدهی به تأمین کننده",
        ),
    ),
}

_NEGATION_CUES = ("نمی", "نیست", "فاقد", "بدون", "ندارم", "ندارد", "نداشت", "عدم", "معاف", "بی تاثیر", "بی تأثیر", "بی اثر")
_EFFECT_CUES = ("اثر", "تاثیر", "موثر", "تأثیر")
_REMOVAL_CUES = ("حذف", "پاک", "رفع", "تسویه", "پرداخت")
_TIME_UNITS = {
    "ساعت": "hour",
    "روز": "day",
    "ماه": "month",
    "سال": "year",
}
_MONEY_UNITS = {"تومان": "toman", "ریال": "rial"}
_MULTIPLIERS = {"هزار": 1_000, "میلیون": 1_000_000, "میلیارد": 1_000_000_000}


def _find_alias_mentions(
    text: str,
    entity_type: EntityType,
    registry: dict[str, tuple[str, ...]],
    canonical_values: dict[str, str] | None = None,
) -> list[EntityMention]:
    candidates: list[EntityMention] = []
    for canonical_id, aliases in registry.items():
        for alias in sorted(aliases, key=len, reverse=True):
            alias_n = normalize_match_text(alias)
            if not alias_n:
                continue
            for match in re.finditer(rf"(?<!\w){re.escape(alias_n)}(?!\w)", text):
                candidates.append(EntityMention(
                    entity_type=entity_type,
                    canonical_id=canonical_id,
                    canonical_value=(canonical_values or {}).get(canonical_id, alias_n),
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    match_type="alias",
                ))
    # Longest/specific non-overlapping spans win. This prevents "صادرات" from
    # stealing "توسعه صادرات".
    candidates.sort(key=lambda m: (-(m.end - m.start), m.start, m.canonical_id))
    selected: list[EntityMention] = []
    for cand in candidates:
        if any(not (cand.end <= x.start or cand.start >= x.end) for x in selected):
            continue
        selected.append(cand)
    return sorted(selected, key=lambda m: (m.start, m.end))


def _filter_bank_mentions(
    text: str,
    mentions: list[EntityMention],
) -> list[EntityMention]:
    """Remove contextually obvious non-bank uses of ambiguous bank aliases."""

    export_cues = (
        "واردات",
        "صادرات/واردات",
        "صادرات و واردات",
        "تعهد ارزی",
        "حجم صادرات",
        "مبلغ صادرات",
        "عدد صادرات",
    )

    out: list[EntityMention] = []

    for mention in mentions:
        raw = normalize_match_text(mention.raw_text)

        # "صادرات" may mean Bank Saderat or the economic activity.
        # Suppress only when context clearly describes export/import and
        # there is no explicit bank cue nearby.
        if mention.canonical_id == "BANK_SADERAT" and raw == "صادرات":
            window = text[
                max(0, mention.start - 24):
                min(len(text), mention.end + 32)
            ]

            if (
                "بانک" not in window
                and any(cue in window for cue in export_cues)
            ):
                continue

        out.append(mention)

    return out


def _rank_mentions(text: str) -> list[EntityMention]:
    result: list[EntityMention] = []
    for match in re.finditer(r"(?<![A-Za-z0-9])([A-Ea-e])\s*([1-3])(?![A-Za-z0-9])", text):
        value = f"{match.group(1).upper()}{match.group(2)}"
        result.append(EntityMention(
            entity_type=EntityType.RANK,
            canonical_id=f"RANK_{value}",
            canonical_value=value,
            raw_text=match.group(0),
            start=match.start(),
            end=match.end(),
            match_type="regex",
        ))
    return result


def _persona_mentions(text: str) -> list[EntityMention]:
    raw = _find_alias_mentions(text, EntityType.PERSONA, _PERSONA_REGISTRY)
    mapped: list[EntityMention] = []
    for m in raw:
        value = Persona.LEGAL.value if m.canonical_id == "PERSONA_LEGAL" else Persona.INDIVIDUAL.value
        mapped.append(replace(m, canonical_value=value))
    return mapped


def _debt_mentions(text: str) -> list[EntityMention]:
    registry = {k: aliases for k, (_, aliases) in _DEBT_REGISTRY.items()}
    canonical = {k: value for k, (value, _) in _DEBT_REGISTRY.items()}
    return _find_alias_mentions(text, EntityType.DEBT, registry, canonical)


def _relation(text: str, entities_count: int) -> Relation:
    if any(x in text for x in ("تفاوت", "مقایسه", "بهتر از", "در مقابل")):
        return Relation.COMPARISON
    if " یا " in f" {text} ":
        return Relation.ALTERNATIVE
    if entities_count > 1:
        return Relation.LIST
    return Relation.SINGLE


def _parse_comparator(prefix: str) -> Comparator:
    if any(x in prefix for x in ("حداقل", "کمتر نباش", "بیشتر یا مساوی")):
        return Comparator.GE
    if any(x in prefix for x in ("حداکثر", "بیشتر نباش", "کمتر یا مساوی")):
        return Comparator.LE
    if any(x in prefix for x in ("بیشتر از", "بالای")):
        return Comparator.GT
    if any(x in prefix for x in ("کمتر از", "زیر")):
        return Comparator.LT
    return Comparator.EQ


def _parse_quantities(text: str) -> list[Quantity]:
    out: list[Quantity] = []
    occupied: list[tuple[int, int]] = []

    # Explicit numeric ranges first.
    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*(?:تا|الی|-)\s*(\d+(?:[.,]\d+)?)\b", text):
        v1 = float(m.group(1).replace(",", ""))
        v2 = float(m.group(2).replace(",", ""))
        if v1.is_integer(): v1 = int(v1)
        if v2.is_integer(): v2 = int(v2)
        nearby = text[max(0, m.start()-20):min(len(text), m.end()+20)]
        qtype = QuantityType.CREDIT_SCORE if any(w in nearby for w in ("امتیاز", "نمره", "رتبه")) else QuantityType.GENERIC_NUMBER
        out.append(Quantity(qtype, v1, "score" if qtype is QuantityType.CREDIT_SCORE else None,
                            Comparator.BETWEEN, m.start(), m.end(), m.group(0), v2))
        occupied.append((m.start(), m.end()))

    num_re = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    for m in num_re.finditer(text):
        if any(a <= m.start() < b for a, b in occupied):
            continue
        raw = m.group(0)
        val_f = float(raw.replace(",", ""))
        value: float | int = int(val_f) if val_f.is_integer() else val_f
        left = text[max(0, m.start()-22):m.start()]
        right = text[m.end():min(len(text), m.end()+28)]
        window = (left + " " + right).strip()
        comparator = _parse_comparator(left[-18:])

        qtype = QuantityType.GENERIC_NUMBER
        unit: str | None = None
        final_value: float | int = value

        time_unit = next((u for fa, u in _TIME_UNITS.items() if re.search(rf"^\s*{fa}\b", right)), None)
        money_unit = next((u for fa, u in _MONEY_UNITS.items() if fa in right[:20]), None)
        multiplier = next((mult for fa, mult in _MULTIPLIERS.items() if fa in right[:20]), None)

        if time_unit:
            qtype, unit = QuantityType.DURATION, time_unit
        elif money_unit:
            qtype, unit = QuantityType.MONEY, money_unit
            if multiplier:
                final_value = value * multiplier
        elif "%" in right[:5] or "درصد" in right[:12]:
            qtype, unit = QuantityType.PERCENTAGE, "percent"
        elif 1300 <= value <= 1500 and any(w in window for w in ("سال", "تاریخ", "از", "تا")):
            qtype, unit = QuantityType.YEAR, "year"
        elif any(w in window for w in ("امتیاز", "نمره", "رتبه", "شدم", "شده")) or (200 <= value <= 900):
            qtype, unit = QuantityType.CREDIT_SCORE, "score"
        elif any(w in window for w in ("تعداد", "چند", "بار", "فقره")):
            qtype = QuantityType.COUNT

        out.append(Quantity(qtype, final_value, unit, comparator, m.start(), m.end(), raw))
    return out


def _intent(text: str, ranks: list[EntityMention], debts: list[EntityMention], quantities: list[Quantity]) -> Intent:
    if any(w in text for w in ("تفاوت", "مقایسه", "بهتر از", "در مقابل")):
        return Intent.COMPARISON
    if (ranks or "رتبه" in text) and any(w in text for w in ("یعنی چی", "چیست", "معنی", "معنا", "چه معنایی", "بازه")):
        return Intent.DEFINE_RANK
    if any(q.quantity_type is QuantityType.CREDIT_SCORE for q in quantities) and any(w in text for w in ("چرا", "علت", "افت", "کم", "پایین", "شدم", "شده")):
        return Intent.EXPLAIN_SCORE
    if debts and any(w in text for w in _REMOVAL_CUES):
        return Intent.DEBT_REMOVAL
    if debts and (any(w in text for w in _EFFECT_CUES) or "چه اتفاق" in text or "چی میشه" in text or "چی می شود" in text):
        return Intent.DEBT_EFFECT
    # Generic debt mentions are intentionally routed to a debt intent so the
    # ambiguity gate can ask for the missing subtype instead of guessing.
    if "بدهی" in text and not debts:
        return Intent.DEBT_EFFECT

    # A score-explanation question does not need to contain a literal numeric
    # score. Examples include "چرا امتیاز ندارم؟" or "چرا بالاترین رتبه را
    # ندارم؟". Keep explicit factor/effect questions out of this branch.
    causal_score_question = (
        any(term in text for term in ("امتیاز", "رتبه", "نمره"))
        and any(cue in text for cue in ("چرا", "علت", "دلیل"))
        and not any(cue in text for cue in _EFFECT_CUES)
    )
    if causal_score_question:
        return Intent.EXPLAIN_SCORE

    if any(w in text for w in ("وام", "تسهیلات")) and any(w in text for w in ("می دهد", "میده", "بگیر", "گرفتن", "شرایط", "دریافت")):
        return Intent.LOAN_ELIGIBILITY
    if any(w in text for w in ("گزارش", "ثبت می شود", "ثبت میشود", "در سابقه")):
        return Intent.REPORT_CONTENT
    if any(w in text for w in ("اصلاح", "تصحیح", "اعتراض", "ویرایش اطلاعات")):
        return Intent.DATA_CORRECTION
    if any(w in text for w in ("چقدر زمان", "چه مدت", "کی", "چه زمانی", "ساعت", "روز", "ماه", "سال")) and any(w in text for w in ("حذف", "باقی", "زمان", "مدت")):
        return Intent.TIMING
    if any(w in text for w in _EFFECT_CUES):
        return Intent.FACTOR_EFFECT
    return Intent.GENERAL_INFO


def _propositions(text: str, debts: list[EntityMention]) -> list[Proposition]:
    # Clause-aware, deterministic scope approximation. This is intentionally
    # conservative: UNKNOWN is preferable to assigning negation to the wrong concept.
    clauses = [c.strip() for c in re.split(r"(?:\bولی\b|\bاما\b|[؛;.!?؟])", text) if c.strip()]
    result: list[Proposition] = []
    for clause in clauses:
        is_neg = any(cue in clause for cue in _NEGATION_CUES)
        is_question = any(cue in clause for cue in ("آیا", "چه تاثیری", "چه تأثیری", "موثره", "موثر است", "تاثیر دارد"))
        polarity = Polarity.NEGATIVE if is_neg else (Polarity.UNKNOWN if is_question else Polarity.POSITIVE)
        matched_debts = [d for d in debts if d.canonical_value in clause or d.raw_text in clause]
        for debt in matched_debts:
            cue = next((x for x in _NEGATION_CUES if x in clause), None)
            result.append(Proposition(concept=f"DEBT:{debt.canonical_value}", polarity=polarity, cue=cue))

        # Generic effect proposition for polarity track.
        if any(x in clause for x in _EFFECT_CUES):
            cue = next((x for x in _NEGATION_CUES if x in clause), None)
            result.append(Proposition(concept="EFFECT", polarity=polarity, cue=cue))
    return result


def _ambiguity(intent: Intent, text: str, debts: list[EntityMention], ranks: list[EntityMention]) -> AmbiguityInfo:
    missing: list[str] = []
    conflicts: list[str] = []

    # Only decision-critical missing slots trigger clarification. Avoid asking for
    # irrelevant information merely because a slot is absent.
    if intent in (Intent.DEBT_EFFECT, Intent.DEBT_REMOVAL) and not debts and "بدهی" in text:
        missing.append("debt_type")
    if intent is Intent.DEFINE_RANK and not ranks:
        missing.append("rank")

    return AmbiguityInfo(
        missing_required_slots=tuple(missing),
        conflicting_slots=tuple(conflicts),
        underspecified=bool(missing or conflicts),
    )


def parse_query(query: str | None) -> QuerySemanticFrame:
    norm_result = normalize_text(query)
    text = norm_result.match_text

    banks = _filter_bank_mentions(
        text,
        _find_alias_mentions(
            text,
            EntityType.BANK,
            _BANK_REGISTRY,
            _BANK_CANONICAL,
        ),
    )
    ranks = _rank_mentions(text)
    personas = _persona_mentions(text)
    debts = _debt_mentions(text)
    quantities = _parse_quantities(text)

    entities_count = len(banks) + len(ranks) + len(personas) + len(debts)
    relation = _relation(text, entities_count)
    intent = _intent(text, ranks, debts, quantities)
    propositions = _propositions(text, debts)
    ambiguity = _ambiguity(intent, text, debts, ranks)

    return QuerySemanticFrame(
        original_text=norm_result.original_text,
        canonical_text=norm_result.canonical_text,
        match_text=norm_result.match_text,
        intent=intent,
        relation=relation,
        banks=banks,
        ranks=ranks,
        personas=personas,
        debt_types=debts,
        quantities=quantities,
        propositions=propositions,
        ambiguity=ambiguity,
        normalization_changes=norm_result.transformations,
    )


def extract_slots(query: str) -> dict[str, Any]:
    """Backward-compatible slot dictionary derived from the typed frame."""
    frame = parse_query(query)
    bank_values = frame.values(EntityType.BANK)
    rank_values = frame.values(EntityType.RANK)
    persona_values = frame.values(EntityType.PERSONA)
    debt_values = frame.values(EntityType.DEBT)

    nums = [str(int(q.value) if isinstance(q.value, float) and q.value.is_integer() else q.value)
            for q in frame.quantities if q.value is not None]

    return {
        "BANK": bank_values[0] if bank_values else None,
        "BANKS": bank_values,
        "RANK": rank_values[0] if rank_values else None,
        "RANKS": rank_values,
        "PERSONA": persona_values[0] if len(persona_values) == 1 else ("both" if len(set(persona_values)) > 1 else None),
        "PERSONAS": persona_values,
        "DEBT": debt_values[0] if debt_values else None,
        "DEBTS": debt_values,
        "AMOUNT": nums[0] if nums else None,
        "TIME": any(q.quantity_type is QuantityType.DURATION for q in frame.quantities) or any(x in frame.match_text for x in _TIME_UNITS),
        "NUMS": nums,
        "NEGATED": any(p.polarity is Polarity.NEGATIVE for p in frame.propositions) or any(c in frame.match_text for c in _NEGATION_CUES),
        "INTENT": frame.intent.value,
        "AMBIGUOUS": frame.ambiguity.underspecified,
        "MISSING_SLOTS": list(frame.ambiguity.missing_required_slots),
    }


def extract_slots_broken(query: str) -> dict:
    """Starter compatibility shim; retained only for imports in untouched code."""
    return {"BANK": None, "RANK": None, "PERSONA": None, "DEBT": None,
            "AMOUNT": None, "TIME": None}
