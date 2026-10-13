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
    if debts:

        has_effect_focus = any(
            cue in text
            for cue in _EFFECT_CUES
        )

        strong_removal_focus = any(
            cue in text
            for cue in (
                "حذف",
                "پاک",
                "رفع",
                "تسویه",
            )
        )

        payment_event = any(
            cue in text
            for cue in (
                "پرداخت کنم",
                "پرداخت کردم",
                "پرداخت شود",
                "پرداخت بشود",
                "پرداخت بشه",
            )
        )

        post_payment_outcome = any(
            cue in text
            for cue in (
                "چه اتفاق",
                "چه می شود",
                "چه می‌شود",
                "چی میشه",
                "چی می شود",
                "چی می‌شود",
                "چه زمانی",
                "چقدر زمان",
                "بعد از",
                "پس از",
            )
        )

        if has_effect_focus:
            return Intent.DEBT_EFFECT

        if (
            strong_removal_focus
            or (
                payment_event
                and post_payment_outcome
            )
        ):
            return Intent.DEBT_REMOVAL

        if any(
            cue in text
            for cue in (
                "چه اتفاق",
                "چی میشه",
                "چی می شود",
                "چی می‌شود",
            )
        ):
            return Intent.DEBT_EFFECT
    if "بدهی" in text and not debts:
        return Intent.DEBT_EFFECT

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


_INTERROGATIVE_CUES = (
    "آیا",
    "چرا",
    "چطور",
    "چگونه",
    "چه تاثیری",
    "چه تأثیری",
    "چه اتفاق",
    "چه می شود",
    "چه می‌شود",
    "چی میشه",
    "چی می شود",
    "چی می‌شود",
)

_EXISTENCE_NEGATIVE_CUES = (
    "ندارم",
    "ندارد",
    "ندارند",
    "نداشت",
    "نداشتم",
    "نداشته",
    "نیست",
    "وجود ندارد",
)

_EXISTENCE_POSITIVE_CUES = (
    "دارم",
    "دارد",
    "دارند",
    "داشتم",
    "داشت",
    "داشته",
    "وجود دارد",
)


def _split_surface_clauses(
    surface_text: str,
) -> list[str]:

    marked = surface_text

    marked = re.sub(
        r"([؛;.!?؟]+)",
        r"\1\n",
        marked,
    )

    marked = re.sub(
        r"\b(?:ولی|اما)\b",
        "\n",
        marked,
    )

    marked = re.sub(
        r"[,،]\s*(?=(?:آیا|چرا|چطور|چگونه)\b)",
        "\n",
        marked,
    )

    marked = re.sub(
        r"\s+(?=(?:آیا|چرا|چطور|چگونه)\b)",
        "\n",
        marked,
    )

    return [
        clause.strip()
        for clause in marked.splitlines()
        if clause.strip()
    ]


def _clause_is_interrogative(
    surface_clause: str,
    match_clause: str,
) -> bool:
    if (
        "؟" in surface_clause
        or "?" in surface_clause
    ):
        return True

    return any(
        cue in match_clause
        for cue in _INTERROGATIVE_CUES
    )


def _word_spans(
    text: str,
    phrases: tuple[str, ...],
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []

    for phrase in phrases:
        normalized = normalize_match_text(
            phrase
        )

        if not normalized:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized)
            + r"(?!\w)"
        )

        for match in re.finditer(
            pattern,
            text,
        ):
            spans.append(
                (
                    match.start(),
                    match.end(),
                    normalized,
                )
            )

    return spans


def _span_distance(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> int:
    if b_end <= a_start:
        return a_start - b_end

    if b_start >= a_end:
        return b_start - a_end

    return 0


def _effect_expression_spans(
    text: str,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []

    for effect_cue in _EFFECT_CUES:
        normalized = normalize_match_text(
            effect_cue
        )

        if not normalized:
            continue

        for match in re.finditer(
            re.escape(normalized),
            text,
        ):
            spans.append(
                (
                    match.start(),
                    match.end(),
                )
            )

    return spans


def _existence_cue_belongs_to_effect(
    text: str,
    cue_start: int,
    cue_end: int,
) -> bool:

    for effect_start, effect_end in (
        _effect_expression_spans(text)
    ):
        if (
            cue_start < effect_end
            and cue_end > effect_start
        ):
            return True

        if (
            effect_end <= cue_start
            and cue_start - effect_end <= 24
        ):
            return True

        if (
            effect_start <= cue_start
            and cue_start - effect_start <= 40
        ):
            return True

    return False


def _debt_polarity(
    clause: str,
    debt: EntityMention,
    *,
    interrogative: bool,
) -> tuple[Polarity, str | None]:

    terms = tuple(
        dict.fromkeys(
            normalize_match_text(x)
            for x in (
                debt.raw_text,
                debt.canonical_value,
            )
            if x
        )
    )

    mention_spans: list[
        tuple[int, int, str]
    ] = []

    for term in terms:
        if not term:
            continue

        for match in re.finditer(
            re.escape(term),
            clause,
        ):
            mention_spans.append(
                (
                    match.start(),
                    match.end(),
                    term,
                )
            )

    if not mention_spans:
        return (
            Polarity.UNKNOWN
            if interrogative
            else Polarity.POSITIVE,
            None,
        )

    cue_candidates: list[
        tuple[int, int, Polarity, str]
    ] = []

    for cue_start, cue_end, cue in _word_spans(
        clause,
        _EXISTENCE_NEGATIVE_CUES,
    ):
        if _existence_cue_belongs_to_effect(
            clause,
            cue_start,
            cue_end,
        ):
            continue

        for mention_start, mention_end, _ in mention_spans:
            distance = _span_distance(
                mention_start,
                mention_end,
                cue_start,
                cue_end,
            )

            cue_candidates.append(
                (
                    distance,
                    0,
                    Polarity.NEGATIVE,
                    cue,
                )
            )

    for cue_start, cue_end, cue in _word_spans(
        clause,
        _EXISTENCE_POSITIVE_CUES,
    ):
        if _existence_cue_belongs_to_effect(
            clause,
            cue_start,
            cue_end,
        ):
            continue

        for mention_start, mention_end, _ in mention_spans:
            distance = _span_distance(
                mention_start,
                mention_end,
                cue_start,
                cue_end,
            )

            cue_candidates.append(
                (
                    distance,
                    1,
                    Polarity.POSITIVE,
                    cue,
                )
            )

    if cue_candidates:
        distance, _, polarity, cue = min(
            cue_candidates,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        if distance <= 28:
            return polarity, cue

    return (
        (
            Polarity.UNKNOWN
            if interrogative
            else Polarity.POSITIVE
        ),
        None,
    )


def _effect_polarity(
    clause: str,
    *,
    interrogative: bool,
) -> tuple[Polarity, str | None]:

    effect_spans: list[
        tuple[int, int]
    ] = []

    for cue in _EFFECT_CUES:
        normalized = normalize_match_text(
            cue
        )

        if not normalized:
            continue

        for match in re.finditer(
            re.escape(normalized),
            clause,
        ):
            effect_spans.append(
                (
                    match.start(),
                    match.end(),
                )
            )

    if not effect_spans:
        return (
            Polarity.UNKNOWN
            if interrogative
            else Polarity.POSITIVE,
            None,
        )

    for effect_start, effect_end in effect_spans:
        window = clause[
            max(0, effect_start - 6):
            min(
                len(clause),
                effect_end + 36,
            )
        ]

        for cue in _NEGATION_CUES:
            normalized = normalize_match_text(
                cue
            )

            if normalized and normalized in window:
                return (
                    Polarity.NEGATIVE,
                    normalized,
                )

    if interrogative:
        return Polarity.UNKNOWN, None

    return Polarity.POSITIVE, None


def _propositions(
    surface_text: str,
    debts: list[EntityMention],
) -> list[Proposition]:

    result: list[Proposition] = []

    for surface_clause in _split_surface_clauses(
        surface_text
    ):
        clause = normalize_match_text(
            surface_clause
        )

        if not clause:
            continue

        interrogative = _clause_is_interrogative(
            surface_clause,
            clause,
        )

        for debt in debts:
            terms = {
                normalize_match_text(
                    debt.raw_text or ""
                ),
                normalize_match_text(
                    debt.canonical_value or ""
                ),
            }

            terms.discard("")

            if not any(
                term in clause
                for term in terms
            ):
                continue

            polarity, cue = _debt_polarity(
                clause,
                debt,
                interrogative=interrogative,
            )

            result.append(
                Proposition(
                    concept=(
                        f"DEBT:{debt.canonical_value}"
                    ),
                    polarity=polarity,
                    cue=cue,
                )
            )

        if any(
            normalize_match_text(cue)
            in clause
            for cue in _EFFECT_CUES
            if normalize_match_text(cue)
        ):
            polarity, cue = _effect_polarity(
                clause,
                interrogative=interrogative,
            )

            result.append(
                Proposition(
                    concept="EFFECT",
                    polarity=polarity,
                    cue=cue,
                )
            )

    return result


def _ambiguity(intent: Intent, text: str, debts: list[EntityMention], ranks: list[EntityMention]) -> AmbiguityInfo:
    missing: list[str] = []
    conflicts: list[str] = []

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
    propositions = _propositions(
        norm_result.original_text,
        debts,
    )
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
    return {"BANK": None, "RANK": None, "PERSONA": None, "DEBT": None,
            "AMOUNT": None, "TIME": None}
