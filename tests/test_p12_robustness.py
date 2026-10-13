from rubbish_rag.corpus import CorpusRecord
from rubbish_rag.models import (
    EntityType,
    Intent,
    Polarity,
    QuantityType,
)
from rubbish_rag.normalize_fa import normalize_match_text
from rubbish_rag.rerank import score_record
from rubbish_rag.slots import parse_query


def _effect_polarities(frame):
    return [
        p.polarity
        for p in frame.propositions
        if p.concept == "EFFECT"
    ]



def test_p12_arabic_and_persian_character_forms_are_equivalent():
    a = normalize_match_text(
        "بانك صادرات و اعتبارسنجي"
    )
    b = normalize_match_text(
        "بانک صادرات و اعتبارسنجی"
    )

    assert a == b


def test_p12_persian_and_ascii_digits_are_equivalent():
    a = normalize_match_text(
        "امتیاز من ۴۸۰ شده است"
    )
    b = normalize_match_text(
        "امتیاز من 480 شده است"
    )

    assert a == b


def test_p12_zwnj_and_space_variants_are_equivalent():
    a = normalize_match_text(
        "چه اتفاقی می\u200cافتد"
    )
    b = normalize_match_text(
        "چه اتفاقی می افتد"
    )

    assert a == b



def test_p12_script_variant_preserves_bank_entity():
    a = parse_query(
        "بانك صادرات وام میده؟"
    )
    b = parse_query(
        "بانک صادرات وام میده؟"
    )

    assert (
        a.values(EntityType.BANK)
        == b.values(EntityType.BANK)
    )


def test_p12_digit_variant_preserves_credit_score_quantity():
    a = parse_query(
        "امتیاز اعتباری من ۴۸۰ شده، چرا؟"
    )
    b = parse_query(
        "امتیاز اعتباری من 480 شده، چرا؟"
    )

    aq = [
        q
        for q in a.quantities
        if q.quantity_type
        is QuantityType.CREDIT_SCORE
    ]

    bq = [
        q
        for q in b.quantities
        if q.quantity_type
        is QuantityType.CREDIT_SCORE
    ]

    assert aq
    assert bq

    assert aq[0].value == bq[0].value
    assert a.intent is b.intent


def test_p12_late_payment_effect_paraphrases_preserve_intent():
    a = parse_query(
        "اقساط را با تأخیر پرداخت کردم؛ "
        "آیا این تأخیر روی امتیاز اثر دارد؟"
    )

    b = parse_query(
        "قسط دیر پرداخت شده ولی بعداً تسویه شده؛ "
        "آیا دیرکرد روی رتبه تاثیر دارد؟"
    )

    assert a.intent is Intent.DEBT_EFFECT
    assert b.intent is Intent.DEBT_EFFECT



def test_p12_effect_polarity_reversal_changes_semantics():
    positive = parse_query(
        "بدهی مالیاتی روی امتیاز تاثیر دارد."
    )

    negative = parse_query(
        "بدهی مالیاتی روی امتیاز تاثیر ندارد."
    )

    assert (
        Polarity.POSITIVE
        in _effect_polarities(positive)
    )

    assert (
        Polarity.NEGATIVE
        in _effect_polarities(negative)
    )


def test_p12_rank_swap_produces_reranker_conflict():
    query = "رتبه C1 یعنی چی؟"

    wrong = CorpusRecord(
        source_doc_id=94001,
        question="رتبه C2 یعنی چی؟",
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="رتبه C2",
        source_row_hash="p12-rank-swap",
    )

    _, parts = score_record(
        query,
        parse_query(query),
        wrong,
        registry=None,
    )

    assert (
        parts.get("rank_reason")
        == "RANK_CONFLICT"
    )


def test_p12_bank_swap_produces_reranker_conflict():
    query = "بانک صادرات به من وام میده؟"

    wrong = CorpusRecord(
        source_doc_id=94002,
        question="بانک ملت به من وام میده؟",
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="بانک ملت",
        source_row_hash="p12-bank-swap",
    )

    _, parts = score_record(
        query,
        parse_query(query),
        wrong,
        registry=None,
    )

    assert (
        parts.get("bank_reason")
        == "BANK_CONFLICT"
    )



def test_p12_exact_question_anchor_survives_digit_normalization():
    query = "امتیاز من 480 شده، چرا؟"

    record = CorpusRecord(
        source_doc_id=94003,
        question="امتیاز من ۴۸۰ شده، چرا؟",
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="امتیاز ۴۸۰",
        source_row_hash="p12-digit-anchor",
    )

    _, parts = score_record(
        query,
        parse_query(query),
        record,
        registry=None,
    )

    assert (
        parts["question_exact_match"]
        == 3.0
    )



def test_p12_semantic_parsing_is_deterministic():
    query = (
        "اگر بدهی مالیاتی را تسویه کنم "
        "چه اتفاقی می افتد؟"
    )

    a = parse_query(query)
    b = parse_query(query)

    assert a.to_dict() == b.to_dict()
