from rubbish_rag.corpus import CorpusRecord
from rubbish_rag.models import Polarity
from rubbish_rag.rerank import score_record
from rubbish_rag.slots import parse_query


def _polarities(frame, concept):
    return [
        p.polarity
        for p in frame.propositions
        if p.concept == concept
    ]


# ---------------------------------------------------------------------------
# Declarative / interrogative distinction
# ---------------------------------------------------------------------------

def test_g_positive_declarative_effect():
    frame = parse_query(
        "بدهی مالیاتی روی امتیاز تاثیر دارد."
    )

    assert (
        Polarity.POSITIVE
        in _polarities(frame, "EFFECT")
    )


def test_g_negative_declarative_effect():
    frame = parse_query(
        "بدهی مالیاتی روی امتیاز تاثیر ندارد."
    )

    assert (
        Polarity.NEGATIVE
        in _polarities(frame, "EFFECT")
    )


def test_g_positive_interrogative_effect_is_unknown():
    frame = parse_query(
        "آیا بدهی مالیاتی روی امتیاز تاثیر دارد؟"
    )

    assert (
        Polarity.UNKNOWN
        in _polarities(frame, "EFFECT")
    )


def test_g_explicit_negative_question_remains_negative():
    frame = parse_query(
        "آیا بدهی مالیاتی روی امتیاز تاثیر ندارد؟"
    )

    assert (
        Polarity.NEGATIVE
        in _polarities(frame, "EFFECT")
    )


# ---------------------------------------------------------------------------
# Clause-scoped modality
# ---------------------------------------------------------------------------

def test_g_positive_premise_is_not_erased_by_following_question():
    frame = parse_query(
        "بدهی مالیاتی دارم؛ "
        "آیا این بدهی روی امتیاز تاثیر دارد؟"
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    assert (
        Polarity.UNKNOWN
        in _polarities(frame, "EFFECT")
    )


def test_g_negative_premise_does_not_negate_following_question():
    frame = parse_query(
        "بدهی مالیاتی ندارم؛ "
        "آیا چک برگشتی روی امتیاز تاثیر دارد؟"
    )

    assert (
        Polarity.NEGATIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    check_polarities = _polarities(
        frame,
        "DEBT:چک",
    )

    assert Polarity.NEGATIVE not in check_polarities

    assert (
        Polarity.UNKNOWN
        in _polarities(frame, "EFFECT")
    )


def test_g_contrastive_debt_negation_remains_scoped():
    frame = parse_query(
        "بدهی مالیاتی ندارم "
        "ولی چک برگشتی دارم"
    )

    assert (
        Polarity.NEGATIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "DEBT:چک",
        )
    )


# ---------------------------------------------------------------------------
# Event negation must not negate entity existence
# ---------------------------------------------------------------------------

def test_g_unpaid_debt_still_means_debt_exists():
    frame = parse_query(
        "بدهی مالیاتی پرداخت نشده دارم"
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )


# ---------------------------------------------------------------------------
# Contrastive reranker behavior
# ---------------------------------------------------------------------------

def test_g_positive_vs_negative_effect_is_conflict():
    query = (
        "بدهی مالیاتی روی امتیاز "
        "تاثیر دارد."
    )

    record = CorpusRecord(
        source_doc_id=96001,
        question=(
            "بدهی مالیاتی روی امتیاز "
            "تاثیر ندارد."
        ),
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="بدهی مالیاتی",
        source_row_hash="g-positive-negative",
    )

    _, parts = score_record(
        query,
        parse_query(query),
        record,
        registry=None,
    )

    assert (
        parts.get("polarity_reason")
        == "POLARITY_CONFLICT"
    )


def test_g_negative_vs_positive_effect_is_conflict():
    query = (
        "بدهی مالیاتی روی امتیاز "
        "تاثیر ندارد."
    )

    record = CorpusRecord(
        source_doc_id=96002,
        question=(
            "بدهی مالیاتی روی امتیاز "
            "تاثیر دارد."
        ),
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="بدهی مالیاتی",
        source_row_hash="g-negative-positive",
    )

    _, parts = score_record(
        query,
        parse_query(query),
        record,
        registry=None,
    )

    assert (
        parts.get("polarity_reason")
        == "POLARITY_CONFLICT"
    )


def test_g_open_question_does_not_fake_polarity_conflict():
    query = (
        "آیا بدهی مالیاتی روی امتیاز "
        "تاثیر دارد؟"
    )

    record = CorpusRecord(
        source_doc_id=96003,
        question=(
            "بدهی مالیاتی روی امتیاز "
            "تاثیر دارد."
        ),
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="بدهی مالیاتی",
        source_row_hash="g-question-positive",
    )

    _, parts = score_record(
        query,
        parse_query(query),
        record,
        registry=None,
    )

    assert (
        parts.get("polarity_reason")
        != "POLARITY_CONFLICT"
    )


# ---------------------------------------------------------------------------
# Answer polarity must not contaminate question polarity
# ---------------------------------------------------------------------------

def test_g_answer_negation_cannot_create_question_conflict():
    query = (
        "آیا بدهی مالیاتی روی امتیاز "
        "تاثیر دارد؟"
    )

    record = CorpusRecord(
        source_doc_id=96004,
        question=query,
        category="test",
        brief_answer=(
            "پس از تسویه، دیگر تاثیری ندارد."
        ),
        answer=(
            "پس از پرداخت بدهی، اثر آن "
            "بر امتیاز باقی نمی‌ماند."
        ),
        keyword="بدهی مالیاتی",
        source_row_hash="g-answer-negation",
    )

    _, parts = score_record(
        query,
        parse_query(query),
        record,
        registry=None,
    )

    assert (
        parts.get("polarity_reason")
        != "POLARITY_CONFLICT"
    )


def test_g_comma_before_question_preserves_positive_premise():
    frame = parse_query(
        "بدهی مالیاتی دارم، "
        "آیا روی امتیاز تاثیر دارد؟"
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    assert (
        Polarity.UNKNOWN
        in _polarities(
            frame,
            "EFFECT",
        )
    )


def test_g_two_debts_joined_with_and_have_independent_existence():
    frame = parse_query(
        "بدهی مالیاتی ندارم و "
        "چک برگشتی دارم"
    )

    assert (
        Polarity.NEGATIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "DEBT:چک",
        )
    )


def test_g_unpaid_event_does_not_negate_positive_effect():
    frame = parse_query(
        "بدهی مالیاتی پرداخت نشده دارم و "
        "روی امتیاز تاثیر دارد."
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "EFFECT",
        )
    )


def test_g_effect_verb_does_not_assert_debt_existence_in_question():
    frame = parse_query(
        "آیا چک برگشتی روی امتیاز تاثیر دارد؟"
    )

    assert (
        Polarity.UNKNOWN
        in _polarities(
            frame,
            "DEBT:چک",
        )
    )

    assert (
        Polarity.UNKNOWN
        in _polarities(
            frame,
            "EFFECT",
        )
    )


def test_g_effect_negation_does_not_negate_debt_existence():
    frame = parse_query(
        "بدهی مالیاتی روی امتیاز تاثیر ندارد."
    )

    assert (
        Polarity.POSITIVE
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    assert (
        Polarity.NEGATIVE
        in _polarities(
            frame,
            "EFFECT",
        )
    )


def test_g_negative_effect_question_does_not_negate_debt_entity():
    frame = parse_query(
        "آیا بدهی مالیاتی روی امتیاز تاثیر ندارد؟"
    )

    assert (
        Polarity.UNKNOWN
        in _polarities(
            frame,
            "DEBT:مالیاتی",
        )
    )

    assert (
        Polarity.NEGATIVE
        in _polarities(
            frame,
            "EFFECT",
        )
    )
