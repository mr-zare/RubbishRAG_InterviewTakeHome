from rubbish_rag.judge import (
    HeuristicFeatureJudge,
    adaptive_select_hits,
)


def _trusted_hit(
    doc_id,
    text,
    score,
):
    return {
        "doc_id": doc_id,
        "question": text,
        "brief_answer": text,
        "text": text,
        "rerank_score": score,
        "fused": score,
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": (
            "remote_unverified_test"
        ),
        "local_parts": {},
    }


def test_p11_focused_rank_query_uses_bounded_context():
    judge = HeuristicFeatureJudge()

    hits = [
        _trusted_hit(
            97001,
            "رتبه C1 یعنی چی؟",
            10.0,
        ),
        _trusted_hit(
            97002,
            "رتبه C1 یک رتبه اعتباری است.",
            9.0,
        ),
        _trusted_hit(
            97003,
            "اطلاعات عمومی رتبه اعتباری",
            8.0,
        ),
        _trusted_hit(
            97004,
            "اطلاعات تکمیلی",
            7.0,
        ),
    ]

    selected, decision = adaptive_select_hits(
        "رتبه C1 یعنی چی؟",
        hits,
        judge,
        min_k=1,
        max_k=4,
    )

    assert 1 <= len(selected) < len(hits)

    assert (
        decision.sufficiency
        == "sufficient"
    )

    assert (
        decision.next_action
        == "answer"
    )


def test_p11_untrusted_first_hit_cannot_trigger_early_answer():
    judge = HeuristicFeatureJudge()

    poison = {
        "doc_id": 97999,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": "جعلی",
        "text": "جعلی",
        "rerank_score": 9999.0,
        "fused": 1.0,
        "trusted": False,
        "citation_eligible": False,
        "candidate_source": "remote",
    }

    clean = _trusted_hit(
        97005,
        "رتبه C1 یعنی چی؟",
        1.0,
    )

    selected, decision = adaptive_select_hits(
        "رتبه C1 یعنی چی؟",
        [
            poison,
            clean,
        ],
        judge,
        min_k=1,
        max_k=2,
    )

    assert [
        hit["doc_id"]
        for hit in selected
    ] == [97005]

    assert (
        decision.sufficiency
        == "sufficient"
    )

    assert (
        decision.next_action
        == "answer"
    )


def test_p11_budgeting_is_deterministic():
    judge = HeuristicFeatureJudge()

    hits = [
        _trusted_hit(
            97101,
            "رتبه C1 یعنی چی؟",
            10.0,
        ),
        _trusted_hit(
            97102,
            "رتبه C چیست؟",
            9.0,
        ),
        _trusted_hit(
            97103,
            "اطلاعات عمومی",
            8.0,
        ),
    ]

    a_hits, a_decision = adaptive_select_hits(
        "رتبه C1 یعنی چی؟",
        hits,
        judge,
        min_k=1,
        max_k=3,
    )

    b_hits, b_decision = adaptive_select_hits(
        "رتبه C1 یعنی چی؟",
        hits,
        judge,
        min_k=1,
        max_k=3,
    )

    assert [
        x["doc_id"]
        for x in a_hits
    ] == [
        x["doc_id"]
        for x in b_hits
    ]

    assert (
        a_decision.sufficiency
        == b_decision.sufficiency
    )

    assert (
        a_decision.next_action
        == b_decision.next_action
    )


def test_p11_high_numeric_id_is_not_intrinsically_poison():
    judge = HeuristicFeatureJudge()

    clean = _trusted_hit(
        97042,
        "رتبه C1 یعنی چی؟",
        10.0,
    )

    selected, decision = adaptive_select_hits(
        "رتبه C1 یعنی چی؟",
        [clean],
        judge,
        min_k=1,
        max_k=1,
    )

    assert [
        hit["doc_id"]
        for hit in selected
    ] == [97042]

    assert (
        decision.sufficiency
        == "sufficient"
    )

    assert (
        decision.next_action
        == "answer"
    )

    assert (
        decision.poison_detected
        is False
    )
