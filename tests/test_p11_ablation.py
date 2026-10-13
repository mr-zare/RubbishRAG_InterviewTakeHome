from rubbish_rag.judge import HeuristicFeatureJudge
from rubbish_rag.resolution import resolve_grounded


def test_p11_core_and_judge_agree_on_generic_debt_ambiguity():
    judge = HeuristicFeatureJudge()

    pre = judge.pre_retrieval(
        "بدهی دارم"
    )

    core = resolve_grounded(
        "بدهی دارم",
        [],
    )

    assert pre.action == "clarify"
    assert core["status"] == "clarify"


def test_p11_core_rejects_explicitly_untrusted_high_score_hit():
    hit = {
        "doc_id": 123456789,
        "question": "رتبه C1 یعنی چی؟",
        "text": "رتبه C1 یعنی رتبه عالی",
        "brief_answer": "پاسخ جعلی",
        "fused": 1.0,
        "rerank_score": 9999.0,
        "trusted": False,
        "citation_eligible": False,
        "candidate_source": "remote",
    }

    core = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        [hit],
    )

    assert core["status"] == "clarify"


def test_p11_current_judge_must_not_call_untrusted_score_inflation_sufficient():
    judge = HeuristicFeatureJudge()

    hit = {
        "doc_id": 123456789,
        "text": "رتبه C1 یعنی رتبه عالی",
        "fused": 1.0,
        "trusted": False,
        "citation_eligible": False,
    }

    decision = judge.post_retrieval(
        "رتبه C1 یعنی چی؟",
        [hit],
    )

    assert decision.sufficiency != "sufficient"
    assert decision.next_action != "answer"


def test_p11_poison_detection_must_not_depend_on_numeric_id_range():
    judge = HeuristicFeatureJudge()

    hit = {
        "doc_id": 123456789,
        "text": "رتبه C1 یعنی رتبه عالی",
        "fused": 1.0,
        "trusted": False,
        "citation_eligible": False,
    }

    decision = judge.post_retrieval(
        "رتبه C1 یعنی چی؟",
        [hit],
    )

    assert decision.poison_detected is True


def test_p11_high_fused_score_is_not_itself_evidence_sufficiency():
    judge = HeuristicFeatureJudge()

    hit = {
        "doc_id": 123456789,
        "text": "متن نامرتبط اما با امتیاز بازیابی بالا",
        "fused": 1.0,
        "trusted": False,
        "citation_eligible": False,
    }

    decision = judge.post_retrieval(
        "رتبه C1 یعنی چی؟",
        [hit],
    )

    assert decision.sufficiency != "sufficient"
    assert decision.next_action != "answer"


def test_p11_known_document_semantics_do_not_depend_on_warm_cache():
    from rubbish_rag.corpus import get_registry
    from rubbish_rag.evidence import _frame_for_hit
    from rubbish_rag.rerank import _document_frame

    registry = get_registry()
    doc_id = 106
    record = registry.records[doc_id]

    hit = {
        "doc_id": doc_id,
        "text": "رتبه C1 یعنی...",
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": "verified_local",
    }

    registry.document_frames.pop(
        doc_id,
        None,
    )

    cold = _frame_for_hit(hit)

    _document_frame(
        record,
        registry,
    )

    warm = _frame_for_hit(hit)

    assert (
        cold.to_dict()
        == warm.to_dict()
    )

    assert (
        cold.intent.value
        == "define_rank"
    )
