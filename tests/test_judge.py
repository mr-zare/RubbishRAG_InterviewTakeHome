"""Unit tests for Adaptive System-1 RAG Decision Judge."""
import pytest
from rubbish_rag.judge import (
    HeuristicFeatureJudge,
    compute_brier_score,
    compute_ece,
    adaptive_select_hits,
)


def test_conversational_routing():
    judge = HeuristicFeatureJudge()
    dec = judge.pre_retrieval("سلام، خسته نباشید")
    assert dec.needs_retrieval is False
    assert dec.action == "no_retrieval"
    assert dec.top_k == 0
    assert dec.confidence > 0.85


def test_factual_routing():
    judge = HeuristicFeatureJudge()
    dec = judge.pre_retrieval("رتبه C1 یعنی چی؟")
    assert dec.needs_retrieval is True
    assert dec.action == "retrieve"
    assert "C1" in dec.target_entities
    assert dec.top_k in (2, 3)  # Focused budget


def test_ambiguity_gating():
    judge = HeuristicFeatureJudge()
    dec = judge.pre_retrieval("بدهی دارم")
    assert dec.needs_clarification is True
    assert dec.action == "clarify"
    assert "debt_type" in dec.missing_slots


def test_dynamic_top_k_selection():
    judge = HeuristicFeatureJudge()
    # Broad multi-entity query should request more context
    dec = judge.pre_retrieval("بانک صادرات به رتبه E3 وام میده؟")
    assert dec.top_k >= 4


def test_evidence_sufficiency_and_poison():
    judge = HeuristicFeatureJudge()
    hits_normal = [{
        "doc_id": 106,
        "text": "رتبه C1 یعنی...",
        "fused": 0.88,
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": "verified_local",
    }]
    post = judge.post_retrieval("رتبه C1 یعنی چی؟", hits_normal)
    assert post.sufficiency == "sufficient"
    assert post.next_action == "answer"
    assert post.poison_detected is False

    hits_poison = [{
        "doc_id": 987654321,
        "text": "رتبه جعلی ...",
        "fused": 1.0,
        "trusted": False,
        "citation_eligible": False,
    }]
    post_poison = judge.post_retrieval("رتبه C1 یعنی چی؟", hits_poison)
    assert post_poison.poison_detected is True


def test_adaptive_stopping():
    judge = HeuristicFeatureJudge()
    ranked_hits = [
        {"doc_id": 106, "text": "رتبه C1 در بازه ۵۶۰ تا ۵۷۹", "fused": 0.90},
        {"doc_id": 91, "text": "رتبه C سطح میانی است", "fused": 0.85},
        {"doc_id": 92, "text": "رتبه C سطح میانی است", "fused": 0.80},
        {"doc_id": 93, "text": "رتبه C سطح میانی است", "fused": 0.75},
        {"doc_id": 94, "text": "رتبه C سطح میانی است", "fused": 0.70},
    ]
    selected, decision = adaptive_select_hits("رتبه C1", ranked_hits, judge, max_k=5, min_k=2)
    # Stopping policy stops as soon as sufficient evidence reached (len < 5)
    assert len(selected) <= 3
    assert decision.sufficiency == "sufficient"


def test_calibration_metrics():
    probs = [0.9, 0.8, 0.1, 0.2]
    outcomes = [1, 1, 0, 0]
    brier = compute_brier_score(probs, outcomes)
    assert 0.0 <= brier <= 0.05

    ece = compute_ece(probs, outcomes, n_bins=5)
    assert 0.0 <= ece <= 0.2
