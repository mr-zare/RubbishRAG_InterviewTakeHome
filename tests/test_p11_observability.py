import json

import pytest

from rubbish_rag.judge import (
    EvidenceDecision,
    HeuristicFeatureJudge,
    PreRetrievalDecision,
)
from rubbish_rag.judge_observability import (
    AdaptiveLoopObservation,
    append_observation_jsonl,
    observe_adaptive_loop,
    summarize_observations,
)


class _TwoRoundJudge:
    confidence_threshold = 0.75

    def pre_retrieval(
        self,
        query,
        conversation=None,
        slots=None,
    ):
        return PreRetrievalDecision(
            action="retrieve",
            needs_retrieval=True,
            needs_clarification=False,
            missing_slots=[],
            retrieval_strategy="hybrid",
            query_rewrite=query,
            target_entities=[],
            target_facets=[],
            top_k=2,
            confidence=0.90,
            retrieve_probability=0.95,
            clarify_probability=0.05,
            reason_codes=[
                "OBS_TEST"
            ],
        )

    def post_retrieval(
        self,
        query,
        hits,
        state=None,
    ):
        sufficient = (
            len(hits) >= 2
        )

        return EvidenceDecision(
            sufficiency=(
                "sufficient"
                if sufficient
                else "insufficient"
            ),
            evidence_score=(
                1.0
                if sufficient
                else 0.1
            ),
            conflicting_items=[],
            poison_detected=False,
            repetition_detected=False,
            novel_evidence_chunks=len(
                hits
            ),
            next_action=(
                "answer"
                if sufficient
                else "retrieve_more"
            ),
            confidence=0.90,
            sufficiency_probability=(
                0.90
                if sufficient
                else 0.10
            ),
            reason_codes=[
                "OBS_POST"
            ],
        )


def _hit(doc_id):
    return {
        "doc_id": doc_id,
        "text": f"evidence {doc_id}",
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": (
            "remote_unverified_test"
        ),
    }


def test_observability_counts_zero_retrieval():
    calls = []

    def retriever(
        query,
        top_k,
    ):
        calls.append(
            top_k
        )
        return []

    observation, result = (
        observe_adaptive_loop(
            "سلام، خسته نباشید",
            retriever,
            judge=(
                HeuristicFeatureJudge()
            ),
        )
    )

    assert (
        result.action
        == "no_retrieval"
    )

    assert (
        observation.retrieval_calls
        == 0
    )

    assert (
        observation.rounds
        == 0
    )

    assert calls == []

    assert (
        observation.elapsed_ms
        >= 0.0
    )


def test_observability_records_two_rounds():
    calls = []

    def retriever(
        query,
        top_k,
    ):
        calls.append(
            top_k
        )

        return [
            _hit(
                99000
                + len(calls)
            )
        ]

    observation, result = (
        observe_adaptive_loop(
            "factual query",
            retriever,
            judge=_TwoRoundJudge(),
            max_rounds=2,
        )
    )

    assert (
        result.action
        == "answer"
    )

    assert (
        observation.rounds
        == 2
    )

    assert (
        observation.retrieval_calls
        == 2
    )

    assert len(
        observation.requested_top_ks
    ) == 2

    assert (
        observation.requested_top_ks[1]
        > observation.requested_top_ks[0]
    )


def test_summary_metrics_are_consistent():
    observations = [
        AdaptiveLoopObservation(
            query="a",
            action="no_retrieval",
            elapsed_ms=1.0,
            rounds=0,
            retrieval_calls=0,
            requested_top_ks=[],
            total_requested_top_k=0,
            total_candidates_seen=0,
            selected_count=0,
            escalated=False,
            backend_status="heuristic",
            fallback_reason=None,
            reason_codes=[],
        ),
        AdaptiveLoopObservation(
            query="b",
            action="answer",
            elapsed_ms=3.0,
            rounds=1,
            retrieval_calls=1,
            requested_top_ks=[3],
            total_requested_top_k=3,
            total_candidates_seen=3,
            selected_count=2,
            escalated=False,
            backend_status="heuristic",
            fallback_reason=None,
            reason_codes=[],
        ),
        AdaptiveLoopObservation(
            query="c",
            action="abstain",
            elapsed_ms=5.0,
            rounds=2,
            retrieval_calls=2,
            requested_top_ks=[
                2,
                4,
            ],
            total_requested_top_k=6,
            total_candidates_seen=2,
            selected_count=2,
            escalated=True,
            backend_status="heuristic",
            fallback_reason=None,
            reason_codes=[],
        ),
    ]

    summary = summarize_observations(
        observations,
        fixed_top_k_baseline=5,
    )

    assert summary.n == 3
    assert summary.mean_latency_ms == 3.0
    assert summary.p50_latency_ms == 3.0
    assert summary.p95_latency_ms > 4.0
    assert (
        summary.p99_latency_ms
        >= summary.p95_latency_ms
    )

    assert (
        summary.max_observed_rounds
        == 2
    )

    assert (
        summary.retrieval_free_rate
        == pytest.approx(
            1 / 3,
            abs=1e-6,
        )
    )

    assert (
        summary.escalation_rate
        == pytest.approx(
            1 / 3,
            abs=1e-6,
        )
    )

    assert (
        summary.first_round_token_savings_pct
        > 60.0
    )


def test_observation_jsonl_is_valid(
    tmp_path,
):
    observation = AdaptiveLoopObservation(
        query="q",
        action="clarify",
        elapsed_ms=1.25,
        rounds=0,
        retrieval_calls=0,
        requested_top_ks=[],
        total_requested_top_k=0,
        total_candidates_seen=0,
        selected_count=0,
        escalated=True,
        backend_status="test",
        fallback_reason=None,
        reason_codes=[
            "TEST"
        ],
    )

    path = (
        tmp_path
        / "telemetry.jsonl"
    )

    append_observation_jsonl(
        path,
        observation,
    )

    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(rows) == 1
    assert (
        rows[0]["action"]
        == "clarify"
    )
    assert (
        rows[0]["query"]
        == "q"
    )
