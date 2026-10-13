from rubbish_rag.judge import (
    EvidenceDecision,
    HeuristicFeatureJudge,
    PreRetrievalDecision,
    escalate_pre_retrieval_decision,
    run_bounded_adaptive_loop,
)


def _hit(doc_id, text):
    return {
        "doc_id": doc_id,
        "question": text,
        "brief_answer": text,
        "text": text,
        "fused": 1.0,
        "rerank_score": 10.0,
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": (
            "remote_unverified_test"
        ),
        "local_parts": {},
    }


class _ScriptedJudge:
    confidence_threshold = 0.75

    def __init__(
        self,
        *,
        pre=None,
        post_mode="insufficient_then_sufficient",
    ):
        self._pre = pre
        self.post_mode = post_mode

    def pre_retrieval(
        self,
        query,
        conversation=None,
        slots=None,
    ):
        if self._pre is not None:
            return self._pre

        return PreRetrievalDecision(
            action="retrieve",
            needs_retrieval=True,
            needs_clarification=False,
            missing_slots=[],
            retrieval_strategy="hybrid",
            query_rewrite=query,
            target_entities=[],
            target_facets=[
                "credit_regulations"
            ],
            top_k=2,
            confidence=0.90,
            retrieve_probability=0.95,
            clarify_probability=0.05,
            reason_codes=[
                "TEST_RETRIEVE"
            ],
        )

    def post_retrieval(
        self,
        query,
        hits,
        state=None,
    ):
        if self.post_mode == "always_insufficient":
            sufficient = False
        elif self.post_mode == "conflicting":
            return EvidenceDecision(
                sufficiency="conflicting",
                evidence_score=0.5,
                conflicting_items=[1],
                poison_detected=False,
                repetition_detected=False,
                novel_evidence_chunks=len(hits),
                next_action="abstain",
                confidence=0.95,
                sufficiency_probability=0.10,
                reason_codes=[
                    "TEST_CONFLICT"
                ],
            )
        else:
            sufficient = len(hits) >= 2

        return EvidenceDecision(
            sufficiency=(
                "sufficient"
                if sufficient
                else "insufficient"
            ),
            evidence_score=(
                1.0
                if sufficient
                else 0.2
            ),
            conflicting_items=[],
            poison_detected=False,
            repetition_detected=False,
            novel_evidence_chunks=len(hits),
            next_action=(
                "answer"
                if sufficient
                else "retrieve_more"
            ),
            confidence=0.90,
            sufficiency_probability=(
                0.90
                if sufficient
                else 0.15
            ),
            reason_codes=[
                (
                    "TEST_SUFFICIENT"
                    if sufficient
                    else "TEST_INSUFFICIENT"
                )
            ],
        )


def test_p11_state_machine_no_retrieval_uses_zero_rounds():
    calls = []

    def retriever(query, top_k):
        calls.append((query, top_k))
        return []

    judge = HeuristicFeatureJudge()

    result = run_bounded_adaptive_loop(
        "سلام، خسته نباشید",
        retriever,
        judge=judge,
    )

    assert result.action == "no_retrieval"
    assert result.rounds == 0
    assert calls == []


def test_p11_state_machine_clarification_uses_zero_rounds():
    calls = []

    def retriever(query, top_k):
        calls.append((query, top_k))
        return []

    judge = HeuristicFeatureJudge()

    result = run_bounded_adaptive_loop(
        "بدهی دارم",
        retriever,
        judge=judge,
    )

    assert result.action == "clarify"
    assert result.rounds == 0
    assert calls == []


def test_p11_low_confidence_retrieval_gets_broader_budget():
    decision = PreRetrievalDecision(
        action="retrieve",
        needs_retrieval=True,
        needs_clarification=False,
        missing_slots=[],
        retrieval_strategy="hybrid",
        query_rewrite="q",
        target_entities=[],
        target_facets=[],
        top_k=2,
        confidence=0.50,
        retrieve_probability=0.55,
        clarify_probability=0.10,
        reason_codes=[],
    )

    escalated, changed = (
        escalate_pre_retrieval_decision(
            decision,
            confidence_threshold=0.75,
            max_k=8,
        )
    )

    assert changed is True
    assert escalated.action == "retrieve"
    assert escalated.top_k >= 5
    assert (
        "LOW_CONFIDENCE_BROADER_RETRIEVAL"
        in escalated.reason_codes
    )


def test_p11_state_machine_retrieves_at_most_two_rounds():
    calls = []

    batches = [
        [
            _hit(
                98001,
                "evidence one",
            )
        ],
        [
            _hit(
                98002,
                "evidence two",
            )
        ],
    ]

    def retriever(query, top_k):
        calls.append(top_k)
        return batches[
            min(
                len(calls) - 1,
                len(batches) - 1,
            )
        ]

    result = run_bounded_adaptive_loop(
        "test query",
        retriever,
        judge=_ScriptedJudge(),
        max_rounds=2,
    )

    assert result.action == "answer"
    assert result.rounds == 2
    assert len(calls) == 2
    assert (
        result.requested_top_ks[1]
        > result.requested_top_ks[0]
    )


def test_p11_state_machine_stops_after_bounded_failure():
    calls = []

    def retriever(query, top_k):
        calls.append(top_k)
        return [
            _hit(
                98100 + len(calls),
                f"evidence {len(calls)}",
            )
        ]

    result = run_bounded_adaptive_loop(
        "test query",
        retriever,
        judge=_ScriptedJudge(
            post_mode="always_insufficient"
        ),
        max_rounds=2,
    )

    assert result.action == "abstain"
    assert result.rounds == 2
    assert len(calls) == 2
    assert (
        "BOUNDED_RETRIEVAL_EXHAUSTED"
        in result.reason_codes
    )


def test_p11_state_machine_conflict_abstains_without_extra_round():
    calls = []

    def retriever(query, top_k):
        calls.append(top_k)
        return [
            _hit(
                98201,
                "conflicting evidence",
            )
        ]

    result = run_bounded_adaptive_loop(
        "test query",
        retriever,
        judge=_ScriptedJudge(
            post_mode="conflicting"
        ),
        max_rounds=2,
    )

    assert result.action == "abstain"
    assert result.rounds == 1
    assert len(calls) == 1


def test_p11_state_machine_rejects_more_than_two_rounds():
    def retriever(query, top_k):
        return []

    try:
        run_bounded_adaptive_loop(
            "test",
            retriever,
            max_rounds=3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "max_rounds=3 must be rejected"
        )
