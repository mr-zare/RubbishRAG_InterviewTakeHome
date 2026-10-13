from rubbish_rag.judge import (
    OpenWeightLLMJudge,
)


def test_openweight_without_local_model_falls_back_cleanly():
    judge = OpenWeightLLMJudge(
        force_model=True,
    )

    decision = judge.pre_retrieval(
        "سلام، خسته نباشید"
    )

    assert decision.action == "no_retrieval"

    assert judge.backend_status == (
        "fallback:no_local_model"
    )

    assert (
        judge.last_fallback_reason
        == "NO_LOCAL_MODEL_CONFIGURED"
    )


def test_openweight_constructor_is_lazy():
    judge = OpenWeightLLMJudge(
        local_model_path=(
            "/definitely/not/a/model"
        )
    )

    assert judge._load_attempted is False
    assert judge._model is None
    assert judge._tokenizer is None


def test_openweight_injected_generator_drives_pre_decision():
    def generator(prompt):
        assert (
            "PRE_RETRIEVAL"
            in prompt
        )

        return """
        {
          "action": "retrieve",
          "missing_slots": [],
          "retrieval_strategy": "hybrid",
          "query_rewrite": "رتبه C1",
          "target_entities": ["C1"],
          "target_facets": ["credit_rank"],
          "top_k": 8,
          "confidence": 0.91,
          "retrieve_probability": 0.95,
          "clarify_probability": 0.04,
          "reason_codes": ["LOCAL_TEST"]
        }
        """

    judge = OpenWeightLLMJudge(
        generator=generator,
        force_model=True,
    )

    decision = judge.pre_retrieval(
        "رتبه C1 یعنی چی؟"
    )

    assert decision.action == "retrieve"
    assert decision.top_k == 8
    assert decision.confidence == 0.91
    assert "C1" in decision.target_entities
    assert (
        "OPEN_WEIGHT_LOCAL_DECISION"
        in decision.reason_codes
    )

    assert (
        judge.backend_status
        == "injected_generator"
    )


def test_openweight_injected_generator_drives_post_decision():
    def generator(prompt):
        assert (
            "POST_RETRIEVAL"
            in prompt
        )

        return """
        {
          "sufficiency": "conflicting",
          "evidence_score": 0.31,
          "conflicting_items": [11, 12],
          "poison_detected": false,
          "repetition_detected": false,
          "novel_evidence_chunks": 2,
          "next_action": "abstain",
          "confidence": 0.93,
          "sufficiency_probability": 0.12,
          "reason_codes": ["LOCAL_CONFLICT"]
        }
        """

    judge = OpenWeightLLMJudge(
        generator=generator,
        force_model=True,
    )

    decision = judge.post_retrieval(
        "تست",
        [
            {
                "doc_id": 11,
                "text": "الف",
            },
            {
                "doc_id": 12,
                "text": "ب",
            },
        ],
    )

    assert (
        decision.sufficiency
        == "conflicting"
    )
    assert (
        decision.next_action
        == "abstain"
    )
    assert decision.confidence == 0.93
    assert decision.conflicting_items == [
        11,
        12,
    ]


def test_openweight_invalid_json_falls_back():
    def generator(prompt):
        return (
            "this is not JSON"
        )

    judge = OpenWeightLLMJudge(
        generator=generator,
        force_model=True,
    )

    decision = judge.pre_retrieval(
        "سلام"
    )

    assert (
        decision.action
        == "no_retrieval"
    )

    assert (
        judge.last_fallback_reason
        == "INVALID_LOCAL_MODEL_JSON"
    )


def test_openweight_invalid_action_falls_back():
    def generator(prompt):
        return """
        {
          "action": "invent_answer",
          "confidence": 0.99
        }
        """

    judge = OpenWeightLLMJudge(
        generator=generator,
        force_model=True,
    )

    decision = judge.pre_retrieval(
        "رتبه C1 یعنی چی؟"
    )

    assert decision.action == "retrieve"

    assert (
        judge.last_fallback_reason
        == "INVALID_PRE_ACTION"
    )


def test_openweight_json_fence_is_supported():
    raw = """```json
    {"action":"clarify"}
    ```"""

    payload = (
        OpenWeightLLMJudge
        ._extract_json_object(
            raw
        )
    )

    assert payload == {
        "action": "clarify"
    }


def test_openweight_source_is_strictly_local_only():
    from pathlib import Path

    source = Path(
        "rubbish_rag/judge.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "local_files_only=True"
        in source
    )

    assert (
        "trust_remote_code=True"
        not in source
    )


def test_openweight_default_uses_high_confidence_heuristic_fast_path():
    called = False

    def generator(prompt):
        nonlocal called
        called = True
        raise AssertionError(
            "high-confidence fast path must not invoke local model"
        )

    judge = OpenWeightLLMJudge(
        generator=generator,
    )

    decision = judge.pre_retrieval(
        "سلام، خسته نباشید"
    )

    assert decision.action == "no_retrieval"
    assert called is False

    assert (
        judge.backend_status
        == "heuristic_fast_path"
    )

    assert (
        "HEURISTIC_FAST_PATH"
        in decision.reason_codes
    )


def test_openweight_force_model_bypasses_fast_path():
    called = False

    def generator(prompt):
        nonlocal called
        called = True

        return """
        {
          "action": "retrieve",
          "missing_slots": [],
          "retrieval_strategy": "hybrid",
          "query_rewrite": "سلام",
          "target_entities": [],
          "target_facets": [],
          "top_k": 2,
          "confidence": 0.9,
          "retrieve_probability": 0.9,
          "clarify_probability": 0.1,
          "reason_codes": ["FORCED_MODEL"]
        }
        """

    judge = OpenWeightLLMJudge(
        generator=generator,
        force_model=True,
    )

    decision = judge.pre_retrieval(
        "سلام، خسته نباشید"
    )

    assert called is True
    assert decision.action == "retrieve"

    assert (
        "OPEN_WEIGHT_LOCAL_DECISION"
        in decision.reason_codes
    )



def test_openweight_low_confidence_actually_escalates_to_model():
    from rubbish_rag.judge import (
        PreRetrievalDecision,
    )

    called = False

    class LowConfidenceFallback:
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
                top_k=3,
                confidence=0.40,
                retrieve_probability=0.55,
                clarify_probability=0.10,
                reason_codes=[
                    "LOW_CONFIDENCE_TEST",
                ],
            )

        def post_retrieval(
            self,
            query,
            hits,
            state=None,
        ):
            raise AssertionError(
                "post_retrieval is not used in this test"
            )

    def generator(prompt):
        nonlocal called
        called = True

        return """
        {
          "action": "retrieve",
          "missing_slots": [],
          "retrieval_strategy": "hybrid",
          "query_rewrite": "test",
          "target_entities": [],
          "target_facets": [],
          "top_k": 5,
          "confidence": 0.91,
          "retrieve_probability": 0.95,
          "clarify_probability": 0.03,
          "reason_codes": ["ESCALATED_MODEL"]
        }
        """

    judge = OpenWeightLLMJudge(
        fallback=LowConfidenceFallback(),
        generator=generator,
        escalation_threshold=0.75,
    )

    decision = judge.pre_retrieval(
        "test"
    )

    assert called is True
    assert decision.action == "retrieve"
    assert decision.top_k == 5

    assert (
        "OPEN_WEIGHT_LOCAL_DECISION"
        in decision.reason_codes
    )
