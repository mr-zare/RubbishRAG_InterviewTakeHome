from pathlib import Path

import rubbish_rag as R
import rubbish_rag.pipeline  # noqa: F401


def test_production_stages_are_registered():
    assert R.get("chunker") is not None
    assert R.get("retriever") is not None
    assert R.get("reranker") is not None
    assert R.get("resolver") is not None


def test_production_pipeline_does_not_import_judge():
    source = Path(
        "rubbish_rag/pipeline.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "from .judge" not in source
    assert "import rubbish_rag.judge" not in source


def test_orchestrator_does_not_delegate_answerability_to_judge():
    source = Path(
        "rubbish_rag/orchestrator.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "HeuristicFeatureJudge" not in source
    assert "OpenWeightLLMJudge" not in source
    assert "adaptive_select_hits" not in source


def test_runtime_code_does_not_read_visible_benchmark():
    package = Path("rubbish_rag")

    offenders = []

    for path in package.rglob("*.py"):
        text = path.read_text(
            encoding="utf-8"
        )

        if (
            "visible_bench" in text
            or "gold_doc_id" in text
        ):
            offenders.append(
                str(path)
            )

    assert offenders == []


def test_runtime_has_no_magic_poison_id_threshold():
    package = Path("rubbish_rag")

    suspicious = []

    needles = (
        ">= 9000",
        "> 9000",
        ">=9000",
        ">9000",
        "9004",
    )

    for path in package.rglob("*.py"):
        text = path.read_text(
            encoding="utf-8"
        )

        if any(
            needle in text
            for needle in needles
        ):
            suspicious.append(
                str(path)
            )

    assert suspicious == []


def test_final_unknown_poison_cannot_answer_even_with_extreme_score():
    from rubbish_rag.resolution import resolve_grounded

    poison = {
        "doc_id": 888888888,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": "پاسخ جعلی",
        "text": (
            "رتبه C1 " * 100
        ),
        "rerank_score": 10**12,
        "fused": 1.0,
        "trusted": False,
        "citation_eligible": False,
        "candidate_source": "remote",
    }

    result = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        [poison],
    )

    assert result["status"] == "clarify"
    assert result["cites"] == []


def test_final_conflicting_high_score_cannot_beat_clean_support():
    from rubbish_rag.resolution import resolve_grounded

    bad = {
        "doc_id": 98001,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": "ناسازگار",
        "text": "ناسازگار",
        "rerank_score": 99999.0,
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": "remote_unverified_test",
        "local_parts": {
            "rank_reason": "RANK_CONFLICT",
        },
    }

    good = {
        "doc_id": 98002,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": (
            "رتبه C1 یک رتبه اعتباری است."
        ),
        "text": (
            "رتبه C1 یک رتبه اعتباری است."
        ),
        "rerank_score": 1.0,
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": "remote_unverified_test",
        "local_parts": {},
    }

    result = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        [bad, good],
    )

    assert result["status"] == "answer"
    assert result["cites"] == [98002]


def test_final_duplicate_sources_never_duplicate_citations():
    from rubbish_rag.resolution import resolve_grounded

    hit = {
        "doc_id": 98003,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": (
            "رتبه C1 یک رتبه اعتباری است."
        ),
        "text": (
            "رتبه C1 یک رتبه اعتباری است."
        ),
        "rerank_score": 5.0,
        "trusted": True,
        "citation_eligible": True,
        "candidate_source": "remote_unverified_test",
        "local_parts": {},
    }

    result = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        [
            dict(hit),
            dict(hit),
            dict(hit),
        ],
    )

    assert result["status"] == "answer"
    assert result["cites"] == [98003]


def test_runtime_judge_has_no_legacy_score_as_confidence_logic():
    source = Path(
        "rubbish_rag/judge.py"
    ).read_text(
        encoding="utf-8"
    )

    forbidden = (
        "HIGH_CONFIDENCE_TOP_HIT",
        "MODERATE_RETRIEVAL_SUPPORT",
        "LOW_RETRIEVAL_SIMILARITY",
        "top_score >= 0.70",
        "score_margin >= 0.15",
    )

    found = [
        item
        for item in forbidden
        if item in source
    ]

    assert found == []
