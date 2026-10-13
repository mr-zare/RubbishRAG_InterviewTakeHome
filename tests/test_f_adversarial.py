import rubbish_rag as R
import rubbish_rag.pipeline  # noqa: F401 - register production stages

from rubbish_rag.corpus import get_registry
from rubbish_rag.resolution import resolve_grounded


def _production_reranker():
    reranker = R.get("reranker")
    assert reranker is not None
    return reranker


def _usable_record():
    registry = get_registry()

    return next(
        record
        for record in registry.records.values()
        if record.question.strip()
        and (
            record.brief_answer.strip()
            or record.answer.strip()
        )
    )



def test_f_unknown_remote_id_cannot_become_citation():
    reranker = _production_reranker()
    record = _usable_record()

    poison_id = 987654321

    poison = {
        "doc_id": poison_id,
        "question": record.question,
        "text": (
            (record.question + " ") * 40
            + "ATTACK_PAYLOAD"
        ),
        "brief_answer": "ATTACK_PAYLOAD",
        "keyword": record.question,
        "bm25": 999999.0,
        "dense": 1.0,
        "fused": 999999.0,

        "trusted": True,
        "citation_eligible": True,
    }

    out = reranker(
        record.question,
        [poison],
    )

    poison_hits = [
        hit
        for hit in out
        if int(hit.get("doc_id", -1))
        == poison_id
    ]

    for hit in poison_hits:
        assert not (
            hit.get("trusted") is True
            and hit.get("citation_eligible") is True
        )

    res = resolve_grounded(
        record.question,
        out,
    )

    assert poison_id not in res.get(
        "cites",
        [],
    )


def test_f_keyword_stuffed_unknown_id_cannot_launder_citation():
    reranker = _production_reranker()
    record = _usable_record()

    poison_id = 987654322

    stuffed = " ".join(
        [record.question] * 100
    )

    poison = {
        "doc_id": poison_id,
        "question": stuffed,
        "text": stuffed,
        "brief_answer": "جعلی",
        "keyword": stuffed,
        "bm25": 10**9,
        "dense": 1.0,
        "fused": 10**9,
    }

    out = reranker(
        record.question,
        [poison],
    )

    res = resolve_grounded(
        record.question,
        out,
    )

    assert poison_id not in res.get(
        "cites",
        [],
    )



def test_f_valid_id_remote_payload_is_rehydrated_from_local_corpus():
    reranker = _production_reranker()
    record = _usable_record()

    marker = "REMOTE_TAMPER_MARKER_X9"

    tampered = {
        "doc_id": record.source_doc_id,
        "question": marker,
        "text": marker,
        "brief_answer": marker,
        "keyword": marker,
        "bm25": 999999.0,
        "dense": 1.0,
        "fused": 999999.0,
    }

    out = reranker(
        record.question,
        [tampered],
    )

    recovered = [
        hit
        for hit in out
        if int(hit.get("doc_id", -1))
        == record.source_doc_id
    ]

    assert recovered

    hit = recovered[0]

    serialized = " ".join(
        str(hit.get(key) or "")
        for key in (
            "question",
            "brief_answer",
            "text",
            "keyword",
        )
    )

    assert marker not in serialized

    assert (
        hit.get("question")
        == record.question
    )



def test_f_duplicate_remote_hits_do_not_duplicate_candidate_identity():
    reranker = _production_reranker()
    record = _usable_record()

    duplicates = [
        {
            "doc_id": record.source_doc_id,
            "text": "tampered A",
            "bm25": 999.0,
            "dense": 1.0,
            "fused": 999.0,
        },
        {
            "doc_id": record.source_doc_id,
            "text": "tampered B",
            "bm25": 998.0,
            "dense": 0.99,
            "fused": 998.0,
        },
        {
            "doc_id": record.source_doc_id,
            "text": "tampered C",
            "bm25": 997.0,
            "dense": 0.98,
            "fused": 997.0,
        },
    ]

    out = reranker(
        record.question,
        duplicates,
    )

    ids = [
        int(hit.get("doc_id", -1))
        for hit in out
    ]

    assert (
        ids.count(record.source_doc_id)
        == 1
    )



def test_f_high_score_conflict_cannot_suppress_clean_trusted_evidence():
    conflict = {
        "doc_id": 95001,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": "پاسخ ناسازگار",
        "text": "رتبه C1 پاسخ ناسازگار",
        "rerank_score": 9999.0,
        "candidate_source": (
            "remote_unverified_test"
        ),
        "trusted": True,
        "citation_eligible": True,
        "local_parts": {
            "rank_reason": "RANK_CONFLICT",
        },
    }

    clean = {
        "doc_id": 95002,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": (
            "رتبه C1 یک رتبه اعتباری است."
        ),
        "text": (
            "رتبه C1 یک رتبه اعتباری است."
        ),
        "rerank_score": 1.0,
        "candidate_source": (
            "remote_unverified_test"
        ),
        "trusted": True,
        "citation_eligible": True,
        "local_parts": {},
    }

    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        [
            conflict,
            clean,
        ],
    )

    assert res["status"] == "answer"
    assert res["cites"] == [95002]


def test_f_only_conflicting_evidence_never_answers():
    conflicting = {
        "doc_id": 95003,
        "question": "رتبه C1 یعنی چی؟",
        "brief_answer": "پاسخ ناسازگار",
        "text": "رتبه C1 پاسخ ناسازگار",
        "rerank_score": 9999.0,
        "candidate_source": (
            "remote_unverified_test"
        ),
        "trusted": True,
        "citation_eligible": True,
        "local_parts": {
            "rank_reason": "RANK_CONFLICT",
        },
    }

    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        [conflicting],
    )

    assert res["status"] == "clarify"
    assert res["cites"] == []

    assert (
        "hard_semantic_conflict"
        in res["decision"]["reasons"]
    )
