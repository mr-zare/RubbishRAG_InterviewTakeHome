import rubbish_rag as R
import rubbish_rag.pipeline  # register stages
from rubbish_rag.chunking import chunk_documents
from rubbish_rag.fusion import rrf_fuse_hits
from rubbish_rag.resolution import resolve_grounded


def test_chunker_sentence_aware_and_metadata():
    docs = [{
        "doc_id": 42,
        "Question": "زمان حذف بدهی چیست؟",
        "Category": "بدهی",
        "BriefAnswer": "تا 24 ساعت.",
        "Answer": "اگر بدهی پرداخت شود، اثر آن حذف می‌شود. این فرایند حداکثر 24 ساعت زمان می‌برد.",
        "Keyword": "مالیات، حذف",
    }]
    chunks = chunk_documents(docs, soft_target=55, hard_max=90)
    assert chunks
    assert all(c["doc_id"] == 42 for c in chunks)
    assert all("question" in c and "lexical_text" in c for c in chunks)
    assert any("24 ساعت" in c["text"] for c in chunks)


def test_rrf_is_rank_based_and_deterministic():
    hits = [
        {"doc_id": 1, "bm25": 100.0, "dense": 0.1, "fused": 1.0, "text": "a"},
        {"doc_id": 2, "bm25": 10.0, "dense": 0.9, "fused": 0.8, "text": "b"},
    ]
    a = rrf_fuse_hits(hits)
    b = rrf_fuse_hits(hits)
    assert [(x["doc_id"], x["rrf_score"]) for x in a] == [(x["doc_id"], x["rrf_score"]) for x in b]
    assert all("server_fused" in x for x in a)


def test_registered_reranker_contract_without_corpus():
    reranker = R.get("reranker")
    hits = [
        {"doc_id": 1, "text": "رتبه اعتباری و امتیاز", "fused": 0.8},
        {"doc_id": 2, "text": "سایر اطلاعات بی ارتباط", "fused": 0.5},
    ]
    out = reranker("رتبه اعتباری", hits)
    assert out[0]["doc_id"] == 1
    assert "rerank_score" in out[0]


def test_resolver_deduplicates_citations():
    hits = [
        {"doc_id": 7, "text": "پاسخ", "brief_answer": "پاسخ", "rerank_score": 3.0},
        {"doc_id": 7, "text": "پاسخ دیگر", "brief_answer": "پاسخ دیگر", "rerank_score": 2.0},
        {"doc_id": 8, "text": "پاسخ مکمل", "brief_answer": "پاسخ مکمل", "rerank_score": 1.0},
    ]
    res = resolve_grounded("سوال مشخص", hits)
    assert res["status"] == "answer"
    assert len(res["cites"]) == len(set(res["cites"]))


def test_ambiguous_debt_clarifies_before_answering():
    res = resolve_grounded("بدهی دارم", [{"doc_id": 1, "text": "هر چیزی", "rerank_score": 9.0}])
    assert res["status"] == "clarify"
    assert "debt_type" in res["missing_slots"]


def test_evidence_stops_after_complete_primary():
    hits = [
        {
            "doc_id": 101,
            "question": "رتبه C1 چیست؟",
            "text": "رتبه C1",
            "brief_answer": "C1",
            "rerank_score": 9.0,
            "candidate_source": "remote_unverified_test",
        },
        {
            "doc_id": 102,
            "question": "رتبه C چیست؟",
            "text": "رتبه C",
            "brief_answer": "C",
            "rerank_score": 8.0,
            "candidate_source": "remote_unverified_test",
        },
    ]

    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        hits,
    )

    assert res["status"] == "answer"
    assert res["cites"] == [101]


def test_exact_faq_question_gets_specificity_bonus():
    from rubbish_rag.corpus import CorpusRecord
    from rubbish_rag.rerank import score_record
    from rubbish_rag.slots import parse_query

    query = "اگر بدهی مالیاتی را پرداخت کنم چه اتفاقی می افتد؟"
    frame = parse_query(query)

    exact = CorpusRecord(
        source_doc_id=90001,
        question=query,
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="بدهی مالیاتی",
        source_row_hash="exact",
    )

    related = CorpusRecord(
        source_doc_id=90002,
        question="بدهی مالیاتی چه تاثیری بر امتیاز دارد؟",
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="بدهی مالیاتی",
        source_row_hash="related",
    )

    exact_score, exact_parts = score_record(
        query,
        frame,
        exact,
        registry=None,
    )

    related_score, related_parts = score_record(
        query,
        frame,
        related,
        registry=None,
    )

    assert exact_parts["question_exact_match"] == 3.0
    assert related_parts["question_exact_match"] == 0.0
    assert exact_score > related_score


def test_exact_faq_question_cannot_conflict_with_its_own_polarity():
    from rubbish_rag.corpus import CorpusRecord
    from rubbish_rag.rerank import score_record
    from rubbish_rag.slots import parse_query

    query = (
        "اگر بدهی مالیاتی پرداخت نشده را پرداخت کنم "
        "چه اتفاقی می افتد؟"
    )

    record = CorpusRecord(
        source_doc_id=91001,
        question=query,
        category="test",
        brief_answer=(
            "پس از پرداخت، این بدهی دیگر تاثیری "
            "در امتیاز نخواهد داشت."
        ),
        answer=(
            "پس از پرداخت، اثر منفی بدهی حذف می‌شود."
        ),
        keyword="بدهی مالیاتی پرداخت نشده",
        source_row_hash="polarity-self-match",
    )

    score, parts = score_record(
        query,
        parse_query(query),
        record,
        registry=None,
    )

    assert parts.get("polarity_reason") != "POLARITY_CONFLICT"
    assert float(parts.get("polarity", 0.0)) >= 0.0


def test_opposite_declarative_polarity_is_penalized():
    from rubbish_rag.corpus import CorpusRecord
    from rubbish_rag.rerank import score_record
    from rubbish_rag.slots import parse_query

    query = "بدهی مالیاتی روی امتیاز تاثیر ندارد."

    record = CorpusRecord(
        source_doc_id=91002,
        question="بدهی مالیاتی روی امتیاز تاثیر دارد.",
        category="test",
        brief_answer="پاسخ تست",
        answer="پاسخ تست",
        keyword="بدهی مالیاتی",
        source_row_hash="polarity-opposite",
    )

    score, parts = score_record(
        query,
        parse_query(query),
        record,
        registry=None,
    )

    assert parts.get("polarity_reason") == "POLARITY_CONFLICT"
    assert float(parts["polarity"]) < 0.0


def test_exact_trusted_faq_is_candidate_generation_anchor():
    from rubbish_rag.corpus import get_registry
    from rubbish_rag.rerank import (
        _verified_exact_question_hits,
        prepare_reranker,
    )
    from rubbish_rag.slots import parse_query

    prepare_reranker()
    registry = get_registry()

    record = next(
        r
        for r in registry.records.values()
        if r.question.strip()
    )

    hits = _verified_exact_question_hits(
        record.question,
        parse_query(record.question),
        registry,
    )

    assert hits
    assert any(
        hit["doc_id"] == record.source_doc_id
        for hit in hits
    )

    matching = [
        hit
        for hit in hits
        if hit["doc_id"] == record.source_doc_id
    ][0]

    assert matching["trusted"] is True
    assert matching["citation_eligible"] is True
    assert (
        matching["candidate_source"]
        == "verified_exact_question"
    )
    assert matching["exact_question_anchor"] is True


def test_p10_no_evidence_never_answers():
    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        [],
    )

    assert res["status"] == "clarify"
    assert res["cites"] == []


def test_p10_uncovered_critical_requirement_never_answers():
    hits = [
        {
            "doc_id": 92001,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "رتبه C1 یک رتبه اعتباری است.",
            "text": "رتبه C1 یک رتبه اعتباری است.",
            "rerank_score": 10.0,
            "candidate_source": "remote_unverified_test",
        }
    ]

    res = resolve_grounded(
        "رتبه C1 بانک صادرات یعنی چی؟",
        hits,
    )

    assert res["status"] == "clarify"
    assert (
        "critical_requirements_uncovered"
        in res["decision"]["reasons"]
    )

    assert any(
        x.startswith("bank:")
        for x in res["decision"][
            "critical_uncovered"
        ]
    )


def test_p10_explicitly_untrusted_evidence_never_answers():
    hits = [
        {
            "doc_id": 92002,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "رتبه C1 یک رتبه اعتباری است.",
            "text": "رتبه C1 یک رتبه اعتباری است.",
            "rerank_score": 100.0,
            "candidate_source": "remote_unverified_test",
            "trusted": False,
            "citation_eligible": False,
        }
    ]

    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        hits,
    )

    assert res["status"] == "clarify"
    assert (
        "no_trusted_citable_evidence"
        in res["decision"]["reasons"]
    )
    assert res["cites"] == []


def test_p10_hard_rank_conflict_never_answers():
    hits = [
        {
            "doc_id": 92003,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "پاسخ تست",
            "text": "رتبه C1",
            "rerank_score": 20.0,
            "candidate_source": "remote_unverified_test",
            "trusted": True,
            "citation_eligible": True,
            "local_parts": {
                "rank_reason": "RANK_CONFLICT",
            },
        }
    ]

    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        hits,
    )

    assert res["status"] == "clarify"
    assert (
        "hard_semantic_conflict"
        in res["decision"]["reasons"]
    )
    assert any(
        "RANK_CONFLICT" in x
        for x in res["decision"][
            "conflicts"
        ]
    )


def test_p10_polarity_conflict_never_answers():
    hits = [
        {
            "doc_id": 92004,
            "question": "بدهی مالیاتی روی امتیاز تاثیر ندارد.",
            "brief_answer": "پاسخ تست",
            "text": "پاسخ تست",
            "rerank_score": 20.0,
            "candidate_source": "remote_unverified_test",
            "trusted": True,
            "citation_eligible": True,
            "local_parts": {
                "polarity_reason": (
                    "POLARITY_CONFLICT"
                ),
            },
        }
    ]

    res = resolve_grounded(
        "بدهی مالیاتی روی امتیاز تاثیر دارد.",
        hits,
    )

    assert res["status"] == "clarify"
    assert (
        "hard_semantic_conflict"
        in res["decision"]["reasons"]
    )


def test_p10_complete_trusted_evidence_can_answer():
    hits = [
        {
            "doc_id": 92005,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "رتبه C1 یک رتبه اعتباری است.",
            "text": "رتبه C1 یک رتبه اعتباری است.",
            "rerank_score": 10.0,
            "candidate_source": "remote_unverified_test",
            "trusted": True,
            "citation_eligible": True,
            "local_parts": {},
        }
    ]

    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        hits,
    )

    assert res["status"] == "answer"
    assert res["cites"] == [92005]
    assert (
        res["evidence"]["sufficiency"][
            "sufficient"
        ]
        is True
    )


def test_p10_duplicates_do_not_manufacture_sufficiency():
    hits = [
        {
            "doc_id": 92006,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "رتبه C1",
            "text": "رتبه C1",
            "rerank_score": 10.0,
            "candidate_source": "remote_unverified_test",
        },
        {
            "doc_id": 92006,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "رتبه C1",
            "text": "رتبه C1",
            "rerank_score": 9.0,
            "candidate_source": "remote_unverified_test",
        },
    ]

    res = resolve_grounded(
        "رتبه C1 بانک صادرات یعنی چی؟",
        hits,
    )

    assert res["status"] == "clarify"
    assert (
        "critical_requirements_uncovered"
        in res["decision"]["reasons"]
    )


def test_p10_untrusted_top_hit_cannot_suppress_trusted_evidence():
    hits = [
        {
            "doc_id": 99999,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "پاسخ جعلی",
            "text": "رتبه C1 پاسخ جعلی",
            "rerank_score": 9999.0,
            "trusted": False,
            "citation_eligible": False,
            "candidate_source": "remote",
        },
        {
            "doc_id": 93001,
            "question": "رتبه C1 یعنی چی؟",
            "brief_answer": "رتبه C1 یک رتبه اعتباری است.",
            "text": "رتبه C1 یک رتبه اعتباری است.",
            "rerank_score": 1.0,
            "trusted": True,
            "citation_eligible": True,
            "candidate_source": "remote_unverified_test",
            "local_parts": {},
        },
    ]

    res = resolve_grounded(
        "رتبه C1 یعنی چی؟",
        hits,
    )

    assert res["status"] == "answer"
    assert res["cites"] == [93001]
    assert 99999 not in res["cites"]
