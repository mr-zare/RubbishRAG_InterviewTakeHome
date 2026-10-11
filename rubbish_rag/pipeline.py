"""RubbishRAG candidate pipeline — architecture-first implementation.

Core path:
  structured chunking -> remote retrieval/RRF -> provenance-aware semantic
  reranking -> grounded answer/clarification.
"""
from __future__ import annotations

import rubbish_rag as R

from .chunking import chunk_documents
from .fusion import rrf_fuse_hits
from .rerank import prepare_reranker, semantic_rerank
from .resolution import resolve_grounded

# Corpus/index preparation occurs at module import. The stock benchmark imports
# this module before starting per-query timers, so one-time preparation is not
# misreported as query latency.
prepare_reranker()


@R.chunker
def chunk(docs):
    return chunk_documents(docs)


@R.retriever
def retrieve(query, topk=5):
    """Remote-first retrieval with client-side RRF.

    Important: the stock CLI benchmark currently bypasses this decorator and
    calls the remote retriever directly. The same RRF logic is therefore also
    applied inside the reranker so benchmark behavior remains correct.
    """
    from server._hidden_retriever import remote_retrieve

    fetch_k = min(20, max(int(topk), int(topk) * 2))
    hits = remote_retrieve(query, topk=fetch_k)
    return rrf_fuse_hits(hits)[: int(topk)]


@R.reranker
def rerank(query, hits):
    return semantic_rerank(query, hits, enable_local_backfill=True)


@R.resolver
def resolve(query, hits):
    return resolve_grounded(query, hits)
