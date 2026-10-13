from __future__ import annotations

import rubbish_rag as R

from .chunking import chunk_documents
from .fusion import rrf_fuse_hits
from .rerank import prepare_reranker, semantic_rerank
from .resolution import resolve_grounded

prepare_reranker()


@R.chunker
def chunk(docs):
    return chunk_documents(docs)


@R.retriever
def retrieve(query, topk=5):
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
