"""Rank-based hybrid fusion utilities (P6)."""
from __future__ import annotations

from typing import Any


def _rank_map(hits: list[dict[str, Any]], field: str) -> dict[int, int]:
    ordered = sorted(
        enumerate(hits),
        key=lambda pair: (-float(pair[1].get(field, 0.0) or 0.0), pair[0], int(pair[1].get("doc_id", 10**12))),
    )
    return {idx: rank for rank, (idx, _) in enumerate(ordered, start=1)}


def rrf_fuse_hits(hits: list[dict[str, Any]], *, rank_constant: int = 60) -> list[dict[str, Any]]:
    """Fuse BM25 and dense rankings without mixing incompatible raw scales."""
    if not hits:
        return []
    bm25_ranks = _rank_map(hits, "bm25")
    dense_ranks = _rank_map(hits, "dense")
    fused: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits):
        out = dict(hit)
        rb = bm25_ranks[idx]
        rd = dense_ranks[idx]
        score = 1.0 / (rank_constant + rb) + 1.0 / (rank_constant + rd)
        out["server_fused"] = float(hit.get("fused", 0.0) or 0.0)
        out["bm25_rank"] = rb
        out["dense_rank"] = rd
        out["rrf_score"] = score
        fused.append(out)
    fused.sort(key=lambda h: (-h["rrf_score"], min(h["bm25_rank"], h["dense_rank"]), int(h.get("doc_id", 10**12))))
    for rank, hit in enumerate(fused, start=1):
        hit["rrf_rank"] = rank
    return fused
