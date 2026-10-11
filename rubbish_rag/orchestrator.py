"""Thin end-to-end orchestrator using the same registered production stages.

This intentionally avoids duplicating normalization/reranking/security logic.
"""
from __future__ import annotations

from typing import Any

import rubbish_rag as R
import rubbish_rag.pipeline  # noqa: F401 - registers stages


class RubbishRAGOrchestrator:
    def __init__(self, api_key: str = "demo-key"):
        self.api_key = api_key

    def run(self, query: str, topk: int = 5) -> dict[str, Any]:
        # Use the sealed service directly here so the requested API key is
        # preserved, then reuse exactly the same candidate-side stages as bench.
        from server._hidden_retriever import remote_retrieve
        fetch_k = min(20, max(int(topk), int(topk) * 2))
        hits = remote_retrieve(query, topk=fetch_k, api_key=self.api_key)
        hits = R.get("reranker")(query, hits)
        return R.get("resolver")(query, hits)

    def invoke(self, input_data: Any) -> dict[str, Any]:
        if isinstance(input_data, dict):
            query = input_data.get("query") or input_data.get("question") or ""
        else:
            query = str(input_data)
        return self.run(query)


def get_orchestrator(api_key: str = "demo-key") -> RubbishRAGOrchestrator:
    return RubbishRAGOrchestrator(api_key=api_key)
