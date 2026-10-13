"""Observability helpers for the optional Adaptive RAG Judge.

This module deliberately stays outside the authoritative production answer
path.  It instruments the bounded adaptive controller so Staff-Track cost and
latency claims can be measured without changing retrieval or answer semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .judge import (
    AdaptiveLoopResult,
    RAGDecisionJudge,
    run_bounded_adaptive_loop,
)


@dataclass
class AdaptiveLoopObservation:
    query: str
    action: str
    elapsed_ms: float
    rounds: int
    retrieval_calls: int
    requested_top_ks: List[int]
    total_requested_top_k: int
    total_candidates_seen: int
    selected_count: int
    escalated: bool
    backend_status: str
    fallback_reason: Optional[str]
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveLoopSummary:
    n: int

    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    mean_rounds: float
    mean_retrieval_calls: float

    mean_first_top_k: float
    mean_total_requested_top_k: float
    mean_selected_context_k: float

    fixed_top_k_baseline: int
    first_round_token_savings_pct: float
    selected_context_savings_pct: float

    retrieval_free_rate: float
    escalation_rate: float

    answer_rate: float
    clarify_rate: float
    no_retrieval_rate: float
    abstain_rate: float

    max_observed_rounds: int
    backend_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _percentile(
    values: List[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        float(v)
        for v in values
    )

    if len(ordered) == 1:
        return ordered[0]

    rank = (
        percentile / 100.0
    ) * (
        len(ordered) - 1
    )

    lo = int(
        math.floor(rank)
    )

    hi = int(
        math.ceil(rank)
    )

    if lo == hi:
        return ordered[lo]

    fraction = rank - lo

    return (
        ordered[lo]
        + (
            ordered[hi]
            - ordered[lo]
        )
        * fraction
    )


def observe_adaptive_loop(
    query: str,
    retriever: Callable[
        [str, int],
        List[Dict[str, Any]],
    ],
    *,
    judge: RAGDecisionJudge,
    max_rounds: int = 2,
    max_k: int = 8,
    min_k: int = 1,
    confidence_threshold: Optional[
        float
    ] = None,
) -> Tuple[
    AdaptiveLoopObservation,
    AdaptiveLoopResult,
]:
    """Run one bounded adaptive decision while recording control-plane cost."""

    retrieval_calls = 0

    def counted_retriever(
        q: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        nonlocal retrieval_calls

        retrieval_calls += 1

        return list(
            retriever(
                q,
                top_k,
            )
            or []
        )

    started = time.perf_counter()

    result = run_bounded_adaptive_loop(
        query,
        counted_retriever,
        judge=judge,
        max_rounds=max_rounds,
        max_k=max_k,
        min_k=min_k,
        confidence_threshold=(
            confidence_threshold
        ),
    )

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    requested = list(
        result.requested_top_ks
    )

    backend_status = str(
        getattr(
            judge,
            "backend_status",
            judge.__class__.__name__,
        )
    )

    fallback_reason = getattr(
        judge,
        "last_fallback_reason",
        None,
    )

    observation = (
        AdaptiveLoopObservation(
            query=query,
            action=result.action,
            elapsed_ms=round(
                elapsed_ms,
                6,
            ),
            rounds=int(
                result.rounds
            ),
            retrieval_calls=int(
                retrieval_calls
            ),
            requested_top_ks=requested,
            total_requested_top_k=sum(
                requested
            ),
            total_candidates_seen=int(
                result.total_candidates_seen
            ),
            selected_count=len(
                result.selected_hits
            ),
            escalated=bool(
                result.escalated
            ),
            backend_status=(
                backend_status
            ),
            fallback_reason=(
                fallback_reason
            ),
            reason_codes=list(
                result.reason_codes
            ),
        )
    )

    return (
        observation,
        result,
    )


def summarize_observations(
    observations: List[
        AdaptiveLoopObservation
    ],
    *,
    fixed_top_k_baseline: int = 5,
) -> AdaptiveLoopSummary:
    """Aggregate latency, retrieval budget, and routing telemetry."""

    if not observations:
        return AdaptiveLoopSummary(
            n=0,
            mean_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            mean_rounds=0.0,
            mean_retrieval_calls=0.0,
            mean_first_top_k=0.0,
            mean_total_requested_top_k=0.0,
            mean_selected_context_k=0.0,
            fixed_top_k_baseline=(
                int(
                    fixed_top_k_baseline
                )
            ),
            first_round_token_savings_pct=0.0,
            selected_context_savings_pct=0.0,
            retrieval_free_rate=0.0,
            escalation_rate=0.0,
            answer_rate=0.0,
            clarify_rate=0.0,
            no_retrieval_rate=0.0,
            abstain_rate=0.0,
            max_observed_rounds=0,
            backend_counts={},
        )

    n = len(
        observations
    )

    latencies = [
        item.elapsed_ms
        for item in observations
    ]

    rounds = [
        item.rounds
        for item in observations
    ]

    retrieval_calls = [
        item.retrieval_calls
        for item in observations
    ]

    first_top_ks = [
        (
            item.requested_top_ks[0]
            if item.requested_top_ks
            else 0
        )
        for item in observations
    ]

    total_top_ks = [
        item.total_requested_top_k
        for item in observations
    ]

    selected_counts = [
        item.selected_count
        for item in observations
    ]

    baseline = max(
        1,
        int(
            fixed_top_k_baseline
        ),
    )

    mean_first = (
        sum(first_top_ks)
        / n
    )

    mean_selected = (
        sum(selected_counts)
        / n
    )

    backend_counts: Dict[
        str,
        int,
    ] = {}

    for item in observations:
        backend_counts[
            item.backend_status
        ] = (
            backend_counts.get(
                item.backend_status,
                0,
            )
            + 1
        )

    def rate(
        predicate: Callable[
            [
                AdaptiveLoopObservation
            ],
            bool,
        ],
    ) -> float:
        count = sum(
            1
            for item in observations
            if predicate(item)
        )

        return (
            count / n
        )

    return AdaptiveLoopSummary(
        n=n,
        mean_latency_ms=round(
            sum(latencies)
            / n,
            6,
        ),
        p50_latency_ms=round(
            _percentile(
                latencies,
                50.0,
            ),
            6,
        ),
        p95_latency_ms=round(
            _percentile(
                latencies,
                95.0,
            ),
            6,
        ),
        p99_latency_ms=round(
            _percentile(
                latencies,
                99.0,
            ),
            6,
        ),
        mean_rounds=round(
            sum(rounds)
            / n,
            6,
        ),
        mean_retrieval_calls=round(
            sum(
                retrieval_calls
            )
            / n,
            6,
        ),
        mean_first_top_k=round(
            mean_first,
            6,
        ),
        mean_total_requested_top_k=round(
            sum(total_top_ks)
            / n,
            6,
        ),
        mean_selected_context_k=round(
            mean_selected,
            6,
        ),
        fixed_top_k_baseline=baseline,
        first_round_token_savings_pct=round(
            (
                1.0
                - (
                    mean_first
                    / baseline
                )
            )
            * 100.0,
            3,
        ),
        selected_context_savings_pct=round(
            (
                1.0
                - (
                    mean_selected
                    / baseline
                )
            )
            * 100.0,
            3,
        ),
        retrieval_free_rate=round(
            rate(
                lambda item:
                item.retrieval_calls
                == 0
            ),
            6,
        ),
        escalation_rate=round(
            rate(
                lambda item:
                item.escalated
            ),
            6,
        ),
        answer_rate=round(
            rate(
                lambda item:
                item.action
                == "answer"
            ),
            6,
        ),
        clarify_rate=round(
            rate(
                lambda item:
                item.action
                == "clarify"
            ),
            6,
        ),
        no_retrieval_rate=round(
            rate(
                lambda item:
                item.action
                == "no_retrieval"
            ),
            6,
        ),
        abstain_rate=round(
            rate(
                lambda item:
                item.action
                == "abstain"
            ),
            6,
        ),
        max_observed_rounds=max(
            rounds
        ),
        backend_counts=(
            backend_counts
        ),
    )


def append_observation_jsonl(
    path: str | Path,
    observation: AdaptiveLoopObservation,
) -> None:
    """Persist one Staff-track telemetry event as append-only JSONL."""

    output = Path(
        path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                observation.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
