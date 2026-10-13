"""Offline evaluation metrics used by the final RubbishRAG evidence pass.

These functions are evaluation-only utilities.  They do not participate in
runtime retrieval, ranking, or answerability.
"""

from __future__ import annotations

import math
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any


def percentile(
    values: Sequence[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    if not 0.0 <= p <= 100.0:
        raise ValueError(
            "percentile must be in [0, 100]"
        )

    ordered = sorted(
        float(value)
        for value in values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * p
        / 100.0
    )

    lo = int(
        math.floor(position)
    )

    hi = int(
        math.ceil(position)
    )

    if lo == hi:
        return ordered[lo]

    fraction = (
        position - lo
    )

    return (
        ordered[lo]
        + (
            ordered[hi]
            - ordered[lo]
        )
        * fraction
    )


def hit_at_k(
    ranked_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int,
) -> float:
    relevant = set(
        relevant_ids
    )

    if not relevant:
        return 0.0

    return float(
        any(
            item in relevant
            for item in ranked_ids[
                : max(0, int(k))
            ]
        )
    )


def reciprocal_rank(
    ranked_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int | None = None,
) -> float:
    relevant = set(
        relevant_ids
    )

    if not relevant:
        return 0.0

    candidates = (
        ranked_ids
        if k is None
        else ranked_ids[
            : max(
                0,
                int(k),
            )
        ]
    )

    for rank, item in enumerate(
        candidates,
        start=1,
    ):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[Any],
    relevance: (
        Mapping[Any, float]
        | Iterable[Any]
    ),
    k: int,
) -> float:
    """Normalized DCG using the linear gain definition in the challenge docs."""

    if isinstance(
        relevance,
        Mapping,
    ):
        gains = {
            key: float(value)
            for key, value
            in relevance.items()
            if float(value) > 0.0
        }
    else:
        gains = {
            item: 1.0
            for item in relevance
        }

    if not gains:
        return 0.0

    limit = max(
        0,
        int(k),
    )

    dcg = 0.0

    for rank, item in enumerate(
        ranked_ids[:limit],
        start=1,
    ):
        gain = gains.get(
            item,
            0.0,
        )

        if gain <= 0.0:
            continue

        dcg += (
            gain
            / math.log2(
                rank + 1
            )
        )

    ideal_gains = sorted(
        gains.values(),
        reverse=True,
    )[:limit]

    idcg = sum(
        gain
        / math.log2(
            rank + 1
        )
        for rank, gain
        in enumerate(
            ideal_gains,
            start=1,
        )
    )

    if idcg <= 0.0:
        return 0.0

    return dcg / idcg


def risk_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float],
) -> list[dict[str, float | int | None]]:
    if len(confidences) != len(correct):
        raise ValueError(
            "confidence/correct length mismatch"
        )

    total = len(
        confidences
    )

    points: list[
        dict[str, float | int | None]
    ] = []

    for threshold in thresholds:
        accepted = [
            index
            for index, confidence
            in enumerate(confidences)
            if float(confidence)
            >= float(threshold)
        ]

        accepted_count = len(
            accepted
        )

        coverage = (
            accepted_count / total
            if total
            else 0.0
        )

        if not accepted:
            risk = None
            accuracy = None
        else:
            errors = sum(
                1
                for index in accepted
                if not bool(
                    correct[index]
                )
            )

            risk = (
                errors
                / accepted_count
            )

            accuracy = (
                1.0 - risk
            )

        points.append(
            {
                "threshold": float(
                    threshold
                ),
                "coverage": float(
                    coverage
                ),
                "selective_risk": (
                    None
                    if risk is None
                    else float(risk)
                ),
                "accepted_accuracy": (
                    None
                    if accuracy is None
                    else float(
                        accuracy
                    )
                ),
                "accepted_count": (
                    accepted_count
                ),
                "total": total,
            }
        )

    return points


def token_savings_pct(
    fixed_tokens: int,
    dynamic_tokens: int,
) -> float:
    if fixed_tokens <= 0:
        return 0.0

    return (
        1.0
        - (
            float(dynamic_tokens)
            / float(fixed_tokens)
        )
    ) * 100.0


def pareto_frontier(
    configs: Sequence[
        Mapping[str, Any]
    ],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[dict[str, Any]]:
    """Return all non-dominated configurations."""

    rows = [
        dict(config)
        for config in configs
    ]

    def dominates(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> bool:
        all_not_worse = True
        strictly_better = False

        for key in maximize:
            lv = float(
                left[key]
            )

            rv = float(
                right[key]
            )

            if lv < rv:
                all_not_worse = False

            if lv > rv:
                strictly_better = True

        for key in minimize:
            lv = float(
                left[key]
            )

            rv = float(
                right[key]
            )

            if lv > rv:
                all_not_worse = False

            if lv < rv:
                strictly_better = True

        return (
            all_not_worse
            and strictly_better
        )

    frontier = []

    for index, candidate in enumerate(
        rows
    ):
        dominated = any(
            dominates(
                other,
                candidate,
            )
            for other_index, other
            in enumerate(rows)
            if other_index != index
        )

        if not dominated:
            frontier.append(
                candidate
            )

    return frontier
