import pytest

from rubbish_rag.eval_metrics import (
    hit_at_k,
    ndcg_at_k,
    pareto_frontier,
    percentile,
    reciprocal_rank,
    risk_coverage_curve,
    token_savings_pct,
)


def test_hit_at_k():
    ranked = [
        1,
        2,
        3,
        4,
    ]

    assert (
        hit_at_k(
            ranked,
            {3},
            3,
        )
        == 1.0
    )

    assert (
        hit_at_k(
            ranked,
            {4},
            3,
        )
        == 0.0
    )


def test_reciprocal_rank():
    assert reciprocal_rank(
        [10, 20, 30],
        {20},
        k=3,
    ) == pytest.approx(
        0.5
    )


def test_ndcg_perfect_and_late():
    perfect = ndcg_at_k(
        [7, 8, 9],
        {7},
        3,
    )

    late = ndcg_at_k(
        [8, 7, 9],
        {7},
        3,
    )

    assert perfect == pytest.approx(
        1.0
    )

    assert 0.0 < late < perfect


def test_ndcg_supports_graded_relevance():
    score = ndcg_at_k(
        ["b", "a"],
        {
            "a": 2.0,
            "b": 1.0,
        },
        2,
    )

    assert 0.0 < score < 1.0


def test_risk_coverage_curve():
    points = risk_coverage_curve(
        [
            0.95,
            0.90,
            0.60,
            0.55,
        ],
        [
            True,
            True,
            False,
            False,
        ],
        [
            0.50,
            0.80,
        ],
    )

    assert points[0][
        "coverage"
    ] == 1.0

    assert points[0][
        "selective_risk"
    ] == 0.5

    assert points[1][
        "coverage"
    ] == 0.5

    assert points[1][
        "selective_risk"
    ] == 0.0


def test_percentile_tail():
    values = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    assert percentile(
        values,
        50,
    ) == 3.0

    assert percentile(
        values,
        99,
    ) >= percentile(
        values,
        95,
    )


def test_token_savings():
    assert token_savings_pct(
        1000,
        680,
    ) == pytest.approx(
        32.0
    )


def test_pareto_frontier():
    configs = [
        {
            "name": "a",
            "accuracy": 0.90,
            "latency": 10.0,
            "tokens": 100.0,
        },
        {
            "name": "b",
            "accuracy": 0.90,
            "latency": 20.0,
            "tokens": 100.0,
        },
        {
            "name": "c",
            "accuracy": 0.95,
            "latency": 30.0,
            "tokens": 80.0,
        },
    ]

    frontier = pareto_frontier(
        configs,
        maximize=[
            "accuracy",
        ],
        minimize=[
            "latency",
            "tokens",
        ],
    )

    names = {
        row["name"]
        for row in frontier
    }

    assert "a" in names
    assert "b" not in names
    assert "c" in names
