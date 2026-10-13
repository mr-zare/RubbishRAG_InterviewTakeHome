#!/usr/bin/env python3
"""Final offline metric pass for the RubbishRAG glossary.

No remote retrieval is performed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# Allow this repository script to be executed directly:
#   python scripts/final_glossary_metrics.py
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from rubbish_rag.corpus import (
    get_registry,
)
from rubbish_rag.eval_metrics import (
    hit_at_k,
    ndcg_at_k,
    pareto_frontier,
    percentile,
    reciprocal_rank,
    risk_coverage_curve,
    token_savings_pct,
)
from rubbish_rag.judge import (
    HeuristicFeatureJudge,
    evaluate_judge,
)
from rubbish_rag.rerank import (
    local_verified_search,
    prepare_reranker,
)
from rubbish_rag.slots import (
    parse_query,
)


ROOT = REPO_ROOT


def load_cases(path: Path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(
        data,
        list,
    ):
        return data

    for key in (
        "items",
        "cases",
        "queries",
        "data",
        "examples",
    ):
        value = data.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            return value

    raise RuntimeError(
        f"Unknown list structure: {path}"
    )


def latest_trace_rows(
    path: Path,
):
    latest = {}

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        try:
            row = json.loads(
                line
            )
        except json.JSONDecodeError:
            continue

        query = row.get(
            "q"
        )

        hits = row.get(
            "hits"
        )

        if (
            not isinstance(
                query,
                str,
            )
            or not isinstance(
                hits,
                list,
            )
        ):
            continue

        row = dict(row)
        row["_line"] = line_number

        latest[query] = row

    return latest


def doc_id_of(hit):
    value = hit.get(
        "doc_id",
        hit.get(
            "doc",
        ),
    )

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def aggregate_ir(
    rows,
):
    if not rows:
        return {
            "n": 0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr_at_5": 0.0,
            "ndcg_at_3": 0.0,
            "ndcg_at_5": 0.0,
        }

    return {
        "n": len(rows),
        "hit_at_3": round(
            sum(
                row[
                    "hit_at_3"
                ]
                for row in rows
            )
            / len(rows),
            6,
        ),
        "hit_at_5": round(
            sum(
                row[
                    "hit_at_5"
                ]
                for row in rows
            )
            / len(rows),
            6,
        ),
        "mrr_at_5": round(
            sum(
                row[
                    "rr_at_5"
                ]
                for row in rows
            )
            / len(rows),
            6,
        ),
        "ndcg_at_3": round(
            sum(
                row[
                    "ndcg_at_3"
                ]
                for row in rows
            )
            / len(rows),
            6,
        ),
        "ndcg_at_5": round(
            sum(
                row[
                    "ndcg_at_5"
                ]
                for row in rows
            )
            / len(rows),
            6,
        ),
    }


def evaluate_ranked_ids(
    case,
    ranked_ids,
):
    gold = int(
        case[
            "gold_doc_id"
        ]
    )

    relevance = {
        gold: 1.0,
    }

    return {
        "id": case.get(
            "id"
        ),
        "query": case[
            "query"
        ],
        "gold_doc_id": gold,
        "ranked_ids": ranked_ids,
        "hit_at_3": hit_at_k(
            ranked_ids,
            {gold},
            3,
        ),
        "hit_at_5": hit_at_k(
            ranked_ids,
            {gold},
            5,
        ),
        "rr_at_5": reciprocal_rank(
            ranked_ids,
            {gold},
            k=5,
        ),
        "ndcg_at_3": ndcg_at_k(
            ranked_ids,
            relevance,
            3,
        ),
        "ndcg_at_5": ndcg_at_k(
            ranked_ids,
            relevance,
            5,
        ),
    }


def binary_confidence(
    positive_probability,
    predicted_positive,
):
    p = float(
        positive_probability
    )

    return (
        p
        if predicted_positive
        else 1.0 - p
    )


def build_plots(
    metrics,
    ablation,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "PLOT SKIP:",
            type(exc).__name__,
            exc,
        )

        return

    plots_dir = (
        ROOT / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Risk-Coverage
    # --------------------------------------------------------
    retrieval_points = metrics[
        "risk_coverage"
    ][
        "retrieval"
    ]

    valid = [
        point
        for point
        in retrieval_points
        if point[
            "selective_risk"
        ] is not None
    ]

    plt.figure(
        figsize=(
            7.5,
            5.0,
        )
    )

    plt.plot(
        [
            point[
                "coverage"
            ]
            for point in valid
        ],
        [
            point[
                "selective_risk"
            ]
            for point in valid
        ],
        marker="o",
    )

    for point in valid:
        plt.annotate(
            f'θ={point["threshold"]:.2f}',
            (
                point[
                    "coverage"
                ],
                point[
                    "selective_risk"
                ],
            ),
        )

    plt.xlabel(
        "Coverage"
    )

    plt.ylabel(
        "Selective risk"
    )

    plt.title(
        "Retrieval Routing Risk–Coverage"
    )

    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "judge_risk_coverage.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # Pareto view
    # --------------------------------------------------------
    if ablation:
        names = (
            "heuristic",
            "qwen_raw",
            "hybrid_selective",
        )

        rows = [
            (
                name,
                ablation[name],
            )
            for name in names
            if name in ablation
        ]

        plt.figure(
            figsize=(
                7.5,
                5.0,
            )
        )

        for name, row in rows:
            x = float(
                row[
                    "latency_mean_ms"
                ]
            )

            y = float(
                row[
                    "routing_accuracy"
                ]
            )

            plt.scatter(
                [x],
                [y],
                s=80,
            )

            plt.annotate(
                (
                    f"{name}\n"
                    f'k={row["mean_top_k"]}'
                ),
                (
                    x,
                    y,
                ),
            )

        plt.xscale(
            "log"
        )

        plt.xlabel(
            "Mean routing latency (ms, log scale)"
        )

        plt.ylabel(
            "Three-way action accuracy"
        )

        plt.title(
            "Judge Backend Accuracy–Latency Trade-off"
        )

        plt.tight_layout()

        plt.savefig(
            plots_dir
            / "judge_pareto_frontier.png",
            dpi=160,
        )

        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help=(
            "Local tokenizer/model directory. "
            "Loaded with local_files_only=True."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Prepare trusted local evaluation index
    # ========================================================
    prepare_reranker()

    registry = get_registry()

    heuristic = (
        HeuristicFeatureJudge()
    )

    # ========================================================
    # Official Staff Judge metrics
    # ========================================================
    official = evaluate_judge(
        judge=heuristic,
        split="test",
    )

    # ========================================================
    # IR metrics: stored raw remote retrieval replay
    # ========================================================
    visible = load_cases(
        ROOT
        / "server"
        / "visible_bench.json"
    )

    answer_cases = [
        case
        for case in visible
        if (
            case.get(
                "expect"
            )
            == "answer"
            and case.get(
                "gold_doc_id"
            )
            is not None
        )
    ]

    trace_rows = latest_trace_rows(
        ROOT
        / "traces"
        / "trace.jsonl"
    )

    remote_eval = []

    missing_trace_queries = []

    for case in answer_cases:
        query = case[
            "query"
        ]

        trace = trace_rows.get(
            query
        )

        if trace is None:
            missing_trace_queries.append(
                case.get(
                    "id"
                )
            )

            continue

        ranked_ids = [
            doc_id
            for doc_id in (
                doc_id_of(hit)
                for hit in trace[
                    "hits"
                ]
            )
            if doc_id is not None
        ]

        row = evaluate_ranked_ids(
            case,
            ranked_ids,
        )

        row[
            "trace_line"
        ] = trace[
            "_line"
        ]

        remote_eval.append(
            row
        )

    # ========================================================
    # IR metrics: current trusted-local ranking replay
    # ========================================================
    local_eval = []

    local_rankings = {}

    for case in answer_cases:
        query = case[
            "query"
        ]

        frame = parse_query(
            query
        )

        hits = local_verified_search(
            query,
            frame,
            registry,
            limit=5,
        )

        ranked_ids = [
            int(
                hit[
                    "doc_id"
                ]
            )
            for hit in hits
        ]

        local_rankings[
            query
        ] = hits

        local_eval.append(
            evaluate_ranked_ids(
                case,
                ranked_ids,
            )
        )

    # ========================================================
    # Risk-Coverage on official judge test split
    # ========================================================
    judge_rows = [
        row
        for row in load_cases(
            ROOT
            / "data"
            / "judge_dataset.json"
        )
        if row.get(
            "split"
        )
        == "test"
    ]

    retrieval_confidences = []
    retrieval_correct = []

    clarify_confidences = []
    clarify_correct = []

    pre_decisions = {}

    for row in judge_rows:
        query = row[
            "query"
        ]

        decision = (
            heuristic.pre_retrieval(
                query
            )
        )

        pre_decisions[
            query
        ] = decision

        gold_retrieve = (
            row.get(
                "expected_retrieval"
            )
            == "RETRIEVE"
        )

        pred_retrieve = bool(
            decision.needs_retrieval
        )

        retrieval_confidences.append(
            binary_confidence(
                decision.retrieve_probability,
                pred_retrieve,
            )
        )

        retrieval_correct.append(
            pred_retrieve
            == gold_retrieve
        )

        gold_clarify = (
            row.get(
                "expected_clarification"
            )
            == "CLARIFY"
        )

        pred_clarify = bool(
            decision.needs_clarification
        )

        clarify_confidences.append(
            binary_confidence(
                decision.clarify_probability,
                pred_clarify,
            )
        )

        clarify_correct.append(
            pred_clarify
            == gold_clarify
        )

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
    ]

    retrieval_curve = (
        risk_coverage_curve(
            retrieval_confidences,
            retrieval_correct,
            thresholds,
        )
    )

    clarification_curve = (
        risk_coverage_curve(
            clarify_confidences,
            clarify_correct,
            thresholds,
        )
    )

    # ========================================================
    # Existing OpenWeight / Hybrid ablation + p99
    # ========================================================
    ablation_path = (
        ROOT
        / "traces"
        / "judge_backend_ablation.json"
    )

    if ablation_path.exists():
        ablation = json.loads(
            ablation_path.read_text(
                encoding="utf-8"
            )
        )
    else:
        ablation = {}

    latency_percentiles = {}

    pareto_rows = []

    for name in (
        "heuristic",
        "qwen_raw",
        "hybrid_selective",
    ):
        row = ablation.get(
            name
        )

        if not isinstance(
            row,
            dict,
        ):
            continue

        latencies = [
            float(
                record[
                    "latency_ms"
                ]
            )
            for record
            in row.get(
                "records",
                []
            )
            if record.get(
                "latency_ms"
            )
            is not None
        ]

        latency_percentiles[
            name
        ] = {
            "n": len(
                latencies
            ),
            "p50_ms": round(
                percentile(
                    latencies,
                    50,
                ),
                3,
            ),
            "p95_ms": round(
                percentile(
                    latencies,
                    95,
                ),
                3,
            ),
            "p99_ms": round(
                percentile(
                    latencies,
                    99,
                ),
                3,
            ),
        }

        pareto_rows.append(
            {
                "name": name,
                # This ablation uses stricter three-way exact action matching.
                "accuracy": float(
                    row[
                        "routing_accuracy"
                    ]
                ),
                "latency_ms": float(
                    row[
                        "latency_mean_ms"
                    ]
                ),
                "mean_top_k": float(
                    row[
                        "mean_top_k"
                    ]
                ),
            }
        )

    pareto_3d = pareto_frontier(
        pareto_rows,
        maximize=[
            "accuracy",
        ],
        minimize=[
            "latency_ms",
            "mean_top_k",
        ],
    )

    pareto_2d = pareto_frontier(
        pareto_rows,
        maximize=[
            "accuracy",
        ],
        minimize=[
            "latency_ms",
        ],
    )

    # ========================================================
    # Actual context-token savings with local Qwen tokenizer
    # ========================================================
    token_metric = {
        "status": (
            "skipped_no_tokenizer_model"
        )
    }

    if args.tokenizer_model:
        from transformers import (
            AutoTokenizer,
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                args.tokenizer_model,
                local_files_only=True,
            )
        )

        fixed_tokens_total = 0
        dynamic_tokens_total = 0

        per_query = []

        for row in judge_rows:
            query = row[
                "query"
            ]

            decision = (
                pre_decisions[
                    query
                ]
            )

            frame = parse_query(
                query
            )

            limit = max(
                5,
                int(
                    decision.top_k
                ),
            )

            hits = local_verified_search(
                query,
                frame,
                registry,
                limit=limit,
            )

            def text_for_hit(hit):
                try:
                    doc_id = int(
                        hit.get(
                            "doc_id",
                            -1,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    doc_id = -1

                record = (
                    registry.get(
                        doc_id
                    )
                    if doc_id >= 0
                    else None
                )

                if record is not None:
                    return str(
                        record.retrieval_text
                    )

                return " ".join(
                    str(
                        hit.get(
                            key
                        )
                        or ""
                    )
                    for key in (
                        "question",
                        "keyword",
                        "brief_answer",
                        "text",
                    )
                )

            def count_hits(
                selected_hits
            ):
                total = 0

                for hit in selected_hits:
                    text = text_for_hit(
                        hit
                    )

                    total += len(
                        tokenizer.encode(
                            text,
                            add_special_tokens=False,
                        )
                    )

                return total

            fixed_tokens = count_hits(
                hits[:5]
            )

            dynamic_tokens = count_hits(
                hits[
                    : int(
                        decision.top_k
                    )
                ]
            )

            fixed_tokens_total += (
                fixed_tokens
            )

            dynamic_tokens_total += (
                dynamic_tokens
            )

            per_query.append(
                {
                    "id": row.get(
                        "id"
                    ),
                    "query": query,
                    "dynamic_top_k": (
                        decision.top_k
                    ),
                    "fixed_k5_tokens": (
                        fixed_tokens
                    ),
                    "dynamic_tokens": (
                        dynamic_tokens
                    ),
                }
            )

        token_metric = {
            "status": "measured",
            "tokenizer": str(
                Path(
                    args.tokenizer_model
                ).name
            ),
            "n": len(
                judge_rows
            ),
            "fixed_k5_tokens": (
                fixed_tokens_total
            ),
            "dynamic_tokens": (
                dynamic_tokens_total
            ),
            "actual_token_savings_pct": round(
                token_savings_pct(
                    fixed_tokens_total,
                    dynamic_tokens_total,
                ),
                3,
            ),
            "per_query": per_query,
        }

    # ========================================================
    # Final artifact
    # ========================================================
    metrics = {
        "scope": {
            "network_retrieval_used": False,
            "runtime_behavior_modified_by_this_script": False,
            "notes": [
                (
                    "IR trace replay uses already-recorded remote candidate "
                    "lists; no new retrieval is issued."
                ),
                (
                    "verified_local_replay evaluates the current trusted-local "
                    "ranking subsystem only."
                ),
                (
                    "backend ablation accuracy is stricter three-way action "
                    "matching and is not the official binary routing metric."
                ),
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "official_judge_test": official,
        "ir": {
            "stored_remote_trace_replay": {
                "summary": aggregate_ir(
                    remote_eval
                ),
                "missing_trace_case_ids": (
                    missing_trace_queries
                ),
                "rows": remote_eval,
            },
            "verified_local_replay": {
                "summary": aggregate_ir(
                    local_eval
                ),
                "rows": local_eval,
            },
        },
        "risk_coverage": {
            "retrieval": retrieval_curve,
            "clarification": (
                clarification_curve
            ),
        },
        "backend_latency_percentiles": (
            latency_percentiles
        ),
        "actual_context_token_savings": (
            token_metric
        ),
        "pareto": {
            "objectives": {
                "maximize": [
                    "accuracy",
                ],
                "minimize": [
                    "latency_ms",
                    "mean_top_k",
                ],
            },
            "all_configs": (
                pareto_rows
            ),
            "frontier_3d": (
                pareto_3d
            ),
            "accuracy_latency_frontier": (
                pareto_2d
            ),
        },
        "glossary_coverage": {
            "hit_at_k": True,
            "mrr": True,
            "ndcg": True,
            "rrf": True,
            "brier": True,
            "ece": True,
            "selective_prediction": True,
            "risk_coverage": True,
            "latency_p50": True,
            "latency_p95": True,
            "latency_p99": True,
            "context_token_savings": (
                token_metric.get(
                    "status"
                )
                == "measured"
            ),
            "pareto_frontier": bool(
                pareto_rows
            ),
        },
    }

    out = (
        ROOT
        / "traces"
        / "glossary_metrics.json"
    )

    out.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    build_plots(
        metrics,
        ablation,
    )

    print(
        json.dumps(
            {
                "official_judge_test": (
                    metrics[
                        "official_judge_test"
                    ]
                ),
                "ir_remote": (
                    metrics[
                        "ir"
                    ][
                        "stored_remote_trace_replay"
                    ][
                        "summary"
                    ]
                ),
                "ir_verified_local": (
                    metrics[
                        "ir"
                    ][
                        "verified_local_replay"
                    ][
                        "summary"
                    ]
                ),
                "latency_percentiles": (
                    latency_percentiles
                ),
                "token_savings": (
                    token_metric
                ),
                "pareto_3d": (
                    pareto_3d
                ),
                "glossary_coverage": (
                    metrics[
                        "glossary_coverage"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nSAVED:",
        out,
    )


if __name__ == "__main__":
    main()
