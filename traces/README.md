# Experiment Metric Snapshots

This directory contains immutable, named metric snapshots retained as engineering evidence.

`metrics.json` is intentionally excluded because it is a mutable runtime artifact overwritten by later benchmark executions.

JSONL retrieval traces are also excluded from the submission repository; the named metric snapshots and plots are the stable evidence artifacts.

| File | Status | Accuracy | Mean ms | P50 ms | P95 ms | SHA-256/12 |
|---|---|---:|---:|---:|---:|---|
| `metrics_aborted_v1_5_patch_failed.json` | Rejected / invalid experiment | 0.95 | 80.5 | 72.4 | 119.2 | `8843a61363fb` |
| `metrics_aborted_v1_6A_interrogative_bug.json` | Rejected / invalid experiment | 0.95 | 63.0 | 47.1 | 120.5 | `e26f28db724e` |
| `metrics_baseline.json` | Baseline | 0.3 | 43.2 | 44.6 | 59.0 | `de62868d2c11` |
| `metrics_core_p3_1.json` | Last explicitly named network checkpoint | 1.0 | 60.2 | 53.4 | 96.8 | `a69264cddc88` |
| `metrics_core_v1_1.json` | Historical milestone | 0.55 | 41.4 | 41.1 | 49.2 | `15fdda4a5d22` |
| `metrics_core_v1_2.json` | Historical milestone | 0.85 | 73.3 | 43.6 | 186.3 | `3867c7ccc21d` |
| `metrics_core_v1_3.json` | Historical milestone | 0.85 | 44.9 | 39.6 | 73.0 | `fc7a9bc8e25c` |
| `metrics_core_v1_4.json` | Historical milestone | 0.95 | 48.9 | 41.4 | 74.9 | `6bb6caeef872` |
| `metrics_core_v1_5.json` | Historical milestone | 1.0 | 63.1 | 46.1 | 98.8 | `7ac7a919f2c2` |
| `metrics_core_v1_55.json` | Historical milestone | 0.55 | 95.7 | 97.7 | 118.5 | `bfcea88a3ff0` |
| `metrics_core_v1_6B.json` | Historical milestone | 0.95 | 67.7 | 57.8 | 109.7 | `eb3c2c1acda9` |
| `metrics_core_v1_6B_run1.json` | Repeated partial v1.6B validation | 0.95 | 76.2 | 56.3 | 139.2 | `06aa9a33ab78` |
| `metrics_core_v1_6B_run2.json` | Repeated partial v1.6B validation | 0.95 | 57.4 | 42.8 | 102.3 | `cd32c7e1331f` |
| `metrics_core_v1_6B_run3.json` | Repeated partial v1.6B validation | 0.95 | 67.9 | 49.2 | 127.2 | `76503e746d6a` |
| `metrics_core_v1_6C.json` | Historical milestone | 1.0 | 43.1 | 42.2 | 48.6 | `e34da2b61cc3` |
| `metrics_core_v1_6C_run1.json` | Repeated v1.6C validation | 1.0 | 58.0 | 54.0 | 89.9 | `1c14a9b80553` |
| `metrics_core_v1_6C_run2.json` | Repeated v1.6C validation | 1.0 | 41.5 | 40.8 | 53.3 | `7edbaad1dd41` |
| `metrics_core_v1_6C_run3.json` | Repeated v1.6C validation | 1.0 | 38.7 | 38.2 | 43.8 | `0379d0e7ede6` |
| `metrics_core_v1_6_run1.json` | Historical milestone | 0.95 | 71.1 | 62.5 | 119.2 | `e22bb64a1017` |
| `metrics_core_v1_6_run2.json` | Historical milestone | 0.95 | 56.8 | 44.7 | 97.6 | `ac3169b1acfa` |
| `metrics_core_v1_6_run3.json` | Historical milestone | 0.95 | 56.9 | 44.3 | 98.7 | `d5710b530c99` |

## Interpretation

- Rejected/aborted snapshots are intentionally retained as negative evidence.
- A 100% visible score is not treated as proof of hidden robustness.
- Post-P3.1 hardening was primarily validated through offline regression, metamorphic, adversarial, polarity, provenance, and determinism tests.
- P11 remains production-OFF pending an end-to-end network cost/latency ablation.
