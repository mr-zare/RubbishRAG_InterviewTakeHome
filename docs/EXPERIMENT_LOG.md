# RubbishRAG Experiment Log

This document records the experimental progression of the solution.
Each iteration is evaluated before being accepted or rejected.

## P0 — Baseline

- Visible benchmark: 6/20 = 30%
- Mean latency: 43.2 ms
- P95 latency: 59.0 ms
- Raw retrieval on diagnostic probes:
  - Hit@1: 5.88%
  - Hit@3: 41.18%
  - Hit@5: 47.06%
  - MRR@5: 0.2275
- Baseline reranker:
  - Hit@1: 0%
  - Hit@3: 29.41%
  - Hit@5: 47.06%
  - MRR@5: 0.1755
  - Improved: 0
  - Worsened: 2
  - Unchanged: 15

Key findings:
- Baseline reranker harmed ranking quality.
- Poison documents could dominate retrieval.
- Fused score was not a calibrated confidence measure.
- Persian digits, ZWNJ, query length and phrasing materially changed retrieval.

Decision:
REJECT baseline reranker design.

---

## Core v1.1 — Structured deterministic RAG core

Main changes:
- Trusted corpus registry and provenance handling.
- Persian normalization with original/canonical/match representations.
- Typed query semantic frame.
- Intent/entity/persona/rank/debt/quantity/polarity parsing.
- RRF-based retrieval prior.
- Typed semantic reranking.
- Parent/evidence deduplication.
- Grounded deterministic resolver.
- Dynamic clarification for missing critical slots.
- Evidence requirement coverage.

Validation:
- Tests: 48/48 passed
- Visible benchmark: 11/20 = 55%
- Mean latency: 41.4 ms
- P95 latency: 49.2 ms

Important fixes:
- Poison result for C1 was removed from answer evidence.
- C1 resolved to trusted doc 106.
- Ambiguous "بدهی دارم" correctly triggered CLARIFY.
- False Bank Saderat detection in export/import context fixed.
- Supplier debt added to debt taxonomy.
- Evidence selection changed from fixed top-3 to smallest useful set.

Decision:
ACCEPT architecture, but candidate recall still insufficient.

---

## Core v1.2 — Conditional verified local recall recovery

Hypothesis:
Remote top-5 candidate recall is the dominant remaining bottleneck.
A verified local fallback should run only when the remote pool does not
contain a candidate covering the critical semantic requirements.

Main changes:
- Requirement-aware backfill trigger.
- Verified local corpus fallback.
- No benchmark-label or document-ID hardcoding.
- Existing semantic scorer reused for local candidates.

Key recovered examples:
- Export/import factor query -> doc 34 ranked #1 locally.
- Supplier debt query -> doc 123 ranked #1 locally.
- "480 شدم. چرا؟" -> doc 311 ranked #1 locally.

Validation:
- Tests: 48/48 passed
- Visible benchmark: 17/20 = 85%
- Mean latency: 73.3 ms
- P95 latency: 186.3 ms

Finding:
Accuracy improved dramatically, but exhaustive semantic scoring over all
330 trusted documents produced unacceptable tail latency.

Decision:
ACCEPT recall strategy, REJECT exhaustive implementation.

---

## Core v1.3 — Indexed two-stage verified backfill

Hypothesis:
Precomputed lexical/semantic postings can preserve local recall while
restricting expensive semantic scoring to a small high-recall shortlist.

Main changes:
- Precomputed token postings.
- Precomputed intent postings.
- Precomputed entity postings.
- Precomputed typed-quantity postings.
- Cheap query-dependent shortlist.
- Full semantic scorer applied only to shortlist.
- Maximum shortlist size: 96.

Local microbenchmarks:

| Probe | Shortlist | Local mean | Local p95 | Expected doc |
|---|---:|---:|---:|---:|
| Export/import | 44 | 6.74 ms | 7.56 ms | 34 |
| Supplier debt | 96 | 28.14 ms | 31.66 ms | 123 |
| Score 480 | 13 | 1.75 ms | 2.15 ms | 311 |

End-to-end validation:
- Tests: 48/48 passed
- Visible benchmark: 17/20 = 85%
- Mean latency: 44.9 ms
- P95 latency: 73.0 ms
- Accuracy delta vs v1.2: 0
- Mean latency delta vs v1.2: -38.7%
- P95 latency delta vs v1.2: -60.8%

Remaining visible failures:
- v6
- v17
- v18

Decision:
FREEZE.
Accuracy is preserved and CPU p95 is below the 100 ms target.

