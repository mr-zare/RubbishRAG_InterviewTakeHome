# RubbishRAG — Engineering Proof

## 1. Submission Summary

This submission replaces a weak retrieval-first baseline with a deterministic,
provenance-aware Persian RAG pipeline.

The final design separates four concerns:

1. query understanding and Persian normalization;
2. candidate retrieval and semantic reranking;
3. trusted evidence construction and answerability;
4. robustness, adversarial validation, and experimental control.

The production path does not use an LLM judge for answerability. P3 structured
semantics, P9 evidence construction, and P10 deterministic sufficiency remain
the authoritative decision stack.

---

## 2. Baseline

The initial visible benchmark result was:

| Metric | Baseline |
|---|---:|
| Accuracy | 30% |
| Answer accuracy | 30% |
| Mean latency | 43.2 ms |
| P50 latency | 44.6 ms |
| P95 latency | 59.0 ms |

Initial retrieval diagnostics also showed poor ranking quality and vulnerability
to adversarial keyword/repetition effects.

The baseline therefore had both a relevance problem and a trust problem.

Evidence:
`traces/metrics_baseline.json`

---

## 3. Architecture

The implemented architecture includes:

- deterministic configuration and provenance;
- trusted local corpus registry;
- Persian multi-view normalization;
- typed semantic query frames;
- structured chunking;
- remote-first retrieval and RRF;
- provenance-aware semantic reranking;
- trusted local recall recovery;
- indexed local candidate generation;
- smallest-sufficient evidence construction;
- deterministic answer / clarify sufficiency gating;
- metamorphic and adversarial regression tests.

Important invariant:

> retrieval score is a ranking signal, not evidence confidence.

Important trust invariant:

> document-ID values do not establish trust.

---

## 4. Measured Network Progression

| Checkpoint | Accuracy | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|---:|
| Baseline | 30% | 43.2 | 44.6 | 59.0 |
| Core v1.1 | 55% | 41.4 | 41.1 | 49.2 |
| Core v1.2 | 85% | 73.3 | 43.6 | 186.3 |
| Core v1.3 | 85% | 44.9 | 39.6 | 73.0 |
| Core v1.4 | 95% | 48.9 | 41.4 | 74.9 |
| Core v1.5 | 100% | 63.1 | 46.1 | 98.8 |
| Core v1.6C run 1 | 100% | 58.0 | 54.0 | 89.9 |
| Core v1.6C run 2 | 100% | 41.5 | 40.8 | 53.3 |
| Core v1.6C run 3 | 100% | 38.7 | 38.2 | 43.8 |
| P3.1 | 100% | 60.2 | 53.4 | 96.8 |

The three independent v1.6C runs all achieved 100% visible accuracy.

Their median:
- accuracy: 100%;
- mean latency: 41.5 ms;
- P95 latency: 53.3 ms.

Named metric snapshots under `traces/` are the source of truth for these
measurements.

`traces/metrics.json` is intentionally treated as a mutable runtime artifact.

---

## 5. Key Engineering Iterations

### Core v1.1

Added typed semantics for previously ambiguous cases, including supplier debt,
and corrected the lexical collision between the word "صادرات" and Bank Saderat.

Evidence construction was changed to stop when query requirements were already
covered instead of adding unnecessary context.

Result:
55% visible accuracy.

### Core v1.2

Added conditional verified-local recall recovery when remote candidates failed
to cover the critical semantic query frame.

Result:
85% accuracy, but P95 increased to 186.3 ms.

This confirmed the recall hypothesis but exposed an efficiency problem.

### Core v1.3

Replaced exhaustive local scoring with indexed candidate generation using
token, intent, entity, and quantity postings.

Result:
accuracy remained 85% while P95 dropped from 186.3 ms to 73.0 ms.

### Core v1.4

Added non-numeric causal score/rank intent detection and question-level source
specificity.

Result:
95% accuracy.

### Core v1.5

Added a normalized exact FAQ-question specificity feature.

An initial patch attempt was invalid because the intended source block was not
modified. That experiment was explicitly rejected and retained as negative
evidence.

The corrected implementation reached 100% visible accuracy.

### Core v1.6

Polarity work exposed an important distinction between:
- document answer polarity;
- FAQ-question polarity;
- interrogative semantics.

An initial interrogative implementation was rejected.

v1.6C moved exact trusted FAQ admission into candidate generation and restored
100% visible accuracy across three repeated runs.

### P3.1

Refined intent focus to distinguish debt removal/payment questions from
questions asking about the effect of debt or delay on credit score/rank.

Visible benchmark remained 100%.

---

## 6. P10 — Deterministic Evidence Sufficiency

Answerability is based on semantic evidence rather than retrieval thresholds.

Critical requirements include relevant:
- intent;
- bank;
- rank;
- debt type;
- typed quantity.

Evidence selection prioritizes:

1. trusted / citation-eligible evidence;
2. semantically consistent evidence;
3. requirement coverage;
4. reranking score.

A high-scoring conflicting candidate therefore cannot suppress clean supporting
evidence.

The resolver returns clarification when trusted evidence is insufficient.

---

## 7. Elective F — Adversarial / Noise Defense

The adversarial suite covers:

- unknown remote document IDs;
- keyword-stuffed untrusted candidates;
- forged high retrieval/reranking scores;
- duplicate candidate injection;
- known-ID payload tampering;
- conflicting evidence;
- citation laundering attempts.

A red test exposed that an equally trusted but semantically conflicting
high-scoring candidate could suppress clean evidence.

The fix changed evidence selection to prefer semantic consistency before score.

Result:
adversarial tests pass and untrusted evidence cannot become a citation.

---

## 8. Elective G — Polarity and Negation

Persian polarity handling was validated beyond simple negation keywords.

The implementation distinguishes:
- debt existence;
- effect predicates;
- declarative polarity;
- interrogative uncertainty;
- clause scope;
- predicate target scope.

Example:

`بدهی مالیاتی روی امتیاز تاثیر ندارد.`

is represented as:
- tax debt existence: positive;
- EFFECT: negative.

The verb `ندارد` negates the effect, not existence of the debt.

Final targeted polarity suite:
18 / 18 passed.

---

## 9. P11 — Adaptive Judge Ablation

The legacy Judge contained two unsafe assumptions:

1. retrieval/fusion score was treated as calibrated evidence confidence;
2. numeric document-ID ranges were used as poison indicators.

Red tests showed that an explicitly untrusted candidate with a high fused score
could incorrectly be considered sufficient.

The Judge was repaired to reuse the authoritative P3/P9/P10 semantics.

A further cold/warm regression exposed cache-dependent document semantics.
Trusted-document frames are now deterministically reconstructed from the local
trusted corpus, so cache state affects performance only, not meaning.

Three independent cold P11 runs were identical.

Adaptive context-budget tests also validated deterministic early stopping.

Production decision:

> P11 remains production-OFF.

Its only remaining unique potential benefit is adaptive retrieval/context
budgeting, which requires a controlled network cost/latency ablation before
activation.

---

## 10. Final Security Hardening

A final static audit found one residual rule in `adaptive_select_hits`:

`doc_id >= 9000 -> reject`

This was removed.

Adaptive candidate admission is now provenance-based.

Final direct behavioral probe confirmed:

- trusted synthetic evidence with a high numeric ID is accepted;
- explicitly untrusted evidence with a low numeric ID is rejected.

After the fix:

- no runtime magic poison-ID thresholds remain;
- no legacy fused-score-as-confidence rules remain;
- no runtime visible-benchmark gold access remains.

---

## 11. Final Offline Validation

Final offline suite:

**119 / 119 passed**

Three independent cold-process runs:

- 119 / 119;
- 119 / 119;
- 119 / 119.

A clean detached Git worktree built from committed HEAD also produced:

- Python compilation: PASS;
- offline suite: 119 / 119;
- packaging test: PASS.

This demonstrates that the implementation does not depend on local untracked
files or a pre-warmed process state.

---

## 12. Negative Results Retained

Rejected or invalid experiments are intentionally retained.

Examples include:

- invalid v1.5 patch attempt;
- v1.6A interrogative/polarity regression;
- v1.6B partial success at 95%;
- initial P11 score-as-confidence behavior;
- warm-cache semantic nondeterminism;
- magic document-ID trust heuristic.

Negative results are part of the engineering evidence rather than being hidden
from the final report.

Evidence:
- `traces/metrics_aborted_v1_5_patch_failed.json`
- `traces/metrics_aborted_v1_6A_interrogative_bug.json`
- `docs/EXPERIMENT_LOG.md`

---

## 13. Evidence Artifacts

Stable evidence committed in the repository includes:

- `traces/README.md`
- named `traces/metrics_*.json` snapshots;
- `plots/report_card.png`;
- `plots/score_spread.png`;
- `plots/length_vs_bm25.png`;
- `plots/repetition_vs_bm25.png`;
- `docs/EXPERIMENT_LOG.md`.

The mutable runtime files `traces/metrics.json` and append-only runtime traces
are not used as historical checkpoint evidence.

---

## 14. Validation Boundary

The final P10/P12/F/G/P11/H1-H3 hardening stages were validated primarily with
offline regression, metamorphic, polarity, adversarial, provenance, and
determinism tests.

Remote retrieval quota was unavailable for a final post-hardening network
benchmark.

Therefore this submission does NOT claim a new remote visible-benchmark
measurement after the last named network checkpoint.

This distinction is intentional.

---

## 15. Final Decision

The submitted production architecture keeps:

- deterministic Persian semantic parsing;
- trusted-corpus provenance;
- semantic reranking;
- verified recall recovery;
- smallest-sufficient evidence;
- deterministic answerability;
- adversarial defenses;
- polarity-aware semantics.

The adaptive Judge remains experimental and production-OFF.

The final implementation favors deterministic, explainable, reproducible
behavior over adding an additional decision layer without demonstrated
end-to-end benefit.
