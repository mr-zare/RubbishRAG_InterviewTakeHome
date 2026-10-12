# RubbishRAG — Engineering Proof


<!-- FINAL-COMPLIANCE:METADATA -->

## Submission Metadata and Declared Scope

| Field | Value |
|---|---|
| Candidate | Erfan Zare |
| GitHub | `@mr-zare` |
| Submission branch | `solution/mr-zare` |
| Submission date | 2026-10-12 |
| Runtime | Python 3.11 |
| Execution target | CPU-only, standard developer machine |
| External model/API dependency | None |
| Production Judge | OFF; deterministic P3/P9/P10 stack is authoritative |

### Track Scope

- [x] Required Core A — Retrieval characterization and diagnostic probing
- [x] Required Core B — Hybrid retrieval / rank fusion and retrieval engineering
- [x] Required Core C — Persian normalization and typed semantic extraction
- [x] Required Core D — Structured chunking and metadata preservation
- [x] Elective F — Adversarial / noise / provenance defense
- [x] Elective G — Polarity and negation handling
- [x] Staff Track — Adaptive System-1 Judge implemented and benchmarked
- [ ] Elective H is not claimed as one of the two selected electives, although
      deterministic clarification behavior exists in the production resolver.


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

<!-- FINAL-COMPLIANCE:EVIDENCE -->

## Evaluator-Facing Evidence Index

### Diagnostic Inventory

The final `traces/trace.jsonl` contains:

- **1022 valid trace rows**
- **33 distinct diagnostic / benchmark queries**

This exceeds the minimum diagnostic-probe requirement.

The references below use the trace file line number because
`rubbish.py inspect-trace <line>` can reproduce the corresponding logged
retrieval event without issuing a new retrieval request.

### Exact Trace References

| Trace | Diagnostic purpose | Raw remote observation | Reproduce |
|---:|---|---|---|
| 1 | C1 rank / adversarial retrieval | poison-style doc `9004` ranked above trusted doc `106` | `python rubbish.py inspect-trace 1` |
| 2 | colloquial tax-debt removal | doc `66` top-1 | `python rubbish.py inspect-trace 2` |
| 4 | returned-check / 24-hour condition | doc `282` top-1 | `python rubbish.py inspect-trace 4` |
| 5 | short numeric score explanation (`۴۸۰ شدم. چرا؟`) | doc `110` top-1 | `python rubbish.py inspect-trace 5` |
| 7 | guarantor + delayed installments | doc `276` top-1 | `python rubbish.py inspect-trace 7` |
| 8 | profitability / missing-score causal question | doc `266` top-1 | `python rubbish.py inspect-trace 8` |
| 11 | colloquial returned-check removal | doc `47` top-1 | `python rubbish.py inspect-trace 11` |
| 14 | exports/imports lexical ambiguity | doc `301` top-1; adversarial docs also retrieved | `python rubbish.py inspect-trace 14` |
| 17 | C3 adversarial retrieval | doc `9004` top-1, trusted C3 doc `108` below it | `python rubbish.py inspect-trace 17` |
| 18 | supplier-debt semantics | doc `65` top-1 | `python rubbish.py inspect-trace 18` |
| 19 | paid tax debt + E3 removal | generic tax doc `173` top-1 | `python rubbish.py inspect-trace 19` |
| 20 | no previous delay but not maximum score | doc `16` top-1 | `python rubbish.py inspect-trace 20` |
| 22 | A2 adversarial retrieval | doc `9004` top-1, trusted A2 doc `101` second | `python rubbish.py inspect-trace 22` |
| 90 | Arabic/Persian-character normalization variant | docs `165/166` retrieved identically | `python rubbish.py inspect-trace 90` |
| 94 | formal export/import wording | doc `301` top-1 | `python rubbish.py inspect-trace 94` |
| 95 | opposing-polarity export/import wording | retrieval order changes under negation | `python rubbish.py inspect-trace 95` |
| 100 | intentionally underspecified debt query | mixed debt evidence retrieved | `python rubbish.py inspect-trace 100` |
| 101 | Bank Saderat + E3 entity query | doc `303` top-1 | `python rubbish.py inspect-trace 101` |
| 102 | Bank Melli + E3 contrast query | same generic doc `303` initially top-1 | `python rubbish.py inspect-trace 102` |

These traces demonstrate why retrieval score alone cannot be treated as
answer confidence: high-fused candidates may be adversarial, overly generic,
entity-incompatible, polarity-incompatible, or simply less specific than
trusted local evidence.

### Baseline Retrieval Characterization

The diagnostic probe suite measured the baseline retrieval behavior as:

| Metric | Raw retrieval | Baseline reranker |
|---|---:|---:|
| Hit@1 | 5.88% | 0.00% |
| Hit@3 | 41.18% | 29.41% |
| Hit@5 | 47.06% | 47.06% |
| MRR@5 | 0.2275 | 0.1755 |

The baseline reranker improved **0** diagnostic cases, worsened **2**, and left
**15** unchanged.

This result motivated semantic reranking, provenance-aware candidate admission,
and verified trusted-local recall recovery rather than further tuning raw fused
score thresholds.

### Final Named Network Checkpoint

The last explicitly named network checkpoint before retrieval quota exhaustion
was P3.1:

| Metric | P3.1 |
|---|---:|
| Visible accuracy | 100% |
| Mean latency | 60.2 ms |
| P50 latency | 53.4 ms |
| P95 latency | 96.8 ms |

Post-P3.1 hardening was validated offline and is **not** presented as a new
remote benchmark measurement.

### Elective F — Adversarial / Provenance Result

Targeted adversarial tests cover:

- unknown remote IDs;
- keyword stuffing;
- forged high scores;
- trusted-ID content tampering;
- duplicate evidence;
- semantically conflicting evidence;
- low-ID untrusted candidates;
- high-ID trusted candidates.

Observed targeted-suite result:

- **untrusted / poison citation rate: 0%**
- magic numeric document-ID trust rules: **0 remaining**

Trust is determined by provenance and trusted-corpus membership, not document
number ranges or retrieval score.

### Elective G — Polarity / Negation Result

Final targeted polarity suite:

- **18 / 18 passed**
- opposing-polarity evidence accepted as citation in covered regression cases:
  **0**

The parser distinguishes proposition existence from predicate effect polarity
and scopes negation to the relevant clause/target.

### Staff Track — Adaptive Judge Benchmark

The Staff Track Judge was measured on the provided test split:

| Metric | Result |
|---|---:|
| Total queries | 15 |
| Routing Accuracy | 93.3% |
| Clarify Precision | 66.7% |
| Clarify Recall | 66.7% |
| Clarify F1 | 0.667 |
| Mean Dynamic top-k | 3.4 |
| Fixed baseline top-k | 5 |
| Context Token Savings | 32.0% |
| Brier Score | 0.0624 |
| Expected Calibration Error | 0.022 |

Despite the positive routing and budget results, the production pipeline keeps
the Judge **OFF** because P3/P9/P10 already provide deterministic semantic
understanding, trusted evidence construction, and sufficiency decisions.

P11 therefore remains an experimentally validated adaptive-budget layer rather
than an additional production authority.

### Final Offline Reproducibility

Final validation:

- offline regression suite: **119 / 119**
- three independent cold-process runs: **119 / 119 each**
- clean detached Git worktree: **119 / 119**
- submission packaging test: PASS
- packaged production modules: compile successfully
- archive secret scan: PASS
- archive source-integrity scan: PASS

Network-dependent CLI tests were not re-run after retrieval quota exhaustion.
No test was modified or bypassed to hide the quota limitation.
