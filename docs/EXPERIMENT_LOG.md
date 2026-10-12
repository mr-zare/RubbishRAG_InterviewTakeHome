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


---

## Core v1.3 — Remaining Failure Analysis

Visible failures:
- v6
- v17
- v18

### Diagnostic tooling issue

The first failure-analysis script paired benchmark cases with trace rows by
position in an append-only trace file. Because the trace contained prior probe
runs, this produced incorrect case/trace associations.

The diagnostic was corrected to match traces by exact query text.

Lesson:
Observability tooling itself must have stable identifiers; positional alignment
is unsafe for append-only traces.

### v6 — Missing explanatory intent

Query:
"شرکت سودآوره ولی امتیاز نداره چرا؟"

Observed:
The query parser classified the query as GENERAL_INFO. As a result, the
requirement model did not encode the "why is my score missing/low?" information
need. A company-related but semantically wrong remote result was therefore
considered sufficient.

Root cause:
EXPLAIN_SCORE detection required an explicit numeric credit score.

Required fix:
Support causal score/rank questions without requiring a numeric score.

### v17 — Semantic coverage was not enough for source specificity

Query:
A paid tax-debt / E3 removal question.

Observed:
Remote doc 173 covered TAX_DEBT + E3 + DEBT_REMOVAL and produced a semantically
reasonable answer, but the benchmark expected the more specific FAQ doc 237.

Root cause:
The verified-backfill trigger considered typed semantic slots sufficient and
did not consider question-level lexical specificity.

Required fix:
A remote candidate must satisfy both critical semantic constraints and a
minimum human-FAQ-question coverage before suppressing local recall recovery.

This is a source-selection precision problem, not an answer hallucination.

### v18 — Missing explanatory intent

Query:
"چرا وقتی شرکت هیچ سابقه تاخیری ندارد، بالاترین امتیاز رو نداره؟"

Observed:
The parser again produced a frame too generic to represent the causal
score-explanation requirement. A tax-related document was selected.

Root cause:
Same non-numeric EXPLAIN_SCORE gap as v6.

Decision:
Implement Core v1.4 with:
1. non-numeric causal score/rank intent detection;
2. question-specificity-aware verified backfill triggering.


---

## Core v1.4 — Semantic Completeness and Causal Intent

Hypothesis:
Two remaining failures were caused by incomplete query semantics, while another
was caused by treating typed-slot compatibility as sufficient source
specificity.

Changes:
- Extended EXPLAIN_SCORE to causal non-numeric score/rank questions.
- Added question-level specificity to the verified-backfill completeness test.
- Preserved DEBT_REMOVAL precedence for tax-debt payment questions.

Validation:
- Tests: 51/51 passed.
- Visible benchmark: 19/20 = 95%.
- Mean latency: 48.9 ms.
- P50 latency: 41.4 ms.
- P95 latency: 74.9 ms.

Delta vs Core v1.3:
- Accuracy: 85% -> 95%.
- Mean latency: 44.9 ms -> 48.9 ms.
- P95 latency: 73.0 ms -> 74.9 ms.

Recovered cases:
- v6: non-numeric company score explanation -> doc 86.
- v18: missing highest-score explanation -> doc 67.

Remaining failure:
- v17.

v17 analysis:
- Intent is correctly DEBT_REMOVAL.
- TAX debt and E3 slots are correctly extracted.
- Expected doc 237 is present in verified local retrieval at rank 2.
- Similar remote doc 173 remains ranked above it.
- This is now a reranking/source-specificity problem rather than a recall or
  query-understanding problem.

Decision:
FREEZE Core v1.4 and isolate the remaining issue to P8 reranking specificity.

---

## Core v1.5 Attempt A — Invalid / Aborted Experiment

The first attempt to add an exact FAQ-question specificity feature did not
modify the reranker.

Observed:
- The patch script terminated with:
  `ERROR: score_record lexical block not found`.
- The regression test was added afterwards.
- The new test failed because `question_exact_match` was absent from the
  scorer output.
- Local retrieval remained unchanged:
  - doc 173 rank 1
  - doc 237 rank 2
- The subsequent benchmark still reported 95% accuracy.

Important:
The benchmark produced after the failed patch is NOT a valid Core v1.5
measurement and must not be used to attribute latency or accuracy changes to
the proposed feature.

Engineering lesson:
A patch application success must be verified before running downstream
benchmarks. Regression tests correctly exposed that the intended code path had
not changed.

Decision:
REJECT / INVALID EXPERIMENT.
Retry the same hypothesis using a structurally robust patch.

---

## Core v1.5 Attempt A — Invalid / Aborted Experiment

The first attempt to add an exact FAQ-question specificity feature did not
modify the reranker.

Observed:
- The patch script terminated with:
  `ERROR: score_record lexical block not found`.
- The regression test was added afterwards.
- The new test failed because `question_exact_match` was absent from the
  scorer output.
- Local retrieval remained unchanged:
  - doc 173 rank 1
  - doc 237 rank 2
- The subsequent benchmark still reported 95% accuracy.

Important:
The benchmark produced after the failed patch is NOT a valid Core v1.5
measurement and must not be used to attribute latency or accuracy changes to
the proposed feature.

Engineering lesson:
A patch application success must be verified before running downstream
benchmarks. Regression tests correctly exposed that the intended code path had
not changed.

Decision:
REJECT / INVALID EXPERIMENT.
Retry the same hypothesis using a structurally robust patch.

---

## Core v1.5 — FAQ Question Specificity Anchor

Hypothesis:
Typed semantic compatibility can identify a relevant FAQ family but may still
prefer a less specific source. Human-written FAQ question identity provides a
high-precision source-specificity signal.

Change:
- Added a normalized exact FAQ-question match feature to the semantic reranker.
- Exact match is content-based and independent of benchmark IDs.
- Added a regression test for the feature.

Validation:
- Tests: 52/52 passed.
- Visible benchmark: 20/20 = 100%.
- Mean latency: 63.1 ms.
- P50 latency: 46.1 ms.
- P95 latency: 98.8 ms.

Recovered case:
- Tax-debt removal query:
  - doc 237 moved from local rank 2 to rank 1.
  - exact-question contribution: +3.0.
  - final citation changed from doc 173 to doc 237.

Important diagnostic finding:
Although query and FAQ question were identical, doc 237 received a polarity
score of -3.0 (POLARITY_CONFLICT). The exact-question signal compensated for
this error.

Interpretation:
The remaining polarity representation is unsafe for use as a hard P10 gate.
Document polarity currently mixes FAQ question semantics with answer semantics;
an answer may legitimately negate or resolve the proposition asked by the
question.

Decision:
FREEZE visible-ranking behavior at Core v1.5.
Do not optimize further against the visible benchmark.
Next priority: polarity robustness before hard sufficiency enforcement.

---

## Core v1.6B — Surface-Aware Polarity, Partial Success

Goal:
Separate question polarity from answer polarity and preserve interrogative
semantics before match normalization removes punctuation.

Successful validation:
- Focused polarity tests: 5/5 passed.
- Full regression suite: 57/57 passed.
- Truth table:
  - positive declarative -> POSITIVE
  - negative declarative -> NEGATIVE
  - positive interrogative -> UNKNOWN
  - interrogative without punctuation -> UNKNOWN
  - explicit negation inside question -> NEGATIVE

Remaining regression:
- Visible benchmark: 19/20 = 95%.
- Failing case: v5.
- v5 contains a declarative/conditional premise followed by an interrogative
  effect clause.

Current hypothesis:
Interrogative status is still query-global. This may incorrectly mark premise
propositions as UNKNOWN. Polarity/questionhood likely needs clause-level scope.

Performance:
Repeated benchmark runs showed significant variance.
Latest three p95 values:
- 139.2 ms
- 102.3 ms
- 127.2 ms

Median p95: 127.2 ms.

Decision:
KEEP the question-anchored polarity architecture and surface-aware question
detection, but DO NOT freeze v1.6 yet.
Investigate clause-level polarity scope and the v5 regression first.

---

## Core v1.6C — Candidate Inclusion Diagnosis

v5 diagnosis disproved the earlier hypothesis that clause-level polarity was
the immediate cause of the visible regression.

Observed:
- Gold doc 9 is local verified rank 1.
- Gold local score: 12.0.
- Exact FAQ-question contribution: +3.0.
- No polarity conflict exists for the gold source.
- Despite this, the end-to-end candidate set does not contain doc 9.
- The resolver therefore selects remote doc 299.

Root cause:
The exact-question feature currently exists only in the reranking score.
It has no effect if the trusted exact FAQ document is never admitted into the
candidate set.

Architectural consequence:
High-confidence source anchors must participate in candidate generation, not
only candidate scoring.

Decision:
Add an O(1) normalized FAQ-question index. A trusted exact FAQ match is always
admitted as a candidate. Full local backfill remains conditional for
non-exact/paraphrased queries.

---

## Core v1.6C — Exact Trusted FAQ Candidate Anchor

Hypothesis:
A source-specificity feature in the reranker cannot help when the corresponding
trusted FAQ document never enters the candidate set.

Observed before the change:
- v5 gold doc 9 was verified-local rank 1.
- Gold score: 12.0.
- Exact-question contribution: +3.0.
- No polarity conflict was present.
- Nevertheless, doc 9 was absent from the end-to-end candidate set.
- Remote doc 299 was selected instead.

Change:
- Added an O(1) normalized trusted FAQ-question index.
- Exact trusted FAQ matches are admitted directly into the candidate set.
- Exact matching remains content-based and independent of benchmark IDs.
- Full verified-local backfill remains conditional for non-exact queries.
- Added a regression test for candidate-generation anchoring.

Validation:
- Tests: 58/58 passed.
- v5 now admits doc 9 and cites doc 9.
- Visible benchmark: 20/20 = 100%.
- Single-run mean latency: 43.1 ms.
- Single-run p95 latency: 48.6 ms.

Architectural lesson:
High-specificity retrieval signals must participate in candidate generation,
not only final reranking. A perfect scorer cannot recover a source that was
never admitted into the candidate pool.

Decision:
FREEZE candidate-generation behavior.
Do not further optimize against the visible benchmark.

---

## P3.1 — Intent Focus Routing

Problem:
Debt-related queries containing generic payment words were over-classified as
DEBT_REMOVAL, even when the user's actual information need was the effect of
late payment on credit score/rank.

Change:
- Intent selection now prioritizes explicit effect focus over incidental
  payment/removal vocabulary.
- Strong removal semantics remain DEBT_REMOVAL.
- Late-payment effect questions are routed to DEBT_EFFECT.

Validation:
- Tests: 63/63 passed.
- Intent truth-set:
  - paid tax debt + removal timing -> DEBT_REMOVAL
  - tax settlement outcome -> DEBT_REMOVAL
  - late installment + score effect -> DEBT_EFFECT
  - late then settled + rank effect -> DEBT_EFFECT
  - tax debt + score effect -> DEBT_EFFECT
- Visible benchmark: 20/20 = 100%.
- Mean latency: 60.2 ms.
- P50 latency: 53.4 ms.
- P95 latency: 96.8 ms.

Decision:
FREEZE P3 semantic intent routing.

---

## P10 — Deterministic Evidence Sufficiency Gate

Goal:
Separate ranking from answerability. A highly ranked result is not sufficient
unless evidence is trusted, covers critical semantic requirements, and has no
hard semantic conflict.

Implemented:
- deterministic SufficiencyAssessment;
- critical requirement coverage gate;
- trusted/citation-eligible evidence requirement;
- bank/rank/debt/quantity/polarity conflict gate;
- resolver clarification on insufficient evidence;
- trust-aware evidence selection so an untrusted high-score hit cannot
  suppress a lower-ranked trusted source.

Decision model:
- ambiguity -> CLARIFY
- no evidence -> CLARIFY
- no trusted/citable evidence -> CLARIFY
- critical requirement uncovered -> CLARIFY
- hard semantic conflict -> CLARIFY
- otherwise -> ANSWER

Validation note:
The full pytest invocation reported two CLI failures in bench/judge because the
sealed remote retriever returned `RuntimeError: quota exceeded`.

These are external retrieval-quota failures, not P10 logic failures.
Network-dependent benchmark validation is deferred until retrieval quota is
available.

Offline P10 and core validation remain the release gate while the remote quota
is unavailable.

---

## P12 — Metamorphic and Contrastive Robustness

Goal:
Test semantic invariants and contrastive mutations independently of the
visible benchmark.

Implemented properties:
- Arabic/Persian character normalization invariance;
- Persian/ASCII digit invariance;
- ZWNJ/space normalization invariance;
- bank entity preservation under script variants;
- credit-score preservation under digit variants;
- late-payment paraphrase intent invariance;
- polarity reversal detection;
- rank-swap conflict detection;
- bank-swap conflict detection;
- exact-question anchor invariance under normalized digits;
- deterministic semantic parsing.

Validation:
- P12 targeted tests: 11/11 passed.
- Full offline suite excluding network-dependent CLI tests: 76 passed.

Interpretation:
The current semantic pipeline preserves the tested benign transformations and
responds to meaning-changing contrastive mutations.

Decision:
FREEZE the base P12 metamorphic suite.
Extend robustness next with adversarial provenance and noise-defense tests.

---

## Elective F — Adversarial and Noise Defense

Goal:
Prevent retrieval-score manipulation, poisoned remote payloads, duplicate
injection, and semantic-conflict injection from becoming grounded citations.

Adversarial tests:
- unknown remote document ID;
- forged trusted/citation-eligible metadata;
- keyword-stuffed unknown source;
- valid document ID with tampered remote payload;
- duplicate remote candidates for the same source;
- high-score semantic conflict versus lower-score clean evidence;
- conflicting-only evidence.

Initial finding:
The first adversarial run passed 5/6 tests.

A high-score candidate marked as trusted but carrying a hard semantic conflict
could suppress a lower-score trusted and semantically consistent candidate.

Root cause:
P10.1 evidence ordering preferred:
  provenance -> coverage -> rerank score

but did not prefer semantic consistency between equally trusted candidates.

Change:
Evidence selection now prefers:
  trusted/citation-eligible
  -> semantically consistent
  -> requirement coverage
  -> rerank score

Validation:
- Elective F targeted tests: 6/6 passed.
- Full offline suite excluding network-dependent CLI tests: 82 passed.

Security properties demonstrated:
- unknown remote IDs cannot be cited;
- externally supplied trust flags do not establish corpus trust;
- known IDs are rehydrated from the trusted local corpus;
- remote tampering is not propagated into cited content;
- duplicate candidates do not duplicate source identity;
- semantic-conflict score inflation cannot suppress clean trusted evidence;
- conflicting-only evidence cannot produce an answer.

Decision:
FREEZE adversarial provenance/evidence-selection behavior.

---

## Elective G — Clause-Scoped Negation and Polarity

Goal:
Prevent interrogative and negation scope from leaking across independent
propositions in multi-clause Persian queries.

Initial red-team result:
- 10/12 tests passed.
- Two clause-scope failures were found.

Failure 1:
A positive premise followed by a question:
  "بدهی مالیاتی دارم؛ آیا ... تاثیر دارد؟"
was represented with the debt proposition as UNKNOWN.

Failure 2:
In:
  "بدهی مالیاتی ندارم؛ آیا چک ... تاثیر دارد؟"
the first clause's negation leaked into the check-debt proposition.

Root cause:
- interrogative status was query-global;
- proposition extraction operated on match_text after punctuation had already
  been removed;
- negation was therefore able to cross semantic clause boundaries.

Change:
- clause segmentation now occurs on the original surface before punctuation
  removal;
- each clause is normalized independently;
- interrogative modality is clause-local;
- debt existence polarity is resolved around the specific debt mention;
- effect negation is scoped locally around the effect expression;
- event negation such as "پرداخت نشده" does not imply non-existence of debt.

Additional stress cases:
- comma before an interrogative clause;
- independent debt polarity joined with "و";
- unpaid-event negation combined with a positive effect statement.

Decision:
FREEZE clause-scoped polarity after targeted and full offline validation pass.

### Elective G v2.1 — Target-Aware Predicate Scope

Additional validation after the clause-scoped implementation exposed a subtler
target-scope issue.

Observed:
- In "آیا چک برگشتی روی امتیاز تاثیر دارد؟", the verb "دارد" was incorrectly
  treated as evidence that the check debt itself exists.
- In "بدهی مالیاتی روی امتیاز تاثیر ندارد", "ندارد" was incorrectly attached
  both to debt existence and to the EFFECT proposition.

Root cause:
Existence-polarity extraction used nearby "دارد/ندارد" without distinguishing
entity-existence predicates from effect predicates.

Change:
- effect-expression spans are now identified explicitly;
- existence cues governed by an effect predicate are excluded from debt
  existence polarity;
- interrogative debt topics remain UNKNOWN unless existence is actually
  asserted;
- effect negation remains attached to EFFECT.

This completes clause-scoped and target-scoped polarity handling.
