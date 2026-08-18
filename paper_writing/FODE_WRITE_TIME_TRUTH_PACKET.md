# FODE-D-26-00032 — Write-Time Truth Packet (v2.1 reconciliation pass)

This packet contains ACTUAL VALUES (not paths) for every number needed to write the manuscript.
It supersedes any conflicting number in `FODE_FINAL_WRITING_PACKET.md` (v2.1 pre-reconciliation).

## 1. Locked title

**Who Finds It Hard? Mapping Human–Machine Alignment in Question Difficulty**

## 2. Locked scientific narrative

Study A (EeDi, n=944 same-item) = primary Human–Machine Alignment evidence.
Study B (RACE, n=4,887) = difficulty-provenance / model-record robustness evidence.
Effect in Study A is real but weak (rho≈0.11–0.14); Study B shows exam-band ≠ intrinsic
difficulty and discrete region labels are seed-sensitive; continuous dynamics are primary encoder
evidence.

## 3. Study A exact sample

```text
n_response_rows = 1,377,653
n_unique_students = 4,918
n_unique_questions = 944
n_unique_student_question_pairs = 1,377,653
n_repeated_student_question_pairs = 0
maximum_repeats_for_one_student_question = 1
```

Attempts per question: min=4, Q1=321.0, median=1684.0, Q3=2345.5, P90=2648.1, P95=2829.25,
max=2966, mean=1459.16 (=1,377,653/944), SD not separately stored (median/IQR reported instead,
per policy of not over-claiming normal-distribution summaries on a right-skewed count variable).

Four excluded questions (all excluded for the SAME reason: >1 distinct raw `CorrectAnswer` value
in the source response log, no unambiguous ground-truth key, not resolved by manual/majority
adjudication):

| QuestionId | Distinct CorrectAnswer values (value: frequency) | Response rows excluded |
|---|---|---:|
| 43 | {2: 230, 3: 2} | 232 |
| 84 | {1: 2175, 4: 452} | 2,627 |
| 206 | {3: 1455, 1: 745} | 2,200 |
| 860 | {2: 10, 3: 5} | 15 |

**Confirmed TRUE:** all human difficulty estimates (IRT, EB, empirical) used in the final
same-item analysis were fitted on the 944-item universe with these 4 items already excluded
BEFORE fitting (verified in `same_item_alignment/audit/00_item_mapping.md` and independently in
`v21_c_human_universe.py`: the frozen `same_item_integrated_948.parquet` filtered to 944 rows
matches the IRT/EB/empirical estimate tables row-for-row). → **NOT a stop condition.**

Response-source rule: `train_task_3_4.csv` ONLY, filtered to the 944 QuestionIds. Task 1/2 is not
merged in (different question pool, no recovered content assets for same-item purposes).

Repeated-response sensitivity: N/A in effect — there are zero repeated (student, question) pairs,
so "all observed" and "first observed" are identical by construction (rho=1.0000 for
empirical/EB; rho=0.99999973 for IRT, reflecting floating-point/refit noise, not a real
difference). Item-difficulty rankings and the primary alignment rho (0.138236 all-responses vs.
0.138214 first-response-only) do NOT materially change.

## 4. Study A exact human method

**Model:** 1PL/Rasch. `P(correct_ij=1) = 1 / (1 + exp(-(theta_i - b_j)))`, i=student, j=item.
**Objective:** penalized joint maximum likelihood (student ability theta and item difficulty b
jointly estimated), `L = sum(log P) - 0.5*(theta/sigma_theta)^2` summed over all 1,377,653
observed responses (no aggregation, no repeated-response deduplication needed since there are no
repeats).
**Theta prior/penalty:** Gaussian, sigma_theta=3.0 (weak, regularizing only, not informative).
**Item penalty:** none.
**Identifiability constraint:** post-hoc mean-centering of item difficulties (b) after
convergence.
**Initialization:** all theta and b initialized to 0.
**Optimizer:** Adam, learning rate=0.05.
**Max iterations:** 3,000 (budget); **convergence rule:** loss-plateau, |Δloss| < 1e-7;
**actual convergence:** iteration 481; **total iterations run:** 503 (20-iteration stability
confirmation buffer after the plateau trigger).
**Random seed:** 20260818 (all-responses run) / same seed for first-response-only sensitivity run
(converged at iter 469, total 491 — negligibly different due to fewer effective observations).
**Missing responses:** no missingness (every retained item has a full observed response column;
students who didn't attempt an item simply contribute no term for that item-student pair).
**Software:** PyTorch (custom joint-MLE implementation), device=GPU (CUDA), no library-version
pin recorded beyond what is in `same_item_alignment/scripts/s2_human_difficulty.py`.
**Item uncertainty:** observed-Fisher-information Wald standard errors — **diagnostic only**,
NOT used as the paper's central uncertainty claim (theta uncertainty is ignored by this
approximation, so these SEs understate true uncertainty). Policy: report IRT point estimates,
support with EB/empirical estimator-robustness rather than IRT analytic CIs.

**Beta-Binomial EB prior (944-item subset, method-of-moments):** alpha0 = 4.652854,
beta0 = 4.522257 (shrinkage prior toward global mean correctness ≈ 4.6528/(4.6528+4.5223) = 0.507).

**Estimator agreement (n=944 for all three pairs):**

```text
Spearman(IRT, EB)        = 0.9865   (p < 1e-300, effectively 0)
Spearman(IRT, empirical)  = 0.9928   (p < 1e-300)
Spearman(EB, empirical)   = 0.9932   (p < 1e-300)
```

## 5. Study A exact machine method

Three VLMs, all native `transformers` model classes (no `trust_remote_code`), bfloat16, no
quantization, temperature=0.0, top_p=1.0, max_new_tokens=4, do_sample=False (deterministic
greedy), random-seed field 20260818 (decoding is deterministic so this is a nominal/logging seed,
not a stochastic-sampling seed).

| Solver | HF model ID | Revision | Params | Role |
|---|---|---|---|---|
| Qwen | `Qwen/Qwen2-VL-7B-Instruct` | `main` (exact commit hash NOT RECORDED) | 7B | PRIMARY |
| InternVL | `OpenGVLab/InternVL3-8B-hf` | `main` (exact commit hash NOT RECORDED) | 8B | PRIMARY |
| SmolVLM2 | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `main` (exact commit hash NOT RECORDED) | 2.2B | NEGATIVE CONTROL / failed pilot gate |

Prompt (verbatim, identical for all 3 solvers, no separate system prompt):

```text
You are answering a multiple-choice school mathematics question shown in the image.
The image contains the question text and four answer options.
Look at the image carefully and determine which option is correct.
Respond with ONLY one character: A, B, C, or D. Do not explain your reasoning. Do not output anything else.
```

Hardware: rented RunPod RTX 4090 24GB (local machine was unstable; this was a compute
substitution, not a change of data/methodology — see `same_item_alignment/audit/content_handling.md`).
Parser: single-character letter extraction (A/B/C/D); parse_success = 100% for all three solvers
on the full 944-item run (`solver_quality_gate.csv`), i.e. zero invalid/missing outputs after
parsing — no invalid-output policy branch was actually exercised in the final run.

## 6. Study A primary alignment results

**Primary rank alignment (n=944, 2-solver Qwen+InternVL machine-error, seed=20260818):**

```text
Spearman rho (IRT × machine-error)  = 0.138236
bootstrap 95% CI                     = [0.076617, 0.201134]  (5,000 replicates)
permutation p                        = 0.0002   (5,000 permutations)
parametric p (asymptotic)            = 2.03e-05
```

```text
EB rho          = 0.137710  (CI [0.077059, 0.199755], perm p=0.0002)
empirical rho   = 0.137247  (CI [0.075966, 0.199284], perm p=0.0002)
```

**Primary binomial regression (n_items=944, n_trials=2):**

```text
model: Errors_i ~ Binomial(2, p_i); logit(p_i) = b0 + b1*z(IRT_i)
IRT standardization: z-score (mean 0, SD 1) over the 944-item IRT difficulty distribution
beta1 (z-IRT coefficient)  = 0.259278
analytic SE                = 0.048420
OR per 1 SD IRT            = 1.2960
95% bootstrap CI (item-level, 5,000 reps) = [1.1654, 1.4504]
p                          = 8.57e-08
Pearson overdispersion ratio = 1.403  (mild; handled via item-level bootstrap, not assumed-binomial SE)
random seed                = 20260818
```

**Solver-fixed-effect sensitivity (2-solver pooled item-solver rows, n_rows=1,888, n_item_clusters=944):**

```text
machine_error ~ z(IRT) + solver_fixed_effect
IRT coefficient (beta) = 0.259427; OR = 1.29619
95% CI (item-clustered bootstrap, 5,000 reps) = [1.16963, 1.45503]
p = 8.50e-08
```

**Per-solver (n=944 each):**

| Solver | accuracy | rank-biserial rho | logit OR per 1SD IRT | 95% CI | p |
|---|---:|---:|---:|---|---:|
| Qwen2-VL-7B (PRIMARY) | 39.41% | 0.0941 | 1.2062 | [1.0649, 1.3862] (bootstrap) | 0.0054 |
| InternVL3-8B (PRIMARY) | 41.74% | 0.1376 | 1.3962 | [1.2249, 1.5949] (bootstrap) | 1.8e-06 |
| SmolVLM2-2.2B (**NEGATIVE CONTROL / failed pilot quality gate**) | 25.11% | 0.0089 (n.s., p=0.784) | 1.0487 | [0.8947, 1.2164] (CI includes 1) | 0.523 |

## 7. Study A robustness (3-solver)

```text
n = 944, rho = 0.109994, 95% CI = [0.048693, 0.171866], permutation p = 0.0014
```
(independently recomputed from the frozen integrated table; exactly reproduces the previously
frozen v2 headline number.)

Solver-fixed-effect 3-solver sensitivity: `three_solver_regression_sensitivity.csv` gives a
directionally consistent, slightly attenuated OR (kept as sensitivity only, not headline).

## 8. Study A disagreement results

Extreme-quartile diagnostic, denominator=472 total (236 human_easy + 236 human_hard); the middle
472 items (IRT interquartile range) are explicitly NOT part of this 6-cell partition:

```text
human_easy_machine_easy   = 78   (33.05% of human_easy)
human_easy_machine_mixed  = 76   (32.20% of human_easy)
human_easy_machine_hard   = 82   (34.75% of human_easy)
human_hard_machine_easy   = 45   (19.07% of human_hard)
human_hard_machine_mixed  = 70   (29.66% of human_hard)
human_hard_machine_hard   = 121  (51.27% of human_hard)
```

`67 → 103` explanation: recomputing the OLD 3-solver taxonomy in isolation gives two DIFFERENT
cells — `aligned_easy = 103` (human-easy & machine-easy) and `human_hard_machine_easy = 67`
(human-hard & machine-easy). These were never the same number changing; they are two distinct
cells that were juxtaposed confusingly in an earlier draft summary.

**Categorical agreement (SECONDARY, n=944, tertile human IRT × 3-state machine score, linear-weighted Cohen's kappa):**

3×3 confusion matrix:

```text
                Machine easy  Machine middle  Machine hard
Human easy            97            97            121
Human middle          86            84            144
Human hard            66            87            162
```

```text
observed agreement       = 0.363347
expected agreement       = 0.333386  (chance-level)
weighted kappa (linear)  = 0.0782
95% CI (bootstrap, 5,000 reps) = [0.0284, 0.1271]
n = 944
weighting scheme = linear
```

## 9. Study A selection-bias limits (944 subset vs. remaining 26,669 EeDi questions)

| variable | subset mean/prop | comparison mean/prop | subset SD | comparison SD | SMD |
|---|---:|---:|---:|---:|---:|
| empirical correctness | 0.5668 | 0.6720 | 0.1345 | 0.1608 | -0.710 |
| EB difficulty (shrunk rate) | 0.5675 | 0.6711 | 0.1332 | 0.1532 | -0.722 |
| attempt count | 2,021.95 | 575.27 | 1,179.92 | 655.21 | **+1.516** |
| Human Easy (empirical bucket) | 3.60% (34/944) | 24.52% (6,540/26,669) | — | — | category gap |
| Human Mid (empirical bucket) | 86.02% (812/944) | 69.92% (18,648/26,669) | — | — | category gap |
| Human Hard (empirical bucket) | 10.38% (98/944) | 5.55% (1,481/26,669) | — | — | category gap |

**max |SMD| = 1.516 ≈ 1.52, variable = attempt_count (n_attempts).** Manuscript-safe conclusion:
the 944-item subset is not representative of the full 27,613-item EeDi pool; the alignment effect
must not be transported beyond the 944-item subset.

## 10. Study B RACE dataset

```text
RACE dev total = 4,887; MIDDLE = 1,436; HIGH = 3,451
```
Split counts and raw source hashes unchanged from v1 freeze (`audit/evidence/race_split_counts.json`).

## 11. Study B encoder method

Longformer (`allenai/longformer-base-4096`), 3 seeds (0/1/2). Per-seed exact results:

| seed | overall val accuracy | MIDDLE accuracy | HIGH accuracy | checkpoint sha256 (first 12 chars) |
|---:|---:|---:|---:|---|
| 0 | 74.34% | 79.04% | 72.38% | 5f581e39bfe1 |
| 1 | 74.16% | 78.48% | 72.36% | 3da6c59bb509 |
| 2 | 73.73% | 78.76% | 71.69% | 6a0db2f60fe3 |

```text
mean overall accuracy = 74.07%  (0.740741)
SD                     = 0.32%  (0.003150)
```

BigBird (`google/bigbird-roberta-base`), single run: overall/MIDDLE=72.35%, HIGH=66.21%.

## 12. Study B continuous-dynamics results (PRIMARY encoder evidence)

Aggregation rule: mean/SD/min/max ACROSS the 3 seeds (no single-seed cherry-pick); per-seed
values reported individually below because seed is itself a robustness axis.

| seed | mean_prob MIDDLE | mean_prob HIGH | acc MIDDLE | acc HIGH | point-biserial(HIGH vs mean_prob) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7305 | 0.6619 | 0.7904 | 0.7238 | -0.0937 |
| 1 | 0.7199 | 0.6543 | 0.7848 | 0.7236 | -0.0903 |
| 2 | 0.7196 | 0.6529 | 0.7834 | 0.7181 | -0.0917 |

MIDDLE consistently HIGHER than HIGH on both mean held-out gold probability and accuracy, across
all 3 seeds independently — this is an exam-source grade-band pattern, not evidence that HIGH
questions are intrinsically "harder" for the encoder in a calibrated-difficulty sense.

**LLM-failure ~ continuous dynamics model (per seed, n=4,887 each, outcome = LLM consensus
incorrect, UNCONDITIONAL definition consistent with R2 PRIMARY policy — no-consensus items
retained with `llm_error=1`):**

| seed | predictor | OR | 95% CI | p |
|---:|---|---:|---|---:|
| 0 | z(mean_prob) | 0.2156 | [0.1638, 0.2839] | 7.8e-28 |
| 0 | z(std_prob) | 0.9772 | [0.8383, 1.1390] | 0.768 |
| 0 | last_correct | 2.6857 | [1.5543, 4.6407] | 4.0e-04 |
| 1 | z(mean_prob) | 0.2476 | [0.1891, 0.3244] | 3.9e-24 |
| 1 | z(std_prob) | 0.9189 | [0.7832, 1.0781] | 0.300 |
| 1 | last_correct | 2.2669 | [1.2997, 3.9541] | 3.9e-03 |
| 2 | z(mean_prob) | 0.2452 | [0.1850, 0.3250] | 1.4e-22 |
| 2 | z(std_prob) | 1.0046 | [0.8622, 1.1706] | 0.953 |
| 2 | last_correct | 1.7625 | [1.0189, 3.0490] | 4.3e-02 |

Interpretation: higher encoder mean held-out probability (item easier for the encoder) is
strongly, consistently associated with LOWER odds of LLM failure (OR≈0.22–0.25, all seeds,
p<1e-21) — encoder-side and LLM-side failure are correlated in the SAME direction across all 3
seeds. Variability (std_prob) is not a robust independent predictor.

## 13. Study B categorical region results (canonical 3-seed majority; see item 5 above for source table)

**MIDDLE × region (canonical Longformer 3-seed majority; `outputs/race_final/band_region_crosstab.csv`):**

| | easy | middle | hard | ambiguous | total |
|---|---:|---:|---:|---:|---:|
| MIDDLE | 498 | 359 | 204 | 375 | 1,436 |
| HIGH | 682 | 909 | 637 | 1,223 | 3,451 |
| **TOTAL** | **1,180** | **1,268** | **841** | **1,598** | **4,887** |

```text
band x region (canonical majority): chi2=131.78, df=3, p=2.24e-28, Cramer's V=0.164, n=4887
```

**Conditional LLM-error × region (canonical majority, n=4,873 consensus-only, SENSITIVITY):**

| region | correct | incorrect |
|---|---:|---:|
| ambiguous | 1,524 | 71 |
| easy | 1,175 | 4 |
| hard | 707 | 127 |
| middle | 1,241 | 24 |

```text
chi2 = 282.345, df=3, p=6.58e-61, Cramer's V=0.2407
```

**Unconditional LLM-failure × region (canonical majority, n=4,887, PRIMARY):**

| region | fail | success |
|---|---:|---:|
| ambiguous | 74 | 1,524 |
| easy | 5 | 1,175 |
| hard | 134 | 707 |
| middle | 27 | 1,241 |

```text
chi2 = 290.960, df=3, p=9.00e-63, Cramer's V=0.2440
```

## 14. Study B region instability

```text
3-seed exact agreement          = 46.80%  (2,287/4,887)
seed0-seed1 agreement           = 63.19%
seed0-seed2 agreement           = 62.59%
seed1-seed2 agreement           = 62.64%
all-three-different count        = 253
2-of-3-majority-resolved count    = 2,347
tie-to-middle count               = 253  (=100% of the all-different cases)
majority vs. legacy-region agreement = 65.83%
```

## 15. Study B architecture/threshold robustness

**BigBird vs. Longformer-majority:** switch rate = 53.35% (same-region 46.65%); BigBird
band×region Cramer's V = 0.1084; BigBird LLM-failure×region V = 0.1503 (tercile spec).
BigBird overall accuracy 72.35%(MIDDLE)/66.21%(HIGH), n=4,887, single run.

**Threshold sensitivity (band×region V / LLM-failure×region V), Longformer 3-way percentile-cut
region rule (simplified easy/middle/hard, no ambiguous bucket — distinct from the canonical
4-category cartography rule used above):**

| architecture | seed | threshold spec | band×region V | LLM-failure×region V |
|---|---:|---|---:|---:|
| Longformer | 0 | 20/80 | 0.1767 | 0.2031 |
| Longformer | 0 | 25/75 | 0.1735 | 0.1951 |
| Longformer | 0 | 33/67 | 0.1574 | 0.1828 |
| Longformer | 1 | 20/80 | 0.1659 | 0.1928 |
| Longformer | 1 | 25/75 | 0.1533 | 0.1922 |
| Longformer | 1 | 33/67 | 0.1479 | 0.1782 |
| Longformer | 2 | 20/80 | 0.1597 | 0.2118 |
| Longformer | 2 | 25/75 | 0.1511 | 0.2017 |
| Longformer | 2 | 33/67 | 0.1425 | 0.1821 |
| BigBird | single run | tercile | 0.1084 | 0.1503 |
| BigBird | single run | quartile | 0.1032 | 0.1561 |
| BigBird | single run | p40/60 | 0.0961 | 0.1500 |

Conclusion: individual region LABELS are unstable across seed/architecture/threshold (46.8%
exact 3-seed agreement, 53% BigBird switch rate), but the underlying band×region and
LLM-failure×region ASSOCIATIONS remain positive and in a narrow moderate range (V≈0.10–0.24)
across every specification tested — i.e., the qualitative finding is robust even though the
exact discrete region assigned to any one item is not.

## 16. Study B LLM method

| backend | provider | API model string requested | dated snapshot captured? | access date |
|---|---|---|---|---|
| DeepSeek | deepseek | `deepseek/deepseek-chat` | **NOT immutable — API alias used; immutable backend snapshot not available** | 2026-08-06 |
| GPT | openai_compatible | `gpt-4o` | **NOT immutable — API alias used; immutable backend snapshot not available** | 2026-08-06 |
| Doubao | volcengine_ark | `doubao-seed-2-0-pro-260215` | Dated snapshot string IS embedded in the model ID itself (`260215`); still an API alias, not a locally verifiable hash | 2026-08-15 |

temperature=0.0 (all 3), top_p not separately configurable via these APIs / recorded as default,
max_tokens=4 (letter-only), prompt = shared RACE MCQ prompt (`race_prepared/race_llm_prompts_val.jsonl`
prompt template), parser = single-letter extraction, retry policy = same decoding params reused
(never varies temperature to force a parse).

## 17. Study B LLM exact results

| backend | valid | missing | parse_success | accuracy_parsed |
|---|---:|---:|---:|---:|
| DeepSeek | 4,884 | 3 | 99.94% | 93.78% |
| GPT | 4,887 | 0 | 100.00% | 93.10% |
| Doubao | 4,887 | 0 | 100.00% | 96.28% |

Raw API rows: DeepSeek 4,887, GPT 4,887, Doubao 10,949 (563-recovery-wave inflated; final usable
votes still 4,887 after last-success reconciliation).

**Reconciled agreement states (R1, n=4,887):**

```text
A three_valid_all_same           = 4,561
B three_valid_exactly_two_same   =   309
C three_valid_all_different      =    14
D two_valid_same                 =     3
E two_valid_different            =     0
F fewer_than_two_valid           =     0
consensus (A+B+D)                 = 4,873
no_consensus (C+E+F)               =    14
```

No-consensus breakdown: by grade band HIGH=13, MIDDLE=1; by canonical majority region hard=7,
middle=3, ambiguous=3, easy=1.

## 18. Exact manuscript-safe claims

- "On a genuine same-item subset of 944 EeDi questions, human item difficulty and machine
  (2-solver VLM) error are positively, significantly, but only weakly rank-correlated
  (rho=0.14, 95% CI [0.08, 0.20], p<0.001)."
- "This weak alignment is directionally robust across three independent human-difficulty
  estimators (IRT/EB/empirical, all rho≈0.14) and a 3-solver robustness ensemble (rho=0.11)."
- "The 944-item subset is not representative of the full 27,613-item EeDi pool and the alignment
  effect should not be generalized beyond it (max SMD=1.52 on attempt count)."
- "On the RACE dev split, MIDDLE-band items show higher encoder confidence and higher LLM/encoder
  accuracy than HIGH-band items across all 3 training seeds, reflecting an exam-source band
  effect rather than calibrated item difficulty."
- "Discrete Cartography-style region labels are unstable across seed (46.8% 3-seed exact
  agreement) and architecture (53% BigBird/Longformer switch rate); the qualitative
  band-region and LLM-failure-region associations are more robust than any single item's
  region label (V≈0.10–0.24 across every seed/architecture/threshold specification tested)."

## 19. Exact forbidden claims

- Bridge-RACE / E6 as evidence of any kind (NOT USABLE).
- "3-way full agreement = 4,870" (that number is A+B, not unanimity; unanimity = 4,561).
- "4,870 consensus / 17 no-consensus" (correct: 4,873/14).
- Treating chi2=282.35 or 290.96 as if only one is "correct" (both are correct, different estimands).
- Generalizing the 944-item alignment effect to the full 27,613-item EeDi pool.
- Calling RACE MIDDLE/HIGH "ground-truth" or "calibrated" difficulty.
- Claiming disagreement-triggered review improves learning outcomes (no RCT exists).
- The legacy 3-cluster GEE regression (OR=1.2523) as a headline result.
- Any single-seed Longformer region-count table (1189/1375/1175/1148, or any one seed's own
  counts) presented as "the" RACE region distribution — only the 3-seed majority is canonical.

## 20. Exact abstract-safe numbers

n=944 EeDi same-item questions; rho=0.14 (2-solver primary alignment); n=4,887 RACE dev items;
Longformer mean accuracy 74.1% (3 seeds); LLM-ensemble consensus accuracy 95.4% (4,873/4,887
consensus achieved).

## 21. Exact Results-safe numbers

All values in sections 3–17 above are Results-safe; each carries its exact n/denominator inline.

## 22. Final table plan

Tables A–F as staged in `outputs/revision_candidate_v21/tables/` (unchanged from prior packet);
Table F (RACE association/robustness) should now explicitly carry BOTH the conditional (282.35)
and unconditional (290.96) LLM×region statistics with estimand labels, per R2.

## 23. Final figure plan

Figures 1–4 unchanged from prior packet (`outputs/revision_candidate_v21/figures/`); Figure 2
must add marginal counts per Editor D / R2.6.

## 24. Reviewer evidence map

`paper_writing/reviewer_evidence_matrix.md` — now includes Editor D (`D. PRESENTATION`) as its
own row, cross-referenced with R2.6.
