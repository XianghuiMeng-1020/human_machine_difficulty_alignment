# FODE-D-26-00032 — Final Pre-Writing Fact Pack (v2.1)

Title (LOCKED): **Who Finds It Hard? Mapping Human–Machine Alignment in Question Difficulty**

## S1. Locked scientific narrative

**Study A — EeDi same-item Human–Machine Alignment**
- 944 EeDi Tasks-3/4 questions with recovered content assets, real student response logs, AND
  real VLM-solver predictions on the identical question image.
- Primary machine ensemble = Qwen2-VL-7B-Instruct + InternVL3-8B-hf (both pass the pre-registered
  pilot quality gate: clearly above chance, no answer collapse).
- SmolVLM2-2.2B-Instruct fails the same pilot gate (near-chance, answer-collapse) -> excluded from
  primary, reported as negative control, retained in a 3-solver robustness ensemble.
- Primary finding: human IRT item difficulty and 2-solver machine error rate are POSITIVELY,
  SIGNIFICANTLY, but only WEAKLY rank-correlated (rho=0.1382, 95% CI [0.077,0.201], n=944).
- Robust in direction across human-difficulty estimators (IRT/EB/empirical) and across the
  3-solver robustness ensemble (rho=0.1100).
- The 944-item subset is NOT representative of the full 27,613-item EeDi pool (max SMD=1.52 on
  attempt count) -> effect must not be generalized beyond this subset.

**Study B — RACE difficulty-provenance / model-record robustness**
- Official RACE dev split (n=4,887; MIDDLE=1,436; HIGH=3,451), unchanged from v1 freeze.
- Longformer encoder, 3 seeds, mean accuracy 74.07% (SD 0.32%); BigBird architecture check.
- 3-backend frozen LLM ensemble (DeepSeek/GPT-4o/Doubao), consensus 4,873/4,887 (corrected from an
  earlier 4,870/17 draft figure).
- Discrete Cartography-style regions are seed-sensitive (46.8% 3-seed exact agreement) and must be
  SECONDARY to continuous held-out confidence/generalization dynamics.
- Bridge-RACE and E6 are NOT USABLE (undocumented ethics/provenance) and must not appear anywhere.

## S2. Exact RQs supported by the final evidence (title unchanged)

- RQ1 (unchanged, RACE): How does the exam-source grade-band tag relate to encoder-based
  data-map region labels on the official RACE validation split?
- RQ2 (unchanged, RACE): How do LLM-ensemble answer errors relate to the grade-band tag and
  data-map regions on the same RACE questions?
- RQ3 (NEW — replaces the Bridge-RACE-based alignment RQ): On a genuine same-item EeDi subset
  (n=944), how strongly does human (student) item difficulty align with machine (VLM) item
  error, and does this alignment depend on which validated solver(s) or human-difficulty
  estimator is used?
- RQ4 (secondary/framing, EeDi full pool): How does a large EeDi student-outcome record
  characterize learner-facing question difficulty as a cross-corpus reference (unchanged from v1,
  n=27,613)?

## S3. Dataset facts

- EeDi same-item: 948 content-available -> 944 retained (4 excluded: QuestionId 43, 84, 206, 860,
  each with >1 distinct raw `CorrectAnswer` value, no majority-vote resolution performed).
- EeDi same-item response log: 1,377,653 rows, 4,918 unique students, 944 unique questions,
  0 repeated (student, question) pairs (every student attempted each retained item at most once).
- EeDi full pool (cross-corpus reference, unchanged v1): 27,613 questions (Easy 6,574 / Mid
  19,460 / Hard 1,579).
- RACE: 4,887 dev questions (1,436 MIDDLE / 3,451 HIGH), unchanged from v1 freeze.
- Response-source rule (same-item): `train_task_3_4.csv` ONLY, filtered to the 944 QuestionIds;
  Task 1/2 responses are not merged in (no recovered content assets for that pool).

## S4. Human difficulty method (EeDi same-item)

1PL/Rasch, penalized joint-MLE, Adam optimizer (lr=0.05), initialized at 0, weak Gaussian penalty
on student ability only (sigma_theta=3.0), post-hoc mean-centering for identifiability, converged
at iteration 481 (20-iteration stability buffer, ~501-503 total), loss-plateau tolerance 1e-7,
observed-Fisher-information item SEs (Wald-style, ignores theta uncertainty — point estimates
recommended as primary, not the analytic CIs). EB (Beta-Binomial, moment-matched prior:
alpha0=4.653, beta0=4.522) and raw empirical correctness are robustness estimators. All three
estimators agree very strongly on the clean 944-item universe (Spearman IRT-EB=0.9865,
IRT-empirical=0.9928, EB-empirical=0.9932). Full detail: `audit/evidence/rasch_method_full.md`.

## S5. Same-item machine method (EeDi VLM solvers)

- Qwen2-VL-7B-Instruct (`Qwen/Qwen2-VL-7B-Instruct`), InternVL3-8B-hf
  (`OpenGVLab/InternVL3-8B-hf`), SmolVLM2-2.2B-Instruct (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`).
- All native `transformers` classes (no `trust_remote_code`), bfloat16, temperature=0.0, top_p=1.0,
  max_new_tokens=4, deterministic greedy decoding, single shared prompt
  (`same_item_alignment/configs/prompts/mcq_prompt.txt`).
- Executed on a rented RunPod RTX 4090 (transformers 5.15.0, torch 2.6.0+cu124, CUDA 12.4) due to
  local-machine instability; no third-party inference API used.
- Full exact counts: `outputs/same_item_final/vlm_exact_counts.csv`; full method:
  `audit/evidence/vlm_inference_method_full.md`.

## S6. Same-item alignment results (all primary and robustness numbers)

| Statistic | Value | 95% CI | p | n |
|---|---|---|---|---|
| Primary rho (IRT x 2-solver error) | 0.1382 | [0.0766, 0.2011] | perm 0.0002 | 944 |
| Robustness rho (IRT x 3-solver error) | 0.1100 | [0.0487, 0.1719] | perm 0.0014 | 944 |
| EB x primary error | 0.1372 (see `alignment_by_human_estimator.csv`) | -- | -- | 944 |
| Empirical x primary error | 0.1381 | -- | -- | 944 |
| Primary binomial OR (per 1 SD IRT) | 1.296 | [1.165, 1.450] (item bootstrap) | 8.6e-8 | 944 |
| Gradient trend (decile rank vs. 2-solver accuracy) | rho=-0.818 | -- | 0.0038 | 10 bins |
| Weighted kappa (secondary, tertile) | 0.078 | [0.028, 0.127] | -- | 944 |

Per-solver: Qwen rho=0.0941 (p=0.004), InternVL rho=0.1376 (p<0.001), SmolVLM2 rho=0.0089
(n.s., p=0.78 — do not interpret as evidence against alignment; solver is near-chance).

## S7. RACE encoder method

Longformer (`allenai/longformer-base-4096`), 3 seeds (0/1/2), max_len=1024, 400-word passage
truncation, AdamW lr=2e-5, 4 epochs, effective batch 16. Mean val accuracy 74.07% (SD 0.32%,
range [73.73%, 74.34%]). BigBird (`google/bigbird-roberta-base`) architecture check: max_len=512,
200-word truncation, single seed, MIDDLE 72.35% / HIGH 66.21% (last epoch).

## S8. RACE LLM method

DeepSeek (`deepseek/deepseek-chat`), GPT (`gpt-4o`), Doubao (`doubao-seed-2-0-pro-260215`).
temperature=0.0, top_p=1.0, max_tokens=4, retries reuse identical decoding params (never vary
temperature to recover a parse). Coverage: DeepSeek 4,884/4,887 valid parses; GPT 4,887/4,887;
Doubao 4,887/4,887 (from 10,949 raw logged rows, including a 563-item recovery wave and 1,528
accidental duplicate-success rows). Full detail: `audit/evidence/race_llm_method_full.md`.

## S9. RACE statistical results (exact final numbers)

- Consensus: 4,873/4,887 (14 no-consensus: 13 HIGH, 1 MIDDLE); **CORRECTED (R1 reconciliation,
  see `audit/evidence/race_llm_agreement_state_reconciliation.md`): strict three-way unanimous
  agreement = 4,561 (all 3 backends valid AND identical). The figure "4,870" is NOT unanimous
  agreement — it equals 4,561 + 309 = items where the three-valid subset reached ANY 2-of-3-or-better
  majority, a superset of unanimity. "4,870" must never again be called "three-way full agreement."**
- Conditional (consensus-only) accuracy: 95.36%; unconditional: 95.09%.
- Band x region: chi2=131.78, p=2.24e-28, Cramer's V=0.164.
- LLM x region: **RESOLVED (R2 reconciliation,** `audit/evidence/llm_region_estimand_reconciliation.md`**)** —
  chi2=290.96, V=0.244 is the UNCONDITIONAL failure analysis (n=4887, no-consensus counted as
  failure; adopted as PRIMARY); chi2=282.35, V=0.241 is the CONDITIONAL consensus-error analysis
  (n=4873, consensus-only; retained as SENSITIVITY). Both use the identical canonical
  3-seed-majority region definition; the two chi2 values differ only by estimand, not by error.
- Grade-band inversion: MIDDLE > HIGH holds for BOTH encoder (0.786 vs 0.722, 3-seed mean) and
  LLM (consensus-conditioned) across all 3 seeds — exam-source band, not intrinsic difficulty.

## S10. Robustness results

- Seeds: Longformer 3-seed mean/SD above; 3-seed exact region agreement 46.8%; pairwise ~63%.
- Architecture: BigBird vs. Longformer region switch rate 53.3%.
- Threshold: p20/80, quartile 25/75, tercile 33/67 all tested for both architectures
  (`outputs/race_final/threshold_sensitivity_full.csv`); band/LLM-region associations are stable
  in sign and magnitude (V in a narrow ~0.10-0.24 band) even though individual item region labels
  are unstable.
- Human estimators (EeDi): IRT/EB/empirical give nearly identical primary alignment rho
  (0.1382/0.1372/0.1381).
- Solver competence: SmolVLM2 fails the pilot gate on both accuracy and answer-collapse criteria;
  excluding it does not remove the primary alignment signal (it is present, and slightly
  stronger, without it).

## S11. Limitations that MUST appear

- The same-item alignment effect is weak/partial (rho ~0.11-0.14), not strong.
- The 944-item subset is confirmed NOT representative of the full 27,613-item EeDi pool.
- Same-item results are scoped to the content-available EeDi subset only.
- SmolVLM2 is near chance with answer collapse and is excluded from the primary ensemble.
- Local/rented-GPU VLMs may have unknown benchmark exposure to EeDi content.
- RACE MIDDLE/HIGH are exam-source educational levels, not calibrated item difficulty.
- Held-out dynamics are generalization dynamics, not original Dataset Cartography training
  dynamics.
- Discrete region labels are seed- and architecture-sensitive (46.8% 3-seed exact agreement).
- MCQ-only scope throughout (no open-ended/constructed-response items).
- No controlled (RCT/deployment) evidence that acting on disagreement improves learning or
  recommendation quality — any such claim must be narrowed to "offline association only."
- No Bridge-RACE or E6 evidence anywhere in this manuscript (NOT USABLE, undocumented ethics).

## S12. Forbidden claims

- Do not call the same-item alignment "strong," "substantial," or "confirms machines find the
  same questions hard as students."
- Do not cite Bridge-RACE (320x30/9,600 responses), E6 (kappa=0.538, 66.7%/30.0% flaw rates), the
  machine/LLM-simulated 160-person IRT panel, or the Paper-1 LLM-vs-human item-generation
  comparison anywhere in the manuscript, tables, figures, or supplement.
- Do not claim the 944-item same-item alignment effect generalizes to the full 27,613-item EeDi
  pool or to RACE.
- Do not claim disagreement-triggered review improves student learning outcomes (no RCT exists).
- Do not describe RACE LLM retries as varying temperature 0.1-1.5 (contradicts the frozen,
  audited protocol, which forbids this).
- Do not report the legacy 3-cluster GEE regression (OR=1.2523) as a primary/headline result.
- Do not report "4,870 consensus / 17 no-consensus" (superseded by 4,873/14) or call 4,870
  "three-way full agreement" (that is 4,561; 4,870 = a different, broader quantity — see R1
  reconciliation) or "aligned_easy: 67 -> 103" (a mislabeling of two different taxonomy cells,
  not a real number change).

## S13. Exact numbers safe for abstract

n=944 same-item EeDi questions; primary rho=0.14 (2-solver ensemble); RACE n=4,887; Longformer
74.1% (mean of 3 seeds); LLM ensemble consensus accuracy 95.4% (4,873/4,887 consensus).

## S14. Exact numbers safe for Results (complete table)

See `outputs/same_item_final/*.csv` and `outputs/race_final/*.csv` in full; headline subset
duplicated in S6/S9 above and in `paper_writing/generated_numbers.tex`.

## S15. Candidate tables (final manuscript decision deferred to author)

- Table A: same-item sample and learner estimates — `outputs/revision_candidate_v21/tables/table_A_same_item_sample.*`
- Table B: same-item machine solver performance — `table_B_solver_performance.*`
- Table C: human-machine alignment statistics — `table_C_alignment_statistics.*`
- Table D: disagreement / categorical agreement — `table_D_disagreement_agreement.*`
- Table E: RACE encoder and LLM performance — `table_E_race_encoder_llm_performance.*`
- Table F: RACE association and robustness summary — `table_F_race_association_robustness.*`

## S16. Candidate figures

- Figure 1 (same-item alignment gradient): `outputs/revision_candidate_v21/figures/eedi_alignment_gradient.{pdf,png}`, source `eedi_alignment_gradient_source.csv`.
- Figure 2 (same-item disagreement map): `eedi_disagreement_map.{pdf,png}`, source `eedi_disagreement_source.csv`.
- Figure 3 (RACE continuous held-out dynamics): `race_continuous_dynamics.{pdf,png}`, source `race_continuous_dynamics_source.csv`.
- Figure 4 (robustness: seed/architecture/threshold): `race_robustness_panel.{pdf,png}`, source `race_robustness_source.csv`.
