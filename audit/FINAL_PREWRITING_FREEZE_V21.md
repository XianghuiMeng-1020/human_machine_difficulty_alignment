# FINAL PRE-WRITING SCIENTIFIC FREEZE v2.1

Parent snapshot: `c8553b1cf1e3cf71bbeabbdea30517e574cfee0f` / `fode-r1-science-freeze-v2`
This snapshot: new commit on `revision/fode-same-item-alignment`, tag `fode-r1-science-freeze-v2.1`

## Scope of this freeze

Offline recomputation and writing-information consolidation ONLY. No manuscript edits. No new
GPU inference. No new human data collection. All numbers below are reproducible from frozen
predictions/canonical data already present at the v2 freeze, via the scripts listed.

## What changed vs. v2

1. Primary same-item machine ensemble redefined as 2 validated solvers (Qwen2-VL-7B-Instruct +
   InternVL3-8B-hf); SmolVLM2-2.2B-Instruct explicitly demoted to negative control / 3-solver
   robustness only (Part A/B).
2. 944-item human universe, response-log characteristics, and repeated-response sensitivity
   independently re-verified (Part C) — `same_item_alignment/scripts/v21_c_human_universe.py`.
3. Full Rasch/IRT method disclosed with exact optimizer/convergence/SE-caveat detail; independent
   estimator agreement recomputed (Part D) — `same_item_alignment/scripts/v21_d_rasch_method.py`.
4. Primary alignment, regression, gradient, disagreement taxonomy, categorical agreement, and
   selection bias all independently recomputed on the 2-solver primary definition (Parts E-J) —
   `same_item_alignment/scripts/v21_core_analysis.py`.
5. GEE 3-cluster regression demoted from headline result to audit history; replaced by item-level
   binomial GLM with item bootstrap as primary inferential regression (Part F).
6. VLM inference method and content-equivalence audit fully documented (Parts K/L) —
   `same_item_alignment/scripts/v21_kl_inference_content.py`.
7. RACE writing packet consolidated from frozen v1 outputs only, no RACE model retraining
   (Parts M-O) — `same_item_alignment/scripts/v21_race_final.py`.
8. Bridge-RACE/E6 forbidden-data scan performed and documented (Part P).
9. Manuscript change map, reviewer evidence matrix, full writing packet, LaTeX staging tables,
   number macros, and figure candidates generated (Parts Q-V).

## Reproducibility guarantee

* Original RACE model outputs (Longformer/BigBird checkpoints, LLM raw logs) UNCHANGED.
* No new human data collected or used.
* No new VLM inference performed; all Part A-L numbers derive from the frozen prediction files
  already present at `fode-r1-science-freeze-v2`.
* All new CSV/MD/JSON/tex outputs are regenerable by re-running the `v21_*.py` scripts in
  `same_item_alignment/scripts/`.

## Working-tree exceptions (explicitly documented, not part of this commit)

* `说明.md` — pre-existing local personal note file unrelated to the v2/v2.1 scientific pipeline;
  left untouched and untracked, not added to this commit.
* `outputs/revision_candidate_v21/figures/**` (all `.pdf`/`.png`/`_source.csv`) — matched by the
  pre-existing repo-wide `.gitignore` rule `figures/` (consistent with the v2 freeze convention
  documented in `audit/evidence/gitignored_large_artifacts.md`); figures remain fully
  regenerable via `same_item_alignment/scripts/v21_figures_234.py` and the Part G gradient script.

## Files added in this freeze (see `git show --stat <tag>` for the exact list)

* `outputs/same_item_final/*` — Parts A-J, K-L (EeDi same-item consolidation)
* `outputs/race_final/*` — Parts M-O (RACE writing packet)
* `outputs/revision_candidate_v21/tables/*`, `outputs/revision_candidate_v21/figures/*` — Parts T, V
* `audit/evidence/*_final.md`, `rasch_method_full.md`, `eedi944_human_universe.md`,
  `disagreement_count_trace.md`, `regression_method_final.md`, `selection_bias_interpretation.md`,
  `vlm_inference_method_full.md`, `same_item_content_equivalence.md`, `bigbird_method_full.md`,
  `race_llm_method_full.md`, `forbidden_human_data_scan.md` — audit trail for Parts B-P
* `paper_writing/original_to_revision_map.md`, `reviewer_evidence_matrix.md`,
  `FODE_FINAL_WRITING_PACKET.md`, `generated_numbers.tex` — Parts Q, R, S, U
* `same_item_alignment/scripts/v21_*.py` — all generation scripts for this freeze

## Unresolved ambiguity carried forward (see final structured report Part X.19)

* LLM-incorrect x region Cramer's V: independently recomputed 0.244 (chi2=290.96) vs. frozen
  `g6_stats.json` 0.238 (chi2=282.35) — likely a minor consensus-definition difference; both
  support the same qualitative conclusion (moderate LLM-error/region association) and should be
  reconciled or reported with an explicit note before final submission.
