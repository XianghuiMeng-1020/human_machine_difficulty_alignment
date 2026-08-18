# Full 944-item machine solver runs (Gate S4)

## Execution environment

Because the local Windows workstation was experiencing repeated remote-control-induced
blue-screen crashes/restarts, the full 944-item x 3-solver inference run (the long,
uninterruptible workload) was executed on a rented cloud GPU pod rather than the local
RTX 5090, to guarantee the run could complete without data loss. This is a pure compute
substitution: same open-weight checkpoints, same prompt, same deterministic decoding
settings, same manifest/images transferred byte-for-byte (SHA-256 hashes match the
locally-verified `eedi948_item_manifest.parquet`). No third-party API was used; this
remains local/self-hosted inference on a single rented GPU instance under our control
(consistent with `content_handling.md`).

- Provider: RunPod (on-demand GPU pod)
- GPU: 1x NVIDIA GeForce RTX 4090, 24564 MiB VRAM, driver 550.127.05
- CPU/RAM: 128 vCPU, 503 GiB RAM
- OS: Linux container, Python 3.11.10
- torch: 2.6.0+cu124 (upgraded from the image default 2.4.1+cu124 because transformers
  5.15.0 requires torch>=2.5; torchaudio was uninstalled after the upgrade broke its ABI,
  since it is not used for anything in this vision-only pipeline)
- transformers: 5.15.0
- Data transferred: `eedi948_item_manifest.parquet`/`.csv`, `configs/solvers.json`,
  `configs/prompts/mcq_prompt.txt`, `s3_run_solver.py`, and a 47.2 MB zip of the 944
  retained question-image assets (extracted to an identical relative directory layout
  so `content_asset_path` resolution in the manifest required no code changes)
- Access: SSH (key-based, ed25519) over the pod's exposed TCP port; no other network
  service was used to move question content

## Run integrity note (documented, not hidden)

During setup, two premature/duplicate launch attempts (before the working launch script
was finalized) produced a corrupted, header-less `solver1_full.csv`
(a rm-vs-append race between two accidentally-concurrent invocations). This was detected
by the post-hoc integrity check (`pod_check_csv.py`, verifying `question_id` column
presence and exact 944-row/944-unique-id counts) before any downstream analysis was run.
The corrupted file was deleted and solver 1 was re-run from a clean, single-process
launch. All three final `*_full.csv` files were verified post-hoc to contain exactly
944 rows, 944 unique `question_id` values, and a valid header, with **zero** stray or
duplicate rows, before being used in `s7_build_integrated.py`.

## Per-solver full-run results (944/944 items each, 0 missing)

| solver_id | checkpoint | n | parse_success_rate | accuracy_on_parsed | answer_distribution (option:count) |
|---|---|---|---|---|---|
| solver_1_qwen2vl7b | Qwen/Qwen2-VL-7B-Instruct | 944 | 1.000 | 0.3941 | see raw_predictions/solver1_full.csv |
| solver_2_internvl3_8b | OpenGVLab/InternVL3-8B-hf | 944 | 1.000 | 0.4174 | 3:351, 2:295, 4:205, 1:93 |
| solver_3_smolvlm2_2b | HuggingFaceTB/SmolVLM2-2.2B-Instruct | 944 | 1.000 | 0.2511 | 2:655, 4:219, 3:70 |

All three solvers achieved 100% parse success (deterministic 1-4 token decoding, no
missing/unparseable outputs) across the full 944-item run. Solver 3
(SmolVLM2-2.2B) reproduces the answer-skew pattern flagged in the 50-item pilot
(`02_pilot50.md`): 69.4% of its full-run predictions are option "2", and its
accuracy (25.1%) is close to the 25% random-chance baseline for 4-option MCQs. This
is carried forward transparently into the ensemble/robustness analyses (Sections 13-14)
rather than excluded post hoc: the leave-one-solver-out and per-solver Spearman results
(`06_robustness.md`, `alignment_per_solver.csv`) show the primary alignment finding is
**not** solely dependent on solver 3, and is in fact *weaker* when only solver 3 is
considered in isolation (rho=0.0089, n.s.), while it strengthens when solver 3 is the
one left out (rho=0.1382 vs. the full-ensemble 0.1100).

## Ensemble machine difficulty (see s7_status.json / same_item_integrated_948.parquet)

- n items integrated = 944, n solvers = 3, n items with any missing solver output = 0
- mean ensemble machine_error_rate = 0.6458

## Gate S4 verdict: **PASS**

Full same-item machine inference completed for all 944 retained items across all 3
frozen solvers with auditable per-item, per-solver outputs saved
(`same_item_alignment/data/raw_predictions/solver{1,2,3}_full.csv` ->
`machine_predictions_948.parquet`), zero missing outputs, 100% parse success, and a
documented/corrected data-integrity incident during setup.
