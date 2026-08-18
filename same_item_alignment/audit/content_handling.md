# Content-handling decision (Gate S10 input)

## Decision

All EeDi question-content image assets (`data/images/{QuestionId}.jpg`, from the
official NeurIPS 2020 Eedi public release, `data/eedi_public_download/data.zip`)
are processed **exclusively with local, open-weight multimodal models run on the
local RTX 5090 GPU**. No question-content image or derived representation is
uploaded to any third-party/commercial API (OpenAI, Anthropic, Google, etc.) in
this extension.

## Rationale

1. No documented data-use review exists that authorizes uploading EeDi's licensed
   NeurIPS-competition content to external commercial inference APIs.
2. The task instructions explicitly require a local-inference default absent such
   a review.
3. Local inference on the 5090 is technically sufficient for 7-8B-class
   multimodal solvers at the required throughput (944 items x 3 solvers).

## What stays local

- All 948/944 question images.
- All prompts constructed from those images.
- All model weights and inference (Hugging Face `transformers`, local GPU).

## What (if anything) leaves the machine

- Nothing question-content-bearing is sent to any third-party model/inference API.
- Addendum (Gate S4 execution): due to repeated local-machine instability
  (remote-control-induced blue-screen restarts), the full 944-item x 3-solver
  inference run was executed on a rented cloud GPU pod (RunPod) that we
  exclusively controlled via SSH for the run's duration, rather than a
  commercial third-party inference/model API. The question-content image
  assets were transferred once (SCP over SSH) to that rented, single-tenant
  GPU instance and used only for local `transformers` inference identical to
  the local-machine plan; no image or prompt was ever sent to any commercial
  AI vendor's API endpoint. Only aggregated, de-identified statistical outputs
  (accuracy rates, Spearman rho, regression coefficients, etc.) and the raw
  per-item prediction CSVs (question id, parsed A/B/C/D letter, correctness
  flag - no image data) were retrieved back to the local repository.

## Solvers selected under this constraint

See `same_item_alignment/audit/03_machine_solver_runs.md` and
`same_item_alignment/configs/solvers.json` for exact checkpoints, revisions, and
inference settings. All three are open-weight checkpoints downloaded once from
Hugging Face Hub and run entirely offline thereafter.
