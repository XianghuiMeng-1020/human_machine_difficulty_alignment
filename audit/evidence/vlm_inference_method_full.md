# VLM Inference — Full Exact Method (Part K, v2.1)

This documents exact configuration for the final 944-item clean universe. No new inference was run to produce this document; all fields are read from the frozen `configs/solvers.json`, `configs/prompts/mcq_prompt.txt`, and per-solver run audits.

## solver_1_qwen2vl7b

- Hugging Face model ID: `Qwen/Qwen2-VL-7B-Instruct`
- Revision/commit: `main` (tag-level pin, not a pinned commit SHA -- exact per-file HF commit hash was not separately recorded at run time; this is a disclosed gap, not a fabricated hash)
- Parameter count: 7B (dense, vision+language)
- Quantization: none (bfloat16)
- dtype: bfloat16
- Inference library: transformers (Qwen2VLForConditionalGeneration) + qwen-vl-utils
- transformers version: 5.15.0 (RunPod full-run environment; upgraded from 5.9.0 used in local pilot smoke-tests, which lacked native support for InternVL3-hf / SmolVLM2 classes)
- torch version: 2.6.0+cu124 (upgraded on the pod from image default 2.4.1+cu124)
- CUDA version: 12.4 (driver 550.127.05)
- Image preprocessing / resize rule: `AutoProcessor.from_pretrained(checkpoint)` with no custom resize/crop pipeline. Exact processor `size` / `max_pixels` / normalization constants were **NOT RECORDED** at execution time (processor defaults used, not logged).
- Maximum image size: **NOT RECORDED** (not manually capped; default max_pixels not logged)
- Prompt text: see `configs/prompts/mcq_prompt.txt` (identical across all 3 solvers)
- System prompt: none separate from the single user-turn prompt below
- temperature: 0.0
- top_p: 1.0
- max_new_tokens: 4
- do_sample: False (temperature=0.0 deterministic greedy decoding)
- random seed: 20260818
- batch size: 1 (sequential per-item requests; see `runtime_ms` column in `raw_predictions/solver1_full.csv` for per-item latency)

## solver_2_internvl3_8b

- Hugging Face model ID: `OpenGVLab/InternVL3-8B-hf`
- Revision/commit: `main` (tag-level pin, not a pinned commit SHA -- exact per-file HF commit hash was not separately recorded at run time; this is a disclosed gap, not a fabricated hash)
- Parameter count: 8B (dense, vision+language)
- Quantization: none (bfloat16)
- dtype: bfloat16
- Inference library: transformers (InternVLForConditionalGeneration, native class, no trust_remote_code)
- transformers version: 5.15.0 (RunPod full-run environment; upgraded from 5.9.0 used in local pilot smoke-tests, which lacked native support for InternVL3-hf / SmolVLM2 classes)
- torch version: 2.6.0+cu124 (upgraded on the pod from image default 2.4.1+cu124)
- CUDA version: 12.4 (driver 550.127.05)
- Image preprocessing / resize rule: `AutoProcessor.from_pretrained(checkpoint)` with no custom resize/crop pipeline. Exact processor `size` / `max_pixels` / normalization constants were **NOT RECORDED** at execution time (processor defaults used, not logged).
- Maximum image size: **NOT RECORDED** (not manually capped; default max_pixels not logged)
- Prompt text: see `configs/prompts/mcq_prompt.txt` (identical across all 3 solvers)
- System prompt: none separate from the single user-turn prompt below
- temperature: 0.0
- top_p: 1.0
- max_new_tokens: 4
- do_sample: False (temperature=0.0 deterministic greedy decoding)
- random seed: 20260818
- batch size: 1 (sequential per-item requests; see `runtime_ms` column in `raw_predictions/solver2_full.csv` for per-item latency)
- Model-swap history: Original selection OpenGVLab/InternVL2_5-4B used trust_remote_code custom modeling code incompatible with the installed transformers 5.9.0 internal tied-weights API (AttributeError: all_tied_weights_keys). The executed frozen full-run solver is OpenGVLab/InternVL3-8B-hf (native transformers class InternVLForConditionalGeneration, no trust_remote_code). InternVL3-2B-hf was only a discarded intermediate pilot candidate, not the 944-item executed model.

## solver_3_smolvlm2_2b

- Hugging Face model ID: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Revision/commit: `main` (tag-level pin, not a pinned commit SHA -- exact per-file HF commit hash was not separately recorded at run time; this is a disclosed gap, not a fabricated hash)
- Parameter count: 2.2B (dense, vision+language)
- Quantization: none (bfloat16)
- dtype: bfloat16
- Inference library: transformers (SmolVLMForConditionalGeneration, native class, no trust_remote_code)
- transformers version: 5.15.0 (RunPod full-run environment; upgraded from 5.9.0 used in local pilot smoke-tests, which lacked native support for InternVL3-hf / SmolVLM2 classes)
- torch version: 2.6.0+cu124 (upgraded on the pod from image default 2.4.1+cu124)
- CUDA version: 12.4 (driver 550.127.05)
- Image preprocessing / resize rule: `AutoProcessor.from_pretrained(checkpoint)` with no custom resize/crop pipeline. Exact processor `size` / `max_pixels` / normalization constants were **NOT RECORDED** at execution time (processor defaults used, not logged).
- Maximum image size: **NOT RECORDED** (not manually capped; default max_pixels not logged)
- Prompt text: see `configs/prompts/mcq_prompt.txt` (identical across all 3 solvers)
- System prompt: none separate from the single user-turn prompt below
- temperature: 0.0
- top_p: 1.0
- max_new_tokens: 4
- do_sample: False (temperature=0.0 deterministic greedy decoding)
- random seed: 20260818
- batch size: 1 (sequential per-item requests; see `runtime_ms` column in `raw_predictions/solver3_full.csv` for per-item latency)
- Model-swap history: Original selection openbmb/MiniCPM-V-2_6 used trust_remote_code custom modeling code with the same tied-weights incompatibility risk as InternVL2_5-4B under transformers 5.9.0. Replacement 1, native openbmb/MiniCPM-V-4.6, turned out to be a much smaller efficient/linear-attention checkpoint that performed at chance (24% on the 50-item pilot). Replacement 2, llava-hf/llava-v1.6-mistral-7b-hf (LLaVA-NeXT/Mistral-7B), scored below chance (20%) and collapsed onto a single answer option (41/50 = 82% predicted 'A') -- a documented pilot failure mode (Sec. 5 criteria), not silently accepted. Replaced with HuggingFaceTB/SmolVLM2-2.2B-Instruct (native class, appropriately-sized, independently-trained/maintained solver family) for the retained solver 3.

## Shared prompt text (identical for all 3 solvers)

```
You are answering a multiple-choice school mathematics question shown in the image.
The image contains the question text and four answer options.
Look at the image carefully and determine which option is correct.
Respond with ONLY one character: A, B, C, or D. Do not explain your reasoning. Do not output anything else.
```

## Parser rule

Raw model text output is matched against the single-character options A/B/C/D (case-insensitive, leading/trailing whitespace stripped); the first matching letter is mapped to option index 1-4. `parse_success=True` iff exactly one of A/B/C/D is identifiable in the raw output.

## Invalid-output handling

All three solvers achieved 100% parse success on both the 50-item pilot and the full 944-item run (`03_machine_solver_runs.md`); the invalid-parse path (drop from analysis denominator, do not count as correct or incorrect) is implemented but was never triggered in the frozen runs.

## GPU / environment

RunPod on-demand pod, 1x NVIDIA GeForce RTX 4090 (24564 MiB VRAM), driver 550.127.05, 128 vCPU / 503 GiB RAM host, Linux container, Python 3.11.10. SSH (ed25519 key-based) access only; no third-party inference API used (see `content_handling.md`).

## Exact inference counts (final 944-item clean universe)

| solver_id             | hf_model_id                          | revision   |   questions_submitted |   successful_parses |   invalid_parses |   missing_outputs |   correct_count |   incorrect_count |   accuracy |
|:----------------------|:-------------------------------------|:-----------|----------------------:|--------------------:|-----------------:|------------------:|----------------:|------------------:|-----------:|
| solver_1_qwen2vl7b    | Qwen/Qwen2-VL-7B-Instruct            | main       |                   944 |                 944 |                0 |                 0 |             372 |               572 |   0.394068 |
| solver_2_internvl3_8b | OpenGVLab/InternVL3-8B-hf            | main       |                   944 |                 944 |                0 |                 0 |             394 |               550 |   0.417373 |
| solver_3_smolvlm2_2b  | HuggingFaceTB/SmolVLM2-2.2B-Instruct | main       |                   944 |                 944 |                0 |                 0 |             237 |               707 |   0.251059 |

## Corrupted-file incident (reproducibility record only -- do not put in manuscript)

During RunPod setup, two premature/duplicate launch attempts produced a corrupted, header-less `solver1_full.csv` (a race between two accidentally-concurrent invocations). Caught by post-hoc integrity check (`pod_check_csv.py`: verifies `question_id` column presence and exact 944-row/944-unique-id counts) before any analysis; file deleted and solver 1 was re-run cleanly. This is a documented compute-environment/setup-process incident, unrelated to any local-PC hardware failure, and belongs only in the audit/reproducibility record, never in the manuscript Methods/Results.

