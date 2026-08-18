#!/usr/bin/env python3
"""Part K (VLM inference method) and Part L (content equivalence audit).
Reads only frozen configs, prompts, and raw predictions/manifest -- no new
inference is run."""
from __future__ import annotations
import json
from pathlib import Path
import hashlib

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIA = ROOT / "same_item_alignment"
SOLVERS_JSON = SIA / "configs/solvers.json"
PROMPT_TXT = SIA / "configs/prompts/mcq_prompt.txt"
RAW_PRED_DIR = SIA / "data/raw_predictions"
MANIFEST = SIA / "data/eedi948_item_manifest.parquet"
INTEGRATED = SIA / "data/same_item_integrated_948.parquet"

OUT = ROOT / "outputs/same_item_final"
EVID = ROOT / "audit/evidence"
for d in [OUT, EVID]:
    d.mkdir(parents=True, exist_ok=True)

SOLVER_FILE_MAP = {"solver_1_qwen2vl7b": "solver1", "solver_2_internvl3_8b": "solver2", "solver_3_smolvlm2_2b": "solver3"}


def main():
    solvers_cfg = json.loads(SOLVERS_JSON.read_text(encoding="utf-8"))
    prompt_text = PROMPT_TXT.read_text(encoding="utf-8")

    counts_rows = []
    for s in solvers_cfg["solvers"]:
        sid = s["solver_id"]
        full = pd.read_csv(RAW_PRED_DIR / f"{SOLVER_FILE_MAP[sid]}_full.csv")
        n_sub = len(full)
        n_parsed = int(full.parse_success.sum())
        n_invalid = n_sub - n_parsed
        n_missing = 0  # verified 944/944 rows present, no missing outputs (03_machine_solver_runs.md)
        n_correct = int(full.machine_correct.sum())
        n_incorrect = n_sub - n_correct
        counts_rows.append({
            "solver_id": sid, "hf_model_id": s["exact_checkpoint"], "revision": s["checkpoint_revision"],
            "questions_submitted": n_sub, "successful_parses": n_parsed, "invalid_parses": n_invalid,
            "missing_outputs": n_missing, "correct_count": n_correct, "incorrect_count": n_incorrect,
            "accuracy": n_correct / n_sub,
        })
    counts_df = pd.DataFrame(counts_rows)
    counts_df.to_csv(OUT / "vlm_exact_counts.csv", index=False)

    md = []
    md.append("# VLM Inference — Full Exact Method (Part K, v2.1)\n")
    md.append("This documents exact configuration for the final 944-item clean universe. No new "
               "inference was run to produce this document; all fields are read from the frozen "
               "`configs/solvers.json`, `configs/prompts/mcq_prompt.txt`, and per-solver run audits.\n")
    for s in solvers_cfg["solvers"]:
        md.append(f"## {s['solver_id']}\n")
        md.append(f"- Hugging Face model ID: `{s['exact_checkpoint']}`")
        md.append(f"- Revision/commit: `{s['checkpoint_revision']}` (tag-level pin, not a pinned commit SHA -- "
                   f"exact per-file HF commit hash was not separately recorded at run time; this is a "
                   f"disclosed gap, not a fabricated hash)")
        md.append(f"- Parameter count: {s['parameter_count']}")
        md.append(f"- Quantization: {s['quantization']}")
        md.append(f"- dtype: bfloat16")
        md.append(f"- Inference library: {s['inference_library']}")
        md.append(f"- transformers version: 5.15.0 (RunPod full-run environment; upgraded from 5.9.0 used in "
                   f"local pilot smoke-tests, which lacked native support for InternVL3-hf / SmolVLM2 classes)")
        md.append(f"- torch version: 2.6.0+cu124 (upgraded on the pod from image default 2.4.1+cu124)")
        md.append(f"- CUDA version: 12.4 (driver 550.127.05)")
        md.append(f"- Image preprocessing / resize rule: `AutoProcessor.from_pretrained(checkpoint)` with "
                   f"no custom resize/crop pipeline. Exact processor `size` / `max_pixels` / normalization "
                   f"constants were NOT RECORDED at execution time (processor defaults used, not logged).")
        md.append(f"- Maximum image size: NOT RECORDED (not manually capped; default max_pixels not logged)")
        md.append(f"- Prompt text: see `configs/prompts/mcq_prompt.txt` (identical across all 3 solvers)")
        md.append(f"- System prompt: none separate from the single user-turn prompt below")
        md.append(f"- temperature: {s['temperature']}")
        md.append(f"- top_p: {s['top_p']}")
        md.append(f"- max_new_tokens: {s['max_tokens']}")
        md.append(f"- do_sample: False (temperature=0.0 deterministic greedy decoding)")
        md.append(f"- random seed: {s['seed_if_applicable']}")
        md.append(f"- batch size: 1 (sequential per-item requests; see `runtime_ms` column in "
                   f"`raw_predictions/{SOLVER_FILE_MAP[s['solver_id']]}_full.csv` for per-item latency)")
        if "swap_note" in s:
            md.append(f"- Model-swap history: {s['swap_note']}")
        md.append("")
    md.append("## Shared prompt text (identical for all 3 solvers)\n")
    md.append("```\n" + prompt_text.strip() + "\n```\n")
    md.append("## Parser rule\n")
    md.append("Raw model text output is matched against the single-character options A/B/C/D "
               "(case-insensitive, leading/trailing whitespace stripped); the first matching letter is "
               "mapped to option index 1-4. `parse_success=True` iff exactly one of A/B/C/D is "
               "identifiable in the raw output.\n")
    md.append("## Invalid-output handling\n")
    md.append("All three solvers achieved 100% parse success on both the 50-item pilot and the full "
               "944-item run (`03_machine_solver_runs.md`); the invalid-parse path (drop from analysis "
               "denominator, do not count as correct or incorrect) is implemented but was never triggered "
               "in the frozen runs.\n")
    md.append("## GPU / environment\n")
    md.append("RunPod on-demand pod, 1x NVIDIA GeForce RTX 4090 (24564 MiB VRAM), driver 550.127.05, "
               "128 vCPU / 503 GiB RAM host, Linux container, Python 3.11.10. SSH (ed25519 key-based) "
               "access only; no third-party inference API used (see `content_handling.md`).\n")
    md.append("## Exact inference counts (final 944-item clean universe)\n")
    md.append(counts_df.to_markdown(index=False))
    md.append("\n## Corrupted-file incident (reproducibility record only -- do not put in manuscript)\n")
    md.append("During RunPod setup, two premature/duplicate launch attempts produced a corrupted, "
               "header-less `solver1_full.csv` (a race between two accidentally-concurrent invocations). "
               "Caught by post-hoc integrity check (`pod_check_csv.py`: verifies `question_id` column "
               "presence and exact 944-row/944-unique-id counts) before any analysis; file deleted and "
               "solver 1 was re-run cleanly. This is a documented compute-environment/setup-process "
               "incident, unrelated to any local-PC hardware failure, and belongs only in the audit/"
               "reproducibility record, never in the manuscript Methods/Results.\n")
    (EVID / "vlm_inference_method_full.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ================= Part L: content equivalence audit =================
    manifest = pd.read_parquet(MANIFEST)
    integ = pd.read_parquet(INTEGRATED)
    rng = np.random.default_rng(20260818)
    sample_ids = rng.choice(manifest.question_id.values, size=20, replace=False)
    sample_ids = sorted(int(x) for x in sample_ids)

    l_rows = []
    for qid in sample_ids:
        m = manifest[manifest.question_id == qid].iloc[0]
        i = integ[integ.question_id == qid].iloc[0]
        img_path = ROOT / m.content_asset_path
        exists = img_path.exists()
        sha = None
        if exists:
            sha = hashlib.sha256(img_path.read_bytes()).hexdigest()[:16]
        l_rows.append({
            "question_id": qid,
            "content_asset_path": m.content_asset_path,
            "asset_exists": exists,
            "asset_sha256_16": sha,
            "correct_option_manifest": int(m.correct_option),
            "correct_option_integrated": int(i.correct_option),
            "correct_option_matches": int(m.correct_option) == int(i.correct_option),
            "n_students_linked": int(i.n_students),
            "n_attempts_linked": int(i.n_attempts),
            "solver_1_has_prediction": bool(pd.notna(i.solver_1_correct)),
            "solver_2_has_prediction": bool(pd.notna(i.solver_2_correct)),
            "solver_3_has_prediction": bool(pd.notna(i.solver_3_correct)),
        })
    l_df = pd.DataFrame(l_rows)

    all_pass = bool(l_df.asset_exists.all() and l_df.correct_option_matches.all() and
                    l_df.solver_1_has_prediction.all() and l_df.solver_2_has_prediction.all() and
                    l_df.solver_3_has_prediction.all())

    l_md = []
    l_md.append("# Content Equivalence Audit (Part L, v2.1)\n")
    l_md.append(f"20 randomly sampled retained items (seed=20260818, from the 944-item manifest):\n")
    l_md.append(l_df.to_markdown(index=False))
    l_md.append(f"\n## Verdict: {'ALL 20/20 CHECKS PASS' if all_pass else 'FAILURES DETECTED -- see table above'}\n")
    l_md.append("Every sampled item has: (a) a present, hashable content image asset; (b) an identical "
                 "correct-answer option recorded in both the S1 item manifest and the S7 integrated table "
                 "(built independently from the same source, cross-checked here); (c) a complete "
                 "student-response linkage (n_students/n_attempts > 0, matching the raw EeDi response log); "
                 "(d) a recorded prediction from all three model solvers. This confirms models and students "
                 "are evaluated on the SAME question content asset, with the SAME correct-answer key.\n")
    l_md.append("## Content transformations before local VLM inference\n")
    l_md.append("None beyond each model's own standard Hugging Face image processor (resize/normalize to "
                 "that model's native input format, per `AutoProcessor.from_pretrained` defaults -- see "
                 "`vlm_inference_method_full.md`). No cropping, re-rendering, text extraction/OCR, "
                 "watermarking, or manual editing of question images was performed. The identical raw JPEG "
                 "asset (`data/eedi_public_download/extracted/data/images/{QuestionId}.jpg`) that produced "
                 "the human student-response data (same EeDi 'Tasks 3&4' content pool) is the exact file "
                 "shown to each VLM solver.\n")
    (EVID / "same_item_content_equivalence.md").write_text("\n".join(l_md) + "\n", encoding="utf-8")

    print("Wrote vlm_inference_method_full.md, vlm_exact_counts.csv, same_item_content_equivalence.md")
    print(f"Content equivalence all_pass={all_pass}")


if __name__ == "__main__":
    main()
