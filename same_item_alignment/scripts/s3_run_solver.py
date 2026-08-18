#!/usr/bin/env python3
"""Run one local multimodal solver over a set of same-item questions.

Usage:
    python s3_run_solver.py --solver_id solver_1_qwen2vl7b --items_csv outputs/pilot/pilot50_items.csv --out outputs/pilot/pilot50_predictions_solver1.csv
    python s3_run_solver.py --solver_id solver_1_qwen2vl7b --items_parquet data/eedi948_item_manifest.parquet --out data/raw_predictions/solver1_full.csv

Deterministic / near-deterministic decoding (do_sample=False <=> temperature=0).
Output is restricted to a single letter A/B/C/D via prompt + parsing; no chain-of-thought stored.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOLVERS_CFG = ROOT / "same_item_alignment/configs/solvers.json"
PROMPT_PATH = ROOT / "same_item_alignment/configs/prompts/mcq_prompt.txt"

LETTER_TO_OPTION = {"A": 1, "B": 2, "C": 3, "D": 4}


def parse_letter(raw: str) -> int | None:
    if not raw:
        return None
    m = re.search(r"[ABCD]", raw.upper())
    return LETTER_TO_OPTION[m.group(0)] if m else None


def load_qwen2vl(checkpoint: str):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(checkpoint)

    def infer(image_path: Path, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    return infer, model


def _generic_native_vlm_loader(checkpoint: str):
    """Generic loader for natively-supported (non trust_remote_code) transformers VLMs
    that expose the standard chat-template + processor(text=..., images=...) API,
    e.g. InternVLForConditionalGeneration, MiniCPMV4_6ForConditionalGeneration."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(checkpoint)

    def infer(image_path: Path, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    return infer, model


def load_internvl(checkpoint: str):
    return _generic_native_vlm_loader(checkpoint)


def load_minicpmv(checkpoint: str):
    return _generic_native_vlm_loader(checkpoint)


def load_llavanext(checkpoint: str):
    return _generic_native_vlm_loader(checkpoint)


def load_smolvlm(checkpoint: str):
    return _generic_native_vlm_loader(checkpoint)


LOADERS = {
    "solver_1_qwen2vl7b": load_qwen2vl,
    "solver_2_internvl3_8b": load_internvl,
    "solver_3_llavanext_mistral7b": load_llavanext,
    "solver_3_smolvlm2_2b": load_smolvlm,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver_id", required=True)
    ap.add_argument("--items_csv")
    ap.add_argument("--items_parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = json.loads(SOLVERS_CFG.read_text(encoding="utf-8"))
    solver_cfg = next(s for s in cfg["solvers"] if s["solver_id"] == args.solver_id)
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()

    if args.items_csv:
        items = pd.read_csv(ROOT / args.items_csv)
    else:
        items = pd.read_parquet(ROOT / args.items_parquet)
    if args.limit:
        items = items.head(args.limit)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip question_ids already written (crash/interrupt safe).
    done_qids = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done_qids = set(prev.question_id.tolist())
        print(f"Resuming: {len(done_qids)} items already completed in {out_path}")

    remaining = items[~items.question_id.isin(done_qids)]
    if len(remaining) == 0:
        print("Nothing to do, all items already completed.")
        return

    print(f"Loading {solver_cfg['exact_checkpoint']} ...")
    infer, model = LOADERS[args.solver_id](solver_cfg["exact_checkpoint"])
    print(f"Loaded. Running inference on {len(remaining)} remaining items ...")

    write_header = not out_path.exists()
    buffer = []
    FLUSH_EVERY = 10

    def flush():
        nonlocal buffer, write_header
        if not buffer:
            return
        pd.DataFrame(buffer).to_csv(out_path, mode="a", header=write_header, index=False)
        write_header = False
        buffer = []

    for idx, r in enumerate(remaining.itertuples()):
        qid = r.question_id
        img_path = ROOT / r.content_asset_path
        t0 = time.time()
        raw_output = None
        parsed = None
        parse_success = False
        try:
            raw_output = infer(img_path, prompt)
            parsed = parse_letter(raw_output)
            parse_success = parsed is not None
        except Exception as e:
            raw_output = f"__ERROR__:{type(e).__name__}:{e}"
        runtime_ms = (time.time() - t0) * 1000

        correct_option = int(r.correct_option)
        machine_correct = bool(parsed == correct_option) if parse_success else None

        buffer.append({
            "question_id": int(qid),
            "solver_id": args.solver_id,
            "raw_output": raw_output,
            "parsed_option": parsed,
            "parse_success": parse_success,
            "correct_option": correct_option,
            "machine_correct": machine_correct,
            "request_index": len(done_qids) + idx,
            "runtime_ms": runtime_ms,
        })
        if idx % 25 == 0:
            print(f"  [{idx+1}/{len(remaining)}] qid={qid} raw={raw_output!r} parsed={parsed} correct={correct_option}")
        if len(buffer) >= FLUSH_EVERY:
            flush()
    flush()

    out_df = pd.read_csv(out_path)
    print(f"Wrote {len(out_df)} total predictions -> {out_path}")

    n_parsed = int(out_df.parse_success.sum())
    n_correct = int(out_df.machine_correct.fillna(False).sum())
    print(json.dumps({
        "solver_id": args.solver_id,
        "n_items": len(out_df),
        "parse_success_rate": n_parsed / len(out_df) if len(out_df) else None,
        "accuracy_on_parsed": n_correct / n_parsed if n_parsed else None,
        "answer_distribution": out_df.parsed_option.value_counts(dropna=False).to_dict(),
        "mean_runtime_ms": float(out_df.runtime_ms.mean()),
    }, indent=2, default=str))

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
