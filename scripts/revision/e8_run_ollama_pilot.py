#!/usr/bin/env python3
"""E8: run local Ollama to (1) produce short answers (2) grade them.

This replaces SIMULATED scores with real local-model outputs for the open-ended
pilot. It does NOT substitute for GPT/Doubao/DeepSeek in RQ3.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from common import REVISION_ROOT, ensure_dir, save_table
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--items_csv",
        default=str(REVISION_ROOT / "open_ended/e8_open_ended_pilot_items.csv"),
    )
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out_csv",
        default=str(REVISION_ROOT / "open_ended/e8_ollama_scores.csv"),
    )
    return ap.parse_args()


def ollama_chat(model: str, prompt: str, timeout: int = 300) -> str:
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:] or f"ollama exit {proc.returncode}")
    return (proc.stdout or "").strip()


def parse_score(text: str) -> int | None:
    m = re.search(r"\b([012])\b", text.strip())
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    items = pd.read_csv(args.items_csv)
    if args.limit:
        items = items.head(args.limit)
    out_path = Path(args.out_csv)
    ensure_dir(out_path.parent)

    done = set()
    rows_out = []
    if out_path.is_file():
        prev = pd.read_csv(out_path)
        done = set(prev["item_id"].astype(str))
        rows_out = prev.to_dict("records")

    print(f"[E8-ollama] model={args.model} items={len(items)} resume_done={len(done)}")
    for i, row in items.iterrows():
        item_id = str(row["item_id"])
        if item_id in done:
            continue
        ans_prompt = (
            "Read the passage and answer the question in one short sentence.\n\n"
            f"Passage:\n{str(row['passage'])[:2500]}\n\n"
            f"Question:\n{row['question']}\n\n"
            "Answer:"
        )
        t0 = time.time()
        student_answer = ollama_chat(args.model, ans_prompt)
        grade_prompt = (
            "Grade the student short answer using the rubric 0/1/2.\n"
            f"Passage:\n{str(row['passage'])[:2000]}\n\n"
            f"Question:\n{row['question']}\n\n"
            f"Reference answer:\n{row['reference_answer']}\n\n"
            f"Student answer:\n{student_answer}\n\n"
            "Return only an integer 0, 1, or 2."
        )
        grade_raw = ollama_chat(args.model, grade_prompt)
        score = parse_score(grade_raw)
        rows_out.append(
            {
                "item_id": item_id,
                "designer_band": row.get("designer_band"),
                "reference_answer": row.get("reference_answer"),
                "student_answer": student_answer,
                "grader_raw": grade_raw,
                "score_0_1_2": score,
                "model": args.model,
                "seconds": round(time.time() - t0, 2),
            }
        )
        pd.DataFrame(rows_out).to_csv(out_path, index=False)
        print(f"[{len(rows_out)}/{len(items)}] {item_id} score={score}")

    df = pd.DataFrame(rows_out)
    save_table(df, out_path)
    summary = {
        "n": len(df),
        "n_scored": int(df["score_0_1_2"].notna().sum()),
        "mean_score": float(df["score_0_1_2"].dropna().mean()) if len(df) else None,
        "model": args.model,
    }
    (REVISION_ROOT / "tables" / "table_e8_ollama_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[OK]", summary)


if __name__ == "__main__":
    main()
