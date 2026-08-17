#!/usr/bin/env python3
"""Merge base GPT val jsonl with HIGH-fill jsonl into a single coverage file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REPO_ROOT, ensure_dir


def qid_of(row: dict) -> str:
    if row.get("question_id"):
        return str(row["question_id"])
    base = row.get("base_id") or ""
    qi = row.get("question_index", 0)
    return f"{base}_q{qi}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=str(REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt.jsonl"),
    )
    ap.add_argument(
        "--fills",
        nargs="+",
        default=[
            str(REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_high_fill.jsonl"),
        ],
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_merged.jsonl"),
    )
    args = ap.parse_args()

    by_qid = {}
    for path in [args.base, *args.fills]:
        p = Path(path)
        if not p.is_file():
            print(f"[WARN] skip missing {p}")
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                by_qid[qid_of(row)] = row
    out = Path(args.out)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        for qid in sorted(by_qid):
            f.write(json.dumps(by_qid[qid], ensure_ascii=False) + "\n")
    print(f"[OK] merged {len(by_qid)} -> {out}")


if __name__ == "__main__":
    main()
