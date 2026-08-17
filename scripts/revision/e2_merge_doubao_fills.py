#!/usr/bin/env python3
"""Merge base Doubao val jsonl with HIGH-fill jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REPO_ROOT, ensure_dir


def qid_of(row: dict) -> str:
    if row.get("question_id"):
        return str(row["question_id"])
    return f"{row.get('base_id','')}_q{row.get('question_index',0)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=str(REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao.jsonl"),
    )
    ap.add_argument(
        "--fills",
        nargs="+",
        default=[
            str(REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_high_fill.jsonl"),
        ],
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_merged.jsonl"),
    )
    args = ap.parse_args()
    by = {}
    for path in [args.base, *args.fills]:
        p = Path(path)
        if not p.is_file():
            print(f"[WARN] skip {p}")
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                by[qid_of(row)] = row
    out = Path(args.out)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        for qid in sorted(by):
            f.write(json.dumps(by[qid], ensure_ascii=False) + "\n")
    print(f"[OK] merged {len(by)} -> {out}")


if __name__ == "__main__":
    main()
