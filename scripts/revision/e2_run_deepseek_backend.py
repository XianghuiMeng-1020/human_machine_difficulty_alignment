#!/usr/bin/env python3
"""E2 helper: run third LLM backend (DeepSeek) on official RACE val prompts.

Requires API credentials via env:
  DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL (optional), DEEPSEEK_MODEL (optional)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from common import REPO_ROOT, REVISION_ROOT, ensure_dir, parse_option_letter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompts_jsonl",
        default=str(REVISION_ROOT / "artifacts/llm_deepseek_val_TODO.jsonl"),
    )
    ap.add_argument(
        "--out_jsonl",
        default=str(REVISION_ROOT / "artifacts/llm_deepseek_val.jsonl"),
    )
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--temperature", type=float, default=0.1)
    return ap.parse_args()


def call_deepseek(prompt: str, temperature: float) -> tuple[str, float, int]:
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) to run third backend")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    client = OpenAI(api_key=api_key, base_url=base_url)

    instruction = (
        "You are an expert in reading comprehension. "
        "Choose the most accurate answer from A, B, C, or D. "
        "Reply with only one letter."
    )
    temp = temperature
    last = ""
    for attempt in range(8):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt + "\nOnly tell me the final answer."},
            ],
            temperature=temp,
            max_tokens=16,
        )
        last = (resp.choices[0].message.content or "").strip()
        if parse_option_letter(last):
            return last, temp, attempt
        temp = round(min(1.5, max(0.1, temp + 0.2)), 2)
        time.sleep(0.5)
    return last, temp, 8


def main():
    args = parse_args()
    in_path = Path(args.prompts_jsonl)
    if not in_path.is_file():
        # fallback: build from race val
        val = REPO_ROOT / "race_prepared/race_llm_prompts_val.jsonl"
        in_path = val
    rows = []
    with in_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out_jsonl)
    ensure_dir(out_path.parent)
    access_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def work(obj):
        prompt = obj.get("prompt")
        if not prompt:
            return None
        raw, temp, n_retries = call_deepseek(prompt, args.temperature)
        out = dict(obj)
        out.update(
            {
                "llm_label": raw,
                "pred_answer": parse_option_letter(raw),
                "temperature": temp,
                "n_retries": n_retries,
                "access_date": access_date,
                "model_version": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                "backend": "deepseek",
            }
        )
        return out

    done = 0
    with out_path.open("w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = [ex.submit(work, r) for r in rows]
            for fut in as_completed(futs):
                res = fut.result()
                if res is None:
                    continue
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                done += 1
                if done % 50 == 0:
                    print(f"[INFO] wrote {done}/{len(rows)}")
    print(f"[OK] DeepSeek outputs -> {out_path} ({done} rows)")


if __name__ == "__main__":
    main()
