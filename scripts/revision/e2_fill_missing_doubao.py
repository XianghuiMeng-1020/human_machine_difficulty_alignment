#!/usr/bin/env python3
"""Fill missing Doubao outputs on official RACE val HIGH items.

Requires env:
  ARK_API_KEY or DOUBAO_API_KEY
  ARK_BASE_URL (default https://ark.cn-beijing.volces.com/api/v3)
  DOUBAO_MODEL (default ep-20250703174207-j75k9 from LLM_request.py)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from common import REPO_ROOT, ensure_dir, parse_option_letter


INSTRUCTION = (
    "You are an expert in reading comprehension. Carefully analyze each RACE "
    "passage and corresponding questions, then choose the most accurate answer "
    "from the given options."
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompts_jsonl",
        default=str(REPO_ROOT / "race_prepared/race_llm_prompts_val.jsonl"),
    )
    ap.add_argument(
        "--existing_jsonl",
        default=str(REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao.jsonl"),
    )
    ap.add_argument(
        "--out_jsonl",
        default=str(REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_high_fill.jsonl"),
    )
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def qid_of(row: dict) -> str:
    for k in ("question_id", "id"):
        if row.get(k):
            return str(row[k])
    return f"{row.get('base_id','')}_q{row.get('question_index',0)}"


def main():
    args = parse_args()
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
    if not api_key:
        raise RuntimeError("Set ARK_API_KEY or DOUBAO_API_KEY")
    base_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model = os.environ.get("DOUBAO_MODEL", "ep-20250703174207-j75k9")
    client = OpenAI(api_key=api_key, base_url=base_url)

    existing = set()
    if Path(args.existing_jsonl).is_file():
        with open(args.existing_jsonl, encoding="utf-8") as f:
            for line in f:
                existing.add(qid_of(json.loads(line)))

    todos = []
    with open(args.prompts_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = qid_of(row)
            if qid in existing or qid.startswith("middle"):
                continue
            band = row.get("designer_difficulty_str") or row.get("designer_difficulty")
            # Keep HIGH only (prompt jsonl uses designer_difficulty 0/1 or MIDDLE/HIGH)
            if qid.startswith("high"):
                pass
            elif str(band).upper() in {"HIGH", "1"} or band == 1:
                pass
            else:
                continue
            todos.append((qid, row.get("prompt", ""), row))
    if args.limit:
        todos = todos[: args.limit]

    out_path = Path(args.out_jsonl)
    ensure_dir(out_path.parent)
    done = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                done.add(qid_of(json.loads(line)))
    todos = [t for t in todos if t[0] not in done]
    print(f"[E2-Doubao] remaining={len(todos)}")
    access_date = datetime.now(timezone.utc).date().isoformat()

    def work(item):
        qid, prompt, row = item
        temp = 0.1
        last = ""
        n_retries = 0
        for attempt in range(8):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": prompt + "\nOnly tell me the final answer.\n"},
                ],
                temperature=temp,
            )
            last = (resp.choices[0].message.content or "").strip()
            if parse_option_letter(last):
                n_retries = attempt
                break
            temp = round(min(1.5, temp + 0.2), 2)
            time.sleep(0.3)
        return {
            "question_id": qid,
            "llm_label": last,
            "parsed_letter": parse_option_letter(last),
            "temperature": temp,
            "n_retries": n_retries,
            "retried": int(n_retries > 0),
            "model": model,
            "access_date": access_date,
            "backend": "doubao",
            "label": row.get("label"),
            "answer_letter": row.get("answer_letter"),
            "designer_difficulty_str": row.get("designer_difficulty_str"),
        }

    ok = 0
    with out_path.open("a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(work, t) for t in todos]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            ok += 1
            if i % 50 == 0:
                print(f"[E2-Doubao] {i}/{len(todos)}")
    print(f"[OK] doubao fill ok={ok} -> {out_path}")


if __name__ == "__main__":
    main()
