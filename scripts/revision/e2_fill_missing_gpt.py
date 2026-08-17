#!/usr/bin/env python3
"""Fill missing GPT-4o outputs on official RACE val (esp. HIGH band).

Uses ByteDance OpenAPI crawl endpoint (same path as scripts/LLM_request.py).
Credentials via env:
  BYTEDANCE_GPT_AK  (required)
  BYTEDANCE_GPT_URL (optional)
  GPT_MODEL         (default gpt-4o-2024-11-20)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import REPO_ROOT, REVISION_ROOT, ensure_dir, parse_option_letter


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
        default=str(REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt.jsonl"),
    )
    ap.add_argument(
        "--out_jsonl",
        default=str(REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_high_fill.jsonl"),
    )
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--only_high", action="store_true", default=True)
    return ap.parse_args()


def qid_of(row: dict) -> str:
    for k in ("question_id", "id"):
        if row.get(k):
            return str(row[k])
    base = row.get("base_id") or row.get("passage_file") or ""
    qi = row.get("question_index", row.get("q_index", 0))
    return f"{base}_q{qi}"


def request_gpt_bytedance(
    prompt: str, ak: str, url: str, model: str, temperature: float
) -> tuple[str, float, int]:
    headers = {"Content-Type": "application/json"}
    temp = temperature
    last = ""
    for attempt in range(8):
        data = {
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": prompt + "\nOnly tell me the final answer.\n"},
            ],
            "model": model,
            "max_tokens": 32,
            "temperature": temp,
            "top_p": 0,
            "n": 1,
            "stream": False,
        }
        try:
            resp = requests.post(f"{url}?ak={ak}", headers=headers, json=data, timeout=120)
            res = resp.json()
            last = res["choices"][0]["message"]["content"].strip()
            if parse_option_letter(last):
                return last, temp, attempt
        except Exception as e:
            last = f"ERROR: {e}"
            time.sleep(2)
        temp = round(min(1.5, max(0.1, (temp or 0.1) + 0.2)), 2)
        time.sleep(0.3)
    return last, temp, 8


def request_gpt_openai(
    prompt: str, api_key: str, base_url: str, model: str, temperature: float
) -> tuple[str, float, int]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    temp = temperature
    last = ""
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": prompt + "\nOnly tell me the final answer.\n"},
                ],
                max_tokens=32,
                temperature=temp,
            )
            last = (resp.choices[0].message.content or "").strip()
            if parse_option_letter(last):
                return last, temp, attempt
        except Exception as e:
            last = f"ERROR: {e}"
            time.sleep(2)
        temp = round(min(1.5, max(0.1, (temp or 0.1) + 0.2)), 2)
        time.sleep(0.3)
    return last, temp, 8


def main():
    args = parse_args()
    openai_key = (
        os.environ.get("GPT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    bytedance_ak = os.environ.get("BYTEDANCE_GPT_AK") or os.environ.get("GPT_AK")
    use_openai = bool(openai_key) and not bytedance_ak
    if not openai_key and not bytedance_ak:
        raise RuntimeError(
            "Set GPT_API_KEY/OPENAI_API_KEY (OpenAI-compatible) or BYTEDANCE_GPT_AK"
        )
    url = os.environ.get(
        "BYTEDANCE_GPT_URL",
        "https://search.bytedance.net/gpt/openapi/online/v2/crawl",
    )
    base_url = os.environ.get("GPT_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("GPT_MODEL", "gpt-4o" if use_openai else "gpt-4o-2024-11-20")
    print(
        f"[E2-GPT] transport={'openai_compatible' if use_openai else 'bytedance'} "
        f"model={model} base={base_url if use_openai else url}"
    )

    existing = set()
    ex_path = Path(args.existing_jsonl)
    if ex_path.is_file():
        with ex_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(qid_of(json.loads(line)))
                except Exception:
                    continue

    todos = []
    with Path(args.prompts_jsonl).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = qid_of(row)
            if qid in existing:
                continue
            band = str(
                row.get("designer_difficulty_str")
                or row.get("designer_difficulty")
                or row.get("grade_band")
                or ""
            )
            if args.only_high:
                is_high = (
                    band.upper() in {"HIGH", "H", "RACE-H", "1", "TRUE"}
                    or qid.startswith("high")
                    or str(band) == "1"
                )
                # designer_difficulty may be int 1=HIGH / 0=MIDDLE in prompt jsonl
                try:
                    if int(band) == 1:
                        is_high = True
                except Exception:
                    pass
                if not is_high:
                    continue
            prompt = row.get("prompt") or row.get("input") or ""
            if not prompt:
                continue
            todos.append((qid, prompt, row))

    if args.limit:
        todos = todos[: args.limit]
    print(f"[E2-GPT] missing to fill: {len(todos)} (existing={len(existing)})")
    if not todos:
        print("[E2-GPT] nothing to do")
        return

    out_path = Path(args.out_jsonl)
    ensure_dir(out_path.parent)
    done = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(qid_of(json.loads(line)))
                except Exception:
                    continue
    todos = [t for t in todos if t[0] not in done]
    print(f"[E2-GPT] remaining after resume: {len(todos)}")

    access_date = datetime.now(timezone.utc).date().isoformat()

    def work(item):
        qid, prompt, row = item
        if use_openai:
            raw, temp, n_retries = request_gpt_openai(
                prompt, openai_key, base_url, model, args.temperature
            )
        else:
            raw, temp, n_retries = request_gpt_bytedance(
                prompt, bytedance_ak, url, model, args.temperature
            )
        letter = parse_option_letter(raw)
        out = {
            **{k: row.get(k) for k in ("question_id", "base_id", "label", "answer_letter", "designer_difficulty_str") if k in row},
            "question_id": qid,
            "llm_label": raw,
            "parsed_letter": letter,
            "temperature": temp,
            "n_retries": n_retries,
            "retried": int(n_retries > 0),
            "model": model,
            "access_date": access_date,
            "backend": "gpt4o",
        }
        return out

    with out_path.open("a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(work, t): t[0] for t in todos}
        ok = 0
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                ok += 1
            except Exception as e:
                print(f"[ERROR] {futs[fut]}: {e}")
            if i % 50 == 0:
                print(f"[E2-GPT] progress {i}/{len(todos)} ok={ok}")

    print(f"[OK] wrote fill outputs -> {out_path} ok={ok}/{len(todos)}")


if __name__ == "__main__":
    main()
