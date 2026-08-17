#!/usr/bin/env python3
"""E2: Aggregate multi-backend LLM outputs, voting, retry/temperature audit.

Supports existing GPT-4o + Doubao logs; scaffolds a third backend slot (DeepSeek).
Writes revision/artifacts/llm_vote_val.csv and reproducibility table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPO_ROOT,
    REVISION_ROOT,
    ensure_dir,
    letter_to_label,
    load_jsonl,
    parse_option_letter,
    save_table,
    write_jsonl,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--race_val_csv",
        default=str(REPO_ROOT / "race_prepared/race_mcq_val.csv"),
    )
    ap.add_argument(
        "--gpt_jsonl",
        default=str(
            REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_merged.jsonl"
            if (REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_merged.jsonl").is_file()
            else REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt.jsonl"
        ),
    )
    ap.add_argument(
        "--gpt_back_jsonl",
        default=str(REPO_ROOT / "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt_back.jsonl"),
    )
    ap.add_argument(
        "--doubao_jsonl",
        default=str(
            REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_merged.jsonl"
            if (REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_merged.jsonl").is_file()
            else REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao.jsonl"
        ),
    )
    ap.add_argument(
        "--doubao_back_jsonl",
        default=str(REPO_ROOT / "LLM_out/doubao_1.8/race_llm_prompts_val_doubao_back.jsonl"),
    )
    ap.add_argument(
        "--third_jsonl",
        default=str(REVISION_ROOT / "artifacts/llm_deepseek_val.jsonl"),
        help="Third backend outputs (created when API run completes)",
    )
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def normalize_backend_rows(path: Path, backend: str, is_retry_file: bool = False) -> pd.DataFrame:
    if not path.is_file():
        print(f"[WARN] missing {backend}: {path}")
        return pd.DataFrame(
            columns=[
                "question_id",
                "backend",
                "raw_response",
                "parsed_letter",
                "pred_label",
                "is_retry_file",
                "source_path",
            ]
        )
    rows = []
    for obj in load_jsonl(path):
        qid = obj.get("question_id") or obj.get("id")
        raw = obj.get("llm_label") or obj.get("pred_answer") or obj.get("answer")
        letter = parse_option_letter(raw)
        rows.append(
            {
                "question_id": qid,
                "backend": backend,
                "raw_response": raw,
                "parsed_letter": letter,
                "pred_label": letter_to_label(letter),
                "is_retry_file": int(is_retry_file),
                "temperature": obj.get("temperature"),
                "n_retries": obj.get("n_retries"),
                "access_date": obj.get("access_date"),
                "model_version": obj.get("model_version") or obj.get("model"),
                "source_path": str(path),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("question_id", keep="first")
    print(f"[INFO] {backend}: {len(df)} rows from {path.name}; parsed={df['pred_label'].notna().sum()}")
    return df


def majority_label(labels: list) -> int | None:
    vals = [int(x) for x in labels if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not vals:
        return None
    counts = Counter(vals)
    top = counts.most_common()
    # ties -> lowest index (deterministic audit rule)
    best_count = top[0][1]
    tied = sorted([lab for lab, c in top if c == best_count])
    return tied[0]


def two_of_three(votes: dict[str, int | None]) -> tuple[int | None, int]:
    vals = [v for v in votes.values() if v is not None]
    if len(vals) < 2:
        return None, 1
    counts = Counter(vals)
    lab, cnt = counts.most_common(1)[0]
    if cnt >= 2:
        return lab, 0
    return None, 1


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    art = ensure_dir(out_dir / "artifacts")
    tables = ensure_dir(out_dir / "tables")

    val = pd.read_csv(args.race_val_csv)
    gold = dict(zip(val["question_id"], val["label"]))

    backends = {
        "gpt4o": normalize_backend_rows(Path(args.gpt_jsonl), "gpt4o", False),
        "gpt4o_back": normalize_backend_rows(Path(args.gpt_back_jsonl), "gpt4o", True),
        "doubao": normalize_backend_rows(Path(args.doubao_jsonl), "doubao", False),
        "doubao_back": normalize_backend_rows(Path(args.doubao_back_jsonl), "doubao", True),
        "deepseek": normalize_backend_rows(Path(args.third_jsonl), "deepseek", False),
    }

    # Stage 1: consolidate primary + retry file per backend by majority / prefer primary parse
    def consolidate(primary: pd.DataFrame, retry: pd.DataFrame, name: str) -> pd.DataFrame:
        p = primary.set_index("question_id") if len(primary) else pd.DataFrame()
        r = retry.set_index("question_id") if len(retry) else pd.DataFrame()
        qids = sorted(set(p.index.tolist()) | set(r.index.tolist()))
        rows = []
        for qid in qids:
            labs = []
            temps = []
            retries = 0
            raws = []
            for src, flag in [(p, 0), (r, 1)]:
                if qid in src.index:
                    row = src.loc[qid]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    labs.append(row.get("pred_label"))
                    raws.append(row.get("raw_response"))
                    if row.get("temperature") is not None:
                        temps.append(row.get("temperature"))
                    if flag:
                        retries += 1
                    if row.get("n_retries") is not None:
                        try:
                            retries = max(retries, int(row.get("n_retries")))
                        except Exception:
                            pass
            # Prefer primary parsed label; else retry; else majority
            pred = None
            if qid in p.index:
                prow = p.loc[qid]
                if isinstance(prow, pd.DataFrame):
                    prow = prow.iloc[0]
                pred = prow.get("pred_label")
            if pred is None and qid in r.index:
                rrow = r.loc[qid]
                if isinstance(rrow, pd.DataFrame):
                    rrow = rrow.iloc[0]
                pred = rrow.get("pred_label")
            if pred is None or (isinstance(pred, float) and np.isnan(pred)) or pd.isna(pred):
                pred = majority_label(labs)
            if pred is not None and (isinstance(pred, float) and np.isnan(pred) or pd.isna(pred)):
                pred = None
            used_retry = int(qid in r.index and (qid not in p.index or pd.isna(p.loc[qid].get("pred_label") if not isinstance(p.loc[qid], pd.DataFrame) else p.loc[qid].iloc[0].get("pred_label"))))
            letter = None
            if pred is not None and not pd.isna(pred):
                try:
                    letter = "ABCD"[int(pred)]
                except (ValueError, TypeError, IndexError):
                    letter = None
                    pred = None
            rows.append(
                {
                    "question_id": qid,
                    f"llm_{name}_pred": pred,
                    f"llm_{name}_letter": letter,
                    f"llm_{name}_retried": used_retry,
                    f"llm_{name}_n_retries": retries,
                    f"llm_{name}_temperature": temps[-1] if temps else np.nan,
                }
            )
        return pd.DataFrame(rows)

    gpt = consolidate(backends["gpt4o"], backends["gpt4o_back"], "gpt4o")
    doubao = consolidate(backends["doubao"], backends["doubao_back"], "doubao")
    deepseek = backends["deepseek"].rename(
        columns={
            "pred_label": "llm_deepseek_pred",
            "parsed_letter": "llm_deepseek_letter",
        }
    )
    if len(deepseek):
        deepseek = deepseek[
            ["question_id", "llm_deepseek_pred", "llm_deepseek_letter"]
        ].copy()
        deepseek["llm_deepseek_retried"] = 0
        deepseek["llm_deepseek_n_retries"] = 0
        deepseek["llm_deepseek_temperature"] = np.nan
    else:
        deepseek = pd.DataFrame(
            columns=[
                "question_id",
                "llm_deepseek_pred",
                "llm_deepseek_letter",
                "llm_deepseek_retried",
                "llm_deepseek_n_retries",
                "llm_deepseek_temperature",
            ]
        )

    vote = val[["question_id", "label", "designer_difficulty_str"]].merge(gpt, on="question_id", how="left")
    vote = vote.merge(doubao, on="question_id", how="left")
    vote = vote.merge(deepseek, on="question_id", how="left")

    preds = []
    no_cons = []
    n_backends_used = []
    for _, row in vote.iterrows():
        backend_votes = {
            "gpt4o": row.get("llm_gpt4o_pred"),
            "doubao": row.get("llm_doubao_pred"),
            "deepseek": row.get("llm_deepseek_pred"),
        }
        # If third backend missing, fall back to 2-of-2 agreement for available backends
        available = {k: v for k, v in backend_votes.items() if pd.notna(v)}
        n_backends_used.append(len(available))
        if len(available) >= 3:
            lab, nc = two_of_three(backend_votes)
        elif len(available) == 2:
            vals = list(available.values())
            if vals[0] == vals[1]:
                lab, nc = int(vals[0]), 0
            else:
                lab, nc = None, 1
        elif len(available) == 1:
            # Single backend: record prediction but mark no-consensus under multi-backend rule
            lab, nc = int(list(available.values())[0]), 1
        else:
            lab, nc = None, 1
        preds.append(lab)
        no_cons.append(nc)

    vote["llm_pred"] = preds
    vote["llm_no_consensus"] = no_cons
    vote["llm_n_backends"] = n_backends_used
    vote["llm_letter"] = vote["llm_pred"].apply(
        lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else "ABCD"[int(x)]
    )
    vote["llm_correct"] = [
        (float(p) == float(g)) if (p is not None and not (isinstance(p, float) and np.isnan(p))) else np.nan
        for p, g in zip(vote["llm_pred"], vote["label"])
    ]
    vote["llm_any_retried"] = (
        vote.filter(like="_retried").fillna(0).astype(int).max(axis=1)
    )

    save_table(vote, art / "llm_vote_val.csv")

    # Retry audit summary
    retry_summary = pd.DataFrame(
        [
            {
                "n_questions": len(vote),
                "n_with_gpt": int(vote["llm_gpt4o_pred"].notna().sum()),
                "n_with_doubao": int(vote["llm_doubao_pred"].notna().sum()),
                "n_with_deepseek": int(vote["llm_deepseek_pred"].notna().sum()),
                "n_consensus": int((vote["llm_no_consensus"] == 0).sum()),
                "no_consensus_rate": float(vote["llm_no_consensus"].mean()),
                "retry_rate": float(vote["llm_any_retried"].mean()),
                "llm_acc_consensus": float(
                    vote.loc[vote["llm_no_consensus"] == 0, "llm_correct"].mean()
                ),
                "llm_acc_no_cons_as_wrong": float(
                    np.where(vote["llm_no_consensus"] == 1, 0.0, vote["llm_correct"].fillna(0)).mean()
                ),
            }
        ]
    )
    save_table(retry_summary, tables / "table_llm_retry_and_consensus.csv")

    # Reproducibility table (R2-5)
    repro = pd.DataFrame(
        [
            {
                "backend": "gpt4o",
                "model_name": "gpt-4o-2024-11-20",
                "provider": "OpenAI-compatible gateway",
                "access_date": "see source logs / manuscript Methods",
                "temperature_policy": "retry may raise temperature in [0.1, 1.5]",
                "n_val_outputs": int(vote["llm_gpt4o_pred"].notna().sum()),
                "source_file": args.gpt_jsonl,
            },
            {
                "backend": "doubao",
                "model_name": "Doubao-1.8 (ep endpoint in LLM_request.py)",
                "provider": "ByteDance Volcengine",
                "access_date": "see source logs / manuscript Methods",
                "temperature_policy": "provider default unless logged",
                "n_val_outputs": int(vote["llm_doubao_pred"].notna().sum()),
                "source_file": args.doubao_jsonl,
            },
            {
                "backend": "deepseek",
                "model_name": "DeepSeek-R1 (third backend; fill after API run)",
                "provider": "DeepSeek / configured endpoint",
                "access_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "temperature_policy": "same letter-only protocol",
                "n_val_outputs": int(vote["llm_deepseek_pred"].notna().sum()),
                "source_file": args.third_jsonl,
            },
        ]
    )
    save_table(repro, tables / "table_llm_reproducibility.csv")

    # Prompt file for missing HIGH items / third backend
    missing_third = val[~val["question_id"].isin(set(deepseek["question_id"]))] if len(deepseek) else val
    prompts = []
    for _, row in missing_third.iterrows():
        prompts.append(
            {
                "question_id": row["question_id"],
                "prompt": row["prompt"],
                "gold_answer": row["answer_letter"],
                "designer_difficulty": row["designer_difficulty_str"],
                "backend_target": "deepseek",
            }
        )
    write_jsonl(art / "llm_deepseek_val_TODO.jsonl", prompts)
    print(f"[OK] wrote {len(prompts)} DeepSeek TODO prompts")

    # Coverage gap vs official 4887
    coverage = pd.DataFrame(
        [
            {
                "official_n": 4887,
                "gpt_coverage": int(vote["llm_gpt4o_pred"].notna().sum()),
                "doubao_coverage": int(vote["llm_doubao_pred"].notna().sum()),
                "deepseek_coverage": int(vote["llm_deepseek_pred"].notna().sum()),
                "note": "Legacy GPT/Doubao logs were collected on buggy 2872 split; re-run on full 4887 HIGH items.",
            }
        ]
    )
    save_table(coverage, tables / "table_llm_coverage_gap.csv")
    print("[OK] E2 vote aggregation complete")


if __name__ == "__main__":
    main()
