#!/usr/bin/env python3
"""E8: Open-ended / constructed-response provenance pilot.

Builds a 50–100 item pilot protocol and, when a public short-answer bank is
unavailable, synthesizes a structured pilot frame from RACE stems converted to
short-answer form (remove options; answer = gold option text) for process demo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, REVISION_ROOT, ensure_dir, save_table, write_jsonl  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--race_val_csv",
        default=str(REPO_ROOT / "race_prepared/race_mcq_val.csv"),
    )
    ap.add_argument("--n_items", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    od = ensure_dir(out_dir / "open_ended")
    tables = ensure_dir(out_dir / "tables")

    val = pd.read_csv(args.race_val_csv)
    sample = (
        val.groupby("designer_difficulty_str", group_keys=False)
        .apply(lambda g: g.sample(n=min(args.n_items // 2, len(g)), random_state=args.seed))
        .reset_index(drop=True)
    )
    if len(sample) < args.n_items:
        extra = val[~val.question_id.isin(sample.question_id)].sample(
            n=min(args.n_items - len(sample), len(val)), random_state=args.seed
        )
        sample = pd.concat([sample, extra], ignore_index=True)

    rows = []
    for _, row in sample.iterrows():
        gold_text = {
            "A": row["option_a"],
            "B": row["option_b"],
            "C": row["option_c"],
            "D": row["option_d"],
        }[row["answer_letter"]]
        rows.append(
            {
                "item_id": row["question_id"],
                "format": "short_answer_from_mcq_stem",
                "passage": row["article"],
                "question": row["question"]
                + "\n\nWrite a short answer based on the passage (do not choose A/B/C/D).",
                "reference_answer": gold_text,
                "designer_band": row["designer_difficulty_str"],
                "rubric": {
                    "correct": "answer semantically matches reference evidence in passage",
                    "partial": "related but missing key evidence",
                    "incorrect": "contradicts passage or unsupported",
                },
            }
        )

    write_jsonl(od / "e8_open_ended_pilot_items.jsonl", rows)
    save_table(pd.DataFrame(rows).drop(columns=["rubric"]), od / "e8_open_ended_pilot_items.csv")

    # Human scoring template
    score_tmpl = pd.DataFrame(
        {
            "item_id": [r["item_id"] for r in rows],
            "rater_id": "",
            "score_0_1_2": "",
            "notes": "",
        }
    )
    save_table(score_tmpl, od / "e8_human_scores_TEMPLATE.csv")

    # LLM scoring prompt pack
    llm_prompts = []
    for r in rows:
        llm_prompts.append(
            {
                "item_id": r["item_id"],
                "prompt": (
                    "Grade the student short answer using the rubric 0/1/2.\n"
                    f"Passage:\n{r['passage'][:2000]}\n\n"
                    f"Question:\n{r['question']}\n\n"
                    f"Reference answer:\n{r['reference_answer']}\n\n"
                    "Student answer:\n{{STUDENT_ANSWER}}\n\n"
                    "Return only an integer 0, 1, or 2."
                ),
            }
        )
    write_jsonl(od / "e8_llm_grader_prompts.jsonl", llm_prompts)

    protocol = {
        "n_items": len(rows),
        "human_record": "rubric score aggregated over raters -> correctness rate / difficulty bucket",
        "machine_record": "LLM grader score + multi-backend agreement",
        "provenance": "store source=human_rubric|llm_grader and rule version",
        "analysis": [
            "item-level agreement (kappa) human vs LLM grader",
            "flag high-disagreement items for review",
            "demonstrate provenance fields for non-MCQ",
        ],
        "limitations": "Pilot converts MCQ stems to short-answer; not a full open-ended corpus",
    }
    (od / "PROTOCOL_e8.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # Simulated demo scores to prove analysis pipeline (replace with real ratings)
    rng = np.random.default_rng(args.seed)
    demo = []
    for r in rows:
        human = int(rng.integers(0, 3))
        llm = human if rng.random() > 0.35 else int(rng.integers(0, 3))
        demo.append(
            {
                "item_id": r["item_id"],
                "human_score": human,
                "llm_score": llm,
                "agree": int(human == llm),
                "designer_band": r["designer_band"],
            }
        )
    demo_df = pd.DataFrame(demo)
    save_table(demo_df, od / "e8_demo_scores_SIMULATED.csv")
    summary = pd.DataFrame(
        [
            {
                "n_items": len(demo_df),
                "exact_agreement": float(demo_df["agree"].mean()),
                "note": "SIMULATED for pipeline check; replace with real human/LLM grades",
            }
        ]
    )
    save_table(summary, tables / "table_e8_open_ended_pilot_summary.csv")
    print(f"[OK] E8 pilot pack n={len(rows)}")


if __name__ == "__main__":
    main()
