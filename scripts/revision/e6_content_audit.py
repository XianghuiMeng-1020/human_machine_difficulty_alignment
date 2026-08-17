#!/usr/bin/env python3
"""E6: Content-validity audit sampling (30 high-disagreement + 30 low-disagreement)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REVISION_ROOT, bootstrap_kappa, ensure_dir, save_table  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--integrated_csv",
        default=str(REVISION_ROOT / "artifacts/race_val_integrated.csv"),
    )
    ap.add_argument("--n_per_arm", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def disagreement_score(row) -> float:
    """Higher = more multi-source disagreement (pre-registered rule)."""
    score = 0.0
    # Designer HIGH but encoder easy (or reverse)
    region = row.get("datamap_region")
    band = row.get("designer_difficulty_str")
    if band == "HIGH" and region == "easy":
        score += 2
    if band == "MIDDLE" and region == "hard":
        score += 2
    if band == "HIGH" and region == "ambiguous":
        score += 1
    if band == "MIDDLE" and region == "ambiguous":
        score += 0.5
    # LLM incorrect under consensus
    if row.get("llm_no_consensus") == 0 and row.get("llm_correct") == 0:
        score += 2
    if row.get("llm_no_consensus") == 1:
        score += 1.5
    # Encoder incorrect
    if row.get("enc_correct") == 0:
        score += 1
    return score


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    audit = ensure_dir(out_dir / "audit")
    tables = ensure_dir(out_dir / "tables")
    df = pd.read_csv(args.integrated_csv)
    df["disagreement_score"] = df.apply(disagreement_score, axis=1)

    # High = top scores; Low = score==0 pool
    high_pool = df.sort_values("disagreement_score", ascending=False)
    low_pool = df[df["disagreement_score"] == 0]
    if len(low_pool) < args.n_per_arm:
        low_pool = df.nsmallest(max(args.n_per_arm * 3, args.n_per_arm), "disagreement_score")

    rng = np.random.default_rng(args.seed)
    high = high_pool.head(max(args.n_per_arm * 3, args.n_per_arm)).sample(
        n=min(args.n_per_arm, len(high_pool)), random_state=args.seed
    )
    low = low_pool.sample(n=min(args.n_per_arm, len(low_pool)), random_state=args.seed)
    high = high.copy()
    low = low.copy()
    high["audit_arm"] = "high_disagreement"
    low["audit_arm"] = "low_disagreement"
    sample = pd.concat([high, low], ignore_index=True)

    # Blind packs (hide arm + model labels from raters)
    public_cols = [
        c
        for c in [
            "question_id",
            "article",
            "question",
            "option_a",
            "option_b",
            "option_c",
            "option_d",
            "answer_letter",
        ]
        if c in sample.columns
    ]
    blind = sample[public_cols].copy()
    blind = blind.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    blind.insert(0, "item_order", np.arange(1, len(blind) + 1))
    save_table(blind, audit / "e6_blind_items_for_raters.csv")

    key = sample[
        ["question_id", "audit_arm", "disagreement_score"]
        + [c for c in ["designer_difficulty_str", "datamap_region", "llm_correct", "llm_no_consensus", "enc_correct"] if c in sample.columns]
    ].copy()
    save_table(key, audit / "e6_arm_key_HIDDEN.csv")

    rubric = {
        "flaw_codes": [
            "ambiguous_key",
            "flawed_distractors",
            "evidence_not_locatable_in_passage",
            "multiple_plausible_answers",
            "other_item_flaw",
            "no_flaw",
        ],
        "procedure": [
            "Two or more raters independently code each item blind to audit_arm",
            "Primary outcome: any_flaw = 1 if any flaw code except no_flaw",
            "Compare flaw prevalence high vs low (Fisher exact / chi-square)",
            "Report Cohen kappa among raters on any_flaw",
        ],
    }
    (audit / "e6_coding_rubric.json").write_text(json.dumps(rubric, indent=2), encoding="utf-8")

    rating_template = pd.DataFrame(
        {
            "item_order": blind["item_order"],
            "question_id": blind["question_id"],
            "rater_id": "",
            "ambiguous_key": "",
            "flawed_distractors": "",
            "evidence_not_locatable_in_passage": "",
            "multiple_plausible_answers": "",
            "other_item_flaw": "",
            "no_flaw": "",
            "notes": "",
        }
    )
    save_table(rating_template, audit / "e6_ratings_TEMPLATE.csv")

    # If ratings already collected, analyze
    ratings_path = audit / "e6_ratings.csv"
    if ratings_path.is_file():
        ratings = pd.read_csv(ratings_path)
        flaw_cols = [
            "ambiguous_key",
            "flawed_distractors",
            "evidence_not_locatable_in_passage",
            "multiple_plausible_answers",
            "other_item_flaw",
        ]
        for c in flaw_cols:
            if c in ratings.columns:
                ratings[c] = pd.to_numeric(ratings[c], errors="coerce").fillna(0)
        ratings["any_flaw"] = ratings[flaw_cols].max(axis=1) if set(flaw_cols) <= set(ratings.columns) else np.nan
        merged = ratings.merge(key, on="question_id", how="left")
        # Majority vote across raters per item
        item = (
            merged.groupby(["question_id", "audit_arm"])["any_flaw"]
            .mean()
            .reset_index()
        )
        item["any_flaw_bin"] = (item["any_flaw"] >= 0.5).astype(int)
        tab = pd.crosstab(item["audit_arm"], item["any_flaw_bin"])
        save_table(tab.reset_index(), tables / "table_e6_flaw_by_arm.csv")
        # Inter-rater kappa if 2 raters
        if "rater_id" in merged.columns and merged["rater_id"].nunique() >= 2:
            raters = sorted(merged["rater_id"].dropna().unique())[:2]
            wide = merged[merged.rater_id.isin(raters)].pivot_table(
                index="question_id", columns="rater_id", values="any_flaw", aggfunc="max"
            )
            if wide.shape[1] >= 2:
                k = bootstrap_kappa(wide.iloc[:, 0], wide.iloc[:, 1])
                save_table(pd.DataFrame([{**k, "rater_a": raters[0], "rater_b": raters[1]}]), tables / "table_e6_interrater_kappa.csv")
    else:
        save_table(
            pd.DataFrame(
                [
                    {
                        "status": "awaiting_ratings",
                        "n_high": int((sample.audit_arm == "high_disagreement").sum()),
                        "n_low": int((sample.audit_arm == "low_disagreement").sum()),
                        "mean_score_high": float(high.disagreement_score.mean()),
                        "mean_score_low": float(low.disagreement_score.mean()),
                    }
                ]
            ),
            tables / "table_e6_sample_ready.csv",
        )

    # Pre-register disagreement rule
    (audit / "e6_disagreement_rule.json").write_text(
        json.dumps(
            {
                "rule": "disagreement_score additive components as in e6_content_audit.disagreement_score",
                "high_arm": "sample from highest scores",
                "low_arm": "sample from score==0 when available",
                "n_per_arm": args.n_per_arm,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] E6 sample ready: {len(sample)} items")


if __name__ == "__main__":
    main()
