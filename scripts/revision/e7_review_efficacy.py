#!/usr/bin/env python3
"""E7: Review efficacy — offline proxy simulation (+ optional small A/B scaffold).

Uses Bridge/IRT human difficulty when available; otherwise uses EeDi IRT proxy
or a synthetic held-out correctness proxy from encoder/LLM agreement patterns
to demonstrate the evaluation harness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REVISION_ROOT, ensure_dir, save_table  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--integrated_csv",
        default=str(REVISION_ROOT / "artifacts/race_val_integrated.csv"),
    )
    ap.add_argument(
        "--eedi_irt_csv",
        default=str(REVISION_ROOT / "artifacts/eedi_irt_b_proxy.csv"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def true_difficulty_proxy(df: pd.DataFrame) -> pd.Series:
    """Construct a continuous 'true difficulty' proxy in [0,1] (1=hard).

    Preference order:
      1) human_correct_rate from bridge if present
      2) 1 - llm_correct (consensus) blended with region
      3) region ordinal
    """
    if "human_correct_rate" in df.columns and df["human_correct_rate"].notna().any():
        return 1.0 - df["human_correct_rate"].astype(float)

    region_map = {"easy": 0.2, "middle": 0.45, "ambiguous": 0.65, "hard": 0.85}
    base = df.get("datamap_region", pd.Series(["middle"] * len(df))).map(region_map).fillna(0.5)
    if "llm_correct" in df.columns:
        llm_hard = 1.0 - pd.to_numeric(df["llm_correct"], errors="coerce").fillna(0.5)
        if "llm_no_consensus" in df.columns:
            llm_hard = np.where(df["llm_no_consensus"] == 1, 0.75, llm_hard)
        base = 0.5 * base + 0.5 * llm_hard
    if "enc_correct" in df.columns:
        enc_hard = 1.0 - pd.to_numeric(df["enc_correct"], errors="coerce").fillna(0.5)
        base = 0.7 * base + 0.3 * enc_hard
    return pd.Series(base, index=df.index).astype(float)


def estimate_under_policy(df: pd.DataFrame, policy: str, rng: np.random.Generator) -> np.ndarray:
    """Return difficulty estimates in [0,1] under a labeling policy."""
    n = len(df)
    if policy == "random":
        return rng.uniform(0, 1, size=n)

    if policy == "designer_only":
        return df["designer_difficulty_str"].map({"MIDDLE": 0.35, "HIGH": 0.7}).fillna(0.5).to_numpy()

    if policy == "llm_only":
        est = np.full(n, 0.5)
        if "llm_correct" in df.columns:
            est = 1.0 - pd.to_numeric(df["llm_correct"], errors="coerce").fillna(0.5).to_numpy()
            if "llm_no_consensus" in df.columns:
                est = np.where(df["llm_no_consensus"].to_numpy() == 1, 0.7, est)
        return est

    if policy == "encoder_only":
        region_map = {"easy": 0.2, "middle": 0.45, "ambiguous": 0.65, "hard": 0.85}
        return (
            df.get("datamap_region", pd.Series(["middle"] * n))
            .map(region_map)
            .fillna(0.5)
            .to_numpy()
        )

    raise ValueError(policy)


def disagreement_score_row(row) -> float:
    score = 0.0
    region = row.get("datamap_region")
    band = row.get("designer_difficulty_str")
    if band == "HIGH" and region == "easy":
        score += 2
    if band == "MIDDLE" and region == "hard":
        score += 2
    if row.get("llm_no_consensus") == 0 and row.get("llm_correct") == 0:
        score += 2
    if row.get("llm_no_consensus") == 1:
        score += 1.5
    if row.get("enc_correct") == 0:
        score += 1
    return score


def policy_disagreement_review(df: pd.DataFrame, true_diff: np.ndarray, top_frac: float = 0.2) -> np.ndarray:
    scores = df.apply(disagreement_score_row, axis=1).to_numpy()
    n_review = max(1, int(round(len(df) * top_frac)))
    # Take exact top-fraction by rank (stable under ties)
    order = np.argsort(-scores, kind="mergesort")
    reviewed = np.zeros(len(df), dtype=bool)
    reviewed[order[:n_review]] = True
    designer = df["designer_difficulty_str"].map({"MIDDLE": 0.35, "HIGH": 0.7}).fillna(0.5).to_numpy()
    est = designer.copy()
    est[reviewed] = true_diff[reviewed]
    return est, reviewed


def metrics(y_true, y_est):
    y_true = np.asarray(y_true, dtype=float)
    y_est = np.asarray(y_est, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_est)))
    rmse = float(np.sqrt(np.mean((y_true - y_est) ** 2)))
    # Mislabel rate under tercile hard/easy
    true_hard = y_true >= np.quantile(y_true, 0.67)
    est_hard = y_est >= np.quantile(y_est, 0.67)
    mislabel = float(np.mean(true_hard != est_hard))
    return {"mae": mae, "rmse": rmse, "hard_mislabel_rate": mislabel}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    efficacy = ensure_dir(out_dir / "efficacy")
    tables = ensure_dir(out_dir / "tables")
    df = pd.read_csv(args.integrated_csv)
    rng = np.random.default_rng(args.seed)

    # Prefer bridge human rates if present
    bridge_rates = out_dir / "bridge/bridge_race_human_rates.csv"
    if bridge_rates.is_file():
        br = pd.read_csv(bridge_rates)
        df = df.merge(br[["question_id", "human_correct_rate"]], on="question_id", how="left")

    true_diff = true_difficulty_proxy(df).to_numpy()
    df["true_difficulty_proxy"] = true_diff

    results = []
    for policy in ["random", "designer_only", "llm_only", "encoder_only"]:
        if policy == "llm_only" and "llm_correct" not in df.columns:
            continue
        if policy == "encoder_only" and "datamap_region" not in df.columns:
            continue
        if policy == "random":
            est = rng.uniform(0, 1, size=len(df))
        elif policy == "designer_only":
            est = df["designer_difficulty_str"].map({"MIDDLE": 0.35, "HIGH": 0.7}).fillna(0.5).to_numpy()
        elif policy == "llm_only":
            est = 1.0 - pd.to_numeric(df["llm_correct"], errors="coerce").fillna(0.5).to_numpy()
            if "llm_no_consensus" in df.columns:
                est = np.where(df["llm_no_consensus"].to_numpy() == 1, 0.7, est)
        else:
            region_map = {"easy": 0.2, "middle": 0.45, "ambiguous": 0.65, "hard": 0.85}
            est = df["datamap_region"].map(region_map).fillna(0.5).to_numpy()
        m = metrics(true_diff, est)
        results.append({"policy": policy, "n_reviewed": 0, **m})

    est_rev, reviewed = policy_disagreement_review(df, true_diff, top_frac=0.2)
    m = metrics(true_diff, est_rev)
    results.append({"policy": "disagreement_review_top20pct", "n_reviewed": int(reviewed.sum()), **m})

    res_df = pd.DataFrame(results).sort_values("mae")
    save_table(res_df, tables / "table_e7a_offline_policy_metrics.csv")
    save_table(df[["question_id", "true_difficulty_proxy"]], efficacy / "e7_true_difficulty_proxy.csv")

    # Next-item matching proxy: rank correlation to true difficulty
    def _spearman(a, b):
        a = pd.Series(a).rank()
        b = pd.Series(b).rank()
        return float(a.corr(b))

    rank_rows = []
    for _, row in res_df.iterrows():
        policy = row["policy"]
        if policy == "random":
            est = rng.uniform(0, 1, size=len(df))
        elif policy == "designer_only":
            est = df["designer_difficulty_str"].map({"MIDDLE": 0.35, "HIGH": 0.7}).fillna(0.5).to_numpy()
        elif policy == "llm_only":
            est = 1.0 - pd.to_numeric(df.get("llm_correct"), errors="coerce").fillna(0.5).to_numpy()
        elif policy == "encoder_only":
            region_map = {"easy": 0.2, "middle": 0.45, "ambiguous": 0.65, "hard": 0.85}
            est = df.get("datamap_region", pd.Series(["middle"] * len(df))).map(region_map).fillna(0.5).to_numpy()
        else:
            est = est_rev
        rank_rows.append({"policy": policy, "spearman_vs_true": _spearman(true_diff, est)})
    save_table(pd.DataFrame(rank_rows), tables / "table_e7a_rank_correlation.csv")

    # E7b small A/B protocol scaffold
    ab = {
        "design": "between-subjects lab study",
        "arms": [
            "no_review: practice queue ordered by designer grade-band only",
            "review: high-disagreement items human-reviewed then reinserted with corrected labels",
        ],
        "n_participants_target": 60,
        "session": "40-minute reading comprehension practice + post-test",
        "outcomes": [
            "post-test accuracy",
            "absolute error of mastery estimate vs held-out performance",
            "time-on-task",
        ],
        "analysis": "ANCOVA with pretest covariate; report effect size",
        "status": "protocol_ready_awaiting_IRB_and_recruitment",
    }
    (efficacy / "e7b_lab_ab_protocol.json").write_text(json.dumps(ab, indent=2), encoding="utf-8")
    save_table(
        pd.DataFrame([{"status": ab["status"], "n_target": ab["n_participants_target"]}]),
        tables / "table_e7b_ab_status.csv",
    )
    print("[OK] E7 offline simulation complete")


if __name__ == "__main__":
    main()
