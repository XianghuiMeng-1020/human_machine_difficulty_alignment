#!/usr/bin/env python3
"""E5: EeDi difficulty label reliability — attempt filters, shrinkage, sensitivity."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, REVISION_ROOT, ensure_dir, save_table  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[str(REPO_ROOT / "data/eedi/train_data/train_task_3_4.csv")],
    )
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def load_attempts(paths) -> pd.DataFrame:
    dfs = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"[WARN] missing {path}")
            continue
        df = pd.read_csv(path)
        rename = {}
        if "QuestionId" in df.columns:
            rename["QuestionId"] = "question_id"
        if "IsCorrect" in df.columns:
            rename["IsCorrect"] = "is_correct"
        if "UserId" in df.columns:
            rename["UserId"] = "student_id"
        df = df.rename(columns=rename)
        dfs.append(df[["question_id", "is_correct"]].assign(is_correct=lambda x: x["is_correct"].astype(int)))
    if not dfs:
        raise FileNotFoundError("No EeDi attempt files found")
    return pd.concat(dfs, ignore_index=True)


def empirical_bayes_rate(n_correct, n_attempts, alpha: float, beta: float):
    return (n_correct + alpha) / (n_attempts + alpha + beta)


def bucket(rate, easy_thr=0.8, hard_thr=0.4):
    if rate >= easy_thr:
        return "Human Easy"
    if rate <= hard_thr:
        return "Human Hard"
    return "Human Mid"


def analyze(df: pd.DataFrame, tables: Path, artifacts: Path) -> None:
    agg = (
        df.groupby("question_id")["is_correct"]
        .agg(n_attempts="count", n_correct="sum", mean_correct="mean")
        .reset_index()
    )
    save_table(agg, artifacts / "eedi_attempt_distribution_raw.csv")

    # Attempt-count distribution summary
    dist = pd.DataFrame(
        [
            {
                "n_questions": len(agg),
                "min_attempts": int(agg.n_attempts.min()),
                "p10": float(np.quantile(agg.n_attempts, 0.1)),
                "median": float(np.median(agg.n_attempts)),
                "p90": float(np.quantile(agg.n_attempts, 0.9)),
                "mean": float(agg.n_attempts.mean()),
                "max": int(agg.n_attempts.max()),
            }
        ]
    )
    save_table(dist, tables / "table_e5_attempt_count_distribution.csv")

    # Global Beta prior from method-of-moments on raw rates (for shrinkage)
    mu = float(agg["mean_correct"].mean())
    var = float(agg["mean_correct"].var())
    if var > 0 and var < mu * (1 - mu):
        common = mu * (1 - mu) / var - 1
        alpha = max(mu * common, 1e-3)
        beta = max((1 - mu) * common, 1e-3)
    else:
        alpha, beta = 1.0, 1.0

    rows = []
    for min_att in [5, 15, 30, 50]:
        for easy_thr, hard_thr in [(0.80, 0.40), (0.75, 0.35), (0.85, 0.45)]:
            sub = agg[agg.n_attempts >= min_att].copy()
            sub["raw_bucket"] = [
                bucket(r, easy_thr, hard_thr) for r in sub["mean_correct"]
            ]
            sub["shrunk_rate"] = empirical_bayes_rate(
                sub["n_correct"], sub["n_attempts"], alpha, beta
            )
            sub["shrunk_bucket"] = [
                bucket(r, easy_thr, hard_thr) for r in sub["shrunk_rate"]
            ]
            for mode, col in [("raw", "raw_bucket"), ("shrunk", "shrunk_bucket")]:
                vc = sub[col].value_counts()
                rows.append(
                    {
                        "min_attempts": min_att,
                        "easy_thr": easy_thr,
                        "hard_thr": hard_thr,
                        "mode": mode,
                        "n_questions": len(sub),
                        "n_easy": int(vc.get("Human Easy", 0)),
                        "n_mid": int(vc.get("Human Mid", 0)),
                        "n_hard": int(vc.get("Human Hard", 0)),
                        "share_easy": float(vc.get("Human Easy", 0) / len(sub)) if len(sub) else np.nan,
                        "share_mid": float(vc.get("Human Mid", 0) / len(sub)) if len(sub) else np.nan,
                        "share_hard": float(vc.get("Human Hard", 0) / len(sub)) if len(sub) else np.nan,
                        "alpha": alpha,
                        "beta": beta,
                    }
                )
            if min_att == 30 and easy_thr == 0.8:
                save_table(sub, artifacts / "eedi_difficulty_min30_main.csv")

    sens = pd.DataFrame(rows)
    save_table(sens, tables / "table_e5_threshold_sensitivity.csv")

    # Stability: share_hard under min_attempts sweep with fixed thresholds
    hard_sweep = sens[
        np.isclose(sens["easy_thr"], 0.8)
        & np.isclose(sens["hard_thr"], 0.4)
        & (sens["mode"] == "raw")
    ][["min_attempts", "n_questions", "share_hard", "share_mid", "share_easy"]]
    save_table(hard_sweep, tables / "table_e5_min_attempts_sweep.csv")

    # Optional simple 1PL-ish difficulty proxy: logit(shrunk_rate)
    main = agg[agg.n_attempts >= 30].copy()
    main["shrunk_rate"] = empirical_bayes_rate(main["n_correct"], main["n_attempts"], alpha, beta)
    eps = 1e-3
    p = main["shrunk_rate"].clip(eps, 1 - eps)
    main["irt_b_proxy"] = -np.log(p / (1 - p))  # higher => harder
    save_table(
        main[["question_id", "n_attempts", "mean_correct", "shrunk_rate", "irt_b_proxy"]],
        artifacts / "eedi_irt_b_proxy.csv",
    )
    corr = float(np.corrcoef(main["mean_correct"], main["irt_b_proxy"])[0, 1])
    save_table(
        pd.DataFrame([{"corr_raw_rate_vs_irt_b_proxy": corr, "n": len(main)}]),
        tables / "table_e5_irt_proxy_correlation.csv",
    )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables = ensure_dir(out_dir / "tables")
    artifacts = ensure_dir(out_dir / "artifacts")
    df = load_attempts(args.inputs)
    print(f"[INFO] loaded {len(df)} attempts")
    analyze(df, tables, artifacts)
    print("[OK] E5 complete")


if __name__ == "__main__":
    main()
