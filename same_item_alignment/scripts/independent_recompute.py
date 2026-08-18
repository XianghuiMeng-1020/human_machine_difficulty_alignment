#!/usr/bin/env python3
"""Independent recomputation script (Section 22).

Reads ONLY the final integrated table `same_item_integrated_948.parquet` and
recomputes all headline results from scratch using its own code path. Does NOT
import any function from s1_item_manifest.py, s2_human_difficulty.py, or
s8_15_alignment_analysis.py. Used to cross-check that the primary analysis
pipeline's headline numbers are reproducible independently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
INTEGRATED = ROOT / "same_item_alignment/data/same_item_integrated_948.parquet"


def independent_spearman_ci(x, y, n_boot=5000, seed=99):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = stats.spearmanr(x[idx], y[idx]).correlation
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, lo, hi


def main():
    t = pd.read_parquet(INTEGRATED)
    report = {}

    report["item_count"] = int(len(t))

    report["learner_difficulty_summary"] = {
        "irt_mean": float(t.human_irt_difficulty.mean()),
        "irt_sd": float(t.human_irt_difficulty.std()),
        "irt_min": float(t.human_irt_difficulty.min()),
        "irt_max": float(t.human_irt_difficulty.max()),
        "eb_mean": float(t.human_eb_difficulty.mean()),
        "empirical_mean": float(t.human_empirical_difficulty.mean()),
    }

    solver_correct_cols = [c for c in t.columns if c.startswith("solver_") and c.endswith("_correct")]
    acc_by_solver = {}
    for c in solver_correct_cols:
        sid = t[c.replace("_correct", "_id")].dropna().iloc[0]
        vals = t[c].dropna().astype(bool)
        acc_by_solver[sid] = {"n": int(len(vals)), "accuracy": float(vals.mean())}
    report["per_solver_accuracy"] = acc_by_solver

    report["ensemble_error_distribution"] = {
        "mean": float(t.machine_error_rate.mean(skipna=True)),
        "sd": float(t.machine_error_rate.std(skipna=True)),
        "quantiles": {str(q): float(t.machine_error_rate.quantile(q)) for q in [0, 0.25, 0.5, 0.75, 1.0]},
    }

    x = t.human_irt_difficulty.values
    y = t.machine_error_rate.values
    valid = ~np.isnan(y)
    rho, lo, hi = independent_spearman_ci(x[valid], y[valid])
    report["primary_spearman"] = {"n": int(valid.sum()), "rho": rho, "ci_lower": lo, "ci_upper": hi}

    rho_eb, lo_eb, hi_eb = independent_spearman_ci(t.human_eb_difficulty.values[valid], y[valid])
    rho_emp, lo_emp, hi_emp = independent_spearman_ci(t.human_empirical_difficulty.values[valid], y[valid])
    report["robustness_spearman"] = {
        "eb": {"rho": rho_eb, "ci": [lo_eb, hi_eb]},
        "empirical": {"rho": rho_emp, "ci": [lo_emp, hi_emp]},
    }

    # regression dataset counts (long format) independently reconstructed
    n_pairs = 0
    for c in solver_correct_cols:
        n_pairs += int(t[c].notna().sum())
    report["regression_dataset_n_rows"] = n_pairs

    t2 = t.copy()
    t2["irt_decile"] = pd.qcut(t2.human_irt_difficulty, 10, labels=list(range(1, 11)))
    dec = t2.groupby("irt_decile", observed=True).agg(
        n_items=("question_id", "count"),
        machine_accuracy=("machine_error_rate", lambda s: 1 - s.mean()),
    )
    report["decile_machine_accuracy"] = dec["machine_accuracy"].round(4).to_dict()

    t2["irt_quintile"] = pd.qcut(t2.human_irt_difficulty, 5, labels=list(range(1, 6)))
    quint = t2.groupby("irt_quintile", observed=True).agg(
        n_items=("question_id", "count"),
        machine_accuracy=("machine_error_rate", lambda s: 1 - s.mean()),
    )
    report["quintile_machine_accuracy"] = quint["machine_accuracy"].round(4).to_dict()

    q_human = t2.human_irt_difficulty.quantile([0.25, 0.75])
    human_easy = t2.human_irt_difficulty <= q_human.loc[0.25]
    human_hard = t2.human_irt_difficulty >= q_human.loc[0.75]
    machine_easy = t2.machine_majority_correct == True
    machine_hard = t2.machine_majority_correct == False
    report["disagreement_group_counts"] = {
        "aligned_easy": int((human_easy & machine_easy).sum()),
        "aligned_hard": int((human_hard & machine_hard).sum()),
        "human_hard_machine_easy": int((human_hard & machine_easy).sum()),
        "human_easy_machine_hard": int((human_easy & machine_hard).sum()),
    }

    print(json.dumps(report, indent=2, default=str))
    out_path = ROOT / "same_item_alignment/audit/independent_recompute_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
