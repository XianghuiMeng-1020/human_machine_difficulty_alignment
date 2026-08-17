#!/usr/bin/env python3
"""P0-3: BigBird core-claim check from existing dynamics + seed-run launcher/status."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT_ARCH = ROOT / "outputs/encoder/architecture_check"
OUT_SEED = ROOT / "outputs/encoder/seed_runs"
OUT_ARCH.mkdir(parents=True, exist_ok=True)
OUT_SEED.mkdir(parents=True, exist_ok=True)


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table.values)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1)))) if n and min(r, k) > 1 else float("nan")


def assign_regions(df, mu_lo, mu_hi, sig_lo, sig_hi):
    labs = []
    for _, r in df.iterrows():
        if r.std_prob >= sig_hi:
            labs.append("ambiguous")
        elif r.mean_prob >= mu_hi and r.std_prob <= sig_lo:
            labs.append("easy")
        elif r.mean_prob <= mu_lo and r.frac_correct < 0.5:
            labs.append("hard")
        else:
            labs.append("middle")
    return labs


def bigbird_core_check():
    td = pd.read_csv(
        ROOT / "revision/artifacts/encoder_competitive/google_bigbird-roberta-base/training_dynamics_val.csv"
    )
    integ = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv")
    agg = (
        td.groupby("question_id")
        .agg(
            mean_prob=("prob_correct", "mean"),
            std_prob=("prob_correct", "std"),
            frac_correct=("is_correct", "mean"),
        )
        .reset_index()
    )
    df = agg.merge(
        integ[["question_id", "designer_difficulty_str", "llm_no_consensus", "llm_correct"]],
        on="question_id",
        how="inner",
    )
    mu_lo, mu_hi = df.mean_prob.quantile([0.33, 0.67])
    sig_lo, sig_hi = df.std_prob.quantile([0.33, 0.67])
    df["region"] = assign_regions(df, mu_lo, mu_hi, sig_lo, sig_hi)
    # also quartile / 40-60
    sens = []
    for name, lo, hi in [("tercile", 0.33, 0.67), ("quartile", 0.25, 0.75), ("p40_60", 0.40, 0.60)]:
        mlo, mhi = df.mean_prob.quantile([lo, hi])
        slo, shi = df.std_prob.quantile([lo, hi])
        labs = assign_regions(df, mlo, mhi, slo, shi)
        tmp = df.copy()
        tmp["region_alt"] = labs
        ct = pd.crosstab(tmp.designer_difficulty_str, tmp.region_alt)
        cons = tmp[tmp.llm_no_consensus.fillna(0).astype(int) == 0]
        cons = cons.assign(llm_incorrect=(~cons.llm_correct.astype(bool)).astype(int))
        ct2 = pd.crosstab(cons.llm_incorrect, cons.region_alt)
        sens.append(
            {
                "spec": name,
                "cramers_v_band_region": cramers_v(ct),
                "cramers_v_llm_incorrect_region": cramers_v(ct2),
                "region_counts": json.dumps(tmp.region_alt.value_counts().to_dict()),
            }
        )
    # stability vs longformer regions
    df = df.merge(integ[["question_id", "datamap_region"]], on="question_id")
    switch = float((df.region != df.datamap_region).mean())
    ct_band = pd.crosstab(df.designer_difficulty_str, df.region)
    ct_band.to_csv(OUT_ARCH / "bigbird_band_x_region.csv")
    pd.DataFrame(sens).to_csv(OUT_ARCH / "bigbird_threshold_sensitivity.csv", index=False)
    # final accuracy by band for BigBird from last-epoch correctness mean ~= not final pred; use frac at epoch4
    ep4 = td[td.epoch == td.epoch.max()].merge(integ[["question_id", "designer_difficulty_str"]], on="question_id")
    acc_band = ep4.groupby("designer_difficulty_str")["is_correct"].mean().to_dict()
    out = {
        "n": int(len(df)),
        "switch_rate_vs_longformer_region": switch,
        "band_x_region_cramers_v": cramers_v(ct_band),
        "acc_by_band_epoch_last": acc_band,
        "construct": "held-out confidence/generalization dynamics (not original train Cartography)",
    }
    (OUT_ARCH / "bigbird_core_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # region stability table
    pd.DataFrame(
        [{"item_agreement_with_longformer": 1 - switch, "switch_rate": switch, "n": len(df)}]
    ).to_csv(ROOT / "outputs/encoder/region_stability.csv", index=False)
    return out


def seed_status_and_launch(launch: bool = False):
    """Check existing seed runs; optionally launch missing seeds via e1_train_mc."""
    summary_rows = []
    for seed in [0, 1, 2]:
        d = OUT_SEED / f"longformer_seed{seed}"
        meta = d / "run_meta.json"
        row = {"seed": seed, "dir": str(d), "complete": meta.is_file()}
        if meta.is_file():
            obj = json.loads(meta.read_text(encoding="utf-8"))
            row["val_accuracy"] = obj.get("val_accuracy")
        summary_rows.append(row)
    # also register legacy seed-unknown run
    legacy = ROOT / "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/run_meta.json"
    if legacy.is_file():
        obj = json.loads(legacy.read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "seed": None,
                "dir": str(legacy.parent),
                "complete": True,
                "val_accuracy": obj.get("val_accuracy"),
                "note": "historical run; seed not recorded in run_meta",
            }
        )
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(ROOT / "outputs/encoder/seed_summary.csv", index=False)

    if launch:
        for seed in [0, 1, 2]:
            d = OUT_SEED / f"longformer_seed{seed}"
            if (d / "run_meta.json").is_file():
                continue
            d.mkdir(parents=True, exist_ok=True)
            # Primary competitive protocol (matches historical Longformer 74.1% run)
            cmd = [
                "python",
                "scripts/revision/e1_train_mc.py",
                "--model_name",
                "allenai/longformer-base-4096",
                "--seed",
                str(seed),
                "--out_dir",
                str(d),
                "--max_len",
                "1024",
                "--article_words",
                "400",
                "--epochs",
                "4",
                "--lr",
                "2e-5",
                "--batch_size",
                "2",
                "--grad_accum",
                "8",
            ]
            (d / "launch_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
    return sdf


def main():
    bb = bigbird_core_check()
    seeds = seed_status_and_launch(launch=True)
    print(json.dumps({"bigbird": bb, "seeds": seeds.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
