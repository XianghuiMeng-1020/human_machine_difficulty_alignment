#!/usr/bin/env python3
"""Independent stats: band×region, LLM×region, tercile sensitivity, grade-band inversion model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EVID = Path(__file__).resolve().parents[1] / "evidence"
EVID.mkdir(parents=True, exist_ok=True)


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table.values)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r - 1, k - 1))))) if n and min(r, k) > 1 else float("nan")


def assign_regions(df: pd.DataFrame, mu_lo, mu_hi, sig_lo, sig_hi):
    """Precedence: ambiguous > easy > hard > middle (matches manuscript rule)."""
    labels = []
    for _, r in df.iterrows():
        mu, sig, fc = r["mean_prob"], r["std_prob"], r["frac_correct"]
        if sig >= sig_hi:
            labels.append("ambiguous")
        elif mu >= mu_hi and sig <= sig_lo:
            labels.append("easy")
        elif mu <= mu_lo and fc < 0.5:
            labels.append("hard")
        else:
            labels.append("middle")
    return labels


def main():
    path = ROOT / "revision/artifacts/race_val_integrated.csv"
    df = pd.read_csv(path)
    out = {}

    # grade band × region
    ct = pd.crosstab(df["designer_difficulty_str"], df["datamap_region"])
    ct.to_csv(EVID / "band_x_region_contingency.csv")
    chi2, p, dof, exp = stats.chi2_contingency(ct.values)
    out["band_x_region"] = {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "cramers_v": cramers_v(ct),
        "table": ct.to_dict(),
    }

    # LLM incorrect × region (consensus only)
    cons = df[df["llm_no_consensus"].fillna(0).astype(int) == 0].copy()
    cons["llm_incorrect"] = (~cons["llm_correct"].astype(bool)).astype(int)
    ct2 = pd.crosstab(cons["llm_incorrect"], cons["datamap_region"])
    ct2.to_csv(EVID / "llm_incorrect_x_region.csv")
    chi2b, pb, dofb, _ = stats.chi2_contingency(ct2.values)
    out["llm_incorrect_x_region"] = {"chi2": float(chi2b), "p": float(pb), "dof": int(dofb), "cramers_v": cramers_v(ct2)}

    # grade-band accuracy inversion check
    enc_by = df.groupby("designer_difficulty_str")["enc_correct"].agg(["mean", "count"])
    llm_by = cons.groupby("designer_difficulty_str")["llm_correct"].agg(["mean", "count"])
    out["encoder_acc_by_band"] = enc_by.to_dict()
    out["llm_acc_by_band_consensus"] = llm_by.to_dict()
    out["encoder_middle_gt_high"] = bool(
        enc_by.loc["MIDDLE", "mean"] > enc_by.loc["HIGH", "mean"]
    ) if set(["MIDDLE", "HIGH"]).issubset(enc_by.index) else None

    # truncation by band if present
    if "likely_truncated_2048" in df.columns:
        out["truncation_by_band"] = df.groupby("designer_difficulty_str")["likely_truncated_2048"].mean().to_dict()
    if "passage_approx_tokens" in df.columns:
        out["passage_tokens_by_band"] = df.groupby("designer_difficulty_str")["passage_approx_tokens"].describe().to_dict()

    # simple logistic: enc_error ~ band + length + trunc + mean_prob + std_prob
    work = df.dropna(subset=["enc_correct", "mean_prob", "std_prob"]).copy()
    work["enc_error"] = 1 - work["enc_correct"].astype(int)
    work["band_high"] = (work["designer_difficulty_str"] == "HIGH").astype(int)
    work["len_z"] = (work.get("passage_approx_tokens", pd.Series(0, index=work.index)).astype(float))
    if work["len_z"].std() > 0:
        work["len_z"] = (work["len_z"] - work["len_z"].mean()) / work["len_z"].std()
    work["trunc"] = work["likely_truncated_2048"].fillna(0).astype(int) if "likely_truncated_2048" in work.columns else 0
    # use statsmodels if available else skip
    try:
        import statsmodels.api as sm

        X = sm.add_constant(work[["band_high", "len_z", "trunc", "mean_prob", "std_prob"]])
        model = sm.Logit(work["enc_error"], X).fit(disp=False)
        out["logit_enc_error"] = {
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "conf_int": model.conf_int().to_dict(),
            "pseudo_r2": float(model.prsquared),
            "n": int(model.nobs),
        }
    except Exception as e:
        out["logit_enc_error_error"] = str(e)

    # tercile sensitivity
    sens_rows = []
    base = df.dropna(subset=["mean_prob", "std_prob", "frac_correct"]).copy()
    specs = {
        "p20_80": (0.20, 0.80),
        "p25_75": (0.25, 0.75),
        "p33_67": (0.33, 0.67),
    }
    label_sets = {}
    for name, (lo, hi) in specs.items():
        mu_lo, mu_hi = base["mean_prob"].quantile([lo, hi])
        sig_lo, sig_hi = base["std_prob"].quantile([lo, hi])
        labs = assign_regions(base, mu_lo, mu_hi, sig_lo, sig_hi)
        label_sets[name] = pd.Series(labs, index=base.index)
        tmp = base.copy()
        tmp["region_alt"] = labs
        ct = pd.crosstab(tmp["designer_difficulty_str"], tmp["region_alt"])
        # LLM error rate by region
        tmp2 = tmp.merge(cons[["question_id", "llm_incorrect"]], on="question_id", how="left")
        err = tmp2.groupby("region_alt")["llm_incorrect"].mean().to_dict()
        sens_rows.append(
            {
                "spec": name,
                "mu_lo": float(mu_lo),
                "mu_hi": float(mu_hi),
                "sig_lo": float(sig_lo),
                "sig_hi": float(sig_hi),
                "cramers_v_band": cramers_v(ct),
                "region_counts": json.dumps(tmp["region_alt"].value_counts().to_dict()),
                "llm_incorrect_by_region": json.dumps({k: float(v) if v == v else None for k, v in err.items()}),
            }
        )
    # switch rates vs manuscript region
    for name, ser in label_sets.items():
        aligned = base.join(df.set_index("question_id")["datamap_region"], on="question_id", rsuffix="_ms")
        # simpler: compare on index
        ms = df.loc[base.index, "datamap_region"].astype(str)
        switch = float((ser.astype(str).values != ms.values).mean())
        for row in sens_rows:
            if row["spec"] == name:
                row["switch_rate_vs_manuscript_region"] = switch
    pd.DataFrame(sens_rows).to_csv(EVID / "tercile_sensitivity_independent.csv", index=False)
    out["sensitivity"] = sens_rows

    # continuous association without bins: point-biserial HIGH vs mean_prob
    y = (df["designer_difficulty_str"] == "HIGH").astype(int)
    r_pb, p_pb = stats.pointbiserialr(y, df["mean_prob"].fillna(df["mean_prob"].median()))
    out["pointbiserial_high_vs_mean_prob"] = {"r": float(r_pb), "p": float(p_pb)}

    # no-consensus by band/region
    out["nocon_by_band"] = df.groupby("designer_difficulty_str")["llm_no_consensus"].mean().astype(float).to_dict()
    out["nocon_by_region"] = df.groupby("datamap_region")["llm_no_consensus"].mean().astype(float).to_dict()

    (EVID / "stats_sensitivity_audit.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["band_x_region", "encoder_middle_gt_high", "encoder_acc_by_band"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
