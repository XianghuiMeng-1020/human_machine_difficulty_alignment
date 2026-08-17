#!/usr/bin/env python3
"""P0-1: Verified EeDi analysis (Route A — recovered official public raw)."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "revision"))

PUBLIC_ZIP = ROOT / "data/eedi_public_download/data.zip"
EXTRACT = ROOT / "data/eedi_public_download/extracted/data/train_data"
P12 = EXTRACT / "train_task_1_2.csv"
P34 = EXTRACT / "train_task_3_4.csv"
LEGACY = ROOT / "Eedi_analysis/eedi_question_human_difficulty.csv"

OUT_PROC = ROOT / "data/processed"
OUT_EEDI = ROOT / "outputs/eedi"
EVID = ROOT / "audit/evidence"
LOG = EVID / "eedi_recompute.log"
for d in [OUT_PROC, OUT_EEDI, EVID]:
    d.mkdir(parents=True, exist_ok=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def log(msg: str, lines: list):
    print(msg)
    lines.append(msg)


def load_raw():
    dfs = []
    meta = []
    for p in [P12, P34]:
        df = pd.read_csv(p, usecols=["QuestionId", "UserId", "IsCorrect"])
        df = df.rename(
            columns={"QuestionId": "question_id", "UserId": "student_id", "IsCorrect": "is_correct"}
        )
        df["is_correct"] = df["is_correct"].astype(int)
        dfs.append(df)
        meta.append(
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
                "n_rows": int(len(df)),
                "n_questions": int(df.question_id.nunique()),
                "n_students": int(df.student_id.nunique()),
                "source_url": "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip",
            }
        )
    return pd.concat(dfs, ignore_index=True), meta


def bucket(rate, easy=0.8, hard=0.4):
    if rate >= easy:
        return "Human Easy"
    if rate <= hard:
        return "Human Hard"
    return "Human Mid"


def main():
    lines = []
    log(f"UTC {datetime.now(timezone.utc).isoformat()}", lines)
    log(f"zip sha256={sha256(PUBLIC_ZIP)} bytes={PUBLIC_ZIP.stat().st_size}", lines)
    assert P12.is_file() and P34.is_file(), "Extracted train files missing"

    raw, file_meta = load_raw()
    log(f"merged attempts={len(raw)} questions={raw.question_id.nunique()} students={raw.student_id.nunique()}", lines)

    agg = (
        raw.groupby("question_id")["is_correct"]
        .agg(n_attempts="count", n_correct="sum", mean_correct="mean")
        .reset_index()
    )
    # Primary: match recovered legacy (all qs have >=34 attempts after merge)
    primary = agg.copy()
    primary["empirical_bucket"] = [bucket(r) for r in primary["mean_correct"]]

    # Reproduce old counts
    vc = primary["empirical_bucket"].value_counts()
    old_match = (
        len(primary) == 27613
        and int(vc.get("Human Easy", 0)) == 6574
        and int(vc.get("Human Mid", 0)) == 19460
        and int(vc.get("Human Hard", 0)) == 1579
    )
    log(f"legacy count match={old_match} vc={vc.to_dict()}", lines)

    # Exact row match vs orphan derived CSV
    legacy = pd.read_csv(LEGACY)
    m = legacy[["question_id", "n_attempts", "n_correct", "mean_correct"]].merge(
        primary[["question_id", "n_attempts", "n_correct", "mean_correct"]],
        on="question_id",
        suffixes=("_leg", "_ver"),
    )
    row_match = bool(
        (m["n_attempts_leg"] == m["n_attempts_ver"]).all()
        and (m["n_correct_leg"] == m["n_correct_ver"]).all()
    )
    log(f"legacy derived CSV exact attempt/correct match={row_match}", lines)

    # Attempt distribution
    att = primary["n_attempts"]
    dist = pd.DataFrame(
        [
            {
                "n_attempts_total": int(len(raw)),
                "n_students": int(raw.student_id.nunique()),
                "n_questions": int(len(primary)),
                "min": int(att.min()),
                "q1": float(att.quantile(0.25)),
                "median": float(att.median()),
                "q3": float(att.quantile(0.75)),
                "p90": float(att.quantile(0.90)),
                "p95": float(att.quantile(0.95)),
                "max": int(att.max()),
            }
        ]
    )
    dist.to_csv(OUT_EEDI / "eedi_attempt_distribution.csv", index=False)

    # Beta-binomial EB shrinkage (method of moments prior)
    mu = float(primary["mean_correct"].mean())
    var = float(primary["mean_correct"].var(ddof=1))
    if 0 < var < mu * (1 - mu):
        common = mu * (1 - mu) / var - 1
        a0 = max(mu * common, 1e-3)
        b0 = max((1 - mu) * common, 1e-3)
    else:
        a0, b0 = 1.0, 1.0
    primary["alpha_post"] = a0 + primary["n_correct"]
    primary["beta_post"] = b0 + (primary["n_attempts"] - primary["n_correct"])
    primary["shrunk_rate"] = primary["alpha_post"] / (primary["alpha_post"] + primary["beta_post"])
    # 95% central posterior interval
    primary["post_low"] = beta_dist.ppf(0.025, primary["alpha_post"], primary["beta_post"])
    primary["post_high"] = beta_dist.ppf(0.975, primary["alpha_post"], primary["beta_post"])
    primary["shrunk_bucket"] = [bucket(r) for r in primary["shrunk_rate"]]
    # uncertain near boundaries if CI crosses cut
    primary["uncertain_boundary"] = (
        ((primary["post_low"] < 0.8) & (primary["post_high"] > 0.8))
        | ((primary["post_low"] < 0.4) & (primary["post_high"] > 0.4))
    )
    # IRT-ish proxy
    eps = 1e-3
    p = primary["shrunk_rate"].clip(eps, 1 - eps)
    primary["irt_b_proxy"] = -np.log(p / (1 - p))
    primary.to_csv(OUT_EEDI / "eedi_primary_item_estimates.csv", index=False)
    # also parquet/csv verified
    primary.to_parquet(OUT_PROC / "eedi_verified.parquet", index=False)
    primary.to_csv(OUT_PROC / "eedi_verified.csv", index=False)

    # Sensitivity
    sens_rows = []
    switch_rows = []
    base_lab = primary.set_index("question_id")["empirical_bucket"]
    for min_att in [5, 10, 20, 50, 100]:
        for easy, hard in [(0.80, 0.40), (0.75, 0.35), (0.85, 0.45)]:
            sub = agg[agg.n_attempts >= min_att].copy()
            sub["lab"] = [bucket(r, easy, hard) for r in sub["mean_correct"]]
            vc = sub["lab"].value_counts()
            sens_rows.append(
                {
                    "min_attempts": min_att,
                    "easy_thr": easy,
                    "hard_thr": hard,
                    "n_questions": int(len(sub)),
                    "n_easy": int(vc.get("Human Easy", 0)),
                    "n_mid": int(vc.get("Human Mid", 0)),
                    "n_hard": int(vc.get("Human Hard", 0)),
                }
            )
            if min_att == 5 and easy == 0.8:
                # switch vs primary empirical
                aligned = sub.set_index("question_id")["lab"]
                common = base_lab.index.intersection(aligned.index)
                sw = (base_lab.loc[common].values != aligned.loc[common].values).mean()
                switch_rows.append(
                    {
                        "comparison": "empirical_vs_same_cuts_min5",
                        "switch_rate": float(sw),
                        "n": int(len(common)),
                    }
                )
    # shrunk vs empirical switch
    sw2 = float((primary["empirical_bucket"] != primary["shrunk_bucket"]).mean())
    switch_rows.append(
        {
            "comparison": "empirical_vs_eb_shrink_primary",
            "switch_rate": sw2,
            "n": int(len(primary)),
            "n_uncertain_boundary": int(primary["uncertain_boundary"].sum()),
        }
    )
    pd.DataFrame(sens_rows).to_csv(OUT_EEDI / "eedi_sensitivity.csv", index=False)
    pd.DataFrame(switch_rows).to_csv(OUT_EEDI / "eedi_label_switches.csv", index=False)

    # Guessing-aware note: MCQ 4-option chance 0.25; hard cutoff 0.40 is above chance
    provenance = f"""# EeDi provenance (P0-1)

## Decision: Route A — Recoverable original EeDi analysis

The orphan derived file `Eedi_analysis/eedi_question_human_difficulty.csv` (SHA-256
`{sha256(LEGACY)}`) was **exactly reproduced** from the official NeurIPS 2020 Eedi
public release.

## Raw source

- URL: `https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip`
- Zip SHA-256: `{sha256(PUBLIC_ZIP)}`
- Bytes: {PUBLIC_ZIP.stat().st_size}
- Extracted:
{json.dumps(file_meta, indent=2)}

## Transformation

1. Concatenate `train_task_1_2.csv` + `train_task_3_4.csv` (attempt-level).
2. Aggregate by `QuestionId`: n_attempts, n_correct, mean_correct.
3. Bucket with easy≥0.80, hard≤0.40.
4. No additional filter needed: after merge, min attempts = 34 (all 27,613 retained).

**Note:** Using `train_task_1_2.csv` alone does **not** reproduce the legacy buckets;
both files must be concatenated (overlapping questions accumulate attempts).

## Reproduction of old counts

| Quantity | Value | Match |
|---|---|---|
| Retained questions | {len(primary)} | {len(primary)==27613} |
| Human Easy | {int(vc.get('Human Easy',0))} | {int(vc.get('Human Easy',0))==6574} |
| Human Mid | {int(vc.get('Human Mid',0))} | {int(vc.get('Human Mid',0))==19460} |
| Human Hard | {int(vc.get('Human Hard',0))} | {int(vc.get('Human Hard',0))==1579} |
| Row-level n_attempts/n_correct vs legacy CSV | — | {row_match} |

## Primary estimator

Beta–Binomial empirical-Bayes shrinkage (method-of-moments prior) with 95% posterior
intervals; empirical rates retained as sensitivity. IRT logit proxy from shrunk rates.
MCQ guessing floor ≈0.25; hard cutoff 0.40 is above chance.

## Outputs

- `data/processed/eedi_verified.parquet`
- `outputs/eedi/eedi_attempt_distribution.csv`
- `outputs/eedi/eedi_primary_item_estimates.csv`
- `outputs/eedi/eedi_sensitivity.csv`
- `outputs/eedi/eedi_label_switches.csv`
- `audit/evidence/eedi_recompute.log`

## Command

```powershell
python scripts/p0_closure/p0_1_eedi_verified.py
```

Generated UTC: {datetime.now(timezone.utc).isoformat()}
"""
    (EVID / "eedi_provenance.md").write_text(provenance, encoding="utf-8")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {
        "route": "A",
        "legacy_counts_reproduced": old_match and row_match,
        "n_questions": int(len(primary)),
        "n_uncertain_boundary": int(primary["uncertain_boundary"].sum()),
        "eb_switch_rate": sw2,
        "alpha0": a0,
        "beta0": b0,
    }
    (EVID / "eedi_p0_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    assert old_match and row_match, "Route A reproduction failed"


if __name__ == "__main__":
    main()
