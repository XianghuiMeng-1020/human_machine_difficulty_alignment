#!/usr/bin/env python3
"""S7: Build machine-difficulty records and the integrated same-item table.

Reads the per-solver raw prediction CSVs (one row per question_id x solver_id,
produced by s3_run_solver.py over the full 944-item manifest) and:
  - concatenates them into machine_predictions_948.parquet (no overwriting of
    failed outputs; missing/parse-failed rows are kept as missing, not coerced
    to incorrect)
  - computes per-item ensemble machine-difficulty quantities
  - joins with human_difficulty_948 to produce same_item_integrated_948.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "same_item_alignment/data"
RAW_PRED_DIR = DATA / "raw_predictions"
HUMAN = DATA / "human_difficulty_948.parquet"
AUDIT = ROOT / "same_item_alignment/audit"


def main():
    files = sorted(RAW_PRED_DIR.glob("*_full.csv"))
    if not files:
        raise SystemExit(f"No full-run prediction files found in {RAW_PRED_DIR}. Run s3_run_solver.py first.")

    frames = [pd.read_csv(f) for f in files]
    preds = pd.concat(frames, ignore_index=True)
    preds.to_parquet(DATA / "machine_predictions_948.parquet", index=False)
    print(f"machine_predictions_948.parquet: {len(preds)} rows from {len(files)} solver files")

    solver_ids = sorted(preds.solver_id.unique().tolist())
    human = pd.read_parquet(HUMAN)
    item_ids = sorted(human.question_id.unique().tolist())

    per_item_rows = []
    per_solver_binary = {}
    for sid in solver_ids:
        s = preds[preds.solver_id == sid].set_index("question_id")
        per_solver_binary[sid] = s["machine_correct"]

    for qid in item_ids:
        valid_flags = []
        correct_flags = []
        answers = []
        row = {"question_id": qid}
        for i, sid in enumerate(solver_ids, start=1):
            series = per_solver_binary[sid]
            val = series.get(qid, np.nan)
            is_valid = pd.notna(val)
            row[f"solver_{i}_id"] = sid
            row[f"solver_{i}_correct"] = bool(val) if is_valid else None
            if is_valid:
                valid_flags.append(True)
                correct_flags.append(bool(val))
        n_valid = len(valid_flags)
        n_correct = int(sum(correct_flags))
        row["n_valid_solvers"] = n_valid
        row["n_correct_solvers"] = n_correct
        row["machine_error_rate"] = (1 - n_correct / n_valid) if n_valid > 0 else None
        row["machine_majority_correct"] = (n_correct > n_valid / 2) if n_valid > 0 else None
        row["all_agree"] = (n_correct == n_valid or n_correct == 0) if n_valid > 0 else None
        per_item_rows.append(row)

    machine_df = pd.DataFrame(per_item_rows)

    integrated = human.merge(machine_df, on="question_id", how="inner")
    integrated = integrated.rename(columns={
        "empirical_difficulty": "human_empirical_difficulty",
        "eb_difficulty": "human_eb_difficulty",
        "irt_item_difficulty": "human_irt_difficulty",
    })
    integrated.to_parquet(DATA / "same_item_integrated_948.parquet", index=False)

    n_missing_any = int((integrated.n_valid_solvers < len(solver_ids)).sum())
    status = {
        "n_items_integrated": int(len(integrated)),
        "n_solvers": len(solver_ids),
        "solver_ids": solver_ids,
        "n_items_with_missing_solver_output": n_missing_any,
        "mean_machine_error_rate": float(integrated.machine_error_rate.mean(skipna=True)),
    }
    (AUDIT / "s7_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
