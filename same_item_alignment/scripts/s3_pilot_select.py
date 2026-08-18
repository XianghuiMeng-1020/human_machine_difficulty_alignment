#!/usr/bin/env python3
"""S3a: Deterministic 50-item pilot sample, stratified by frozen human IRT difficulty.

10 items each from quintiles Q1 (easiest) .. Q5 (hardest) of irt_item_difficulty,
selected deterministically (fixed seed) from the frozen human_difficulty_948 table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HUMAN = ROOT / "same_item_alignment/data/human_difficulty_948.parquet"
OUT_PILOT = ROOT / "same_item_alignment/outputs/pilot"
OUT_PILOT.mkdir(parents=True, exist_ok=True)

SEED = 20260818


def main():
    df = pd.read_parquet(HUMAN)
    df = df.sort_values("irt_item_difficulty").reset_index(drop=True)
    df["irt_quintile"] = pd.qcut(df["irt_item_difficulty"], 5, labels=[1, 2, 3, 4, 5])

    rng = np.random.default_rng(SEED)
    picks = []
    for q in [1, 2, 3, 4, 5]:
        pool = df[df.irt_quintile == q]
        n = min(10, len(pool))
        idx = rng.choice(pool.index.values, size=n, replace=False)
        picks.append(df.loc[idx])
    pilot = pd.concat(picks).sort_values("question_id").reset_index(drop=True)

    pilot_out = pilot[[
        "question_id", "content_asset_path", "correct_option",
        "n_attempts", "n_students", "empirical_correctness",
        "eb_difficulty", "irt_item_difficulty", "irt_quintile",
    ]]
    pilot_out.to_csv(OUT_PILOT / "pilot50_items.csv", index=False)

    print(f"Pilot n={len(pilot_out)}")
    print(pilot_out.groupby("irt_quintile").size())
    (OUT_PILOT / "pilot50_status.json").write_text(
        json.dumps({"n_pilot_items": int(len(pilot_out)),
                     "per_quintile": pilot_out.groupby("irt_quintile").size().to_dict()}, indent=2, default=int),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
