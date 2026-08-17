#!/usr/bin/env python3
"""E0: Build integrated question-level table and exact manuscript tables.

Merges official RACE val (4887), encoder predictions/TD (when available),
and multi-backend LLM votes into revision/artifacts/race_val_integrated.csv.
Exports exact counts for manuscript tables (no hand-rounded figures).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPO_ROOT,
    REVISION_ROOT,
    assign_datamap_region,
    compute_td_metrics,
    ensure_dir,
    letter_to_label,
    parse_option_letter,
    save_table,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race_val_csv", default=str(REPO_ROOT / "race_prepared/race_mcq_val.csv"))
    ap.add_argument(
        "--bert_pred_csv",
        default=str(
            REPO_ROOT
            / "race_trainedmodels_5e-4_e5_256bs/models_longformer-base-4096/val_predictions.csv"
        ),
    )
    ap.add_argument(
        "--bert_td_csv",
        default=str(
            REPO_ROOT
            / "race_trainedmodels_5e-4_e5_256bs/models_longformer-base-4096/training_dynamics_val.csv"
        ),
    )
    ap.add_argument(
        "--llm_vote_csv",
        default=str(REVISION_ROOT / "artifacts/llm_vote_val.csv"),
        help="Built by e2_llm_vote_aggregate.py",
    )
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def load_optional_csv(path: Path, required_cols=None) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"[WARN] missing optional file: {path}")
        return None
    df = pd.read_csv(path)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[WARN] {path} missing cols {missing}")
            return None
    return df


def export_exact_tables(df: pd.DataFrame, tables_dir: Path) -> None:
    ensure_dir(tables_dir)

    # Split sanity (A1)
    split_counts = (
        df["designer_difficulty_str"]
        .value_counts()
        .reindex(["MIDDLE", "HIGH"])
        .fillna(0)
        .astype(int)
        .rename("n")
        .reset_index()
        .rename(columns={"index": "designer_difficulty_str"})
    )
    if "designer_difficulty_str" not in split_counts.columns:
        split_counts.columns = ["designer_difficulty_str", "n"]
    split_counts["share"] = split_counts["n"] / split_counts["n"].sum()
    save_table(split_counts, tables_dir / "table_a1_official_split_counts.csv")

    # Region sizes (Table-like panel a)
    if "datamap_region" in df.columns and df["datamap_region"].notna().any():
        region = (
            df["datamap_region"]
            .fillna("missing")
            .value_counts()
            .rename("n")
            .reset_index()
            .rename(columns={"index": "datamap_region"})
        )
        if "datamap_region" not in region.columns:
            region.columns = ["datamap_region", "n"]
        region["share"] = region["n"] / len(df)
        save_table(region, tables_dir / "table_region_sizes.csv")

        cross = (
            pd.crosstab(df["designer_difficulty_str"], df["datamap_region"], margins=True)
            .reset_index()
        )
        save_table(cross, tables_dir / "table_designer_x_region.csv")

        # Region medians
        med = (
            df.groupby("datamap_region")[["mean_prob", "std_prob", "frac_correct"]]
            .median()
            .reset_index()
        )
        save_table(med, tables_dir / "table_region_medians.csv")

    # Encoder option-level PR (Table 4 style)
    if "enc_pred" in df.columns and df["enc_pred"].notna().any():
        rows = []
        for cls in range(4):
            gold_pos = df["label"] == cls
            pred_pos = df["enc_pred"] == cls
            tp = int((gold_pos & pred_pos).sum())
            fp = int((~gold_pos & pred_pos).sum())
            fn = int((gold_pos & ~pred_pos).sum())
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            rows.append(
                {
                    "class": "ABCD"[cls],
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
        enc_pr = pd.DataFrame(rows)
        enc_pr.attrs = {}
        overall_acc = float((df["enc_pred"] == df["label"]).mean())
        enc_pr.loc[len(enc_pr)] = {
            "class": "OVERALL_ACC",
            "precision": overall_acc,
            "recall": overall_acc,
            "tp": int((df["enc_pred"] == df["label"]).sum()),
            "fp": "",
            "fn": "",
        }
        save_table(enc_pr, tables_dir / "table_encoder_class_pr.csv")

    # LLM consensus subset tables (B1/B2)
    if "llm_no_consensus" in df.columns:
        cons = df[df["llm_no_consensus"] == 0].copy()
        save_table(
            pd.DataFrame(
                [
                    {
                        "universe_n": len(df),
                        "consensus_n": len(cons),
                        "no_consensus_n": int((df["llm_no_consensus"] == 1).sum()),
                        "no_consensus_rate": float(df["llm_no_consensus"].mean()),
                        "llm_acc_consensus_only": float(cons["llm_correct"].mean())
                        if len(cons)
                        else float("nan"),
                        "llm_acc_no_consensus_as_incorrect": float(
                            np.where(df["llm_no_consensus"] == 1, 0, df["llm_correct"]).mean()
                        ),
                        "encoder_acc_universe": float((df["enc_pred"] == df["label"]).mean())
                        if "enc_pred" in df.columns
                        else float("nan"),
                        "encoder_acc_consensus": float((cons["enc_pred"] == cons["label"]).mean())
                        if "enc_pred" in cons.columns and len(cons)
                        else float("nan"),
                    }
                ]
            ),
            tables_dir / "table_subset_denominators.csv",
        )

        if len(cons):
            by_band = (
                cons.groupby("designer_difficulty_str")
                .agg(
                    n_questions=("question_id", "count"),
                    longformer_accuracy=("enc_correct", "mean"),
                    llm_accuracy=("llm_correct", "mean"),
                )
                .reset_index()
            )
            if "enc_correct" not in cons.columns and "enc_pred" in cons.columns:
                cons["enc_correct"] = (cons["enc_pred"] == cons["label"]).astype(int)
                by_band = (
                    cons.groupby("designer_difficulty_str")
                    .agg(
                        n_questions=("question_id", "count"),
                        longformer_accuracy=("enc_correct", "mean"),
                        llm_accuracy=("llm_correct", "mean"),
                    )
                    .reset_index()
                )
            save_table(by_band, tables_dir / "table_accuracy_by_grade_band_consensus.csv")

    # Manifest for editorial recomputation
    manifest = {
        "n_val": int(len(df)),
        "n_middle": int((df["designer_difficulty_str"] == "MIDDLE").sum()),
        "n_high": int((df["designer_difficulty_str"] == "HIGH").sum()),
        "columns": list(df.columns),
        "note": "Exact empirical counts; do not round for manuscript tables.",
    }
    ensure_dir(tables_dir)
    (tables_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] manifest: {manifest}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    art = ensure_dir(out_dir / "artifacts")
    tables = ensure_dir(out_dir / "tables")

    val = pd.read_csv(args.race_val_csv)
    assert len(val) == 4887, f"Expected official val n=4887, got {len(val)}"
    m = int((val.designer_difficulty_str == "MIDDLE").sum())
    h = int((val.designer_difficulty_str == "HIGH").sum())
    assert (m, h) == (1436, 3451), f"Expected M/H=1436/3451, got {m}/{h}"

    df = val.copy()

    pred = load_optional_csv(
        Path(args.bert_pred_csv), ["question_id", "pred_label", "prob_correct"]
    )
    if pred is not None:
        pred = pred.drop_duplicates("question_id")
        df = df.merge(
            pred.rename(
                columns={
                    "pred_label": "enc_pred",
                    "prob_correct": "enc_prob_correct_final",
                }
            )[["question_id", "enc_pred", "enc_prob_correct_final"]],
            on="question_id",
            how="left",
        )
        df["enc_correct"] = (df["enc_pred"] == df["label"]).astype(float)
        df.loc[df["enc_pred"].isna(), "enc_correct"] = np.nan
        print(
            f"[INFO] encoder preds matched "
            f"{df['enc_pred'].notna().sum()}/{len(df)} "
            f"(legacy checkpoint may cover buggy split only)"
        )

    td = load_optional_csv(
        Path(args.bert_td_csv), ["question_id", "epoch", "prob_correct", "is_correct"]
    )
    if td is not None:
        dm = compute_td_metrics(td)
        dm, cuts = assign_datamap_region(dm)
        df = df.merge(dm, on="question_id", how="left")
        (art / "datamap_cuts_tercile.json").write_text(json.dumps(cuts, indent=2))
        print(f"[INFO] datamap matched {df['datamap_region'].notna().sum()}/{len(df)}")

    vote = load_optional_csv(Path(args.llm_vote_csv))
    if vote is not None:
        keep = [c for c in vote.columns if c == "question_id" or c.startswith("llm_")]
        df = df.merge(vote[keep], on="question_id", how="left")
        if "llm_no_consensus" not in df.columns and "llm_pred" in df.columns:
            df["llm_no_consensus"] = df["llm_pred"].isna().astype(int)
        if "llm_correct" not in df.columns and "llm_pred" in df.columns:
            df["llm_correct"] = (df["llm_pred"] == df["label"]).astype(float)
            df.loc[df["llm_pred"].isna(), "llm_correct"] = np.nan

    # Passage length / truncation diagnostics (E1c inputs)
    df["passage_chars"] = df["article"].astype(str).str.len()
    df["passage_approx_tokens"] = (df["passage_chars"] / 4.0).round().astype(int)
    df["likely_truncated_2048"] = (df["passage_approx_tokens"] > 1800).astype(int)

    out_csv = art / "race_val_integrated.csv"
    save_table(df, out_csv)
    export_exact_tables(df, tables)

    readme = out_dir / "REPLICATION.md"
    readme.write_text(
        """# Revision replication package (FODE-D-26-00032)

## Official RACE validation split
- Total: **4887**
- MIDDLE (RACE-M): **1436**
- HIGH (RACE-H): **3451**

## Key artifact
- `artifacts/race_val_integrated.csv`: question-level integrated table
- `tables/*.csv`: exact empirical manuscript tables (no hand rounding)

## Rebuild
```bash
python scripts/RACE_prepare_and_designer_stats.py --val_high data/RACE/dev_high.jsonl
python scripts/revision/e2_llm_vote_aggregate.py
python scripts/revision/e0_build_integrated_table.py
```

Encoder retrain (E1) and full 3-backend LLM (E2) must be run on GPU/API before final numbers.
""",
        encoding="utf-8",
    )
    print(f"[OK] E0 complete -> {out_csv}")


if __name__ == "__main__":
    main()
