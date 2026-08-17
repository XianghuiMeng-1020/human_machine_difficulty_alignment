#!/usr/bin/env python3
"""E4: Agreement stats, cartography sensitivity, grade-band inversion, no-consensus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REVISION_ROOT,
    assign_datamap_region,
    bootstrap_kappa,
    ensure_dir,
    save_table,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--integrated_csv",
        default=str(REVISION_ROOT / "artifacts/race_val_integrated.csv"),
    )
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def agreement_tables(df: pd.DataFrame, tables: Path) -> None:
    rows = []
    if "datamap_region" in df.columns and df["datamap_region"].notna().any():
        sub = df.dropna(subset=["datamap_region", "designer_difficulty_str"])
        # Coarse ordinal: MIDDLE~easier intent, HIGH~harder intent vs region
        designer = sub["designer_difficulty_str"].map({"MIDDLE": 0, "HIGH": 1})
        region_hardish = sub["datamap_region"].isin(["hard", "ambiguous"]).astype(int)
        rows.append(
            {
                "pair": "designer_HIGH_vs_region_hard_or_ambiguous",
                **bootstrap_kappa(designer, region_hardish),
            }
        )
        # Multi-class region vs designer (designer expanded)
        rows.append(
            {
                "pair": "designer_vs_region_multiclass_proxy",
                **bootstrap_kappa(
                    sub["designer_difficulty_str"],
                    sub["datamap_region"],
                ),
            }
        )

    if "llm_correct" in df.columns and "datamap_region" in df.columns:
        cons = df[df.get("llm_no_consensus", 0) == 0].dropna(subset=["llm_correct", "datamap_region"])
        if len(cons):
            rows.append(
                {
                    "pair": "llm_incorrect_vs_region_hard",
                    **bootstrap_kappa(
                        (cons["llm_correct"] == 0).astype(int),
                        (cons["datamap_region"] == "hard").astype(int),
                    ),
                }
            )
            rows.append(
                {
                    "pair": "llm_incorrect_vs_region_hard_or_ambiguous",
                    **bootstrap_kappa(
                        (cons["llm_correct"] == 0).astype(int),
                        cons["datamap_region"].isin(["hard", "ambiguous"]).astype(int),
                    ),
                }
            )

    if "enc_correct" in df.columns and "llm_correct" in df.columns:
        both = df.dropna(subset=["enc_correct", "llm_correct"])
        if "llm_no_consensus" in both.columns:
            both = both[both["llm_no_consensus"] == 0]
        if len(both):
            rows.append(
                {
                    "pair": "encoder_correct_vs_llm_correct",
                    **bootstrap_kappa(both["enc_correct"].astype(int), both["llm_correct"].astype(int)),
                }
            )

    save_table(pd.DataFrame(rows), tables / "table_e4a_agreement_kappa.csv")


def sensitivity_analysis(df: pd.DataFrame, tables: Path) -> None:
    need = {"mean_prob", "std_prob", "frac_correct"}
    if not need.issubset(df.columns) or df["mean_prob"].isna().all():
        save_table(
            pd.DataFrame([{"status": "missing_datamap_metrics"}]),
            tables / "table_e4b_cartography_sensitivity.csv",
        )
        return

    base = df.dropna(subset=["mean_prob", "std_prob", "frac_correct"]).copy()
    schemes = {
        "tercile_33_67": (0.33, 0.67, 0.33, 0.67),
        "quartile_25_75": (0.25, 0.75, 0.25, 0.75),
        "split_40_60": (0.40, 0.60, 0.40, 0.60),
    }
    labeled = {}
    rows = []
    for name, (ml, mh, sl, sh) in schemes.items():
        lab, cuts = assign_datamap_region(base, ml, mh, sl, sh)
        labeled[name] = lab.set_index("question_id")["datamap_region"]
        vc = lab["datamap_region"].value_counts()
        rows.append(
            {
                "scheme": name,
                **{f"n_{k}": int(v) for k, v in vc.items()},
                **{f"share_{k}": float(v) / len(lab) for k, v in vc.items()},
                "mu_low": cuts["mu_low"],
                "mu_high": cuts["mu_high"],
                "sigma_low": cuts["sigma_low"],
                "sigma_high": cuts["sigma_high"],
            }
        )
    # Jaccard stability of hard/ambiguous sets vs tercile
    ref_hard = set(labeled["tercile_33_67"][labeled["tercile_33_67"] == "hard"].index)
    ref_amb = set(labeled["tercile_33_67"][labeled["tercile_33_67"] == "ambiguous"].index)
    for name, ser in labeled.items():
        hard = set(ser[ser == "hard"].index)
        amb = set(ser[ser == "ambiguous"].index)

        def jacc(a, b):
            if not a and not b:
                return 1.0
            return len(a & b) / len(a | b)

        rows_idx = next(i for i, r in enumerate(rows) if r["scheme"] == name)
        rows[rows_idx]["jaccard_hard_vs_tercile"] = jacc(hard, ref_hard)
        rows[rows_idx]["jaccard_ambiguous_vs_tercile"] = jacc(amb, ref_amb)

        # Association sign with HIGH
        tmp = base[["question_id", "designer_difficulty_str"]].merge(
            ser.rename("region").reset_index(), on="question_id"
        )
        high_in_hard = float(
            (tmp.designer_difficulty_str == "HIGH")[tmp.region == "hard"].mean()
        ) if (tmp.region == "hard").any() else float("nan")
        rows[rows_idx]["share_HIGH_within_hard"] = high_in_hard

    save_table(pd.DataFrame(rows), tables / "table_e4b_cartography_sensitivity.csv")


def inversion_diagnosis(df: pd.DataFrame, tables: Path) -> None:
    rows = []

    def acc_by_band(sub, pred_col, correct_col=None):
        out = []
        for band in ["MIDDLE", "HIGH"]:
            b = sub[sub.designer_difficulty_str == band]
            if correct_col and correct_col in b.columns:
                acc = float(b[correct_col].mean())
            elif pred_col in b.columns:
                acc = float((b[pred_col] == b["label"]).mean())
            else:
                acc = float("nan")
            out.append({"band": band, "n": len(b), "accuracy": acc})
        return out

    # Universe encoder
    if "enc_pred" in df.columns:
        for r in acc_by_band(df.dropna(subset=["enc_pred"]), "enc_pred", "enc_correct"):
            rows.append({"setting": "encoder_universe", **r})

    # Consensus LLM / encoder
    if "llm_no_consensus" in df.columns:
        cons = df[df["llm_no_consensus"] == 0]
        if "llm_correct" in cons.columns:
            for r in acc_by_band(cons, "llm_pred", "llm_correct"):
                rows.append({"setting": "llm_consensus", **r})
        if "enc_pred" in cons.columns:
            for r in acc_by_band(cons.dropna(subset=["enc_pred"]), "enc_pred", "enc_correct"):
                rows.append({"setting": "encoder_on_llm_consensus", **r})

        # No-consensus as incorrect
        tmp = df.copy()
        tmp["llm_correct_alt"] = np.where(tmp["llm_no_consensus"] == 1, 0, tmp.get("llm_correct"))
        for band in ["MIDDLE", "HIGH"]:
            b = tmp[tmp.designer_difficulty_str == band]
            rows.append(
                {
                    "setting": "llm_no_consensus_as_incorrect",
                    "band": band,
                    "n": len(b),
                    "accuracy": float(pd.to_numeric(b["llm_correct_alt"], errors="coerce").mean()),
                }
            )

    # Length stratification
    if "likely_truncated_2048" in df.columns and "enc_correct" in df.columns:
        for trunc in [0, 1]:
            sub = df[df["likely_truncated_2048"] == trunc].dropna(subset=["enc_correct"])
            for r in acc_by_band(sub, "enc_pred", "enc_correct"):
                rows.append({"setting": f"encoder_trunc{trunc}", **r})

    save_table(pd.DataFrame(rows), tables / "table_e4c_grade_band_inversion.csv")


def no_consensus_section(df: pd.DataFrame, tables: Path, protocols: Path) -> None:
    if "llm_no_consensus" not in df.columns:
        save_table(
            pd.DataFrame([{"status": "missing_llm_vote"}]),
            tables / "table_e4d_no_consensus.csv",
        )
        return

    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "n": len(df),
                "no_consensus_n": int(df["llm_no_consensus"].sum()),
                "no_consensus_rate": float(df["llm_no_consensus"].mean()),
            }
        ]
    )
    by_band = (
        df.groupby("designer_difficulty_str")
        .agg(n=("question_id", "count"), no_consensus_rate=("llm_no_consensus", "mean"))
        .reset_index()
    )
    by_band.insert(0, "scope", "by_grade_band")
    pieces = [overall, by_band]
    if "datamap_region" in df.columns:
        by_reg = (
            df.dropna(subset=["datamap_region"])
            .groupby("datamap_region")
            .agg(n=("question_id", "count"), no_consensus_rate=("llm_no_consensus", "mean"))
            .reset_index()
        )
        by_reg.insert(0, "scope", "by_region")
        pieces.append(by_reg)
    save_table(pd.concat(pieces, ignore_index=True), tables / "table_e4d_no_consensus.csv")

    # Qualitative sample (>=30)
    nc = df[df["llm_no_consensus"] == 1]
    if len(nc):
        sample = nc.sample(n=min(30, len(nc)), random_state=42)
        cols = [
            c
            for c in [
                "question_id",
                "designer_difficulty_str",
                "datamap_region",
                "article",
                "question",
                "option_a",
                "option_b",
                "option_c",
                "option_d",
                "answer_letter",
                "llm_gpt4o_letter",
                "llm_doubao_letter",
                "llm_deepseek_letter",
            ]
            if c in sample.columns
        ]
        save_table(sample[cols], protocols / "e4d_no_consensus_qualitative_sample.csv")
        rubric = {
            "items": int(len(sample)),
            "codes": [
                "ambiguous_key",
                "flawed_distractor",
                "evidence_not_locatable",
                "multiple_plausible",
                "none_genuine_hard",
            ],
            "instruction": "Blind to model outputs when possible; mark all that apply.",
        }
        (protocols / "e4d_no_consensus_coding_rubric.json").write_text(
            json.dumps(rubric, indent=2), encoding="utf-8"
        )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables = ensure_dir(out_dir / "tables")
    protocols = ensure_dir(out_dir / "protocols")
    df = pd.read_csv(args.integrated_csv)

    agreement_tables(df, tables)
    sensitivity_analysis(df, tables)
    inversion_diagnosis(df, tables)
    no_consensus_section(df, tables, protocols)
    print("[OK] E4 analyses complete")


if __name__ == "__main__":
    main()
