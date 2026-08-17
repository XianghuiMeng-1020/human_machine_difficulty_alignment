#!/usr/bin/env python3
"""E3: Human–machine item-level bridges.

E3a Bridge-RACE: stratified sample for human collection + response template.
E3b Bridge-EeDi: export EeDi prompts for encoder/LLM and compute alignment
    once model outputs exist (or simulate protocol tables).
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
    bootstrap_kappa,
    ensure_dir,
    save_table,
    write_jsonl,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--integrated_csv",
        default=str(REVISION_ROOT / "artifacts/race_val_integrated.csv"),
    )
    ap.add_argument(
        "--eedi_difficulty_csv",
        default=str(REPO_ROOT / "Eedi_analysis/eedi_question_human_difficulty.csv"),
    )
    ap.add_argument(
        "--eedi_attempts",
        nargs="*",
        default=[str(REPO_ROOT / "data/eedi/train_data/train_task_3_4.csv")],
    )
    ap.add_argument("--n_per_stratum", type=int, default=40)
    ap.add_argument("--target_attempts_per_item", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


def sample_bridge_race(df: pd.DataFrame, n_per: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    work = df.copy()
    if "datamap_region" not in work.columns or work["datamap_region"].isna().all():
        work["datamap_region"] = "unknown"
    work["datamap_region"] = work["datamap_region"].fillna("unknown")
    parts = []
    for band in ["MIDDLE", "HIGH"]:
        regions = sorted(work.loc[work.designer_difficulty_str == band, "datamap_region"].unique())
        for region in regions:
            sub = work[(work.designer_difficulty_str == band) & (work.datamap_region == region)]
            if len(sub) == 0:
                continue
            take = min(n_per, len(sub))
            idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            parts.append(sub.loc[idx])
        # Ensure HIGH has enough items even when encoder coverage is MIDDLE-only
        band_sub = work[work.designer_difficulty_str == band]
        already = pd.concat(parts)["question_id"] if parts else pd.Series(dtype=object)
        need = max(0, n_per * 2 - int((already.isin(band_sub.question_id)).sum() if len(already) else 0))
        # Simpler: guarantee at least n_per * 2 per band overall
        have = sum(len(p[p.designer_difficulty_str == band]) for p in parts) if parts else 0
        target_band = n_per * max(2, len(regions))
        if have < target_band:
            remain = band_sub[~band_sub.question_id.isin(already if len(already) else [])]
            take = min(target_band - have, len(remain))
            if take > 0:
                idx = rng.choice(remain.index.to_numpy(), size=take, replace=False)
                parts.append(remain.loc[idx])
    sample = pd.concat(parts, ignore_index=True).drop_duplicates("question_id")
    sample["bridge"] = "RACE"
    sample["target_n_human_attempts"] = 30
    return sample


def export_human_collection_pack(sample: pd.DataFrame, bridge_dir: Path) -> None:
    ensure_dir(bridge_dir)
    cols = [
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
        "target_n_human_attempts",
    ]
    cols = [c for c in cols if c in sample.columns]
    save_table(sample[cols], bridge_dir / "bridge_race_items.csv")

    # Annotator-facing JSONL (gold hidden in separate key file)
    public_rows = []
    gold_rows = []
    for _, row in sample.iterrows():
        public_rows.append(
            {
                "question_id": row["question_id"],
                "passage": row.get("article"),
                "question": row.get("question"),
                "options": {
                    "A": row.get("option_a"),
                    "B": row.get("option_b"),
                    "C": row.get("option_c"),
                    "D": row.get("option_d"),
                },
            }
        )
        gold_rows.append(
            {"question_id": row["question_id"], "gold_letter": row.get("answer_letter")}
        )
    write_jsonl(bridge_dir / "bridge_race_public_items.jsonl", public_rows)
    write_jsonl(bridge_dir / "bridge_race_gold_HIDDEN.jsonl", gold_rows)

    protocol = {
        "name": "Bridge-RACE human collection",
        "design": "stratified by grade-band x datamap_region",
        "n_items": int(len(sample)),
        "min_attempts_per_item": 30,
        "eligibility": "English reading comprehension, age-appropriate for grade band",
        "response_format": "single letter A-D",
        "derived_labels": {
            "human_correct_rate": "mean(is_correct)",
            "human_bucket": "easy>=0.80; hard<=0.40; else mid",
        },
        "analysis": "item-level kappa vs designer / encoder region / LLM vote",
    }
    (bridge_dir / "PROTOCOL_bridge_race.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    # Empty response log template
    template = pd.DataFrame(
        columns=[
            "annotator_id",
            "question_id",
            "chosen_letter",
            "timestamp",
            "consent_version",
        ]
    )
    save_table(template, bridge_dir / "bridge_race_responses_TEMPLATE.csv")


def build_eedi_model_prompts(args, bridge_dir: Path) -> pd.DataFrame | None:
    """Export EeDi questions for model solving (E3b)."""
    # Prefer difficulty csv; fall back to attempts aggregation
    qdf = None
    path = Path(args.eedi_difficulty_csv)
    if path.is_file():
        qdf = pd.read_csv(path)
    else:
        print(f"[WARN] missing {path}")
        return None

    # EeDi public challenge often lacks full stem text in attempt logs.
    # We still export IDs + human buckets for alignment once model outputs arrive.
    out = qdf.copy()
    out["bridge"] = "EeDi"
    save_table(out, bridge_dir / "bridge_eedi_items_with_human_labels.csv")

    # Placeholder prompts file for when question text is available
    prompts = []
    id_col = "question_id" if "question_id" in out.columns else out.columns[0]
    for qid in out[id_col].astype(str).tolist()[:5000]:
        prompts.append(
            {
                "question_id": qid,
                "prompt": None,
                "note": "Fill stem/options from EeDi question bank release before LLM/encoder call",
                "backend_target": "all",
            }
        )
    write_jsonl(bridge_dir / "bridge_eedi_model_prompts_TODO.jsonl", prompts)

    protocol = {
        "name": "Bridge-EeDi model collection",
        "goal": "Run encoder+3 LLM backends on EeDi items with existing student correctness buckets",
        "human_side": "EeDi attempt-derived buckets (after E5 filters)",
        "machine_side": "encoder region (if trained) + LLM vote",
        "analysis": "item-level confusion matrices and Cohen kappa",
    }
    (bridge_dir / "PROTOCOL_bridge_eedi.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return out


def analyze_alignment_if_ready(bridge_dir: Path, tables: Path) -> None:
    """If human responses exist, compute item-level rates and kappa vs machine labels."""
    resp_path = bridge_dir / "bridge_race_responses.csv"
    items_path = bridge_dir / "bridge_race_items.csv"
    gold_path = bridge_dir / "bridge_race_gold_HIDDEN.jsonl"
    integrated = Path(REVISION_ROOT / "artifacts/race_val_integrated.csv")
    if not (resp_path.is_file() and items_path.is_file() and gold_path.is_file()):
        save_table(
            pd.DataFrame(
                [
                    {
                        "status": "awaiting_human_responses",
                        "expected_file": str(resp_path),
                    }
                ]
            ),
            tables / "table_e3a_bridge_race_alignment.csv",
        )
        return

    resp = pd.read_csv(resp_path)
    gold = pd.DataFrame([json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()])
    merged = resp.merge(gold, on="question_id", how="left")
    merged["is_correct"] = (
        merged["chosen_letter"].astype(str).str.strip().str.upper().str[0]
        == merged["gold_letter"].astype(str).str.strip().str.upper().str[0]
    ).astype(int)
    rates = (
        merged.groupby("question_id")
        .agg(n_attempts=("is_correct", "count"), human_correct_rate=("is_correct", "mean"))
        .reset_index()
    )
    rates["human_bucket"] = np.where(
        rates["human_correct_rate"] >= 0.8,
        "easy",
        np.where(rates["human_correct_rate"] <= 0.4, "hard", "mid"),
    )
    if integrated.is_file():
        integ = pd.read_csv(integrated)
        rates = rates.merge(
            integ[
                [
                    c
                    for c in [
                        "question_id",
                        "designer_difficulty_str",
                        "datamap_region",
                        "llm_correct",
                        "llm_no_consensus",
                    ]
                    if c in integ.columns
                ]
            ],
            on="question_id",
            how="left",
        )
    rows = []
    if "designer_difficulty_str" in rates.columns:
        # Map designer to ordinal for kappa with human bucket via coarse map
        human = rates["human_bucket"].map({"easy": 0, "mid": 1, "hard": 2})
        designer = rates["designer_difficulty_str"].map({"MIDDLE": 0, "HIGH": 2})
        rows.append({"pair": "human_bucket_vs_designer", **bootstrap_kappa(human, designer)})
    if "datamap_region" in rates.columns:
        region = rates["datamap_region"].map(
            {"easy": 0, "middle": 1, "ambiguous": 1, "hard": 2}
        )
        human = rates["human_bucket"].map({"easy": 0, "mid": 1, "hard": 2})
        rows.append({"pair": "human_bucket_vs_region", **bootstrap_kappa(human, region)})
    save_table(pd.DataFrame(rows), tables / "table_e3a_bridge_race_alignment.csv")
    save_table(rates, bridge_dir / "bridge_race_human_rates.csv")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    bridge_dir = ensure_dir(out_dir / "bridge")
    tables = ensure_dir(out_dir / "tables")

    integ_path = Path(args.integrated_csv)
    if integ_path.is_file():
        df = pd.read_csv(integ_path)
    else:
        df = pd.read_csv(REPO_ROOT / "race_prepared/race_mcq_val.csv")

    sample = sample_bridge_race(df, args.n_per_stratum, args.seed)
    export_human_collection_pack(sample, bridge_dir)
    stratum_counts = (
        sample.groupby(["designer_difficulty_str", "datamap_region"])
        .size()
        .reset_index(name="n")
    )
    save_table(stratum_counts, tables / "table_e3a_bridge_race_sample_design.csv")
    print(f"[OK] Bridge-RACE sample n={len(sample)}")

    build_eedi_model_prompts(args, bridge_dir)
    analyze_alignment_if_ready(bridge_dir, tables)

    # Recruitment calculator
    calc = pd.DataFrame(
        [
            {
                "n_items": len(sample),
                "attempts_per_item": args.target_attempts_per_item,
                "total_item_responses": len(sample) * args.target_attempts_per_item,
                "if_50_items_per_annotator": int(
                    np.ceil(len(sample) * args.target_attempts_per_item / 50)
                ),
            }
        ]
    )
    save_table(calc, tables / "table_e3a_recruitment_calculator.csv")
    print("[OK] E3 bridge protocols ready")


if __name__ == "__main__":
    main()
