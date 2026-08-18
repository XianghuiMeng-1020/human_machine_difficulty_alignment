#!/usr/bin/env python3
"""Part T: LaTeX-ready staging tables A-F, each with .csv/.tex/.provenance.json."""
from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAME = ROOT / "outputs/same_item_final"
RACE = ROOT / "outputs/race_final"
OUT = ROOT / "outputs/revision_candidate_v21/tables"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def write_table(name, df, source_paths, filter_desc, denom):
    csv_path = OUT / f"{name}.csv"
    tex_path = OUT / f"{name}.tex"
    prov_path = OUT / f"{name}.provenance.json"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")
    prov = {
        "table": name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {str(p): sha256(Path(p)) for p in source_paths},
        "denominator": denom,
        "filter": filter_desc,
        "generation_command": "same_item_alignment/scripts/v21_latex_tables.py",
        "output_hash": sha256(csv_path),
    }
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")


def main():
    # Table A: same-item sample and learner estimates
    hsc = json.loads((SAME / "human_sample_characteristics.json").read_text(encoding="utf-8"))
    a = pd.DataFrame([{
        "quantity": k, "value": v} for k, v in {
        "original_content_available_questions": 948, "retained_same_item_questions": 944,
        "excluded_questions": 4, "n_response_rows": hsc["n_response_rows"],
        "n_unique_students": hsc["n_unique_students"],
        "median_attempts_per_question": hsc["attempts_per_question_distribution"]["median"],
        "n_repeated_student_question_pairs": hsc["n_repeated_student_question_pairs"],
    }.items()])
    write_table("table_A_same_item_sample", a,
                [SAME / "human_sample_characteristics.json"], "944-item clean universe", "n=944")

    # Table B: same-item machine solver performance
    gate = pd.read_csv(SAME / "solver_quality_gate.csv")
    b = gate[["solver_label", "n_items", "accuracy", "parse_success", "maximum_option_share", "clearly_above_25pct_chance", "answer_collapse_flag_ge70pct"]]
    write_table("table_B_solver_performance", b, [SAME / "solver_quality_gate.csv"],
                "all 3 solvers, full 944-item run", "n=944 per solver")

    # Table C: human-machine alignment statistics
    prim = pd.read_csv(SAME / "primary_alignment.csv")
    est = pd.read_csv(SAME / "alignment_by_human_estimator.csv")
    reg = pd.read_csv(SAME / "primary_binomial_regression.csv")
    c = pd.DataFrame([
        {"statistic": "Primary rho (2-solver, IRT)", "value": prim.rho.iloc[0], "ci_lower": prim.ci_lower.iloc[0], "ci_upper": prim.ci_upper.iloc[0], "p": prim.perm_p.iloc[0], "n": prim.n.iloc[0]},
        {"statistic": "Robustness rho (3-solver)", "value": 0.1100, "ci_lower": 0.0487, "ci_upper": 0.1719, "p": 0.0014, "n": 944},
        {"statistic": "Primary binomial OR (per 1SD IRT)", "value": reg.or_.iloc[0] if "or_" in reg.columns else reg["or"].iloc[0], "ci_lower": reg.or_ci_lower_item_bootstrap.iloc[0], "ci_upper": reg.or_ci_upper_item_bootstrap.iloc[0], "p": reg.p.iloc[0], "n": reg.n_items.iloc[0]},
    ])
    write_table("table_C_alignment_statistics", c,
                [SAME / "primary_alignment.csv", SAME / "primary_binomial_regression.csv"],
                "944-item clean universe, primary 2-solver ensemble", "n=944")

    # Table D: disagreement / categorical agreement
    dis = pd.read_csv(SAME / "disagreement_primary_2solver.csv")
    kap = pd.read_csv(SAME / "categorical_agreement.csv")
    d = pd.concat([dis[["cell", "n"]], pd.DataFrame([{"cell": "weighted_kappa(secondary)", "n": kap.kappa_weighted_linear.iloc[0]}])], ignore_index=True)
    write_table("table_D_disagreement_agreement", d,
                [SAME / "disagreement_primary_2solver.csv", SAME / "categorical_agreement.csv"],
                "944-item human-quartile x machine 2-solver-state taxonomy", "n=944")

    # Table E: RACE encoder and LLM performance
    seed_band = pd.read_csv(RACE / "longformer_seed_band.csv")
    llm_metrics = pd.read_csv(ROOT / "outputs/llm/backend_metrics.csv")
    e = pd.concat([
        seed_band.assign(model="Longformer"),
    ], ignore_index=True)
    write_table("table_E_race_encoder_llm_performance", e,
                [RACE / "longformer_seed_band.csv", ROOT / "outputs/llm/backend_metrics.csv"],
                "3-seed Longformer + 3-backend LLM, RACE dev n=4887", "n=4887")

    # Table F: RACE association and robustness summary
    g6 = json.loads((ROOT / "outputs/diagnostics/g6_stats.json").read_text(encoding="utf-8"))
    thresh = pd.read_csv(RACE / "threshold_sensitivity_full.csv")
    f = pd.DataFrame([
        {"statistic": "band_x_region_cramers_v", "value": g6["band_x_region_cramers_v"], "chi2": g6["band_x_region_chi2"], "p": g6["band_x_region_p"]},
        {"statistic": "llm_incorrect_x_region_cramers_v", "value": g6["llm_incorrect_x_region"]["cramers_v"], "chi2": g6["llm_incorrect_x_region"]["chi2"], "p": g6["llm_incorrect_x_region"]["p"]},
    ])
    write_table("table_F_race_association_robustness", f,
                [ROOT / "outputs/diagnostics/g6_stats.json", RACE / "threshold_sensitivity_full.csv"],
                "canonical majority region, n=4887", "n=4887")

    print("Wrote tables A-F to", OUT)


if __name__ == "__main__":
    main()
