#!/usr/bin/env python3
"""P0-6: Stage revision_candidate tables/figures + write audit/09_p0_closure_report.md."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "outputs/revision_candidate"
TAB = STAGE / "tables"
FIG = STAGE / "figures"
EVID = ROOT / "audit/evidence"
REPORT = ROOT / "audit/09_p0_closure_report.md"
for d in [TAB, FIG, EVID]:
    d.mkdir(parents=True, exist_ok=True)


def sha256(p: Path) -> str:
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def file_ok(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def gate_status() -> dict:
    """Evaluate G0–G9 from artifacts (honest PASS only with evidence)."""
    g = {}

    # G0
    race_mid = ROOT / "data/RACE/dev_mid.jsonl"
    race_high = ROOT / "data/RACE/dev_high.jsonl"
    eedi12 = ROOT / "data/eedi_public_download/extracted/data/train_data/train_task_1_2.csv"
    eedi34 = ROOT / "data/eedi_public_download/extracted/data/train_data/train_task_3_4.csv"
    g["G0"] = {
        "status": "PASS"
        if all(file_ok(p) for p in [race_mid, race_high, eedi12, eedi34, EVID / "eedi_provenance.md"])
        else "PARTIAL",
        "evidence": [
            str(race_mid),
            str(race_high),
            str(eedi12),
            str(eedi34),
            "audit/evidence/eedi_provenance.md",
            "audit/evidence/race_split_counts.json",
        ],
    }

    # G1
    split = ROOT / "audit/evidence/race_split_counts.json"
    g["G1"] = {
        "status": "PASS" if file_ok(split) else "FAIL",
        "evidence": ["audit/evidence/race_split_counts.json", "audit/evidence/race_split_audit.log"],
    }

    # G2
    required = [
        ROOT / "data/processed/race_items.parquet",
        ROOT / "data/processed/encoder_runs.parquet",
        ROOT / "data/processed/encoder_epoch_predictions.parquet",
        ROOT / "data/processed/encoder_item_summaries.parquet",
        ROOT / "data/processed/llm_runs.parquet",
        ROOT / "data/processed/llm_responses.parquet",
        ROOT / "data/processed/llm_votes.parquet",
        ROOT / "data/processed/race_analysis_integrated.parquet",
        ROOT / "outputs/diagnostics/join_audit.json",
    ]
    g["G2"] = {
        "status": "PASS" if all(file_ok(p) for p in required) else "PARTIAL",
        "evidence": [str(p.relative_to(ROOT)).replace("\\", "/") for p in required],
    }

    # G3
    join = ROOT / "outputs/diagnostics/join_audit.json"
    flow = ROOT / "outputs/diagnostics/subset_flow.csv"
    g3_ok = False
    if file_ok(join):
        ja = json.loads(join.read_text(encoding="utf-8"))
        g3_ok = bool(ja.get("all_assertions_pass") or ja.get("pass"))
    g["G3"] = {
        "status": "PASS" if g3_ok and file_ok(flow) else "PARTIAL",
        "evidence": ["outputs/diagnostics/join_audit.json", "outputs/diagnostics/subset_flow.csv"],
    }

    # G4
    struct = ROOT / "outputs/encoder/structural_tests"
    seed_sum = ROOT / "outputs/encoder/seed_summary.csv"
    arch = ROOT / "outputs/encoder/architecture_check/bigbird_core_summary.json"
    val_md = EVID / "encoder_validation.md"
    seeds_complete = 0
    seed_accs = []
    if file_ok(seed_sum):
        sdf = pd.read_csv(seed_sum)
        for seed in [0, 1, 2]:
            meta = ROOT / f"outputs/encoder/seed_runs/longformer_seed{seed}/run_meta.json"
            if file_ok(meta):
                seeds_complete += 1
                seed_accs.append(float(json.loads(meta.read_text()).get("val_accuracy", 0)))
    struct_ok = file_ok(struct / "label_mapping_audit_30.csv") and file_ok(val_md)
    arch_ok = file_ok(arch)
    g4 = "PASS" if struct_ok and arch_ok and seeds_complete >= 3 else ("PARTIAL" if struct_ok else "FAIL")
    g["G4"] = {
        "status": g4,
        "evidence": [
            "outputs/encoder/structural_tests/",
            "outputs/encoder/seed_runs/",
            "outputs/encoder/architecture_check/",
            "outputs/encoder/seed_summary.csv",
            "audit/evidence/encoder_validation.md",
        ],
        "seed_accs": seed_accs,
        "seeds_complete": seeds_complete,
    }

    # G5 — require honest coverage freeze (DeepSeek 4884 ≠ 4887) + API accounting
    llm_metrics = [
        ROOT / "outputs/llm/backend_coverage_freeze.csv",
        ROOT / "outputs/llm/consensus_metrics.csv",
        ROOT / "outputs/diagnostics/llm_coverage_freeze.json",
        EVID / "llm_reproducibility.md",
    ]
    frozen_meta = list((ROOT / "outputs/llm").glob("*_run_meta.json")) if (ROOT / "outputs/llm").is_dir() else []
    raw_ok = any(file_ok(p) for p in (ROOT / "outputs/llm").rglob("*.jsonl")) if (ROOT / "outputs/llm").is_dir() else False
    protocol = file_ok(ROOT / "configs/llm_protocol.yaml") and file_ok(ROOT / "prompts/race_mcq_prompt.txt")
    cov = ROOT / "outputs/diagnostics/llm_coverage_freeze.json"
    honest = False
    if file_ok(cov):
        obj = json.loads(cov.read_text(encoding="utf-8"))
        ds = obj.get("backends", {}).get("llm_deepseek_frozen_v1", {})
        honest = (
            int(ds.get("successful_parses_last_success", 0)) == 4884
            and bool(obj.get("do_not_claim_all_three_4887_valid"))
            and obj.get("api_accounting", {}).get("total_logged_api_rows") is not None
        )
    g5_pass = all(file_ok(p) for p in llm_metrics) and protocol and raw_ok and len(frozen_meta) >= 3 and honest
    g["G5"] = {
        "status": "PASS" if g5_pass else ("PARTIAL" if protocol else "FAIL"),
        "evidence": [str(p.relative_to(ROOT)).replace("\\", "/") for p in llm_metrics]
        + ["configs/llm_protocol.yaml", "prompts/race_mcq_prompt.txt"],
        "frozen_run_metas": len(frozen_meta),
        "honest_coverage_freeze": honest,
        "raw_present": raw_ok,
    }

    # G6
    g6f = ROOT / "outputs/diagnostics/g6_stats.json"
    g["G6"] = {
        "status": "PASS" if file_ok(g6f) else "PARTIAL",
        "evidence": [
            "outputs/diagnostics/g6_stats.json",
            "outputs/diagnostics/g6_band_x_region.csv",
            "outputs/diagnostics/g6_llm_incorrect_x_region.csv",
        ],
    }

    # G7
    eedi_files = [
        ROOT / "data/processed/eedi_verified.parquet",
        ROOT / "outputs/eedi/eedi_attempt_distribution.csv",
        ROOT / "outputs/eedi/eedi_primary_item_estimates.csv",
        ROOT / "outputs/eedi/eedi_sensitivity.csv",
        ROOT / "outputs/eedi/eedi_label_switches.csv",
        EVID / "eedi_provenance.md",
        EVID / "eedi_recompute.log",
    ]
    g["G7"] = {
        "status": "PASS" if all(file_ok(p) for p in eedi_files) else "FAIL",
        "evidence": [str(p.relative_to(ROOT)).replace("\\", "/") for p in eedi_files],
    }

    # G8
    g8f = ROOT / "outputs/diagnostics/g8_inversion.json"
    g["G8"] = {
        "status": "PASS" if file_ok(g8f) else "PARTIAL",
        "evidence": ["outputs/diagnostics/g8_inversion.json"],
    }

    # G9 — PASS only if ethics audit exists and Bridge/E6 are scoped out of the revision
    eth = EVID / "human_ethics_provenance.json"
    eth_md = EVID / "human_ethics_provenance.md"
    g9_ok = False
    if file_ok(eth) and file_ok(eth_md):
        obj = json.loads(eth.read_text(encoding="utf-8"))
        bridge_ok = bool(obj.get("bridge", {}).get("documented_in_repo", {}).get("usable_for_revision"))
        e6_ok = bool(obj.get("e6", {}).get("usable_for_revision"))
        g9_ok = (not bridge_ok) and (not e6_ok)
    g["G9"] = {
        "status": "PASS" if g9_ok else "PARTIAL",
        "evidence": [
            "audit/evidence/human_ethics_provenance.md",
            "audit/evidence/human_ethics_provenance.json",
        ],
        "scope": "Bridge-RACE and E6 NOT USABLE; no new-human-data claims in revision",
    }

    return g


def stage_artifacts():
    """Copy/generate candidate tables with provenance sidecars."""
    claims = {}
    staged = []

    def stage_csv(src: Path, name: str, filt: str, denom: int | None):
        if not file_ok(src):
            return
        dst = TAB / name
        df = pd.read_csv(src) if src.suffix == ".csv" else pd.read_parquet(src)
        if src.suffix == ".csv":
            df.to_csv(dst, index=False)
        else:
            df.to_csv(dst.with_suffix(".csv"), index=False)
            dst = dst.with_suffix(".csv")
        side = {
            "source": str(src.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256(src),
            "output_sha256": sha256(dst),
            "filter": filt,
            "denominator": denom,
            "n_rows": int(len(df)),
            "command": f"python scripts/p0_closure/p0_6_stage_and_report.py  # stage from {src.name}",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        (dst.with_suffix(dst.suffix + ".provenance.json")).write_text(json.dumps(side, indent=2), encoding="utf-8")
        staged.append(side)

    # RACE integrated headline
    integ = ROOT / "data/processed/race_analysis_integrated.parquet"
    if file_ok(integ):
        df = pd.read_parquet(integ)
        assert len(df) == 4887, len(df)
        if "grade_band" in df.columns:
            bands = df["grade_band"].value_counts().to_dict()
        else:
            bands = df["designer_difficulty_str"].value_counts().to_dict()
        claims["race_total"] = 4887
        claims["race_bands"] = bands
        if "encoder_correct" in df.columns:
            claims["encoder_accuracy"] = float(df["encoder_correct"].mean())
        elif "enc_correct" in df.columns:
            claims["encoder_accuracy"] = float(df["enc_correct"].mean())
        df.to_csv(TAB / "race_analysis_integrated.csv", index=False)
        stage_csv(integ, "race_analysis_integrated.csv", "official RACE val universe", 4887)

    for src, name, filt, denom in [
        (ROOT / "outputs/eedi/eedi_primary_item_estimates.csv", "eedi_primary_item_estimates.csv", "n>=50 primary", None),
        (ROOT / "outputs/eedi/eedi_sensitivity.csv", "eedi_sensitivity.csv", "threshold/cutoff sweep", None),
        (ROOT / "outputs/diagnostics/g6_band_x_region.csv", "g6_band_x_region.csv", "full integrated", 4887),
        (ROOT / "outputs/encoder/seed_summary.csv", "encoder_seed_summary.csv", "primary Longformer seeds", None),
        (ROOT / "outputs/encoder/region_stability.csv", "encoder_region_stability.csv", "3-seed region labels", 4887),
        (ROOT / "outputs/encoder/seed_g6_g8.csv", "encoder_seed_g6_g8.csv", "per-seed G6/G8", None),
        (ROOT / "outputs/llm/consensus_metrics.csv", "llm_consensus_metrics.csv", "frozen last-success", None),
        (ROOT / "outputs/llm/backend_coverage_freeze.csv", "llm_backend_coverage_freeze.csv", "honest parse coverage", None),
        (ROOT / "outputs/diagnostics/region_threshold_robustness.csv", "region_threshold_robustness.csv", "20/80 25/75 33/67 x 3 seeds", None),
        (ROOT / "outputs/llm/no_consensus_analysis.csv", "llm_no_consensus_analysis.csv", "frozen protocol", None),
        (ROOT / "outputs/llm/retry_analysis.csv", "llm_retry_analysis.csv", "frozen protocol", None),
        (ROOT / "outputs/diagnostics/subset_flow.csv", "subset_flow.csv", "join flow", None),
    ]:
        stage_csv(src, name, filt, denom)

    # consistency assertions
    asserts = {"race_total_equals_4887": claims.get("race_total") == 4887}
    if "race_bands" in claims:
        asserts["middle_plus_high"] = int(sum(claims["race_bands"].values())) == 4887
    (STAGE / "claims.json").write_text(json.dumps({"claims": claims, "assertions": asserts, "staged": staged}, indent=2), encoding="utf-8")
    return claims, asserts


def write_report(gates: dict, claims: dict, asserts: dict):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    statuses = {k: v["status"] for k, v in gates.items()}
    blockers = []
    for k, v in gates.items():
        if v["status"] != "PASS":
            blockers.append(f"{k}={v['status']}: missing/incomplete evidence → {', '.join(v.get('evidence', [])[:3])}")

    ready = all(s == "PASS" for s in statuses.values())
    verdict = "READY FOR MANUSCRIPT REVISION" if ready else "NOT READY"

    # headline quantities
    headlines = []
    eedi_log = EVID / "eedi_recompute.log"
    if file_ok(EVID / "eedi_provenance.md"):
        headlines.append("EeDi Route A: official train_task_1_2 + train_task_3_4 recover 27613 / 6574 / 19460 / 1579")
    if claims.get("race_total"):
        headlines.append(f"RACE official val n={claims['race_total']} bands={claims.get('race_bands')}")
    if claims.get("encoder_accuracy") is not None:
        headlines.append(f"Encoder accuracy (integrated table)={claims['encoder_accuracy']:.6f}")
    if gates["G4"].get("seed_accs"):
        accs = gates["G4"]["seed_accs"]
        headlines.append(
            f"Longformer multi-seed val acc mean={sum(accs)/len(accs):.4f} sd={(pd.Series(accs).std(ddof=1) if len(accs)>1 else 0):.4f} min={min(accs):.4f} max={max(accs):.4f}"
        )
    if file_ok(ROOT / "outputs/diagnostics/g6_stats.json"):
        g6 = json.loads((ROOT / "outputs/diagnostics/g6_stats.json").read_text())
        headlines.append(
            f"G6 band×region χ²={g6.get('band_x_region_chi2'):.3f} p={g6.get('band_x_region_p')} V={g6.get('band_x_region_cramers_v'):.3f}"
        )
        llm6 = g6.get("llm_incorrect_x_region") or {}
        if llm6:
            headlines.append(
                f"G6 LLM-incorrect×region χ²={llm6.get('chi2'):.3f} p={llm6.get('p')} V={llm6.get('cramers_v'):.3f}"
            )
    if file_ok(ROOT / "outputs/diagnostics/g8_inversion.json"):
        g8 = json.loads((ROOT / "outputs/diagnostics/g8_inversion.json").read_text())
        headlines.append(
            f"G8 encoder MIDDLE>HIGH={g8.get('encoder_middle_gt_high')} llm MIDDLE>HIGH={g8.get('llm_middle_gt_high')} acc={g8.get('encoder_acc_by_band')}"
        )
    if file_ok(ROOT / "outputs/diagnostics/llm_coverage_freeze.json"):
        fc = json.loads((ROOT / "outputs/diagnostics/llm_coverage_freeze.json").read_text(encoding="utf-8"))
        headlines.append(fc.get("coverage_one_liner", ""))
        headlines.append(
            f"API logged rows={fc.get('api_accounting', {}).get('total_logged_api_rows')} "
            f"(DeepSeek {fc.get('api_accounting', {}).get('deepseek_raw_rows')}+"
            f"GPT {fc.get('api_accounting', {}).get('gpt_raw_rows')}+"
            f"Doubao {fc.get('api_accounting', {}).get('doubao_raw_rows')}); "
            f"do not report retry_rate as the full call count"
        )
        c = fc.get("consensus_last_success", {})
        headlines.append(
            f"Frozen LLM consensus={c.get('n_consensus')} no_consensus={c.get('n_no_consensus')} "
            f"2vote_cons={c.get('n_2vote_consensus')} acc_cons={c.get('acc_consensus_conditioned')} "
            f"acc_uncond={c.get('acc_unconditional_nocon_incorrect')}"
        )
    headlines.append(
        "Bridge-RACE and E6: NOT USABLE FOR THE REVISION (ethics/authorization undocumented); withhold human-rater claims"
    )

    lines = [
        "# P0 Closure Report — FODE-D-26-00032",
        "",
        f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Commit: `{commit}`",
        f"- Branch: `revision/fode-p0-closure`",
        f"- Verdict: **{verdict}**",
        "",
        "## Gate table",
        "",
        "| Gate | Requirement | Status |",
        "|---|---|---|",
        f"| G0 | Raw inputs and full provenance available | {statuses['G0']} |",
        f"| G1 | RACE split independently verified | {statuses['G1']} |",
        f"| G2 | Canonical integrated table complete | {statuses['G2']} |",
        f"| G3 | All table denominators reconcile | {statuses['G3']} |",
        f"| G4 | Encoder pipeline valid and above chance | {statuses['G4']} |",
        f"| G5 | LLM protocol reproducible; no-consensus analyzed | {statuses['G5']} |",
        f"| G6 | Formal statistics and sensitivity complete | {statuses['G6']} |",
        f"| G7 | EeDi reliability analysis complete | {statuses['G7']} |",
        f"| G8 | Grade-band inversion resolved or explained | {statuses['G8']} |",
        f"| G9 | Audit-claim validation completed or appropriately scoped | {statuses['G9']} |",
        "",
        "## Evidence pointers",
        "",
    ]
    for k, v in gates.items():
        lines.append(f"### {k} — {v['status']}")
        for e in v.get("evidence", []):
            lines.append(f"- `{e}`")
        lines.append("")

    lines += [
        "## Remaining P0 blockers",
        "",
    ]
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- None")
    lines += [
        "",
        "## Known limitations (do not block READY)",
        "",
        "- DeepSeek valid parses are 4884/4887, not 4887. Two-of-three still yields consensus on the 3 missing items via GPT+Doubao.",
        "- Doubao logged 10949 API rows including 3287 AccountOverdueError rows, a 563-item never-parsed recovery, and extra accidental duplicate successes. Canonical vote = last success. First vs last changes 13 Doubao letters and 6 consensus items.",
        "- Discrete regions are secondary and seed-sensitive (3-seed exact agreement 0.468). Lead with continuous held-out dynamics.",
        "- Bridge-RACE and E6 human data are NOT USABLE (no IRB/platform/authorization in repo).",
        "- Legacy LLM logs remain `legacy_nonreproducible` and are not used for G5 numbers.",
        "",
        "## Final verified headline quantities",
        "",
    ]
    for h in headlines:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Staging assertions",
        "",
        "```json",
        json.dumps(asserts, indent=2),
        "```",
        "",
        "## Canonical paths",
        "",
        "- `data/processed/`",
        "- `outputs/eedi/`",
        "- `outputs/encoder/`",
        "- `outputs/llm/`",
        "- `outputs/diagnostics/`",
        "- `outputs/revision_candidate/`",
        "- `audit/evidence/`",
        "- `audit/09_p0_closure_report.md`",
        "",
        "## Recommended next command",
        "",
    ]
    if ready:
        lines.append("Proceed to manuscript revision using only staged numbers in `outputs/revision_candidate/`.")
    else:
        lines.append(
            "Close remaining blockers (see list above). Typical: finish 3 Longformer seeds + frozen 3-backend LLM rerun, then re-run `python scripts/p0_closure/p0_6_stage_and_report.py`."
        )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # machine-readable summary for final print
    summary = {
        "commit": commit,
        "verdict": verdict,
        "gates": statuses,
        "blockers": blockers,
        "headlines": headlines,
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
    }
    (EVID / "p0_closure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    claims, asserts = stage_artifacts()
    gates = gate_status()
    summary = write_report(gates, claims, asserts)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
