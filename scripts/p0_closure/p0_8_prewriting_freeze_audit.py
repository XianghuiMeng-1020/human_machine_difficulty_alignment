#!/usr/bin/env python3
"""P0-8 pre-writing freeze audit: LLM coverage, API accounting, region robustness.

Does not invent ethics/IRB status. Writes machine-readable evidence only.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/diagnostics"
EVID = ROOT / "audit/evidence"
LLM = ROOT / "outputs/llm"
ENC = ROOT / "outputs/encoder"
LETTER = list("ABCD")
for d in (OUT, EVID):
    d.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(p: Path) -> dict:
    if not p.is_file():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.relative_to(ROOT)).replace("\\", "/"),
        "exists": True,
        "bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(p),
    }


def is_ok(rec: dict) -> bool:
    return bool(rec.get("parse_success")) and str(rec.get("parsed_option") or "")[:1] in LETTER


def classify_row(rec: dict) -> str:
    raw = str(rec.get("raw_response") or "")
    reason = rec.get("retry_reason")
    att = int(rec.get("attempt_index") or 0)
    if "AccountOverdueError" in raw:
        return "account_overdue_error"
    if reason == "transport_error" or "Error code:" in raw:
        return "transport_error"
    if att > 0 and reason:
        return "same_request_retry"
    if att == 0 and is_ok(rec):
        return "initial_success"
    if att == 0 and not is_ok(rec):
        return "initial_fail_unparsed"
    if att > 0 and is_ok(rec):
        return "retry_success"
    return "other"


def cramers_v(table: pd.DataFrame) -> float:
    if table.size == 0 or table.values.sum() == 0:
        return float("nan")
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def audit_human_files() -> dict:
    bridge = ROOT / "revision/bridge/bridge_race_responses.csv"
    items = ROOT / "revision/bridge/bridge_race_items.csv"
    proto = ROOT / "revision/bridge/PROTOCOL_bridge_race.json"
    e6 = ROOT / "revision/audit/e6_ratings.csv"
    e6r1 = ROOT / "revision/audit/e6_ratings_R1.csv"
    e6r2 = ROOT / "revision/audit/e6_ratings_R2.csv"
    e6arm = ROOT / "revision/audit/e6_arm_key_HIDDEN.csv"
    e6rub = ROOT / "revision/audit/e6_coding_rubric.json"
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ethics_inference": "none — only file-level facts; IRB/HREC/platform not present in repository",
        "bridge": {},
        "e6": {},
    }
    if bridge.is_file():
        df = pd.read_csv(bridge)
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        out["bridge"] = {
            "raw_file": file_meta(bridge),
            "items_file": file_meta(items),
            "protocol_file": file_meta(proto),
            "schema": list(df.columns),
            "n_rows": int(len(df)),
            "n_items": int(df.question_id.nunique()),
            "n_annotators": int(df.annotator_id.nunique()),
            "attempts_per_item_min": int(df.groupby("question_id").size().min()),
            "attempts_per_item_max": int(df.groupby("question_id").size().max()),
            "timestamp_min": None if ts.isna().all() else str(ts.min()),
            "timestamp_max": None if ts.isna().all() else str(ts.max()),
            "consent_versions": sorted(df["consent_version"].astype(str).unique().tolist())
            if "consent_version" in df.columns
            else [],
            "recruitment_platform_in_file": None,
            "irb_hrec_in_file": None,
            "protocol_keys": list(json.loads(proto.read_text(encoding="utf-8")).keys()) if proto.is_file() else [],
            "documented_in_repo": {
                "origin": "revision/bridge/ files dated 2026-07; protocol describes stratified 320x30 design only",
                "collection_date": "timestamps in CSV only",
                "newly_collected_vs_preexisting": "UNDOCUMENTED in ethics sense; files live under revision/ and timestamps are 2026-07",
                "recruitment_source_platform": "NOT FOUND in repository",
                "consent_status": "column consent_version present; no consent form / information sheet in repo",
                "ethics_irb_hrec": "NOT FOUND in repository (do not infer)",
                "usable_for_revision": False,
                "usable_reason": "cannot demonstrate recruitment platform or IRB/HREC/authorization",
            },
        }
    if e6.is_file():
        ed = pd.read_csv(e6)
        out["e6"] = {
            "ratings_file": file_meta(e6),
            "r1_file": file_meta(e6r1),
            "r2_file": file_meta(e6r2),
            "arm_key_file": file_meta(e6arm),
            "rubric_file": file_meta(e6rub),
            "schema": list(ed.columns),
            "n_rows": int(len(ed)),
            "raters": sorted(ed.rater_id.astype(str).unique().tolist()),
            "n_items": int(ed.question_id.nunique()),
            "r1_r2_identity": "UNDOCUMENTED — labels R1/R2 only; notes mix English/Chinese; cannot classify study-team vs recruited",
            "recruitment_source_platform": "NOT FOUND",
            "ethics_irb_hrec": "NOT FOUND in repository (do not infer)",
            "usable_for_revision": False,
            "usable_reason": "human-rater identity and ethics/authorization not documented",
        }
    (EVID / "human_ethics_provenance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def audit_llm() -> dict:
    gold = pd.read_csv(
        ROOT / "revision/artifacts/race_val_integrated.csv",
        usecols=["question_id", "answer_letter"],
    )
    gold_map = gold.set_index("question_id")["answer_letter"].astype(str).str.upper().str[0]
    all_q = set(gold.question_id.astype(str))
    backends = {
        "llm_deepseek_frozen_v1": LLM / "llm_deepseek_frozen_v1_responses.jsonl",
        "llm_gpt_frozen_v1": LLM / "llm_gpt_frozen_v1_responses.jsonl",
        "llm_doubao_frozen_v1": LLM / "llm_doubao_frozen_v1_responses.jsonl",
    }
    per = {}
    first_votes = {}
    last_votes = {}
    for run_id, path in backends.items():
        rows = load_jsonl(path)
        by_q = defaultdict(list)
        class_counts = Counter()
        for r in rows:
            by_q[str(r["question_id"])].append(r)
            class_counts[classify_row(r)] += 1
        unique = set(by_q)
        last_ok = {}
        first_ok = {}
        n_multi_success = 0
        n_multi_agree = 0
        missing = []
        for q, recs in by_q.items():
            oks = [r for r in recs if is_ok(r)]
            if oks:
                first_ok[q] = str(oks[0]["parsed_option"])[:1]
                last_ok[q] = str(oks[-1]["parsed_option"])[:1]
                if len(oks) > 1:
                    n_multi_success += 1
                    letters = [str(r["parsed_option"])[:1] for r in oks]
                    if len(set(letters)) == 1:
                        n_multi_agree += 1
            else:
                missing.append(q)
        # items never seen
        never = sorted(all_q - unique)
        first_votes[run_id] = first_ok
        last_votes[run_id] = last_ok
        per[run_id] = {
            "raw_file": file_meta(path),
            "n_raw_rows": len(rows),
            "unique_items_in_log": len(unique),
            "unique_items_expected": 4887,
            "successful_parses_last_success": len(last_ok),
            "successful_parses_first_success": len(first_ok),
            "missing_parses": len(missing) + len(never),
            "missing_question_ids": sorted(missing) + never,
            "final_usable_votes": len(last_ok),
            "row_class_counts": dict(class_counts),
            "items_with_gt1_successful_response": n_multi_success,
            "multi_success_all_agree": n_multi_agree,
            "multi_success_agreement_rate": (n_multi_agree / n_multi_success) if n_multi_success else None,
            "first_vs_last_vote_changes": int(
                sum(1 for q in last_ok if first_ok.get(q) != last_ok.get(q))
            ),
        }

    def consensus(vote_maps):
        rows = []
        run_ids = list(vote_maps)
        for q in sorted(all_q):
            votes = [vote_maps[r].get(q) for r in run_ids]
            present = [v for v in votes if v in LETTER]
            if not present:
                status, opt = "no_consensus", None
            else:
                vc = pd.Series(present).value_counts()
                if vc.iloc[0] >= 2:
                    status, opt = "consensus", str(vc.index[0])
                elif len(present) == 1:
                    status, opt = "no_consensus_single_vote", str(present[0])
                else:
                    status, opt = "no_consensus", None
            rows.append(
                {
                    "question_id": q,
                    "n_votes": len(present),
                    "consensus_status": status,
                    "consensus_option": opt,
                    "llm_correct": int(opt == gold_map.get(q, "")) if opt in LETTER else 0,
                }
            )
        return pd.DataFrame(rows)

    cons_last = consensus(last_votes)
    cons_first = consensus(first_votes)
    merge = cons_last.merge(cons_first, on="question_id", suffixes=("_last", "_first"))
    vote_or_cons_change = int(
        (
            (merge.consensus_status_last != merge.consensus_status_first)
            | (merge.consensus_option_last.fillna("") != merge.consensus_option_first.fillna(""))
        ).sum()
    )

    # two-of-three definition for missing backend
    n_2vote_cons = int(((cons_last.n_votes == 2) & (cons_last.consensus_status == "consensus")).sum())
    n_2vote_nocon = int(((cons_last.n_votes == 2) & (cons_last.consensus_status != "consensus")).sum())
    n_1vote = int((cons_last.n_votes == 1).sum())
    n_0vote = int((cons_last.n_votes == 0).sum())
    n_3vote = int((cons_last.n_votes == 3).sum())

    # Doubao recovery: items whose first successful parse appears after an AccountOverdueError on that item
    doubao_rows = load_jsonl(backends["llm_doubao_frozen_v1"])
    by_d = defaultdict(list)
    for r in doubao_rows:
        by_d[str(r["question_id"])].append(r)
    n_recovery_items = 0
    n_recovery_success_rows = 0
    for q, recs in by_d.items():
        had_overdue = any("AccountOverdueError" in str(r.get("raw_response") or "") for r in recs)
        oks = [r for r in recs if is_ok(r)]
        if had_overdue and oks:
            n_recovery_items += 1
            n_recovery_success_rows += len(oks)
    # 563 was the never-parsed-at-restart count; recompute never-ok-before-last-wave if possible via request_time
    # treat rows with request_time >= 2026-08-15T15:45 as recovery wave
    recov_wave = 0
    recov_wave_ok = 0
    for r in doubao_rows:
        t = str(r.get("request_time") or "")
        if t >= "2026-08-15T15:45":
            recov_wave += 1
            if is_ok(r):
                recov_wave_ok += 1

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backends": per,
        "coverage_one_liner": (
            f"DeepSeek unique={per['llm_deepseek_frozen_v1']['unique_items_in_log']} "
            f"parse_ok={per['llm_deepseek_frozen_v1']['successful_parses_last_success']}/4887; "
            f"GPT unique={per['llm_gpt_frozen_v1']['unique_items_in_log']} "
            f"parse_ok={per['llm_gpt_frozen_v1']['successful_parses_last_success']}/4887; "
            f"Doubao unique={per['llm_doubao_frozen_v1']['unique_items_in_log']} "
            f"parse_ok={per['llm_doubao_frozen_v1']['successful_parses_last_success']}/4887"
        ),
        "do_not_claim_all_three_4887_valid": per["llm_deepseek_frozen_v1"]["successful_parses_last_success"] != 4887,
        "two_of_three_rule": (
            "Consensus if at least two backends return the same A-D letter. "
            "A missing/unparsed backend is a non-vote. "
            "One vote is not consensus. Zero votes is no_consensus. "
            "Two agreeing votes (third missing) is consensus."
        ),
        "consensus_last_success": {
            "n_consensus": int((cons_last.consensus_status == "consensus").sum()),
            "n_no_consensus": int((cons_last.consensus_status != "consensus").sum()),
            "n_3vote": n_3vote,
            "n_2vote_consensus": n_2vote_cons,
            "n_2vote_no_consensus": n_2vote_nocon,
            "n_1vote": n_1vote,
            "n_0vote": n_0vote,
            "acc_consensus_conditioned": float(
                cons_last.loc[cons_last.consensus_status == "consensus", "llm_correct"].mean()
            ),
            "acc_unconditional_nocon_incorrect": float(cons_last.llm_correct.mean()),
            "three_way_full_agreement": int(
                ((cons_last.n_votes == 3) & (cons_last.consensus_status == "consensus")).sum()
            ),
        },
        "first_vs_last_success": {
            "consensus_or_option_changes": vote_or_cons_change,
            "deepseek_vote_changes": per["llm_deepseek_frozen_v1"]["first_vs_last_vote_changes"],
            "gpt_vote_changes": per["llm_gpt_frozen_v1"]["first_vs_last_vote_changes"],
            "doubao_vote_changes": per["llm_doubao_frozen_v1"]["first_vs_last_vote_changes"],
        },
        "api_accounting": {
            "deepseek_raw_rows": per["llm_deepseek_frozen_v1"]["n_raw_rows"],
            "gpt_raw_rows": per["llm_gpt_frozen_v1"]["n_raw_rows"],
            "doubao_raw_rows": per["llm_doubao_frozen_v1"]["n_raw_rows"],
            "total_logged_api_rows": sum(per[k]["n_raw_rows"] for k in per),
            "note": "Each jsonl row is one logged completion attempt (initial, same-request retry, or later recovery). This is the complete logged call count, not the backend_metrics.retry_rate field.",
            "doubao_row_classes": per["llm_doubao_frozen_v1"]["row_class_counts"],
            "doubao_items_with_overdue_then_success": n_recovery_items,
            "doubao_recovery_wave_rows_request_time_ge_20260815T1545": recov_wave,
            "doubao_recovery_wave_ok": recov_wave_ok,
            "historical_563_never_parsed_at_restart": 563,
        },
    }
    cons_last.to_csv(OUT / "freeze_consensus_last_success.csv", index=False)
    (OUT / "llm_coverage_freeze.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # overwrite headline metrics with last-success, honest coverage
    pd.DataFrame(
        [
            {
                "llm_run_id": k,
                "n_unique": v["unique_items_in_log"],
                "parse_ok_last_success": v["successful_parses_last_success"],
                "missing_parses": v["missing_parses"],
                "final_usable_votes": v["final_usable_votes"],
                "n_raw_api_rows": v["n_raw_rows"],
            }
            for k, v in per.items()
        ]
    ).to_csv(LLM / "backend_coverage_freeze.csv", index=False)
    pd.DataFrame([summary["consensus_last_success"]]).to_csv(LLM / "consensus_metrics.csv", index=False)
    return summary


def audit_regions() -> dict:
    integ = pd.read_csv(
        ROOT / "revision/artifacts/race_val_integrated.csv",
        usecols=["question_id", "designer_difficulty_str", "llm_no_consensus", "llm_correct", "datamap_region"],
    )
    # official 3-seed regions from existing table if present
    seed_reg = ENC / "seed_item_regions.csv"
    if seed_reg.is_file():
        sr = pd.read_csv(seed_reg)
        wide = sr.pivot(index="question_id", columns="seed", values="region")
        wide.columns = [f"region_seed{int(c)}" for c in wide.columns]
        m = wide.reset_index().merge(integ, on="question_id", how="inner")
        pair01 = float((m.region_seed0 == m.region_seed1).mean())
        pair02 = float((m.region_seed0 == m.region_seed2).mean())
        pair12 = float((m.region_seed1 == m.region_seed2).mean())
        all3 = float(((m.region_seed0 == m.region_seed1) & (m.region_seed1 == m.region_seed2)).mean())
        maj = m[["region_seed0", "region_seed1", "region_seed2"]].mode(axis=1)[0]
        # ties: mode may return multiple; if NaN in col1 after mode, it's a tie — already handled in p0_7 as middle
    else:
        pair01 = pair02 = pair12 = all3 = float("nan")
        m = integ.copy()
        maj = None

    # BigBird switch
    bb = json.loads((ENC / "architecture_check/bigbird_core_summary.json").read_text(encoding="utf-8"))

    def dynamics_from_seed(seed: int) -> pd.DataFrame:
        td = pd.read_csv(ENC / f"seed_runs/longformer_seed{seed}/training_dynamics_val.csv")
        agg = (
            td.groupby("question_id")
            .agg(mean_prob=("prob_correct", "mean"), std_prob=("prob_correct", "std"), frac_correct=("is_correct", "mean"))
            .reset_index()
        )
        last = td[td.epoch == td.epoch.max()][["question_id", "is_correct"]].rename(columns={"is_correct": "last_correct"})
        return agg.merge(last, on="question_id").merge(integ, on="question_id")

    robustness = []
    for seed in (0, 1, 2):
        df = dynamics_from_seed(seed)
        for name, lo, hi in [("p20_80", 0.20, 0.80), ("quartile_25_75", 0.25, 0.75), ("tercile_33_67", 0.33, 0.67)]:
            mlo, mhi = df.mean_prob.quantile([lo, hi])
            slo, shi = df.std_prob.quantile([lo, hi])
            labs = assign_regions(df, mlo, mhi, slo, shi)
            tmp = df.copy()
            tmp["region"] = labs
            ct = pd.crosstab(tmp.designer_difficulty_str, tmp.region)
            cons = tmp[tmp.llm_no_consensus.fillna(0).astype(int) == 0].copy()
            cons["llm_incorrect"] = (~cons.llm_correct.astype(bool)).astype(int)
            ct2 = pd.crosstab(cons.llm_incorrect, cons.region)
            acc = tmp.groupby("designer_difficulty_str")["last_correct"].mean().to_dict()
            robustness.append(
                {
                    "seed": seed,
                    "threshold": name,
                    "band_x_region_cramers_v": cramers_v(ct),
                    "llm_incorrect_x_region_cramers_v": cramers_v(ct2),
                    "encoder_acc_MIDDLE": float(acc.get("MIDDLE", float("nan"))),
                    "encoder_acc_HIGH": float(acc.get("HIGH", float("nan"))),
                    "middle_gt_high": float(acc.get("MIDDLE", 0)) > float(acc.get("HIGH", 0)),
                    "spearman_mean_prob_vs_enc_correct": float(
                        stats.spearmanr(tmp.mean_prob, tmp.last_correct).correlation
                    ),
                }
            )
    rdf = pd.DataFrame(robustness)
    rdf.to_csv(OUT / "region_threshold_robustness.csv", index=False)

    # continuous primary: mean_prob association with band (point-biserial / rank)
    cont = []
    for seed in (0, 1, 2):
        df = dynamics_from_seed(seed)
        y = (df.designer_difficulty_str == "HIGH").astype(int)
        r_pb = float(stats.pointbiserialr(y, df.mean_prob).correlation)
        cont.append(
            {
                "seed": seed,
                "pointbiserial_HIGH_vs_mean_prob": r_pb,
                "mean_prob_MIDDLE": float(df.loc[df.designer_difficulty_str == "MIDDLE", "mean_prob"].mean()),
                "mean_prob_HIGH": float(df.loc[df.designer_difficulty_str == "HIGH", "mean_prob"].mean()),
                "acc_MIDDLE": float(df.loc[df.designer_difficulty_str == "MIDDLE", "last_correct"].mean()),
                "acc_HIGH": float(df.loc[df.designer_difficulty_str == "HIGH", "last_correct"].mean()),
            }
        )
    cdf = pd.DataFrame(cont)
    cdf.to_csv(OUT / "continuous_dynamics_by_band.csv", index=False)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verified_3seed_exact_agreement": all3,
        "verified_pairwise": {"seed0_1": pair01, "seed0_2": pair02, "seed1_2": pair12},
        "verified_bigbird_vs_longformer_switch": bb.get("switch_rate_vs_longformer_region"),
        "matches_claimed_0_468": abs(all3 - 0.468) < 0.002 if pd.notna(all3) else False,
        "matches_claimed_pairwise_0_63": abs(((pair01 + pair02 + pair12) / 3) - 0.63) < 0.01
        if pd.notna(pair01)
        else False,
        "matches_claimed_bb_switch_0_5335": abs(float(bb.get("switch_rate_vs_longformer_region", 0)) - 0.5335) < 0.001,
        "hierarchy": {
            "PRIMARY": "continuous held-out confidence/generalization-dynamics (mean_prob, std_prob, last_correct)",
            "SECONDARY": "discrete regions — analysis labels only, not intrinsic item-difficulty ground truth",
        },
        "robustness_middle_gt_high_all_cells": bool(rdf.middle_gt_high.all()),
        "robustness_band_region_v_min": float(rdf.band_x_region_cramers_v.min()),
        "robustness_band_region_v_max": float(rdf.band_x_region_cramers_v.max()),
        "continuous_by_seed": cont,
        "recommended_manuscript_language": (
            "Do not treat discrete cartography-style regions as intrinsic difficulty. "
            "Lead with continuous held-out dynamics; report discrete regions as a secondary, seed-sensitive partition."
        ),
    }
    (OUT / "region_stability_freeze.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def write_sha_manifest() -> dict:
    """SHA-256 for empirical inputs that may stay outside Git."""
    patterns = [
        ROOT / "data/RACE/dev_mid.jsonl",
        ROOT / "data/RACE/dev_high.jsonl",
        ROOT / "data/RACE/train_mid.jsonl",
        ROOT / "data/RACE/train_high.jsonl",
        ROOT / "data/eedi_public_download/extracted/data/train_data/train_task_1_2.csv",
        ROOT / "data/eedi_public_download/extracted/data/train_data/train_task_3_4.csv",
        ROOT / "data/eedi/train_data/train_task_3_4.csv",
        ROOT / "revision/artifacts/race_val_integrated.csv",
        ROOT / "revision/bridge/bridge_race_responses.csv",
        ROOT / "revision/bridge/bridge_race_items.csv",
        ROOT / "revision/audit/e6_ratings.csv",
        ROOT / "outputs/llm/llm_deepseek_frozen_v1_responses.jsonl",
        ROOT / "outputs/llm/llm_gpt_frozen_v1_responses.jsonl",
        ROOT / "outputs/llm/llm_doubao_frozen_v1_responses.jsonl",
    ]
    for seed in (0, 1, 2):
        patterns.append(ENC / f"seed_runs/longformer_seed{seed}/model_epoch4.pt")
        patterns.append(ENC / f"seed_runs/longformer_seed{seed}/run_meta.json")
        patterns.append(ENC / f"seed_runs/longformer_seed{seed}/val_predictions.csv")
        patterns.append(ENC / f"seed_runs/longformer_seed{seed}/training_dynamics_val.csv")
    rows = [file_meta(p) for p in patterns]
    man = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "SHA-256 manifest for empirical inputs; large binaries may be gitignored",
        "files": rows,
    }
    (EVID / "scientific_freeze_sha_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(EVID / "scientific_freeze_sha_manifest.csv", index=False)
    return man


def main():
    human = audit_human_files()
    llm = audit_llm()
    regions = audit_regions()
    man = write_sha_manifest()
    print(
        json.dumps(
            {
                "bridge_usable": human.get("bridge", {}).get("documented_in_repo", {}).get("usable_for_revision"),
                "e6_usable": human.get("e6", {}).get("usable_for_revision"),
                "llm_coverage": llm["coverage_one_liner"],
                "deepseek_missing": llm["backends"]["llm_deepseek_frozen_v1"]["missing_question_ids"],
                "consensus": llm["consensus_last_success"],
                "first_vs_last_changes": llm["first_vs_last_success"],
                "api_total_rows": llm["api_accounting"]["total_logged_api_rows"],
                "doubao_recovery_wave": llm["api_accounting"]["doubao_recovery_wave_rows_request_time_ge_20260815T1545"],
                "region": {
                    "all3": regions["verified_3seed_exact_agreement"],
                    "pairwise": regions["verified_pairwise"],
                    "bb_switch": regions["verified_bigbird_vs_longformer_switch"],
                    "robust_middle_gt_high": regions["robustness_middle_gt_high_all_cells"],
                },
                "manifest_n": len(man["files"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
