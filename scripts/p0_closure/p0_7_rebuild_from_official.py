#!/usr/bin/env python3
"""Rebuild canonical tables from official 3-seed Longformer + frozen LLM runs.

Does not invent metadata. Official seeds are the G4 source of truth.
Frozen LLM votes replace legacy votes in the analysis table when all 3
backends have 4887 unique items.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
DIAG = ROOT / "outputs/diagnostics"
ENC = ROOT / "outputs/encoder"
EVID = ROOT / "audit/evidence"
LLM = ROOT / "outputs/llm"
for d in [PROC, DIAG, ENC, EVID]:
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
LETTER = list("ABCD")


def sha256_file(p: Path) -> str:
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table.values)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1)))) if n and min(r, k) > 1 else float("nan")


def assign_regions(df: pd.DataFrame, mu_lo, mu_hi, sig_lo, sig_hi):
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


def majority_label(vals):
    vals = [v for v in vals if pd.notna(v)]
    if not vals:
        return None
    c = Counter(vals)
    top = c.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return "middle"  # tie → middle, do not invent a stronger label


def seed_item_summary(seed: int) -> pd.DataFrame:
    d = ENC / "seed_runs" / f"longformer_seed{seed}"
    td = pd.read_csv(d / "training_dynamics_val.csv")
    pred = pd.read_csv(d / "val_predictions.csv")
    agg = (
        td.groupby("question_id")
        .agg(
            mean_prob=("prob_correct", "mean"),
            std_prob=("prob_correct", "std"),
            frac_correct=("is_correct", "mean"),
        )
        .reset_index()
    )
    last = td[td.epoch == td.epoch.max()][["question_id", "is_correct", "prob_correct"]].rename(
        columns={"is_correct": "last_correct", "prob_correct": "last_prob"}
    )
    pred = pred.rename(columns={"prob_correct": "pred_prob"})
    pred["pred_correct"] = (pred["gold_label"] == pred["pred_label"]).astype(int)
    out = agg.merge(last, on="question_id", how="inner").merge(
        pred[["question_id", "pred_correct", "pred_prob"]], on="question_id", how="left"
    )
    mu_lo, mu_hi = out.mean_prob.quantile([0.33, 0.67])
    sig_lo, sig_hi = out.std_prob.quantile([0.33, 0.67])
    out["region"] = assign_regions(out, mu_lo, mu_hi, sig_lo, sig_hi)
    out["seed"] = seed
    out["encoder_run_id"] = f"enc_longformer_seed{seed}"
    return out


def compute_seed_regions_and_stability(items: pd.DataFrame) -> pd.DataFrame:
    frames = [seed_item_summary(s) for s in SEEDS]
    all_s = pd.concat(frames, ignore_index=True)
    all_s.to_csv(ENC / "seed_item_regions.csv", index=False)

    wide_r = all_s.pivot(index="question_id", columns="seed", values="region")
    wide_r.columns = [f"region_seed{c}" for c in wide_r.columns]
    wide_c = all_s.pivot(index="question_id", columns="seed", values="last_correct")
    wide_c.columns = [f"last_correct_seed{c}" for c in wide_c.columns]
    wide_p = all_s.pivot(index="question_id", columns="seed", values="pred_correct")
    wide_p.columns = [f"pred_correct_seed{c}" for c in wide_p.columns]

    canon = wide_r.copy()
    canon["region_majority"] = wide_r.apply(lambda r: majority_label(r.tolist()), axis=1)
    canon["encoder_correct_mean"] = wide_c.mean(axis=1)
    canon["encoder_correct_majority"] = (wide_c.mean(axis=1) >= 0.5).astype(int)
    canon = canon.join(wide_c).join(wide_p)
    canon = canon.reset_index()

    # pairwise region agreement
    pair_rows = []
    cols = [f"region_seed{s}" for s in SEEDS]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            both = wide_r[[a.replace("region_", "region_") if False else a, b]].dropna()
            pair_rows.append(
                {
                    "a": a,
                    "b": b,
                    "n": int(len(both)),
                    "agreement": float((both[a] == both[b]).mean()),
                }
            )
    # vs legacy manuscript region
    integ = pd.read_csv(
        ROOT / "revision/artifacts/race_val_integrated.csv",
        usecols=["question_id", "datamap_region", "designer_difficulty_str"],
    )
    m = canon.merge(integ, on="question_id", how="left")
    vs_legacy = float((m["region_majority"] == m["datamap_region"]).mean())
    all_three = float(
        ((m["region_seed0"] == m["region_seed1"]) & (m["region_seed1"] == m["region_seed2"])).mean()
    )

    stab = pd.DataFrame(pair_rows)
    stab.loc[len(stab)] = {
        "a": "majority",
        "b": "legacy_datamap_region",
        "n": int(len(m)),
        "agreement": vs_legacy,
    }
    stab.loc[len(stab)] = {
        "a": "seed0",
        "b": "seed1_and_seed2_all_equal",
        "n": int(len(m)),
        "agreement": all_three,
    }
    stab.to_csv(ENC / "region_stability.csv", index=False)

    # per-seed G6/G8 encoder side
    seed_stats = []
    for s in SEEDS:
        tmp = m.merge(
            all_s.loc[all_s.seed == s, ["question_id", "last_correct", "region"]],
            on="question_id",
            suffixes=("", f"_s{s}"),
        )
        # last_correct already from wide; use seed-specific region
        reg = tmp[f"region_seed{s}"]
        ct = pd.crosstab(tmp["designer_difficulty_str"], reg)
        acc = tmp.groupby("designer_difficulty_str")[f"last_correct_seed{s}"].mean().to_dict()
        seed_stats.append(
            {
                "seed": s,
                "val_acc": float(tmp[f"last_correct_seed{s}"].mean()),
                "band_x_region_cramers_v": cramers_v(ct),
                "encoder_acc_MIDDLE": float(acc.get("MIDDLE", float("nan"))),
                "encoder_acc_HIGH": float(acc.get("HIGH", float("nan"))),
                "middle_gt_high": float(acc.get("MIDDLE", 0)) > float(acc.get("HIGH", 0)),
            }
        )
    pd.DataFrame(seed_stats).to_csv(ENC / "seed_g6_g8.csv", index=False)

    # majority-region G6 encoder
    ct_m = pd.crosstab(m["designer_difficulty_str"], m["region_majority"])
    ct_m.to_csv(DIAG / "g6_band_x_region_majority_seeds.csv")
    chi2, p, dof, _ = stats.chi2_contingency(ct_m.values)
    maj_g6 = {
        "source": "majority region across official Longformer seeds 0/1/2",
        "n": int(len(m)),
        "band_x_region_chi2": float(chi2),
        "band_x_region_p": float(p),
        "band_x_region_dof": int(dof),
        "band_x_region_cramers_v": cramers_v(ct_m),
        "three_seed_region_agreement": all_three,
        "majority_vs_legacy_region": vs_legacy,
        "per_seed": seed_stats,
    }
    (DIAG / "g6_encoder_multiseed.json").write_text(json.dumps(maj_g6, indent=2), encoding="utf-8")

    # official seed_summary with local paths
    rows = []
    accs = []
    for s in SEEDS:
        meta_p = ENC / "seed_runs" / f"longformer_seed{s}" / "run_meta.json"
        obj = json.loads(meta_p.read_text(encoding="utf-8"))
        accs.append(float(obj["val_accuracy"]))
        ckpt = ENC / "seed_runs" / f"longformer_seed{s}" / "model_epoch4.pt"
        if not ckpt.is_file():
            ckpt = ENC / "seed_runs" / f"longformer_seed{s}" / "hf_model" / "model.safetensors"
        rows.append(
            {
                "seed": s,
                "complete": True,
                "dir": str((ENC / "seed_runs" / f"longformer_seed{s}").relative_to(ROOT)).replace("\\", "/"),
                "val_accuracy": obj.get("val_accuracy"),
                "best_val_accuracy": obj.get("best_val_accuracy"),
                "finished_at": obj.get("finished_at"),
                "checkpoint_sha256": sha256_file(ckpt) if ckpt.is_file() else None,
                "run_meta_sha256": sha256_file(meta_p),
            }
        )
    sdf = pd.DataFrame(rows)
    sdf.to_csv(ENC / "seed_summary.csv", index=False)
    summary = {
        "aggregation_rule": "mean/SD/min/max across seeds 0,1,2; do not pick a single seed",
        "n_complete": 3,
        "mean_val_accuracy": float(np.mean(accs)),
        "sd_val_accuracy": float(np.std(accs, ddof=1)),
        "min_val_accuracy": float(min(accs)),
        "max_val_accuracy": float(max(accs)),
        "seeds": rows,
        "region_rule": "heldout_tercile_precedence_v1 per seed; canonical = majority; ties = middle",
        "three_seed_region_agreement": all_three,
        "majority_vs_legacy_region": vs_legacy,
    }
    (ENC / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return canon, maj_g6, summary


def frozen_backend_counts() -> dict:
    out = {}
    for run_id in ["llm_deepseek_frozen_v1", "llm_gpt_frozen_v1", "llm_doubao_frozen_v1"]:
        p = LLM / f"{run_id}_responses.jsonl"
        if not p.is_file():
            out[run_id] = {"n": 0, "unique": 0, "parse_ok": 0, "complete": False}
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        # last attempt per question
        by_q = {}
        for r in rows:
            q = r["question_id"]
            ok = bool(r.get("parse_success")) and str(r.get("parsed_option") or "")[:1] in LETTER
            prev = by_q.get(q)
            if prev is None:
                by_q[q] = r
                continue
            prev_ok = bool(prev.get("parse_success")) and str(prev.get("parsed_option") or "")[:1] in LETTER
            if ok and not prev_ok:
                by_q[q] = r
            elif ok == prev_ok and int(r.get("attempt_index") or 0) >= int(prev.get("attempt_index") or 0):
                by_q[q] = r
        parse_ok = sum(1 for r in by_q.values() if r.get("parse_success") and r.get("parsed_option") in LETTER)
        out[run_id] = {
            "n": len(rows),
            "unique": len(by_q),
            "parse_ok": parse_ok,
            "complete": len(by_q) == 4887,
            "parse_complete": parse_ok == 4887,
        }
    return out


def load_frozen_last() -> pd.DataFrame:
    frames = []
    for p in sorted(LLM.glob("llm_*_frozen_v1_responses.jsonl")):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    resp = pd.concat(frames, ignore_index=True)
    ok = resp[resp["parse_success"].astype(bool) & resp["parsed_option"].isin(LETTER)]
    fail = resp.sort_values("attempt_index").drop_duplicates(["question_id", "llm_run_id"], keep="last")
    resp = pd.concat([fail, ok], ignore_index=True).drop_duplicates(["question_id", "llm_run_id"], keep="last")
    return resp


def rebuild_llm_tables(items: pd.DataFrame, counts: dict):
    """Ingest frozen runs as primary; keep legacy rows marked non-reproducible."""
    # start from existing p0_2 tables if present, else empty
    legacy_runs = pd.read_parquet(PROC / "llm_runs.parquet") if (PROC / "llm_runs.parquet").is_file() else pd.DataFrame()
    legacy_resp = (
        pd.read_parquet(PROC / "llm_responses.parquet") if (PROC / "llm_responses.parquet").is_file() else pd.DataFrame()
    )

    frozen_run_rows = []
    frozen_resp_rows = []
    for run_id, info in counts.items():
        meta_p = LLM / f"{run_id}_run_meta.json"
        if not meta_p.is_file():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        frozen_run_rows.append(
            {
                "llm_run_id": run_id,
                "provider": meta.get("provider"),
                "exact_model_id": meta.get("exact_model_id"),
                "model_snapshot_or_version": meta.get("model_snapshot_or_version"),
                "access_date": meta.get("access_date"),
                "prompt_sha256": meta.get("prompt_sha256"),
                "system_prompt_sha256": meta.get("system_prompt_sha256"),
                "temperature": meta.get("temperature"),
                "top_p": meta.get("top_p"),
                "max_tokens": meta.get("max_tokens"),
                "response_format": meta.get("response_format"),
                "timeout": meta.get("timeout"),
                "max_retries": meta.get("max_retries"),
                "environment_sha256": None,
                "n_parsed_rows_ingested": info["unique"],
                "legacy_nonreproducible": False,
                "reason": None,
            }
        )
    resp = load_frozen_last()
    if len(resp):
        for _, r in resp.iterrows():
            frozen_resp_rows.append(
                {
                    "question_id": r["question_id"],
                    "llm_run_id": r["llm_run_id"],
                    "attempt_index": int(r.get("attempt_index") or 0),
                    "request_time": r.get("request_time"),
                    "retry_reason": r.get("retry_reason"),
                    "raw_response": r.get("raw_response"),
                    "raw_response_sha256": r.get("raw_response_sha256"),
                    "parse_success": bool(r.get("parse_success")),
                    "parsed_option": r.get("parsed_option"),
                    "temperature": r.get("temperature"),
                    "source_file": f"outputs/llm/{r['llm_run_id']}_responses.jsonl",
                }
            )

    runs = pd.concat([legacy_runs, pd.DataFrame(frozen_run_rows)], ignore_index=True, sort=False)
    # drop duplicate run ids preferring frozen
    runs = runs.drop_duplicates("llm_run_id", keep="last")
    resps = pd.concat([legacy_resp, pd.DataFrame(frozen_resp_rows)], ignore_index=True, sort=False)

    complete3 = all(counts[k]["complete"] for k in counts)
    gold = items.set_index("question_id")["answer_letter"].astype(str).str.upper().str[0]
    if complete3 and len(resp):
        wide = resp.pivot_table(index="question_id", columns="llm_run_id", values="parsed_option", aggfunc="last")
        vote_rows = []
        for qid, row in wide.iterrows():
            votes = [v for v in row.dropna().tolist() if v in LETTER]
            if not votes:
                status, opt = "no_consensus", None
            else:
                vc = pd.Series(votes).value_counts()
                if vc.iloc[0] >= 2:
                    status, opt = "consensus", vc.index[0]
                elif len(votes) == 1:
                    status, opt = "consensus_single_backend", vc.index[0]
                else:
                    status, opt = "no_consensus", None
            vote_rows.append(
                {
                    "question_id": qid,
                    "backend_vote_1": row.get("llm_deepseek_frozen_v1"),
                    "backend_vote_2": row.get("llm_gpt_frozen_v1"),
                    "backend_vote_3": row.get("llm_doubao_frozen_v1"),
                    "consensus_option": opt,
                    "consensus_status": status,
                    "llm_correct": int(opt == gold.get(qid, "")) if opt else 0,
                    "vote_source": "frozen_v1_two_of_three",
                }
            )
        votes = pd.DataFrame(vote_rows)
    else:
        votes = pd.read_parquet(PROC / "llm_votes.parquet")
        votes["vote_source"] = "legacy_pending_frozen_complete"

    runs.to_parquet(PROC / "llm_runs.parquet", index=False)
    resps.to_parquet(PROC / "llm_responses.parquet", index=False)
    votes.to_parquet(PROC / "llm_votes.parquet", index=False)
    return votes, complete3


def rebuild_encoder_tables(canon: pd.DataFrame):
    """Append official seed runs to encoder provenance tables."""
    enc_runs = pd.read_parquet(PROC / "encoder_runs.parquet") if (PROC / "encoder_runs.parquet").is_file() else pd.DataFrame()
    enc_epoch = (
        pd.read_parquet(PROC / "encoder_epoch_predictions.parquet")
        if (PROC / "encoder_epoch_predictions.parquet").is_file()
        else pd.DataFrame()
    )
    enc_sum = (
        pd.read_parquet(PROC / "encoder_item_summaries.parquet")
        if (PROC / "encoder_item_summaries.parquet").is_file()
        else pd.DataFrame()
    )

    run_rows = []
    epoch_rows = []
    summary_rows = []
    for s in SEEDS:
        d = ENC / "seed_runs" / f"longformer_seed{s}"
        meta_p = d / "run_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        run_id = f"enc_longformer_seed{s}"
        ckpt = d / "model_epoch4.pt"
        if not ckpt.is_file():
            ckpt = d / "hf_model" / "model.safetensors"
        run_rows.append(
            {
                "encoder_run_id": run_id,
                "architecture": meta.get("model_name"),
                "exact_checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/") if ckpt.is_file() else None,
                "library_version": None,
                "seed": s,
                "training_config_sha256": sha256_file(meta_p),
                "start_time": None,
                "end_time": meta.get("finished_at"),
                "hardware": "runpod_rtx4090_bs8_accum2_amp_gradckpt",
                "best_checkpoint_rule": "best_val_accuracy_in_run_meta",
                "checkpoint_sha256": sha256_file(ckpt) if ckpt.is_file() else None,
                "reported_val_accuracy": meta.get("val_accuracy"),
                "max_len": meta.get("max_len"),
                "article_words": meta.get("article_words"),
                "epochs": meta.get("epochs"),
                "trainer": meta.get("trainer"),
                "provenance_complete": True,
                "missing_fields": "library_version,start_time",
            }
        )
        td = pd.read_csv(d / "training_dynamics_val.csv")
        for _, r in td.iterrows():
            epoch_rows.append(
                {
                    "question_id": r["question_id"],
                    "encoder_run_id": run_id,
                    "epoch": int(r["epoch"]),
                    "gold_probability": float(r["prob_correct"]),
                    "predicted_option": None,
                    "encoder_correct": int(r["is_correct"]),
                }
            )
        sub = canon[
            [
                "question_id",
                f"region_seed{s}",
            ]
        ].rename(columns={f"region_seed{s}": "region_label"})
        dyn = (
            td.groupby("question_id")
            .agg(
                mean_gold_probability=("prob_correct", "mean"),
                std_gold_probability=("prob_correct", "std"),
                fraction_epochs_correct=("is_correct", "mean"),
            )
            .reset_index()
        )
        dyn["encoder_run_id"] = run_id
        dyn["region_rule_id"] = "heldout_tercile_precedence_v1"
        dyn = dyn.merge(sub, on="question_id", how="left")
        summary_rows.extend(dyn.to_dict("records"))

    enc_runs = pd.concat([enc_runs, pd.DataFrame(run_rows)], ignore_index=True, sort=False)
    enc_runs = enc_runs.drop_duplicates("encoder_run_id", keep="last")
    if len(enc_epoch):
        enc_epoch = enc_epoch[~enc_epoch.encoder_run_id.isin([f"enc_longformer_seed{s}" for s in SEEDS])]
    enc_epoch = pd.concat([enc_epoch, pd.DataFrame(epoch_rows)], ignore_index=True, sort=False)
    if len(enc_sum):
        enc_sum = enc_sum[~enc_sum.encoder_run_id.isin([f"enc_longformer_seed{s}" for s in SEEDS])]
    enc_sum = pd.concat([enc_sum, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)

    enc_runs.to_parquet(PROC / "encoder_runs.parquet", index=False)
    enc_epoch.to_parquet(PROC / "encoder_epoch_predictions.parquet", index=False)
    enc_sum.to_parquet(PROC / "encoder_item_summaries.parquet", index=False)
    return enc_runs, enc_sum


def rebuild_integrated(items: pd.DataFrame, canon: pd.DataFrame, votes: pd.DataFrame):
    race_items = pd.read_parquet(PROC / "race_items.parquet")
    integ_src = pd.read_csv(
        ROOT / "revision/artifacts/race_val_integrated.csv",
        usecols=[
            "question_id",
            "designer_difficulty_str",
            "passage_approx_tokens",
            "likely_truncated_2048",
            "datamap_region",
        ],
    )
    out = race_items.merge(canon, on="question_id", how="left")
    out = out.merge(votes, on="question_id", how="left")
    out = out.merge(integ_src, on="question_id", how="left")
    if "datamap_region" in out.columns:
        out["legacy_datamap_region"] = out["datamap_region"]
    out["region_label"] = out["region_majority"]
    out["datamap_region"] = out["region_majority"]
    out["enc_correct"] = out["encoder_correct_mean"]
    out["encoder_correct"] = out["encoder_correct_mean"]
    if "grade_band" not in out.columns:
        out["grade_band"] = out["designer_difficulty_str"]
    if "designer_difficulty_str" not in out.columns:
        out["designer_difficulty_str"] = out["grade_band"]
    if "passage_length_tokens" not in out.columns and "passage_approx_tokens" in out.columns:
        out["passage_length_tokens"] = out["passage_approx_tokens"]
    if "truncated" not in out.columns and "likely_truncated_2048" in out.columns:
        out["truncated"] = out["likely_truncated_2048"]

    assert len(out) == 4887, len(out)
    assert out.question_id.is_unique
    out.to_parquet(PROC / "race_analysis_integrated.parquet", index=False)

    flow = [
        {"stage": "race_items", "n": int(len(race_items))},
        {"stage": "official_seed_items", "n": int(canon.question_id.nunique())},
        {"stage": "llm_votes", "n": int(len(votes))},
        {"stage": "integrated", "n": int(len(out))},
        {"stage": "integrated_with_region", "n": int(out["region_label"].notna().sum())},
        {"stage": "llm_consensus", "n": int(out["consensus_status"].astype(str).str.startswith("consensus").sum())},
    ]
    pd.DataFrame(flow).to_csv(DIAG / "subset_flow.csv", index=False)
    assertions = {
        "race_items_unique": bool(race_items.question_id.is_unique),
        "assert_4887": int(len(out)) == 4887,
        "assert_m_h": int((out["grade_band"] == "MIDDLE").sum()) == 1436
        and int((out["grade_band"] == "HIGH").sum()) == 3451,
        "integrated_equals_items": int(len(out)) == int(len(race_items)),
        "no_duplicate_integrated": int(len(out) - out.question_id.nunique()) == 0,
        "unmatched_region_zero": int(out["region_label"].isna().sum()) == 0,
        "consensus_plus_nocon_equals_llm_votes": int(len(votes))
        == int(votes["consensus_status"].astype(str).str.startswith("consensus").sum())
        + int((~votes["consensus_status"].astype(str).str.startswith("consensus")).sum()),
    }
    join_audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "race_items_n": int(len(race_items)),
        "integrated_n": int(len(out)),
        "vote_source": str(votes["vote_source"].iloc[0]) if "vote_source" in votes.columns and len(votes) else None,
        "region_source": "majority of official Longformer seeds 0/1/2",
        "assertions": assertions,
        "all_assertions_pass": bool(all(assertions.values())),
        "pass": bool(all(assertions.values())),
        "g5_requires_frozen_rerun": False,
    }
    (DIAG / "join_audit.json").write_text(json.dumps(join_audit, indent=2), encoding="utf-8")
    return out, join_audit


def update_encoder_validation(summary: dict, maj_g6: dict):
    md = EVID / "encoder_validation.md"
    prev = md.read_text(encoding="utf-8") if md.is_file() else ""
    extra = f"""
## Official 3-seed Longformer (P0-3 / G4)

Aggregation rule: mean/SD/min/max across seeds 0, 1, 2. Do **not** pick a single seed.

| Seed | val acc | best | finished_at UTC |
|---|---:|---:|---|
| 0 | {summary['seeds'][0]['val_accuracy']:.6f} | {summary['seeds'][0]['best_val_accuracy']:.6f} | {summary['seeds'][0].get('finished_at')} |
| 1 | {summary['seeds'][1]['val_accuracy']:.6f} | {summary['seeds'][1]['best_val_accuracy']:.6f} | {summary['seeds'][1].get('finished_at')} |
| 2 | {summary['seeds'][2]['val_accuracy']:.6f} | {summary['seeds'][2]['best_val_accuracy']:.6f} | {summary['seeds'][2].get('finished_at')} |

- mean = {summary['mean_val_accuracy']:.6f}
- SD = {summary['sd_val_accuracy']:.6f}
- min = {summary['min_val_accuracy']:.6f}
- max = {summary['max_val_accuracy']:.6f}

Region rule: held-out tercile precedence per seed; canonical region = majority; ties = middle.
Three-seed exact region agreement = {maj_g6['three_seed_region_agreement']:.4f}.
Majority vs legacy integrated region = {maj_g6['majority_vs_legacy_region']:.4f}.

Artifacts: `outputs/encoder/seed_runs/longformer_seed{{0,1,2}}/`, `seed_summary.csv`, `region_stability.csv`, `seed_g6_g8.csv`.
"""
    if "## Official 3-seed Longformer" in prev:
        head = prev.split("## Official 3-seed Longformer")[0].rstrip()
        md.write_text(head + "\n" + extra, encoding="utf-8")
    else:
        md.write_text(prev.rstrip() + "\n" + extra, encoding="utf-8")


def main():
    items = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv", usecols=["question_id", "answer_letter", "designer_difficulty_str"])
    canon, maj_g6, summary = compute_seed_regions_and_stability(items)
    rebuild_encoder_tables(canon)
    update_encoder_validation(summary, maj_g6)
    counts = frozen_backend_counts()
    (DIAG / "frozen_llm_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    votes, complete3 = rebuild_llm_tables(items, counts)
    integ, join_audit = rebuild_integrated(items, canon, votes)
    print(
        json.dumps(
            {
                "seed_summary": {
                    "mean": summary["mean_val_accuracy"],
                    "sd": summary["sd_val_accuracy"],
                    "min": summary["min_val_accuracy"],
                    "max": summary["max_val_accuracy"],
                },
                "g6_encoder_multiseed": maj_g6,
                "frozen_llm_counts": counts,
                "frozen_complete3": complete3,
                "join_audit": join_audit,
                "integrated_n": int(len(integ)),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
