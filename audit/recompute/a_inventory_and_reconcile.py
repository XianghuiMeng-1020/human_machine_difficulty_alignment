#!/usr/bin/env python3
"""Inventory key artifacts + independent reconciliation from canonical integrated table."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parents[1]
EVID = AUDIT / "evidence"
EVID.mkdir(parents=True, exist_ok=True)

SUSPICIOUS = [
    5000, 3360, 1640, 1800, 1300, 750, 1150, 1530, 780, 300, 270, 520, 450, 400, 850, 1200, 2050,
    "62.3", "75.1", "78.5", "88.2",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: Path):
    if not path.is_file():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": True,
        "bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def inventory():
    patterns = [
        "data/RACE/*.jsonl",
        "data/eedi/**/*.csv",
        "race_prepared/*",
        "revision/artifacts/*",
        "revision/tables/*",
        "revision/bridge/*",
        "revision/audit/*",
        "LLM_out/**/*.jsonl",
        "scripts/revision/*.py",
        "scripts/RACE_*.py",
        "revision/STATUS.md",
        "revision/manuscript/*",
        "_revision_materials/sn-article.tex",
        "environment*.yml",
        "requirements*.txt",
        "pyproject.toml",
        "build_env.sh",
    ]
    rows = []
    seen = set()
    for pat in patterns:
        for p in ROOT.glob(pat):
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            # skip huge train csv hashing? still hash but slow - skip >200MB content hash partial
            meta = {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size,
                    "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()}
            if p.stat().st_size < 80_000_000:
                meta["sha256"] = sha256(p)
            else:
                meta["sha256"] = "SKIPPED_LARGE_FILE"
            rows.append(meta)
    out = pd.DataFrame(rows).sort_values("path")
    out.to_csv(EVID / "file_inventory.csv", index=False)
    return out


def hardcode_scan():
    hits = []
    roots = [ROOT / "scripts", ROOT / "revision", ROOT / "_revision_materials"]
    text_ext = {".py", ".md", ".tex", ".json", ".csv", ".txt", ".yml", ".yaml"}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in text_ext:
                continue
            if p.stat().st_size > 5_000_000:
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for tok in SUSPICIOUS:
                s = str(tok)
                if s in txt:
                    # classify roughly
                    rel = str(p.relative_to(ROOT)).replace("\\", "/")
                    cls = "documentation"
                    if p.suffix == ".py":
                        cls = "code"
                    elif p.suffix == ".tex":
                        cls = "manuscript_text"
                    elif "tables" in rel or rel.endswith(".csv"):
                        cls = "artifact_or_table"
                    hits.append({"token": s, "path": rel, "class_guess": cls, "count": txt.count(s)})
    df = pd.DataFrame(hits)
    if len(df):
        df.to_csv(EVID / "hardcode_scan_hits.csv", index=False)
    else:
        pd.DataFrame(columns=["token", "path", "class_guess", "count"]).to_csv(
            EVID / "hardcode_scan_hits.csv", index=False
        )
    return df


def reconcile_integrated():
    path = ROOT / "revision/artifacts/race_val_integrated.csv"
    report = {"canonical_path": str(path), "exists": path.is_file()}
    if not path.is_file():
        (EVID / "reconcile_integrated.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    df = pd.read_csv(path)
    report["n_rows"] = int(len(df))
    report["n_unique_qid"] = int(df["question_id"].nunique())
    report["duplicate_qid"] = int(len(df) - df["question_id"].nunique())
    report["sha256"] = sha256(path)
    report["columns"] = list(df.columns)

    # required fields presence
    required = [
        "question_id", "split", "designer_difficulty_str", "answer_letter", "label",
        "enc_pred", "enc_correct", "mean_prob", "std_prob", "frac_correct", "datamap_region",
        "llm_pred", "llm_no_consensus", "llm_correct",
        "llm_gpt4o_letter", "llm_doubao_letter", "llm_deepseek_letter",
    ]
    report["missing_required_fields"] = [c for c in required if c not in df.columns]
    soft = [
        "llm_exact_model_id", "llm_provider", "llm_access_date", "llm_raw_response",
        "retry_reason", "backend_vote", "epoch", "encoder_model", "seed",
    ]
    report["missing_provenance_fields"] = [c for c in soft if c not in df.columns]

    # band counts
    vc = df["designer_difficulty_str"].value_counts().to_dict()
    report["band_counts"] = {str(k): int(v) for k, v in vc.items()}

    # encoder
    enc = df.dropna(subset=["enc_correct"])
    report["encoder"] = {
        "n": int(len(enc)),
        "accuracy": float(enc["enc_correct"].mean()) if len(enc) else None,
        "by_band": enc.groupby("designer_difficulty_str")["enc_correct"].mean().astype(float).to_dict(),
    }
    # confusion if enc_pred and label
    if "enc_pred" in df.columns and "label" in df.columns:
        sub = df.dropna(subset=["enc_pred", "label"]).copy()
        sub["enc_pred"] = sub["enc_pred"].astype(int)
        sub["label"] = sub["label"].astype(int)
        cm = pd.crosstab(sub["label"], sub["enc_pred"])
        report["encoder_confusion_sum"] = int(cm.values.sum())
        report["encoder_confusion"] = cm.to_dict()
        report["assert_confusion_sum_eq_n"] = int(cm.values.sum()) == int(len(sub))

    # datamap
    if "datamap_region" in df.columns:
        report["region_counts"] = {str(k): int(v) for k, v in df["datamap_region"].value_counts(dropna=False).to_dict().items()}
        report["region_missing"] = int(df["datamap_region"].isna().sum())

    # LLM
    llm = {}
    if "llm_no_consensus" in df.columns:
        llm["n_total"] = int(len(df))
        llm["n_no_consensus"] = int(df["llm_no_consensus"].fillna(0).astype(int).sum())
        llm["n_consensus"] = int(llm["n_total"] - llm["n_no_consensus"])
        cons = df[df["llm_no_consensus"].fillna(0).astype(int) == 0]
        if "llm_correct" in cons.columns:
            llm["acc_consensus_only"] = float(cons["llm_correct"].mean())
            # unconditional: no consensus as incorrect
            unc = df.copy()
            unc["_c"] = np.where(unc["llm_no_consensus"].fillna(0).astype(int) == 1, 0, unc["llm_correct"].fillna(0))
            llm["acc_unconditional_nocon_as_incorrect"] = float(unc["_c"].mean())
        # per backend if letter present
        for name, col in [("gpt4o", "llm_gpt4o_letter"), ("doubao", "llm_doubao_letter"), ("deepseek", "llm_deepseek_letter")]:
            if col in df.columns and "answer_letter" in df.columns:
                m = df[col].notna() & df["answer_letter"].notna()
                correct = (
                    df.loc[m, col].astype(str).str.strip().str.upper().str[0]
                    == df.loc[m, "answer_letter"].astype(str).str.strip().str.upper().str[0]
                )
                llm[f"acc_{name}"] = float(correct.mean()) if m.sum() else None
                llm[f"n_{name}"] = int(m.sum())
        # retry fields
        for name, col in [("gpt4o", "llm_gpt4o_retried"), ("doubao", "llm_doubao_retried"), ("deepseek", "llm_deepseek_retried")]:
            if col in df.columns:
                llm[f"retry_rate_{name}"] = float(df[col].fillna(0).astype(float).mean())
        if "llm_any_retried" in df.columns:
            llm["retry_rate_any"] = float(df["llm_any_retried"].fillna(0).astype(float).mean())
    report["llm"] = llm

    # subset flow
    flow = []
    flow.append({"stage": "integrated_rows", "n": int(len(df))})
    flow.append({"stage": "has_enc_pred", "n": int(df["enc_pred"].notna().sum()) if "enc_pred" in df.columns else None})
    flow.append({"stage": "has_datamap_region", "n": int(df["datamap_region"].notna().sum()) if "datamap_region" in df.columns else None})
    flow.append({"stage": "llm_consensus", "n": int((df["llm_no_consensus"].fillna(0).astype(int) == 0).sum()) if "llm_no_consensus" in df.columns else None})
    flow.append({"stage": "enc_and_llm_pred", "n": int((df["enc_pred"].notna() & df["llm_pred"].notna()).sum()) if "llm_pred" in df.columns else None})
    pd.DataFrame(flow).to_csv(EVID / "subset_flow.csv", index=False)

    # assertions vs claimed tables
    claims = {
        "n_val_4887": report["n_rows"] == 4887,
        "middle_1436": report["band_counts"].get("MIDDLE") == 1436,
        "high_3451": report["band_counts"].get("HIGH") == 3451,
        "unique_qid": report["duplicate_qid"] == 0,
    }
    if report.get("region_counts"):
        # claimed 1189/1375/1148/1175 - normalize keys
        rc = {str(k).lower(): v for k, v in report["region_counts"].items()}
        claims["region_easy_1189"] = rc.get("easy") == 1189
        claims["region_ambiguous_1375"] = rc.get("ambiguous") == 1375
        claims["region_hard_1148"] = rc.get("hard") == 1148
        # middle/other naming
        mid = rc.get("middle") or rc.get("middle/other") or rc.get("middle_other")
        claims["region_middle_1175"] = mid == 1175
    if llm:
        claims["consensus_4870"] = llm.get("n_consensus") == 4870
        if llm.get("acc_consensus_only") is not None:
            claims["llm_acc_consensus_near_0.954"] = abs(llm["acc_consensus_only"] - 0.954) < 0.005
    if report["encoder"]["accuracy"] is not None:
        claims["encoder_acc_near_0.741"] = abs(report["encoder"]["accuracy"] - 0.741) < 0.01

    report["claim_checks"] = claims
    (EVID / "reconcile_integrated.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # also write compact machine assertions
    asserts = {
        "sum_band_counts": sum(report["band_counts"].values()) == report["n_rows"],
        "confusion_ok": report.get("assert_confusion_sum_eq_n"),
        "claim_checks": claims,
    }
    (EVID / "reconcile_assertions.json").write_text(json.dumps(asserts, indent=2), encoding="utf-8")
    return report


def audit_bridge_e6_eedi():
    out = {}
    # Bridge
    br = ROOT / "revision/bridge/bridge_race_responses.csv"
    bi = ROOT / "revision/bridge/bridge_race_items.csv"
    out["bridge_responses"] = file_meta(br)
    out["bridge_items"] = file_meta(bi)
    if br.is_file() and bi.is_file():
        resp = pd.read_csv(br)
        items = pd.read_csv(bi)
        out["bridge_n_responses"] = int(len(resp))
        out["bridge_n_items"] = int(items["question_id"].nunique())
        out["bridge_n_annotators"] = int(resp["annotator_id"].nunique()) if "annotator_id" in resp.columns else None
        out["bridge_attempts_per_item"] = resp.groupby("question_id").size().describe().to_dict()
        # score
        gold = items.set_index("question_id")["answer_letter"].astype(str).str.upper().str[0]
        resp = resp.copy()
        resp["ok"] = [
            int(str(c).strip().upper()[:1] == gold.get(q, ""))
            for q, c in zip(resp["question_id"], resp["chosen_letter"])
        ]
        out["bridge_mean_correct"] = float(resp["ok"].mean())

    # E6
    e6 = ROOT / "revision/audit/e6_ratings.csv"
    out["e6_ratings"] = file_meta(e6)
    if e6.is_file():
        r = pd.read_csv(e6)
        out["e6_n_rows"] = int(len(r))
        out["e6_raters"] = sorted(r["rater_id"].astype(str).unique().tolist()) if "rater_id" in r.columns else []
        if "no_flaw" in r.columns and "item_order" in r.columns:
            # any flaw = not no_flaw, need arm - from arm key
            arm = ROOT / "revision/audit/e6_arm_key_HIDDEN.csv"
            if arm.is_file():
                a = pd.read_csv(arm)
                out["e6_arm_meta"] = file_meta(arm)
                # merge if possible
                if "question_id" in r.columns and "question_id" in a.columns:
                    m = r.merge(a, on="question_id", how="left")
                    out["e6_arm_counts"] = m.get("arm", m.get("disagreement_arm", pd.Series(dtype=object))).value_counts(dropna=False).to_dict()

    # EeDi
    eedi = ROOT / "data/eedi/train_data/train_task_3_4.csv"
    out["eedi_raw"] = file_meta(eedi)
    if eedi.is_file():
        # sample read for speed then full groupby on usecols
        df = pd.read_csv(eedi, usecols=["QuestionId", "UserId", "IsCorrect"])
        out["eedi_n_attempts"] = int(len(df))
        out["eedi_n_students"] = int(df["UserId"].nunique())
        out["eedi_n_questions"] = int(df["QuestionId"].nunique())
        att = df.groupby("QuestionId").size()
        out["eedi_attempts_per_question"] = {
            "min": int(att.min()),
            "p25": float(att.quantile(0.25)),
            "median": float(att.median()),
            "p90": float(att.quantile(0.90)),
            "p95": float(att.quantile(0.95)),
            "max": int(att.max()),
        }
        # threshold sweeps
        rates = df.groupby("QuestionId")["IsCorrect"].mean()
        sweeps = []
        for nmin in [5, 10, 20, 50, 100]:
            keep = att[att >= nmin].index
            sub = rates.loc[keep]
            easy = int((sub >= 0.8).sum())
            hard = int((sub <= 0.4).sum())
            mid = int(len(sub) - easy - hard)
            sweeps.append({"min_attempts": nmin, "n_questions": int(len(sub)), "n_easy": easy, "n_mid": mid, "n_hard": hard})
        out["eedi_threshold_sweep"] = sweeps
        pd.DataFrame(sweeps).to_csv(EVID / "eedi_threshold_sweep_independent.csv", index=False)

    # preregistration search
    preg_hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt", ".pdf", ".json", ".yml"}:
            continue
        name = p.name.lower()
        if "prereg" in name or "pre-reg" in name or "osf" in name:
            preg_hits.append(str(p.relative_to(ROOT)))
    out["preregistration_filename_hits"] = preg_hits

    (EVID / "bridge_e6_eedi_audit.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def check_llm_raw_logs():
    out = {"backends": {}}
    roots = [
        ("gpt4o", ROOT / "LLM_out/gpt4o_1124"),
        ("doubao", ROOT / "LLM_out/doubao_1.8"),
        ("deepseek", ROOT / "revision/artifacts"),
    ]
    for name, d in roots:
        info = {"dir": str(d), "exists": d.is_dir() if d.is_dir() else d.exists()}
        if d.is_dir():
            files = list(d.glob("*.jsonl")) + list(d.glob("*deepseek*"))
            info["files"] = [file_meta(f) for f in files if f.is_file()][:20]
            # peek one jsonl
            for f in files:
                if f.suffix == ".jsonl" and f.stat().st_size < 50_000_000:
                    keys = set()
                    n = 0
                    temps = []
                    with f.open("r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh):
                            if i >= 200:
                                break
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            n += 1
                            keys |= set(obj.keys())
                            if "temperature" in obj:
                                temps.append(obj["temperature"])
                    info["sample_keys"] = sorted(keys)
                    info["sample_n"] = n
                    info["temperature_values_sample"] = sorted(set(temps))[:20]
                    break
        out["backends"][name] = info
    # also revision llm vote
    vote = ROOT / "revision/artifacts/llm_vote_val.csv"
    out["llm_vote_val"] = file_meta(vote)
    (EVID / "llm_raw_log_audit.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def encoder_checkpoint_audit():
    out = {}
    enc_root = ROOT / "revision/artifacts/encoder_competitive"
    out["dir_exists"] = enc_root.is_dir()
    if enc_root.is_dir():
        metas = []
        for meta in enc_root.rglob("run_meta.json"):
            obj = json.loads(meta.read_text(encoding="utf-8"))
            metas.append({"path": str(meta.relative_to(ROOT)).replace("\\", "/"), **obj})
        out["run_metas"] = metas
    # seeds?
    seeds = []
    for p in (ROOT / "scripts/revision").glob("e1*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "seed" in txt:
            seeds.append(p.name)
    out["scripts_mentioning_seed"] = seeds
    # datamap audit report if any
    ar = ROOT / "race_datamap_audit/AUDIT_REPORT.md"
    out["legacy_datamap_audit"] = file_meta(ar)
    (EVID / "encoder_checkpoint_audit.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main():
    print("[1] inventory")
    inv = inventory()
    print(" files=", len(inv))
    print("[2] hardcode scan")
    hc = hardcode_scan()
    print(" hits=", len(hc))
    print("[3] reconcile integrated")
    rep = reconcile_integrated()
    print(" n=", rep.get("n_rows"), "claims=", rep.get("claim_checks"))
    print("[4] bridge/e6/eedi")
    be = audit_bridge_e6_eedi()
    print(" eedi_attempts=", be.get("eedi_n_attempts"), "bridge=", be.get("bridge_n_responses"))
    print("[5] llm logs")
    check_llm_raw_logs()
    print("[6] encoder checkpoints")
    encoder_checkpoint_audit()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "head_commit": (ROOT / ".git/HEAD").read_text(encoding="utf-8").strip() if (ROOT / ".git/HEAD").exists() else None,
        "n_inventory_files": int(len(inv)),
        "reconcile_claim_checks": rep.get("claim_checks"),
        "missing_provenance_fields": rep.get("missing_provenance_fields"),
    }
    (EVID / "audit_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
