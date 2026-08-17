#!/usr/bin/env python3
"""P0-2: Build normalized provenance tables without fabricating metadata."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
DIAG = ROOT / "outputs/diagnostics"
EVID = ROOT / "audit/evidence"
for d in [PROC, DIAG, EVID]:
    d.mkdir(parents=True, exist_ok=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    # --- race items from raw + integrated ---
    man = EVID / "race_raw_manifest.csv"
    if not man.is_file():
        import subprocess

        subprocess.check_call(["python", "audit/recompute/a1_race_raw_split_audit.py"], cwd=ROOT)
    raw = pd.read_csv(man)
    dev = raw[(raw.split == "dev") & (raw.valid)].copy()
    integ = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv")
    race_sha = {
        "dev_mid": sha256(ROOT / "data/RACE/dev_mid.jsonl"),
        "dev_high": sha256(ROOT / "data/RACE/dev_high.jsonl"),
    }
    items = dev.rename(
        columns={
            "grade_band": "grade_band",
            "source_file": "source_file",
            "question_index": "question_index",
            "gold_option": "gold_option",
        }
    )[
        ["question_id", "split", "grade_band", "source_file", "question_index", "gold_option"]
    ].merge(
        integ[
            [
                "question_id",
                "passage_approx_tokens",
                "likely_truncated_2048",
            ]
        ],
        on="question_id",
        how="left",
    )
    items = items.rename(
        columns={
            "passage_approx_tokens": "passage_length_tokens",
            "likely_truncated_2048": "truncated",
        }
    )
    items["raw_data_sha256"] = items["source_file"].map(
        {"dev_mid.jsonl": race_sha["dev_mid"], "dev_high.jsonl": race_sha["dev_high"]}
    )
    assert len(items) == 4887, len(items)
    assert items.question_id.is_unique
    items.to_parquet(PROC / "race_items.parquet", index=False)

    # --- encoder runs ---
    enc_root = ROOT / "revision/artifacts/encoder_competitive"
    run_rows = []
    epoch_rows = []
    summary_rows = []
    for meta_path in enc_root.rglob("run_meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_dir = meta_path.parent
        run_id = f"enc_{model_dir.name}"
        ckpts = sorted(model_dir.glob("model_epoch*.pt"))
        best = model_dir / "model_epoch4.pt" if (model_dir / "model_epoch4.pt").is_file() else (ckpts[-1] if ckpts else None)
        run_rows.append(
            {
                "encoder_run_id": run_id,
                "architecture": meta.get("model_name"),
                "exact_checkpoint": str(best.relative_to(ROOT)).replace("\\", "/") if best else None,
                "library_version": None,  # not recorded historically — do not invent
                "seed": None,  # not in run_meta — do not invent
                "training_config_sha256": sha256(meta_path),
                "start_time": None,
                "end_time": meta.get("finished_at"),
                "hardware": None,
                "best_checkpoint_rule": "last_epoch_equals_reported_best_in_run_meta",
                "checkpoint_sha256": sha256(best) if best and best.is_file() else None,
                "reported_val_accuracy": meta.get("val_accuracy"),
                "max_len": meta.get("max_len"),
                "article_words": meta.get("article_words"),
                "epochs": meta.get("epochs"),
                "trainer": meta.get("trainer"),
                "provenance_complete": False,
                "missing_fields": "seed,library_version,hardware,start_time",
            }
        )
        td = model_dir / "training_dynamics_val.csv"
        if td.is_file():
            dyn = pd.read_csv(td)
            # expected columns question_id, epoch, prob_correct, is_correct
            for _, r in dyn.iterrows():
                epoch_rows.append(
                    {
                        "question_id": r["question_id"],
                        "encoder_run_id": run_id,
                        "epoch": int(r["epoch"]),
                        "gold_probability": float(r["prob_correct"]),
                        "predicted_option": None,  # not stored in dynamics file
                        "encoder_correct": int(r["is_correct"]),
                    }
                )
            g = dyn.groupby("question_id")
            summ = g.agg(
                mean_gold_probability=("prob_correct", "mean"),
                std_gold_probability=("prob_correct", "std"),
                fraction_epochs_correct=("is_correct", "mean"),
            ).reset_index()
            summ["encoder_run_id"] = run_id
            summ["region_rule_id"] = "heldout_tercile_precedence_v1" if "longformer" in run_id else None
            if "longformer" in run_id:
                # attach manuscript regions from integrated
                summ = summ.merge(
                    integ[["question_id", "datamap_region"]], on="question_id", how="left"
                ).rename(columns={"datamap_region": "region_label"})
            else:
                summ["region_label"] = None
            summary_rows.extend(summ.to_dict("records"))

    enc_runs = pd.DataFrame(run_rows)
    enc_epoch = pd.DataFrame(epoch_rows)
    enc_sum = pd.DataFrame(summary_rows)
    enc_runs.to_parquet(PROC / "encoder_runs.parquet", index=False)
    enc_epoch.to_parquet(PROC / "encoder_epoch_predictions.parquet", index=False)
    enc_sum.to_parquet(PROC / "encoder_item_summaries.parquet", index=False)

    # --- LLM runs / responses from available logs (nulls where unknown) ---
    llm_run_rows = []
    llm_resp_rows = []
    # Document known backends from folder names + response letter; do not invent access dates
    backends = [
        {
            "llm_run_id": "llm_gpt4o_legacy_val",
            "provider": None,
            "exact_model_id": None,
            "model_snapshot_or_version": "gpt4o_1124_folder_label_only",
            "access_date": None,
            "path_glob": "LLM_out/gpt4o_1124/race_llm_prompts_val_gpt*.jsonl",
            "legacy_nonreproducible": True,
        },
        {
            "llm_run_id": "llm_doubao_legacy_val",
            "provider": None,
            "exact_model_id": None,
            "model_snapshot_or_version": "doubao_1.8_folder_label_only",
            "access_date": None,
            "path_glob": "LLM_out/doubao_1.8/race_llm_prompts_val_doubao*.jsonl",
            "legacy_nonreproducible": True,
        },
        {
            "llm_run_id": "llm_deepseek_integrated_cols",
            "provider": None,
            "exact_model_id": None,
            "model_snapshot_or_version": None,
            "access_date": None,
            "path_glob": None,
            "legacy_nonreproducible": True,
        },
    ]

    def ingest_jsonl(run_id, paths):
        n = 0
        for p in paths:
            if not p.is_file():
                continue
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = obj.get("id") or obj.get("question_id")
                    raw = obj.get("raw") or obj.get("response") or obj.get("text")
                    # many logs only store pred_answer
                    pred = obj.get("pred_answer") or obj.get("llm_label") or obj.get("letter")
                    raw_s = json.dumps(obj, ensure_ascii=False) if raw is None else str(raw)
                    llm_resp_rows.append(
                        {
                            "question_id": qid,
                            "llm_run_id": run_id,
                            "attempt_index": int(obj.get("n_retries") or obj.get("attempt") or 0),
                            "request_time": obj.get("timestamp") or obj.get("request_time"),
                            "retry_reason": obj.get("retry_reason"),
                            "raw_response": raw if raw is not None else None,
                            "raw_response_sha256": hashlib.sha256(raw_s.encode("utf-8", errors="ignore")).hexdigest(),
                            "parse_success": pred is not None and str(pred).strip() != "",
                            "parsed_option": str(pred).strip().upper()[:1] if pred is not None else None,
                            "temperature": obj.get("temperature"),
                            "source_file": str(p.relative_to(ROOT)).replace("\\", "/"),
                        }
                    )
                    n += 1
        return n

    for b in backends:
        paths = list(ROOT.glob(b["path_glob"])) if b.get("path_glob") else []
        n = ingest_jsonl(b["llm_run_id"], paths) if paths else 0
        # prefer merged files
        llm_run_rows.append(
            {
                "llm_run_id": b["llm_run_id"],
                "provider": b["provider"],
                "exact_model_id": b["exact_model_id"],
                "model_snapshot_or_version": b["model_snapshot_or_version"],
                "access_date": b["access_date"],
                "prompt_sha256": None,
                "system_prompt_sha256": None,
                "temperature": None,
                "top_p": None,
                "max_tokens": None,
                "response_format": "letter_only_claimed",
                "timeout": None,
                "max_retries": None,
                "environment_sha256": None,
                "n_parsed_rows_ingested": n,
                "legacy_nonreproducible": True,
                "reason": "missing provider/exact_model_id/access_date/frozen decoding in historical logs",
            }
        )

    # Votes from integrated (auditable derived consolidation)
    votes = integ[
        [
            "question_id",
            "llm_gpt4o_letter",
            "llm_doubao_letter",
            "llm_deepseek_letter",
            "llm_no_consensus",
            "llm_letter",
            "llm_correct",
        ]
    ].rename(
        columns={
            "llm_gpt4o_letter": "backend_vote_1",
            "llm_doubao_letter": "backend_vote_2",
            "llm_deepseek_letter": "backend_vote_3",
            "llm_letter": "consensus_option",
        }
    )
    votes["consensus_status"] = np.where(votes["llm_no_consensus"].fillna(0).astype(int) == 1, "no_consensus", "consensus")
    votes = votes.drop(columns=["llm_no_consensus"])

    pd.DataFrame(llm_run_rows).to_parquet(PROC / "llm_runs.parquet", index=False)
    pd.DataFrame(llm_resp_rows).to_parquet(PROC / "llm_responses.parquet", index=False)
    votes.to_parquet(PROC / "llm_votes.parquet", index=False)

    # Integrated analysis table (join)
    longformer_sum = enc_sum[enc_sum.encoder_run_id.str.contains("longformer", na=False)]
    integrated = items.merge(longformer_sum, on="question_id", how="left").merge(votes, on="question_id", how="left")
    # also attach encoder final from integ
    integrated = integrated.merge(
        integ[["question_id", "enc_pred", "enc_correct", "mean_prob", "std_prob", "frac_correct"]],
        on="question_id",
        how="left",
    )
    integrated.to_parquet(PROC / "race_analysis_integrated.parquet", index=False)

    # Join audit / subset flow
    flow = [
        {"stage": "raw_dev_valid", "n": int(len(dev))},
        {"stage": "race_items", "n": int(len(items))},
        {"stage": "longformer_summaries", "n": int(longformer_sum.question_id.nunique())},
        {"stage": "llm_votes", "n": int(len(votes))},
        {"stage": "integrated", "n": int(len(integrated))},
        {"stage": "integrated_with_region", "n": int(integrated["region_label"].notna().sum())},
        {
            "stage": "llm_consensus",
            "n": int((integrated["consensus_status"] == "consensus").sum()),
        },
    ]
    pd.DataFrame(flow).to_csv(DIAG / "subset_flow.csv", index=False)

    assertions = {
        "race_items_unique": bool(items.question_id.is_unique),
        "assert_4887": int(len(items)) == 4887,
        "assert_m_h": int((items.grade_band == "MIDDLE").sum()) == 1436
        and int((items.grade_band == "HIGH").sum()) == 3451,
        "integrated_equals_items": int(len(integrated)) == int(len(items)),
        "no_duplicate_integrated": int(len(integrated) - integrated.question_id.nunique()) == 0,
        "unmatched_region_zero": int(integrated["region_label"].isna().sum()) == 0,
        "consensus_plus_nocon_equals_llm_votes": int(len(votes))
        == int((votes["consensus_status"] == "consensus").sum())
        + int((votes["consensus_status"] == "no_consensus").sum()),
    }
    join_audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "race_items_n": int(len(items)),
        "encoder_runs_n": int(len(enc_runs)),
        "encoder_epoch_rows": int(len(enc_epoch)),
        "llm_runs_all_legacy_nonreproducible": bool(pd.DataFrame(llm_run_rows)["legacy_nonreproducible"].all())
        if llm_run_rows
        else True,
        "llm_response_rows_ingested": int(len(llm_resp_rows)),
        "integrated_n": int(len(integrated)),
        "unmatched_region": int(integrated["region_label"].isna().sum()),
        "duplicate_integrated": int(len(integrated) - integrated.question_id.nunique()),
        "assertions": assertions,
        "all_assertions_pass": bool(all(assertions.values())),
        "pass": bool(all(assertions.values())),
        "g5_requires_frozen_rerun": True,
        "note": "Legacy LLM metadata left null/non-reproducible; frozen rerun updates llm_* tables later.",
    }
    (DIAG / "join_audit.json").write_text(json.dumps(join_audit, indent=2), encoding="utf-8")
    print(json.dumps(join_audit, indent=2))


if __name__ == "__main__":
    main()
