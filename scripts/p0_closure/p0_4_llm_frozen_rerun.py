#!/usr/bin/env python3
"""P0-4: Frozen-protocol LLM rerun (DeepSeek primary; others if keys work)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/llm"
PROC = ROOT / "data/processed"
EVID = ROOT / "audit/evidence"
OUT.mkdir(parents=True, exist_ok=True)
EVID.mkdir(parents=True, exist_ok=True)

PROTOCOL = yaml.safe_load((ROOT / "configs/llm_protocol.yaml").read_text(encoding="utf-8"))
PROMPT_TMPL = (ROOT / "prompts/race_mcq_prompt.txt").read_text(encoding="utf-8")


def load_dotenv():
    envp = ROOT / ".env"
    if not envp.is_file():
        return
    for line in envp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_letter(text: str):
    if not text:
        return None
    s = str(text).strip().upper()
    if s[:1] in "ABCD":
        return s[:1]
    m = re.search(r"\b([ABCD])\b", s)
    return m.group(1) if m else None


def render(row):
    return (
        PROMPT_TMPL.replace("{{passage}}", str(row["article"])[:3500])
        .replace("{{question}}", str(row["question"]))
        .replace("{{option_a}}", str(row["option_a"]))
        .replace("{{option_b}}", str(row["option_b"]))
        .replace("{{option_c}}", str(row["option_c"]))
        .replace("{{option_d}}", str(row["option_d"]))
    )


def run_backend(backend_cfg, items: pd.DataFrame, workers=8, retry_unparsed=False):
    load_dotenv()
    key = os.environ.get(backend_cfg["api_key_env"])
    if not key:
        return None, f"missing {backend_cfg['api_key_env']}"
    base = os.environ.get(backend_cfg.get("base_url_env", ""), None)
    model = os.environ.get(backend_cfg["exact_model_id_env"])
    if not model:
        return None, f"missing {backend_cfg['exact_model_id_env']}"
    client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
    dec = PROTOCOL["decoding"]
    access_date = datetime.now(timezone.utc).date().isoformat()
    run_id = backend_cfg["llm_run_id"]
    cache = OUT / f"{run_id}_responses.jsonl"
    done = set()
    last_ok = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec["question_id"]
            done.add(qid)
            ok = bool(rec.get("parse_success")) and str(rec.get("parsed_option") or "")[:1] in "ABCD"
            prev = last_ok.get(qid)
            if prev is None or int(rec.get("attempt_index") or 0) >= int(prev[0]):
                last_ok[qid] = (int(rec.get("attempt_index") or 0), ok)
    if retry_unparsed:
        # keep any historically successful parse; do not rerun those
        any_ok = set()
        if cache.is_file():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if bool(rec.get("parse_success")) and str(rec.get("parsed_option") or "")[:1] in "ABCD":
                    any_ok.add(rec["question_id"])
        done = any_ok

    run_meta = {
        "llm_run_id": run_id,
        "provider": backend_cfg["provider"],
        "exact_model_id": model,
        "model_snapshot_or_version": model,
        "access_date": access_date,
        "prompt_sha256": hashlib.sha256(PROMPT_TMPL.encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(b"").hexdigest(),
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
        "max_tokens": dec["max_tokens"],
        "response_format": PROTOCOL["response_format"],
        "timeout": PROTOCOL["timeout_seconds"],
        "max_retries": PROTOCOL["max_retries"],
        "protocol_id": PROTOCOL["protocol_id"],
        "legacy_nonreproducible": False,
    }
    (OUT / f"{run_id}_run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    jobs = [r for _, r in items.iterrows() if r["question_id"] not in done]
    print(f"[{run_id}] pending={len(jobs)} cached={len(done)} model={model}")

    def one(row):
        prompt = render(row)
        last_err = None
        for attempt in range(PROTOCOL["max_retries"] + 1):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=dec["temperature"],
                    top_p=dec["top_p"],
                    max_tokens=dec["max_tokens"],
                    timeout=PROTOCOL["timeout_seconds"],
                )
                raw = (resp.choices[0].message.content or "").strip()
                letter = parse_letter(raw)
                return {
                    "question_id": row["question_id"],
                    "llm_run_id": run_id,
                    "attempt_index": attempt,
                    "request_time": datetime.now(timezone.utc).isoformat(),
                    "retry_reason": None if attempt == 0 else "parse_or_transport_retry_same_decoding",
                    "raw_response": raw,
                    "raw_response_sha256": hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest(),
                    "parse_success": letter is not None,
                    "parsed_option": letter,
                    "latency_s": time.time() - t0,
                    "temperature": dec["temperature"],
                }
            except Exception as e:
                last_err = str(e)
                if attempt >= PROTOCOL["max_retries"]:
                    return {
                        "question_id": row["question_id"],
                        "llm_run_id": run_id,
                        "attempt_index": attempt,
                        "request_time": datetime.now(timezone.utc).isoformat(),
                        "retry_reason": "transport_error",
                        "raw_response": last_err[:500],
                        "raw_response_sha256": hashlib.sha256(last_err.encode()).hexdigest(),
                        "parse_success": False,
                        "parsed_option": None,
                        "temperature": dec["temperature"],
                    }
        return None

    # Chunked submit: avoid thousands of queued futures / rate-limit storms
    chunk = max(workers * 8, 16)
    done_n = 0
    with cache.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for start in range(0, len(jobs), chunk):
                batch = jobs[start : start + chunk]
                futs = [ex.submit(one, row) for row in batch]
                for fut in as_completed(futs):
                    rec = fut.result()
                    if rec is None:
                        continue
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done_n += 1
                    if done_n % 25 == 0:
                        fh.flush()
                        print(f"[{run_id}] {done_n}/{len(jobs)}", flush=True)
            fh.flush()
    print(f"[{run_id}] finished wrote={done_n}", flush=True)
    return run_meta, None


def aggregate_and_metrics(items: pd.DataFrame):
    # Load frozen deepseek (+ others if present)
    resp_frames = []
    for p in OUT.glob("llm_*_frozen_v1_responses.jsonl"):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        resp_frames.append(pd.DataFrame(rows))
    if not resp_frames:
        return
    resp = pd.concat(resp_frames, ignore_index=True)
    # keep last attempt per question/run
    resp = resp.sort_values(["parse_success", "attempt_index"], ascending=[True, True])
    # prefer last successful parse; fall back to last attempt if none succeeded
    ok = resp[resp["parse_success"].astype(bool) & resp["parsed_option"].isin(list("ABCD"))]
    fail = resp.drop_duplicates(["question_id", "llm_run_id"], keep="last")
    resp = pd.concat([fail, ok], ignore_index=True).drop_duplicates(["question_id", "llm_run_id"], keep="last")
    gold = items.set_index("question_id")["answer_letter"].astype(str).str.upper().str[0]

    backend_rows = []
    for run_id, g in resp.groupby("llm_run_id"):
        g = g.copy()
        g["correct"] = [
            int(str(o) == gold.get(q, "")) if o in list("ABCD") else 0
            for q, o in zip(g.question_id, g.parsed_option.fillna(""))
        ]
        backend_rows.append(
            {
                "llm_run_id": run_id,
                "n": int(len(g)),
                "parse_success_rate": float(g.parse_success.mean()),
                "accuracy_parsed": float(g.loc[g.parse_success, "correct"].mean()) if g.parse_success.any() else None,
                "retry_rate": float((g.attempt_index > 0).mean()),
            }
        )
    pd.DataFrame(backend_rows).to_csv(OUT / "backend_metrics.csv", index=False)

    # If only one backend, consensus = that backend
    wide = resp.pivot_table(index="question_id", columns="llm_run_id", values="parsed_option", aggfunc="last")
    cons_rows = []
    for qid, row in wide.iterrows():
        votes = [v for v in row.dropna().tolist() if v in list("ABCD")]
        if not votes:
            status, opt = "no_consensus", None
        else:
            # majority
            vc = pd.Series(votes).value_counts()
            if vc.iloc[0] >= 2 or len(votes) == 1:
                status, opt = ("consensus" if len(set(votes)) == 1 or vc.iloc[0] >= 2 else "no_consensus"), vc.index[0]
                if len(votes) == 1:
                    status = "consensus_single_backend"
            else:
                status, opt = "no_consensus", None
        cons_rows.append(
            {
                "question_id": qid,
                "consensus_status": status,
                "consensus_option": opt,
                "llm_correct": int(opt == gold.get(qid, "")) if opt else 0,
                "n_votes": len(votes),
            }
        )
    cons = pd.DataFrame(cons_rows)
    cons = cons.merge(items[["question_id", "designer_difficulty_str"]], on="question_id", how="left")
    # attach longformer region if available
    integ = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv", usecols=["question_id", "datamap_region"])
    cons = cons.merge(integ, on="question_id", how="left")
    cons.to_csv(OUT / "consensus_by_item.csv", index=False)

    # Pairwise backend agreement
    cols = list(wide.columns)
    pair_rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            both = wide[[a, b]].dropna()
            both = both[both[a].isin(list("ABCD")) & both[b].isin(list("ABCD"))]
            pair_rows.append(
                {
                    "backend_a": a,
                    "backend_b": b,
                    "n_paired": int(len(both)),
                    "agreement_rate": float((both[a] == both[b]).mean()) if len(both) else None,
                }
            )
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(OUT / "pairwise_agreement.csv", index=False)

    summary = {
        "n_items": int(len(cons)),
        "n_consensus": int(cons.consensus_status.str.startswith("consensus").sum()),
        "n_no_consensus": int((~cons.consensus_status.str.startswith("consensus")).sum()),
        "three_way_full_agreement": int(
            (
                (wide.nunique(axis=1) == 1)
                & (wide.notna().sum(axis=1) == wide.shape[1])
                & wide.apply(lambda r: all(v in list("ABCD") for v in r.dropna()), axis=1)
            ).sum()
        )
        if wide.shape[1] >= 3
        else None,
        "acc_consensus_conditioned": float(
            cons.loc[cons.consensus_status.str.startswith("consensus"), "llm_correct"].mean()
        )
        if cons.consensus_status.str.startswith("consensus").any()
        else None,
        "acc_unconditional_nocon_as_incorrect": float(cons["llm_correct"].mean()),
        "n_backends_present": int(wide.shape[1]),
    }
    pd.DataFrame([summary]).to_csv(OUT / "consensus_metrics.csv", index=False)

    nocon = cons.copy()
    nocon["is_nocon"] = ~nocon.consensus_status.str.startswith("consensus")
    overall = pd.DataFrame(
        [
            {
                "n_items": int(len(cons)),
                "n_consensus": int((~nocon.is_nocon).sum()),
                "n_no_consensus": int(nocon.is_nocon.sum()),
                "no_consensus_rate": float(nocon.is_nocon.mean()),
                "acc_consensus_only": float(cons.loc[~nocon.is_nocon, "llm_correct"].mean()) if (~nocon.is_nocon).any() else None,
                "acc_unconditional_nocon_incorrect": float(cons["llm_correct"].mean()),
            }
        ]
    )
    by_band = nocon.groupby("designer_difficulty_str")["is_nocon"].mean().reset_index(name="no_consensus_rate")
    by_region = nocon.groupby("datamap_region")["is_nocon"].mean().reset_index(name="no_consensus_rate")
    pd.concat(
        [overall.assign(scope="overall"), by_band.assign(scope="by_band"), by_region.assign(scope="by_region")],
        ignore_index=True,
        sort=False,
    ).to_csv(OUT / "no_consensus_analysis.csv", index=False)

    retry = resp.assign(retried=resp.attempt_index > 0)
    retry.groupby("retried").agg(n=("question_id", "count"), parse_rate=("parse_success", "mean")).reset_index().to_csv(
        OUT / "retry_analysis.csv", index=False
    )

    (EVID / "llm_reproducibility.md").write_text(
        f"""# LLM reproducibility (P0-4)

## Frozen protocol

- `configs/llm_protocol.yaml`
- `prompts/race_mcq_prompt.txt`
- Decoding fixed: temperature=0.0, top_p=1.0, max_tokens=4
- Retries do **not** change decoding parameters

## Legacy runs

Historical `LLM_out/*` and integrated vote columns are marked **legacy_nonreproducible**
(missing exact provider/model snapshot/access_date/frozen decoding). They remain available
for exploratory comparison only and are **not** used for G5 PASS.

## Frozen rerun artifacts

- `outputs/llm/*_run_meta.json`
- `outputs/llm/*_responses.jsonl`
- `outputs/llm/backend_metrics.csv`
- `outputs/llm/consensus_metrics.csv`
- `outputs/llm/no_consensus_analysis.csv`
- `outputs/llm/retry_analysis.csv`

## Contamination note

RACE is a public benchmark. High LLM accuracy may reflect benchmark exposure/contamination
and must not be interpreted as pure general reasoning ability.

Generated: {datetime.now(timezone.utc).isoformat()}
""",
        encoding="utf-8",
    )


def main():
    load_dotenv()
    items = pd.read_csv(ROOT / "revision/artifacts/race_val_integrated.csv")
    # Prefer DeepSeek frozen rerun for G5 closure; attempt others if keys exist
    for b in PROTOCOL["backends"]:
        if b["provider"] != "deepseek":
            # still try if key present but DeepSeek first priority
            continue
        meta, err = run_backend(b, items, workers=12)
        print("deepseek", meta is not None, err)
    # optional others
    for b in PROTOCOL["backends"]:
        if b["provider"] == "deepseek":
            continue
        meta, err = run_backend(b, items, workers=8)
        print(b["llm_run_id"], meta is not None, err)
    aggregate_and_metrics(items)
    print("[OK] P0-4 metrics written")


if __name__ == "__main__":
    main()
