#!/usr/bin/env python3
"""E11 / Paper-1: Difficulty(LLM-generated) vs Difficulty(Human-created) via IRT.

Design
------
Same respondent panel answers both:
  - Human-authored Bridge-RACE anchors (stratified MIDDLE/HIGH)
  - LLM-generated MCQs from held-out RACE passages (matched target band)

Primary analysis: concurrent 1PL on the crossed matrix; compare b distributions.
Robustness: 2PL; KS / Welch / TOST-style equivalence band on mean b.

Outputs under revision/artifacts/irt/ and revision/tables/table_e11_*.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPO_ROOT,
    REVISION_ROOT,
    ensure_dir,
    parse_option_letter,
    save_table,
)
from e9_irt_model_family import (  # noqa: E402
    build_dense_matrix,
    fit_models,
    mean_sigma_link,
)
from e10_expand_machine_students import (  # noqa: E402
    LETTERS,
    build_mc_prompt,
    dump_encoder_logits,
    load_dotenv,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_human_anchors", type=int, default=80)
    ap.add_argument("--n_llm_items", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_workers", type=int, default=16)
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--skip_answer", action="store_true")
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_human_anchors(bridge_items: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    per = n // 2
    for band in ["MIDDLE", "HIGH"]:
        sub = bridge_items[bridge_items.designer_difficulty_str == band]
        idx = rng.choice(sub.index.to_numpy(), size=min(per, len(sub)), replace=False)
        parts.append(sub.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    out["item_source"] = "human_created"
    out["target_band"] = out["designer_difficulty_str"]
    return out


def sample_generation_passages(integrated: pd.DataFrame, exclude_ids, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    work = integrated[~integrated.question_id.isin(exclude_ids)].copy()
    # one passage family: use base_id if present else question_id stem
    work["passage_key"] = work["question_id"].astype(str).str.replace(r"_q\d+$", "", regex=True)
    # unique passages
    passages = work.drop_duplicates("passage_key")
    parts = []
    per = n // 2
    for band in ["MIDDLE", "HIGH"]:
        sub = passages[passages.designer_difficulty_str == band]
        take = min(per, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        parts.append(sub.loc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.head(n)


# ---------------------------------------------------------------------------
# LLM item generation
# ---------------------------------------------------------------------------

GEN_SYSTEM = (
    "You write multiple-choice reading comprehension questions. "
    "Return ONLY valid JSON with keys: question, option_a, option_b, option_c, option_d, answer_letter. "
    "answer_letter must be A, B, C, or D. Options must be short and mutually exclusive."
)


def gen_prompt(article: str, target_band: str) -> str:
    level = "middle-school" if target_band == "MIDDLE" else "high-school"
    hard = "moderately challenging" if target_band == "MIDDLE" else "challenging"
    art = article if len(article) <= 1800 else article[:1800] + "..."
    return (
        f"{GEN_SYSTEM}\n\n"
        f"Target examinee level: {level} ({hard}).\n"
        f"Write ONE new MCQ that can be answered from the passage alone.\n\n"
        f"Passage:\n{art}\n\n"
        "JSON:"
    )


def parse_gen_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    # fenced
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    need = ["question", "option_a", "option_b", "option_c", "option_d", "answer_letter"]
    if not all(k in obj and str(obj[k]).strip() for k in need):
        return None
    letter = str(obj["answer_letter"]).strip().upper()[:1]
    if letter not in LETTERS:
        return None
    return {
        "question": str(obj["question"]).strip(),
        "option_a": str(obj["option_a"]).strip(),
        "option_b": str(obj["option_b"]).strip(),
        "option_c": str(obj["option_c"]).strip(),
        "option_d": str(obj["option_d"]).strip(),
        "answer_letter": letter,
    }


def deepseek_chat(client, model: str, prompt: str, temperature: float, max_tokens: int = 400) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (r.choices[0].message.content or "").strip()


def generate_llm_items(passages: pd.DataFrame, cache_path: Path, workers: int) -> pd.DataFrame:
    from openai import OpenAI

    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    done = {}
    rows = []
    if cache_path.is_file():
        prev = pd.read_csv(cache_path)
        done = set(prev["source_passage_key"].astype(str))
        rows = prev.to_dict("records")

    jobs = []
    for _, r in passages.iterrows():
        key = str(r["passage_key"])
        if key in done:
            continue
        jobs.append(r)

    print(f"[GEN] pending={len(jobs)} cached={len(done)}")

    def _one(r):
        prompt = gen_prompt(str(r["article"]), str(r["designer_difficulty_str"]))
        raw = deepseek_chat(client, model, prompt, temperature=0.7, max_tokens=500)
        parsed = parse_gen_json(raw)
        if parsed is None:
            # one retry colder
            raw = deepseek_chat(client, model, prompt + "\nReturn JSON only.", temperature=0.2, max_tokens=500)
            parsed = parse_gen_json(raw)
        if parsed is None:
            return None
        qid = f"LLMgen_{r['passage_key']}"
        return {
            "question_id": qid,
            "source_passage_key": str(r["passage_key"]),
            "item_source": "llm_generated",
            "target_band": r["designer_difficulty_str"],
            "designer_difficulty_str": r["designer_difficulty_str"],
            "article": str(r["article"])[:4000],
            **parsed,
            "raw_gen": raw[:200],
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, r) for r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec:
                rows.append(rec)
            if i % 10 == 0:
                pd.DataFrame(rows).to_csv(cache_path, index=False)
                print(f"[GEN] {i}/{len(jobs)} ok={len(rows)}")
    df = pd.DataFrame(rows).drop_duplicates("question_id")
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Answering LLM items with same panel
# ---------------------------------------------------------------------------

def answer_with_encoder_sims(llm_items: pd.DataFrame, roster: pd.DataFrame, artifacts: Path, seed: int):
    enc_roster = roster[roster.source == "encoder_sim"].copy()
    if enc_roster.empty:
        return pd.DataFrame()
    # map backend dir
    backend_dirs = {
        "bert-base-uncased": artifacts.parent / "encoder_competitive" / "bert-base-uncased" / "hf_model",
        # roster may truncate name
    }
    # fix truncated names in roster
    long_rows = []
    for backend_key, model_dir in [
        ("bert-base-uncased", artifacts.parent / "encoder_competitive" / "bert-base-uncased" / "hf_model"),
        ("allenai_longformer-base-4096", artifacts.parent / "encoder_competitive" / "allenai_longformer-base-4096" / "hf_model"),
    ]:
        sub_roster = enc_roster[enc_roster.backend.astype(str).str.contains(backend_key.split("-")[0] if "bert" in backend_key else "longformer", case=False, na=False)]
        # more reliable match
        if "bert" in backend_key:
            sub_roster = enc_roster[enc_roster.backend.astype(str).str.contains("bert", case=False, na=False)]
        else:
            sub_roster = enc_roster[enc_roster.backend.astype(str).str.contains("longformer", case=False, na=False)]
        if sub_roster.empty or not model_dir.is_dir():
            print(f"[WARN] skip encoder backend {backend_key}")
            continue
        cache = artifacts / f"e11_logits_{backend_key}.csv"
        # build item frame expected by dump_encoder_logits
        items = llm_items.rename(columns={})  # already has fields
        if "answer_letter" not in items.columns:
            raise ValueError("llm items need answer_letter")
        if cache.is_file():
            logits = pd.read_csv(cache)
        else:
            max_len = 512 if "longformer" in backend_key else 320
            art_w = 200 if "longformer" in backend_key else 120
            logits = dump_encoder_logits(items, model_dir, article_words=art_w, max_len=max_len)
            logits.to_csv(cache, index=False)
        # simulate each person with stored temp/noise
        L = logits[["logit_A", "logit_B", "logit_C", "logit_D"]].to_numpy(float)
        gold = logits["gold"].tolist() if "gold" in logits.columns else logits.merge(
            items[["question_id", "answer_letter"]], on="question_id"
        )["answer_letter"].tolist()
        qids = logits["question_id"].tolist()
        for _, person in sub_roster.iterrows():
            rng = np.random.default_rng(int(person["seed"]) + 10007)
            temp = float(person["temperature"]) if pd.notna(person["temperature"]) else 1.0
            noise = float(person["noise_scale"]) if pd.notna(person["noise_scale"]) else 0.3
            noisy = L + rng.normal(0.0, noise, size=L.shape)
            z = noisy / max(temp, 1e-3)
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(axis=1, keepdims=True)
            cdf = np.cumsum(p, axis=1)
            u = rng.random(len(qids))
            choices = (u[:, None] > cdf[:, :-1]).sum(axis=1)
            for i, qid in enumerate(qids):
                letter = LETTERS[int(choices[i])]
                g = str(gold[i]).strip().upper()[:1]
                long_rows.append(
                    {
                        "person_id": person["person_id"],
                        "question_id": qid,
                        "chosen_letter": letter,
                        "is_correct": int(letter == g),
                        "group": "machine",
                        "source": "encoder_sim",
                        "item_source": "llm_generated",
                    }
                )
    return pd.DataFrame(long_rows)


def answer_with_llm_sims(llm_items: pd.DataFrame, roster: pd.DataFrame, cache_path: Path, workers: int):
    from openai import OpenAI

    llm_roster = roster[roster.source == "llm_sim"].copy()
    if llm_roster.empty:
        return pd.DataFrame()
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    if cache_path.is_file():
        prev = pd.read_csv(cache_path)
        done = set(zip(prev["person_id"], prev["question_id"]))
        rows = prev.to_dict("records")
    else:
        done, rows = set(), []

    item_recs = llm_items.to_dict("records")
    jobs = []
    for _, person in llm_roster.iterrows():
        sys_text = person.get("system_text")
        if pd.isna(sys_text) or not str(sys_text).strip():
            sys_text = "You are a student. Answer with ONE letter only."
        for row in item_recs:
            if (person["person_id"], row["question_id"]) in done:
                continue
            jobs.append((person, row, str(sys_text)))

    print(f"[ANS-LLM] pending={len(jobs)} cached={len(done)}")

    def _one(person, row, sys_text):
        prompt = build_mc_prompt(row, sys_text)
        raw = deepseek_chat(
            client,
            model,
            prompt,
            temperature=float(person["temperature"]) if pd.notna(person["temperature"]) else 0.8,
            max_tokens=4,
        )
        letter = parse_option_letter(raw) or ""
        gold = str(row["answer_letter"]).strip().upper()[:1]
        return {
            "person_id": person["person_id"],
            "question_id": row["question_id"],
            "chosen_letter": letter,
            "raw": raw[:40],
            "is_correct": int(letter == gold) if letter in LETTERS else 0,
            "group": "machine",
            "source": "llm_sim",
            "item_source": "llm_generated",
            "persona": person.get("persona"),
            "temperature": person.get("temperature"),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, p, r, s) for p, r, s in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"[WARN] ans fail: {exc}")
            if i % 50 == 0:
                pd.DataFrame(rows).to_csv(cache_path, index=False)
                print(f"[ANS-LLM] {i}/{len(jobs)}")
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# IRT Paper-1 analysis
# ---------------------------------------------------------------------------

def tost_equivalence(x, y, delta=0.5):
    """Two one-sided tests for mean difference within [-delta, delta] on logit scale."""
    # TOST via Welch
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    if se <= 0:
        return {"diff": float(diff), "equivalent_at_delta": False, "delta": delta}
    t1 = (diff - (-delta)) / se  # H0: diff <= -delta
    t2 = (delta - diff) / se  # H0: diff >= delta
    df = len(x) + len(y) - 2
    p1 = stats.t.sf(t1, df)
    p2 = stats.t.sf(t2, df)
    p = max(p1, p2)
    return {
        "diff": float(diff),
        "se": float(se),
        "delta": float(delta),
        "p_tost": float(p),
        "equivalent_at_delta": bool(p < 0.05),
        "ci95_low": float(diff - 1.96 * se),
        "ci95_high": float(diff + 1.96 * se),
    }


def run_paper1_irt(human_long, llm_long, human_meta, llm_meta, tables, artifacts):
    # crossed long
    h = human_long.copy()
    h["item_source"] = "human_created"
    m = llm_long.copy()
    m["item_source"] = "llm_generated"
    all_long = pd.concat(
        [
            h[["person_id", "question_id", "is_correct", "item_source"]],
            m[["person_id", "question_id", "is_correct", "item_source"]],
        ],
        ignore_index=True,
    )
    # keep persons who answered both sources
    both = (
        all_long.groupby("person_id")["item_source"]
        .nunique()
        .reset_index()
    )
    keep_persons = both.loc[both["item_source"] >= 2, "person_id"]
    all_long = all_long[all_long.person_id.isin(keep_persons)]
    print(f"[IRT] crossed persons={all_long.person_id.nunique()} "
          f"items={all_long.question_id.nunique()} obs={len(all_long)}")

    mat, items, persons = build_dense_matrix(
        all_long, "person_id", "question_id", "is_correct", min_item=10, min_person=10
    )
    fits = fit_models(mat, models=("1PL", "2PL"))

    meta = pd.concat(
        [
            human_meta[["question_id", "item_source", "target_band"]],
            llm_meta[["question_id", "item_source", "target_band"]],
        ],
        ignore_index=True,
    ).drop_duplicates("question_id")
    meta_m = meta.set_index("question_id")

    rows = []
    for model, fit in fits.items():
        if not model.endswith("PL"):
            continue
        for i, qid in enumerate(items):
            rows.append(
                {
                    "model": model,
                    "question_id": qid,
                    "a": float(fit["a"][i]),
                    "b": float(fit["b"][i]),
                    "item_source": meta_m.loc[qid, "item_source"] if qid in meta_m.index else None,
                    "target_band": meta_m.loc[qid, "target_band"] if qid in meta_m.index else None,
                }
            )
    params = pd.DataFrame(rows)
    save_table(params, artifacts / "e11_joint_item_params.csv")

    # Primary 1PL comparison
    b1 = params[params.model == "1PL"].dropna(subset=["item_source", "b"])
    human_b = b1.loc[b1.item_source == "human_created", "b"]
    llm_b = b1.loc[b1.item_source == "llm_generated", "b"]
    ks = stats.ks_2samp(human_b, llm_b)
    welch = stats.ttest_ind(llm_b, human_b, equal_var=False)
    tost = tost_equivalence(llm_b.to_numpy(), human_b.to_numpy(), delta=0.5)
    # within matched target band
    band_rows = []
    for band in ["MIDDLE", "HIGH"]:
        hb = b1.loc[(b1.item_source == "human_created") & (b1.target_band == band), "b"]
        lb = b1.loc[(b1.item_source == "llm_generated") & (b1.target_band == band), "b"]
        if len(hb) < 3 or len(lb) < 3:
            continue
        ksb = stats.ks_2samp(hb, lb)
        wb = stats.ttest_ind(lb, hb, equal_var=False)
        tb = tost_equivalence(lb.to_numpy(), hb.to_numpy(), delta=0.5)
        band_rows.append(
            {
                "target_band": band,
                "n_human": int(len(hb)),
                "n_llm": int(len(lb)),
                "mean_b_human": float(hb.mean()),
                "mean_b_llm": float(lb.mean()),
                "diff_llm_minus_human": float(lb.mean() - hb.mean()),
                "welch_p": float(wb.pvalue),
                "ks_stat": float(ksb.statistic),
                "ks_p": float(ksb.pvalue),
                "tost_equivalent_delta0.5": bool(tb["equivalent_at_delta"]),
                "tost_p": float(tb["p_tost"]),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "model": "1PL",
                "n_persons": len(persons),
                "n_human_items": int((b1.item_source == "human_created").sum()),
                "n_llm_items": int((b1.item_source == "llm_generated").sum()),
                "mean_b_human": float(human_b.mean()),
                "mean_b_llm": float(llm_b.mean()),
                "sd_b_human": float(human_b.std(ddof=1)),
                "sd_b_llm": float(llm_b.std(ddof=1)),
                "diff_llm_minus_human": float(llm_b.mean() - human_b.mean()),
                "welch_t": float(welch.statistic),
                "welch_p": float(welch.pvalue),
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
                "spearman_within_band_proxy": float(
                    stats.spearmanr(
                        b1.sort_values("question_id")["b"],
                        b1.sort_values("question_id")["b"],
                    )[0]
                ),
                **{f"tost_{k}": v for k, v in tost.items()},
                "hypothesis_support": (
                    "aligned"
                    if tost["equivalent_at_delta"] or (welch.pvalue >= 0.05 and ks.pvalue >= 0.05)
                    else "not_aligned"
                ),
            }
        ]
    )
    # 2PL robustness
    b2 = params[params.model == "2PL"].dropna(subset=["item_source", "b"])
    h2 = b2.loc[b2.item_source == "human_created", "b"]
    l2 = b2.loc[b2.item_source == "llm_generated", "b"]
    # correlate ranks of overlapping... different items, compare mean diff sign
    robust = pd.DataFrame(
        [
            {
                "model": "2PL",
                "mean_b_human": float(h2.mean()),
                "mean_b_llm": float(l2.mean()),
                "diff_llm_minus_human": float(l2.mean() - h2.mean()),
                "welch_p": float(stats.ttest_ind(l2, h2, equal_var=False).pvalue),
                "ks_p": float(stats.ks_2samp(h2, l2).pvalue),
                "same_sign_diff_as_1pl": bool(
                    np.sign(l2.mean() - h2.mean()) == np.sign(llm_b.mean() - human_b.mean())
                ),
            }
        ]
    )

    save_table(summary, tables / "table_e11_paper1_1pl_alignment.csv")
    save_table(pd.DataFrame(band_rows), tables / "table_e11_paper1_by_band.csv")
    save_table(robust, tables / "table_e11_paper1_2pl_robustness.csv")

    # empirical p-correct by source (classical check)
    classical = (
        all_long.groupby("item_source")["is_correct"]
        .agg(n_obs="count", mean_correct="mean")
        .reset_index()
    )
    save_table(classical, tables / "table_e11_classical_accuracy_by_source.csv")

    return summary, params, band_rows


def write_report(out_dir: Path, summary: pd.DataFrame, band_df: pd.DataFrame, n_gen: int, n_hum: int):
    s = summary.iloc[0].to_dict()
    path = out_dir / "artifacts" / "irt" / "E11_PAPER1_REPORT.md"
    lines = [
        "# E11 / Paper-1 — Difficulty(LLM-generated) vs Difficulty(Human-created)",
        "",
        "## Design",
        "",
        f"- Human-created anchors: **{n_hum}** Bridge-RACE items (stratified MIDDLE/HIGH)",
        f"- LLM-generated items: **{n_gen}** MCQs from held-out RACE passages (DeepSeek)",
        f"- Same respondent panel answering both sources: **{int(s['n_persons'])}** machine/LLM-simulated students",
        "- Primary scale: **concurrent 1PL**; robustness: 2PL",
        "",
        "## Hypothesis",
        "",
        "When target difficulty bands are matched, the IRT difficulty distribution of "
        "LLM-generated items should be similar to that of human-created items.",
        "",
        "## Results (1PL)",
        "",
        f"- mean b human = {s['mean_b_human']:.3f}, mean b LLM = {s['mean_b_llm']:.3f}",
        f"- Δ (LLM−human) = {s['diff_llm_minus_human']:.3f}",
        f"- Welch p = {s['welch_p']:.4g}, KS p = {s['ks_p']:.4g}",
        f"- TOST equivalence at |Δ|<0.5 logits: **{s['tost_equivalent_at_delta']}** (p={s['tost_p_tost']:.4g})",
        f"- Hypothesis support label: **{s['hypothesis_support']}**",
        "",
        "## By target band",
        "",
        band_df.to_markdown(index=False) if hasattr(band_df, "to_markdown") else band_df.to_string(index=False),
        "",
        "## Files",
        "",
        "- `tables/table_e11_*.csv`",
        "- `artifacts/irt/e11_joint_item_params.csv`",
        "- `artifacts/irt/e11_llm_items.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {path}")


def main():
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    out_dir = Path(args.out_dir)
    tables = ensure_dir(out_dir / "tables")
    artifacts = ensure_dir(out_dir / "artifacts" / "irt")

    bridge = pd.read_csv(REVISION_ROOT / "bridge" / "bridge_race_items.csv")
    integrated = pd.read_csv(REVISION_ROOT / "artifacts" / "race_val_integrated.csv")
    roster = pd.read_csv(artifacts / "machine_panel_roster.csv")
    panel_long = pd.read_csv(artifacts / "machine_panel_long.csv")

    # 1) human anchors
    anchors = sample_human_anchors(bridge, args.n_human_anchors, args.seed)
    save_table(anchors, artifacts / "e11_human_anchors.csv")

    # reuse existing panel responses on anchors
    human_long = panel_long[panel_long.question_id.isin(anchors.question_id)].copy()
    print(f"[INFO] human-anchor responses reused: {len(human_long)} "
          f"persons={human_long.person_id.nunique()}")

    # 2) generate LLM items
    gen_cache = artifacts / "e11_llm_items.csv"
    if args.skip_generate and gen_cache.is_file():
        llm_items = pd.read_csv(gen_cache)
    else:
        passages = sample_generation_passages(
            integrated, set(anchors.question_id), args.n_llm_items, args.seed
        )
        llm_items = generate_llm_items(passages, gen_cache, args.llm_workers)
    print(f"[INFO] LLM items: {len(llm_items)} "
          f"{llm_items.target_band.value_counts().to_dict() if 'target_band' in llm_items.columns else ''}")
    if len(llm_items) < max(20, args.n_llm_items // 2):
        raise SystemExit(f"Too few LLM items generated: {len(llm_items)}")

    # 3) answer LLM items with same panel
    ans_cache = artifacts / "e11_llm_item_responses_llm_sims.csv"
    enc_cache = artifacts / "e11_llm_item_responses_encoder_sims.csv"
    if args.skip_answer and ans_cache.is_file() and enc_cache.is_file():
        llm_ans_llm = pd.read_csv(ans_cache)
        llm_ans_enc = pd.read_csv(enc_cache)
    else:
        llm_ans_enc = answer_with_encoder_sims(llm_items, roster, artifacts, args.seed)
        llm_ans_enc.to_csv(enc_cache, index=False)
        llm_ans_llm = answer_with_llm_sims(llm_items, roster, ans_cache, args.llm_workers)
    llm_long = pd.concat([llm_ans_enc, llm_ans_llm], ignore_index=True, sort=False)
    llm_long = llm_long.drop_duplicates(["person_id", "question_id"], keep="last")
    save_table(llm_long, artifacts / "e11_llm_item_responses_all.csv")
    print(f"[INFO] LLM-item responses: {len(llm_long)} "
          f"persons={llm_long.person_id.nunique()} items={llm_long.question_id.nunique()}")

    # 4) IRT
    human_meta = anchors[["question_id", "item_source", "target_band"]]
    llm_meta = llm_items[["question_id", "item_source", "target_band"]]
    summary, params, band_rows = run_paper1_irt(
        human_long, llm_long, human_meta, llm_meta, tables, artifacts
    )
    band_df = pd.DataFrame(band_rows)
    write_report(out_dir, summary, band_df, len(llm_items), len(anchors))

    # status
    status = {
        "E11": "done",
        "paper1": True,
        "hypothesis_support": summary.iloc[0]["hypothesis_support"],
        "n_human_items": int(summary.iloc[0]["n_human_items"]),
        "n_llm_items": int(summary.iloc[0]["n_llm_items"]),
        "n_persons": int(summary.iloc[0]["n_persons"]),
        "report": "revision/artifacts/irt/E11_PAPER1_REPORT.md",
    }
    (artifacts / "e11_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("[OK] E11 Paper-1 complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
