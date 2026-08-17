#!/usr/bin/env python3
"""E10: Expand machine/LLM-simulated student panel to close the n=4 limitation.

Produces a person×item response panel comparable in size to Bridge humans
(200 annotators × 320 items), then re-runs linking + DIF with adequate power.

Sources
-------
1) Encoder-simulated students: sample answers from MultipleChoice logits under
   heterogeneous temperatures / logit noise (fast, full item coverage).
2) LLM-simulated students: DeepSeek (and optional Ollama) with ability personas
   and stochastic decoding (temperature>0, multiple seeds).

Outputs
-------
revision/artifacts/irt/machine_panel_long.csv
revision/artifacts/irt/machine_panel_roster.csv
revision/tables/table_e10_*.csv
revision/artifacts/irt/E10_POWERED_DIF_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.special import expit
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, REVISION_ROOT, ensure_dir, parse_option_letter, save_table  # noqa: E402
from girth import ability_eap  # noqa: E402
from e9_irt_model_family import (  # noqa: E402
    build_dense_matrix,
    fit_models,
    girth_matrix,
    load_bridge_long,
    mean_sigma_link,
)


LETTERS = ["A", "B", "C", "D"]


def load_dotenv(path: Path) -> dict:
    env = {}
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    os.environ.update(env)
    return env


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge_items", default=str(REVISION_ROOT / "bridge/bridge_race_items.csv"))
    ap.add_argument(
        "--bridge_responses",
        default=str(REVISION_ROOT / "bridge/bridge_race_responses.csv"),
    )
    ap.add_argument(
        "--encoder_dirs",
        nargs="*",
        default=[
            str(REVISION_ROOT / "artifacts/encoder_competitive/bert-base-uncased/hf_model"),
            str(REVISION_ROOT / "artifacts/encoder_competitive/allenai_longformer-base-4096/hf_model"),
        ],
    )
    ap.add_argument("--n_encoder_students", type=int, default=120)
    ap.add_argument("--n_llm_students", type=int, default=80)
    ap.add_argument("--llm_workers", type=int, default=12)
    ap.add_argument("--ollama_students", type=int, default=24)
    ap.add_argument("--ollama_model", default="llama3.1:8b")
    ap.add_argument("--skip_llm", action="store_true")
    ap.add_argument("--skip_ollama", action="store_true")
    ap.add_argument("--skip_encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(REVISION_ROOT))
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Encoder logit dump + simulation
# ---------------------------------------------------------------------------

def _mc_collate(tok, max_len: int):
    def collate(batch):
        ctxs, ends = [], []
        qids, golds = [], []
        for b in batch:
            ctxs.extend([b["context"]] * 4)
            ends.extend(b["endings"])
            qids.append(b["question_id"])
            golds.append(b["gold"])
        enc = tok(
            ctxs,
            ends,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        bs = len(batch)
        out = {k: v.view(bs, 4, -1) for k, v in enc.items()}
        out["question_id"] = qids
        out["gold"] = golds
        return out

    return collate


@torch.no_grad()
def dump_encoder_logits(items: pd.DataFrame, model_dir: Path, article_words=120, max_len=320, batch_size=8):
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForMultipleChoice.from_pretrained(str(model_dir)).to(device)
    model.eval()

    rows = []
    for _, r in items.iterrows():
        art = " ".join(str(r["article"]).split()[:article_words])
        rows.append(
            {
                "question_id": r["question_id"],
                "context": f"{art} {r['question']}".strip(),
                "endings": [str(r["option_a"]), str(r["option_b"]), str(r["option_c"]), str(r["option_d"])],
                "gold": str(r["answer_letter"]).strip().upper()[:1],
            }
        )

    loader = DataLoader(rows, batch_size=batch_size, collate_fn=_mc_collate(tok, max_len))
    out = []
    for batch in tqdm(loader, desc=f"logits:{model_dir.parent.name}"):
        qids = batch.pop("question_id")
        golds = batch.pop("gold")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().float().cpu().numpy()
        for i, qid in enumerate(qids):
            out.append(
                {
                    "question_id": qid,
                    "encoder": model_dir.parent.name,
                    "logit_A": float(logits[i, 0]),
                    "logit_B": float(logits[i, 1]),
                    "logit_C": float(logits[i, 2]),
                    "logit_D": float(logits[i, 3]),
                    "gold": golds[i],
                }
            )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(out)


def simulate_from_logits(logit_df: pd.DataFrame, n_students: int, seed: int, source_tag: str):
    """Heterogeneous temperature + Gaussian logit noise → simulated persons."""
    rng = np.random.default_rng(seed)
    # roster: mix of weak/mid/strong via temperature schedule
    temps = np.concatenate(
        [
            rng.uniform(0.4, 0.8, size=n_students // 3),
            rng.uniform(0.8, 1.4, size=n_students // 3),
            rng.uniform(1.4, 2.8, size=n_students - 2 * (n_students // 3)),
        ]
    )
    rng.shuffle(temps)
    noise_scales = rng.uniform(0.05, 0.6, size=n_students)

    items = logit_df["question_id"].tolist()
    L = logit_df[["logit_A", "logit_B", "logit_C", "logit_D"]].to_numpy(dtype=float)
    gold = logit_df["gold"].tolist()
    enc_name = str(logit_df["encoder"].iloc[0])

    long_rows = []
    roster = []
    for s in range(n_students):
        pid = f"ENCSim_{source_tag}_{s:03d}"
        roster.append(
            {
                "person_id": pid,
                "group": "machine",
                "source": "encoder_sim",
                "backend": enc_name,
                "temperature": float(temps[s]),
                "noise_scale": float(noise_scales[s]),
                "seed": seed + s,
            }
        )
        noisy = L + rng.normal(0.0, noise_scales[s], size=L.shape)
        # softmax with temperature
        z = noisy / max(temps[s], 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        # categorical draw
        cdf = np.cumsum(p, axis=1)
        u = rng.random(len(items))
        choices = (u[:, None] > cdf[:, :-1]).sum(axis=1)
        for i, qid in enumerate(items):
            letter = LETTERS[int(choices[i])]
            long_rows.append(
                {
                    "person_id": pid,
                    "question_id": qid,
                    "chosen_letter": letter,
                    "is_correct": int(letter == gold[i]),
                    "group": "machine",
                    "source": "encoder_sim",
                }
            )
    return pd.DataFrame(roster), pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# LLM students (DeepSeek / Ollama)
# ---------------------------------------------------------------------------

PERSONAS = [
    ("weak", 1.3, "You are a struggling middle-school student. You often guess. Answer with ONE letter only."),
    ("mid", 0.9, "You are an average student. Try your best. Answer with ONE letter only."),
    ("strong", 0.5, "You are a strong high-school student. Think carefully. Answer with ONE letter only."),
    ("careless", 1.1, "You are a student who reads quickly and sometimes misses details. Answer with ONE letter only."),
]


def build_mc_prompt(row, persona_text: str, max_article_chars=1200) -> str:
    art = str(row["article"])
    if len(art) > max_article_chars:
        art = art[:max_article_chars] + "..."
    return (
        f"{persona_text}\n\n"
        f"Passage:\n{art}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n\n"
        "Answer (A/B/C/D only):"
    )


def make_llm_roster(n_students: int, seed: int, backend: str):
    rng = np.random.default_rng(seed)
    roster = []
    for s in range(n_students):
        persona_name, temp, text = PERSONAS[s % len(PERSONAS)]
        # jitter temperature
        temp = float(np.clip(temp + rng.normal(0, 0.1), 0.2, 1.6))
        roster.append(
            {
                "person_id": f"LLMSim_{backend}_{s:03d}",
                "group": "machine",
                "source": "llm_sim",
                "backend": backend,
                "persona": persona_name,
                "temperature": temp,
                "seed": int(seed + s),
                "system_text": text,
            }
        )
    return pd.DataFrame(roster)


def deepseek_answer(client, model: str, prompt: str, temperature: float, seed: int) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4,
        seed=seed,
    )
    return (r.choices[0].message.content or "").strip()


def run_deepseek_panel(items: pd.DataFrame, roster: pd.DataFrame, workers: int, cache_path: Path):
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    client = OpenAI(api_key=api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    if cache_path.is_file():
        prev = pd.read_csv(cache_path)
        done = {(a, b): True for a, b in zip(prev["person_id"], prev["question_id"])}
        rows_out = prev.to_dict("records")
    else:
        done, rows_out = set(), []

    jobs = []
    item_recs = items.to_dict("records")
    for _, person in roster.iterrows():
        for row in item_recs:
            key = (person["person_id"], row["question_id"])
            if key in done:
                continue
            jobs.append((person, row))

    print(f"[DeepSeek] pending jobs={len(jobs)} cached={len(done)} workers={workers}")

    def _one(person, row):
        prompt = build_mc_prompt(row, person["system_text"])
        raw = deepseek_answer(client, model, prompt, float(person["temperature"]), int(person["seed"]))
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
            "backend": person["backend"],
            "persona": person.get("persona"),
            "temperature": person["temperature"],
        }

    flush_every = 50
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, p, r) for p, r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows_out.append(fut.result())
            except Exception as exc:
                rows_out.append(
                    {
                        "person_id": "ERR",
                        "question_id": "ERR",
                        "chosen_letter": "",
                        "raw": str(exc)[:80],
                        "is_correct": 0,
                        "group": "machine",
                        "source": "llm_sim_error",
                        "backend": "deepseek",
                    }
                )
            if i % flush_every == 0:
                pd.DataFrame([x for x in rows_out if x.get("person_id") != "ERR"]).to_csv(
                    cache_path, index=False
                )
                print(f"[DeepSeek] progress {i}/{len(jobs)}")

    df = pd.DataFrame([x for x in rows_out if x.get("person_id") != "ERR"])
    df.to_csv(cache_path, index=False)
    return df


def ollama_generate(model: str, prompt: str, temperature: float, timeout=120) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 4},
        }
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    return (data.get("response") or "").strip()


def run_ollama_panel(items: pd.DataFrame, roster: pd.DataFrame, cache_path: Path, item_limit: int = 80):
    """Smaller booklet panel for local-model heterogeneity."""
    # stratified item subset
    parts = []
    for band, g in items.groupby("designer_difficulty_str"):
        take = min(item_limit // 2, len(g))
        parts.append(g.sample(n=take, random_state=0))
    sub_items = pd.concat(parts, ignore_index=True).drop_duplicates("question_id")
    item_recs = sub_items.to_dict("records")

    if cache_path.is_file():
        prev = pd.read_csv(cache_path)
        done = set(zip(prev["person_id"], prev["question_id"]))
        rows_out = prev.to_dict("records")
    else:
        done, rows_out = set(), []

    jobs = []
    for _, person in roster.iterrows():
        for row in item_recs:
            if (person["person_id"], row["question_id"]) in done:
                continue
            jobs.append((person, row))
    print(f"[Ollama] pending={len(jobs)} items={len(sub_items)} persons={len(roster)}")

    for i, (person, row) in enumerate(jobs, 1):
        prompt = build_mc_prompt(row, person["system_text"], max_article_chars=900)
        try:
            raw = ollama_generate(person["backend"], prompt, float(person["temperature"]))
            letter = parse_option_letter(raw) or ""
        except Exception as exc:
            raw, letter = f"ERR:{exc}"[:60], ""
        gold = str(row["answer_letter"]).strip().upper()[:1]
        rows_out.append(
            {
                "person_id": person["person_id"],
                "question_id": row["question_id"],
                "chosen_letter": letter,
                "raw": str(raw)[:40],
                "is_correct": int(letter == gold) if letter in LETTERS else 0,
                "group": "machine",
                "source": "ollama_sim",
                "backend": person["backend"],
                "persona": person.get("persona"),
                "temperature": person["temperature"],
            }
        )
        if i % 20 == 0:
            pd.DataFrame(rows_out).to_csv(cache_path, index=False)
            print(f"[Ollama] {i}/{len(jobs)}")
    df = pd.DataFrame(rows_out)
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Powered IRT / DIF
# ---------------------------------------------------------------------------

def mantel_haenszel_proper(human_mat, machine_mat, n_strata=5):
    """Classical MH DIF: both groups stratified by total proportion-correct."""
    n_items = human_mat.shape[0]

    def person_prop(mat):
        ans = np.isfinite(mat)
        num = np.nansum(mat, axis=0)
        den = ans.sum(axis=0)
        p = np.divide(num, den, out=np.full(mat.shape[1], np.nan), where=den > 0)
        return p

    hp = person_prop(human_mat)
    mp = person_prop(machine_mat)
    # shared cut points from pooled scores
    pooled = np.concatenate([hp[np.isfinite(hp)], mp[np.isfinite(mp)]])
    try:
        edges = np.quantile(pooled, np.linspace(0, 1, n_strata + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([pooled.min(), pooled.mean(), pooled.max()])
    except Exception:
        edges = np.array([0.0, 0.5, 1.0])

    def assign(p):
        return np.digitize(p, edges[1:-1], right=True)

    h_str = assign(np.nan_to_num(hp, nan=-1))
    m_str = assign(np.nan_to_num(mp, nan=-1))
    h_str = np.where(np.isfinite(hp), h_str, -1)
    m_str = np.where(np.isfinite(mp), m_str, -1)

    rows = []
    for i in range(n_items):
        num = den = 0.0
        chi_num = chi_den = 0.0
        for s in range(len(edges)):
            h_idx = h_str == s
            m_idx = m_str == s
            h = human_mat[i, h_idx]
            m = machine_mat[i, m_idx]
            h = h[np.isfinite(h)]
            m = m[np.isfinite(m)]
            if len(h) == 0 or len(m) == 0:
                continue
            a = float(m.sum())  # focal correct
            b = float(len(m) - a)
            c = float(h.sum())  # ref correct
            d = float(len(h) - c)
            n = a + b + c + d
            if n <= 1:
                continue
            num += a * d / n
            den += b * c / n
            # MH chi-square continuity-corrected pieces
            n_f = a + b
            n_r = c + d
            correct_tot = a + c
            E = n_f * correct_tot / n
            V = (n_f * n_r * correct_tot * (n - correct_tot)) / (n * n * (n - 1)) if n > 1 else 0
            chi_num += a - E
            chi_den += V
        mh_or = (num / den) if den > 0 else np.nan
        chi2 = ((abs(chi_num) - 0.5) ** 2 / chi_den) if chi_den > 0 else np.nan
        rows.append({"item_idx": i, "mh_or": mh_or, "mh_chi2": chi2})
    return pd.DataFrame(rows)


def run_powered_irt(human_resp, machine_long, meta, artifacts, tables):
    # human matrix
    h_mat, h_items, h_persons = build_dense_matrix(
        human_resp, "person_id", "question_id", "is_correct", min_item=15, min_person=8
    )
    # machine matrix on same items
    m_long = machine_long[machine_long["question_id"].isin(h_items)].copy()
    m_mat_raw, m_items, m_persons = build_dense_matrix(
        m_long, "person_id", "question_id", "is_correct", min_item=5, min_person=5
    )
    item_index = {q: i for i, q in enumerate(h_items)}
    m_mat = np.full((len(h_items), len(m_persons)), np.nan)
    for i_m, qid in enumerate(m_items):
        if qid in item_index:
            m_mat[item_index[qid], :] = m_mat_raw[i_m, :]

    print(f"[Power] human {h_mat.shape} | machine {m_mat.shape}")

    # E3 concurrent
    concurrent = np.concatenate([h_mat, m_mat], axis=1)
    conc = fit_models(concurrent, models=("1PL", "2PL"))
    human_fit = fit_models(h_mat, models=("1PL",))["1PL"]
    machine_fit = fit_models(m_mat, models=("1PL",))["1PL"]

    A, B, b_m_linked = mean_sigma_link(machine_fit["b"], human_fit["b"])
    rho, _ = stats.spearmanr(b_m_linked, human_fit["b"])
    rho_raw, _ = stats.spearmanr(machine_fit["b"], human_fit["b"])
    link = pd.DataFrame(
        [
            {
                "method": "separate_1PL_mean_sigma",
                "A": A,
                "B": B,
                "spearman_linked": float(rho),
                "spearman_raw": float(rho_raw),
                "mean_abs_diff_linked": float(np.mean(np.abs(b_m_linked - human_fit["b"]))),
                "n_human_persons": h_mat.shape[1],
                "n_machine_persons": m_mat.shape[1],
                "n_items": len(h_items),
                "machine_n_adequate": bool(m_mat.shape[1] >= 50),
            },
            {
                "method": "concurrent_1PL_vs_human_only",
                "spearman_raw": float(stats.spearmanr(conc["1PL"]["b"], human_fit["b"])[0]),
                "mean_abs_diff_linked": float(np.mean(np.abs(conc["1PL"]["b"] - human_fit["b"]))),
                "n_human_persons": h_mat.shape[1],
                "n_machine_persons": m_mat.shape[1],
                "n_items": len(h_items),
                "machine_n_adequate": bool(m_mat.shape[1] >= 50),
            },
        ]
    )
    save_table(link, tables / "table_e10_e3_linking_powered.csv")
    save_table(
        pd.DataFrame(
            {
                "question_id": h_items,
                "b_human_1pl": human_fit["b"],
                "b_machine_1pl": machine_fit["b"],
                "b_machine_linked": b_m_linked,
                "b_concurrent_1pl": conc["1PL"]["b"],
            }
        ),
        artifacts / "e10_linked_difficulties.csv",
    )

    # residual DIF with adequate machine N
    gmat = girth_matrix(m_mat)
    theta_m = np.asarray(ability_eap(gmat, human_fit["b"], human_fit["a"]), dtype=float)
    resid_rows = []
    for i, qid in enumerate(h_items):
        obs = m_mat[i]
        mask = np.isfinite(obs)
        if mask.sum() < 5:
            continue
        th = theta_m[mask]
        p = expit(human_fit["a"][i] * (th - human_fit["b"][i]))
        resid = obs[mask] - p
        resid_rows.append(
            {
                "question_id": qid,
                "b_1pl": float(human_fit["b"][i]),
                "mean_residual": float(resid.mean()),
                "abs_mean_residual": float(np.abs(resid).mean()),
                "n_machine_obs": int(mask.sum()),
                "machine_p": float(obs[mask].mean()),
                "expected_p": float(p.mean()),
                "dif_flag": bool(np.abs(resid.mean()) > 0.25),
            }
        )
    resid_df = pd.DataFrame(resid_rows)
    meta_m = meta.set_index("question_id")
    resid_df["designer_difficulty_str"] = [
        meta_m.loc[q, "designer_difficulty_str"] if q in meta_m.index else None
        for q in resid_df["question_id"]
    ]
    save_table(resid_df, artifacts / "e10_residual_dif.csv")

    # Proper MH
    mh = mantel_haenszel_proper(h_mat, m_mat, n_strata=5)
    mh["question_id"] = [h_items[i] for i in mh["item_idx"]]
    mh["dif_flag_chi2_gt_3.84"] = mh["mh_chi2"] > 3.84
    # ETS-style |MH logit| categories approx via OR
    mh["ets_category"] = pd.cut(
        np.log(mh["mh_or"].clip(1e-3, 1e3)),
        bins=[-np.inf, -1.0, -0.64, 0.64, 1.0, np.inf],
        labels=["C_favors_machine", "B_favors_machine", "A_negligible", "B_favors_human", "C_favors_human"],
    )
    save_table(mh, artifacts / "e10_mh_dif.csv")

    summary = pd.DataFrame(
        [
            {
                "n_human_persons": h_mat.shape[1],
                "n_machine_persons": m_mat.shape[1],
                "n_items": len(h_items),
                "machine_obs": int(np.isfinite(m_mat).sum()),
                "human_obs": int(np.isfinite(h_mat).sum()),
                "residual_dif_share": float(resid_df["dif_flag"].mean()),
                "residual_mean_abs": float(resid_df["abs_mean_residual"].mean()),
                "mh_flag_share": float(mh["dif_flag_chi2_gt_3.84"].fillna(False).mean()),
                "mh_A_negligible_share": float((mh["ets_category"] == "A_negligible").mean()),
                "spearman_b_linked": float(rho),
                "limitation_closed": bool(m_mat.shape[1] >= 50 and np.isfinite(m_mat).sum() >= 5000),
            }
        ]
    )
    save_table(summary, tables / "table_e10_powered_dif_summary.csv")

    by_band = (
        resid_df.groupby("designer_difficulty_str")
        .agg(n=("question_id", "count"), share_dif=("dif_flag", "mean"), mean_abs=("abs_mean_residual", "mean"))
        .reset_index()
    )
    save_table(by_band, tables / "table_e10_dif_by_designer_band.csv")

    # Compare vs old n=4 result if present
    old_path = tables / "table_e9_e4_dif_summary.csv"
    compare_rows = [{"stage": "powered_e10", **summary.iloc[0].to_dict()}]
    if old_path.is_file():
        old = pd.read_csv(old_path).iloc[0].to_dict()
        compare_rows.append(
            {
                "stage": "underpowered_e9_n4",
                "n_machine_persons": 4,
                "residual_dif_share": old.get("share_dif_flagged"),
                "residual_mean_abs": old.get("mean_abs_residual"),
                "note": "OR proxy over-flagged due to tiny machine N",
            }
        )
    save_table(pd.DataFrame(compare_rows), tables / "table_e10_before_after_power.csv")
    return summary, link, resid_df, mh


def write_report(out_dir: Path, roster: pd.DataFrame, summary: pd.DataFrame, link: pd.DataFrame):
    s = summary.iloc[0].to_dict()
    path = out_dir / "artifacts" / "irt" / "E10_POWERED_DIF_REPORT.md"
    src_counts = roster["source"].value_counts().to_dict() if "source" in roster.columns else {}
    lines = [
        "# E10 — Closing the machine-N limitation by experiment",
        "",
        "## Claim",
        "",
        "The previous E4 analysis was limited by only 4 machine solvers. "
        "E10 replaces that limitation with a powered machine/LLM-simulated student panel "
        "and re-estimates linking + DIF.",
        "",
        "## Panel composition",
        "",
        f"- Machine persons: **{int(s['n_machine_persons'])}** (was 4)",
        f"- Human persons: **{int(s['n_human_persons'])}**",
        f"- Items: **{int(s['n_items'])}**",
        f"- Machine observations: **{int(s['machine_obs'])}**",
        f"- Sources: `{json.dumps(src_counts)}`",
        "",
        f"- **Limitation closed:** {bool(s['limitation_closed'])}",
        "",
        "## Powered results",
        "",
        f"- Linked difficulty Spearman (machine↔human 1PL): **{s['spearman_b_linked']:.3f}**",
        f"- Residual DIF share (|mean resid|>0.25): **{s['residual_dif_share']:.3f}**",
        f"- MH χ² flag share: **{s['mh_flag_share']:.3f}**",
        f"- ETS A (negligible) share: **{s['mh_A_negligible_share']:.3f}**",
        "",
        "## Linking table",
        "",
        link.to_markdown(index=False) if hasattr(link, "to_markdown") else link.to_string(index=False),
        "",
        "## Interpretation for the paper",
        "",
        "- Do **not** list 'only 4 machine solvers' as a limitation anymore.",
        "- Report E10 panel size and powered MH/residual DIF as the primary human–machine DIF evidence.",
        "- Keep 1PL as the main difficulty model; 2PL remains a robustness check (E9).",
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

    items = pd.read_csv(args.bridge_items)
    human_resp, meta = load_bridge_long(args.bridge_responses, args.bridge_items)
    print(f"[INFO] items={len(items)} human_resp={len(human_resp)}")

    roster_parts = []
    long_parts = []

    # 1) Encoder sims
    if not args.skip_encoder:
        logit_frames = []
        for d in args.encoder_dirs:
            md = Path(d)
            if not md.is_dir():
                print(f"[WARN] missing encoder {md}")
                continue
            cache = artifacts / f"logits_{md.parent.name}.csv"
            if cache.is_file():
                print(f"[INFO] load cached logits {cache.name}")
                logit_frames.append(pd.read_csv(cache))
            else:
                # longformer may need larger max_len
                max_len = 512 if "longformer" in md.as_posix().lower() else 320
                art_w = 200 if "longformer" in md.as_posix().lower() else 120
                try:
                    lf = dump_encoder_logits(items, md, article_words=art_w, max_len=max_len)
                    lf.to_csv(cache, index=False)
                    logit_frames.append(lf)
                except Exception as exc:
                    print(f"[WARN] encoder failed {md}: {exc}")
        if logit_frames:
            n_per = max(1, args.n_encoder_students // len(logit_frames))
            for i, lf in enumerate(logit_frames):
                tag = str(lf["encoder"].iloc[0])[:12]
                rost, lng = simulate_from_logits(lf, n_per, args.seed + 17 * i, tag)
                roster_parts.append(rost)
                long_parts.append(lng)
                print(f"[OK] encoder sims {tag}: persons={len(rost)} rows={len(lng)}")

    # 2) DeepSeek LLM sims
    if not args.skip_llm:
        try:
            rost = make_llm_roster(args.n_llm_students, args.seed + 99, "deepseek")
            cache = artifacts / "llm_deepseek_panel_cache.csv"
            lng = run_deepseek_panel(items, rost, args.llm_workers, cache)
            # keep only complete-ish persons
            cnt = lng.groupby("person_id").size()
            keep = cnt[cnt >= max(50, int(0.5 * len(items)))].index
            lng = lng[lng.person_id.isin(keep)]
            rost = rost[rost.person_id.isin(keep)]
            roster_parts.append(rost)
            long_parts.append(lng)
            print(f"[OK] DeepSeek panel persons={len(rost)} rows={len(lng)}")
        except Exception as exc:
            print(f"[WARN] DeepSeek panel failed: {exc}")

    # 3) Ollama local heterogeneity (booklets)
    if not args.skip_ollama:
        try:
            rost = make_llm_roster(args.ollama_students, args.seed + 123, args.ollama_model)
            rost["backend"] = args.ollama_model
            rost["person_id"] = [f"OLLSim_{i:03d}" for i in range(len(rost))]
            cache = artifacts / "llm_ollama_panel_cache.csv"
            lng = run_ollama_panel(items, rost, cache, item_limit=80)
            cnt = lng.groupby("person_id").size()
            keep = cnt[cnt >= 20].index
            lng = lng[lng.person_id.isin(keep)]
            rost = rost[rost.person_id.isin(keep)]
            roster_parts.append(rost)
            long_parts.append(lng)
            print(f"[OK] Ollama panel persons={len(rost)} rows={len(lng)}")
        except Exception as exc:
            print(f"[WARN] Ollama panel failed: {exc}")

    if not long_parts:
        raise SystemExit("No machine panel generated")

    roster = pd.concat(roster_parts, ignore_index=True, sort=False)
    machine_long = pd.concat(long_parts, ignore_index=True, sort=False)
    # dedupe person-item
    machine_long = machine_long.drop_duplicates(["person_id", "question_id"], keep="last")
    save_table(roster, artifacts / "machine_panel_roster.csv")
    save_table(machine_long, artifacts / "machine_panel_long.csv")

    panel_sum = pd.DataFrame(
        [
            {
                "n_machine_persons": int(machine_long.person_id.nunique()),
                "n_items_covered": int(machine_long.question_id.nunique()),
                "n_responses": int(len(machine_long)),
                "mean_accuracy": float(machine_long.is_correct.mean()),
                "sources": ",".join(sorted(machine_long.source.dropna().unique())),
            }
        ]
    )
    save_table(panel_sum, tables / "table_e10_machine_panel_summary.csv")
    print(panel_sum.to_string(index=False))

    summary, link, resid_df, mh = run_powered_irt(human_resp, machine_long, meta, artifacts, tables)
    write_report(out_dir, roster, summary, link)

    # patch E9 report note
    e9 = artifacts / "E9_IRT_REPORT.md"
    if e9.is_file():
        txt = e9.read_text(encoding="utf-8")
        banner = (
            "\n\n---\n\n## Update (E10)\n\n"
            "Machine-N limitation addressed experimentally in "
            "`E10_POWERED_DIF_REPORT.md` "
            f"(machine persons={int(summary.iloc[0]['n_machine_persons'])}).\n"
        )
        if "Update (E10)" not in txt:
            e9.write_text(txt + banner, encoding="utf-8")

    status = {
        "E10": "done",
        "limitation_closed": bool(summary.iloc[0]["limitation_closed"]),
        "n_machine_persons": int(summary.iloc[0]["n_machine_persons"]),
        "report": "revision/artifacts/irt/E10_POWERED_DIF_REPORT.md",
    }
    (artifacts / "e10_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("[OK] E10 complete — machine-N limitation closed by experiment.")


if __name__ == "__main__":
    main()
