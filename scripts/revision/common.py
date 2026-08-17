"""Shared utilities for FODE revision experiments (E0–E8)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = REPO_ROOT / "revision"
LETTER2LABEL = {"A": 0, "B": 1, "C": 2, "D": 3}
LABEL2LETTER = {v: k for k, v in LETTER2LABEL.items()}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_option_letter(text) -> Optional[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    s = str(text).strip().upper()
    if not s:
        return None
    if s in LETTER2LABEL:
        return s
    # Common LLM formats: "C.", "Answer: B", "(A)", "The answer is D"
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])\.", s)
    if m:
        return m.group(1)
    return None


def letter_to_label(letter: Optional[str]) -> Optional[int]:
    if letter is None:
        return None
    return LETTER2LABEL.get(letter)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cohen_kappa(y1: np.ndarray, y2: np.ndarray, labels=None) -> float:
    """Cohen's kappa for categorical labels (no sklearn dependency required)."""
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    mask = ~(pd.isna(y1) | pd.isna(y2))
    y1, y2 = y1[mask], y2[mask]
    if len(y1) == 0:
        return float("nan")
    if labels is None:
        labels = sorted(set(y1.tolist()) | set(y2.tolist()))
    label_index = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        if a not in label_index or b not in label_index:
            continue
        cm[label_index[a], label_index[b]] += 1
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n ** 2)
    if pe == 1:
        return 1.0 if po == 1 else float("nan")
    return float((po - pe) / (1 - pe))


def bootstrap_kappa(y1, y2, n_boot: int = 200, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    mask = ~(pd.isna(y1) | pd.isna(y2))
    y1, y2 = y1[mask], y2[mask]
    point = cohen_kappa(y1, y2)
    if len(y1) < 5:
        return {"kappa": point, "ci_low": float("nan"), "ci_high": float("nan"), "n": len(y1)}
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y1), len(y1))
        boots.append(cohen_kappa(y1[idx], y2[idx]))
    boots = np.asarray(boots, dtype=float)
    return {
        "kappa": point,
        "ci_low": float(np.nanpercentile(boots, 2.5)),
        "ci_high": float(np.nanpercentile(boots, 97.5)),
        "n": int(len(y1)),
    }


def assign_datamap_region(
    df: pd.DataFrame,
    mu_q_low: float = 0.33,
    mu_q_high: float = 0.67,
    sigma_q_low: float = 0.33,
    sigma_q_high: float = 0.67,
    frac_hard: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Assign easy/ambiguous/hard/middle using quantile cut points."""
    out = df.copy()
    mu_low = float(np.quantile(out["mean_prob"], mu_q_low))
    mu_high = float(np.quantile(out["mean_prob"], mu_q_high))
    sigma_low = float(np.quantile(out["std_prob"], sigma_q_low))
    sigma_high = float(np.quantile(out["std_prob"], sigma_q_high))

    def _assign(row):
        mu, sigma, frac = row["mean_prob"], row["std_prob"], row["frac_correct"]
        region = "middle"
        if sigma >= sigma_high:
            region = "ambiguous"
        if mu <= mu_low and frac < frac_hard:
            region = "hard"
        if mu >= mu_high and sigma <= sigma_low:
            region = "easy"
        return region

    out["datamap_region"] = out.apply(_assign, axis=1)
    cuts = {
        "mu_low": mu_low,
        "mu_high": mu_high,
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
        "mu_q_low": mu_q_low,
        "mu_q_high": mu_q_high,
        "sigma_q_low": sigma_q_low,
        "sigma_q_high": sigma_q_high,
    }
    return out, cuts


def compute_td_metrics(td_df: pd.DataFrame) -> pd.DataFrame:
    td = td_df.drop_duplicates(subset=["question_id", "epoch"], keep="first")
    g = td.groupby("question_id")
    agg = g.agg(
        mean_prob=("prob_correct", "mean"),
        std_prob=("prob_correct", "std"),
        frac_correct=("is_correct", "mean"),
        num_epochs=("epoch", "nunique"),
    ).reset_index()
    agg["std_prob"] = agg["std_prob"].fillna(0.0)
    return agg


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[OK] wrote {path} ({len(df)} rows)")
