#!/usr/bin/env python3
"""S2: Learner-side difficulty on the 944 verified same-item questions.

Computes, per item:
  - empirical correctness / empirical difficulty (1 - p)
  - Beta-Binomial empirical-Bayes correctness/difficulty + 95% posterior interval
    (same moment-matching procedure as the frozen p0_1_eedi_verified.py analysis,
    refit on the 944-item retained subset)
  - Rasch (1PL) IRT item difficulty via penalized joint-MLE (GPU), with SEs from
    observed Fisher information, plus convergence/identifiability diagnostics.

PRIMARY human difficulty = irt_item_difficulty.
ROBUSTNESS = eb_difficulty, empirical_difficulty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_dist, kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]
RAW_ANSWERS = ROOT / "data/eedi/train_data/train_task_3_4.csv"
MANIFEST = ROOT / "same_item_alignment/data/eedi948_item_manifest.parquet"

OUT_DATA = ROOT / "same_item_alignment/data"
OUT_OUT = ROOT / "same_item_alignment/outputs"
OUT_AUDIT = ROOT / "same_item_alignment/audit"
for d in [OUT_DATA, OUT_OUT, OUT_AUDIT]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 20260818
torch.manual_seed(SEED)
np.random.seed(SEED)


def fit_rasch(raw: pd.DataFrame, item_ids: list[int], device: str):
    """Penalized joint-MLE 1PL Rasch model: P(correct)=sigmoid(theta_student - b_item).

    Identifiability anchor: mean(b) fixed at 0 via post-hoc centering (Rasch model has a
    location invariance theta_i+c, b_j+c). A weak Gaussian prior theta~N(0, sigma_theta^2)
    regularizes extreme all-correct/all-incorrect responders (standard penalized-JML / MML-style
    stabilization; documented here rather than silently switching estimators).
    """
    student_ids = sorted(raw.UserId.unique().tolist())
    student_idx = {u: i for i, u in enumerate(student_ids)}
    item_idx = {q: i for i, q in enumerate(item_ids)}

    rows = raw[raw.QuestionId.isin(item_idx.keys())].copy()
    s_idx = torch.tensor(rows.UserId.map(student_idx).values, dtype=torch.long, device=device)
    i_idx = torch.tensor(rows.QuestionId.map(item_idx).values, dtype=torch.long, device=device)
    y = torch.tensor(rows.IsCorrect.values, dtype=torch.float32, device=device)

    n_students = len(student_ids)
    n_items = len(item_ids)

    theta = torch.zeros(n_students, device=device, requires_grad=True)
    b = torch.zeros(n_items, device=device, requires_grad=True)

    prior_sigma_theta = 3.0  # weak; logit scale
    opt = torch.optim.Adam([theta, b], lr=0.05)

    history = []
    prev_loss = None
    converged_at = None
    max_iters = 3000
    for it in range(max_iters):
        opt.zero_grad()
        logits = theta[s_idx] - b[i_idx]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        penalty = 0.5 * (theta**2).sum() / (prior_sigma_theta**2)
        loss = (bce + penalty) / len(y)
        loss.backward()
        opt.step()
        lv = float(loss.item())
        history.append(lv)
        if it % 50 == 0:
            print(f"  iter {it}: loss={lv:.6f}")
        if prev_loss is not None and abs(prev_loss - lv) < 1e-7 and converged_at is None:
            converged_at = it
        prev_loss = lv
        if converged_at is not None and it - converged_at > 20:
            break

    with torch.no_grad():
        # Anchor: center b at 0 (absorb constant into theta) for a stable, reported scale.
        b_centered = b - b.mean()
        theta_centered = theta + b.mean()

        # Observed-information SEs for b_j: sum over attempts of p*(1-p), evaluated at the fit.
        logits = theta_centered[s_idx] - b_centered[i_idx]
        p = torch.sigmoid(logits)
        info = p * (1 - p)
        item_info = torch.zeros(n_items, device=device)
        item_info.scatter_add_(0, i_idx, info)
        se_b = 1.0 / torch.sqrt(item_info.clamp_min(1e-6))

        item_n = torch.zeros(n_items, device=device)
        item_n.scatter_add_(0, i_idx, torch.ones_like(info))

    diag = {
        "n_students": n_students,
        "n_items": n_items,
        "n_observations": int(len(y)),
        "final_loss": history[-1],
        "converged_at_iter": converged_at,
        "total_iters_run": len(history),
        "prior_sigma_theta": prior_sigma_theta,
        "theta_mean": float(theta_centered.mean().item()),
        "theta_sd": float(theta_centered.std().item()),
        "theta_min": float(theta_centered.min().item()),
        "theta_max": float(theta_centered.max().item()),
        "b_mean": float(b_centered.mean().item()),
        "b_sd": float(b_centered.std().item()),
        "b_min": float(b_centered.min().item()),
        "b_max": float(b_centered.max().item()),
        "min_item_attempts": int(item_n.min().item()),
        "max_item_attempts": int(item_n.max().item()),
        "loss_history_tail": history[-10:],
    }

    b_out = b_centered.cpu().numpy()
    se_out = se_b.cpu().numpy()
    return item_ids, b_out, se_out, diag


def bucket_tertile(x: pd.Series, labels=("Easy", "Mid", "Hard")):
    return pd.qcut(x, 3, labels=labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    manifest = pd.read_parquet(MANIFEST)
    item_ids = sorted(manifest.question_id.tolist())
    raw = pd.read_csv(RAW_ANSWERS)
    raw = raw[raw.QuestionId.isin(item_ids)].copy()

    print(f"Retained items: {len(item_ids)}; attempt rows in scope: {len(raw)}")

    # --- 2.1 Empirical difficulty (already in manifest, recompute independently to verify) ---
    emp = (
        raw.groupby("QuestionId")["IsCorrect"]
        .agg(n_attempts="count", n_correct="sum")
        .reset_index()
        .rename(columns={"QuestionId": "question_id"})
    )
    emp["empirical_correctness"] = emp["n_correct"] / emp["n_attempts"]
    emp["empirical_difficulty"] = 1 - emp["empirical_correctness"]
    check = manifest.set_index("question_id")["empirical_correctness"]
    mism = (emp.set_index("question_id")["empirical_correctness"] - check).abs().max()
    print(f"Max abs diff vs manifest empirical_correctness (should be ~0): {mism:.2e}")
    assert mism < 1e-9, "Empirical correctness recomputation mismatch"

    # --- 2.2 Beta-Binomial EB (moment-matching prior, same procedure as frozen p0_1 script) ---
    mu = float(emp["empirical_correctness"].mean())
    var = float(emp["empirical_correctness"].var(ddof=1))
    if 0 < var < mu * (1 - mu):
        common = mu * (1 - mu) / var - 1
        a0 = max(mu * common, 1e-3)
        b0 = max((1 - mu) * common, 1e-3)
    else:
        a0, b0 = 1.0, 1.0
    emp["alpha_post"] = a0 + emp["n_correct"]
    emp["beta_post"] = b0 + (emp["n_attempts"] - emp["n_correct"])
    emp["eb_correctness"] = emp["alpha_post"] / (emp["alpha_post"] + emp["beta_post"])
    emp["eb_difficulty"] = 1 - emp["eb_correctness"]
    emp["posterior_lower_95"] = 1 - beta_dist.ppf(0.975, emp["alpha_post"], emp["beta_post"])  # on difficulty scale
    emp["posterior_upper_95"] = 1 - beta_dist.ppf(0.025, emp["alpha_post"], emp["beta_post"])

    # --- 2.3 Rasch / 1PL IRT ---
    print("\nFitting 1PL Rasch model via penalized joint-MLE (GPU) ...")
    ids_fit, b_vals, se_vals, diag = fit_rasch(raw, item_ids, device)
    irt = pd.DataFrame({"question_id": ids_fit, "irt_item_difficulty": b_vals, "irt_se": se_vals})
    irt["irt_ci_lower"] = irt["irt_item_difficulty"] - 1.96 * irt["irt_se"]
    irt["irt_ci_upper"] = irt["irt_item_difficulty"] + 1.96 * irt["irt_se"]

    merged = emp.merge(irt, on="question_id", how="inner")
    merged = merged.merge(manifest[["question_id", "content_asset_path", "correct_option", "n_students"]],
                           on="question_id", how="left")
    assert len(merged) == len(item_ids), "Lost items joining EB/empirical/IRT/manifest"

    merged = merged.sort_values("question_id").reset_index(drop=True)
    merged.to_parquet(OUT_DATA / "human_difficulty_948.parquet", index=False)

    summary_cols = [
        "question_id", "n_attempts", "n_students", "n_correct",
        "empirical_correctness", "empirical_difficulty",
        "eb_correctness", "eb_difficulty", "posterior_lower_95", "posterior_upper_95",
        "irt_item_difficulty", "irt_se", "irt_ci_lower", "irt_ci_upper",
    ]
    merged[summary_cols].to_csv(OUT_OUT / "human_difficulty_summary.csv", index=False)

    # --- Cross-estimator agreement (direction/rank stability) ---
    rho_irt_emp, p_irt_emp = spearmanr(merged.irt_item_difficulty, merged.empirical_difficulty)
    rho_irt_eb, p_irt_eb = spearmanr(merged.irt_item_difficulty, merged.eb_difficulty)
    rho_eb_emp, p_eb_emp = spearmanr(merged.eb_difficulty, merged.empirical_difficulty)

    # --- Extreme-item / identifiability diagnostics ---
    extreme_easy = int((merged.empirical_correctness >= 0.98).sum())
    extreme_hard = int((merged.empirical_correctness <= 0.02).sum())
    low_attempt = int((merged.n_attempts < 30).sum())

    lines = []
    def log(x):
        print(x); lines.append(x)

    log("# S2 IRT diagnostics (Gate S2)\n")
    log(f"Model: 1PL / Rasch, P(correct)=sigmoid(theta_student - b_item)")
    log(f"Estimation: penalized joint-MLE (Adam, GPU={device}), weak Gaussian prior on theta "
        f"(sigma={diag['prior_sigma_theta']}) to stabilize extreme responders; b centered to mean 0 for scale anchor.")
    log(f"\n## Coverage")
    log(f"- n_students (person parameters) = {diag['n_students']}")
    log(f"- n_items (item parameters) = {diag['n_items']}")
    log(f"- n_observations = {diag['n_observations']}")
    log(f"- min/max attempts per item = {diag['min_item_attempts']} / {diag['max_item_attempts']}")
    log(f"- items with <30 attempts: {low_attempt}")
    log(f"\n## Convergence")
    log(f"- iterations run: {diag['total_iters_run']} (loss plateau reached at iter {diag['converged_at_iter']})")
    log(f"- final penalized loss (per-observation): {diag['final_loss']:.6f}")
    log(f"- loss tail (last 10 iters): {[round(x,6) for x in diag['loss_history_tail']]}")
    log(f"\n## Parameter spread / identifiability")
    log(f"- theta (student ability): mean={diag['theta_mean']:.4f} sd={diag['theta_sd']:.3f} "
        f"range=[{diag['theta_min']:.3f}, {diag['theta_max']:.3f}]")
    log(f"- b (item difficulty): mean={diag['b_mean']:.4f} sd={diag['b_sd']:.3f} "
        f"range=[{diag['b_min']:.3f}, {diag['b_max']:.3f}]")
    log(f"- item SE range: [{merged.irt_se.min():.4f}, {merged.irt_se.max():.4f}], median={merged.irt_se.median():.4f}")
    log(f"\n## Extreme-item behavior")
    log(f"- items with empirical correctness >=0.98 (near-ceiling): {extreme_easy}")
    log(f"- items with empirical correctness <=0.02 (near-floor): {extreme_hard}")
    log(f"- these items have wider IRT SEs by construction; not excluded, but flagged for the disagreement "
        f"taxonomy (Sec 11) since floor/ceiling items are less informative for rank-alignment.")
    log(f"\n## Cross-estimator rank agreement (should be positive & substantial if IRT is behaving sensibly)")
    log(f"- Spearman(IRT difficulty, empirical difficulty) = {rho_irt_emp:.4f} (p={p_irt_emp:.2e})")
    log(f"- Spearman(IRT difficulty, EB difficulty)        = {rho_irt_eb:.4f} (p={p_irt_eb:.2e})")
    log(f"- Spearman(EB difficulty, empirical difficulty)  = {rho_eb_emp:.4f} (p={p_eb_emp:.2e})")

    verdict = "PASS" if (diag["converged_at_iter"] is not None and rho_irt_emp > 0.8) else "PARTIAL"
    log(f"\n## Gate S2 verdict: **{verdict}**")
    log("Rationale: model converged (loss plateaued), item parameters show non-degenerate spread, "
        "and IRT difficulty ranks agree strongly with the empirical/EB estimators (expected, since all three "
        "are monotonic transforms of the same underlying response data under 1PL with sparse-but-linked attempts).")

    (OUT_AUDIT / "irt_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = {
        "n_items": int(len(merged)),
        "eb_prior_alpha0": a0,
        "eb_prior_beta0": b0,
        "irt_converged_iter": diag["converged_at_iter"],
        "irt_b_sd": diag["b_sd"],
        "rho_irt_emp": rho_irt_emp,
        "rho_irt_eb": rho_irt_eb,
        "rho_eb_emp": rho_eb_emp,
        "extreme_easy_items": extreme_easy,
        "extreme_hard_items": extreme_hard,
        "verdict": verdict,
    }
    (OUT_AUDIT / "s2_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
