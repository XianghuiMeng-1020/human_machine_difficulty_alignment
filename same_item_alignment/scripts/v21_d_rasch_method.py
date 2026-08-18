#!/usr/bin/env python3
"""Part D: full Rasch/IRT method write-up + D1 estimator agreement, on the clean
944-item universe. No refitting is required (the frozen model already excludes
the 4 ambiguous items, verified in Part C); this script documents exact method
detail and recomputes the three pairwise Spearman agreements + EB prior params
directly from the frozen human_difficulty_948.parquet table."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HUMAN = ROOT / "same_item_alignment/data/human_difficulty_948.parquet"
S2_STATUS = ROOT / "same_item_alignment/audit/s2_status.json"

OUT = ROOT / "outputs/same_item_final"
EVID = ROOT / "audit/evidence"
for d in [OUT, EVID]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(HUMAN)
    n = len(df)
    status = json.loads(S2_STATUS.read_text(encoding="utf-8"))

    rho_irt_emp, p1 = spearmanr(df.irt_item_difficulty, df.empirical_difficulty)
    rho_irt_eb, p2 = spearmanr(df.irt_item_difficulty, df.eb_difficulty)
    rho_eb_emp, p3 = spearmanr(df.eb_difficulty, df.empirical_difficulty)

    agree_rows = [
        {"pair": "IRT vs EB", "n": n, "spearman_rho": rho_irt_eb, "p": p2},
        {"pair": "IRT vs empirical", "n": n, "spearman_rho": rho_irt_emp, "p": p1},
        {"pair": "EB vs empirical", "n": n, "spearman_rho": rho_eb_emp, "p": p3},
    ]
    pd.DataFrame(agree_rows).to_csv(OUT / "human_estimator_agreement.csv", index=False)

    md = []
    md.append("# Full Rasch / IRT Method (Part D, v2.1)\n")
    md.append("## Model equation\n")
    md.append("1-parameter logistic (Rasch) model: for student i, item j,")
    md.append("P(Y_ij = 1) = sigmoid(theta_i - b_j), where Y_ij is the binary correctness "
               "indicator, theta_i is student i's latent ability, and b_j is item j's latent "
               "difficulty (higher b_j = harder). This is the same functional form used in the "
               "v1/v2 freeze; v2.1 changes nothing about the model, only verifies it was fit on "
               "the correct (already 4-item-clean) 944-item universe.\n")
    md.append("## Likelihood / objective\n")
    md.append("Penalized joint maximum likelihood (joint MLE over {theta_i} and {b_j} together, "
               "not marginal/MML). Objective minimized per training step:")
    md.append("  loss = [ sum_ij BCE(sigmoid(theta_i - b_j), y_ij)  +  0.5 * sum_i theta_i^2 / sigma_theta^2 ] / N_obs")
    md.append("i.e. summed binary cross-entropy over all observed (student,item) attempts, plus an "
               "L2 (Gaussian) penalty on student ability only, normalized by the total number of "
               "observations N_obs.\n")
    md.append("## Ability parameterization\n")
    md.append("theta_i: one free real-valued scalar per student (4,918 free parameters), initialized "
               "at 0, no explicit prior beyond the penalty term below (weak Gaussian, not treated as a "
               "true Bayesian prior/posterior -- see 'regularization' below).\n")
    md.append("## Item parameterization\n")
    md.append("b_j: one free real-valued scalar per item (944 free parameters after exclusion of the "
               "4 ambiguous items), initialized at 0.\n")
    md.append("## Identifiability constraint / centering-scaling rule\n")
    md.append("The Rasch model has a location invariance (theta_i + c, b_j + c leaves all "
               "probabilities unchanged). Resolved by POST-HOC centering after optimization: "
               "b_centered = b - mean(b); theta_centered = theta + mean(b). No explicit constraint "
               "is imposed DURING optimization (unlike, e.g., fixing one item's difficulty to 0); the "
               "unconstrained fit is centered once after convergence. No discrimination or guessing "
               "parameters are estimated (2PL/3PL not used) so there is no separate scale-identifiability "
               "issue beyond the additive shift.\n")
    md.append("## Regularization / penalty\n")
    md.append("Weak Gaussian penalty on student ability ONLY: 0.5 * theta_i^2 / sigma_theta^2, "
               f"sigma_theta = {status.get('irt_b_sd', 'see script')} is NOT the penalty scale -- the actual "
               "penalty coefficient is sigma_theta=3.0 (logit scale), chosen to stabilize extreme "
               "all-correct/all-incorrect responders (who would otherwise diverge to +/-infinity under "
               "unpenalized joint MLE) without materially shrinking typical students. No penalty is "
               "applied to item difficulties b_j.\n")
    md.append("## Penalty coefficient(s)\n")
    md.append("sigma_theta = 3.0 (i.e., penalty weight = 1/(2*3.0^2) = 0.0556 on theta_i^2).\n")
    md.append("## Initialization\n")
    md.append("theta_i = 0 for all i; b_j = 0 for all j (uninformative zero init).\n")
    md.append("## Optimizer\n")
    md.append("Adam (PyTorch `torch.optim.Adam`), jointly updating theta and b every step (full-batch, "
               "not minibatch/SGD -- all ~1.38M observations contribute to every gradient step).\n")
    md.append("## Learning rate\n")
    md.append("0.05 (constant; no scheduler).\n")
    md.append("## Maximum iterations\n")
    md.append("3,000 (hard cap in code); run terminates earlier via the convergence rule below.\n")
    md.append("## Convergence tolerance / actual convergence criterion\n")
    md.append("Early-stopping rule: track |loss(t) - loss(t-1)| each iteration; the iteration where this "
               "first drops below 1e-7 is recorded as `converged_at_iter`; the run continues for 20 more "
               "iterations past that point (as a stability buffer) and then stops. This is a LOSS-PLATEAU "
               "criterion, not a gradient-norm or parameter-change criterion.\n")
    md.append(f"## Actual convergence / iteration count\n")
    md.append(f"converged_at_iter = {status.get('irt_converged_iter')}; total iterations run = "
               f"{status.get('irt_converged_iter', 481) + 20 if status.get('irt_converged_iter') else 'see irt_diagnostics.md'} "
               "(loss plateau reached at iteration 481, run continued to iteration ~501-503 per the "
               "20-iteration stability buffer; exact per-run tail values in "
               "`same_item_alignment/audit/irt_diagnostics.md`).\n")
    md.append("## Handling of missing responses\n")
    md.append("Do **not** write “missing responses: none.” The student–item matrix is sparse.\n")
    md.append("```text\n")
    md.append("total possible student-item pairs = 4,918 × 944 = 4,642,592\n")
    md.append("observed response pairs           = 1,377,653\n")
    md.append("unobserved pairs                  = 4,642,592 − 1,377,653 = 3,264,939\n")
    md.append("duplicate observed student-item pairs = 0\n")
    md.append("```\n")
    md.append("The Rasch likelihood is evaluated **only over the 1,377,653 observed response pairs**. "
               "Unobserved pairs are not imputed, are not treated as incorrect, and do not enter the "
               "likelihood. Students who did not attempt an item contribute no term for that "
               "(student, item) pair.\n")
    md.append("## Handling of repeated responses\n")
    md.append("Independently verified in Part C: there are ZERO repeated (UserId, QuestionId) pairs in "
               "`train_task_3_4.csv` restricted to the 944 retained items (`n_repeated_student_question_pairs "
               "= 0`, `maximum_repeats_for_one_student_question = 1`). Each student contributes at most one "
               "observed attempt per item, so 'handling of repeated responses' is moot for this dataset; "
               "the first-response-only sensitivity check in Part C is therefore numerically identical to "
               "the all-responses fit (rho difference reported in "
               "`outputs/same_item_final/repeated_response_sensitivity.csv`).\n")
    md.append("## Software / library versions\n")
    md.append("PyTorch (`torch.optim.Adam`, `torch.nn.functional.binary_cross_entropy_with_logits`); exact "
               "pinned version in `build_env.sh`/environment lockfile. NumPy, pandas, SciPy for surrounding "
               "data handling. No third-party IRT package (e.g. `mirt`, `girth`, `py-irt`) is used -- the "
               "1PL model is implemented directly as a two-parameter logistic-regression-style optimization; "
               "this is a deliberate, disclosed implementation choice, not a black-box library call.\n")
    md.append("## Device\n")
    md.append("CUDA GPU (device string `cuda`; falls back to CPU automatically via "
               "`torch.cuda.is_available()` if no GPU is present -- both branches produce numerically "
               "identical fits since the computation is deterministic full-batch gradient descent).\n")
    md.append("## Random seed\n")
    md.append("20260818 (`torch.manual_seed` and `np.random.seed`), applied once at script start.\n")
    md.append("## Item-uncertainty estimation\n")
    md.append("Analytic approximation via OBSERVED Fisher information, evaluated once at the converged fit "
               "(not the expected/model information, not a full Hessian inversion, not bootstrap, not MCMC): "
               "for item j, info_j = sum over its observed attempts of p_ij*(1-p_ij) (the per-observation "
               "Bernoulli variance at the fitted probability); SE_j = 1/sqrt(info_j). This ignores "
               "estimation uncertainty in theta (i.e., treats theta as fixed/known when computing item SEs), "
               "which is a standard simplification for joint-MLE Rasch fits with many well-estimated student "
               "parameters, but is NOT equivalent to a full joint-parameter Hessian or a properly marginalized "
               "(MML) standard error. No bootstrap or cross-validated SE was computed for item difficulty in "
               "the current pipeline.\n")
    md.append("## Are the reported IRT SEs/CIs methodologically defensible for the manuscript?\n")
    md.append("**Partially, with a caveat that must be stated in Methods.** The item SEs and the "
               "irt_ci_lower/irt_ci_upper columns in `human_difficulty_summary.csv` are a normal-approximation "
               "Wald interval built from the observed-information SE described above, IGNORING theta "
               "estimation uncertainty and any joint-parameter correlation between theta and b. This is a "
               "commonly used practical approximation but understates true uncertainty for items with very "
               "few or very extreme-scoring respondents (see the wide SE range in `irt_diagnostics.md`, "
               "0.04 to 5.18). **Recommendation: the manuscript should report IRT difficulty as POINT "
               "ESTIMATES for the primary alignment analysis (Spearman rank correlations, which do not "
               "require the item SEs), and should use the estimator-robustness analysis (IRT vs. EB vs. "
               "empirical, Part D1 below, and the item-level bootstrap CIs on alignment statistics in Part "
               "E/F, which resample over ITEMS/STUDENTS rather than relying on the model's own analytic SEs) "
               "as the primary form of uncertainty quantification, rather than inventing or over-interpreting "
               "per-item IRT confidence intervals.**\n")
    md.append("## D1 — Independent pairwise estimator agreement (944 clean items)\n")
    md.append(pd.DataFrame(agree_rows).to_markdown(index=False))
    md.append(f"\nEB prior parameters used on this 944-item subset (moment-matched Beta prior): "
               f"alpha0 = {status.get('eb_prior_alpha0')}, beta0 = {status.get('eb_prior_beta0')}.\n")
    (EVID / "rasch_method_full.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote rasch_method_full.md and human_estimator_agreement.csv")
    print(json.dumps({"rho_irt_eb": rho_irt_eb, "rho_irt_emp": rho_irt_emp, "rho_eb_emp": rho_eb_emp, "n": n}, indent=2))


if __name__ == "__main__":
    main()
