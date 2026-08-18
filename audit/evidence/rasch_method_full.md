# Full Rasch / IRT Method (Part D, v2.1)

## Model equation

1-parameter logistic (Rasch) model: for student i, item j,
P(Y_ij = 1) = sigmoid(theta_i - b_j), where Y_ij is the binary correctness indicator, theta_i is student i's latent ability, and b_j is item j's latent difficulty (higher b_j = harder). This is the same functional form used in the v1/v2 freeze; v2.1 changes nothing about the model, only verifies it was fit on the correct (already 4-item-clean) 944-item universe.

## Likelihood / objective

Penalized joint maximum likelihood (joint MLE over {theta_i} and {b_j} together, not marginal/MML). Objective minimized per training step:
  loss = [ sum_ij BCE(sigmoid(theta_i - b_j), y_ij)  +  0.5 * sum_i theta_i^2 / sigma_theta^2 ] / N_obs
i.e. summed binary cross-entropy over all observed (student,item) attempts, plus an L2 (Gaussian) penalty on student ability only, normalized by the total number of observations N_obs.

## Ability parameterization

theta_i: one free real-valued scalar per student (4,918 free parameters), initialized at 0, no explicit prior beyond the penalty term below (weak Gaussian, not treated as a true Bayesian prior/posterior -- see 'regularization' below).

## Item parameterization

b_j: one free real-valued scalar per item (944 free parameters after exclusion of the 4 ambiguous items), initialized at 0.

## Identifiability constraint / centering-scaling rule

The Rasch model has a location invariance (theta_i + c, b_j + c leaves all probabilities unchanged). Resolved by POST-HOC centering after optimization: b_centered = b - mean(b); theta_centered = theta + mean(b). No explicit constraint is imposed DURING optimization (unlike, e.g., fixing one item's difficulty to 0); the unconstrained fit is centered once after convergence. No discrimination or guessing parameters are estimated (2PL/3PL not used) so there is no separate scale-identifiability issue beyond the additive shift.

## Regularization / penalty

Weak Gaussian penalty on student ability ONLY: 0.5 * theta_i^2 / sigma_theta^2, sigma_theta = 0.8516403436660767 is NOT the penalty scale -- the actual penalty coefficient is sigma_theta=3.0 (logit scale), chosen to stabilize extreme all-correct/all-incorrect responders (who would otherwise diverge to +/-infinity under unpenalized joint MLE) without materially shrinking typical students. No penalty is applied to item difficulties b_j.

## Penalty coefficient(s)

sigma_theta = 3.0 (i.e., penalty weight = 1/(2*3.0^2) = 0.0556 on theta_i^2).

## Initialization

theta_i = 0 for all i; b_j = 0 for all j (uninformative zero init).

## Optimizer

Adam (PyTorch `torch.optim.Adam`), jointly updating theta and b every step (full-batch, not minibatch/SGD -- all ~1.38M observations contribute to every gradient step).

## Learning rate

0.05 (constant; no scheduler).

## Maximum iterations

3,000 (hard cap in code); run terminates earlier via the convergence rule below.

## Convergence tolerance / actual convergence criterion

Early-stopping rule: track |loss(t) - loss(t-1)| each iteration; the iteration where this first drops below 1e-7 is recorded as `converged_at_iter`; the run continues for 20 more iterations past that point (as a stability buffer) and then stops. This is a LOSS-PLATEAU criterion, not a gradient-norm or parameter-change criterion.

## Actual convergence / iteration count

converged_at_iter = 481; total iterations run = 501 (loss plateau reached at iteration 481, run continued to iteration ~501-503 per the 20-iteration stability buffer; exact per-run tail values in `same_item_alignment/audit/irt_diagnostics.md`).

## Handling of missing responses

Do **not** write “missing responses: none.” The student–item matrix is sparse.

```text
total possible student-item pairs = 4,918 × 944 = 4,642,592
observed response pairs           = 1,377,653
unobserved pairs                  = 4,642,592 − 1,377,653 = 3,264,939
duplicate observed student-item pairs = 0
```

The Rasch likelihood is evaluated **only over the 1,377,653 observed response pairs**. Unobserved pairs are not imputed, are not treated as incorrect, and do not enter the likelihood. Students who did not attempt an item contribute no term for that (student, item) pair.

## Handling of repeated responses

Independently verified in Part C: there are ZERO repeated (UserId, QuestionId) pairs in `train_task_3_4.csv` restricted to the 944 retained items (`n_repeated_student_question_pairs = 0`, `maximum_repeats_for_one_student_question = 1`). Each student contributes at most one observed attempt per item, so 'handling of repeated responses' is moot for this dataset; the first-response-only sensitivity check in Part C is therefore numerically identical to the all-responses fit (rho difference reported in `outputs/same_item_final/repeated_response_sensitivity.csv`).

## Software / library versions

PyTorch (`torch.optim.Adam`, `torch.nn.functional.binary_cross_entropy_with_logits`); exact pinned version in `build_env.sh`/environment lockfile. NumPy, pandas, SciPy for surrounding data handling. No third-party IRT package (e.g. `mirt`, `girth`, `py-irt`) is used -- the 1PL model is implemented directly as a two-parameter logistic-regression-style optimization; this is a deliberate, disclosed implementation choice, not a black-box library call.

## Device

CUDA GPU (device string `cuda`; falls back to CPU automatically via `torch.cuda.is_available()` if no GPU is present -- both branches produce numerically identical fits since the computation is deterministic full-batch gradient descent).

## Random seed

20260818 (`torch.manual_seed` and `np.random.seed`), applied once at script start.

## Item-uncertainty estimation

Analytic approximation via OBSERVED Fisher information, evaluated once at the converged fit (not the expected/model information, not a full Hessian inversion, not bootstrap, not MCMC): for item j, info_j = sum over its observed attempts of p_ij*(1-p_ij) (the per-observation Bernoulli variance at the fitted probability); SE_j = 1/sqrt(info_j). This ignores estimation uncertainty in theta (i.e., treats theta as fixed/known when computing item SEs), which is a standard simplification for joint-MLE Rasch fits with many well-estimated student parameters, but is NOT equivalent to a full joint-parameter Hessian or a properly marginalized (MML) standard error. No bootstrap or cross-validated SE was computed for item difficulty in the current pipeline.

## Are the reported IRT SEs/CIs methodologically defensible for the manuscript?

**Partially, with a caveat that must be stated in Methods.** The item SEs and the irt_ci_lower/irt_ci_upper columns in `human_difficulty_summary.csv` are a normal-approximation Wald interval built from the observed-information SE described above, IGNORING theta estimation uncertainty and any joint-parameter correlation between theta and b. This is a commonly used practical approximation but understates true uncertainty for items with very few or very extreme-scoring respondents (see the wide SE range in `irt_diagnostics.md`, 0.04 to 5.18). **Recommendation: the manuscript should report IRT difficulty as POINT ESTIMATES for the primary alignment analysis (Spearman rank correlations, which do not require the item SEs), and should use the estimator-robustness analysis (IRT vs. EB vs. empirical, Part D1 below, and the item-level bootstrap CIs on alignment statistics in Part E/F, which resample over ITEMS/STUDENTS rather than relying on the model's own analytic SEs) as the primary form of uncertainty quantification, rather than inventing or over-interpreting per-item IRT confidence intervals.**

## D1 — Independent pairwise estimator agreement (944 clean items)

| pair             |   n |   spearman_rho |   p |
|:-----------------|----:|---------------:|----:|
| IRT vs EB        | 944 |       0.986512 |   0 |
| IRT vs empirical | 944 |       0.992795 |   0 |
| EB vs empirical  | 944 |       0.993235 |   0 |

EB prior parameters used on this 944-item subset (moment-matched Beta prior): alpha0 = 4.652854469530791, beta0 = 4.522257227137732.

