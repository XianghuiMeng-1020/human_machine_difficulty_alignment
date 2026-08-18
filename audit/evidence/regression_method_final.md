# Regression Method — Final (Part F, v2.1)

## Why the old 3-cluster GEE was replaced

The v2 freeze used `GEE(error ~ IRT, groups=solver_id)` with exchangeable working correlation and exactly 3 solver clusters (OR=1.2523). Cluster-robust sandwich variance estimators are asymptotically justified as the NUMBER OF CLUSTERS grows; with only 3 clusters the sandwich SE is not reliable, so that OR/CI/p should not be the headline inferential regression. It is retained only as archived sensitivity history, not reported as a primary manuscript number.

## F1 — Primary item-level binomial regression (validated 2-solver ensemble)

Errors_i ~ Binomial(2, p_i); logit(p_i) = b0 + b1*z(IRT_i). n_items=944, n_trials/item=2.
- beta = 0.2593, analytic SE = 0.0484
- OR per 1 SD IRT difficulty = 1.2960
- 95% CI (analytic, model-based) = [1.1787, 1.4250]
- 95% CI (item-level bootstrap, 5000 reps, seed=20260818) = [1.1654, 1.4504]
- p = 8.567e-08
- Overdispersion (Pearson chi2/df at item level) = 1.403 (not material (<=1.5); analytic and bootstrap CIs should agree closely).

## F2 — Solver fixed-effect robustness (item-level bootstrap, 944 clusters)

machine_error ~ z(IRT) + solver_fixed_effect (solver_1, solver_2 pooled long-form, n_rows=1888).
- OR per 1 SD IRT = 1.2962, item-bootstrap 95% CI = [1.1696, 1.4550], p=8.496e-08

## F3 — Per-solver regression

| solver_id             |   n |     beta |        se |      or |   or_ci_lower_analytic |   or_ci_upper_analytic |   or_ci_lower_item_bootstrap |   or_ci_upper_item_bootstrap |           p |
|:----------------------|----:|---------:|----------:|--------:|-----------------------:|-----------------------:|-----------------------------:|-----------------------------:|------------:|
| solver_1_qwen2vl7b    | 944 | 0.18748  | 0.0674232 | 1.20621 |                1.05689 |                1.37662 |                      1.06084 |                      1.38638 | 0.00542504  |
| solver_2_internvl3_8b | 944 | 0.333738 | 0.0699222 | 1.39618 |                1.21737 |                1.60125 |                      1.22485 |                      1.59489 | 1.81517e-06 |

## F4 — Three-solver sensitivity (archived robustness, NOT headline)

- Pooled (no FE): OR=1.2112, p=1.244e-06, n_rows=2832
- + solver FE: OR=1.2170, p=9.233e-07

The legacy 3-cluster GEE result (OR=1.2523, 95% CI [1.0804,1.4515], p=0.0028) remains available in `same_item_alignment/outputs/regression_summary.csv` for audit history only.

