# Alignment statistics (Sections 8-10)

n items (same-item integrated table) = 944

## PRIMARY: IRT difficulty x ensemble machine error rate
- n = 944
- Spearman rho = 0.1100
- 95% bootstrap CI = [0.0487, 0.1719]
- permutation p = 0.0014 (parametric p = 0.0007109)
- Effect-size interpretation: **weak** rank alignment (do not upgrade this label if rho falls below 0.50; a moderate/substantial rho supports partial, not complete, alignment).

## Robustness: human_eb_difficulty x machine_error_rate
- n=944 rho=0.1085 CI=[0.0477,0.1698] perm_p=0.0009998

## Robustness: human_empirical_difficulty x machine_error_rate
- n=944 rho=0.1097 CI=[0.0488,0.1716] perm_p=0.0014

## Per-model: IRT difficulty x solver_1_qwen2vl7b error
- n=944 rho=0.0941 p=0.003816

## Per-model: IRT difficulty x solver_2_internvl3_8b error
- n=944 rho=0.1376 p=2.223e-05

## Per-model: IRT difficulty x solver_3_smolvlm2_2b error
- n=944 rho=0.0089 p=0.7839

## Section 9: gradient test
- Spearman(decile rank, machine accuracy) = -0.8182, p=0.003815
- Monotonic-decrease interpretation: supported (machine accuracy expected to fall as human difficulty rises; rho should be negative and significant).

## Section 10: regression (pooled GEE, solver-clustered; used in place of a mixed-effects logit because GEE with an exchangeable working correlation gives population-averaged, solver-clustered inference that converges reliably for K=3 clusters)
- beta(human_irt_difficulty) = 0.2250, SE=0.0753
- OR = 1.2523 (95% CI [1.0804, 1.4515])
- p = 0.002823
- n (item x solver rows) = 2832
