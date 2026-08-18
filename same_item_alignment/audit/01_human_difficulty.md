# S2 IRT diagnostics (Gate S2)

Model: 1PL / Rasch, P(correct)=sigmoid(theta_student - b_item)
Estimation: penalized joint-MLE (Adam, GPU=cuda), weak Gaussian prior on theta (sigma=3.0) to stabilize extreme responders; b centered to mean 0 for scale anchor.

## Coverage
- n_students (person parameters) = 4918
- n_items (item parameters) = 944
- n_observations = 1377653
- min/max attempts per item = 4 / 2966
- items with <30 attempts: 47

## Convergence
- iterations run: 503 (loss plateau reached at iter 481)
- final penalized loss (per-observation): 0.559087
- loss tail (last 10 iters): [0.559088, 0.559088, 0.559088, 0.559088, 0.559088, 0.559088, 0.559087, 0.559087, 0.559087, 0.559087]

## Parameter spread / identifiability
- theta (student ability): mean=-0.3780 sd=1.098 range=[-2.633, 4.984]
- b (item difficulty): mean=0.0000 sd=0.852 range=[-5.370, 2.251]
- item SE range: [0.0405, 5.1793], median=0.0584

## Extreme-item behavior
- items with empirical correctness >=0.98 (near-ceiling): 3
- items with empirical correctness <=0.02 (near-floor): 0
- these items have wider IRT SEs by construction; not excluded, but flagged for the disagreement taxonomy (Sec 11) since floor/ceiling items are less informative for rank-alignment.

## Cross-estimator rank agreement (should be positive & substantial if IRT is behaving sensibly)
- Spearman(IRT difficulty, empirical difficulty) = 0.9928 (p=0.00e+00)
- Spearman(IRT difficulty, EB difficulty)        = 0.9865 (p=0.00e+00)
- Spearman(EB difficulty, empirical difficulty)  = 0.9932 (p=0.00e+00)

## Gate S2 verdict: **PASS**
Rationale: model converged (loss plateaued), item parameters show non-degenerate spread, and IRT difficulty ranks agree strongly with the empirical/EB estimators (expected, since all three are monotonic transforms of the same underlying response data under 1PL with sparse-but-linked attempts).
