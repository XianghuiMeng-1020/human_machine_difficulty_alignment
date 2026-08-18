# R2 — LLM × Region Estimand Reconciliation (chi2=290.96 vs 282.35)

Both numbers use the **same** region definition (canonical Longformer 3-seed-majority region, tie→middle;
see `audit/evidence/region_count_reconciliation.md`). The discrepancy is **not** a computational
error and **not** a region-definition difference — it is two different, both legitimate, outcome
estimands. Verified by direct independent recomputation from `outputs/llm/consensus_by_item.csv` +
`outputs/encoder/seed_item_regions.csv`.

## Analysis A — CONDITIONAL consensus error (SENSITIVITY)

Universe: only the 4,873 items with a valid 2-of-3 consensus. Outcome: consensus answer
incorrect=1 / correct=0.

| region (canonical majority) | correct | incorrect |
|---|---:|---:|
| ambiguous | 1524 | 71 |
| easy | 1175 | 4 |
| hard | 707 | 127 |
| middle | 1241 | 24 |

```text
n = 4873
chi2 = 282.345, df = 3, p = 6.58e-61
Cramer's V = 0.2407
```

**This is the value previously frozen in `outputs/diagnostics/g6_stats.json` as
`llm_incorrect_x_region` (chi2=282.35, V≈0.238–0.241 depending on rounding).**

## Analysis B — UNCONDITIONAL solver failure (PRIMARY)

Universe: all 4,887 RACE items. Outcome: (incorrect consensus OR no-consensus)=1 /
correct consensus=0.

| region (canonical majority) | fail | success |
|---|---:|---:|
| ambiguous | 74 | 1524 |
| easy | 5 | 1175 |
| hard | 134 | 707 |
| middle | 27 | 1241 |

```text
n = 4887
chi2 = 290.960, df = 3, p = 9.00e-63
Cramer's V = 0.2440
```

**This is the value obtained in the v2.1 independent recomputation (chi2=290.96, V=0.244).**

## Conclusion

290.96/0.244 = unconditional (all 4,887 items, no-consensus counted as failure);
282.35/0.241 = conditional (4,873 consensus-only items). Both are internally consistent,
correctly computed, and qualitatively identical (moderate LLM-failure/region association,
concentrated in the `hard` region). Neither is "wrong."

## Manuscript policy (adopted)

Per the governing preference stated in R2 and consistent with Reviewer 2's explicit objection to
hiding no-consensus cases:

```text
PRIMARY   = Analysis B (unconditional failure, n=4887, chi2=290.96, V=0.244)
SENSITIVITY = Analysis A (conditional consensus-error, n=4873, chi2=282.35, V=0.241)
```

This ordering is scientifically appropriate here because the "unconditional failure" definition
does not silently drop the 14 no-consensus items into a denominator hole, and the two estimates
are close enough (ΔV=0.003) that the choice does not change the qualitative conclusion — it only
changes which one is reported first.

Files: `outputs/race_final/llm_region_conditional.csv`, `outputs/race_final/llm_region_unconditional.csv`.
