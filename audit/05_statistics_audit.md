# 05 Statistics audit

**Command:** `python audit/recompute/a_stats_sensitivity.py`

## Band × region

χ²=123.9356253652872, p=1.0957759785844783e-26, Cramér's V=0.1592490725553449

## LLM incorrect × region

{
  "chi2": 269.09753153551173,
  "p": 4.838529463302273e-58,
  "dof": 3,
  "cramers_v": 0.2350663026461587
}

## Inversion C2

encoder_middle_gt_high=True  
encoder_acc_by_band={'mean': {'HIGH': 0.7226890756302521, 'MIDDLE': 0.7848189415041783}, 'count': {'HIGH': 3451, 'MIDDLE': 1436}}  
llm_acc_by_band={'mean': {'HIGH': 0.9484866123399301, 'MIDDLE': 0.9686192468619247}, 'count': {'HIGH': 3436, 'MIDDLE': 1434}}  
truncation_by_band={'HIGH': 0.0, 'MIDDLE': 0.0}

Logit model singular (collinearity with mean_prob) — report as blocked diagnostic, not as failed inversion.

## Sensitivity R2.4

`evidence/tercile_sensitivity_independent.csv` + project `table_e4b_cartography_sensitivity.csv`.

## κ tables

Project `table_e4a_agreement_kappa.csv` present with bootstrap CIs.

## Verdict

**R2.2/R2.4/C2: PASS** (with C2 wording constraints). Formal package adequate for experiment gate; presentation polishing later.
