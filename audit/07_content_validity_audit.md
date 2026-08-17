# 07 Content-validity audit

## Artifacts

- Items/arm key: `revision/audit/e6_arm_key_HIDDEN.csv`
- Ratings: `revision/audit/e6_ratings.csv` (+ R1/R2)
- Rubric: `revision/audit/e6_coding_rubric.json`
- Tables: `table_e6_flaw_by_arm.csv`, `table_e6_interrater_kappa.csv`

## Independent recompute (this audit)

- Item-level any-flaw (max across raters): high 20/30=66.7%, low 9/30=30.0%
- Fisher exact OR=4.667, p=0.00921
- Cohen κ(R1,R2)=0.5379

Matches project tables.

## Blinding / protocol

Collector/apps and HIDDEN arm key indicate blinding design. Full protocol JSON present.

## Claim boundary

Supports enrichment of item-quality problems among high-disagreement flags.  
Does **not** support improved learning outcomes (R1.2 remains offline-only).

## Verdict

**C6: PASS**
