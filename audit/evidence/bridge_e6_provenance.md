# Bridge / E6 provenance (G9)

## What “Bridge” means

**Bridge-RACE** is a same-item human answering collection on a stratified sample of official
RACE validation MCQs. Adult participants select A/B/C/D without seeing gold keys or model labels.

## What 320×30 means

- **320 items** sampled from RACE validation, stratified by grade band × data-map region
  (8 strata × 40 items; see `revision/bridge/PROTOCOL_bridge_race.json` / sample design tables).
- **30 independent human attempts per item** → 9,600 responses
  (`revision/bridge/bridge_race_responses.csv`).
- Annotators: **200** distinct `annotator_id` values in the response file (adult participants).

## Independent Bridge recompute

- Responses: 9600
- Items: 320
- Attempts/item: min=30, max=30
- Mean correctness: 0.415312
- Project alignment table (for κ definitions):  
                    pair    kappa   ci_low  ci_high   n
human_bucket_vs_designer 0.088235 0.052605 0.127843 320
  human_bucket_vs_region 0.263682 0.193699 0.333842 320

## What E6 means

**E6** is a blinded content-validity audit of multi-source disagreement flags.

- Sample: 30 high-disagreement + 30 low-disagreement items (`e6_arm_key_HIDDEN.csv`).
- Raters: **human** raters labeled `R1` and `R2` (not LLM judges).
- Blinding: arm key stored separately as HIDDEN; rating UI designed to hide audit arm.
- Rubric codes: ambiguous_key, flawed_distractors, evidence_not_locatable_in_passage,
  multiple_plausible_answers, other_item_flaw, no_flaw (`e6_coding_rubric.json`).
- Primary outcome: any-flaw = not `no_flaw`.
- Item-level any-flaw = max across the two raters.

## Independent E6 recompute

| Arm | n | any-flaw | rate |
|---|---:|---:|---:|
| high_disagreement | 30 | 20 | 0.6667 |
| low_disagreement | 30 | 9 | 0.3000 |

- 2×2 counts: high_flaw=20, high_nofław=10, low_flaw=9, low_nofław=21
- Fisher exact OR=4.666667, p=0.00920597
- Cohen κ(R1,R2) on any-flaw=0.537915

## Claim boundary

Supports: disagreement flags enrich for detectable item-quality problems under this rubric.  
Does **not** support: improved learning outcomes or live recommendation quality (no RCT).

## Ethics / admin

Consent version recorded with Bridge responses (`consent_version` column). Secondary RACE/EeDi
analyses use public datasets. Full institutional documentation is outside this machine audit.

## Status

Human raters confirmed → E6 can support G9 PASS for content-validity enrichment claim.
