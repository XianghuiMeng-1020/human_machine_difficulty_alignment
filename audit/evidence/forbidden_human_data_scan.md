# Forbidden Human-Data Scan: Bridge-RACE / E6 (Part P, v2.1)

## Status (unchanged from P0 closure)
Bridge-RACE and E6 are **NOT USABLE** as scientific evidence (no IRB/platform/authorization
documentation in repo; `audit/09_p0_closure_report.md`, `audit/evidence/human_ethics_provenance.md`).
This scan verifies they do not leak into manuscript-candidate outputs.

## What the flagged numbers mean (traced to source)
- **320** = Bridge-RACE sampled items (8 strata x 40 items).
- **9600** = 320 items x 30 independent human attempts/item.
- **66.7% / 30.0%** = E6 any-flaw rate, high-disagreement arm (20/30) vs low-disagreement arm (9/30).
- **0.00921** = E6 Fisher exact p-value (0.00920597) for the high-vs-low-disagreement flaw-rate difference.
- **0.538** = E6 Cohen's kappa (0.537915) between the two human raters (R1, R2) on any-flaw.
- **E6** = the blinded content-validity audit itself (30+30 items, two human raters).

## Scan scope and results

| Location class | Path(s) searched | Bridge/E6 markers found | Verdict |
|---|---|---|---|
| Manuscript-candidate staging | `outputs/revision_candidate/`, `outputs/revision_candidate_v21/` | **NONE** | CLEAN |
| v2.1 same-item outputs | `outputs/same_item_final/` | **NONE** (unrelated dataset entirely) | CLEAN |
| v2.1 RACE outputs | `outputs/race_final/` | **NONE** | CLEAN |
| Audit-history / experimental staging | `revision/bridge/`, `revision/tables/table_e3a_*`, `table_e6_*`, `table_e9_bridge_model_ic.csv`, `table_e10_*` | Bridge/E6 counts present (320, 9600, 66.7%, 30.0%, kappa 0.538, p=0.00921) | **HARMLESS AUDIT HISTORY** — these are the original experimental artifacts documenting why Bridge/E6 are unusable; they are not manuscript inputs |
| Frozen audit evidence | `audit/evidence/bridge_e6_provenance.md`, `audit/evidence/bridge_e6_eedi_audit.json` | Full provenance + numbers, explicitly labeled NOT USABLE | **HARMLESS AUDIT HISTORY** (this is the correct home for these numbers) |

## Conclusion
No Bridge-RACE or E6 number appears in any manuscript-candidate table, figure, or writing-packet
file produced in this v2.1 pass (`outputs/revision_candidate_v21/`, `outputs/same_item_final/`,
`outputs/race_final/`, `paper_writing/`). All occurrences of the flagged markers are confined to
clearly-labeled audit-history/staging locations (`revision/`, `audit/evidence/`) that already
carry an explicit NOT-USABLE status. **No manuscript, table, figure, supplement, or headline claim
in this packet cites Bridge-RACE or E6 evidence.**
