# Human-data / ethics provenance (freeze audit)

**Rule:** file-level facts only. No inferred IRB/HREC/exemption.

Generated from `scripts/p0_closure/p0_8_prewriting_freeze_audit.py` → `human_ethics_provenance.json`.

## Bridge-RACE — NOT USABLE FOR THE REVISION

| Field | What the repository actually contains |
|---|---|
| Exact origin | `revision/bridge/bridge_race_responses.csv` and `bridge_race_items.csv`. Protocol JSON describes a 320×30 stratified design only. |
| Collection date | CSV `timestamp` min/max are recorded in `human_ethics_provenance.json` (values fall in 2026-07). No separate collection log. |
| Newly collected vs pre-existing | **UNDOCUMENTED** as an ethics category. Files sit under `revision/` with 2026-07 timestamps. That is not authorization. |
| Recruitment / source / platform | **NOT FOUND** (no Prolific/MTurk/lab roster, no payment record, no advertisement). |
| Consent status | Column `consent_version` exists (value `v2026.07.1` on rows). **No consent form, information sheet, or consent script in the repo.** |
| Ethics / IRB / HREC | **NOT FOUND.** `revision/efficacy/e7b_lab_ab_protocol.json` says `protocol_ready_awaiting_IRB_and_recruitment` for a **different** (A/B) protocol, not Bridge-RACE. |
| Annotators | 200 distinct `annotator_id` values (`H###`). Identity, age verification, and recruitment path **not documented**. |
| Raw file | See JSON sidecar: path, SHA-256, bytes, mtime, schema `annotator_id,question_id,chosen_letter,timestamp,consent_version`. |

**Verdict: NOT USABLE FOR THE REVISION.**

Cannot demonstrate valid provenance plus appropriate authorization.  
Do **not** use 320×30 / κ / human-bucket tables in the revision.  
If Bridge-RACE is not used, the original wording “no new human data collection” may be retained. If it were used, that wording would be false.

## E6 content-validity ratings — NOT USABLE FOR THE REVISION

| Field | What the repository actually contains |
|---|---|
| Exact origin | `revision/audit/e6_ratings.csv` plus `e6_ratings_R1.csv`, `e6_ratings_R2.csv`, `e6_arm_key_HIDDEN.csv`, `e6_coding_rubric.json`. |
| Collection date | File mtimes ~2026-07-27. No rater session log. |
| Newly collected vs pre-existing | **UNDOCUMENTED.** |
| Recruitment / platform | **NOT FOUND.** |
| Consent | **NOT FOUND.** |
| Ethics / IRB / HREC | **NOT FOUND.** Do not infer that study-team coding of public items is exempt. |
| R1 / R2 | Labels only. Notes mix English and Chinese. **Cannot classify as study-team vs recruited participants.** |
| Raw file | Schema includes rubric flags + `notes`. SHA-256 in JSON sidecar. |

**Verdict: NOT USABLE FOR THE REVISION.**

Internal Fisher/κ recomputes may stay in the audit folder as diagnostics. They are **not** manuscript-safe claims.

## G9 scope after this audit

G9 is **scoped**: the revision will not use Bridge-RACE or E6 human-rater enrichment claims.  
Secondary RACE/EeDi analyses remain public-dataset secondary use (separate from Bridge/E6).
