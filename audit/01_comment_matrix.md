# 01 Comment matrix

| ID | Concern | Required evidence | Current artifact | Independent verification | Status | Blocker/next |
|---|---|---|---|---|---|---|
| A1 | Official RACE dev split counts | Raw JSONL enumeration | evidence/race_split_counts.json | Independent script matches 4887/1436/3451 | PASS | None |
| A2 | Round / fabricated counts | Exact empirical tables; no hardcoded analysis counts | evidence/hardcode_scan_hits.csv; reconcile claim_checks | Region/LLM/encoder counts recompute exactly; Response EeDi 27613 unsupported | PARTIAL | Locate EeDi extract used for 27613 or retract that denominator |
| A3 | Integrated table + code; prereg | Canonical table + scripts; registry ID or remove claim | race_val_integrated.csv; scripts/revision/* | Table exists & reconciles; missing provenance cols; no registry ID found (removal OK) | PARTIAL | Add epoch/model/seed/provider/access_date/raw_response fields or normalize multi-table schema |
| B1 | Encoder confusion vs accuracy | Confusion sum == n; accuracy == correct/n | reconcile_integrated.json | confusion_sum=4887; acc=0.7409 | PASS | None |
| B2 | Table5/6 LLM denominators | Consensus/unconditional acc + n | reconcile_integrated.json; table_e4d | consensus 4870; acc_cons=0.9544; nocon=17 | PASS | None |
| B3 | Encoder valid trained reader | Above chance; tiny-overfit; ≥3 seeds; checkpoint match | encoder_checkpoint_audit.json; epoch_metrics | 74.1% Longformer / 68% BigBird; learning curve rises; **only 1 seed**; structural tiny-overfit not re-executed this audit | PARTIAL | P0: run ≥3 seeds; document tiny-overfit/gradient audits |
| C1 | Held-out vs original Cartography | Construct label + val dynamics files | training_dynamics_val.csv; e1 scripts | Val dynamics present (held-out). Must not claim original train Cartography | PARTIAL | Ensure all claims say held-out generalization dynamics |
| C2 | Grade-band accuracy inversion | Recompute band accuracies + explanation | stats_sensitivity_audit.json; table_e4c | MIDDLE>HIGH encoder 78.5>72.3; LLM 96.9>94.8; truncation 0%; length HIGH>MIDDLE | PASS | Keep wording as exam-source band, not 'harder' |
| C3 | Consensus filter + retries | Retry rates, temps, no-cons by band/region | table_e4d; llm_raw_log_audit; integrated retry cols | no-cons analyzed; retry cols exist; older GPT jsonl lacks temperature; access_date not in canonical | PARTIAL | Merge raw logs with model ID/access timestamps into provenance table |
| C4 | EeDi reliability | Attempt dist; threshold sweep; IRT/shrinkage | bridge_e6_eedi_audit.json; table_e5_* | Independent sweep on 948 qs; **cannot verify 27613**; min attempts=4 in raw | FAIL | P0: recover full EeDi extract or rewrite RQ1 denominators to available file |
| C5 | RQ1 role | Data-grounded route A/B decision | evidence/rq1_role_decision.md | Route B only (cross-corpus); no same-item EeDi↔RACE | PARTIAL | Lock Route B in revision prose later (after gates) |
| C6 | Blinded content-validity audit | Frozen rule, blinding, κ, effect sizes | e6_*; independent fisher/kappa | 30+30; κ=0.538; flaw 66.7% vs 30%; Fisher p=0.009 OR=4.67 | PASS | None |
| R1.1 | No same-item human/model | Bridge same-item human answers | bridge_*; table_e3a | 320 items ×30; κ human–region 0.264 > designer 0.088 | PASS | None |
| R1.2 | Review improves learning | RCT or scoped offline claim | table_e7a_* | Offline Spearman only; no live learning RCT | PARTIAL | Keep claim scoped; do not assert learning gains |
| R1.3 | MCQ-only scope | Scope decision | Response_to_Reviewers.md | Intentional MCQ scope; experimental | NOT APPLICABLE | Presentation later |
| R2.1 | Overstated alignment framing | Bridge grounding + RQ1 demotion | table_e3a; rq1_role_decision | Bridge grounds item-level; EeDi must stay cross-corpus | PARTIAL | Depends on C4/C5 lock |
| R2.2 | Formal statistical validation | χ²/V/κ/CI/OR coded outputs | stats_sensitivity_audit.json; table_e4a | χ²+Cramér V; κ tables; bootstrap CIs present | PASS | None |
| R2.3 | LLM identity/provider/date | Exact IDs in data artifacts | llm_raw_log_audit.json; integrated missing fields | Folder names + Response text; **not** in canonical columns | PARTIAL | P0 provenance columns |
| R2.4 | Tercile sensitivity | Multi-cut recomputation | table_e4b; tercile_sensitivity_independent.csv | tercile/quartile/40–60 + independent recompute | PASS | None |
| R2.5 | No-consensus analysis | Rates by band/region + alt accuracy | table_e4d; stats json | overall/band/region rates; unconditional acc | PASS | None |
| R2.6 | Figure/language/denominators | Presentation | — | Not audited as PASS (presentation) | NOT STARTED | After experiment gates |
| I1 | Figure coords vs region table | Same columns/cut points | integrated quantiles; legacy scatter CSV | Cut points compatible with mean/std ranges; PNG not regenerated from audit script | PARTIAL | Regenerate scatter from integrated + overlay cuts; count points==4887 |
| I2 | Tables from named filters | subset_flow + scripts | subset_flow.csv | Flow stages saved; not every table has attached filter JSON | PARTIAL | Emit filter JSON beside each table |
| I3 | BigBird tests core conclusion | Band×dynamics / LLM×dynamics for BigBird | training_dynamics_val.csv exists | Dynamics exist; no first-class BigBird region×band table in revision/tables | PARTIAL | Run e4 analyses on BigBird regions |
| I4 | Percentages from integers | Recompute | reconcile claim_checks | Key % match counts | PASS | None |
| I5 | Exclusion row-flow | subset_flow | subset_flow.csv | Partial flow only | PARTIAL | Expand to full exclusion audit |

Statuses used only: PASS / PARTIAL / FAIL / BLOCKED / NOT STARTED / NOT APPLICABLE.