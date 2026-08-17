#!/usr/bin/env python3
"""Generate audit/*.md reports from evidence/ (no manuscript edits)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parents[1]
EVID = AUDIT / "evidence"


def read_json(name):
    p = EVID / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def git(cmd):
    return subprocess.check_output(["git"] + cmd, cwd=ROOT, text=True).strip()


def main():
    commit = git(["rev-parse", "HEAD"])
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = git(["status", "--short"])
    log = git(["log", "-n", "10", "--oneline"])
    now = datetime.now(timezone.utc).isoformat()

    split = read_json("race_split_counts.json")
    cmp_ = read_json("race_split_comparisons.json")
    rec = read_json("reconcile_integrated.json")
    be = read_json("bridge_e6_eedi_audit.json")
    llm = read_json("llm_raw_log_audit.json")
    enc = read_json("encoder_checkpoint_audit.json")
    stats = read_json("stats_sensitivity_audit.json")
    summary = read_json("audit_run_summary.json")

    # ----- 00 inventory -----
    (AUDIT / "00_repository_inventory.md").write_text(
        f"""# 00 Repository inventory

- **Audit time (UTC):** {now}
- **Repo path:** `{ROOT}`
- **Branch:** `{branch}`
- **HEAD:** `{commit}`
- **Uncommitted work:** preserved (not discarded). `git status --short` lines: {len(status.splitlines()) if status else 0}

## Git freeze

```
{log}
```

## Where is the real experiment repository?

`{ROOT}` (nested under workspace `Digedu SI - DifficultyAlignment`). Primary revision experiments live under `revision/` + `scripts/revision/`. Legacy pre-revision artifacts also exist under `race_prepared/`, `race_analysis_with_datamap/`, `LLM_out/`.

## Raw inputs

| Asset | Path | Role |
|---|---|---|
| RACE raw JSONL | `data/RACE/{{train,dev,test}}_{{mid,high}}.jsonl` | Untouched official-style dumps |
| EeDi attempts | `data/eedi/train_data/train_task_3_4.csv` | Only EeDi attempt file present |
| LLM raw logs | `LLM_out/gpt4o_1124/`, `LLM_out/doubao_1.8/` | Backend response jsonl |
| Bridge humans | `revision/bridge/bridge_race_responses.csv` | New revision human answers |
| E6 ratings | `revision/audit/e6_ratings*.csv` | Blind audit ratings |

Hashes: `audit/evidence/file_inventory.csv`, `race_split_counts.json`.

## Canonical derived data

| Asset | Path | SHA-256 (if hashed) |
|---|---|---|
| Prepared val MCQ | `race_prepared/race_mcq_val.csv` | see comparisons JSON |
| Integrated RQ2/RQ3 table | `revision/artifacts/race_val_integrated.csv` | `{rec.get('sha256','')}` |
| LLM vote aggregate | `revision/artifacts/llm_vote_val.csv` | inventory |
| Encoder dynamics (val) | `revision/artifacts/encoder_competitive/*/training_dynamics_val.csv` | inventory |
| Revision tables | `revision/tables/table_*.csv` | inventory |

## Final outputs (revision experiments)

`revision/tables/*`, `revision/artifacts/*`, Bridge/E6 packs, IRT panels (`table_e9_*`, `table_e10_*`, `table_e11_*`), figures under `race_analysis_with_datamap/`.

## Pre-revision vs new

- **Before revision (mtime ~2025-12 / 2026-03):** `LLM_out/*` base logs, `race_prepared/*`, `race_datamap_audit/*`, early datamap PNGs.
- **New revision (2026-07-26..27):** `revision/**`, encoder_competitive checkpoints, HIGH fills, Bridge/E6, E4–E11 tables, `_revision_materials/` (note: audit does **not** treat TeX edits as evidence).

## Can every headline number be traced?

| Claim family | Traceable from raw/canonical? |
|---|---|
| RACE 4887 / 1436 / 3451 | YES — independent raw enumeration |
| Region 1189/1375/1148/1175 | YES — from integrated |
| Longformer 74.1% | YES — integrated + run_meta |
| LLM consensus 4870 / ~95.4% | YES — integrated |
| Bridge 320×30 | YES — bridge files |
| E6 66.7% vs 30%, κ≈0.54 | YES — independent recompute |
| EeDi n=27,613 ≥34 attempts | **NO** — available raw has 948 questions |
| Multi-seed encoder stability | **NO** — single seed metas only |

Evidence index: `audit/evidence/`.
""",
        encoding="utf-8",
    )

    # ----- 01 matrix -----
    rows = [
        ("A1", "Official RACE dev split counts", "Raw JSONL enumeration", "evidence/race_split_counts.json", "Independent script matches 4887/1436/3451", "PASS", "None"),
        ("A2", "Round / fabricated counts", "Exact empirical tables; no hardcoded analysis counts", "evidence/hardcode_scan_hits.csv; reconcile claim_checks", "Region/LLM/encoder counts recompute exactly; Response EeDi 27613 unsupported", "PARTIAL", "Locate EeDi extract used for 27613 or retract that denominator"),
        ("A3", "Integrated table + code; prereg", "Canonical table + scripts; registry ID or remove claim", "race_val_integrated.csv; scripts/revision/*", "Table exists & reconciles; missing provenance cols; no registry ID found (removal OK)", "PARTIAL", "Add epoch/model/seed/provider/access_date/raw_response fields or normalize multi-table schema"),
        ("B1", "Encoder confusion vs accuracy", "Confusion sum == n; accuracy == correct/n", "reconcile_integrated.json", "confusion_sum=4887; acc=0.7409", "PASS", "None"),
        ("B2", "Table5/6 LLM denominators", "Consensus/unconditional acc + n", "reconcile_integrated.json; table_e4d", "consensus 4870; acc_cons=0.9544; nocon=17", "PASS", "None"),
        ("B3", "Encoder valid trained reader", "Above chance; tiny-overfit; ≥3 seeds; checkpoint match", "encoder_checkpoint_audit.json; epoch_metrics", "74.1% Longformer / 68% BigBird; learning curve rises; **only 1 seed**; structural tiny-overfit not re-executed this audit", "PARTIAL", "P0: run ≥3 seeds; document tiny-overfit/gradient audits"),
        ("C1", "Held-out vs original Cartography", "Construct label + val dynamics files", "training_dynamics_val.csv; e1 scripts", "Val dynamics present (held-out). Must not claim original train Cartography", "PARTIAL", "Ensure all claims say held-out generalization dynamics"),
        ("C2", "Grade-band accuracy inversion", "Recompute band accuracies + explanation", "stats_sensitivity_audit.json; table_e4c", "MIDDLE>HIGH encoder 78.5>72.3; LLM 96.9>94.8; truncation 0%; length HIGH>MIDDLE", "PASS", "Keep wording as exam-source band, not 'harder'"),
        ("C3", "Consensus filter + retries", "Retry rates, temps, no-cons by band/region", "table_e4d; llm_raw_log_audit; integrated retry cols", "no-cons analyzed; retry cols exist; older GPT jsonl lacks temperature; access_date not in canonical", "PARTIAL", "Merge raw logs with model ID/access timestamps into provenance table"),
        ("C4", "EeDi reliability", "Attempt dist; threshold sweep; IRT/shrinkage", "bridge_e6_eedi_audit.json; table_e5_*", "Independent sweep on 948 qs; **cannot verify 27613**; min attempts=4 in raw", "FAIL", "P0: recover full EeDi extract or rewrite RQ1 denominators to available file"),
        ("C5", "RQ1 role", "Data-grounded route A/B decision", "evidence/rq1_role_decision.md", "Route B only (cross-corpus); no same-item EeDi↔RACE", "PARTIAL", "Lock Route B in revision prose later (after gates)"),
        ("C6", "Blinded content-validity audit", "Frozen rule, blinding, κ, effect sizes", "e6_*; independent fisher/kappa", "30+30; κ=0.538; flaw 66.7% vs 30%; Fisher p=0.009 OR=4.67", "PASS", "None"),
        ("R1.1", "No same-item human/model", "Bridge same-item human answers", "bridge_*; table_e3a", "320 items ×30; κ human–region 0.264 > designer 0.088", "PASS", "None"),
        ("R1.2", "Review improves learning", "RCT or scoped offline claim", "table_e7a_*", "Offline Spearman only; no live learning RCT", "PARTIAL", "Keep claim scoped; do not assert learning gains"),
        ("R1.3", "MCQ-only scope", "Scope decision", "Response_to_Reviewers.md", "Intentional MCQ scope; experimental", "NOT APPLICABLE", "Presentation later"),
        ("R2.1", "Overstated alignment framing", "Bridge grounding + RQ1 demotion", "table_e3a; rq1_role_decision", "Bridge grounds item-level; EeDi must stay cross-corpus", "PARTIAL", "Depends on C4/C5 lock"),
        ("R2.2", "Formal statistical validation", "χ²/V/κ/CI/OR coded outputs", "stats_sensitivity_audit.json; table_e4a", "χ²+Cramér V; κ tables; bootstrap CIs present", "PASS", "None"),
        ("R2.3", "LLM identity/provider/date", "Exact IDs in data artifacts", "llm_raw_log_audit.json; integrated missing fields", "Folder names + Response text; **not** in canonical columns", "PARTIAL", "P0 provenance columns"),
        ("R2.4", "Tercile sensitivity", "Multi-cut recomputation", "table_e4b; tercile_sensitivity_independent.csv", "tercile/quartile/40–60 + independent recompute", "PASS", "None"),
        ("R2.5", "No-consensus analysis", "Rates by band/region + alt accuracy", "table_e4d; stats json", "overall/band/region rates; unconditional acc", "PASS", "None"),
        ("R2.6", "Figure/language/denominators", "Presentation", "—", "Not audited as PASS (presentation)", "NOT STARTED", "After experiment gates"),
        ("I1", "Figure coords vs region table", "Same columns/cut points", "integrated quantiles; legacy scatter CSV", "Cut points compatible with mean/std ranges; PNG not regenerated from audit script", "PARTIAL", "Regenerate scatter from integrated + overlay cuts; count points==4887"),
        ("I2", "Tables from named filters", "subset_flow + scripts", "subset_flow.csv", "Flow stages saved; not every table has attached filter JSON", "PARTIAL", "Emit filter JSON beside each table"),
        ("I3", "BigBird tests core conclusion", "Band×dynamics / LLM×dynamics for BigBird", "training_dynamics_val.csv exists", "Dynamics exist; no first-class BigBird region×band table in revision/tables", "PARTIAL", "Run e4 analyses on BigBird regions"),
        ("I4", "Percentages from integers", "Recompute", "reconcile claim_checks", "Key % match counts", "PASS", "None"),
        ("I5", "Exclusion row-flow", "subset_flow", "subset_flow.csv", "Partial flow only", "PARTIAL", "Expand to full exclusion audit"),
    ]

    mat = ["# 01 Comment matrix", "",
           "| ID | Concern | Required evidence | Current artifact | Independent verification | Status | Blocker/next |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        mat.append("| " + " | ".join(r) + " |")
    mat.append("")
    mat.append("Statuses used only: PASS / PARTIAL / FAIL / BLOCKED / NOT STARTED / NOT APPLICABLE.")
    (AUDIT / "01_comment_matrix.md").write_text("\n".join(mat), encoding="utf-8")

    # ----- 02 data -----
    (AUDIT / "02_data_audit.md").write_text(
        f"""# 02 Data audit

## A1 RACE split

**Command:** `python audit/recompute/a1_race_raw_split_audit.py`

**Result:** raw dev = **{split.get('dev_total')}** (MIDDLE **{split.get('dev_middle')}**, HIGH **{split.get('dev_high')}**); `matches_official_dev_claim={split.get('matches_official_dev_claim')}`.

Comparisons: `evidence/race_split_comparisons.json` — `race_mcq_val` and `integrated` both `matches_raw_dev=true`.

**Status A1: PASS**

## Canonical integrated table

Path: `revision/artifacts/race_val_integrated.csv`  
SHA-256: `{rec.get('sha256')}`  
n={rec.get('n_rows')} unique_qid={rec.get('n_unique_qid')} duplicate_qid={rec.get('duplicate_qid')}

Missing provenance fields: `{rec.get('missing_provenance_fields')}`

subset_flow: `evidence/subset_flow.csv`

**Status A3 (table completeness): PARTIAL**

## Hardcoding / round numbers

Scan: `evidence/hardcode_scan_hits.csv` (many hits in docs/tex/legacy).  
Independent reconcile shows key revision tables use exact 4887-region-LLM counts (not 5000/3360 round set).

**Critical unsupported denominator:** Response letter EeDi **n=27,613** not present in repo raw (`train_task_3_4.csv` → 948 questions; `table_e5_*` agrees).

**Status A2: PARTIAL** (RACE OK; EeDi headline FAIL)
""",
        encoding="utf-8",
    )

    # ----- 03 encoder -----
    metas = enc.get("run_metas", [])
    (AUDIT / "03_encoder_audit.md").write_text(
        f"""# 03 Encoder audit

## Checkpoints found

```json
{json.dumps(metas, indent=2)}
```

Longformer epoch curve (`epoch_metrics.jsonl`): 0.695 → 0.737 → 0.740 → 0.741.

## Independent accuracy

From integrated: accuracy={rec.get('encoder',{}).get('accuracy')} by band={rec.get('encoder',{}).get('by_band')}  
Confusion sum OK: {rec.get('assert_confusion_sum_eq_n')}

## Seeds

Only single-run `run_meta.json` per architecture. **<3 seeds** → cannot PASS B3 fully.

## Structural tests this audit

| Test | Status |
|---|---|
| A/B/C/D mapping audit (≥30 items) | NOT STARTED (not re-run here) |
| Input-construction print (≥10) | NOT STARTED |
| Tiny-set overfit | NOT STARTED (smoke artifacts exist under `encoder_mc_smoke/` but not independently verified) |
| One-batch gradient | NOT STARTED |
| Baseline chance/majority | NOT STARTED |
| Truncation by band | PASS via integrated `likely_truncated_2048` all 0.0 |
| Learning curves | PARTIAL (epoch_metrics only; no full train loss JSON) |
| Checkpoint↔table link | PARTIAL (val_accuracy matches integrated) |

## Construct C1

`training_dynamics_val.csv` confirms **held-out validation** dynamics. Do not label as original Dataset Cartography train-set learning dynamics.

## BigBird I3

Val dynamics file exists (4887×4). No dedicated BigBird band×region / LLM×region revision table found → PARTIAL.

## Verdict

**B3: PARTIAL** (above chance, competitive Longformer; missing multi-seed + formal structural battery)
""",
        encoding="utf-8",
    )

    # ----- 04 llm -----
    (AUDIT / "04_llm_audit.md").write_text(
        f"""# 04 LLM audit

## Independent recomputation (integrated)

```json
{json.dumps(rec.get('llm',{}), indent=2)}
```

No-consensus table `revision/tables/table_e4d_no_consensus.csv` matches overall 17/4887.

## Raw logs

See `evidence/llm_raw_log_audit.json`.

- GPT/Doubao jsonl exist under `LLM_out/`.
- Sample GPT keys lack `temperature` / `access_date` / `model_id`.
- HIGH fills dated 2026-07-26 present.
- Canonical integrated has retry/temperature columns but **not** provider/model/access_date/raw_response.

## Retry protocol risk

Manuscript/response states temperature may vary 0.1–1.5 on parse retries. Without per-call temperature retained for all backends in raw logs, deterministic vs stochastic calls cannot be fully separated → **C3/R2.3 PARTIAL**.

## Verdict

**B2/R2.5: PASS** (denominators/no-cons). **C3/R2.3: PARTIAL** (identity+date+raw retention incomplete in canonical package).
""",
        encoding="utf-8",
    )

    # ----- 05 stats -----
    (AUDIT / "05_statistics_audit.md").write_text(
        f"""# 05 Statistics audit

**Command:** `python audit/recompute/a_stats_sensitivity.py`

## Band × region

χ²={stats.get('band_x_region',{}).get('chi2')}, p={stats.get('band_x_region',{}).get('p')}, Cramér's V={stats.get('band_x_region',{}).get('cramers_v')}

## LLM incorrect × region

{json.dumps(stats.get('llm_incorrect_x_region',{}), indent=2)}

## Inversion C2

encoder_middle_gt_high={stats.get('encoder_middle_gt_high')}  
encoder_acc_by_band={stats.get('encoder_acc_by_band')}  
llm_acc_by_band={stats.get('llm_acc_by_band_consensus')}  
truncation_by_band={stats.get('truncation_by_band')}

Logit model singular (collinearity with mean_prob) — report as blocked diagnostic, not as failed inversion.

## Sensitivity R2.4

`evidence/tercile_sensitivity_independent.csv` + project `table_e4b_cartography_sensitivity.csv`.

## κ tables

Project `table_e4a_agreement_kappa.csv` present with bootstrap CIs.

## Verdict

**R2.2/R2.4/C2: PASS** (with C2 wording constraints). Formal package adequate for experiment gate; presentation polishing later.
""",
        encoding="utf-8",
    )

    # ----- 06 eedi -----
    (AUDIT / "evidence" / "rq1_role_decision.md").write_text(
        f"""# RQ1 role decision

## Feasible route

**Route B — separate learner-outcome calibration module.**

## Supporting data

- Available EeDi raw: `data/eedi/train_data/train_task_3_4.csv`
- Attempts={be.get('eedi_n_attempts')}, students={be.get('eedi_n_students')}, questions={be.get('eedi_n_questions')}
- Attempts/question median={be.get('eedi_attempts_per_question',{}).get('median')}
- No item text/options in this extract for RACE-style encoder/LLM evaluation
- No licensed same-item bridge between EeDi and RACE in repo

## Route A feasibility

**Not feasible** with current assets (no shared item IDs/text across EeDi and RACE).

## Required additional work

1. Either recover the extract that allegedly yields n=27,613, **or** rewrite all RQ1 numbers to the available 948-question file.
2. Keep Bridge-RACE as the only item-level human–machine alignment evidence.
3. Do not claim EeDi↔RACE item-level alignment.

## Risks

Publishing 27,613 without the file is a data-integrity failure.

## Recommended revised claim

RQ1 = cross-corpus learner-outcome reference on the **auditable** EeDi extract; item-level alignment = Bridge-RACE only.
""",
        encoding="utf-8",
    )

    (AUDIT / "06_eedi_audit.md").write_text(
        f"""# 06 EeDi audit

## Independent raw summary

```json
{json.dumps({k: be.get(k) for k in ['eedi_n_attempts','eedi_n_students','eedi_n_questions','eedi_attempts_per_question','eedi_threshold_sweep']}, indent=2)}
```

File SHA-256: `{be.get('eedi_raw',{}).get('sha256')}`

## Conflict with revision Response claim

Response claims primary extract **n=27,613** with ≥34 attempts.  
Repo + `table_e5_attempt_count_distribution.csv` show **948** questions (min attempts 4).  
`train_task_1_2.csv` **missing**.

## Estimators present

- Threshold sweeps / shrinkage / IRT-proxy scripts under `scripts/revision/e5_eedi_reliability.py` and tables `table_e5_*` (on 948-q file).
- Full IRT model-family on EeDi subsample exists in E9 (`table_e9_e1_*`) — separate from RQ1 headline.

## Verdict

**C4: FAIL** until denominator is reconciled.  
**C5/R1.1 framing: PARTIAL** — Route B documented in `evidence/rq1_role_decision.md`.
""",
        encoding="utf-8",
    )

    # ----- 07 content -----
    (AUDIT / "07_content_validity_audit.md").write_text(
        f"""# 07 Content-validity audit

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
""",
        encoding="utf-8",
    )

    # ----- 08 final -----
    gates = [
        ("G0", "Raw inputs and full provenance available", "PARTIAL"),
        ("G1", "RACE split independently verified", "PASS"),
        ("G2", "Canonical integrated table complete", "PARTIAL"),
        ("G3", "All table denominators reconcile", "PARTIAL"),
        ("G4", "Encoder pipeline valid and above chance", "PARTIAL"),
        ("G5", "LLM protocol reproducible; no-consensus analyzed", "PARTIAL"),
        ("G6", "Formal statistics and sensitivity complete", "PASS"),
        ("G7", "EeDi reliability analysis complete", "FAIL"),
        ("G8", "Grade-band inversion resolved or explained", "PASS"),
        ("G9", "Audit-claim validation completed or appropriately scoped", "PASS"),
    ]

    gate_md = "\n".join([f"| {a} | {b} | {c} |" for a, b, c in gates])

    (AUDIT / "08_final_verdict.md").write_text(
        f"""# 08 Final verdict

## 1. Audit scope

- **Repository path:** `{ROOT}`
- **Starting commit:** `{commit}`
- **Audit branch:** `{branch}`
- **Environment:** Windows; Python with pandas/scipy/statsmodels/girth as installed in user env
- **Date/time (UTC):** {now}
- **Manuscript:** not modified in this audit (original submitted PDF/TeX treated as non-evidence for experiment PASS)

## 2. Executive verdict

# EXPERIMENTS NOT READY

RACE official-split counts, encoder/LLM headline accuracies on the integrated table, Bridge, E6, and formal band/region statistics **independently recompute**.  
Blocking gaps remain: **EeDi 27,613 denominator not in repo**, **canonical provenance fields incomplete**, **encoder multi-seed + structural battery incomplete**, **BigBird core-association tables missing**.

## 3. Comment matrix

See `01_comment_matrix.md` (full rows). Summary counts:

- PASS: A1, B1, B2, C2, C6, R1.1, R2.2, R2.4, R2.5, I4 (+G6/G8/G9)
- PARTIAL: A2, A3, B3, C1, C3, C5, R1.2, R2.1, R2.3, I1, I2, I3, I5
- FAIL: C4
- NOT STARTED / N/A: R1.3, R2.6 (presentation)

## 4. Data-integrity verdict

- Official split: **verified** 4887 = 1436+3451 from raw JSONL
- Canonical table: present & unique keys; missing provenance columns
- Hardcoding: old round manuscript numbers appear in docs/tex; revision empirical tables match recompute for RACE/LLM/encoder
- EeDi: **integrity break** on claimed 27613

## 5. Encoder verdict

Competitive Longformer 74.1% with rising epoch curve; BigBird 68.0%; confusion reconciles.  
Fails full B3 due to **single seed** and **structural tests not independently executed** this audit.

## 6. LLM verdict

Consensus/no-consensus/accuracies reconcile. Raw logs exist but canonical package lacks exact model/provider/access_date/raw_response; retry temperature provenance incomplete.

## 7. Statistical verdict

χ²/Cramér's V, κ+CIs, sensitivity cuts, inversion recomputed. Adequate for G6 PASS.

## 8. EeDi verdict

Available file supports ~948 questions with rich attempts/question.  
Headline RQ1 n=27,613 **unsupported** → G7 FAIL.

## 9. Remaining experiments

### P0 (must before manuscript writing)

1. **Reconcile EeDi denominator** — recover `train_task_1_2`/full extract OR rewrite all RQ1 numbers to auditable 948-q file; rerun e5; update Response claims.  
   - Script: `scripts/revision/e5_eedi_reliability.py`  
   - Pass: published n equals SHA-hashed raw extract groupby count
2. **Canonical provenance table** — join LLM raw logs + encoder run_meta (model, seed, access_date, raw_response pointers, retry_reason).  
   - Pass: A3 required+soft fields present or normalized multi-table with tested joins
3. **Encoder multi-seed (≥3) Longformer** — recompute region×band & LLM×region stability.  
   - Script: `scripts/revision/e1_train_mc.py --seed {{0,1,2}}` + cartography rebuild  
   - Pass: mean±sd accuracy; region switch rates documented
4. **Structural encoder battery** — mapping/input dump/tiny-overfit/gradient/baselines written under `audit/evidence/encoder_structural/`.  
   - Pass: tiny-set train acc→~1.0; nonzero grads; mapping 30/30 OK

### P1

5. BigBird full core analyses (I3)  
6. Regenerate scatter from integrated with overlay cuts (I1)  
7. Per-table filter JSON (I2/I5)  
8. LLM retry temperature stratification report

### P2

9. Classroom learners for Paper-1 crossed IRT (optional enhancement; not original editor gate)  
10. Live review A/B (R1.2) — remains future work

## 10. Hard release gates

| Gate | Requirement | Status |
|---|---|---|
{gate_md}

## 11. Final instruction

Do **not** modify the manuscript, response letter, title, abstract, or tracked-change files until all **P0** items are complete and gates **G0–G8** are PASS.

---

## Print summary

- **Current commit:** `{commit}`
- **Verdict:** EXPERIMENTS NOT READY
- **P0 blockers:** EeDi 27613 missing; provenance fields; encoder ≥3 seeds + structural battery
- **Audit artifacts:** `{AUDIT}`
- **Next command:**

```powershell
cd "{ROOT}"
python audit/recompute/a1_race_raw_split_audit.py
# then execute P0-1 EeDi reconcile before any manuscript edit
```
""",
        encoding="utf-8",
    )

    print("WROTE audit markdowns")
    print("COMMIT", commit)
    print("VERDICT EXPERIMENTS NOT READY")


if __name__ == "__main__":
    main()
