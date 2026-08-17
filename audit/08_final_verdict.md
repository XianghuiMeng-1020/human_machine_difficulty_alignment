# 08 Final verdict

## 1. Audit scope

- **Repository path:** `E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment`
- **Starting commit:** `9a9a334bffadc648c95b008b581a24948d716c62`
- **Audit branch:** `audit/fode-experiments-20260806`
- **Environment:** Windows; Python with pandas/scipy/statsmodels/girth as installed in user env
- **Date/time (UTC):** 2026-08-06T02:05:54.750055+00:00
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
   - Script: `scripts/revision/e1_train_mc.py --seed {0,1,2}` + cartography rebuild  
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
| G0 | Raw inputs and full provenance available | PARTIAL |
| G1 | RACE split independently verified | PASS |
| G2 | Canonical integrated table complete | PARTIAL |
| G3 | All table denominators reconcile | PARTIAL |
| G4 | Encoder pipeline valid and above chance | PARTIAL |
| G5 | LLM protocol reproducible; no-consensus analyzed | PARTIAL |
| G6 | Formal statistics and sensitivity complete | PASS |
| G7 | EeDi reliability analysis complete | FAIL |
| G8 | Grade-band inversion resolved or explained | PASS |
| G9 | Audit-claim validation completed or appropriately scoped | PASS |

## 11. Final instruction

Do **not** modify the manuscript, response letter, title, abstract, or tracked-change files until all **P0** items are complete and gates **G0–G8** are PASS.

---

## Print summary

- **Current commit:** `9a9a334bffadc648c95b008b581a24948d716c62`
- **Verdict:** EXPERIMENTS NOT READY
- **P0 blockers:** EeDi 27613 missing; provenance fields; encoder ≥3 seeds + structural battery
- **Audit artifacts:** `E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment\audit`
- **Next command:**

```powershell
cd "E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment"
python audit/recompute/a1_race_raw_split_audit.py
# then execute P0-1 EeDi reconcile before any manuscript edit
```
