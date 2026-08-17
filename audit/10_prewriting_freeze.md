# Pre-writing freeze audit — FODE-D-26-00032

Do not edit the manuscript until the scientific-freeze commit is recorded below.

## 1–3. Git (filled after commit)

See `audit/evidence/p0_closure_summary.json` after `p0_6` + git tag.

## 4. G0–G9

Rerun via `python scripts/p0_closure/p0_6_stage_and_report.py` after this audit.

## 5. Bridge-RACE / E6 ethics

| Dataset | Verdict |
|---|---|
| Bridge-RACE | **NOT USABLE** |
| E6 | **NOT USABLE** |

No IRB/HREC, no recruitment platform, no consent form. R1/R2 identity undocumented.  
Details: `audit/evidence/human_ethics_provenance.md`.

## 6. Per-backend LLM coverage (raw jsonl, last success)

| Backend | unique | parse_ok | missing | usable votes |
|---|---:|---:|---:|---:|
| DeepSeek | 4887 | 4884 | 3 | 4884 |
| GPT | 4887 | 4887 | 0 | 4887 |
| Doubao | 4887 | 4887 | 0 | 4887 |

Missing DeepSeek IDs: middle2612.txt_q2, middle291.txt_q4, middle4542.txt_q1.  
Two-of-three: missing backend = non-vote; ≥2 identical letters = consensus. Those 3 items are 2-vote consensus.

## 7. API / recovery accounting

Logged rows: DeepSeek 4887 + GPT 4887 + Doubao 10949 = **20723**.  
Doubao: 7661 initial_success + 3287 AccountOverdueError + 1 same-request retry.  
563 = items with zero successful parse at the 15:45 recovery restart.  
1528 later-wave rows are extra accidental duplicates after recharge.  
2775 Doubao items have >1 success; agreement 0.9953; first vs last changes 13 letters / 6 consensus items.  
Canonical: last success.

## 8. Region stability (verified)

- 3-seed exact agreement = 0.46798
- pairwise = 0.6319 / 0.6259 / 0.6264
- BigBird–Longformer switch = 0.53346
- PRIMARY: continuous dynamics. SECONDARY: discrete regions.
- MIDDLE>HIGH and band×region V>0 on every seed × 20/80, 25/75, 33/67.

## 9. Manuscript-safe headlines

- RACE val n=4887 (MIDDLE 1436 / HIGH 3451)
- EeDi Route A n=27613 (easy 6574 / mid 19460 / hard 1579); EB switch 1.916%
- Longformer 3-seed mean 0.740741 SD 0.003150
- G6 band×region χ²=131.776 p=2.24e-28 V=0.164
- G6 LLM-incorrect×region χ²=282.345 p=6.58e-61 V=0.241
- G8 encoder MIDDLE 0.7862 > HIGH 0.7218 (exam-source wording)
- LLM: DeepSeek 4884/4887, GPT 4887, Doubao 4887; consensus 4873 / 14; acc_cons 0.9536
- Do not use Bridge-RACE or E6 numbers
- Do not call discrete regions ground-truth difficulty

## 10. Verdict

Filled after p0_6 + git freeze.
