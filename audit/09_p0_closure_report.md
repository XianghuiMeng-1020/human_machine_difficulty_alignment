# P0 Closure Report — FODE-D-26-00032

- Generated UTC: 2026-08-17T16:15:36.491483+00:00
- Commit: `9a9a334bffadc648c95b008b581a24948d716c62`
- Branch: `revision/fode-p0-closure`
- Verdict: **READY FOR MANUSCRIPT REVISION**

## Gate table

| Gate | Requirement | Status |
|---|---|---|
| G0 | Raw inputs and full provenance available | PASS |
| G1 | RACE split independently verified | PASS |
| G2 | Canonical integrated table complete | PASS |
| G3 | All table denominators reconcile | PASS |
| G4 | Encoder pipeline valid and above chance | PASS |
| G5 | LLM protocol reproducible; no-consensus analyzed | PASS |
| G6 | Formal statistics and sensitivity complete | PASS |
| G7 | EeDi reliability analysis complete | PASS |
| G8 | Grade-band inversion resolved or explained | PASS |
| G9 | Audit-claim validation completed or appropriately scoped | PASS |

## Evidence pointers

### G0 — PASS
- `E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment\data\RACE\dev_mid.jsonl`
- `E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment\data\RACE\dev_high.jsonl`
- `E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment\data\eedi_public_download\extracted\data\train_data\train_task_1_2.csv`
- `E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment\data\eedi_public_download\extracted\data\train_data\train_task_3_4.csv`
- `audit/evidence/eedi_provenance.md`
- `audit/evidence/race_split_counts.json`

### G1 — PASS
- `audit/evidence/race_split_counts.json`
- `audit/evidence/race_split_audit.log`

### G2 — PASS
- `data/processed/race_items.parquet`
- `data/processed/encoder_runs.parquet`
- `data/processed/encoder_epoch_predictions.parquet`
- `data/processed/encoder_item_summaries.parquet`
- `data/processed/llm_runs.parquet`
- `data/processed/llm_responses.parquet`
- `data/processed/llm_votes.parquet`
- `data/processed/race_analysis_integrated.parquet`
- `outputs/diagnostics/join_audit.json`

### G3 — PASS
- `outputs/diagnostics/join_audit.json`
- `outputs/diagnostics/subset_flow.csv`

### G4 — PASS
- `outputs/encoder/structural_tests/`
- `outputs/encoder/seed_runs/`
- `outputs/encoder/architecture_check/`
- `outputs/encoder/seed_summary.csv`
- `audit/evidence/encoder_validation.md`

### G5 — PASS
- `outputs/llm/backend_coverage_freeze.csv`
- `outputs/llm/consensus_metrics.csv`
- `outputs/diagnostics/llm_coverage_freeze.json`
- `audit/evidence/llm_reproducibility.md`
- `configs/llm_protocol.yaml`
- `prompts/race_mcq_prompt.txt`

### G6 — PASS
- `outputs/diagnostics/g6_stats.json`
- `outputs/diagnostics/g6_band_x_region.csv`
- `outputs/diagnostics/g6_llm_incorrect_x_region.csv`

### G7 — PASS
- `data/processed/eedi_verified.parquet`
- `outputs/eedi/eedi_attempt_distribution.csv`
- `outputs/eedi/eedi_primary_item_estimates.csv`
- `outputs/eedi/eedi_sensitivity.csv`
- `outputs/eedi/eedi_label_switches.csv`
- `audit/evidence/eedi_provenance.md`
- `audit/evidence/eedi_recompute.log`

### G8 — PASS
- `outputs/diagnostics/g8_inversion.json`

### G9 — PASS
- `audit/evidence/human_ethics_provenance.md`
- `audit/evidence/human_ethics_provenance.json`

## Remaining P0 blockers

- None

## Known limitations (do not block READY)

- DeepSeek valid parses are 4884/4887, not 4887. Two-of-three still yields consensus on the 3 missing items via GPT+Doubao.
- Doubao logged 10949 API rows including 3287 AccountOverdueError rows, a 563-item never-parsed recovery, and extra accidental duplicate successes. Canonical vote = last success. First vs last changes 13 Doubao letters and 6 consensus items.
- Discrete regions are secondary and seed-sensitive (3-seed exact agreement 0.468). Lead with continuous held-out dynamics.
- Bridge-RACE and E6 human data are NOT USABLE (no IRB/platform/authorization in repo).
- Legacy LLM logs remain `legacy_nonreproducible` and are not used for G5 numbers.

## Final verified headline quantities

- EeDi Route A: official train_task_1_2 + train_task_3_4 recover 27613 / 6574 / 19460 / 1579
- RACE official val n=4887 bands={'HIGH': 3451, 'MIDDLE': 1436}
- Encoder accuracy (integrated table)=0.740741
- Longformer multi-seed val acc mean=0.7407 sd=0.0032 min=0.7373 max=0.7434
- G6 band×region χ²=131.776 p=2.2407964312210237e-28 V=0.164
- G6 LLM-incorrect×region χ²=282.345 p=6.582008365978675e-61 V=0.241
- G8 encoder MIDDLE>HIGH=True llm MIDDLE>HIGH=True acc={'HIGH': 0.7218197623877137, 'MIDDLE': 0.7862116991643454}
- DeepSeek unique=4887 parse_ok=4884/4887; GPT unique=4887 parse_ok=4887/4887; Doubao unique=4887 parse_ok=4887/4887
- API logged rows=20723 (DeepSeek 4887+GPT 4887+Doubao 10949); do not report retry_rate as the full call count
- Frozen LLM consensus=4873 no_consensus=14 2vote_cons=3 acc_cons=0.9536219987687257 acc_uncond=0.950890116635973
- Bridge-RACE and E6: NOT USABLE FOR THE REVISION (ethics/authorization undocumented); withhold human-rater claims

## Staging assertions

```json
{
  "race_total_equals_4887": true,
  "middle_plus_high": true
}
```

## Canonical paths

- `data/processed/`
- `outputs/eedi/`
- `outputs/encoder/`
- `outputs/llm/`
- `outputs/diagnostics/`
- `outputs/revision_candidate/`
- `audit/evidence/`
- `audit/09_p0_closure_report.md`

## Recommended next command

Proceed to manuscript revision using only staged numbers in `outputs/revision_candidate/`.
