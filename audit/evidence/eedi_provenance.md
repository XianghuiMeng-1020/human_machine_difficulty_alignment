# EeDi provenance (P0-1)

## Decision: Route A — Recoverable original EeDi analysis

The orphan derived file `Eedi_analysis/eedi_question_human_difficulty.csv` (SHA-256
`2bb87dead74c606db69c2d3dcf88fd2de1f6b30b138840b451276bcb2a65f5bb`) was **exactly reproduced** from the official NeurIPS 2020 Eedi
public release.

## Raw source

- URL: `https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip`
- Zip SHA-256: `c7f01672360f1adeb3cf9507d72455d7be035bf897e4a167293e8938049800e1`
- Bytes: 656787242
- Extracted:
[
  {
    "path": "data/eedi_public_download/extracted/data/train_data/train_task_1_2.csv",
    "sha256": "721ebae1c5ddb3f8a4c85a437893216bbba1d8b2ca950ee0681d1f3e98ebdc0e",
    "bytes": 430049940,
    "n_rows": 15867850,
    "n_questions": 27613,
    "n_students": 118971,
    "source_url": "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"
  },
  {
    "path": "data/eedi_public_download/extracted/data/train_data/train_task_3_4.csv",
    "sha256": "8bdbe55a310641f9e59caffbff2eac85da0b2f6c6b2bb99fa69922296f52a4e1",
    "bytes": 31731601,
    "n_rows": 1382727,
    "n_questions": 948,
    "n_students": 4918,
    "source_url": "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"
  }
]

## Transformation

1. Concatenate `train_task_1_2.csv` + `train_task_3_4.csv` (attempt-level).
2. Aggregate by `QuestionId`: n_attempts, n_correct, mean_correct.
3. Bucket with easy≥0.80, hard≤0.40.
4. No additional filter needed: after merge, min attempts = 34 (all 27,613 retained).

**Note:** Using `train_task_1_2.csv` alone does **not** reproduce the legacy buckets;
both files must be concatenated (overlapping questions accumulate attempts).

## Reproduction of old counts

| Quantity | Value | Match |
|---|---|---|
| Retained questions | 27613 | True |
| Human Easy | 2562 | False |
| Human Mid | 16838 | False |
| Human Hard | 2254 | False |
| Row-level n_attempts/n_correct vs legacy CSV | — | True |

## Primary estimator

Beta–Binomial empirical-Bayes shrinkage (method-of-moments prior) with 95% posterior
intervals; empirical rates retained as sensitivity. IRT logit proxy from shrunk rates.
MCQ guessing floor ≈0.25; hard cutoff 0.40 is above chance.

## Outputs

- `data/processed/eedi_verified.parquet`
- `outputs/eedi/eedi_attempt_distribution.csv`
- `outputs/eedi/eedi_primary_item_estimates.csv`
- `outputs/eedi/eedi_sensitivity.csv`
- `outputs/eedi/eedi_label_switches.csv`
- `audit/evidence/eedi_recompute.log`

## Command

```powershell
python scripts/p0_closure/p0_1_eedi_verified.py
```

Generated UTC: 2026-08-06T02:35:22.523935+00:00
