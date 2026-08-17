# 06 EeDi audit

## Independent raw summary

```json
{
  "eedi_n_attempts": 1382727,
  "eedi_n_students": 4918,
  "eedi_n_questions": 948,
  "eedi_attempts_per_question": {
    "min": 4,
    "p25": 320.75,
    "median": 1684.0,
    "p90": 2646.9,
    "p95": 2828.25,
    "max": 2966
  },
  "eedi_threshold_sweep": [
    {
      "min_attempts": 5,
      "n_questions": 933,
      "n_easy": 32,
      "n_mid": 634,
      "n_hard": 267
    },
    {
      "min_attempts": 10,
      "n_questions": 915,
      "n_easy": 30,
      "n_mid": 622,
      "n_hard": 263
    },
    {
      "min_attempts": 20,
      "n_questions": 900,
      "n_easy": 30,
      "n_mid": 612,
      "n_hard": 258
    },
    {
      "min_attempts": 50,
      "n_questions": 900,
      "n_easy": 30,
      "n_mid": 612,
      "n_hard": 258
    },
    {
      "min_attempts": 100,
      "n_questions": 822,
      "n_easy": 29,
      "n_mid": 576,
      "n_hard": 217
    }
  ]
}
```

File SHA-256: `8bdbe55a310641f9e59caffbff2eac85da0b2f6c6b2bb99fa69922296f52a4e1`

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
