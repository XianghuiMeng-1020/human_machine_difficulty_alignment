# 03 Encoder audit

## Checkpoints found

```json
[
  {
    "path": "revision/artifacts/encoder_competitive/allenai_longformer-base-4096/run_meta.json",
    "model_name": "allenai/longformer-base-4096",
    "val_accuracy": 0.7409453392028809,
    "best_val_accuracy": 0.7409453392028809,
    "epochs": 4,
    "max_len": 1024,
    "article_words": 400,
    "lr": 2e-05,
    "trainer": "custom_pytorch_multiple_choice",
    "finished_at": "2026-07-26T20:03:50"
  },
  {
    "path": "revision/artifacts/encoder_competitive/bert-base-uncased/run_meta.json",
    "model_name": "bert-base-uncased",
    "val_accuracy": 0.5932064652442932,
    "best_val_accuracy": 0.5934110879898071,
    "epochs": 4,
    "max_len": 320,
    "article_words": 120,
    "lr": 2e-05,
    "trainer": "custom_pytorch_multiple_choice",
    "finished_at": "2026-07-26T06:15:04"
  },
  {
    "path": "revision/artifacts/encoder_competitive/google_bigbird-roberta-base/run_meta.json",
    "model_name": "google/bigbird-roberta-base",
    "val_accuracy": 0.6801719069480896,
    "best_val_accuracy": 0.6801719069480896,
    "epochs": 4,
    "max_len": 512,
    "article_words": 200,
    "lr": 2e-05,
    "trainer": "custom_pytorch_multiple_choice",
    "finished_at": "2026-07-26T08:20:59"
  }
]
```

Longformer epoch curve (`epoch_metrics.jsonl`): 0.695 → 0.737 → 0.740 → 0.741.

## Independent accuracy

From integrated: accuracy=0.7409453652547575 by band={'HIGH': 0.7226890756302521, 'MIDDLE': 0.7848189415041783}  
Confusion sum OK: True

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
