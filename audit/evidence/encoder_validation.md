# Encoder validation (P0-3 structural)

## Structural tests

Command: `python scripts/p0_closure/p0_3_encoder_structural.py`

| Test | Result |
|---|---|
| 30-item A/B/C/D mapping | mismatches=0 |
| 10-item input construction | `outputs/encoder/structural_tests/input_construction_audit_10.json` |
| One-batch gradient | nonzero_tensors=201, grad_norm=5.4566 |
| Tiny overfit (64, bert-base) | train_acc=1.0000, pass=True |
| Chance / majority baselines | `baselines.csv` |
| Truncation by band | `truncation_by_band.csv` (all 0.0 at flag) |
| Checkpoint identity | run_meta vs integrated abs_diff=2.605187665594144e-08 |

## Construct language

All val dynamics files are **held-out confidence / generalization dynamics**, not original
Dataset Cartography on training examples.

## Official 3-seed Longformer (P0-3 / G4)

Aggregation rule: mean/SD/min/max across seeds 0, 1, 2. Do **not** pick a single seed.

| Seed | val acc | best | finished_at UTC |
|---|---:|---:|---|
| 0 | 0.743401 | 0.743401 | 2026-08-13T10:46:40 |
| 1 | 0.741559 | 0.741559 | 2026-08-14T02:10:26 |
| 2 | 0.737262 | 0.737671 | 2026-08-14T17:31:31 |

- mean = 0.740741
- SD = 0.003150
- min = 0.737262
- max = 0.743401

**Manuscript hierarchy (freeze-verified):** PRIMARY = continuous held-out mean_prob / std_prob / last_correct. SECONDARY = discrete regions (not intrinsic difficulty).

Region rule: held-out tercile precedence per seed; canonical region = majority; ties = middle.
Three-seed exact region agreement = 0.46798 (claimed 0.468).
Pairwise: 0.6319 / 0.6259 / 0.6264 (mean ≈ 0.628).
BigBird vs Longformer region switch = 0.53346.
MIDDLE>HIGH encoder accuracy holds on every seed × {20/80, 25/75, 33/67}.
band×region Cramér's V range across those cells: 0.146–0.175.

Artifacts: `outputs/encoder/seed_runs/longformer_seed{0,1,2}/`, `seed_summary.csv`, `region_stability.csv`, `seed_g6_g8.csv`.
