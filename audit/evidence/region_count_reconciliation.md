# R3 — RACE Region-Count Version Reconciliation (v2.1 write-time truth pass)

All rows independently recomputed from the raw per-epoch training-dynamics files using the single
canonical assignment rule (`scripts/revision/common.py::_assign`, precedence order
ambiguous→hard→easy→middle-default, tercile cut points on `mean_prob`/`std_prob`):

```text
region = "middle"
if std_prob  >= sigma_p67: region = "ambiguous"
if mean_prob <= mu_p33 and frac_correct < 0.5: region = "hard"
if mean_prob >= mu_p67 and std_prob <= sigma_p33: region = "easy"
```

| region source | easy | middle | hard | ambiguous | total | item universe |
|---|---:|---:|---:|---:|---:|---|
| legacy original manuscript | 135 | 597 | 474 | 230 | **1,436** | `race_trainedmodels_5e-4_e5_256bs/.../training_dynamics_val.csv`, 3 epochs — **NOT the canonical 4,887-item dev split** (n=1,436 exactly equals the MIDDLE-band count; this artifact predates P0 canonicalization and must never be quoted as "the RACE region counts") |
| old single-seed revised run | 1,189 | 1,175 | 1,148 | 1,375 | 4,887 | `revision/artifacts/encoder_competitive/allenai_longformer-base-4096/training_dynamics_val.csv`, 4 epochs, single seed, canonical 4,887-item universe — source of the previously-circulated "1189/1375/1175/1148" set |
| Longformer seed 0 | 1,237 | 1,181 | 856 | 1,613 | 4,887 | `outputs/encoder/seed_item_regions.csv`, seed=0 |
| Longformer seed 1 | 1,155 | 1,213 | 906 | 1,613 | 4,887 | `outputs/encoder/seed_item_regions.csv`, seed=1 |
| Longformer seed 2 | 1,225 | 1,188 | 861 | 1,613 | 4,887 | `outputs/encoder/seed_item_regions.csv`, seed=2 |
| **Longformer 3-seed majority (CANONICAL)** | **1,180** | **1,268** | **841** | **1,598** | **4,887** | 2/3-majority vote across seed 0/1/2, ties→middle |
| BigBird (single run, tercile) | 803 | 1,205 | 1,266 | 1,613 | 4,887 | `outputs/encoder/architecture_check/bigbird_threshold_sensitivity.csv`, spec=tercile (BigBird's own default/canonical run) |

## Three-seed majority construction detail (n=4,887)

```text
number with 3/3 same label (all three seeds agree)        = 2,287
number resolved by a genuine 2/3 majority                 = 2,347
number with all three seeds different (no majority exists) =   253
number of those 253 assigned "middle" by the tie rule       =   253  (100% of ties go to middle by convention)
Total: 2,287 + 2,347 + 253 = 4,887 ✓
```

(`outputs/race_final/region_stability_full.csv`, `outputs/encoder/seed_summary.json`)

## Manuscript canonical definition

Per governing policy:

```text
PRIMARY encoder evidence   = continuous held-out confidence/generalization dynamics
                              (per-seed mean_prob / std_prob / final correctness; see Part O below)
SECONDARY categorical repr. = Longformer THREE-SEED MAJORITY region
                              (easy=1,180, middle=1,268, hard=841, ambiguous=1,598; ties→middle)
```

All other rows in the table above are explicitly **legacy** (pre-canonicalization item universe),
**seed-specific** (a single Longformer seed, not the majority), or **architecture-specific**
(BigBird) and must never be substituted for the canonical majority counts in headline Results.
