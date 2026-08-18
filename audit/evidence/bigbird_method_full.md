# BigBird Architecture-Robustness Method (Part N, v2.1)

- Checkpoint: `google/bigbird-roberta-base` (see `03_encoder_audit.md` / `revision/artifacts/encoder_competitive/google_bigbird-roberta-base/run_meta.json`)
- Seed(s): single run (no multi-seed BigBird sweep was performed; this is a disclosed limitation -- BigBird is a ONE-seed architecture-robustness check, not a seed-robustness check)
- Training setup: epochs=4, max_len=512, article_words=200, lr=2e-05, same custom PyTorch multiple-choice trainer as Longformer/BERT baselines
- Overall dev accuracy: (see per-band below; overall = weighted mean of band accuracies)
- MIDDLE accuracy (last epoch): 0.7235
- HIGH accuracy (last epoch): 0.6621
- Held-out dynamics extraction: identical construct to Longformer -- per-epoch validation-set gold-probability trajectory (mean_prob, std_prob, last_correct), NOT original-Cartography training-set dynamics
- Region rule: same `heldout_tercile_precedence_v1`-style rule as Longformer, applied to BigBird's own held-out dynamics
- Same item universe: BigBird was scored on the identical 4,887-item RACE dev set (`race_analysis_integrated.parquet`, question_id-joined); same preprocessing family (same custom multiple-choice trainer), different max_len/article_words tuned per architecture's context-length capability (512 tokens/200 words vs. Longformer's 1024 tokens/400 words) -- this is a comparable-but-not-identical preprocessing budget, disclosed rather than hidden
- BigBird vs Longformer region switch rate: 0.5335 (53.3% of items assigned a DIFFERENT region label by BigBird vs. Longformer's majority region)
- BigBird band x region Cramer's V: 0.1084

| designer_difficulty_str   |   ambiguous |   easy |   hard |   middle |
|:--------------------------|------------:|-------:|-------:|---------:|
| HIGH                      |        1149 |    482 |    951 |      869 |
| MIDDLE                    |         464 |    321 |    315 |      336 |
