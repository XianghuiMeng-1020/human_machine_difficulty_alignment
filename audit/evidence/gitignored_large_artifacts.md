# Gitignored large artifacts (scientific freeze)

These paths stay outside Git. SHA-256 and exact paths are in `scientific_freeze_sha_manifest.csv`.

| Path | Why ignored |
|---|---|
| `data/eedi_public_download/` | Official NeurIPS zip extract (~2.5 GB) |
| `data/eedi/train_data/train_task_1_2.csv` | Already ignored huge CSV |
| `outputs/encoder/seed_runs/longformer_seed{0,1,2}/model_epoch*.pt` | `*.pt` (~595 MB × 4 epochs × 3 seeds) |
| `outputs/encoder/seed_runs/**/hf_model/` | HuggingFace weights/tokenizers |
| `revision/artifacts/race_val_integrated.csv` | Large integrated dump; canonical table is `data/processed/race_analysis_integrated.parquet` |
| `.env` | Secrets |

Local presence is required to regenerate encoder/LLM/EeDi numbers. Manifest hashes bind the ignored files to this freeze.
