# 00 Repository inventory

- **Audit time (UTC):** 2026-08-06T02:05:54.750055+00:00
- **Repo path:** `E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment`
- **Branch:** `audit/fode-experiments-20260806`
- **HEAD:** `9a9a334bffadc648c95b008b581a24948d716c62`
- **Uncommitted work:** preserved (not discarded). `git status --short` lines: 27

## Git freeze

```
9a9a334b Initial commit for DifficultyAlignment (without huge CSVs)
```

## Where is the real experiment repository?

`E:\Digedu SI - DifficultyAlignment\Digedu SI - DifficultyAlignment` (nested under workspace `Digedu SI - DifficultyAlignment`). Primary revision experiments live under `revision/` + `scripts/revision/`. Legacy pre-revision artifacts also exist under `race_prepared/`, `race_analysis_with_datamap/`, `LLM_out/`.

## Raw inputs

| Asset | Path | Role |
|---|---|---|
| RACE raw JSONL | `data/RACE/{train,dev,test}_{mid,high}.jsonl` | Untouched official-style dumps |
| EeDi attempts | `data/eedi/train_data/train_task_3_4.csv` | Only EeDi attempt file present |
| LLM raw logs | `LLM_out/gpt4o_1124/`, `LLM_out/doubao_1.8/` | Backend response jsonl |
| Bridge humans | `revision/bridge/bridge_race_responses.csv` | New revision human answers |
| E6 ratings | `revision/audit/e6_ratings*.csv` | Blind audit ratings |

Hashes: `audit/evidence/file_inventory.csv`, `race_split_counts.json`.

## Canonical derived data

| Asset | Path | SHA-256 (if hashed) |
|---|---|---|
| Prepared val MCQ | `race_prepared/race_mcq_val.csv` | see comparisons JSON |
| Integrated RQ2/RQ3 table | `revision/artifacts/race_val_integrated.csv` | `44fde6a9bd3cfe3c171aa97a69b4d34082051fb779f21af45674641253fa2e8d` |
| LLM vote aggregate | `revision/artifacts/llm_vote_val.csv` | inventory |
| Encoder dynamics (val) | `revision/artifacts/encoder_competitive/*/training_dynamics_val.csv` | inventory |
| Revision tables | `revision/tables/table_*.csv` | inventory |

## Final outputs (revision experiments)

`revision/tables/*`, `revision/artifacts/*`, Bridge/E6 packs, IRT panels (`table_e9_*`, `table_e10_*`, `table_e11_*`), figures under `race_analysis_with_datamap/`.

## Pre-revision vs new

- **Before revision (mtime ~2025-12 / 2026-03):** `LLM_out/*` base logs, `race_prepared/*`, `race_datamap_audit/*`, early datamap PNGs.
- **New revision (2026-07-26..27):** `revision/**`, encoder_competitive checkpoints, HIGH fills, Bridge/E6, E4–E11 tables, `_revision_materials/` (note: audit does **not** treat TeX edits as evidence).

## Can every headline number be traced?

| Claim family | Traceable from raw/canonical? |
|---|---|
| RACE 4887 / 1436 / 3451 | YES — independent raw enumeration |
| Region 1189/1375/1148/1175 | YES — from integrated |
| Longformer 74.1% | YES — integrated + run_meta |
| LLM consensus 4870 / ~95.4% | YES — integrated |
| Bridge 320×30 | YES — bridge files |
| E6 66.7% vs 30%, κ≈0.54 | YES — independent recompute |
| EeDi n=27,613 ≥34 attempts | **NO** — available raw has 948 questions |
| Multi-seed encoder stability | **NO** — single seed metas only |

Evidence index: `audit/evidence/`.
