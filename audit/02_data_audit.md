# 02 Data audit

## A1 RACE split

**Command:** `python audit/recompute/a1_race_raw_split_audit.py`

**Result:** raw dev = **4887** (MIDDLE **1436**, HIGH **3451**); `matches_official_dev_claim=True`.

Comparisons: `evidence/race_split_comparisons.json` — `race_mcq_val` and `integrated` both `matches_raw_dev=true`.

**Status A1: PASS**

## Canonical integrated table

Path: `revision/artifacts/race_val_integrated.csv`  
SHA-256: `44fde6a9bd3cfe3c171aa97a69b4d34082051fb779f21af45674641253fa2e8d`  
n=4887 unique_qid=4887 duplicate_qid=0

Missing provenance fields: `['llm_exact_model_id', 'llm_provider', 'llm_access_date', 'llm_raw_response', 'retry_reason', 'backend_vote', 'epoch', 'encoder_model', 'seed']`

subset_flow: `evidence/subset_flow.csv`

**Status A3 (table completeness): PARTIAL**

## Hardcoding / round numbers

Scan: `evidence/hardcode_scan_hits.csv` (many hits in docs/tex/legacy).  
Independent reconcile shows key revision tables use exact 4887-region-LLM counts (not 5000/3360 round set).

**Critical unsupported denominator:** Response letter EeDi **n=27,613** not present in repo raw (`train_task_3_4.csv` → 948 questions; `table_e5_*` agrees).

**Status A2: PARTIAL** (RACE OK; EeDi headline FAIL)
