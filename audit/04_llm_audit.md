# 04 LLM audit

## Independent recomputation (integrated)

```json
{
  "n_total": 4887,
  "n_no_consensus": 17,
  "n_consensus": 4870,
  "acc_consensus_only": 0.9544147843942505,
  "acc_unconditional_nocon_as_incorrect": 0.9510947411499898,
  "acc_gpt4o": 0.9326647564469914,
  "n_gpt4o": 4886,
  "acc_doubao": 0.9611213423368119,
  "n_doubao": 4887,
  "acc_deepseek": 0.9404542664211173,
  "n_deepseek": 4887,
  "retry_rate_gpt4o": 0.0,
  "retry_rate_doubao": 0.2936361776140782,
  "retry_rate_deepseek": 0.0,
  "retry_rate_any": 0.2936361776140782
}
```

No-consensus table `revision/tables/table_e4d_no_consensus.csv` matches overall 17/4887.

## Raw logs

See `evidence/llm_raw_log_audit.json`.

- GPT/Doubao jsonl exist under `LLM_out/`.
- Sample GPT keys lack `temperature` / `access_date` / `model_id`.
- HIGH fills dated 2026-07-26 present.
- Canonical integrated has retry/temperature columns but **not** provider/model/access_date/raw_response.

## Retry protocol risk

Manuscript/response states temperature may vary 0.1–1.5 on parse retries. Without per-call temperature retained for all backends in raw logs, deterministic vs stochastic calls cannot be fully separated → **C3/R2.3 PARTIAL**.

## Verdict

**B2/R2.5: PASS** (denominators/no-cons). **C3/R2.3: PARTIAL** (identity+date+raw retention incomplete in canonical package).
