# LLM reproducibility (frozen, single number set)

**Canonical coverage (recomputed from raw jsonl, last successful A–D parse):**

| Backend | unique items | successful parses | missing parses | final usable votes | logged API rows |
|---|---:|---:|---:|---:|---:|
| llm_deepseek_frozen_v1 | 4887 | **4884** | **3** | 4884 | 4887 |
| llm_gpt_frozen_v1 | 4887 | 4887 | 0 | 4887 | 4887 |
| llm_doubao_frozen_v1 | 4887 | 4887 | 0 | 4887 | 10949 |

Do **not** write “all three backends 4887/4887 valid parses.” DeepSeek is 4884/4887.

DeepSeek missing IDs: `middle2612.txt_q2`, `middle291.txt_q4`, `middle4542.txt_q1` (logged rows exist; unparseable).

## Two-of-three with a missing backend

- A missing/unparsed backend is a **non-vote**.
- Consensus = at least two backends return the **same** A–D letter.
- One vote ≠ consensus. Zero votes = no consensus.
- The three DeepSeek-missing items have GPT+Doubao votes: all three reached 2-vote consensus (`n_2vote_consensus=3`).

Canonical consensus (last-success): **4873 consensus / 14 no-consensus**; 4884 items have 3 votes; 3 items have 2 votes; 3-vote same-letter agreement = 4870.  
Conditioned accuracy 0.953622; unconditional (no-consensus as incorrect) 0.950890.

## Complete API / recovery accounting

Total **logged** completion rows = **20,723** (DeepSeek 4887 + GPT 4887 + Doubao 10949).  
This is **not** the `backend_metrics.retry_rate` field (that field only flags `attempt_index>0` on the last kept row).

Doubao row classes (each row = one logged call):

| Class | n |
|---|---:|
| initial_success (attempt 0, parseable) | 7661 |
| account_overdue_error | 3287 |
| same_request_retry | 1 |

- Same-request transport/parse retries: almost none (DeepSeek 0 extra rows; GPT 0; Doubao 1 classified retry).
- Later recovery after `AccountOverdueError`: 2724 items have at least one overdue row **and** a later success.
- Explicit 563-item recovery wave: at the 2026-08-15T15:45 restart, 563 items had **never** had a successful parse; those were the intended recovery set.
- An earlier aborted recharge job also wrote extra successful rows (recovery-wave rows with `request_time >= 2026-08-15T15:45`: 1528, all parseable). Those extras are **accidental duplicate reruns**, not the 563-only set.

## First-success vs last-success

Items with >1 successful Doubao response under the same frozen settings: **2775**.  
Agreement among those successful letters: **0.9953** (2762/2775).  
First-success vs last-success **changes the Doubao letter on 13 items** and **changes consensus status or option on 6 items**.

**Canonical rule for the revision:** last successful A–D parse.  
Report the 6-item first-vs-last sensitivity; do not hide recovery reruns behind “retry rate ≈ 0.”

## Frozen protocol

- `configs/llm_protocol.yaml` / `prompts/race_mcq_prompt.txt`
- temperature=0.0, top_p=1.0, max_tokens=4; retries do not change decoding
- Legacy `LLM_out/*` remains `legacy_nonreproducible` and is not used

## Contamination

RACE is public. High LLM accuracy may reflect benchmark exposure.
