# RACE Frozen LLM Method — Full Exact Detail (Part O, v2.1)

## DeepSeek (llm_deepseek_frozen_v1)

- provider: deepseek
- exact model ID: `deepseek/deepseek-chat`
- version/snapshot: `deepseek/deepseek-chat`
- access date: 2026-08-06
- prompt: `prompts/race_mcq_prompt.txt` (sha256 19cb85f7e0cbfa59...)
- system prompt: "" (empty; no separate system turn)
- temperature: 0.0, top_p: 1.0, max_tokens: 4
- response parser: single-letter A-D extraction, `response_format=single_letter_ABCD`
- retry policy: max_retries=2, retries MUST reuse identical decoding params (`configs/llm_protocol.yaml`); temperature is never raised on retry

## GPT (llm_gpt_frozen_v1)

- provider: openai_compatible
- exact model ID: `gpt-4o`
- version/snapshot: `gpt-4o`
- access date: 2026-08-06
- prompt: `prompts/race_mcq_prompt.txt` (sha256 19cb85f7e0cbfa59...)
- system prompt: "" (empty; no separate system turn)
- temperature: 0.0, top_p: 1.0, max_tokens: 4
- response parser: single-letter A-D extraction, `response_format=single_letter_ABCD`
- retry policy: max_retries=2, retries MUST reuse identical decoding params (`configs/llm_protocol.yaml`); temperature is never raised on retry

## Doubao (llm_doubao_frozen_v1)

- provider: volcengine_ark
- exact model ID: `doubao-seed-2-0-pro-260215`
- version/snapshot: `doubao-seed-2-0-pro-260215`
- access date: 2026-08-15
- prompt: `prompts/race_mcq_prompt.txt` (sha256 19cb85f7e0cbfa59...)
- system prompt: "" (empty; no separate system turn)
- temperature: 0.0, top_p: 1.0, max_tokens: 4
- response parser: single-letter A-D extraction, `response_format=single_letter_ABCD`
- retry policy: max_retries=2, retries MUST reuse identical decoding params (`configs/llm_protocol.yaml`); temperature is never raised on retry

## Exact coverage (final, frozen v1)

- DeepSeek: 4883 / 4887 valid parses
- GPT: 4887 / 4887
- Doubao: 4887 / 4887 (from 10,949 raw logged API rows)

| llm_run_id             | exact_model_id             |   n_unique |   parse_ok_last_success |   missing_parses |   accuracy_parsed |
|:-----------------------|:---------------------------|-----------:|------------------------:|-----------------:|------------------:|
| llm_deepseek_frozen_v1 | deepseek/deepseek-chat     |       4887 |                    4884 |                3 |          0.937756 |
| llm_gpt_frozen_v1      | gpt-4o                     |       4887 |                    4887 |                0 |          0.931042 |
| llm_doubao_frozen_v1   | doubao-seed-2-0-pro-260215 |       4887 |                    4887 |                0 |          0.962758 |

## Consensus

- n_consensus = 4873
- n_no_consensus = 14
- three_way_full_agreement = 4870 (NOTE: this differs from a previously circulated '4561' figure in an earlier chat-level summary; the frozen `consensus_metrics.csv` independently recomputed here says 4870, not 4561 -- treat 4561 as SUPERSEDED/incorrect)
- conditional (consensus-only) accuracy = 0.953622
- unconditional (no-consensus-as-incorrect) accuracy = 0.950890

## No-consensus items (14) by grade band / canonical region

By band: {'HIGH': 13, 'MIDDLE': 1}
By region: {'hard': 7, 'middle': 3, 'ambiguous': 3, 'easy': 1}

## O1 — API accounting (reproducibility packet only; NOT for main Results)

- Total logged completion rows = 20,723 (DeepSeek 4887 + GPT 4887 + Doubao 10,949)
- Doubao row classes: initial_success (attempt 0, parseable) = 7,661; account_overdue_error = 3,287; same_request_retry = 1
- Later recovery after AccountOverdueError: 2,724 items have >=1 overdue row AND a later success
- Explicit 563-item recovery wave (2026-08-15T15:45 restart): 563 items had never had a successful parse before this wave; intended recovery set
- An earlier aborted recharge job wrote 1,528 EXTRA accidental-duplicate successful rows (request_time >= 2026-08-15T15:45, all parseable) -- these are NOT the 563-only intended set
- Items with >1 successful Doubao response: 2,775; agreement among those letters = 99.53% (2,762/2,775)
- First-success vs last-success: changes the Doubao letter on 13 items; changes consensus status/option on 6 items
- Canonical rule used throughout: LAST successful A-D parse

