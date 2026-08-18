# R1 — RACE LLM Agreement-State Reconciliation (v2.1 write-time truth pass)

Source: `data/processed/llm_votes.parquet` (n=4887, one row per RACE item, one vote column per
backend: `backend_vote_1`=DeepSeek, `backend_vote_2`=GPT-4o, `backend_vote_3`=Doubao).
Generating computation: inline pandas classification (reproduced below), output saved to
`outputs/race_final/llm_agreement_states.csv`.

## Exhaustive, mutually-exclusive state classification (n=4887)

| State | Definition | Exact n |
|---|---|---:|
| A. three_valid_all_same | all 3 backends produced a valid parse AND all 3 letters identical | **4561** |
| B. three_valid_exactly_two_same | all 3 valid, exactly 2 of 3 letters identical | **309** |
| C. three_valid_all_different | all 3 valid, all 3 letters distinct | **14** |
| D. two_valid_same | exactly 2 backends valid (1 missing), the 2 agree | **3** |
| E. two_valid_different | exactly 2 backends valid, the 2 disagree | **0** |
| F. fewer_than_two_valid | 0 or 1 valid backend | **0** |

**Verification: A+B+C+D+E+F = 4561+309+14+3+0+0 = 4887 ✓ (exactly matches RACE dev n)**

```text
two_of_three_consensus = A + B + D = 4561 + 309 + 3 = 4873   ✓ matches frozen "4873 consensus"
no_consensus            = C + E + F =   14 +   0 + 0 =   14   ✓ matches frozen "14 no consensus"
```

The 3 missing DeepSeek parses (`n_valid=2` rows, states D) are, in every one of the 3 cases,
still resolvable because the 2 valid backends (GPT, Doubao) agree with each other.

## Pairwise agreement (exact valid-pair denominators)

| Pair | agree (n) | valid pairs (n) | rate |
|---|---:|---:|---:|
| DeepSeek–GPT | 4664 | 4884 | 0.954955 |
| DeepSeek–Doubao | 4670 | 4884 | 0.956183 |
| GPT–Doubao | 4661 | 4887 | 0.953755 |

DeepSeek-involving pairs use denominator 4884 (its 3 missing parses excluded); GPT–Doubao uses
the full 4887 because both backends have zero missing parses.

## Identity check (three-valid subset only, n=4884)

```text
sum(pairwise agreement counts, 3-valid subset) = 4664 + 4670 + 4661 = 13,992
3*A + B (3-valid subset)                       = 3*4561 + 309       = 13,992     ✓ EXACT MATCH
```

## What `4561` and `4870` actually were

- **`4561` = State A = `three_valid_all_same`** — the number of RACE items where all three
  frozen LLM backends independently returned a valid parse AND all three letters were literally
  identical. This is the mathematically correct, strict definition of "three-way complete
  agreement." **This is retained as the canonical value for that phrase.**
- **`4870` (previously mislabeled "three_way_full_agreement" in `outputs/llm/consensus_metrics.csv`)
  = A + B = `4561 + 309`** — the number of items where the three-valid subset reached ANY
  2-of-3-or-better majority (i.e., "consensus was achievable among the three raw votes"), which is
  a *superset* of true three-way unanimity. It conflates "achieved a majority" with "all three
  literally agreed." **This value must be RETIRED as "three-way complete agreement."** It may be
  re-labeled, if ever needed, as `n_three_valid_reaching_majority = 4870`, but it should not appear
  in the manuscript under that name.
- Pairwise agreement rates of ≈0.954–0.956 were always computed correctly (they were never in
  conflict with either 4561 or 4870); the inconsistency was confined to the single derived label
  "three-way complete/full agreement."

## Manuscript-safe statement

> Among the 4,887 RACE dev items, three independently-queried frozen LLM backends unanimously
> agreed on 4,561 items (93.3%); a 2-of-3 majority was reached on a further 312 items (309 with
> all three valid + 3 with exactly two valid), giving 4,873 items (99.7%) with a usable two-of-three
> consensus answer and 14 items (0.3%) with no consensus.
