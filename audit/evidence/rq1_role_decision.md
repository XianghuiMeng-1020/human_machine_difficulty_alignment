# RQ1 role decision

## Feasible route

**Route B — separate learner-outcome calibration module.**

## Supporting data

- Available EeDi raw: `data/eedi/train_data/train_task_3_4.csv`
- Attempts=1382727, students=4918, questions=948
- Attempts/question median=1684.0
- No item text/options in this extract for RACE-style encoder/LLM evaluation
- No licensed same-item bridge between EeDi and RACE in repo

## Route A feasibility

**Not feasible** with current assets (no shared item IDs/text across EeDi and RACE).

## Required additional work

1. Either recover the extract that allegedly yields n=27,613, **or** rewrite all RQ1 numbers to the available 948-question file.
2. Keep Bridge-RACE as the only item-level human–machine alignment evidence.
3. Do not claim EeDi↔RACE item-level alignment.

## Risks

Publishing 27,613 without the file is a data-integrity failure.

## Recommended revised claim

RQ1 = cross-corpus learner-outcome reference on the **auditable** EeDi extract; item-level alignment = Bridge-RACE only.
