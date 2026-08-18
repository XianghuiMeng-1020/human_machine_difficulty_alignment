# EeDi 944-Item Human Data Universe (Part C, v2.1)

- Original content-available questions: 948
- Retained same-item questions: 944
- Excluded questions: 4 (QuestionId 43, 84, 206, 860)

## Excluded questions -- exact detail

- QuestionId 43: observed CorrectAnswer value counts = {'2': 230, '3': 2}; raw response log contains >1 distinct CorrectAnswer value for this QuestionId; no single unambiguous ground-truth key exists in source provenance data; excluded rather than resolved by majority vote (would be a fabricated resolution not present in source of truth).
- QuestionId 84: observed CorrectAnswer value counts = {'1': 2175, '4': 452}; raw response log contains >1 distinct CorrectAnswer value for this QuestionId; no single unambiguous ground-truth key exists in source provenance data; excluded rather than resolved by majority vote (would be a fabricated resolution not present in source of truth).
- QuestionId 206: observed CorrectAnswer value counts = {'3': 1455, '1': 745}; raw response log contains >1 distinct CorrectAnswer value for this QuestionId; no single unambiguous ground-truth key exists in source provenance data; excluded rather than resolved by majority vote (would be a fabricated resolution not present in source of truth).
- QuestionId 860: observed CorrectAnswer value counts = {'2': 10, '3': 5}; raw response log contains >1 distinct CorrectAnswer value for this QuestionId; no single unambiguous ground-truth key exists in source provenance data; excluded rather than resolved by majority vote (would be a fabricated resolution not present in source of truth).

No manual adjudication was performed on these four items.

## C1: were the 4 excluded items in the human model fits?

**No.** The 944-item manifest (`eedi948_item_manifest.parquet`, produced by S1 item-mapping) excludes all 4 ambiguous-answer items *before* any empirical-correctness, EB, or IRT/Rasch fitting occurs (`s2_human_difficulty.py` reads the manifest's `question_id` list and filters `train_task_3_4.csv` to exactly those 944 ids before fitting). Independently re-verified here: the Rasch refit on the clean 944-item, all-responses universe used n_items=944, n_students=4918, n_observations=1377653, none of which include the 4 excluded QuestionIds. **No recomputation of the human model was necessary** -- the currently-frozen IRT/EB/empirical model already only ever saw the clean 944-item universe.

## C2: exact student-response sample (944 items, all observed responses)

- n_response_rows = 1377653
- n_unique_students = 4918
- n_unique_questions = 944
- attempts_per_question_min = 4
- Q1 = 321.0
- median = 1684.0
- Q3 = 2345.5
- P90 = 2648.1000000000004
- P95 = 2829.2499999999995
- max = 2966
- n_unique_student_question_pairs = 1377653
- n_repeated_student_question_pairs = 0
- maximum_repeats_for_one_student_question = 1
- total possible student-item pairs = 4918 × 944 = 4,642,592
- unobserved student-item pairs = 3,264,939
- duplicate observed student-item pairs = 0
- Rasch likelihood evaluated only over observed pairs; unobserved pairs are not imputed and do not enter the likelihood

## Response-source rule

Learner responses for the same-item alignment analysis come EXCLUSIVELY from `data/eedi/train_data/train_task_3_4.csv` (EeDi NeurIPS-2020 'Tasks 3 & 4' answer log), filtered to the 944 retained QuestionIds. Task 1/2 answers (`train_task_1_2.csv`) are NOT merged in for this analysis: Tasks 1/2 and Tasks 3/4 are two distinct EeDi question pools with only Tasks-3/4 content assets recovered and byte-verified against local machine-inference images (see `00_item_mapping.md`). Merging in Task 1/2 responses would not add same-item content coverage (no image assets were recovered/hashed for that pool in this extension) and would risk mixing unrelated QuestionId namespaces. This differs from the original (non-same-item) RQ1 EeDi analysis frozen in v1, which used the combined ~27,613-question Route-A pool from both tasks for a population-level (not same-item) difficulty description.

## Repeated-response sensitivity (all observed vs. first observed per student-question pair)

Item-difficulty rankings are essentially unchanged (Spearman all-vs-first: empirical=1.0000, EB=1.0000, IRT=1.0000, all n=944). The primary alignment rho itself moves from 0.1382 (all responses) to 0.1382 (first-response-only), a difference of 0.0000, which is NOT material (<=0.02) and does not change the qualitative (weak, positive, significant) conclusion.

Full sensitivity table: `outputs/same_item_final/repeated_response_sensitivity.csv`.

