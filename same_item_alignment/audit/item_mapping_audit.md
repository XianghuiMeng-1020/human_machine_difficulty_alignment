# Item mapping audit (S1)

Raw answers file: `data\eedi\train_data\train_task_3_4.csv`
SHA-256: `8bdbe55a310641f9e59caffbff2eac85da0b2f6c6b2bb99fa69922296f52a4e1`
Source zip SHA-256: `c7f01672360f1adeb3cf9507d72455d7be035bf897e4a167293e8938049800e1`  (`data\eedi_public_download\data.zip`)

## Independently recomputed raw universe

- n_attempts (rows) = 1382727 (expected 1,382,727)
- n_unique_questions = 948 (expected 948)
- n_unique_students = 4918 (expected 4,918)

## Duplicate response rows (QuestionId,UserId,AnswerId): 0
Rows with AnswerValue outside {1,2,3,4}: 0
Rows with CorrectAnswer outside {1,2,3,4}: 0
Rows where IsCorrect != (AnswerValue==CorrectAnswer): 0

## Questions with >1 distinct CorrectAnswer across records: 4
  - QuestionId 43: CorrectAnswer value counts = {2: 230, 3: 2} (n=232)
  - QuestionId 84: CorrectAnswer value counts = {1: 2175, 4: 452} (n=2627)
  - QuestionId 206: CorrectAnswer value counts = {3: 1455, 1: 745} (n=2200)
  - QuestionId 860: CorrectAnswer value counts = {2: 10, 3: 5} (n=15)

Decision: these questions do NOT have a deterministic, unambiguous correct-answer label in the raw provenance data and are EXCLUDED from the same-item manifest rather than resolved by majority vote (which would be a fabricated resolution not present in the source of truth).

## Content assets found in `data\eedi_public_download\extracted\data\images`: 948
Questions with response records but NO matching image asset: 0 []
Image assets with NO matching response records: 0 []

question_metadata_task_3_4.csv rows: 948, unique QuestionId: 948
QuestionId set equality (answers vs question_metadata): True

## Retained universe after hard checks: 944 / 948
Excluded questions (ambiguous correct answer or missing image asset): [43, 84, 206, 860]

Hashing content assets (sha256) for retained items ...

Wrote manifest: `same_item_alignment/data/eedi948_item_manifest.parquet` (n=944)

## Deterministic spot check on 60 randomly sampled items: 60/60 PASS

## Gate S1 verdict
- Verified same-item denominator: **944 / 948** questions
- 100% deterministic linkage achieved on the retained 944-item subset (unresolved ambiguous-correct-answer items and any image-less items excluded, not fabricated).
- Status: **PASS** (PARTIAL relative to the 948 target because 4 items were excluded for a documented, non-fabricated reason; the retained subset itself is 100% linked)
