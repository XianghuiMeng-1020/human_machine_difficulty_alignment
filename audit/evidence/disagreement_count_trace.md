# Disagreement Count Trace: 'aligned_easy: 67 -> 103' (Part H)

## What each number actually is

Independently recomputing the ORIGINAL (v2 freeze, 3-solver-majority) human-quartile x machine-majority taxonomy from the frozen `same_item_integrated_948.parquet` table gives exactly the same four cell counts documented in `08_final_same_item_audit.md`:

- aligned_easy (human bottom quartile & machine-majority correct) = **103**
- aligned_hard (human top quartile & machine-majority incorrect) = **169**
- human_hard_machine_easy (human top quartile & machine-majority correct) = **67**
- human_easy_machine_hard (human bottom quartile & machine-majority incorrect) = **133**

## Resolution of the '67 -> 103' ambiguity

`103` is `aligned_easy` (103). `67` is `human_hard_machine_easy` (67). These are two DIFFERENT cells of the same 2x2 human-quartile x machine-majority taxonomy, not two versions of the same number. The prior chat-level summary that wrote 'aligned_easy: 67 -> 103' conflated the disagreement cell (`human_hard_machine_easy`=67, a genuinely different, scientifically interesting quantity) with the alignment cell (`aligned_easy`=103) while describing a single label -- almost certainly a copy/paste or mislabeling error in an intermediate summary, not evidence of a changed computation. Both numbers independently recompute correctly and consistently from the same frozen table in this run; there is only ONE correct current value for each of the six/four named cells, and they are reported separately by name (never as an arrow-separated pair) in `disagreement_3solver_sensitivity.csv` and `disagreement_primary_2solver.csv`. This ambiguous notation must not appear in the manuscript; use only the named-cell table format.

## Primary (2-solver) taxonomy for v2.1

| cell                     |   n |   pct_of_human_easy |   pct_of_human_hard |   pct_of_all_944 |
|:-------------------------|----:|--------------------:|--------------------:|-----------------:|
| human_easy_machine_easy  |  78 |             33.0508 |            nan      |          8.26271 |
| human_easy_machine_mixed |  76 |             32.2034 |            nan      |          8.05085 |
| human_easy_machine_hard  |  82 |             34.7458 |            nan      |          8.68644 |
| human_hard_machine_easy  |  45 |            nan      |             19.0678 |          4.76695 |
| human_hard_machine_mixed |  70 |            nan      |             29.661  |          7.41525 |
| human_hard_machine_hard  | 121 |            nan      |             51.2712 |         12.8178  |
