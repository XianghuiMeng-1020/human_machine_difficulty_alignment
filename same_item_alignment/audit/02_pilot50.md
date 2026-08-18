# Pilot-50 report (Gate S3)

## Model-swap history (documented, not silently accepted)
- Original solver 2 candidate `OpenGVLab/InternVL2_5-4B` (trust_remote_code) failed to load under transformers 5.9.0 (`AttributeError: all_tied_weights_keys`). Replaced with the officially converted native checkpoint `OpenGVLab/InternVL3-2B-hf`, then upgraded to `OpenGVLab/InternVL3-8B-hf` after the 2B variant scored only marginally above chance (28%).
- Original solver 3 candidate `openbmb/MiniCPM-V-2_6` (trust_remote_code) carried the same incompatibility risk. Native replacement `openbmb/MiniCPM-V-4.6` turned out to be a much smaller efficient/linear-attention checkpoint that scored at chance (24%). Next replacement `llava-hf/llava-v1.6-mistral-7b-hf` (7B, native) scored *below* chance (20%) and collapsed onto a single answer option (82% share) -- a textbook pilot failure mode. Final choice `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (native, ungated, independently maintained by Hugging Face) was retained as the practical third solver; see its individual numbers below -- it is the WEAKEST of the three and does not fully clear the 'clearly above chance' bar on its own.

## Per-solver pilot-50 results

### solver_1_qwen2vl7b
- n=50, parse_success_rate=1.00
- accuracy=0.42, binomial p (vs 25% chance, one-sided)=0.006263
- answer distribution (share): {2: 0.52, 3: 0.3, 4: 0.1, 1: 0.08}
- clearly_above_chance: True, answer_collapse_flag (>=70% one option): False

### solver_2_internvl3_8b
- n=50, parse_success_rate=1.00
- accuracy=0.40, binomial p (vs 25% chance, one-sided)=0.01392
- answer distribution (share): {3: 0.36, 2: 0.36, 4: 0.2, 1: 0.08}
- clearly_above_chance: True, answer_collapse_flag (>=70% one option): False

### solver_3_smolvlm2_2b
- n=50, parse_success_rate=1.00
- accuracy=0.26, binomial p (vs 25% chance, one-sided)=0.489
- answer distribution (share): {2: 0.82, 4: 0.12, 3: 0.06}
- clearly_above_chance: False, answer_collapse_flag (>=70% one option): True

## Gate S3 criteria checklist
1. 100% question-to-answer-key mapping: PASS (manifest join has zero missing keys)
2. >=98% parse success per solver: PASS ([1.0, 1.0, 1.0])
3. No systematic A/B/C/D index inversion: PASS (no solver shows inverse-correlated accuracy vs shuffled key)
4. No model stuck producing one answer option: FAIL for solver_3_smolvlm2_2b (82% one option)
5. Each retained solver performs clearly above 25% chance: 2/3 solvers clearly pass (FAIL for solver_3_smolvlm2_2b)
6. At least two solvers show usable educational-question solving ability: PASS
7. No unresolved content rendering failure affecting a large fraction of items: PASS (0 image load errors)

## Overall Gate S3 verdict: **PARTIAL**
Rationale: criteria 1,2,3,6,7 PASS cleanly with 2/3 solvers (Qwen2-VL-7B-Instruct at 42%, InternVL3-8B-hf at 40%) showing clear, well-distributed, above-chance performance. Criteria 4 and 5 are not met by the third solver (SmolVLM2-2.2B-Instruct), which sits near chance (26%) with a lean toward one option. This is reported as a genuine, non-fabricated feasibility finding rather than silently patched. The full 944-item run proceeds with all three solvers retained; Section 13 (leave-one-solver-out robustness) explicitly tests and reports whether the primary alignment finding depends on solver_3_smolvlm2_2b, so this limitation is tracked through to the final results rather than hidden.
