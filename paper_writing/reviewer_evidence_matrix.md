# Reviewer/Editor Comment -> Final Evidence Matrix (Part R, v2.1)

Base matrix is `audit/01_comment_matrix.md` (IDs: A1-A3 = Editor A, B1-B3 = Editor B, C1-C6 =
Editor C, R1.1-1.3 = Reviewer 1, R2.1-2.6 = Reviewer 2, I1-I5 = internal integrity checks).
**UPDATE (v2.1 write-time truth pass, R17):** "Editor D" is confirmed to be the decision letter's
`D. PRESENTATION` section (careful language editing; correction of grammatical problems; Figure 2
marginal counts; every table must state its denominator in the caption). It is now added as its
own row below and cross-referenced with the closely related R2.6 (which covers the same
presentation ground from the Reviewer-2 side). Below, each row
is updated to its FINAL v2.1 status; unchanged rows are carried forward from `01_comment_matrix.md`
with a note.

| ID | Comment | Final scientific action | Exact evidence | Exact artifact | Manuscript implication | Claim boundary | Status |
|---|---|---|---|---|---|---|---|
| A1 | Official RACE dev split counts | Independently re-verified (unchanged) | 4887/1436/3451 | `audit/evidence/race_split_counts.json` | Use exact counts in Methods | None (fully supported) | PASS |
| A2 | Round/fabricated counts; EeDi 27,613 denominator | 27,613 now confirmed supported (was PARTIAL) | `data/processed/eedi_verified.parquet` (27,613 rows), P0 closure headline | `audit/09_p0_closure_report.md` | Table 1 (EeDi buckets) numbers are correct as-is | Do not conflate 27,613 (full pool) with 944 (same-item subset) | PASS (upgraded from PARTIAL) |
| A3 | Integrated table + code; missing provenance columns | Still missing epoch/model/seed/provider/access_date columns in canonical `race_val_integrated.csv`; NOT fixed in this pass (out of scope for a same-item alignment consolidation) | `audit/evidence/race_raw_manifest.csv` | n/a | Provenance detail is now available in per-backend `*_run_meta.json` (Part O) even though not merged into one canonical table column-set | Report provenance via `race_llm_method_full.md`, not by claiming the canonical table itself has these columns | PARTIAL (carried forward) |
| B1 | Encoder confusion vs accuracy | Unchanged, PASS | confusion_sum=4887, acc=0.7409 | `audit/evidence/reconcile_integrated.json` | No change | None | PASS |
| B2 | Table 5/6 LLM denominators | Recomputed with corrected consensus (4873/14, not 4870/17) | `outputs/llm/consensus_metrics.csv` | `outputs/race_final/race_llm_exact_metrics.csv` | Manuscript Table 5/6 must use 4873/14, not the draft's 4870/17 | Use "last successful A-D parse" as canonical vote rule | PASS (numbers corrected) |
| B3 | Encoder valid trained reader; need >=3 seeds | RESOLVED — 3 Longformer seeds now exist | mean=0.7407, sd=0.0032, min=0.7373, max=0.7434 | `outputs/encoder/seed_summary.json` | Report seed mean+SD, not a single run | Structural tiny-overfit/gradient battery still not re-executed this pass | PASS (upgraded from PARTIAL; structural sub-item remains PARTIAL) |
| C1 | Held-out vs original Cartography construct | Unchanged; language already correct in draft | Held-out validation dynamics only | `outputs/diagnostics/continuous_dynamics_by_band.csv` | Continuous dynamics now PRIMARY per this v2.1 instruction (Part M3) | Never call this original-Cartography training dynamics | PASS |
| C2 | Grade-band accuracy inversion | Confirmed with multi-seed data: MIDDLE>HIGH holds for encoder AND LLM, across all 3 seeds | encoder MIDDLE .7862/.7834-.7904 vs HIGH .7181-.7238 across seeds | `outputs/diagnostics/g8_inversion.json`, `outputs/encoder/seed_g6_g8.csv` | Wording must say "exam-source band", never "harder" | Do not causally interpret MIDDLE>HIGH as MIDDLE being intrinsically easier text | PASS |
| C3 | Consensus filter + retries; temperature disclosure | Draft's Methods text (temp 0.1-1.5 retry) CONTRADICTS the frozen protocol (temp fixed, never varied on retry) — must be corrected | `configs/llm_protocol.yaml` | `audit/evidence/race_llm_method_full.md` | Methods must say retries reuse IDENTICAL decoding params | Do not describe a temperature-search retry procedure | PARTIAL -> flagged for mandatory manuscript text fix (Part Q) |
| C4 | EeDi reliability (27,613 denominator) | RESOLVED (see A2) | 27,613 confirmed | `outputs/eedi/eedi_attempt_distribution.csv` | No change to Table 1 | Same-item (944) and full-pool (27,613) EeDi analyses must be kept visually/textually distinct | PASS (upgraded from FAIL) |
| C5 | RQ1 role (Route A vs B) | Locked: Route B for RACE (cross-corpus reference); NEW same-item Route added for the EeDi-VLM extension (this is a THIRD route, distinct from A/B) | `audit/evidence/rq1_role_decision.md` | Same-item alignment is same-item WITHIN EeDi (student vs VLM), not EeDi-vs-RACE | RQ1 role decision text needs one additional sentence introducing the new same-item route | Do not claim the same-item EeDi-VLM study resolves the EeDi-vs-RACE cross-corpus question — it does not; it is a separate, self-contained study | PARTIAL -> ACTION NEEDED (add sentence, no data change) |
| C6 | Blinded content-validity audit | **NARROWED per this task**: E6 is NOT USABLE; the content-validity recommendation must be scoped down to "future work: repeat a blinded content-validity audit under documented IRB/ethics review" | `audit/evidence/human_ethics_provenance.md`, `forbidden_human_data_scan.md` | n/a (no substitute audit run) | Do not cite E6's kappa=0.538 or 66.7%/30.0% flaw rates anywhere in the manuscript | Content-validity claim is now scope-limited to future work only | **DOWNGRADED from PASS to NOT USABLE** per this consolidation |
| R1.1 | No same-item human/model comparison | **THIS IS NOW RESOLVED** — but by the NEW same-item EeDi-VLM study (Study A), NOT by Bridge-RACE (which is forbidden) | n=944, rho=0.1382 (primary), rho=0.1100 (3-solver robustness) | `outputs/same_item_final/primary_alignment.csv` | This becomes a headline manuscript result | Alignment is weak/partial, not strong; scoped to the 944-item content-available EeDi subset only, not transportable to all 27,613 EeDi items or to RACE | PASS — but via completely different evidence than the draft currently uses |
| R1.2 | Review improves learning (controlled outcome) | **Marked "claim narrowed"** per this task, NOT resolved | n/a — no RCT/deployment evidence exists anywhere in the repo | n/a | Manuscript must not claim disagreement-triggered review improves learning outcomes | Offline rank-correlation/association findings only; no causal learning-outcome claim | claim narrowed (not PASS, not FAIL — explicit scope limitation) |
| R1.3 | MCQ-only scope | Unchanged | Intentional, disclosed scope | `paper_writing/FODE_FINAL_WRITING_PACKET.md` S11 | State explicitly as a scope limitation, not an oversight | Do not imply generalization to open-ended/constructed-response items | Mark as scope limitation / future work (not fabricated as resolved) |
| R2.1 | Overstated alignment framing | Reframed: same-item EeDi result is WEAK (rho~0.11-0.14) and must be described as partial alignment, consistent with "Human-Machine Alignment" title only under a "partial, not complete" reading | `outputs/same_item_final/primary_alignment.csv`, `human_difficulty_deciles.csv` | Abstract/Discussion wording constraint | Never use "strong" or "substantial" for rho in the 0.10-0.15 range | PASS (framing corrected, grounded in new evidence, not Bridge) |
| R2.2 | Formal statistical validation | RESOLVED with proper item-level inference (replacing 3-cluster GEE) | primary binomial regression OR=1.296, item-bootstrap CI [1.165,1.450], p=8.6e-8 | `outputs/same_item_final/primary_binomial_regression.csv` | Table C (alignment stats) must use this, not the old GEE OR=1.2523 | Old GEE result retained only as audit history | PASS |
| R2.3 | LLM identity/provider/date | RESOLVED for RACE LLM backends (exact model IDs now documented); EeDi VLM identities also now fully documented | exact_model_id per backend | `audit/evidence/race_llm_method_full.md`, `audit/evidence/vlm_inference_method_full.md` | Methods must cite exact model IDs (e.g. `doubao-seed-2-0-pro-260215`), never generic "Doubao" | None | PASS (upgraded from PARTIAL) |
| R2.4 | Tercile sensitivity | Extended to same-item EeDi quintile/decile gradient + RACE threshold sensitivity (20/80,25/75,33/67) | `outputs/same_item_final/human_difficulty_deciles.csv`, `outputs/race_final/threshold_sensitivity_full.csv` | New tables for both studies | None | PASS |
| R2.5 | No-consensus analysis | RESOLVED with corrected 4873/14 (not 4870/17) | `outputs/race_final/race_no_consensus_exact.csv` | by-band: HIGH=13, MIDDLE=1 | Table must use corrected counts | None | PASS |
| R2.6 | Figure/language/denominators (presentation) | Merged with **Editor D** below (same presentation requirement, raised independently by both the editor and Reviewer 2); addressed only at manuscript-writing time | n/a — see Editor D row | n/a — see Editor D row | Deferred to writing | Deferred | READY FOR WRITING (see Editor D row) |
| Editor D | `D. PRESENTATION`: (1) careful language/copy editing; (2) fix grammatical problems throughout; (3) Figure 2 must show marginal counts; (4) every table must state its exact denominator in the caption | Scientific action = none (this is a presentation/formatting fix applied at manuscript-writing time, not a new analysis); this reconciliation task's job is only to make every artifact denominator-traceable so the writer CAN satisfy (4) | Every v2.1 output file/table now carries an explicit `n=`/denominator in its own filename, header row, or `.provenance.json` (`outputs/revision_candidate_v21/tables/*.provenance.json`, `denominator` field); Figure 2 candidate source (`eedi_disagreement_source.csv`) includes full per-cell n so marginal counts can be added when the figure is finalized | At manuscript-writing time: (a) run a full copy-edit/grammar pass; (b) add explicit marginal-count annotations to Figure 2; (c) add a one-line denominator statement to every table caption (e.g. "n=944 same-item questions", "n=4,887 RACE dev items") | Presentation only; does not change any reported number | READY FOR WRITING (mechanical presentation task; all underlying denominators are now documented) |

## Special note: Human–Machine Alignment comment (R1.1 / R2.1)
Per instruction, this comment is answered with **genuine same-item evidence**: 944 identical
EeDi content-available questions, evaluated by BOTH real students (IRT/EB/empirical difficulty)
and real VLM solvers (Qwen2-VL-7B + InternVL3-8B, primary; + SmolVLM2-2.2B, robustness/negative
control). This is not a softened framing of the old Bridge-RACE claim — it is an entirely
different, new dataset and analysis that happens to support the same title. The evidence is weak
(rho ~0.11-0.14) but real, significant, and reproducible from frozen predictions.

## Special note: Controlled-learning-outcome comment (R1.2)
Explicitly marked **"claim narrowed"**, not "resolved." No RCT, A/B test, or deployment log
exists in this repository for either the RACE disagreement-review policy or the EeDi same-item
alignment finding. Any language suggesting learning-outcome improvement must be removed from the
manuscript; the correct scope is "offline association/prioritization signal, not a tested
intervention."

## Special note: Open-ended question comment (R1.3)
Marked as **scope limitation / future work**. No new experiment was run or should be fabricated
to address open-ended/constructed-response items; the MCQ-only scope is disclosed as an
intentional design decision in Limitations.

## Special note: Human content-validity audit comment (C6)
**Narrowed, not resolved.** E6 cannot be used (NOT USABLE, Part P). The manuscript's
content-validity recommendation is downgraded to a Future Work item: "a properly consented,
IRB-documented blinded content-validity audit of high-disagreement items remains a natural next
step; this study does not include one."
