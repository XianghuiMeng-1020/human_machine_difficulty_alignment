# Original Manuscript -> Revision Change Map (Part Q, v2.1)

Source read (NOT edited): `_revision_materials/sn-article.tex` (587 lines read in full; this
appears to be the most recent submitted/near-submission draft in the repo — no earlier `.tex`
found). Title is LOCKED and unchanged: "Who Finds It Hard? Mapping Human–Machine Alignment in
Question Difficulty".

## Top-level finding
This draft's ENTIRE "Human–Machine Alignment" evidence base is Bridge-RACE + E6
(Section `sec:human-validation`, lines 604-689, and the abstract). **Both are now NOT USABLE**
(Part P). This is the single largest structural problem: the title's core claim currently rests
on forbidden data. The new same-item EeDi VLM analysis (Study A, this v2.1 packet) is the
REPLACEMENT evidence base for "Human–Machine Alignment" and does not appear anywhere in this
draft at all — it must be newly written in, not merely edited.

## Section-by-section map

| Location | Original claim/result | Decision | New source-of-truth | New exact number | Reason |
|---|---|---|---|---|---|
| Abstract (l.127) | "Human difficulty buckets align more strongly with encoder regions than with designer grade band (κ=0.264 vs 0.088)... 66.7% vs 30.0%" | **REPLACE** | `outputs/same_item_final/primary_alignment.csv` | rho=0.1382 (2-solver), n=944 | Bridge/E6 forbidden; abstract must lead with the NEW same-item EeDi machine-error vs human-IRT alignment as the primary Human-Machine Alignment evidence |
| Sec 1 Intro, RQ1-RQ3 (l.162-168) | RQ1 (EeDi learner baseline), RQ2 (designer vs encoder region), RQ3 (LLM vs designer/encoder) | **KEEP** (RACE RQ2/RQ3 unaffected by same-item work) but **ADD** a new RQ for same-item EeDi Human-Machine alignment (see Part S2 of writing packet) | n/a | n/a | RACE-side RQs are frozen v1 and must not change; a new RQ is additive |
| Sec 2.1 Datasets (l.238-242) | EeDi used only for RQ1 (cross-corpus, not joined to RACE) | **REWRITE** | `same_item_alignment/audit/00_item_mapping.md` | 944/948 | Must add: EeDi is ALSO now used for a same-item VLM alignment study (separate from the RACE branch); this is a new use of EeDi that this section does not describe |
| Sec 2.3.4 Bridge-RACE/E6 methods (l.324-329) | Bridge-RACE 320x30=9600 responses; E6 30+30 content audit | **DELETE** from primary Methods; may remain ONLY as a disclosed "not used" footnote if the journal requires disclosure of discontinued protocols | `audit/evidence/forbidden_human_data_scan.md` | n/a | NOT USABLE (ethics/provenance) |
| Sec 2.3.5 IRT/DIF/Paper-1 (l.331-342) | Bridge/machine-panel IRT, DIF, Paper-1 LLM-vs-human item generation comparison | **DELETE** | n/a | n/a | Entirely Bridge-dependent; NOT USABLE |
| Sec 3.1 RQ1 EeDi buckets (Table 1, l.369-382) | EeDi n=27,613; Easy 6,574/Mid 19,460/Hard 1,579 | **KEEP** | `data/processed/eedi_verified.parquet`, `audit/09_p0_closure_report.md` | 27613/6574/19460/1579 (confirmed, matches P0 closure "Final verified headline quantities" exactly) | This number IS supported in the current repo (contrary to a stale PARTIAL flag in an older `02_data_audit.md`); no change needed |
| Sec 3.2 RQ2 region table (Table 2/3, l.395-473) | Region counts Easy 1189/Ambig 1375/Hard 1148/Middle 1175; designer x region crosstab | **REWRITE (numbers only)** | `outputs/race_final/region_counts.csv`, `band_region_crosstab.csv` | region_majority counts: easy=1180, hard=841, middle=1268, ambiguous=1598 (current frozen `region_majority`, 3-seed majority rule) — **does not match** the single-run numbers in the draft | The draft's region table used an older single/legacy Longformer run; must be regenerated from the current 3-seed majority-vote canonical table |
| Sec 3.2 kappa table (Table `agreement_kappa`, l.523-537) | Designer-HIGH vs hard∪ambiguous κ=0.102; LLM-incorrect vs hard κ=0.158 | **REWRITE (verify numbers)** | `outputs/diagnostics/g6_stats.json`, `g8_inversion.json` | band×region Cramer's V=0.164 (chi2=131.78,p=2.2e-28); llm-incorrect×region V=0.244 (chi2=290.96) [recomputed here] vs V=0.241 in frozen `g6_llm_incorrect_x_region.csv` | Small (~3%) discrepancy between two independently-computed llm-incorrect x region statistics flagged as unresolved (see Part X.19) |
| Sec 3.3 RQ3 (l.544-568) | Consensus 4,870/4,887; MIDDLE 78.52%/HIGH 72.41% Longformer; MIDDLE 96.86%/HIGH 94.85% LLM | **REWRITE (numbers)** | `outputs/llm/consensus_metrics.csv`, `outputs/diagnostics/g8_inversion.json` | consensus 4873/4887 (not 4870); encoder acc MIDDLE=0.7862/HIGH=0.7218 (multi-seed mean, not single-run 78.52/72.41); LLM acc by band not yet separately regenerated in this pass — **flag for follow-up** | Single-seed drift; must regenerate from the 3-seed canonical table used in this v2.1 packet |
| Sec 3.4 Bridge/E6 results (Table `irt_family`, `irt_dif`, `paper1_irt`, `offline_policy`, l.604-689) | All Bridge/E6/Paper-1 results | **DELETE entirely** | `outputs/same_item_final/*` (whole new subsection) | n=944, rho=0.1382 primary | Replace this whole subsection with the new same-item EeDi Human-Machine Alignment Results (Parts E-J of this packet) |
| Sec 4 Discussion (l.694) | "Those human studies support keeping the provenance claim... human buckets track encoder regions more closely than designer grade band" | **REWRITE** | same-item alignment results | rho=0.1382 (weak, significant) | Discussion must be reframed around the weak-but-significant same-item EeDi finding, not Bridge κ comparisons |
| Sec 5 Limitations (l.700-707) | MCQ-only scope; encoder/LLM protocol sensitivity; panel is simulation not classroom sample; no controlled deployment evidence | **KEEP + ADD** | `paper_writing/FODE_FINAL_WRITING_PACKET.md` S11 | n/a | Existing limitations remain valid; ADD: 944-item non-representativeness, SmolVLM2 negative control, no Bridge/E6 evidence, discrete-region seed-sensitivity |
| Statements: Data availability (l.730-737) | "the revision package releases... Bridge-RACE item list and response file... E6 blind items and ratings... machine/LLM-simulated student panel... Paper-1 LLM-generated item bank" | **REWRITE** | n/a | n/a | Must remove all Bridge/E6/Paper-1/simulated-panel data-availability claims; add same-item EeDi VLM prediction files instead |
| Figures: `difficulty.pdf` (study-setup flowchart, l.275-280) | 7-step pipeline including Bridge-RACE (step 5) and blind audit (step 7) | **REWRITE** | n/a | n/a | Flowchart must drop Bridge/E6 steps and add the same-item EeDi VLM branch |
| Figure `Datamap means vs. variability.jpg` (l.593-598) | Legacy scatter, not regenerated from audit script (flagged PARTIAL in `01_comment_matrix.md` I1) | **REGENERATE** | `outputs/encoder/seed_item_regions.csv` | n=4887 check required | Prior audit flagged this figure was not regenerated from the canonical script; must confirm point count == 4887 before reuse |

## Explicitly-checked "flagged obsolete" tokens
Searched literally in `sn-article.tex` for: `5000, 3360, 1640, 1800, 1300, 750, 1150, 850, 1200,
2050, 62.3, 75.1, 78.5, 88.2`. **None of these tokens appear** in this draft. This strongly
suggests either (a) this draft postdates whatever earlier draft contained those numbers, or (b)
those numbers belong to a chat-level summary of an even earlier version not present in this repo.
**Do not assume they need fixing in THIS file** — but do not assume the manuscript is otherwise
current either (see Bridge/E6 and region-count issues above, which ARE real and present).

## Other flags found
- "official RACE validation split" is used correctly and consistently (n=4,887) — matches current
  frozen split counts. No fix needed.
- "four sources on the same questions" — the draft's framing is four difficulty RECORDS on RACE
  (designer/encoder/LLM) + EeDi (separate, not same-item). This is consistent with the OLD
  scope. The NEW scope adds a genuinely same-item human+machine record on EeDi (Study A) that
  this draft does not have language for at all.
- "designer difficulty" terminology is used consistently; no inconsistency found.
- "Dataset Cartography training dynamics" — draft already correctly hedges this as "held-out
  generalization dynamics... rather than train-set learning dynamics" (l.294). This matches the
  current recommended framing (Part M3); no change needed here.
- "temperature 0.1-1.5... retry" (l.319) — this describes the RACE LLM retry protocol. The
  CURRENT frozen protocol (`configs/llm_protocol.yaml`) explicitly says retries must reuse
  IDENTICAL temperature/top_p/max_tokens and must NOT vary temperature 0.1-1.5 to recover a
  parseable answer. **This is a real, material inconsistency**: the draft's Methods text
  describes a temperature-varying retry protocol that contradicts the currently frozen,
  audited protocol. Must be corrected to match `llm_protocol.yaml` exactly.
- No `???` bibliography placeholders found in this file.
- No repeated-introduction-claim duplication or obvious grammar breakage was found in a
  full-file read; this is a relatively polished draft, and its main problems are EVIDENTIARY
  (Bridge/E6) and NUMERIC-STALENESS (single-seed RACE numbers), not prose quality.

## Net verdict for Part Q
This draft cannot be revised by small edits alone. Section 2.3.4/2.3.5, all of Section 3.4, the
abstract's Human-Machine-Alignment sentence, the Data Availability statement, and the study-setup
figure all require **structural replacement**, not line edits, because they are built on now-
forbidden data. RQ2/RQ3 RACE content is directionally intact but every reported number needs
regeneration from the current frozen 3-seed canonical tables before reuse (do not silently
copy-paste the current draft's Table 2/3/5 values).
