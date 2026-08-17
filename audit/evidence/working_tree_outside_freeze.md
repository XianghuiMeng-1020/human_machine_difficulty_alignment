# Working tree outside the scientific freeze

After `fode-r1-science-freeze`, these classes remain local-only (gitignored or unrestored pre-existing edits):

- Encoder checkpoints / HF folders / EeDi public extract — see `gitignored_large_artifacts.md`
- Pre-existing plots/scripts under `race_analysis_with_datamap/`, `race_prepared/` (not restaged)
- Untracked revision manuscript and older revision artifacts
- Legacy `LLM_out/`

They are not required to regenerate the freeze headline numbers if the SHA-manifested inputs and committed tables are present.
