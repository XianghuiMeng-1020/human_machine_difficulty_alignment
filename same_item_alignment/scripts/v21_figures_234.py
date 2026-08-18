#!/usr/bin/env python3
"""Part V: Figure 2 (disagreement), Figure 3 (RACE continuous dynamics),
Figure 4 (robustness) candidates + source CSVs. Staging only, no LaTeX integration."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIA = ROOT / "same_item_alignment"
FIG = ROOT / "outputs/revision_candidate_v21/figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- Figure 2: same-item disagreement (human difficulty x machine state) ----
df = pd.read_parquet(SIA / "data/same_item_integrated_948.parquet")
n_primary_correct = df.solver_1_correct.astype(int) + df.solver_2_correct.astype(int)
df["machine_state"] = n_primary_correct.map({2: "machine_easy (both correct)", 1: "machine_mixed (one correct)", 0: "machine_hard (both incorrect)"})
q = df["human_irt_difficulty"].quantile([0.25, 0.75])
df["human_group"] = np.select(
    [df.human_irt_difficulty <= q.loc[0.25], df.human_irt_difficulty >= q.loc[0.75]],
    ["human_easy (bottom quartile)", "human_hard (top quartile)"], default="human_middle (excluded from taxonomy)")

fig2_source = df[["question_id", "human_irt_difficulty", "human_group", "machine_state", "empirical_correctness"]]
fig2_source.to_csv(FIG / "eedi_disagreement_source.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
colors = {"machine_easy (both correct)": "#2ca02c", "machine_mixed (one correct)": "#ff7f0e", "machine_hard (both incorrect)": "#d62728"}
for state, g in df.groupby("machine_state"):
    ax.scatter(g.human_irt_difficulty, g.empirical_correctness, s=14, alpha=0.6, label=state, color=colors[state])
ax.axvline(q.loc[0.25], color="gray", linestyle=":", linewidth=1)
ax.axvline(q.loc[0.75], color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Human IRT difficulty")
ax.set_ylabel("Empirical student correctness")
ax.set_title("Same-item disagreement map (descriptive; no causal interpretation implied)\nDotted lines = human bottom/top IRT quartile cuts")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "eedi_disagreement_map.pdf")
fig.savefig(FIG / "eedi_disagreement_map.png", dpi=200)
plt.close(fig)

# ---- Figure 3: RACE continuous held-out dynamics by band ----
cont = pd.read_csv(ROOT / "outputs/diagnostics/continuous_dynamics_by_band.csv")
cont.to_csv(FIG / "race_continuous_dynamics_source.csv", index=False)
fig3, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].plot(cont.seed, cont.mean_prob_MIDDLE, "o-", label="MIDDLE", color="#1f77b4")
axes[0].plot(cont.seed, cont.mean_prob_HIGH, "o-", label="HIGH", color="#d62728")
axes[0].set_xlabel("Longformer seed"); axes[0].set_ylabel("Mean held-out gold probability")
axes[0].set_title("Continuous held-out confidence by grade band"); axes[0].legend()
axes[1].plot(cont.seed, cont.acc_MIDDLE, "o-", label="MIDDLE", color="#1f77b4")
axes[1].plot(cont.seed, cont.acc_HIGH, "o-", label="HIGH", color="#d62728")
axes[1].set_xlabel("Longformer seed"); axes[1].set_ylabel("Held-out accuracy")
axes[1].set_title("Held-out accuracy by grade band"); axes[1].legend()
fig3.suptitle("RACE continuous held-out dynamics (primary; not discrete Cartography regions)")
fig3.tight_layout()
fig3.savefig(FIG / "race_continuous_dynamics.pdf")
fig3.savefig(FIG / "race_continuous_dynamics.png", dpi=200)
plt.close(fig3)

# ---- Figure 4: robustness (seed / architecture / threshold) compact panel ----
thresh = pd.read_csv(ROOT / "outputs/race_final/threshold_sensitivity_full.csv")
thresh.to_csv(FIG / "race_robustness_source.csv", index=False)
seed_summary_v = pd.read_csv(ROOT / "outputs/encoder/seed_g6_g8.csv")
fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4.5))
axes4[0].bar(seed_summary_v.seed.astype(str), seed_summary_v.band_x_region_cramers_v, color="#2b6cb0")
axes4[0].set_xlabel("Longformer seed"); axes4[0].set_ylabel("Band x region Cramer's V")
axes4[0].set_title("Seed sensitivity (tercile rule)")
lf_only = thresh[thresh.architecture == "Longformer"]
piv = lf_only.groupby("spec")["band_x_region_cramers_v_recomputed"].mean()
axes4[1].bar(piv.index.astype(str), piv.values, color="#2ca02c")
axes4[1].set_xlabel("Threshold scheme"); axes4[1].set_ylabel("Mean band x region V across seeds")
axes4[1].set_title("Threshold sensitivity (Longformer)")
fig4.suptitle("Robustness: seed and threshold sensitivity of discrete region associations")
fig4.tight_layout()
fig4.savefig(FIG / "race_robustness_panel.pdf")
fig4.savefig(FIG / "race_robustness_panel.png", dpi=200)
plt.close(fig4)

print("Wrote figures 2/3/4 + source CSVs to", FIG)
