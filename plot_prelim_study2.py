"""
Plot Preliminary Study 2 results.
Style: matches vanilla_vs_utility teaser figure (flat, serif, hatch, legend on top).

Layout: 2 groups (Temporal / Open-domain), each with 4 bars:
  Vanilla (no demo) | Demo: temporal | Demo: open | Demo: mixed
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── data (F1 × 100) ───────────────────────────────────────────────────────────
CONDITIONS   = ["Vanilla", "Temporal", "Open-domain", "Mixed"]
TEMPORAL_F1  = [18.5, 26.6, 19.9, 23.6]
OPEN_F1      = [11.0, 12.7, 16.7, 16.4]

GROUPS = [
    ("Temporal",    TEMPORAL_F1),
    ("Open-domain", OPEN_F1),
]

# ── colours: match teaser palette ────────────────────────────────────────────
# same 4 colours used across conditions, consistent with teaser style
COLORS = [
    (130/255, 176/255, 210/255),   # sky blue   — Vanilla
    (142/255, 207/255, 201/255),   # teal       — Demo: temporal
    (250/255, 127/255, 111/255),   # salmon     — Demo: open
    (255/255, 190/255, 122/255),   # peach      — Demo: mixed
]
HATCHES = ["", "//", "//", "//"]   # vanilla solid, demos hatched

# ── style (identical to teaser) ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "axes.facecolor":   "#EEF0F2",
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.color":       "gray",
    "grid.alpha":       0.5,
    "font.size":        12,
})

fig, ax = plt.subplots(figsize=(10/2, 4.4/2))

# ── bar layout: 2 groups × 4 bars, gap between groups ────────────────────────
n_cond    = len(CONDITIONS)
bar_w     = 0.78 / n_cond          # same density as teaser (0.78 / n_methods)
group_gap = 0.2                   # extra space between the two groups

pos_0 = np.arange(n_cond) * bar_w
pos_1 = pos_0 + n_cond * bar_w + group_gap

for group_idx, (group_label, f1_vals) in enumerate(GROUPS):
    pos = pos_0 if group_idx == 0 else pos_1
    for i, (xc, val) in enumerate(zip(pos, f1_vals)):
        ax.bar(xc, val, bar_w,
               color=COLORS[i],
               hatch=HATCHES[i],
               edgecolor="black", linewidth=1.4,
               label=CONDITIONS[i].replace("\n", " ") if group_idx == 0 else None)
        ax.text(xc, val + 1.0, f"{val:.1f}",
                ha="center", va="bottom", fontsize=9.5, color="#333")

# ── group x-tick labels ───────────────────────────────────────────────────────
group_centres = [pos_0.mean(), pos_1.mean()]
ax.set_xticks(group_centres)
ax.set_xticklabels(["Temporal", "Open-domain"], fontsize=12)
ax.set_xlim(pos_0[0] - bar_w * 0.8, pos_1[-1] + bar_w * 0.8)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_ylabel("F1-Score")
ax.set_ylim(0, 44)
ax.set_yticks([0, 10, 20, 30, 40])
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor("#EEF0F2")
ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)

# ── legend: horizontal row above plot (same as teaser) ───────────────────────
ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=n_cond,
    fontsize=10,
    handlelength=1.5,
    columnspacing=1.2,
)

plt.tight_layout()
out = Path("results/prelim_study2/prelim_study2_bar")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=600)
print(f"Saved: {out}.pdf / .png")