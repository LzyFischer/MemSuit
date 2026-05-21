import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Data -----------------------------------------------------------------
METHODS = ["MemoryBank", "Amem", "Mem0", "SimpleMem", "MemSuit"]

token_per_conv = [13427.70, 41622.60, 42331.60, 32836.00, 36408.00]
f1_score       = [17.73,     19.27,     21.31,     18.68,    25.04]

# ---- Palette --------------------------------------------------------------
palette = [
    (255/255, 190/255, 122/255),  # peach
    (250/255, 127/255, 111/255),  # salmon
    (130/255, 176/255, 210/255),  # sky blue
    (142/255, 207/255, 201/255),  # teal
    (190/255, 184/255, 220/255),  # lavender
    (227/255, 207/255, 187/255),  # beige
]

F1_COLOR    = palette[1]  # salmon
TOKEN_COLOR = palette[2]  # sky blue

# ---- Styling --------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "axes.facecolor": "#EEF0F2",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "gray",
    "grid.alpha": 0.5,
    "font.size": 18,
})

fig, ax_f1 = plt.subplots(figsize=(8, 3))
ax_tok = ax_f1.twinx()

n_methods = len(METHODS)
bar_w = 0.36
x = np.arange(n_methods)

# F1 bars (left, slight left offset)
bars_f1 = ax_f1.bar(
    x - bar_w / 2, f1_score, bar_w,
    label="F1-Score",
    color=F1_COLOR,
    edgecolor="black", linewidth=1.2,
)

# Token bars (right, slight right offset)
bars_tok = ax_tok.bar(
    x + bar_w / 2, token_per_conv, bar_w,
    label="Token",
    color=TOKEN_COLOR,
    edgecolor="black", linewidth=1.2,
    hatch="//",
)

# ---- Value labels ---------------------------------------------------------
f1_ylim_top = 32
tok_ylim_top = 52000

# for rect, h in zip(bars_f1, f1_score):
#     ax_f1.text(
#         rect.get_x() + rect.get_width() / 2,
#         h + f1_ylim_top * 0.012,
#         f"{h:.2f}",
#         ha="center", va="bottom", fontsize=10, color="#333",
#     )

# for rect, h in zip(bars_tok, token_per_conv):
#     ax_tok.text(
#         rect.get_x() + rect.get_width() / 2,
#         h + tok_ylim_top * 0.012,
#         f"{h:,.0f}",
#         ha="center", va="bottom", fontsize=12, color="#333",
#     )

# ---- Axes -----------------------------------------------------------------
ax_f1.set_xticks(x)
ax_f1.set_xticklabels(METHODS, rotation=0, ha="center", fontsize=18)

ax_f1.set_ylabel("F1-Score", color="#333")
ax_f1.set_ylim(0, f1_ylim_top)
ax_f1.set_yticks([0, 10, 20, 30])

ax_tok.set_ylabel("Token", color="#333")
ax_tok.set_ylim(0, tok_ylim_top)
ax_tok.set_yticks([0, 10000, 20000, 30000, 40000, 50000])
ax_tok.set_yticklabels(["0", "10k", "20k", "30k", "40k", "50k"])

# ---- Cosmetics ------------------------------------------------------------
ax_f1.set_facecolor("#EEF0F2")
ax_f1.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
ax_f1.spines[["top"]].set_visible(False)
ax_tok.spines[["top"]].set_visible(False)
ax_tok.grid(False)

# ---- Legend ---------------------------------------------------------------
handles = [bars_f1, bars_tok]
labels = ["F1-Score", "Token"]
ax_f1.legend(
    handles, labels,
    ncol=2, loc="upper center",
    bbox_to_anchor=(0.5, 1.27),
    frameon=False, fontsize=18,
    columnspacing=2.0, handletextpad=0.6,
)

fig.tight_layout()

out_png = Path("results/bar_plot_token_f1.png")
out_pdf = Path("results/bar_plot_token_f1.pdf")
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
