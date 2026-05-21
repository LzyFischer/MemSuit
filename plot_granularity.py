import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Data -----------------------------------------------------------------
# Methods: 1, 5, 10, MemSuit
# Categories: Cat1, Cat2, Cat3, Cat4
METHOD_ORDER = ["1", "5", "10", "MemSuit"]
METHOD_LABELS = {"1": "1", "5": "5", "10": "10", "MemSuit": "MemSuit"}

CATEGORY_ORDER = ["Muli-hop", "Temporal", "Open-domain", "Single-hop"]

data_f1 = {
    "1":       [10.81, 13.92, 10.32, 14.20],
    "5":       [20.07, 28.21, 16.41, 29.08],
    "10":      [19.10, 22.18, 10.82, 24.25],
    "MemSuit": [22.63, 28.66, 15.40, 33.48],
}

data_bleu = {
    "1":       [7.60,  9.99,  8.73,  12.30],
    "5":       [14.52, 18.88, 13.53, 25.44],
    "10":      [13.73, 14.97, 4.36,  21.43],
    "MemSuit": [16.36, 20.40, 13.15, 29.96],
}

# ---- Palette --------------------------------------------------------------
palette = [
    (255/255, 190/255, 122/255),  # peach
    (250/255, 127/255, 111/255),  # salmon
    (130/255, 176/255, 210/255),  # sky blue
    (142/255, 207/255, 201/255),  # teal
    (190/255, 184/255, 220/255),  # lavender
    (227/255, 207/255, 187/255),  # beige
]

METHOD_COLORS = {
    "1":       palette[2],  # sky blue
    "5":       palette[3],  # teal
    "10":      palette[0],  # peach
    "MemSuit": palette[1],  # salmon
}

# A different hatch for each method (last one is the highlight - no hatch)
METHOD_HATCH = {
    "1":       "",
    "5":       "",
    "10":      "",
    "MemSuit": "//",
}


def plot_subplot(ax, data, ylabel, ylim, yticks, show_legend=False):
    n_methods = len(METHOD_ORDER)
    bar_w = 0.78 / n_methods
    x = np.arange(len(CATEGORY_ORDER))

    for i, m in enumerate(METHOD_ORDER):
        means = data[m]
        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, means, bar_w,
            label=METHOD_LABELS[m],
            color=METHOD_COLORS[m],
            edgecolor="black", linewidth=1.6,
            hatch=METHOD_HATCH[m],
        )
        for rect, h in zip(bars, means):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + (ylim[1] * 0.015),
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=12, color="#333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER, rotation=0, ha="center", fontsize=18)

    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    ax.set_facecolor("#EEF0F2")
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)

    if show_legend:
        ax.legend(
            ncol=4, loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            frameon=False, fontsize=16,
            columnspacing=1.5, handletextpad=0.5,
        )


def main():
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

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)

    plot_subplot(ax_top, data_f1, ylabel="F1-Score",
                 ylim=(0, 42), yticks=[0, 10, 20, 30, 40],
                 show_legend=True)
    plot_subplot(ax_bot, data_bleu, ylabel="BLEU",
                 ylim=(0, 38), yticks=[0, 10, 20, 30],
                 show_legend=False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.18)

    out_png = Path("bar_plot_f1_bleu.png")
    out_pdf = Path("bar_plot_f1_bleu.pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
