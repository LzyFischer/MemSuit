import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Data ---------------------------------------------------------------
data = {
    "vanilla": {
        "overall": {"mean": 0.19133026786839167, "std": 0.1991857101451128, "n": 996},
        "cat_1":   {"mean": 0.1988616024557887,  "std": 0.1597207638720661, "n": 185},
        "cat_2":   {"mean": 0.18903701172554494, "std": 0.11428755896758218, "n": 212},
        "cat_3":   {"mean": 0.09637340264596428, "std": 0.15165053084339622, "n": 59},
        "cat_4":   {"mean": 0.27997346870494403, "std": 0.22080530874656965, "n": 540},
    },
    "utility": {
        "overall": {"mean": 0.3737376684097653,  "std": 0.2513278524497521,  "n": 996},
        "cat_1":   {"mean": 0.37419108263074535, "std": 0.2536035929575618,  "n": 185},
        "cat_2":   {"mean": 0.4044847405175016,  "std": 0.24653934328161597, "n": 212},
        "cat_3":   {"mean": 0.2840004916698716,  "std": 0.3085361326304582,  "n": 59},
        "cat_4":   {"mean": 0.43328921009482513, "std": 0.24060665969585435, "n": 540},
    },
}

# ---- Config -------------------------------------------------------------
palette = [
    (255/255, 190/255, 122/255),  # peach
    (250/255, 127/255, 111/255),  # salmon
    (130/255, 176/255, 210/255),  # sky blue
    (142/255, 207/255, 201/255),  # teal
    (190/255, 184/255, 220/255),  # lavender
    (227/255, 207/255, 187/255),  # beige
]

CATEGORY_ORDER = ["overall", "cat_1", "cat_2", "cat_3", "cat_4"]
CATEGORY_SHORT = {
    "overall": "Overall",
    "cat_1": "Multi-hop",
    "cat_2": "Temporal",
    "cat_3": "Open-domain",
    "cat_4": "Single-hop",
}

METHOD_ORDER = ["vanilla", "utility"]
METHOD_LABELS = {"vanilla": "Vanilla", "utility": "Utility"}
METHOD_COLORS = {"vanilla": palette[2], "utility": palette[1]}  # sky blue & salmon


def plot_bars(data, out_path: Path, show_n: bool = True):
    # Convert to percentages and gather stds / ns
    means_by_method = {
        m: [data[m][c]["mean"] * 100 for c in CATEGORY_ORDER] for m in METHOD_ORDER
    }
    stds_by_method = {
        m: [data[m][c]["std"] * 100 for c in CATEGORY_ORDER] for m in METHOD_ORDER
    }
    ns_by_category = {c: data[METHOD_ORDER[0]][c]["n"] for c in CATEGORY_ORDER}

    # ---- Styling --------------------------------------------------------
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
    })
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(10/2, 4.4/2))

    n_methods = len(METHOD_ORDER)
    bar_w = 0.78 / n_methods
    x = np.arange(len(CATEGORY_ORDER))

    for i, m in enumerate(METHOD_ORDER):
        means = means_by_method[m]
        stds = stds_by_method[m]
        offset = (i - (n_methods - 1) / 2) * bar_w
        if i == 0:
            bars = ax.bar(
                x + offset, means, bar_w,
                label=METHOD_LABELS[m],
                color=METHOD_COLORS[m],
                edgecolor="black", linewidth=1.4,
            )
        else:
            bars = ax.bar(
                x + offset, means, bar_w,
                label=METHOD_LABELS[m],
                color=METHOD_COLORS[m],
                edgecolor="black", linewidth=1.4,
                hatch='//'
            )
        for rect, h in zip(bars, means):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 1.5,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=10, color="#333",
            )

    # x-tick labels (with optional "(n=...)")
    xtick_labels = []
    for c in CATEGORY_ORDER:
        if show_n and c in ns_by_category:
            xtick_labels.append(f"{CATEGORY_SHORT[c]}")
        else:
            xtick_labels.append(CATEGORY_SHORT[c])

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=11)

    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60])

    # —— cosmetics ——
    ax.set_facecolor("#EEF0F2")
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # —— Legend: horizontal row above the plot ——
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=n_methods,
        fontsize=11,
        handlelength=1.5,
        columnspacing=2.0,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=600)
    print(f"saved: {out_path.with_suffix('.pdf')} and .png")
    plt.show()


if __name__ == "__main__":
    plot_bars(data, Path("./figures/vanilla_vs_utility"), show_n=True)