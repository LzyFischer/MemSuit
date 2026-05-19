"""
Bar plot for the granularity ablation.

Reads the combined summary JSON produced by
`eval/run_granularity_ablation.py` and writes:

    figures/granularity_ablation.pdf / .png             (overall)
    figures/granularity_ablation_by_category.pdf / .png (with --per-category)

Usage
-----
    python plot_granularity_ablation.py
    python plot_granularity_ablation.py \
        --summary results/granularity_ablation/summary_combined.json \
        --out figures/granularity_ablation \
        --metric f1 \
        --per-category
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# ─── palette + labels (same look as plot_prelim1.py) ────────────────────────
PALETTE = [
    (255 / 255, 190 / 255, 122 / 255),  # peach
    (250 / 255, 127 / 255, 111 / 255),  # salmon
    (130 / 255, 176 / 255, 210 / 255),  # sky blue
    (142 / 255, 207 / 255, 201 / 255),  # teal
    (190 / 255, 184 / 255, 220 / 255),  # lavender
    (227 / 255, 207 / 255, 187 / 255),  # beige
]

# LoCoMo category labels — taken from plot_prelim1.py
CATEGORY_SHORT = {
    "overall": "Overall",
    "cat_1":   "Multi-hop",
    "cat_2":   "Temporal",
    "cat_3":   "Open-domain",
    "cat_4":   "Single-hop",
    "cat_5":   "Adversarial",
}
CATEGORY_ORDER = ["overall", "cat_1", "cat_2", "cat_3", "cat_4", "cat_5"]

METRIC_LABELS = {
    "f1":       "F1-Score",
    "bleu1":    "BLEU-1",
    "rougeL_f": "ROUGE-L",
    "bert_f1":  "BERTScore",
    "sbert":    "SBERT-Sim",
    "llm_judge": "LLM-Judge",
}


def _set_rc() -> None:
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


def _sorted_t_keys(summary: Dict) -> List[str]:
    return sorted(
        [k for k in summary if k.startswith("T_")],
        key=lambda k: int(k.split("_")[1]),
    )


# ─── overall (single-pane) plot ─────────────────────────────────────────────

def plot_overall(summary: Dict, out_path: Path, metric: str = "f1") -> None:
    keys        = _sorted_t_keys(summary)
    turn_counts = [int(k.split("_")[1]) for k in keys]

    def _get(k, field):
        block = summary[k].get("overall", {}).get(metric, {})
        return block.get(field, 0.0)

    means = [_get(k, "mean") * 100 for k in keys]
    stds  = [_get(k, "std")  * 100 for k in keys]
    ns    = [_get(k, "n")            for k in keys]

    _set_rc()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    x       = np.arange(len(turn_counts))
    bar_w   = 0.6
    colors  = [PALETTE[2], PALETTE[3], PALETTE[0], PALETTE[1],
               PALETTE[4], PALETTE[5]][: len(turn_counts)]
    hatches = ["", "//", "\\\\", "xx", "..", "++"][: len(turn_counts)]

    bars = ax.bar(
        x, means, bar_w,
        yerr=stds, capsize=4,
        color=colors,
        edgecolor="black", linewidth=1.4,
    )
    for bar, hp in zip(bars, hatches):
        bar.set_hatch(hp)
    for rect, h, s in zip(bars, means, stds):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + s + 1.5,
            f"{h:.1f}",
            ha="center", va="bottom", fontsize=10, color="#333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"T={t}\n(n={n})" for t, n in zip(turn_counts, ns)],
                       fontsize=11)
    ax.set_xlabel("Window size (turns)  ·  1 memory entry per window")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    max_top = max(m + s for m, s in zip(means, stds)) if means else 0
    ax.set_ylim(0, max(50, int(max_top + 12)))

    ax.set_facecolor("#EEF0F2")
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=600)
    print(f"saved: {out_path.with_suffix('.pdf')} and .png")


# ─── per-category grouped plot ──────────────────────────────────────────────

def plot_per_category(summary: Dict, out_path: Path, metric: str = "f1") -> None:
    keys        = _sorted_t_keys(summary)
    turn_counts = [int(k.split("_")[1]) for k in keys]

    available_cats: List[str] = []
    for c in CATEGORY_ORDER:
        if any(c in summary[k] and metric in summary[k][c] for k in keys):
            available_cats.append(c)

    def _val(k, c, field):
        block = summary[k].get(c, {}).get(metric, {})
        v = block.get(field, 0.0)
        return v * 100 if field in ("mean", "std") else v

    _set_rc()
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(available_cats)), 3.8))

    n_groups = len(turn_counts)
    x        = np.arange(len(available_cats))
    bar_w    = 0.78 / max(n_groups, 1)
    colors   = [PALETTE[2], PALETTE[3], PALETTE[0], PALETTE[1]][: n_groups]
    hatches  = ["", "//", "\\\\", "xx"][: n_groups]

    for i, (k, t) in enumerate(zip(keys, turn_counts)):
        means  = [_val(k, c, "mean") for c in available_cats]
        offset = (i - (n_groups - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, means, bar_w,
            label=f"T={t}",
            color=colors[i],
            edgecolor="black", linewidth=1.2,
            hatch=hatches[i],
        )
        for rect, h in zip(bars, means):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 1.2,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=8.5, color="#333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_SHORT.get(c, c) for c in available_cats],
                       fontsize=11)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    max_h = max(
        (_val(k, c, "mean") for k in keys for c in available_cats),
        default=0,
    )
    ax.set_ylim(0, max(50, int(max_h + 15)))

    ax.set_facecolor("#EEF0F2")
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=n_groups,
        fontsize=11,
        handlelength=1.5,
        columnspacing=2.0,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=600)
    print(f"saved: {out_path.with_suffix('.pdf')} and .png")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary",
                   default="results/granularity_ablation/summary_combined.json")
    p.add_argument("--out", default="figures/granularity_ablation")
    p.add_argument("--metric", default="f1",
                   choices=list(METRIC_LABELS.keys()),
                   help="Which metric column to plot")
    p.add_argument("--per-category", action="store_true",
                   help="Also produce a per-QA-category grouped bar plot")
    args = p.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"summary file not found: {summary_path}")
        print("Run eval/run_granularity_ablation.py first.")
        return 1
    payload = json.loads(summary_path.read_text())
    summary = payload.get("summary", payload)  # tolerate either shape

    plot_overall(summary, Path(args.out), metric=args.metric)
    if args.per_category:
        plot_per_category(summary, Path(args.out + "_by_category"),
                          metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
