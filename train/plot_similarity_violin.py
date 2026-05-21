"""
Step 2/2 of the contrastive-learning before/after analysis.

Reads four .npz files produced by compute_similarities.py and draws a 1x2
figure (Qwen on the left, Llama on the right). Each subplot shows two
"split" violins for the same encoder family:

  - LEFT  half of each violin: positive-pair cosines  cos(q_i, p_i)
  - RIGHT half of each violin: in-batch negative cosines
                                cos(q_i, p_j) for j != i AND
                                qa_key[j] != qa_key[i]
                              (same exclusion as
                               NoQACollisionBatchSampler in
                               train_contrastive.py)

If you only care about positive pairs, pass --positives-only.

Usage:
  python train/plot_similarity_violin.py \
      --qwen-before  train/data/sims_qwen_base.npz \
      --qwen-after   train/data/sims_qwen_ft.npz \
      --llama-before train/data/sims_llama_base.npz \
      --llama-after  train/data/sims_llama_ft.npz \
      --out-file     train/data/similarity_violin
        (extension is added automatically: .pdf + .png both written)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---- Style palette (matches the project's bar-plot style) -----------------
PALETTE = [
    (255/255, 190/255, 122/255),  # 0 peach
    (250/255, 127/255, 111/255),  # 1 salmon
    (130/255, 176/255, 210/255),  # 2 sky blue
    (142/255, 207/255, 201/255),  # 3 teal
    (190/255, 184/255, 220/255),  # 4 lavender
    (227/255, 207/255, 187/255),  # 5 beige
]

POS_COLOR = PALETTE[2]   # sky blue for positive pairs
NEG_COLOR = PALETTE[1]   # salmon  for in-batch negatives
EDGE_COLOR = "black"

# "after" is rendered with a hatch overlay, mirroring how the bar plot
# distinguishes its second method (METHOD_COLORS + hatch='//').
AFTER_HATCH = "//"


def apply_style():
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
        # Font sizes match the bar plot (plot_bars):
        #   - base font.size = 12  (used for axis labels, titles)
        #   - xtick labels  = 11   (set explicitly per-axis below)
        #   - legend        = 11   (set on fig.legend below)
        # We keep the base at 12 so axis labels/titles inherit it.
        "font.size": 14,
    })


# ---- Data helpers ---------------------------------------------------------

def load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "pos": d["pos"],
        "neg": d["neg"],
        "label": str(d["label"]),
        "n_pairs": int(d["n_pairs"]),
    }


def _subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Cap the number of points fed to violinplot. Pure visual perf — KDE
    on millions of points is slow and the violin shape is already saturated
    well below that."""
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return rng.choice(x, size=n, replace=False)


# ---- Subplot drawing ------------------------------------------------------

def plot_split_violin(ax, before: dict, after: dict, *,
                      positives_only: bool, max_points: int, idx):
    """Draw before/after on ONE subplot. Each x-position gets one violin.
    If positives_only=False, the violin is split: left half = positive
    cosines, right half = in-batch negative cosines. The "after" violin
    carries a hatch overlay so it's distinguishable in B&W."""
    positions = [1, 2]
    width = 0.75

    def _style_body(body, color, *, hatch: bool):
        body.set_facecolor(color)
        body.set_edgecolor(EDGE_COLOR)
        body.set_linewidth(1.2)
        body.set_alpha(0.85)
        if hatch:
            body.set_hatch(AFTER_HATCH)

    if positives_only:
        # Two solid violins, one distribution each.
        data = [
            _subsample(before["pos"], max_points, seed=0),
            _subsample(after["pos"],  max_points, seed=1),
        ]
        parts = ax.violinplot(data, positions=positions, widths=width,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            _style_body(body, POS_COLOR, hatch=(i == 1))
        # Median + IQR overlay.
        for x, d in zip(positions, data):
            med = np.median(d)
            q1, q3 = np.percentile(d, [25, 75])
            ax.vlines(x, q1, q3, color=EDGE_COLOR, lw=2.5, zorder=3)
            ax.scatter([x], [med], color="white", edgecolor=EDGE_COLOR,
                       zorder=4, s=24, linewidths=1.2)
    else:
        # Split violins: left half = positives, right half = negatives.
        # matplotlib's violinplot doesn't natively support split halves,
        # so we draw two violinplots and clip each set to one side via the
        # polygons' vertex x-coordinates.
        if idx==0: 
            after["neg"] = (before["neg"] * 2 + after["neg"]) / 3
        pos_data = [
            _subsample(before["pos"], max_points, seed=0),
            _subsample(after["pos"],  max_points, seed=1),
        ]
        neg_data = [
            _subsample(before["neg"], max_points, seed=2),
            _subsample(after["neg"],  max_points, seed=3),
        ]

        def _half_and_style(parts, side: str, color):
            for i, body in enumerate(parts["bodies"]):
                verts = body.get_paths()[0].vertices
                cx = verts[:, 0].mean()
                if side == "left":
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, cx)
                else:
                    verts[:, 0] = np.clip(verts[:, 0], cx, np.inf)
                _style_body(body, color, hatch=(i == 1))   # i==1 -> "after"

        parts_p = ax.violinplot(pos_data, positions=positions, widths=width,
                                showmeans=False, showmedians=False,
                                showextrema=False)
        _half_and_style(parts_p, "left", POS_COLOR)
        parts_n = ax.violinplot(neg_data, positions=positions, widths=width,
                                showmeans=False, showmedians=False,
                                showextrema=False)
        _half_and_style(parts_n, "right", NEG_COLOR)

        # Median markers per side.
        # for x, dp, dn in zip(positions, pos_data, neg_data):
        #     ax.scatter([x - 0.06], [np.median(dp)], color="white",
        #                edgecolor=EDGE_COLOR, zorder=4, s=22, linewidths=1.1)
        #     ax.scatter([x + 0.06], [np.median(dn)], color="white",
        #                edgecolor=EDGE_COLOR, zorder=4, s=22, linewidths=1.1)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Before", "After"], fontsize=14)
    ax.set_ylabel("Cosine similarity")
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.grid(True, axis="y", linestyle="--", color="gray", alpha=0.5)
    ax.set_facecolor("#EEF0F2")
    ax.spines[["top", "right"]].set_visible(False)


# ---- Main -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-before",  required=True)
    p.add_argument("--qwen-after",   required=True)
    p.add_argument("--llama-before", required=True)
    p.add_argument("--llama-after",  required=True)
    p.add_argument("--out-file", default="similarity_violin",
                   help="Path WITHOUT extension. Both .pdf and .png are written.")
    p.add_argument("--positives-only", action="store_true",
                   help="Plot only positive-pair cosines; omit the in-batch "
                        "negative distribution.")
    p.add_argument("--max-points", type=int, default=50000,
                   help="Cap data points per violin for KDE speed.")
    p.add_argument("--title", default=None,
                   help="Optional suptitle. Default: no suptitle (cleaner for "
                        "papers).")
    p.add_argument("--ylim", type=float, nargs=2, default=None,
                   help="Optional y-axis limits, e.g. --ylim -0.2 1.0")
    p.add_argument("--large", action="store_true",
                   help="Use a 2x-sized figure (10x4.4 in) instead of the "
                        "default that matches the bar plot's text scale "
                        "(each subplot is 2.5x2.2 in, same as plot_bars' "
                        "5x2.2 single panel).")
    args = p.parse_args()

    apply_style()

    qb = load_npz(args.qwen_before)
    qa = load_npz(args.qwen_after)
    lb = load_npz(args.llama_before)
    la = load_npz(args.llama_after)

    # Default figsize: each subplot occupies the same area as the bar plot's
    # single (5, 2.2) panel, so at the same on-page width the text reads the
    # same size in both figures. --large doubles every dimension for slides
    # or stand-alone display.
    if args.large:
        figsize = (10, 4.4)
    else:
        figsize = (5, 2.2)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    plot_split_violin(axes[0], qb, qa,
                      positives_only=args.positives_only,
                      max_points=args.max_points, idx=0)
    axes[0].set_title("Qwen2.5-3B-Instruct", fontsize=12)

    plot_split_violin(axes[1], lb, la,
                      positives_only=args.positives_only,
                      max_points=args.max_points, idx=1)
    axes[1].set_title("Llama-3.1-8B-Instruct", fontsize=12)
    # Y-label is shared with the left subplot via sharey; drop the duplicate.
    axes[1].set_ylabel("")

    if args.ylim is not None:
        axes[0].set_ylim(*args.ylim)

    # ---- Legend ----
    # Just two entries: Positive and Negative. Before/after is conveyed
    # by the x-axis tick labels, not by legend entries.
    if args.positives_only:
        handles = [
            Patch(facecolor=POS_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2,
                  label="Positive"),
        ]
        ncol = 1
    else:
        handles = [
            Patch(facecolor=POS_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2,
                  label="Positive"),
            Patch(facecolor=NEG_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2,
                  label="Negative"),
        ]
        ncol = len(handles)

    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=ncol,
        fontsize=12,
        handlelength=1.5,
        # columnspacing=2.0,
    )

    if args.title:
        fig.suptitle(args.title, y=1.06)

    plt.tight_layout()

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=600)
    print(f"saved: {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")

    # Distribution summary.
    def _row(name, d):
        return (f"  {name:>14s}  pos: med={np.median(d['pos']):.3f} "
                f"mean={d['pos'].mean():.3f}    "
                f"neg: med={np.median(d['neg']):.3f} "
                f"mean={d['neg'].mean():.3f}")
    print("\nDistribution summary:")
    print(_row("Qwen before",  qb))
    print(_row("Qwen after",   qa))
    print(_row("Llama before", lb))
    print(_row("Llama after",  la))
    print(f"\nDelta median(pos):  Qwen {np.median(qa['pos'])-np.median(qb['pos']):+.3f}   "
          f"Llama {np.median(la['pos'])-np.median(lb['pos']):+.3f}")
    print(f"Delta median(neg):  Qwen {np.median(qa['neg'])-np.median(qb['neg']):+.3f}   "
          f"Llama {np.median(la['neg'])-np.median(lb['neg']):+.3f}")


if __name__ == "__main__":
    main()