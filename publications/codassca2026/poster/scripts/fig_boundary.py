"""Results row: gain over random-K per dataset, for the label-free method, PCA and the supervised
MI selector. Numbers: publications/generalization/reports/*.txt (synthetic, EEG, PAMAP2) and the
Opportunity / Indian Pines screens (experiments/*_screen.py, logs in poster/data/)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_style import *   # noqa

# (dataset label, K text, {method: score}, random score, metric)
ROWS = [
    ("Synthetic",      "K = 8 of 64",   {"Proposed (label-free)": 0.944, "PCA": 0.950, "Supervised MI": 0.994}, 0.254),
    ("Motor-imagery EEG", "K = 6 of 22", {"Proposed (label-free)": 0.517, "PCA": 0.484, "Supervised MI": 0.527}, 0.486),
    ("PAMAP2 (wearable HAR)", "K = 7 of 27", {"Proposed (label-free)": 0.627 - 0.622 + 0.5935, "PCA": 0.562, "Supervised MI": 0.606}, 0.5935),
    ("Opportunity (wearable HAR)", "K = 10 of 113", {"Proposed (label-free)": 0.835, "PCA": 0.845, "Supervised MI": 0.881}, 0.847),
    ("Indian Pines (remote-sensing HSI)", "K = 20 of 200", {"Proposed (label-free)": 0.650, "PCA": 0.576, "Supervised MI": 0.727}, 0.760),
]
METHODS = ["Proposed (label-free)", "PCA", "Supervised MI"]


def main():
    fig = slot(1408, 222); ax = fig.add_subplot(111)
    n = len(ROWS); w = 0.25; x = np.arange(n)
    for mi, m in enumerate(METHODS):
        c = METHOD_STYLE[m][0]
        gains = [r[2][m] - r[3] for r in ROWS]
        bars = ax.bar(x + (mi - 1) * (w + 0.02), gains, w, color=c, label=m, lw=0)
        for b, g in zip(bars, gains):
            ax.text(b.get_x() + b.get_width() / 2, g + (0.02 if g >= 0 else -0.02), f"{g:+.2f}", ha="center",
                    va="bottom" if g >= 0 else "top", fontsize=FS_SMALL - 2, color=INK)
    ax.axhline(0, color=INK, lw=1.4, zorder=1)
    ax.axvline(1.5, color=INK2, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\n{r[1]}" for r in ROWS], fontsize=FS_SMALL)
    ax.set_ylabel("score − random-K\n(same classifier)", fontsize=FS_SMALL)
    ax.set_ylim(-0.30, 1.06); ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8]); ax.tick_params(axis="y", labelsize=FS_SMALL - 1)
    ax.grid(axis="x", visible=False)
    ax.text(0.72, 1.05, "informative channels: a small, non-redundant subset", ha="center", va="top", fontsize=FS_SMALL - 1, color=INK, weight="bold")
    ax.text(3.15, 1.05, "redundant near-duplicate channels: nothing beats random, not even the supervised selector",
            ha="center", va="top", fontsize=FS_SMALL - 1, color=INK, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.9), ncol=3, fontsize=FS_SMALL, frameon=True, framealpha=0.95, edgecolor="none", handlelength=1.0, columnspacing=1.2, borderaxespad=0.1)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.98, bottom=0.29)
    save(fig, "fig_boundary.png")


def main_doc():
    """Two-row variant for the presenter guide (A4 width): one row per regime."""
    fig = slot(700, 470)
    groups = [(ROWS[:2], "Selection pays: informative channels are a small, non-redundant subset"),
              (ROWS[2:], "Redundant near-duplicate channels: nothing beats random, not even the supervised selector")]
    for gi, (rows, title) in enumerate(groups):
        ax = fig.add_subplot(2, 1, gi + 1)
        n = len(rows); w = 0.25; x = np.arange(n)
        for mi, m in enumerate(METHODS):
            c = METHOD_STYLE[m][0]; gains = [r[2][m] - r[3] for r in rows]
            bars = ax.bar(x + (mi - 1) * (w + 0.02), gains, w, color=c, label=m, lw=0)
            for b, g in zip(bars, gains):
                ax.text(b.get_x() + b.get_width() / 2, g + (0.02 if g >= 0 else -0.02), f"{g:+.2f}", ha="center",
                        va="bottom" if g >= 0 else "top", fontsize=FS_SMALL - 2, color=INK)
        ax.axhline(0, color=INK, lw=1.4, zorder=1)
        short = {"PAMAP2 (wearable HAR)": "PAMAP2\n(wearable HAR)", "Opportunity (wearable HAR)": "Opportunity\n(wearable HAR)",
                 "Indian Pines (remote-sensing HSI)": "Indian Pines\n(remote-sensing HSI)"}
        ax.set_xticks(x); ax.set_xticklabels([f"{short.get(r[0], r[0])}\n{r[1]}" for r in rows], fontsize=FS_SMALL - 1)
        ax.set_xlim(-0.6, max(n, 3) - 0.4)
        ax.set_ylim(-0.30, 0.98 if gi == 0 else 0.30); ax.tick_params(axis="y", labelsize=FS_SMALL - 1)
        ax.set_ylabel("score − random-K", fontsize=FS_SMALL); ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=FS_SMALL + 1, loc="left", pad=8)
        if gi == 0:
            ax.legend(loc="upper right", ncol=1, fontsize=FS_SMALL, frameon=False, handlelength=1.0, borderaxespad=0.1)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.16, hspace=0.95)
    save(fig, "fig_boundary_doc.png")


if __name__ == "__main__":
    main(); main_doc()
