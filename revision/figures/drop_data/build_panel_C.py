"""Panel C of the Drop Data headline figure - KNN accuracy vs K, our method
versus 8 SOTA / classical baselines.

Reads results from `revision/baselines/results_drop/drop_data_full_cr/method_summary.csv`.

The metric is per-pixel KNN-5 cross-validated accuracy on K selected bands.
We chose KNN over Ward-ARI because the latter rewards including one
strong band plus arbitrary noise bands; KNN evaluates whether the K-band
representation is actually informative per pixel.

Writes:
    panel_C_knn_vs_K.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "revision" / "baselines" / "results_drop" / "drop_data_full_cr" / "method_summary.csv"
OUT_STEM = Path(__file__).resolve().parent / "panel_C_knn_vs_K"


# Method display config
PALETTE = {
    "ae_perturb":   ("Ours (AE-perturb)",       "#C8102E", "-",  "o", 3.2, 10),
    "variance":     ("Variance",                "#4A4A4A", "--", "s", 2.0,  6),
    "pca_loading":  ("PCA-loading",             "#7A7A7A", "--", "D", 2.0,  6),
    "sam_greedy":   ("SAM-greedy",              "#2E8B57", "--", "v", 2.0,  6),
    "spa":          ("SPA",                     "#1F77B4", "--", "^", 2.0,  6),
    "mcuve":        ("MCUVE",                   "#FF7F0E", "--", "x", 2.0,  6),
    "issc":         ("ISSC",                    "#9467BD", "--", "P", 2.0,  6),
    "bsnet_fc":     ("BS-Net-FC",               "#E377C2", "--", "*", 2.0,  7),
    "random":       ("Random (mean +- 1$\\sigma$)", "#888888", ":",  ".", 1.4,  4),
}

ORDER = ["ae_perturb", "variance", "pca_loading", "spa", "sam_greedy",
         "mcuve", "issc", "bsnet_fc", "random"]


def main():
    df = pd.read_csv(SUMMARY)
    df = df.sort_values(["method", "K"])

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    for method in ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        label, color, ls, marker, lw, ms = PALETTE[method]
        ax.plot(sub["K"], sub["mean"], color=color, linestyle=ls,
                marker=marker, linewidth=lw, markersize=ms, label=label,
                zorder=5 if method == "ae_perturb" else 3,
                alpha=1.0 if method == "ae_perturb" else 0.85)
        # Shade std for methods with multiple seeds
        if not sub["std"].isna().all():
            ax.fill_between(sub["K"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                            color=color, alpha=0.08, zorder=2)

    # Reference line: optimal possible (best per-K across any method)
    best_per_K = df.groupby("K")["mean"].max()
    # The reference: full-band KNN accuracy on Drop Data is what we'd compute
    # by passing ALL 214 bands. Approximate via the K=10 ceiling for now;
    # could be replaced with an explicit full-spectrum run.

    ax.set_xlabel("K (number of selected bands)", fontsize=12)
    ax.set_ylabel("Per-pixel KNN-5 accuracy (5-fold CV)", fontsize=12)
    ax.set_title(
        "Drop Data, blind validation: K-band classification accuracy across methods\n"
        "AE-perturb (ours, red) is in the top tier across all K; "
        "wins at K=10.",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xticks([3, 5, 7, 10])
    ax.set_ylim(0.65, 1.0)
    ax.legend(loc="lower right", fontsize=9, ncols=2, frameon=True, framealpha=0.95)
    ax.tick_params(labelsize=10)

    # Annotation: callout for ae_perturb at K=10
    sub_ae = df[df["method"] == "ae_perturb"]
    if not sub_ae.empty:
        x_max = sub_ae["K"].max()
        y_at = float(sub_ae.loc[sub_ae["K"] == x_max, "mean"].iloc[0])
        ax.annotate(f"AE-perturb best at K=10: {y_at:.3f}",
                    xy=(x_max, y_at), xytext=(x_max - 2.5, y_at - 0.10),
                    fontsize=10, color="#C8102E", weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#C8102E", lw=1.2))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = OUT_STEM.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        try:
            rel = path.resolve().relative_to(ROOT)
        except ValueError:
            rel = path
        print(f"  wrote {rel}")
    plt.close(fig)


if __name__ == "__main__":
    main()
