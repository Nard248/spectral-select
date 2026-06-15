"""Re-render the Collagen Sponges multi-classifier accuracy curves so the
x-axis spans the full K range (3 to 149) instead of stopping at K=50.

Data: results/Pepsin_Classifier_Comparison/20260407_133016/classifier_comparison.csv
       (10 classifiers × 9 K values × 2 ROIs × 2 band-selection regimes)

We use the 'expanded_all_rows' ROI variant + 'autoencoder' band selection
(this matches the poster's classifier curves panel). The 'all' / baseline
rows give the per-classifier full-spectrum accuracy, which we render as a
dotted horizontal reference line on the legend.

Output: MasterThesis_Narek_Meloyan/figures/collagen_sponges/classifier_curves.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = (ROOT / "results" / "Pepsin_Classifier_Comparison"
       / "20260407_133016" / "classifier_comparison.csv")
OUT = (ROOT / "MasterThesis_Narek_Meloyan" / "figures"
       / "collagen_sponges" / "classifier_curves.png")

# Match the colour palette from the existing poster figure
COLOURS = {
    "GBM":         "#F2A93C",
    "KNN_k5":      "#5BA0FF",
    "KNN_k11":     "#1F77B4",
    "KNN_k11_dist":"#0B3D91",
    "LDA":         "#5DA84A",
    "MLP":         "#E0721E",
    "RF_100":      "#3CB371",
    "RF_300":      "#1F6E3C",
    "SVM_linear":  "#D62728",
    "SVM_rbf":     "#9B1B1B",
}
ORDER = ["GBM", "KNN_k11", "KNN_k11_dist", "KNN_k5", "LDA",
         "MLP", "RF_100", "RF_300", "SVM_linear", "SVM_rbf"]


def main() -> None:
    df = pd.read_csv(CSV)

    # Match the accuracy-envelope protocol (small training-ROI rectangles).
    # The 'expanded_all_rows' variant uses a much larger training set, which
    # raises the full-spectrum baseline to ~94% and obscures the noise-removal
    # benefit visible in the envelope. We use 'original_small' so both plots
    # show the same phenomenon.
    sub = df[(df["roi"] == "original_small")].copy()
    sub_sel = sub[sub["band_selection"] == "autoencoder"].copy()
    sub_base = sub[sub["band_selection"] == "all"].copy()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Determine the spectrum max K (the "all bands" baseline lives at K_max)
    K_max = int(sub_base["n_bands"].iloc[0]) if len(sub_base) > 0 else 158

    for clf in ORDER:
        c = COLOURS[clf]
        # Selection sweep: K = 3..50 from band_selection='autoencoder'
        curve = sub_sel[sub_sel["classifier"] == clf].sort_values("n_bands")
        # Append the per-classifier full-spectrum baseline as the K_max endpoint
        bl_row = sub_base[sub_base["classifier"] == clf]
        if len(bl_row) > 0:
            bl_acc = float(bl_row["accuracy"].iloc[0])
            xs = list(curve["n_bands"]) + [K_max]
            ys = list(curve["accuracy"]) + [bl_acc]
        else:
            xs = list(curve["n_bands"])
            ys = list(curve["accuracy"])
            bl_acc = None
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=1.6, color=c, label="")
        # Dotted horizontal reference line for the baseline
        if bl_acc is not None:
            ax.axhline(bl_acc, linewidth=0.7, linestyle=":", color=c, alpha=0.8)
            label = f"{clf} (bl={bl_acc*100:.1f}%)"
        else:
            label = clf
        # Dummy plot so the marker shows next to the label in the legend
        ax.plot([], [], marker="o", linewidth=1.6, color=c, label=label)

    ax.set_xlabel("Number of Selected Bands", fontsize=13)
    ax.set_ylabel("Classification Accuracy", fontsize=13)
    ax.set_ylim(0.55, 1.0)
    ax.set_xlim(0, K_max + 5)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    ax.legend(loc="lower right", fontsize=8.5, ncols=2,
              frameon=True, framealpha=0.95, columnspacing=1.0)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  K range: {sub_sel['n_bands'].min()} .. {sub_sel['n_bands'].max()}")
    print(f"  K values: {sorted(sub_sel['n_bands'].unique())}")


if __name__ == "__main__":
    main()
