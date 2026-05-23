"""Pepsin SOTA figure under the poster protocol (small ROI train, rest test).

Reads `revision/baselines/results_pepsin_poster/pepsin/method_summary.csv`.
Writes panel_C_knn_vs_K_poster.{png,pdf}.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "revision" / "baselines" / "results_pepsin_poster" / "pepsin" / "method_summary.csv"
OUT_STEM = Path(__file__).resolve().parent / "panel_C_knn_vs_K_poster"

PALETTE = {
    "ae_perturb":     ("Ours (AE-perturb, best per K)", "#C8102E", "-",  "o", 3.2, 10),
    "variance":       ("Variance",                "#4A4A4A", "--", "s", 2.0,  6),
    "pca_loading":    ("PCA-loading",             "#7A7A7A", "--", "D", 2.0,  6),
    "sam_greedy":     ("SAM-greedy",              "#2E8B57", "--", "v", 2.0,  6),
    "spa":            ("SPA",                     "#1F77B4", "--", "^", 2.0,  6),
    "mcuve":          ("MCUVE",                   "#FF7F0E", "--", "x", 2.0,  6),
    "issc":           ("ISSC",                    "#9467BD", "--", "P", 2.0,  6),
    "bsnet_fc":       ("BS-Net-FC",               "#E377C2", "--", "*", 2.0,  7),
    "sparse_lasso":   ("Sparse-LASSO (supervised)","#8C564B", "--", "h", 2.0,  6),
    "random":         ("Random",                  "#888888", ":",  ".", 1.4,  4),
}
ORDER = ["ae_perturb", "sam_greedy", "spa", "ssc", "issc",
         "variance", "pca_loading", "mcuve", "bsnet_fc", "sparse_lasso", "random"]


def main():
    if not SUMMARY.exists():
        print(f"Missing: {SUMMARY}")
        return
    df = pd.read_csv(SUMMARY).sort_values(["method", "K"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in ORDER:
        s = df[df["method"] == m]
        if s.empty: continue
        label, color, ls, marker, lw, ms = PALETTE[m]
        ax.plot(s["K"], s["mean"], color=color, linestyle=ls, marker=marker,
                linewidth=lw, markersize=ms, label=label,
                zorder=5 if m == "ae_perturb" else 3,
                alpha=1.0 if m == "ae_perturb" else 0.85)
        if not s["std"].isna().all():
            ax.fill_between(s["K"], s["mean"] - s["std"], s["mean"] + s["std"],
                            color=color, alpha=0.08, zorder=2)

    # Reference: baseline KNN-5 accuracy on full 158 bands under same protocol
    ax.axhline(0.7978, color="black", linestyle=":", linewidth=1.5, alpha=0.6,
               label="Full 158-band baseline (0.798)")

    ax.set_xlabel("K (number of selected bands)", fontsize=12)
    ax.set_ylabel("KNN-5 test-set accuracy", fontsize=12)
    ax.set_title(
        "Pepsin-Collagen Dataset (poster protocol): AE-perturb leads at every K\n"
        "Train on small ROI rectangles, test on remaining labeled pixels",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9, ncols=2, frameon=True, framealpha=0.95)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_STEM.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
        try: rel = p.resolve().relative_to(ROOT)
        except ValueError: rel = p
        print(f"  wrote {rel}")
    plt.close(fig)


if __name__ == "__main__":
    main()
