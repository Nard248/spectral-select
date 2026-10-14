"""Regenerate Fig. 5 (benchmark results) for the camera-ready: 7.0 in wide, serif, larger fonts.
Data are copied verbatim from reports/paper_figures.py::fig_results (3-seed spatial values); no new runs."""
import pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
F = pathlib.Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})
beat = {"regime": ["clean", "low", "moderate", "high", "severe"],
        "random95": [0.573, 0.479, 0.574, 0.587, 0.592],
        "PCA": [0.699, 0.470, 0.539, 0.549, 0.549], "AE": [0.655, 0.449, 0.548, 0.575, 0.569],
        "mutInfo": [0.696, 0.504, 0.576, 0.604, 0.615]}
comp = {"k": [4, 8, 12, 16, 24, 32, 48, 64, 100],
        "PCA": [0.453, 0.517, 0.552, 0.572, 0.606, 0.616, 0.647, 0.656, 0.662],
        "AE": [0.467, 0.532, 0.552, 0.573, 0.616, 0.632, 0.639, 0.657, 0.667],
        "mutInfo": [0.495, 0.571, 0.604, 0.622, 0.659, 0.658, 0.673, 0.680, 0.680],
        "random": [0.461, 0.509, 0.547, 0.581, 0.638, 0.618, 0.636, 0.662, 0.668], "full": 0.683}
spatial = {"method": ["pixel\nvar.", "pixel\nMI", "random", "tex.\nvar.", "tex.\nMI", "oracle"],
           "F1": [0.516, 0.517, 0.710, 0.657, 0.688, 0.751]}
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0), gridspec_kw=dict(width_ratios=[1.1, 1, 1.05]))
x = np.arange(5); w = 0.2
for i, m in enumerate(["random95", "PCA", "AE", "mutInfo"]):
    ax[0].bar(x + (i - 1.5) * w, beat[m], w, label=("random(95th)" if m == "random95" else m))
ax[0].set_xticks(x); ax[0].set_xticklabels(beat["regime"]); ax[0].set_title("Beat-random across noise ($k$=24)")
ax[0].set_ylabel("best-NL macro-F1"); ax[0].set_ylim(0.35, 0.86); ax[0].legend(ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2)
for m in ["PCA", "AE", "mutInfo", "random"]:
    ax[1].plot(comp["k"], comp[m], marker="o", ms=3, lw=1.1, label=m)
ax[1].axhline(comp["full"], ls="--", c="k", lw=1, label="full-564")
ax[1].set_title("Compression: accuracy vs. #bands"); ax[1].set_xlabel("$k$ (selected bands)")
ax[1].set_ylim(0.43, 0.82); ax[1].legend(ncol=3, loc="upper center", columnspacing=0.7, handlelength=1.3, handletextpad=0.4, borderpad=0.3)
cols = ["#c44", "#c44", "#888", "#4a4", "#4a4", "#444"]
ax[2].bar(range(6), spatial["F1"], color=cols)
ax[2].set_xticks(range(6)); ax[2].set_xticklabels(spatial["method"], fontsize=7)
ax[2].set_title("Spatial-texture regime (3 seeds)"); ax[2].set_ylabel("block-CV macro-F1"); ax[2].set_ylim(0.4, 0.8)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(F / "fig_results_v2.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
fig.savefig(F / "fig_results_v2.pdf", bbox_inches="tight", pad_inches=0.02)
