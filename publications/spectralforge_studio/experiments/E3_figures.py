"""Figures of experiment E3 (called by E3_sensitivity_redox.py; also runnable: python E3_figures.py [--quick])."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
# Okabe-Ito
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YEL, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#7f7f7f"

PARAMS = ["cover_mm", "mus500", "b_mie", "blood", "sto2", "collagen_uM", "nadh_uM", "fad_uM"]
LABEL = {"cover_mm": "cover thickness", "mus500": r"cover $\mu_s'(500)$", "b_mie": r"cover $b_{\rm Mie}$",
         "blood": "cover blood fraction", "sto2": r"cover StO$_2$", "collagen_uM": "collagen fluor. conc.",
         "nadh_uM": "NADH conc.", "fad_uM": "FAD conc."}
GROUP_COL = {p: (VERM if p not in ("nadh_uM", "fad_uM") else BLUE) for p in PARAMS}
TITLE = {"nadh": "NADH band (365 / 460 nm)", "fad": "FAD band (450 / 530 nm)", "rr_raw": "optical redox ratio",
         "peak": "peak emission (365 nm ex.)", "rr_F": "RR, FAD / elastic", "rr_N": "RR, NADH / cover-AF",
         "rr_FN": "RR, both corrections"}


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _bars(ax, ind, key, ylabel=None, title=None, legend=False):
    y = np.arange(len(PARAMS))[::-1]
    h = 0.36
    for off, (idx, lab, hatch) in zip((h / 2, -h / 2), (("S1", "first order $S_1$", None), ("ST", "total $S_T$", None))):
        v = np.array([ind[p][idx] for p in PARAMS])
        ci = np.array([ind[p][idx + "_ci"] for p in PARAMS])
        err = np.vstack([np.maximum(v - ci[:, 0], 0), np.maximum(ci[:, 1] - v, 0)])
        cols = [GROUP_COL[p] for p in PARAMS]
        ax.barh(y + off, np.clip(v, 0, None), height=h, color=cols, alpha=0.95 if idx == "ST" else 0.45,
                edgecolor="none", xerr=err, error_kw=dict(lw=0.6, capsize=1.2, capthick=0.6), label=lab)
    ax.set_yticks(y)
    if ylabel:  # shared y axis: only label the first panel (labelling the others with "" would blank them all)
        ax.set_yticklabels([LABEL[p] for p in PARAMS])
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Sobol' index")
    ax.set_title(title, fontsize=8)
    ax.axvline(0, color="k", lw=0.5)


def make_figures(out: Path, quick: bool = False) -> None:
    out = Path(out)
    res = json.loads((out / "results.json").read_text())
    runs = pd.read_parquet(out / "runs.parquet")
    sob, mor = res["sobol"], res["morris"]
    sfx = "_quick" if quick else ""
    from matplotlib.patches import Patch

    # ---- Fig 1: S1 / ST with CIs for the raw signals and the redox ratio
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.9), sharey=True)
    for ax, o in zip(axs, ["nadh", "fad", "rr_raw", "peak"]):
        _bars(ax, sob[o]["indices"], o, ylabel=(o == "nadh"), title=TITLE[o])
    h = [Patch(color=VERM, label="cover / confounders"), Patch(color=BLUE, label="metabolic state (NADH, FAD)"),
         Patch(facecolor="k", alpha=0.45, label=r"light bar: $S_1$"), Patch(facecolor="k", alpha=0.95, label=r"dark bar: $S_T$")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=0.6)
    _save(fig, f"E3_sobol{sfx}")

    # ---- Fig 2: scatter matrix of the redox ratio against every input
    fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.5), sharey=True)
    for ax, p in zip(axs.ravel(), PARAMS):
        x, y = runs[f"p:{p}"], runs["m:rr_raw"]
        ax.scatter(x, y, s=4, color=GROUP_COL[p], alpha=0.5, lw=0, rasterized=True)
        rho = spearmanr(x, y).statistic
        ax.text(0.04, 0.05, rf"$\rho_s$ = {rho:+.2f}", transform=ax.transAxes, fontsize=7,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set_xlabel(LABEL[p] + {"cover_mm": " (mm)", "mus500": r" (mm$^{-1}$)", "nadh_uM": r" ($\mu$M)",
                                  "fad_uM": r" ($\mu$M)", "collagen_uM": r" ($\mu$M)"}.get(p, ""))
    for ax in axs[:, 0]:
        ax.set_ylabel(r"FAD/(FAD+NADH)")
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    _save(fig, f"E3_scatter{sfx}")

    # ---- Fig 3: before / after correction
    variants = ["rr_raw", "rr_F", "rr_N", "rr_FN"]
    vcol = [GREY, SKY, ORANGE, GREEN]
    fig = plt.figure(figsize=(7.16, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    xs = np.arange(len(PARAMS))
    w = 0.2
    for j, (v, c) in enumerate(zip(variants, vcol)):
        ind = sob[v]["indices"]
        st = np.array([ind[p]["ST"] for p in PARAMS])
        ci = np.array([ind[p]["ST_ci"] for p in PARAMS])
        ax.bar(xs + (j - 1.5) * w, np.clip(st, 0, None), w, color=c, label=TITLE[v],
               yerr=np.vstack([np.maximum(st - ci[:, 0], 0), np.maximum(ci[:, 1] - st, 0)]),
               error_kw=dict(lw=0.5, capsize=1, capthick=0.5))
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[p] for p in PARAMS], rotation=35, ha="right")
    ax.set_ylabel(r"total-order index $S_T$")
    ax.set_title("(a) per-input total effect on the redox ratio")
    ax.set_ylim(0, 2.0)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.5)
    ax = fig.add_subplot(gs[1])
    gi = res["group_surrogate"]
    for j, (v, c) in enumerate(zip(variants, vcol)):
        for gx, g in enumerate(("cover", "metabolism")):
            d = gi[v]["groups"][g]
            val = d["ST"]
            lo, hi = d["ST_ci"]
            ax.bar(gx + (j - 1.5) * w, max(val, 0), w, color=c, yerr=[[max(val - lo, 0)], [max(hi - val, 0)]],
                   error_kw=dict(lw=0.5, capsize=1, capthick=0.5))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cover group\n(6 inputs)", "metabolic group\n(NADH, FAD)"])
    ax.set_ylabel(r"group total index $S_T^{G}$")
    ax.set_ylim(0, 1.15)
    ax.set_title("(b) grouped (GP surrogate)")
    fig.tight_layout()
    _save(fig, f"E3_correction{sfx}")

    # ---- Fig 4: Morris vs Sobol cross-check on the raw redox ratio
    fig, axs = plt.subplots(1, 2, figsize=(3.5 * 2 * 0.9, 2.6))
    mo = mor["rr_raw"]
    ax = axs[0]
    for p in PARAMS:
        ax.errorbar(mo[p]["mu_star"], mo[p]["sigma"], xerr=[[mo[p]["mu_star"] - mo[p]["mu_star_ci"][0]],
                    [mo[p]["mu_star_ci"][1] - mo[p]["mu_star"]]], fmt="o", ms=4, color=GROUP_COL[p], lw=0.6)
        off = {"blood": (95, -6), "sto2": (5, 3), "nadh_uM": (40, -8), "collagen_uM": (60, 2), "b_mie": (8, 6)}.get(p, (3, 3))
        ax.annotate(LABEL[p], (mo[p]["mu_star"], mo[p]["sigma"]), fontsize=6, xytext=off, textcoords="offset points",
                    ha="left", arrowprops=dict(arrowstyle="-", lw=0.4, color=GREY) if p in ("blood", "collagen_uM", "nadh_uM") else None)
    ax.set_ylim(0.065, 0.26)
    ax.set_xlabel(r"Morris $\mu^*$"); ax.set_ylabel(r"Morris $\sigma$"); ax.set_title("(a) Morris screening, redox ratio")
    ax = axs[1]
    stv = np.array([sob["rr_raw"]["indices"][p]["ST"] for p in PARAMS])
    mus = np.array([mo[p]["mu_star"] for p in PARAMS])
    for p, a_, b_ in zip(PARAMS, mus, stv):
        ax.scatter(a_, b_, s=18, color=GROUP_COL[p])
    rho = spearmanr(mus, stv).statistic
    ax.set_xlabel(r"Morris $\mu^*$"); ax.set_ylabel(r"Sobol' $S_T$")
    ax.set_title(rf"(b) rank agreement, $\rho_s$ = {rho:.2f}")
    fig.tight_layout()
    _save(fig, f"E3_morris{sfx}")


if __name__ == "__main__":
    q = "--quick" in sys.argv
    make_figures(ROOT / "results" / "E3" / ("quick" if q else ""), q)
