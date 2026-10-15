"""Figures of E4 (inclusion detectability): E4_cnr_depth, E4_max_depth, E4_panels.
    .venv/bin/python publications/spectralforge_studio/experiments/E4_figures.py [results_dir] [figure_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          verm="#D55E00", purple="#CC79A7")
EXC_COL = {"340": OI["purple"], "365": OI["blue"], "400": OI["green"], "430": OI["verm"]}
W2 = 7.16
PANEL_EX = "365"


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})


def save(fig, figdir, name):
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_cnr_depth(R, figdir):
    res = R["results"]
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.3), sharey=True, sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.12))
    for i, kind in enumerate(("nodule", "vessel")):
        for j, rk in enumerate(sorted(res[kind], key=lambda s: float(s[1:]))):
            ax = axs[i, j]
            for ek, x in sorted(res[kind][rk].items(), key=lambda kv: float(kv[0])):
                d = np.array(x["depths_mm"])
                c = np.array(x["cnr_instrument"], float)
                seeds = np.array(x["cnr_instrument_seeds"], float)
                col = EXC_COL.get(ek, "k")
                ax.fill_between(d, np.maximum(seeds.min(1), 1e-3), np.maximum(seeds.max(1), 1e-3), color=col, alpha=0.15, lw=0)
                ax.plot(d, np.maximum(c, 1e-2), "o-", color=col, label=f"{ek} nm")
                raw = np.array(x["cnr_raw_single"], float)
                ax.plot(d, np.maximum(raw, 1e-3), ":", color=col, lw=0.9)
            ax.axhline(3, color="0.35", lw=0.7, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(0.01, 30)
            ax.set_title(f"{'nodule' if kind == 'nodule' else 'vessel'}, $r={rk[1:]}$ mm", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("CNR")
            if i == 1:
                ax.set_xlabel("depth of inclusion top (mm)")
            if i == 0 and j == 0:
                h, l = ax.get_legend_handles_labels()
                fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05), handlelength=1.2)
                fig.text(0.5, 0.925, "solid: instrument CNR (MC averaged over 3 seeds); dotted: raw single capture; band: seed range", ha="center", fontsize=6.8, color="0.3")
            ax.text(0.03, 3.4, "CNR = 3", transform=ax.get_yaxis_transform(), fontsize=6.5, color="0.35")
    fig.subplots_adjust(top=0.88)
    save(fig, figdir, "E4_cnr_depth")


def _dmax_panel(ax, R, kind, key="dmax_mm", flag="dmax_flag", ci="dmax_ci95_mm"):
    res = R["results"][kind]
    rks = sorted(res, key=lambda s: float(s[1:]))
    rs = [float(k[1:]) for k in rks]
    exs = sorted(res[rks[0]], key=float)
    dmin = min(res[rks[0]][exs[0]]["depths_mm"])
    dmx = max(res[rks[0]][exs[0]]["depths_mm"])
    for e_i, ek in enumerate(exs):
        col = EXC_COL.get(ek, "k")
        for r, rk in zip(rs, rks):
            x = res[rk][ek]
            xr = r * (1 + 0.1 * (e_i - 1.5))
            if x[flag] == "ok":
                c = x.get(ci) or [None, None]
                yerr = None
                if c[0] is not None and c[1] is not None:
                    yerr = [[max(x[key] - c[0], 0)], [max(c[1] - x[key], 0)]]
                ax.errorbar([xr], [x[key]], yerr=yerr, fmt="o", color=col, ms=3.8, lw=0.8, capsize=0)
            elif x[flag] == "below_dmin":
                ax.plot([xr], [dmin], "v", color=col, ms=4.5, mfc="white")
        ax.plot([], [], "o", color=col, label=f"{ek} nm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(rs)
    ax.set_xticklabels([f"{r:g}" for r in rs])
    ax.minorticks_off()
    ax.set_xlabel("inclusion radius (mm)")
    ax.set_ylim(0.03, 3.5)
    ax.set_xlim(min(rs) * 0.7, max(rs) * 1.4)


def fig_max_depth(R, figdir):
    res = R["results"]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0), gridspec_kw=dict(wspace=0.32, hspace=0.45))
    a = axs[0, 0]
    _dmax_panel(a, R, "vessel")
    a.set_ylabel("depth at CNR = 3 (mm)")
    a.set_title("(a) vessel, per-pixel CNR", fontsize=8.5)
    a.legend(loc="upper left", fontsize=6.5, handlelength=0.8, labelspacing=0.2, ncol=2, columnspacing=0.8)
    a.text(0.97, 0.04, "$\\triangledown$: CNR<3 at 0.05 mm", transform=a.transAxes, ha="right", fontsize=6.5)
    a = axs[0, 1]
    _dmax_panel(a, R, "vessel", "dmax_roi_mm", "dmax_roi_flag", "dmax_roi_ci95_mm")
    a.set_title("(b) vessel, ROI-averaged CNR", fontsize=8.5)
    # nodule: peak CNR vs radius
    a = axs[1, 0]
    rks = sorted(res["nodule"], key=lambda s: float(s[1:]))
    rs = [float(k[1:]) for k in rks]
    for e_i, ek in enumerate(sorted(res["nodule"][rks[0]], key=float)):
        col = EXC_COL.get(ek, "k")
        pk = [max(res["nodule"][rk][ek]["cnr_instrument"]) for rk in rks]
        fl = [res["nodule"][rk][ek]["delta_noise_floor_conservative_e"] / res["nodule"][rk][ek]["sigma_pix_e"][0] for rk in rks]
        xs = [r * (1 + 0.1 * (e_i - 1.5)) for r in rs]
        a.plot(xs, pk, "o", color=col, ms=4)
        a.plot(xs, fl, "x", color=col, ms=4, mew=1.0)
    a.axhline(3, color="0.35", lw=0.7, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks(rs)
    a.set_xticklabels([f"{r:g}" for r in rs])
    a.minorticks_off()
    a.set_xlabel("nodule radius (mm)")
    a.set_ylabel("max over depth of CNR")
    a.set_ylim(0.01, 8)
    a.set_title("(c) nodule: no depth reaches CNR = 3", fontsize=8.5)
    a.text(0.03, 0.04, "o peak per-pixel CNR\nx Monte Carlo floor of the 3-seed mean", transform=a.transAxes, fontsize=6.5)
    # decay length fits
    a = axs[1, 1]
    allv = []
    for rk, d_ in res["vessel"].items():
        for ek, x in d_.items():
            f = x["fit_delta_signal"]
            if not f.get("delta"):
                continue
            col = EXC_COL.get(ek, "k")
            ci = x["delta_fit_signal_ci95_mm"]
            yerr = [[f["delta"] - ci[0]], [ci[1] - f["delta"]]] if ci[0] is not None and ci[1] is not None else None
            a.errorbar([x["delta_pred_em_only_mm"]], [f["delta"]], yerr=yerr, fmt="o", color=col, ms=3.8, lw=0.6, capsize=0)
            a.plot([x["delta_pred_twoway_mm"]], [f["delta"]], "s", color=col, ms=3.5, mfc="none")
            allv += [x["delta_pred_em_only_mm"], x["delta_pred_twoway_mm"], f["delta"]]
    lim = [0.04, 6]
    a.plot(lim, lim, color="0.4", lw=0.7)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("diffusion-theory length of the cover (mm)")
    a.set_ylabel("fitted decay length of $|\\Delta|$ (mm)")
    a.plot([], [], "ko", ms=3.8, label="vs $1/\\mu_{\\rm eff}(\\lambda_{em})$")
    a.plot([], [], "ks", ms=3.5, mfc="none", label="vs $1/(\\mu_{\\rm eff}^{ex}+\\mu_{\\rm eff}^{em})$")
    a.legend(loc="upper left", fontsize=6.5, handlelength=0.8)
    a.set_title("(d) vessel contrast decay (colour = excitation)", fontsize=8.5)
    save(fig, figdir, "E4_max_depth")


def fig_panels(R, res_dir, figdir):
    cache = Path(res_dir) / "cache"
    st = R["settings"]
    seeds = st["seeds"]
    dk = st["depths_mm"]
    E = f"E{PANEL_EX}"
    half = 30
    from E4_inclusion_detectability import roi_masks  # noqa: F401

    def pic(kind, r, d, sigma):
        ims = []
        for s in seeds:
            a = np.load(cache / f"{kind}_r{r:g}_d{d:g}_s{s}_crn.npz")
            b = np.load(cache / f"base_r0_d{d:g}_s{s}_crn.npz")
            ims.append(a[E].astype(float) - b[E].astype(float))
        c = np.mean(ims, 0)
        cy, cx = int(round(float(a["cy"]))), int(round(float(a["cx"])))
        return c[cy - half:cy + half + 1, cx - half:cx + half + 1] / sigma, float(a["pitch"])

    res = R["results"]
    kinds = [("nodule", sorted(res["nodule"], key=lambda s: float(s[1:]))), ("vessel", sorted(res["vessel"], key=lambda s: float(s[1:])))]
    ncol = max(len(dk["nodule"]), len(dk["vessel"]))
    fig, axs = plt.subplots(6, ncol, figsize=(W2, 6.6), gridspec_kw=dict(hspace=0.12, wspace=0.05))
    vmax = 6.0
    im = None
    row = 0
    for kind, rks in kinds:
        for rk in rks:
            x = res[kind][rk][PANEL_EX]
            for j in range(ncol):
                ax = axs[row, j]
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j >= len(dk[kind]):
                    ax.axis("off")
                    continue
                d = dk[kind][j]
                idx = x["depths_mm"].index(d)
                sig = x["sigma_pix_e"][idx]
                img, pitch = pic(kind, float(rk[1:]), d, sig)
                im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                th = np.linspace(0, 2 * np.pi, 100)
                r_px = float(rk[1:]) / pitch
                if kind == "nodule":
                    ax.plot(half + r_px * np.cos(th), half + r_px * np.sin(th), color="k", lw=0.5, ls=":")
                else:
                    for sgn in (-1, 1):
                        ax.axvline(half + sgn * r_px, color="k", lw=0.5, ls=":")
                ax.text(0.04, 0.05, f"{x['cnr_instrument'][idx]:.1f}", transform=ax.transAxes, fontsize=6.5, color="k",
                        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.75))
                if row in (0, 3):
                    ax.set_title(f"{d:g}", fontsize=7.5, pad=2)
                if j == 0:
                    ax.text(-0.08, 0.5, f"{kind}\n$r={rk[1:]}$ mm", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=7.5)
            row += 1
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("contrast / sensor noise per pixel ($\\Delta$ image / $\\sigma_{\\rm pix}$)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.075, f"{PANEL_EX} nm excitation, fluorescence bands, matched no-inclusion capture subtracted, mean of {len(seeds)} MC seeds; "
             "column labels = depth d of the inclusion top (mm); white number = instrument CNR; dotted = true inclusion outline; tile = 11.5 mm", ha="center", fontsize=6.8, color="0.25")
    save(fig, figdir, "E4_panels")


def make(res_dir, figdir):
    style()
    R = json.loads((Path(res_dir) / "results.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    fig_cnr_depth(R, figdir)
    fig_max_depth(R, figdir)
    fig_panels(R, res_dir, figdir)


if __name__ == "__main__":
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "E4"
    fd = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "figures"
    make(rd, fd)
