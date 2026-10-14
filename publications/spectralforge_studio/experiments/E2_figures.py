"""Figures of E2 (cover-thickness response). Called by E2_cover_thickness.py or stand-alone:
    .venv/bin/python publications/spectralforge_studio/experiments/E2_figures.py [results_dir] [figure_dir]
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


def arr(d, keys=None):
    ks = sorted(d, key=float)
    return np.array([float(k) for k in ks]), np.array([np.nan if d[k] is None else d[k] for k in ks], float)


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def fig_penetration(R, figdir):
    S = R["sweep"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.55))
    a = ax[0]
    for k in sorted(S, key=float):
        r = S[k]
        t, m = arr(r["S_buried_mean"])
        _, se = arr(r["S_buried_se"]) if r["S_buried_se"] else (t, np.zeros_like(m))
        i0 = r["I0"] if r["I0"] and np.isfinite(r["I0"]) else m[0]
        col = EXC_COL.get(k, "k")
        ok = m > 0
        a.errorbar(t[ok], m[ok] / m[0], yerr=se[ok] / m[0], fmt="o", color=col, ms=3, lw=0.6, capsize=0, label=f"{k} nm")
        if r["delta_eff_mm"] and np.isfinite(r["delta_eff_mm"]):
            lo, hi = r["fit_window_mm"]
            tt = np.linspace(0, min(hi * 1.3, 2.0), 50)
            a.plot(tt, i0 / m[0] * np.exp(-tt / r["delta_eff_mm"]), "-", color=col, lw=0.9)
    a.set_yscale("log")
    a.set_xlim(-0.03, 2.05)
    a.set_ylim(1e-5, 3)
    a.set_xlabel("collagen cover thickness $t$ (mm)")
    a.set_ylabel("buried-layer signal $S(t)/S(0)$")
    a.legend(frameon=False, loc="lower left", ncol=2, handlelength=1.2, columnspacing=0.8)
    a.text(0.98, 0.97, "points below $10^{-5}$ (MC noise floor) clipped", transform=a.transAxes, fontsize=6, ha="right", va="top", color="0.4")
    panel_label(a, "(a)")
    # (b) delta_eff measured vs two-way diffusion prediction
    b = ax[1]
    ks = sorted(S, key=float)
    x = np.arange(len(ks))
    meas = np.array([S[k]["delta_eff_mm"] or np.nan for k in ks], float)
    lo = np.array([S[k]["delta_eff_ci95_mm"][0] if S[k]["delta_eff_ci95_mm"][0] is not None else np.nan for k in ks], float)
    hi = np.array([S[k]["delta_eff_ci95_mm"][1] if S[k]["delta_eff_ci95_mm"][1] is not None else np.nan for k in ks], float)
    two = np.array([S[k]["delta_pred_twoway_mm"] for k in ks], float)
    band = np.array([S[k]["delta_pred_band_resolved_mm"] or np.nan for k in ks], float)
    b.errorbar(x - 0.12, meas, yerr=[np.maximum(meas - lo, 0), np.maximum(hi - meas, 0)], fmt="o", color=OI["black"], capsize=2, label="MC fit")
    b.plot(x + 0.12, two, "s", color=OI["orange"], mfc="none", label="diffusion, 2-way")
    b.plot(x + 0.12, band, "^", color=OI["sky"], mfc="none", label="diffusion, band-resolved")
    b.set_yscale("log")
    b.set_xticks(x)
    b.set_xticklabels([f"{float(k):g}" for k in ks])
    b.set_xlabel("excitation (nm)")
    b.set_ylabel(r"$\delta_{\rm eff}$ (mm)")
    b.set_ylim(0.04, 3.0)
    b.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.0)
    panel_label(b, "(b)")
    # (c) ratio
    c = ax[2]
    ratio = np.array([S[k]["ratio_measured_over_band"] or np.nan for k in ks], float)
    ratio2 = np.array([S[k]["ratio_measured_over_twoway"] or np.nan for k in ks], float)
    c.axhline(1.0, color="0.6", lw=0.6)
    c.plot(x, ratio2, "s-", color=OI["orange"], mfc="none", label="vs 2-way")
    c.plot(x, ratio, "^-", color=OI["sky"], mfc="none", label="vs band-resolved")
    c.set_xticks(x)
    c.set_xticklabels([f"{float(k):g}" for k in ks])
    c.set_xlabel("excitation (nm)")
    c.set_ylabel(r"$\delta_{\rm eff}^{\rm MC}\,/\,\delta^{\rm diff}$")
    c.set_ylim(0.55, 1.25)
    c.legend(frameon=False, loc="lower left", fontsize=6.5)
    panel_label(c, "(c)")
    save(fig, figdir, "E2_penetration")


def fig_fraction_distortion(R, figdir):
    S = R["sweep"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw=dict(wspace=0.5))
    for k in sorted(S, key=float):
        r = S[k]
        col = EXC_COL.get(k, "k")
        t, f = arr(r["f_cover_mean"])
        _, fse = arr(r["f_cover_se"]) if r["f_cover_se"] else (t, np.zeros_like(f))
        ax[0].errorbar(t, f, yerr=fse, fmt="o-", color=col, lw=0.8, ms=3, label=f"{k} nm")
        xo = r["crossover_t_half_mm"]
        if xo and np.isfinite(xo):
            ax[0].plot([xo], [0.5], "v", color=col, ms=5, mec="k", mew=0.4)
        t, pk = arr(r["centroid_nm"]["mean"])
        _, pse = arr(r["centroid_nm"]["se"]) if r["centroid_nm"]["se"] else (t, np.zeros_like(pk))
        ok = np.isfinite(pk)
        # shift relative to t = 0 if the t = 0 value exists
        ax[1].errorbar(t[ok], pk[ok] - pk[ok][0], yerr=pse[ok], fmt="o-", color=col, lw=0.8, ms=3)
        t, br = arr(r["br"]["mean"])
        _, bse = arr(r["br"]["se"]) if r["br"]["se"] else (t, np.zeros_like(br))
        ok = np.isfinite(br)
        ax[2].errorbar(t[ok], br[ok] / br[ok][0], yerr=bse[ok] / br[ok][0], fmt="o-", color=col, lw=0.8, ms=3)
    for a_ in ax:
        a_.set_xscale("symlog", linthresh=0.05, linscale=0.4)
        a_.set_xlim(-0.004, 2.2)
        a_.set_xticks([0, 0.05, 0.1, 0.25, 0.5, 1, 2])
        a_.set_xticklabels(["0", "", "0.1", "0.25", "0.5", "1", "2"])
        a_.set_xlabel("cover thickness $t$ (mm)")
    ax[0].axhline(0.5, color="0.6", lw=0.5, ls=":")
    ax[0].set_ylabel("cover autofluorescence fraction")
    ax[0].legend(frameon=False, loc="lower right", fontsize=6.5)
    ax[0].text(0.25, 0.62, r"$\blacktriangledown$ 50 % crossover", transform=ax[0].transAxes, fontsize=6.5, va="top")
    ax[1].set_ylabel("emission-centroid shift (nm)")
    ax[1].axhline(0, color="0.6", lw=0.5)
    ax[2].set_ylabel("blue/red ratio, rel. to $t=0$")
    ax[2].axhline(1, color="0.6", lw=0.5)
    ax[2].set_yscale("log")
    for a_, s in zip(ax, "abc"):
        panel_label(a_, f"({s})")
    save(fig, figdir, "E2_fraction_distortion")


def fig_blood(R, figdir):
    B = R.get("blood") or {}
    if not B:
        return
    cols = {0.0: OI["black"], 0.01: OI["orange"], 0.03: OI["verm"]}
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.45))
    tref = 0.25
    for ex, ls in ((430.0, "-"), (400.0, "--")):
        for bl in (0.0, 0.01, 0.03):
            key = f"ex{ex:g}_B{bl:g}_t{tref:g}"
            if key not in B:
                continue
            d = B[key]
            em, sp = np.array(d["em_nm"]), np.array(d["spectrum"])
            base = B[f"ex{ex:g}_B0_t{tref:g}"]
            spb = np.array(base["spectrum"])
            if ex == 430.0:
                ax[0].plot(em, sp / spb.max(), ls, color=cols[bl], label=f"B = {bl * 100:g} %")
            ax[1].plot(em, sp / spb, ls, color=cols[bl], label=f"{ex:g} nm" if bl == 0.03 else None)
    ax[0].set_xlabel("emission wavelength (nm)")
    ax[0].set_ylabel("fluorescence (rel. to B=0 max)")
    ax[0].set_title(f"430 nm exc., $t$ = {tref:g} mm", fontsize=8)
    ax[0].legend(frameon=False)
    ax[1].set_xlabel("emission wavelength (nm)")
    ax[1].set_ylabel(r"$S_B(\lambda)/S_{B=0}(\lambda)$")
    ax[1].set_yscale("log")
    for lam in (542, 577):
        ax[1].axvline(lam, color="0.7", lw=0.5, ls=":")
    ax[1].text(546, 1.3, "HbO$_2$ Q bands", fontsize=6.5, color="0.4", va="bottom")
    ax[1].set_ylim(0.002, 2.5)
    ax[1].legend(frameon=False, title="solid 430, dashed 400 nm exc.", title_fontsize=6.5, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1)
    # (c) B/R vs t per B (ex 430)
    for bl in (0.0, 0.01, 0.03):
        t, br, se = [], [], []
        for tt in sorted({v["t_mm"] for v in B.values()}):
            key = f"ex430_B{bl:g}_t{tt:g}"
            if key in B:
                t.append(tt)
                br.append(B[key]["br"])
                se.append(B[key]["br_se"] or 0)
        ax[2].errorbar(t, br, yerr=[np.minimum(se, 0.7*np.array(br)), se], fmt="o-", color=cols[bl], lw=0.9, label=f"B = {bl * 100:g} %")
    ax[2].set_xlabel("cover thickness (mm)")
    ax[2].set_ylabel("blue/red ratio (430 nm exc.)")
    ax[2].set_yscale("log")
    ax[2].legend(frameon=False)
    for a_, s in zip(ax, "abc"):
        panel_label(a_, f"({s})")
    save(fig, figdir, "E2_blood")


def wedge_binned(R, bin_mm=0.2):
    """Seed-averaged, thickness-binned wedge/1D ratio (the per-pixel profiles are dominated by MC noise)."""
    Wd = R.get("wedge") or {}
    out = {}
    edges = np.arange(0.0, 2.0 + 1e-9, bin_mm)
    for kind in Wd:
        for ex in ("400", "430"):
            ps = [p[ex] for p in Wd[kind] if ex in p]
            if not ps:
                continue
            t = np.concatenate([np.array(p["t_local_mm"]) for p in ps])
            y = np.concatenate([np.array(p["wedge_signal"]) for p in ps])
            s1 = np.concatenate([np.array(p["one_d_signal"]) for p in ps])
            ok = (y > 0) & (s1 > 0)
            lr = np.log(y[ok] / s1[ok]); t = t[ok]; y = y[ok]; s1 = s1[ok]
            rows = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (t >= lo) & (t < hi + (1e-9 if hi >= 2.0 else 0))
                if m.sum() >= 3:
                    rows.append(dict(t_mm=float((lo + hi) / 2), n=int(m.sum()), wedge=float(np.exp(np.mean(np.log(y[m])))),
                                     one_d=float(np.exp(np.mean(np.log(s1[m])))), log_ratio=float(lr[m].mean()),
                                     log_ratio_se=float(lr[m].std(ddof=1) / np.sqrt(m.sum()))))
            allr = np.array([r["log_ratio"] for r in rows])
            out[f"{kind}_{ex}"] = dict(bins=rows, n_px=int(len(lr)), mean_log_ratio=float(lr.mean()),
                                       mean_log_ratio_se=float(lr.std(ddof=1) / np.sqrt(len(lr))),
                                       rms_log_ratio_pixel=float(np.sqrt(np.mean(lr ** 2))),
                                       rms_log_ratio_binned=float(np.sqrt(np.mean(allr ** 2))) if len(allr) else None)
    return out


def fig_wedge(R, figdir, res_dir=None):
    Wd = R.get("wedge") or {}
    if not Wd:
        return
    WB = wedge_binned(R)
    if res_dir is not None:
        (Path(res_dir) / "wedge_binned.json").write_text(json.dumps(WB, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.35))
    colk = {"gentle": OI["blue"], "steep": OI["verm"]}
    for kind in Wd:
        for ex, ls, mk in (("400", "-", "o"), ("430", "--", "s")):
            b = WB.get(f"{kind}_{ex}")
            if not b or not b["bins"]:
                continue
            t = np.array([r["t_mm"] for r in b["bins"]])
            ax[0].plot(t, [r["wedge"] for r in b["bins"]], ls, marker=mk, color=colk[kind], ms=3,
                       label=f"{kind}, {ex} nm")
            ax[0].plot(t, [r["one_d"] for r in b["bins"]], ":", color=colk[kind], lw=1.0)
            lr = np.array([r["log_ratio"] for r in b["bins"]]); se = np.array([r["log_ratio_se"] for r in b["bins"]])
            ax[1].errorbar(t + (0.012 if kind == "steep" else 0) + (0.006 if ex == "430" else 0), np.exp(lr),
                           yerr=[np.exp(lr) - np.exp(lr - se), np.exp(lr + se) - np.exp(lr)], fmt=mk + ls, color=colk[kind],
                           ms=3, lw=0.8, mfc="none" if ex == "430" else colk[kind], capsize=0)
    ax[0].plot([], [], ":", color="0.3", label="uniform-block (1D)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("analytic local cover thickness (mm)")
    ax[0].set_ylabel("buried-layer fluorescence (a.u.)")
    ax[0].set_ylim(1.5e-8, 4e-6)
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.6)
    ax[1].axhline(1, color="0.5", lw=0.6)
    ax[1].set_xlabel("analytic local cover thickness (mm)")
    ax[1].set_ylabel("3D wedge / 1D uniform block")
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.25, 4)
    ax[1].set_yticks([0.25, 0.5, 1, 2, 4]); ax[1].set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax[1].text(0.02, 0.97, "bins of 0.2 mm, pooled seeds;\nbars: $\\pm$1 SE", transform=ax[1].transAxes, fontsize=6.5, va="top")
    for a_, s in zip(ax, "ab"):
        panel_label(a_, f"({s})")
    save(fig, figdir, "E2_wedge")


def make(res_dir, fig_dir):
    style()
    R = json.loads((Path(res_dir) / "results.json").read_text())
    fig_penetration(R, fig_dir)
    fig_fraction_distortion(R, fig_dir)
    fig_blood(R, fig_dir)
    fig_wedge(R, fig_dir, res_dir)
    # file names carry the experiment id; the quick run writes to its own results dir


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results" / "E2"
    fd = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "figures"
    make(rd, fd)
