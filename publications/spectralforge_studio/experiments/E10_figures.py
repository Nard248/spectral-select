"""Figures of E10 (performance and convergence)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import E2_figures as F   # noqa: E402

plt = F.plt
OI = F.OI


def make(res_dir, fig_dir):
    F.style()
    res_dir = Path(res_dir)
    R = json.loads((res_dir / "results.json").read_text())
    conv = R["convergence"]
    col = {"peel": OI["blue"], "cone_raw": OI["verm"], "cone_prod": OI["orange"]}
    lab = {"peel": "peel-off", "cone_raw": "cone (raw)", "cone_prod": "cone (engine smoothing)"}

    def series(k, key):
        tab = conv[k]["table"]
        N = np.array(sorted(tab, key=float), float)
        return N, np.array([tab[str(int(n))][key] for n in N])

    fig, ax = plt.subplots(2, 2, figsize=(F.W2, 5.6), gridspec_kw=dict(wspace=0.36, hspace=0.5))
    # (a) runtime
    a = ax[0, 0]
    for k in ("peel", "cone_raw"):
        N, t = series(k, "mc3d_s")
        lf = conv[k]["runtime_linear_fit"]
        a.plot(N, t, "o", color=col[k], label=f"{lab[k]}: {lf['s_per_1e5_packets']:.1f} s / $10^5$ packets")
        a.plot(N, lf["overhead_s"] + lf["s_per_1e5_packets"] * N / 1e5, "-", color=col[k], lw=0.8)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("packets per pass $N$")
    a.set_ylabel("block Monte Carlo time (s)")
    a.legend(frameon=False, loc="upper left", fontsize=7)
    F.panel_label(a, "(a)")
    # (b) noise
    b = ax[0, 1]
    for k, mk in (("peel", "o"), ("cone_raw", "o")):
        N, px = series(k, "pix_rel_sd")
        s = conv[k]["slope_pix_rel_sd"]
        b.plot(N, px, mk, color=col[k], label=f"{lab[k]}: {s['value']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]")
        b.plot(N, px[0] * (N / N[0]) ** -0.5, "-", color=col[k], lw=0.6, alpha=0.6)
    N, se = series("peel", "rel_se_roi")
    s = conv["peel"]["slope_rel_se_roi"]
    b.plot(N, se, "s", color=col["peel"], mfc="none", label=f"peel-off, ROI mean: {s['value']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("packets per pass $N$")
    b.set_ylabel("relative MC noise")
    b.set_ylim(top=3.0, bottom=2e-3)
    b.legend(frameon=False, fontsize=6.3, loc="lower left")
    b.text(0.97, 0.96, r"lines: $N^{-1/2}$; slope [95 % CI]", transform=b.transAxes, ha="right", va="top", fontsize=6.5, color="0.4")
    F.panel_label(b, "(b)")
    # (c) threads
    c = ax[1, 0]
    th = R.get("thread_scaling")
    if th:
        n = np.array(sorted(th["table"], key=float), float)
        T = [th["table"][str(int(k))] for k in n]
        sp = np.array([t["speedup_median"] for t in T])
        lo = np.array([t["speedup_iqr"][0] for t in T])
        hi = np.array([t["speedup_iqr"][1] for t in T])
        c.plot(n, n, ":", color="0.5", label="ideal")
        c.errorbar(n, sp, yerr=[sp - lo, hi - sp], fmt="o-", color=OI["blue"], capsize=2, label="measured (median, IQR)")
        for x, y, t in list(zip(n, sp, T))[1:]:
            c.annotate(f"{t['efficiency_median']:.2f}", (x, y), textcoords="offset points", xytext=(5, -11) if t["efficiency_median"] < 0.999 else (8, 4), fontsize=6.5, color="0.3")
        c.set_xlabel("numba threads")
        c.set_ylabel("speed-up vs 1 thread")
        c.set_xscale("log", base=2)
        c.set_yscale("log", base=2)
        c.set_xticks(n)
        c.set_xticklabels([f"{int(k)}" for k in n])
        c.set_yticks([1, 2, 4, 8, 16])
        c.set_yticklabels(["1", "2", "4", "8", "16"])
        c.legend(frameon=False, loc="upper left")
        c.text(0.97, 0.05, "labels: parallel efficiency", transform=c.transAxes, ha="right", fontsize=6.5, color="0.4")
    F.panel_label(c, "(c)")
    # (d) preview vs full
    d = ax[1, 1]
    p = res_dir / "agreement_maps.npz"
    if p.exists():
        z = np.load(p)
        m = z["mask"]
        pv, fb, fb1 = z["preview"][m], z["full_binned"][m], z["full_binned_seed1"][m]
        d.loglog(fb, fb1, ".", ms=2.5, color=OI["orange"], alpha=0.6, label="full, other seed")
        d.loglog(fb, pv, ".", ms=2.5, color=OI["blue"], alpha=0.6, label="preview")
        lo_, hi_ = min(fb.min(), pv.min()) * 0.7, max(fb.max(), pv.max()) * 1.4
        d.plot([lo_, hi_], [lo_, hi_], "-", color="0.4", lw=0.6)
        d.set_xlabel("full capture (4$\\times$4 binned)")
        d.set_ylabel("comparison capture")
        pa = R["preview_vs_full"]
        d.text(0.04, 0.96, f"RMS rel. diff.\npreview/full {pa['rms_rel_preview_vs_fullbinned']:.2f}\nfull/full {pa['rms_rel_fullbinned_seed0_vs_seed1']:.2f}",
               transform=d.transAxes, va="top", fontsize=6.8)
        d.legend(frameon=False, loc="lower right", fontsize=6.8)
    F.panel_label(d, "(d)")
    F.save(fig, fig_dir, "E10_performance")

    # estimators + inclusion overhead
    eff = R["estimator_comparison"]
    N = np.array(sorted(eff, key=float), float)
    g = lambda key: np.array([eff[str(int(n))][key] for n in N])
    fig, ax = plt.subplots(1, 3, figsize=(F.W2, 2.5), gridspec_kw=dict(wspace=0.5))
    ax[0].semilogx(N, g("pix_sd_ratio_cone_raw_over_peel"), "o-", color=OI["blue"], label="pixel noise, cone / peel")
    ax[0].semilogx(N, 1 / g("time_ratio_cone_raw_over_peel"), "^-", color=OI["green"], label="time, peel / cone")
    ax[0].set_ylim(-0.2, 4.2)
    ax[0].axhline(1, color="0.6", lw=0.6)
    ax[0].set_xlabel("packets per pass $N$")
    ax[0].set_ylabel("ratio at equal packets")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    ax[1].semilogx(N, g("efficiency_peel_over_cone_raw_pix"), "o-", color=OI["blue"], label="per-pixel noise")
    ax[1].axhline(1, color="0.6", lw=0.6)
    ax[1].set_xlabel("packets per pass $N$")
    ax[1].set_ylabel("efficiency gain of peel-off,\n$\\sigma^2 T$ (cone) / $\\sigma^2 T$ (peel)")
    ax[1].set_ylim(0, 6.5)
    ax[1].legend(frameon=False, fontsize=6.5, loc="lower left")
    inc = R.get("inclusion_overhead")
    if inc:
        names = [("sphere_null", "host-filled\nsphere"), ("vessel", "vessel\n$r$=1 mm"), ("sphere", "sphere\n$r$=1 mm")]
        x = np.arange(len(names))
        med = np.array([inc["ratio_vs_none"][k]["median"] for k, _ in names])
        lo = np.array([inc["ratio_vs_none"][k]["iqr"][0] for k, _ in names])
        hi = np.array([inc["ratio_vs_none"][k]["iqr"][1] for k, _ in names])
        ax[2].bar(x, med, 0.55, color=[OI["sky"], OI["purple"], OI["verm"]], yerr=[med - lo, hi - med], capsize=2)
        ax[2].axhline(1.6, color="k", lw=0.8, ls="--")
        ax[2].text(-0.45, 1.66, "target 1.6", ha="left", fontsize=6.8)
        for xi, m_ in zip(x, med):
            ax[2].text(xi, 0.06, f"{m_:.2f}", ha="center", color="w", fontsize=7.5, fontweight="bold")
        ax[2].axhline(1, color="0.6", lw=0.6)
        ax[2].set_xticks(x)
        ax[2].set_xticklabels([n for _, n in names], fontsize=7)
        ax[2].set_ylabel("time ratio vs no inclusion")
        ax[2].set_ylim(0, 2.3)
    for a_, s in zip(ax, "abc"):
        F.panel_label(a_, f"({s})")
    F.save(fig, fig_dir, "E10_estimators_inclusions")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    make(here / "results" / "E10", here / "figures")
