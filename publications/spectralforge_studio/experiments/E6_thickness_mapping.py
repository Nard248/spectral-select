"""E6: cover-thickness mapping on two wedge scenes (SpectralForge Studio paper).

Scenes: (i) tissue-optics wedge (E5 scene, collagen cover 0.1 -> 1.4 mm along x over the NADH/FAD layer, 8 TLS lines x 31
LCTF bands), (ii) the generic layered wedge phantom (fluorescein layer under an Intralipid-like cover, 2 -> 0 mm; 6 lines).
Per-tile retrieval (`inverse.retrieve_map`, tile 4 x 4 output pixels, free multiplicative amplitude = unknown illumination)
with the 1-D thickness surrogate of the same scene (same simulator, uniform-thickness training grid, so the only model error is
the lateral wedge gradient, the MC noise of the surrogate grid and the interpolation).  Truth = the simulator's
`layer_thickness:cover` map.  3 exposure levels (relative to the nominal protocol, signal and dark scale, read noise does not),
two channel sets (all channels; k = 8 information-optimal for the amplitude-free thickness problem at eps = 2 %) plus 5 random
k = 8 draws, and two noise models for the fit (photon-limited weights/SE, eps = 2 % model-error weights/SE).
Reported: bias, RMSE, coverage of the nominal 95 % interval (|err| < 1.96 SE) overall and per thickness bin.

Run: .venv/bin/python publications/spectralforge_studio/experiments/E6_thickness_mapping.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import C3_common as C  # noqa: E402
from spectraforge.studio import information as info  # noqa: E402
from spectraforge.studio import inverse as inv  # noqa: E402

LEVELS = [0.003, 0.03, 1.0]
EPS_FIT = [0.0, 0.02]
TILE = 4
K = 8
N_RAND = 5
BINS = [0.0, 0.5, 1.0, 2.1]      # thickness bins (mm) for bias/RMSE/coverage vs depth


def channel_sets(sur, quick: bool):
    fi0 = info.fisher_from_surrogate(sur, [0.7])
    fa = C.with_eps_fisher(C.augment(fi0, "amp"), 0.02)
    k = 4 if quick else K
    opt = np.zeros(sur.channels.n, bool)
    opt[list(info.optimal_design(fa, k, "target:cover_mm", "channels", exhaustive_check=False).selected)] = True
    rng = np.random.default_rng(5)
    rnd = []
    for _ in range(2 if quick else N_RAND):
        m = np.zeros(sur.channels.n, bool)
        m[rng.choice(sur.channels.n, k, replace=False)] = True
        rnd.append(m)
    return {"all": np.ones(sur.channels.n, bool), f"opt{k}": opt, **{f"rand{k}_{i}": m for i, m in enumerate(rnd)}}


def score(mr, bins=BINS):
    ok = mr.valid & np.isfinite(mr.gt)
    lo, hi = None, None
    t = mr.theta[..., 0][ok]
    g = mr.gt[ok]
    s = mr.se[..., 0][ok]
    use = np.isfinite(t) & (g >= 0.05) & (g <= 1.5) & np.isfinite(s)
    t, g, s = t[use], g[use], s[use]
    e = t - g
    out = {"n": int(e.size), "bias_um": float(e.mean() * 1e3), "rmse_um": float(np.sqrt(np.mean(e ** 2)) * 1e3),
           "median_abs_err_um": float(np.median(np.abs(e)) * 1e3), "mean_se_um": float(s.mean() * 1e3),
           "coverage_95": float(np.mean(np.abs(e) < 1.96 * s)), "bins": []}
    # bootstrap CI of rmse / coverage over tiles
    rng = np.random.default_rng(0)
    idx = rng.integers(0, e.size, (300, e.size))
    r = np.sqrt((e[idx] ** 2).mean(axis=1)) * 1e3
    c = (np.abs(e[idx]) < 1.96 * s[idx]).mean(axis=1)
    out["rmse_um_ci"] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    out["coverage_95_ci"] = [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))]
    for a, b in zip(bins[:-1], bins[1:]):
        m = (g >= a) & (g < b)
        if m.sum() >= 3:
            out["bins"].append({"lo_mm": a, "hi_mm": b, "n": int(m.sum()), "bias_um": float(e[m].mean() * 1e3),
                                "rmse_um": float(np.sqrt(np.mean(e[m] ** 2)) * 1e3), "coverage_95": float(np.mean(np.abs(e[m]) < 1.96 * s[m])),
                                "mean_se_um": float(s[m].mean() * 1e3)})
    return out, (g, t, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    T0 = time.time()
    C.style()
    levels = [0.03, 1.0] if q else LEVELS
    out = {"quick": q, "levels": levels, "eps_fit": EPS_FIT, "tile": TILE, "bins_mm": BINS,
           "units": {"bias_um": "um", "rmse_um": "um", "coverage_95": "fraction"}, "scenes": {}}
    scat = {}
    for kind, builder in (("tissue", C.tissue_surrogate), ("phantom", C.phantom_surrogate)):
        exp0, sp, grid, sur = builder(q)
        sets = channel_sets(sur, q)
        print(f"[{kind}] surrogate {sur.channels.n} ch, {time.time() - T0:.0f} s", flush=True)
        sc = {"n_channels": int(sur.channels.n), "grid_mm": grid, "sets": {k: int(m.sum()) for k, m in sets.items()},
              "opt_labels": [sur.channels.labels()[i] for i in np.flatnonzero(sets[[k for k in sets if k.startswith("opt")][0]])],
              "results": {}}
        for lev in levels:
            cap = C.get_capture(kind, lev, q)
            print(f"[{kind}] capture level {lev:g}: {time.time() - T0:.0f} s", flush=True)
            sl = C.scale_surrogate(sur, lev)
            for eps in EPS_FIT:
                se_ = C.with_eps(sl, eps)
                for name, mask in sets.items():
                    mr = inv.retrieve_map(cap, se_.subset(mask), tile=TILE, gt_name="layer_thickness:cover", kind="adu")
                    st, pts = score(mr)
                    sc["results"].setdefault(f"L{lev:g}", {}).setdefault(f"eps{eps:g}", {})[name] = st
                    if name in ("all", [k for k in sets if k.startswith("opt")][0]):
                        scat[(kind, lev, eps, name)] = pts
                print(f"[{kind}] L{lev:g} eps{eps:g}: {time.time() - T0:.0f} s", flush=True)
        # aggregate the random draws
        for lk, d in sc["results"].items():
            for ek, dd in d.items():
                rs = [v for k, v in dd.items() if k.startswith("rand")]
                dd["rand_mean"] = {"rmse_um": float(np.mean([r["rmse_um"] for r in rs])), "rmse_um_sd": float(np.std([r["rmse_um"] for r in rs])),
                                   "bias_um": float(np.mean([r["bias_um"] for r in rs])), "coverage_95": float(np.mean([r["coverage_95"] for r in rs])),
                                   "n_draws": len(rs)}
        # photon-limited CRLB of the same configurations for context (tile-level, n_pix = TILE^2)
        sc["crlb_um_at_0.7mm"] = {}
        for lev in levels:
            sl = C.scale_surrogate(sur, lev)
            sl = sl.__class__(sl.params, sl.theta, sl.values_e, sl.channels, sl.noise.with_n_pix(TILE * TILE), mode=sl.mode,
                              grid_axes=sl.grid_axes, interp=sl.interp, smooth=sl.smooth, meta=sl.meta)
            f = C.augment(info.fisher_from_surrogate(sl, [0.7]), "amp")
            sc["crlb_um_at_0.7mm"][f"L{lev:g}"] = {f"eps{e:g}": float(C.with_eps_fisher(f, e).sd[0] * 1e3) for e in EPS_FIT}
        out["scenes"][kind] = sc
    out["elapsed_s"] = time.time() - T0
    C.write_results("E6" if not q else "E6_quick", out)
    make_figure(out, scat, q)
    print(f"done {time.time() - T0:.0f} s", flush=True)


def make_figure(out, scat, q):
    import matplotlib.pyplot as plt
    levels = out["levels"]
    fig, ax = plt.subplots(1, 4, figsize=(C.DCOL, 2.2), gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
    ct = {"all": C.OI["blue"], "opt": C.OI["verm"], "rand": C.OI["green"]}
    lev_mid = 1.0
    # (a) scatter, tissue + phantom, all channels, highest SNR, eps=0.02
    a = ax[0]
    for kind, col, mk in (("tissue", C.OI["blue"], "o"), ("phantom", C.OI["orange"], "s")):
        g, t, s = scat[(kind, lev_mid, 0.02, "all")]
        a.plot(g, t, mk, ms=1.8, color=col, alpha=0.6, mew=0, label=kind)
    a.plot([0, 2], [0, 2], "k-", lw=0.6)
    a.set_xlim(0, 2.05); a.set_ylim(0, 2.05)
    a.set_xlabel("true cover (mm)"); a.set_ylabel("retrieved cover (mm)"); a.set_title("(a) all channels, level 1", fontsize=8)
    a.legend(fontsize=7, loc="upper left", markerscale=2)
    for j, (key, ttl, yl) in enumerate((("rmse_um", "(b) RMSE", "RMSE (µm)"), ("coverage_95", "(c) 95 % CI coverage", "coverage"),
                                        ("bias_um", "(d) bias", "bias (µm)"))):
        a = ax[1 + j]
        for kind, ls in (("tissue", "-"), ("phantom", "--")):
            sc = out["scenes"][kind]
            optname = [k for k in sc["sets"] if k.startswith("opt")][0]
            for eps, mk in ((0.0, "o"), (0.02, "s")):
                for name, col in (("all", ct["all"]), (optname, ct["opt"])):
                    v = [sc["results"][f"L{l:g}"][f"eps{eps:g}"][name][key] for l in levels]
                    a.plot(levels, v, ls=ls, marker=mk, ms=3, color=col, mfc=col if eps == 0 else "none", lw=0.9)
        a.set_xscale("log"); a.set_xlabel("exposure level"); a.set_ylabel(yl); a.set_title(ttl, fontsize=8)
        if key == "rmse_um":
            a.set_yscale("log")
        if key == "coverage_95":
            a.axhline(0.95, color="0.5", lw=0.6, ls=":"); a.set_ylim(0, 1.02)
        if key == "bias_um":
            a.axhline(0, color="0.5", lw=0.6)
    h = [plt.Line2D([], [], color=ct["all"], label="all ch."), plt.Line2D([], [], color=ct["opt"], label="$k$ = 8 optimal"),
         plt.Line2D([], [], color="0.3", ls="-", label="tissue"), plt.Line2D([], [], color="0.3", ls="--", label="phantom"),
         plt.Line2D([], [], color="0.3", marker="o", ls="", label="fit $\\epsilon$ = 0"),
         plt.Line2D([], [], color="0.3", marker="s", mfc="none", ls="", label="fit $\\epsilon$ = 2 %")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=0.7)
    C.savefig(fig, "E6_thickness_mapping" + ("_quick" if q else ""))


if __name__ == "__main__":
    main()
