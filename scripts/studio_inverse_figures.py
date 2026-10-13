#!/usr/bin/env python
"""Figures 24-26 of the SpectralForge Studio v1.2 inverse tools -> docs/studio/figures/.

fig24_crlb_vs_thickness.png  CRLB of the collagen-cover thickness vs the true thickness (lab-replica Nuance camera),
                             empirical SD of the ML estimator, and a thickness map retrieved from a noisy wedge capture.
fig25_optimal_channels.png   information heat map over (excitation x emission band), greedy picks, CRLB(k), exposure allocation.
fig26_unmixing.png           joint-EEM NNLS abundance maps vs explain_pixel ground truth (2-dye layered capture).
Preview-scale runs (scale 0.25, 4 excitations); the surrogate is cached on disk (~/.cache/spectralforge/surrogates)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectraforge.studio import information as info
from spectraforge.studio import inverse as inv
from spectraforge.studio import unmix as um
from spectraforge.studio.capture import capture
from spectraforge.studio.presets import preset
from spectraforge.studio.schema import Experiment
from spectraforge.studio.surrogate import ParamSpec, build_surrogate, simulate_channels, apply_params

OUT = Path(__file__).resolve().parents[1] / "docs" / "studio" / "figures"
L = "/objects/0/shape/layers/2/thickness/"
EXC = [340.0, 365.0, 415.0, 430.0]
PF = 0.15
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.dpi": 200})
C1, C2, C3 = "#1f5f8b", "#d1495b", "#3b8b5a"
numbers: dict = {}


def base(pf=PF):
    d = preset("tissue_under_collagen").model_dump(mode="json")
    d["acquisition"]["volume_mc"]["preview_fraction"] = pf
    d["sources"][0]["sequence"] = [s for s in d["sources"][0]["sequence"] if s["wavelength_nm"] in EXC]
    return Experiment.model_validate(d)


def fig24():
    t0 = time.time()
    exp = base()
    sp = ParamSpec("cover_mm", (L + "thickness_mm", L + "end_thickness_mm"), lo=0.05, hi=1.5)
    ov = {L + "kind": "uniform"}
    sur = build_surrogate(exp, [sp], {"cover_mm": np.linspace(0.05, 1.5, 12)}, overrides=ov, scale=0.25,
                          smooth="gcv", excitations=EXC)
    numbers["surrogate"] = {"channels": sur.channels.n, "n_points": 12, "build_s": sur.meta["build_s"],
                            "loo": sur.loo_error(), "n_pix_effective": sur.noise.n_pix}
    th = np.linspace(0.08, 1.47, 60)
    sd = np.array([info.fisher_from_surrogate(sur, [t]).sd[0] for t in th])
    # empirical SD of the estimator; truth from independent simulator runs at off-grid thicknesses (incl. interpolation error)
    tt = [0.2, 0.45, 0.7, 0.95, 1.2]
    emp, crl, bias = [], [], []
    for t in tt:
        truth = simulate_channels(apply_params(exp, [sp], [t], ov), None, scale=0.25, excitations=EXC).mean_e
        mc = inv.monte_carlo_fit(sur, [t], n_real=200, seed=1, truth_fn=lambda _th, tr=truth: tr)
        emp.append(mc["empirical_sd"][0]); crl.append(mc["crlb_sd"][0]); bias.append(mc["bias"][0])
    numbers["efficiency"] = [{"t": t, "emp_sd_um": e * 1e3, "crlb_um": c * 1e3, "bias_um": b * 1e3, "ratio": e / c}
                             for t, e, c, b in zip(tt, emp, crl, bias)]
    # thickness map from a noisy wedge capture
    dw = base(PF).model_dump(mode="json")
    from spectraforge.studio.sweeps import set_pointer
    for s in dw["sources"][0]["sequence"]:
        pass
    wedge = Experiment.model_validate(dw)
    res = capture(wedge, scale=0.25, excitations=EXC)
    m = inv.retrieve_map(res, sur, tile=3, gt_name="layer_thickness:collagen")
    numbers["map"] = m.stats
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    a = ax[0]
    a.plot(th, sd * 1e3, color=C1, lw=2, label="CRLB (surrogate Fisher)")
    a.plot(tt, np.array(emp) * 1e3, "o", color=C2, ms=6, label="empirical SD, 200 fits")
    a.set_yscale("log"); a.set_xlabel("true cover thickness (mm)"); a.set_ylabel("SD of thickness estimate (µm)")
    a.set_title("Precision limit (CRLB)"); a.legend(frameon=False)
    ok = m.valid & np.isfinite(m.gt)
    g, e, s = m.gt[ok], m.theta[..., 0][ok], m.se[..., 0][ok]
    a = ax[1]
    a.plot([0, 1.5], [0, 1.5], color="0.6", lw=1)
    a.errorbar(g, e, yerr=1.96 * s, fmt="o", ms=2.5, color=C1, alpha=0.5, elinewidth=0.6)
    a.set_xlabel("ground-truth thickness (mm)"); a.set_ylabel("retrieved thickness (mm)")
    st = m.stats
    a.set_title(f"Wedge retrieval: bias {st.get('bias', 0) * 1e3:+.0f} µm, RMSE {st.get('rmse', 0) * 1e3:.0f} µm", fontsize=9)
    a = ax[2]
    im = a.imshow(m.theta[..., 0], cmap="viridis", vmin=0, vmax=1.5)
    a.axis("off"); a.set_title("Thickness map (mm)")
    fig.colorbar(im, ax=a, shrink=0.8)
    fig.savefig(OUT / "fig24_crlb_vs_thickness.png")
    plt.close(fig)
    numbers["fig24_s"] = time.time() - t0
    return sur


def fig25(sur):
    t0 = time.time()
    th = 0.5
    fi = info.fisher_from_surrogate(sur, [th])
    ch = fi.channels
    exs = sorted(set(ch.ex_nm))
    ems = sorted(set(ch.em_nm))
    ipc = fi.info_per_channel()[:, 0]
    k = 8
    d = info.optimal_design(fi, k, "D", budget="exposure_time", exhaustive_check=False)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1, 1]})
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    a = ax[0]
    norm = LogNorm(vmin=max(ipc.max() * 1e-6, ipc[ipc > 0].min()), vmax=ipc.max())
    cmap = plt.get_cmap("magma")
    for i, (e, b) in enumerate(zip(ch.ex_nm, ch.em_nm)):
        a.add_patch(Rectangle((b - 5, exs.index(e) - 0.4), 10, 0.8, color=cmap(norm(max(ipc[i], 1e-30))), lw=0))
    a.set_xlim(min(ems) - 5, max(ems) + 5); a.set_ylim(-0.6, len(exs) - 0.4); a.set_facecolor("0.93")
    a.set_yticks(range(len(exs))); a.set_yticklabels([f"{e:g}" for e in exs])
    for r, i in enumerate(d.selected):
        a.plot(ch.em_nm[i], exs.index(ch.ex_nm[i]), "o", mfc="none", mec="cyan", ms=11, mew=1.5)
        a.text(ch.em_nm[i], exs.index(ch.ex_nm[i]), str(r + 1), color="cyan", ha="center", va="center", fontsize=7)
    a.set_xlabel("emission band (nm)"); a.set_ylabel("excitation (nm)")
    a.set_title("Information on thickness at 0.5 mm; greedy D picks")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a, shrink=0.85)
    cb.set_label("e$^{-2}$ mm$^{-2}$")
    a = ax[1]
    ks = np.arange(1, k + 1)
    a.plot(ks, d.crlb_sd[:, 0] * 1e3, "o-", color=C1, label="information-optimal")
    rng = np.random.default_rng(0)
    rnd = [np.median([info.crlb_for_channels(fi, rng.choice(ch.n, kk, replace=False))[0] for _ in range(100)]) for kk in ks]
    a.plot(ks, np.array(rnd) * 1e3, "s--", color="0.5", label="random (median)")
    a.axhline(fi.sd[0] * 1e3, color=C2, lw=1, label=f"all {ch.n} channels")
    a.set_yscale("log"); a.set_xlabel("number of channels k"); a.set_ylabel("CRLB SD (µm)"); a.legend(frameon=False, fontsize=7)
    a.set_title("CRLB(k)")
    ex = d.exposure
    a = ax[2]
    x = np.arange(len(d.selected))
    a.bar(x - 0.2, np.array(ex["t_reference_s"]) * 1e3, 0.4, color="0.6", label="as acquired")
    a.bar(x + 0.2, np.array(ex["t_opt_s"]) * 1e3, 0.4, color=C3, label="optimised")
    a.set_xticks(x); a.set_xticklabels([f"{ch.ex_nm[i]:g}→{ch.em_nm[i]:g}" for i in d.selected], rotation=60, fontsize=6)
    a.set_ylabel("exposure (ms)"); a.legend(frameon=False, fontsize=7)
    a.set_title(f"Exposure: SD {ex['crlb_sd_reference'][0] * 1e3:.2f}→{ex['crlb_sd_opt'][0] * 1e3:.2f} µm", fontsize=9)
    fig.savefig(OUT / "fig25_optimal_channels.png")
    plt.close(fig)
    numbers["design"] = {"theta_mm": th, "k": k, "selected": [f"{ch.ex_nm[i]:g}->{ch.em_nm[i]:g}" for i in d.selected],
                         "crlb_sd_um_vs_k": (d.crlb_sd[:, 0] * 1e3).tolist(), "random_median_um": (np.array(rnd) * 1e3).tolist(),
                         "all_channels_um": fi.sd[0] * 1e3,
                         "exposure": {"sd_ref_um": ex["crlb_sd_reference"][0] * 1e3, "sd_opt_um": ex["crlb_sd_opt"][0] * 1e3,
                                      "sd_equal_um": ex["crlb_sd_equal"][0] * 1e3}}
    numbers["fig25_s"] = time.time() - t0


def fig26():
    t0 = time.time()
    d = preset("tissue_under_collagen").model_dump(mode="json")
    d["acquisition"]["volume_mc"]["preview_fraction"] = 0.3
    for m in d["materials"]:
        if m["id"] == "tissue_fluor_layer":
            m["dyes"] = [{"fluorophore": "NADH", "concentration_uM": 120.0}]
        if m["id"] == "collagen_cover":
            m["dyes"] = [{"fluorophore": "collagen", "concentration_uM": 600.0}]
        if m["id"] == "tissue_substrate":
            m["dyes"] = []
    d["sources"][0]["sequence"] = [dict(s, exposure_ms=s["exposure_ms"] * 8) for s in d["sources"][0]["sequence"]
                                   if s["wavelength_nm"] in EXC]
    exp = Experiment.model_validate(d)
    res = capture(exp, scale=0.25)
    em = um.endmembers_from_forward_model(exp, res, scale=0.25, n_candidates=24)
    om = um.object_mask(res)
    r = um.unmix(res, em, "nnls", mask=om)
    ys, xs = np.nonzero(om)
    pick = np.random.default_rng(0).choice(xs.size, 120, replace=False)
    px = [(int(xs[i]), int(ys[i])) for i in pick]
    gt = um.gt_component_amounts(exp, res, em, px, scale=0.25)
    est = np.array([r.abundances[y, x] for x, y in px])
    cmp = um.compare_with_gt(est, gt, em.names)
    numbers["unmix"] = {"compare": cmp, "mean_residual_fraction": float(np.nanmean(r.residual_fraction[om])),
                        "endmember_angle_deg": float(um.spectral_angle_deg(em.matrix[:, 0], em.matrix[:, 1]))}
    for meth in ("ls", "fcls"):
        rr = um.unmix(res, em, meth, mask=om)
        e2 = np.array([rr.abundances[y, x] for x, y in px])
        numbers["unmix"][meth] = {k: v["pearson_r"] for k, v in um.compare_with_gt(e2, gt, em.names).items()}
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    for j, nm in enumerate(em.names):
        a = ax[0, j]
        mp = np.where(om, r.abundances[..., j], np.nan)
        im = a.imshow(mp, cmap="viridis"); a.axis("off")
        a.set_title(f"{nm} abundance (e$^-$)"); fig.colorbar(im, ax=a, shrink=0.75)
        b = ax[1, j]
        b.plot(gt[:, j], est[:, j], "o", ms=3.5, color=C1, alpha=0.7)
        lim = [0, max(gt[:, j].max(), est[:, j].max()) * 1.05]
        b.plot(lim, lim, color="0.6", lw=1)
        b.set_xlabel("ground truth (explain_pixel, e$^-$)"); b.set_ylabel("NNLS estimate (e$^-$)")
        b.set_title(f"r = {cmp[nm]['pearson_r']:.3f}, slope {cmp[nm]['slope']:.2f}")
    fig.savefig(OUT / "fig26_unmixing.png")
    plt.close(fig)
    numbers["fig26_s"] = time.time() - t0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sur = fig24()
    fig25(sur)
    fig26()
    (OUT / "inverse_numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    print(json.dumps(numbers, indent=2, default=float))
