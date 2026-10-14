#!/usr/bin/env python
"""Figures and numbers for the inclusion / lateral-transport validation of the 3D layered-block Monte Carlo
(SpectralForge Studio v1.2, docs/studio/PHYSICS_VALIDATION.md section 12.9-12.11).

    python scripts/studio_inclusion_figures.py            # everything (~15 min on an idle 14-core machine)
    python scripts/studio_inclusion_figures.py --quick    # fewer photons / depths (~3 min)
    python scripts/studio_inclusion_figures.py --only radial|perturbation|depth|timing

Writes ``docs/studio/figures/fig22_radial_reflectance.png``, ``fig22b_perturbation.png``,
``fig23_inclusion_depth.png`` and ``docs/studio/figures/inclusion_numbers.json``.  The reference codes (independent
MCML-type pencil-beam Monte Carlo, Farrell-Patterson-Wilson diffusion theory) and the scene builders live in
``tests/studio/test_inclusions_validation.py`` so that the tests and the figures use the very same code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tests.studio import test_inclusions_validation as V  # noqa: E402

OUT = ROOT / "docs" / "studio" / "figures"
NUM = OUT / "inclusion_numbers.json"

C_ENG, C_REF, C_FAR = "#1f5fa8", "#d1495b", "#2a9d8f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "font.family": "DejaVu Sans"})


def _load() -> dict:
    return json.loads(NUM.read_text()) if NUM.exists() else {}


def _save(d: dict) -> None:
    NUM.write_text(json.dumps(d, indent=1, default=float))


# ============================================================================================
# fig22: radially resolved diffuse reflectance
# ============================================================================================
def radial(quick: bool) -> dict:
    mua, musp, n = 0.01, 1.0, 1.4
    res = {}
    for g, nph_eng, nph_ref in ((0.0, 300_000 if quick else 4_000_000, 2_000_000 if quick else 20_000_000),):
        t0 = time.time()
        rho, R, tal = V.engine_radial_reflectance(mua, musp, g, n, nph_eng)
        rc, Rr = V.reference_radial_reflectance(mua, musp, g, n, nph_ref)
        spec = tal.info["specular"]
        print(f"g={g}: engine {time.time() - t0:.0f} s, total diffuse R {tal.diffuse_reflectance:.5f}, "
              f"energy balance {tal.info['energy_balance']:.12f}")
        res[g] = dict(rho=rho, R=R, rc=rc, Rr=Rr, spec=spec, Rtot=tal.diffuse_reflectance, mua=mua, musp=musp, n=n,
                      n_eng=nph_eng, n_ref=nph_ref, balance=tal.info["energy_balance"])
    d = res[0.0]
    edges = np.concatenate([np.arange(0.5, 2.0, 0.25), np.arange(2.0, 4.0, 0.5), np.arange(4.0, 10.01, 1.0)])
    eng = V.annulus_means(d["rho"], d["R"], edges)
    ref = V.ref_annulus_means(d["rc"], d["Rr"], edges)
    far = V.farrell_annulus_means(mua, musp, n, edges) * (1 - d["spec"])
    mid = 0.5 * (edges[1:] + edges[:-1])

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    rr = np.linspace(0.3, 10, 400)
    a.semilogy(rr, V.farrell_reflectance(rr, mua, musp, n) * (1 - d["spec"]), color=C_FAR, lw=1.8,
               label="Farrell, Patterson & Wilson (1992) dipole")
    a.semilogy(mid, ref, "s", ms=4.5, mfc="none", color=C_REF, mew=1.2, label=f"independent MCML-type MC ({d['n_ref']:.0e} ph)")
    a.semilogy(mid, eng, "o", ms=3.5, color=C_ENG, label=f"studio 3D engine ({d['n_eng']:.0e} ph)")
    a.axvspan(0, 2, color="0.5", alpha=0.12, lw=0)
    a.text(1.0, 0.04, "$\\rho<2$ mfp$'$\n(diffusion\ntheory not valid)", transform=a.get_xaxis_transform(), ha="center",
           fontsize=7.5, color="0.35")
    a.set_xlabel("radial distance $\\rho$ from the pencil beam (mm)")
    a.set_ylabel("diffuse reflectance $R(\\rho)$  (mm$^{-2}$ per transmitted power)")
    a.set_xlim(0, 10)
    a.legend(frameon=False, fontsize=8, loc="upper right")
    a.set_title("semi-infinite medium, $\\mu_a=0.01$, $\\mu_s'=1$ mm$^{-1}$, $n=1.4$", fontsize=9)
    b = ax[1]
    b.axhline(1.0, color="k", lw=0.8)
    b.axhspan(0.9, 1.1, color=C_FAR, alpha=0.10, lw=0)
    b.axhspan(0.97, 1.03, color=C_ENG, alpha=0.10, lw=0)
    b.plot(mid, eng / ref, "o-", ms=3.5, lw=1.0, color=C_ENG, label="engine / independent MC")
    b.plot(mid, eng / far, "o-", ms=3.5, lw=1.0, color=C_FAR, label="engine / Farrell")
    b.plot(mid, ref / far, "s--", ms=4, lw=0.9, mfc="none", color=C_REF, label="independent MC / Farrell")
    b.axvspan(0, 2, color="0.5", alpha=0.12, lw=0)
    b.set_xlim(0, 10)
    b.set_ylim(0.6, 1.3)
    b.set_xlabel("$\\rho$ (mm)")
    b.set_ylabel("ratio")
    b.legend(frameon=False, fontsize=8, loc="lower right")
    b.set_title("blue band: $\\pm3$ % (engine vs MC); green band: $\\pm10$ %", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig22_radial_reflectance.png")
    plt.close(fig)

    K = V.RHO_EDGES
    eng2, far2 = V.annulus_means(d["rho"], d["R"]), V.farrell_annulus_means(mua, musp, n) * (1 - d["spec"])
    ref2 = V.ref_annulus_means(d["rc"], d["Rr"])
    return {"edges_mm": K.tolist(), "engine": eng2.tolist(), "reference_mc": ref2.tolist(), "farrell": far2.tolist(),
            "engine_over_reference": (eng2 / ref2).tolist(), "engine_over_farrell": (eng2 / far2).tolist(),
            "reference_over_farrell": (ref2 / far2).tolist(), "R_total_engine": d["Rtot"],
            "specular": d["spec"], "n_photons_engine": d["n_eng"], "n_photons_reference": d["n_ref"],
            "energy_balance": d["balance"], "mua": mua, "musp": musp, "g": 0.0, "n": n}


# ============================================================================================
# fig22b: first-order perturbation
# ============================================================================================
def perturbation(quick: bool) -> dict:
    out = V.perturbation_study(n_photons=2_000_000 if quick else 20_000_000, deltas=(0.01, 0.02, 0.04, 0.08), seeds=(1,) if quick else (1, 2))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    d = np.array(out["deltas"])
    ax.errorbar(d, np.array(out["dR_mean"]) * 1e4, yerr=np.array(out["dR_err"]) * 1e4, fmt="o", color=C_ENG, capsize=3,
                label="MC: $R_{\\rm null}-R_{\\rm perturbed}$ (common random numbers)")
    xs = np.linspace(0, d.max() * 1.05, 20)
    ax.plot(xs, xs * out["slope_pred"] * 1e4, "-", color=C_FAR, label="first-order theory $\\Delta\\mu_a\\,\\int\\Phi\\Phi^{\\dagger}dV$")
    ax.set_xlabel("$\\Delta\\mu_a$ of the sphere (mm$^{-1}$)")
    ax.set_ylabel("decrease of diffuse reflectance  ($10^{-4}$)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("nine absorbing spheres ($r=1.5$ mm, 3 mm deep), $\\mu_a=0.1$, $\\mu_s'=2$ mm$^{-1}$", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig22b_perturbation.png")
    plt.close(fig)
    return out


# ============================================================================================
# fig23: buried fluorescent sphere
# ============================================================================================
def depth(quick: bool) -> dict:
    from scipy.ndimage import gaussian_filter

    from spectraforge.studio import block_capture, capture

    depths = [0.5, 1.0, 2.0, 3.0] if quick else [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    show = [0.5, 1.0, 2.0, 3.0] if quick else [0.5, 1.5, 3.0, 4.0]
    scats = [0.5, 1.5, 3.0]
    photons = 400_000 if quick else 3_000_000
    images, table = {}, {}
    for mp in scats:
        table[mp] = []
        for dpt in depths:
            block_capture.clear_cache()
            t0 = time.time()
            res = capture.capture(V.buried_bead_experiment(mp, dpt, photons=photons), scale=0.5)
            on, ring, cm, pitch = V.bead_signal(res)
            ex = res.excitations_nm[0]
            wl = res.emission_nm[ex]
            img = res.radiance[ex][..., (wl > 505.0) & (wl < 565.0)].sum(-1)
            mask = res.ground_truth["inclusion_mask:bead"]
            ys, xs = np.nonzero(mask == 1)
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            of = float(np.nanmean(res.ground_truth["origin_fraction:bead"][mask == 1]))
            table[mp].append(dict(depth=dpt, on=on, ring=ring, excess=on - ring, contrast=cm, origin_fraction=of))
            print(f"mus'={mp} d={dpt}: on {on:.3e} ring {ring:.3e} C {cm:.3f} ({time.time() - t0:.0f} s)", flush=True)
            if dpt in show:
                pad = int(round(8.0 / pitch))
                sm = gaussian_filter(img, 1.0) - ring
                images[(mp, dpt)] = sm[max(cy - pad, 0):cy + pad + 1, max(cx - pad, 0):cx + pad + 1]
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = fig.add_gridspec(len(scats) + 1, len(show), height_ratios=[1] * len(scats) + [1.1], hspace=0.35, wspace=0.08)
    for i, mp in enumerate(scats):
        vmax = max(float(images[(mp, show[0])].max()), 1e-30)
        for j, dpt in enumerate(show):
            ax = fig.add_subplot(gs[i, j])
            im = images[(mp, dpt)]
            h = im.shape[0] * 0.5 * 30.0 / max(im.shape[0], 1)
            ax.imshow(im, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"bead top {dpt:g} mm deep", fontsize=9)
            if j == 0:
                ax.text(-0.08, 0.5, f"$\\mu_s'={mp:g}$ mm$^{{-1}}$", transform=ax.transAxes, rotation=90, va="center",
                        ha="right", fontsize=9)
            c = im.shape[0] // 2
            ax.text(0.04, 0.05, f"centre {100 * im[c - 1:c + 2, c - 1:c + 2].mean() / vmax:.0f} %", transform=ax.transAxes,
                    fontsize=8, color="k")
    fig.text(0.5, 0.955, "excess radiance over the surrounding ring (505-565 nm, smoothed 1 px), each row scaled to its shallowest bead; "
             "16 mm field", ha="center", fontsize=8, color="0.3")
    gsb = gs[len(scats), :].subgridspec(1, 2, wspace=0.3)
    cols = {0.5: "#6baed6", 1.5: "#2171b5", 3.0: "#08306b"}
    a1, a2 = (fig.add_subplot(gsb[0, k]) for k in range(2))
    for mp in scats:
        dd = [r["depth"] for r in table[mp]]
        e0 = table[mp][0]["excess"]
        a1.plot(dd, [r["excess"] / e0 for r in table[mp]], "o-", color=cols[mp], ms=4, label=f"$\\mu_s'={mp:g}$ mm$^{{-1}}$")
        a2.plot(dd, [r["contrast"] for r in table[mp]], "o-", color=cols[mp], ms=4)
    a1.axhline(0, color="k", lw=0.6)
    a2.axhline(0, color="k", lw=0.6)
    a1.set_ylabel("excess radiance / shallowest")
    a2.set_ylabel("Michelson contrast (bead vs ring)")
    for a in (a1, a2):
        a.set_xlabel("depth of the bead top (mm)")
    a1.legend(frameon=False, fontsize=8)
    fig.suptitle("Buried fluorescein bead ($r=1$ mm) in a non-fluorescent turbid host, 490 nm excitation, 3D MC with voxel inclusions",
                 fontsize=10, y=0.995)
    fig.savefig(OUT / "fig23_inclusion_depth.png", bbox_inches="tight")
    plt.close(fig)
    return {str(mp): rows for mp, rows in table.items()}


def _fwhm(profile: np.ndarray, pitch: float) -> float:
    p = np.asarray(profile, float)
    p = p - np.nanmin(p)
    k = int(np.nanargmax(p))
    half = 0.5 * p[k]
    lo = k
    while lo > 0 and p[lo] > half:
        lo -= 1
    hi = k
    while hi < len(p) - 1 and p[hi] > half:
        hi += 1
    return float((hi - lo) * pitch)


# ============================================================================================
# timing
# ============================================================================================
def timing(quick: bool) -> dict:
    return V.timing_study(n_photons=300_000 if quick else 2_000_000, reps=2 if quick else 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["radial", "perturbation", "depth", "timing"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nums = _load()
    steps = {"radial": radial, "perturbation": perturbation, "depth": depth, "timing": timing}
    for k, fn in steps.items():
        if a.only and a.only != k:
            continue
        print(f"=== {k}", flush=True)
        res = fn(a.quick)
        nums = _load()                     # re-read: other runs may have written meanwhile
        nums[k] = res
        _save(nums)


if __name__ == "__main__":
    main()
