#!/usr/bin/env python
"""Regenerate every figure and number of ``docs/studio/PHYSICS_VALIDATION.md``.

    PYTHONPATH=src python scripts/studio_validation_figures.py [--only mcml,diffusion,...] [--quick]

Writes PNGs into ``docs/studio/figures/`` and ``validation_numbers.json`` next to them.  Everything is
computed with the studio engine itself (transport, tracer, capture, sensor); nothing is hard-coded except
the literature reference values.  ``--quick`` lowers the Monte Carlo photon counts (a few minutes -> ~1 min).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectraforge.studio import capture as cap
from spectraforge.studio import optics as optics_mod
from spectraforge.studio import presets, sensor
from spectraforge.studio import transport as T
from spectraforge.studio import transport_mc as mc
from spectraforge.studio.contracts import SPECTRAL_GRID_NM as GRID
from spectraforge.studio.schema import (
    Background, BeamSpec, BoxShape, CameraSpec, CuvetteShape, DyeAmount, ExcitationStep, Experiment,
    LensSpec, LightSourceSpec, MaterialSpec, PhysicsToggles, SceneObject, SensorSpec, SpectralFilterSpec,
    Transform,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "studio" / "figures"
NUM: dict = {}
QUICK = False

# ---------------------------------------------------------------- plot style
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "lines.linewidth": 1.8, "font.family": "DejaVu Sans",
})
# Journal-paper style (7.16 in wide figures, 7-8 pt text at print size), used by the figures built for the Studio manuscript.
PAPER_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
            "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
            "legend.fontsize": 7, "axes.grid": False, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.dpi": 300,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6}
C = {"a": "#1f6fb4", "b": "#d9541e", "c": "#2a9d4b", "d": "#7a3fb0", "k": "#333333", "g": "#8a8a8a"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


NO_NOISE = PhysicsToggles(shot_noise=False, read_noise=False, dark_current=False, fixed_pattern_noise=False,
                          quantization=False, saturation=False, power_drift=False, defocus=False,
                          diffraction=False, chromatic_aberration=False, vignetting=False,
                          filter_angle_shift=False, subsurface_spread=False)


def _white_mc(g, n, no, mu_a, mu_s, d, mu_int=1.0, n_photons=1_000_000):
    """Diffuse reflectance / transmittance / absorption of the baked 'white' Monte Carlo tables."""
    tau = np.array([mu_s * d])
    u = np.sin(np.pi * np.arange(41) / 80) ** 2
    o = mc.bake_arrays(g, n, no, np.array([mu_int]), tau, u, T.ZINF_EDGES, T.RAD_EDGES, n_photons, 7)
    e = T._bin_weights(np.array([mu_a / mu_s]))[:, 0]
    with np.errstate(all="ignore"):
        rd = float((o["rtop_fin"][0, 0].astype(float) @ e).sum())
        tt = float(o["tt_fin"][0, 0].astype(float) @ e)
        a = (mu_a / mu_s) * float((o["m_fin"][0, 0].astype(float) @ e).sum())
    return rd, tt, a


def _semi_inf(g, n, mu_a, mu_s, n_photons):
    u = np.sin(np.pi * np.arange(41) / 80) ** 2
    o = mc.bake_arrays(g, n, 1.0, np.array([1.0]), T.TAU_NODES[:1], u, T.ZINF_EDGES, T.RAD_EDGES, n_photons, 3)
    e = T._bin_weights(np.array([mu_a / mu_s]))[:, 0]
    with np.errstate(all="ignore"):
        mass = o["m_inf"][0].astype(float) @ e
        rd = float((o["rtop_inf"][0].astype(float) @ e).sum())
    return mass, rd


# ================================================================ 1. MCML benchmark table
def fig_mcml():
    npho = 4_000_000 if QUICK else 64_000_000
    rows = []
    r = mc.mcml_slab(1.0, 9.0, 0.75, 1.0, 1.0, 0.2, n_photons=npho, seed=3)
    rows += [("MCML slab (mua=1, mus=9, g=0.75, n=1, d=0.2 mm)", "Rd", r["Rd"], 0.09734, "Wang & Jacques 1995 / Prahl"),
             ("", "Tt", r["Tt"], 0.66096, ""),
             ("", "R+T+A", r["Rd"] + r["Tt"] + r["A"], 1.0, "energy conservation")]
    rd, tt, a = _white_mc(0.75, 1.0, 1.0, 1.0, 9.0, 0.2, n_photons=1_000_000 if QUICK else 4_000_000)
    rows += [("Baked 'white' tables, same slab", "Rd", rd, 0.09734, "as above"),
             ("", "Tt", tt, 0.66096, "")]
    n, mua, d = 1.5, 0.8, 1.0
    R = ((n - 1) / (n + 1)) ** 2
    e = np.exp(-mua * d)
    t_ref = (1 - R) ** 2 * e / (1 - R ** 2 * e ** 2)
    rd_ref = (1 - R) ** 2 * R * e ** 2 / (1 - R ** 2 * e ** 2)
    r2 = mc.mcml_slab(mua, 1e-9, 0.0, n, 1.0, d, n_photons=npho // 4, seed=2)
    rows += [("Index-mismatched clear slab (n=1.5, mua=0.8/mm, d=1 mm)", "Tt", r2["Tt"], t_ref, "incoherent multiple reflections, exact"),
             ("", "Rd", r2["Rd"], rd_ref, "")]
    # white MC vs conventional MCML, mismatched oblique turbid slab
    n_, g_, mua_, mus_, d_, mi = 1.4, 0.8, 0.5, 10.0, 0.6, 0.8
    rd3, tt3, a3 = _white_mc(g_, n_, 1.0, mua_, mus_, d_, mu_int=mi, n_photons=1_000_000 if QUICK else 4_000_000)
    mu_ext = np.sqrt(1 - n_ ** 2 * (1 - mi ** 2))
    ref = mc.mcml_slab(mua_, mus_, g_, n_, 1.0, d_, mu0_ext=mu_ext, n_photons=npho // 2, seed=9)
    f = 1 - ref["Rsp"]
    rows += [("White tables vs conventional MCML (n=1.4, oblique, turbid)", "Rd", rd3, ref["Rd"] / f, "independent code path"),
             ("", "Tt", tt3, ref["Tt"] / f, ""),
             ("", "A", a3, ref["A"] / f, "")]
    # dipole
    for nn, g, ap in ((1.0, 0.9, 0.99), (1.4, 0.9, 0.99)):
        musp = 1.0
        mus = musp / (1 - g)
        mua_d = musp * (1 - ap) / ap
        _, rdm = _semi_inf(g, nn, mua_d, mus, 400_000 if QUICK else 1_500_000)
        rdi = -1.440 / nn ** 2 + 0.710 / nn + 0.668 + 0.0636 * nn if nn != 1.0 else 0.0
        A = (1 + rdi) / (1 - rdi)
        s3 = np.sqrt(3 * (1 - ap))
        dip = 0.5 * ap * (1 + np.exp(-4 / 3 * A * s3)) * np.exp(-s3)
        rows.append((f"Semi-infinite diffuse reflectance, a'={ap}, n={nn}", "Rd", rdm, dip, "Farrell/Patterson albedo formula"))
    NUM["mcml_table"] = [dict(case=c, quantity=q, engine=float(v), reference=float(rf), note=nt,
                              rel_err=float(v / rf - 1)) for c, q, v, rf, nt in rows]
    fig, ax = plt.subplots(figsize=(10.6, 0.62 * len(rows) + 1.1))
    ax.axis("off")
    import textwrap
    cells = [[textwrap.fill(c, 42), q, f"{v:.5f}", f"{rf:.5f}", f"{100 * (v / rf - 1):+.2f} %", nt] for c, q, v, rf, nt in rows]
    tb = ax.table(cellText=cells, colLabels=["case", "quantity", "engine", "reference", "difference", "reference source"],
                  loc="center", cellLoc="left", colWidths=[0.36, 0.07, 0.09, 0.09, 0.09, 0.30])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 2.1)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if i == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(weight="bold")
    save(fig, "fig01_mcml_table.png")


# ================================================================ 2. diffusion limit
def fig_diffusion():
    from scipy.optimize import least_squares
    g, n, mua, musp = 0.9, 1.0, 0.01, 1.0
    mus = musp / (1 - g)
    mass, _ = _semi_inf(g, n, mua, mus, 1_000_000 if QUICK else 3_000_000)
    mu_eff = np.sqrt(3 * mua * (mua + musp))
    edges = T.ZINF_EDGES / mus
    mid = 0.5 * (edges[1:] + edges[:-1])
    dens = mass / np.diff(edges)
    sel = (edges[:-1] > 5.0 / musp) & (edges[1:] < 5.0 / mu_eff)
    lo, hi, m = edges[:-1][sel], edges[1:][sel], mass[sel]

    def resid(p):
        a, mu = p
        return (a * (np.exp(-mu * lo) - np.exp(-mu * hi)) / mu - m) / np.sqrt(np.maximum(m, 1e-12))
    sol = least_squares(resid, [m.sum() * mu_eff, mu_eff])
    a_fit, mu_fit = sol.x
    NUM["diffusion"] = dict(mu_eff_fit=float(mu_fit), mu_eff_diffusion=float(mu_eff), rel_err=float(mu_fit / mu_eff - 1),
                            mua=mua, musp=musp, g=g)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ok = dens > 0
    ax.semilogy(mid[ok], dens[ok], "o", ms=3.2, color=C["a"], label="Monte Carlo (white tables)")
    z = np.linspace(5.0 / musp, mid[ok].max(), 100)
    ax.semilogy(z, a_fit * np.exp(-mu_fit * z), "-", color=C["b"], label=f"fit: $\\mu_{{eff}}$ = {mu_fit:.4f} mm$^{{-1}}$")
    ax.semilogy(z, a_fit * np.exp(-mu_eff * z), "--", color=C["k"], label=f"diffusion: $\\sqrt{{3\\mu_a(\\mu_a+\\mu_s')}}$ = {mu_eff:.4f} mm$^{{-1}}$")
    ax.axvline(5.0 / musp, color=C["g"], lw=0.8)
    ax.text(5.0 / musp + 0.1, ax.get_ylim()[1] * 0.3, "fit starts\n5 transport mfp", fontsize=8, color=C["g"])
    ax.set_xlabel("depth z (mm)")
    ax.set_ylabel("absorbed-weight density (a.u. / mm)")
    ax.set_title(f"Diffusion limit: $\\mu_a$={mua}, $\\mu_s'$={musp} /mm, g={g}")
    ax.legend(fontsize=8)
    save(fig, "fig02_diffusion_decay.png")


# ================================================================ experiment builders
def _fl_material(c_uM, ksv=0.0, dye="fluorescein", raman=False):
    return MaterialSpec(id="fl", name="fl", dyes=[DyeAmount(fluorophore=dye, concentration_uM=c_uM)],
                        refractive_index=1.333, surface_roughness=0.0, quenching_ksv_per_uM=ksv, aqueous=raman)


def _band_filter(fine=False, late=False):
    if fine:   # 4 nm Gaussian bands every 2 nm: samples the emission spectrum
        return SpectralFilterSpec(kind="lctf", band_start_nm=496.0, band_step_nm=2.0, n_bands=100,
                                  band_start_policy="fixed", fwhm_nm=4.0, peak_transmission_curve="flat",
                                  emission_longpass_nm=None)
    return SpectralFilterSpec(kind="lctf", band_start_nm=540.0 if late else 500.0, band_step_nm=10.0, n_bands=18,
                              band_start_policy="fixed", fwhm_nm=20.0, peak_transmission_curve="flat",
                              emission_longpass_nm=None)


def exp_cuvette_90(c_uM, *, beam_z=0.0, fine=False, ksv=0.0, ex=488.0, power=1.0, steps=None, toggles=NO_NOISE):
    """Laser along +x through a single cuvette, camera looking along -z (90 deg)."""
    mats = presets.builtin_materials() + [_fl_material(c_uM, ksv)]
    laser = LightSourceSpec(id="L", name="L", kind="laser", position_mm=(-170, 16, beam_z), target_mm=(0, 16, beam_z),
                            line_fwhm_nm=1.0, lamp_curve="flat", excitation_filter_od=6.0,
                            beam=BeamSpec(profile="gaussian", aperture_diameter_mm=1.0, divergence_half_angle_deg=0.05),
                            sequence=steps or [ExcitationStep(wavelength_nm=ex, power_mW=power, exposure_ms=10.0)])
    cam = CameraSpec(id="C", name="C", position_mm=(0, 16, 300), target_mm=(0, 16, 0),
                     lens=LensSpec(focal_length_mm=50, f_number=2.8, focus_distance_mm=300),
                     sensor=SensorSpec(), filter=_band_filter(fine))
    exp = Experiment(materials=mats, objects=[SceneObject(id="c", name="c", shape=CuvetteShape(), material_id="fl",
                                                           transform=Transform(), vessel_material_id="glass")],
                     background=Background(kind="none", material_id=None), sources=[laser], cameras=[cam],
                     active_source_id="L", active_camera_id="C")
    return exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"physics": toggles})})


def exp_slab_front(c_uM, *, ksv=0.0, fine=False, late=False, steps=None, power=1.0, toggles=NO_NOISE, aperture=20.0, dye="fluorescein", normal=False):
    """Thick slab lit at 30 deg from the normal, camera on the normal (front-face geometry)."""
    mats = presets.builtin_materials() + [_fl_material(c_uM, ksv, dye)]
    src = LightSourceSpec(id="L", name="L", kind="laser", position_mm=(0.0, 130.0, 0.0) if normal else (-86.6, 130.0, 0.0), target_mm=(0, 30, 0),
                          line_fwhm_nm=1.0, lamp_curve="flat", excitation_filter_od=6.0,
                          beam=BeamSpec(profile="flat_top", aperture_diameter_mm=aperture, divergence_half_angle_deg=0.05),
                          sequence=steps or [ExcitationStep(wavelength_nm=488.0, power_mW=power, exposure_ms=10.0)])
    cam = CameraSpec(id="C", name="C", position_mm=(0, 330, 0), target_mm=(0, 30, 0),
                     lens=LensSpec(focal_length_mm=50, f_number=4, focus_distance_mm=300),
                     sensor=SensorSpec(), filter=_band_filter(fine, late))
    exp = Experiment(materials=mats, objects=[SceneObject(id="s", name="s", shape=BoxShape(size_mm=(80, 30, 80)),
                                                           material_id="fl", transform=Transform(position_mm=(0, 15, 0)))],
                     background=Background(kind="none", material_id=None), sources=[src], cameras=[cam],
                     active_source_id="L", active_camera_id="C")
    return exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"physics": toggles})})


def _centre(res, ex=None):
    ex = ex or res.excitations_nm[0]
    cube = res.radiance[ex]
    return cube[cube.shape[0] // 2, cube.shape[1] // 2]


# ================================================================ 5/6. inner filter & reabsorption
def fig_inner_filter():
    conc = np.logspace(-1, np.log10(500.0), 15 if QUICK else 22)
    F90c, F90tot, Fff, Fff_nc = [], [], [], []
    for c in conc:
        r = cap.capture(exp_cuvette_90(c), scale=0.25)
        F90c.append(_centre(r).sum())
        F90tot.append(r.radiance[r.excitations_nm[0]].sum())
        Fff.append(_centre(cap.capture(exp_slab_front(c), scale=0.25)).sum())
        # no-inner-filter reference (absorption toggle off keeps dye absorption for emission only)
        tg = NO_NOISE.model_copy(update={"absorption": False})
        Fff_nc.append(_centre(cap.capture(exp_slab_front(c, toggles=tg), scale=0.25)).sum())
    F90c, F90tot, Fff, Fff_nc = map(np.array, (F90c, F90tot, Fff, Fff_nc))
    # absorbance of 1 cm at 488 nm (10 mm cuvette)
    eps = float(optics_mod.dye_extinction_M_cm(optics_mod.merged_fluorophores(exp_cuvette_90(1.0))["fluorescein"], 488.0))
    A1cm = eps * conc * 1e-6
    lin = Fff_nc / conc
    NUM["inner_filter"] = dict(eps488=eps, conc_uM=conc.tolist(), F90_centre=F90c.tolist(), F90_total=F90tot.tolist(),
                               F_front=Fff.tolist(), F_front_noIF=Fff_nc.tolist(),
                               peak_conc_90_centre_uM=float(conc[np.argmax(F90c)]),
                               peak_conc_90_total_uM=float(conc[np.argmax(F90tot)]))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.1))
    a = ax[0]
    norm = lin[0] * conc  # dilute-limit linear signal of the front-face geometry
    a.loglog(conc, Fff_nc / Fff_nc[0] * conc[0] / conc[0] * 1.0 * 0 + Fff_nc / (lin[0]), "--", color=C["g"], label="front-face, no inner filter (linear)")
    a.loglog(conc, Fff / lin[0], "-o", ms=3, color=C["a"], label="front-face (30° in, camera on normal)")
    s90 = F90tot / (F90tot[0] / conc[0])
    a.loglog(conc, s90, "-s", ms=3, color=C["b"], label="90° geometry, whole cuvette")
    c90 = F90c / (F90c[0] / conc[0])
    a.loglog(conc, c90, "-^", ms=3, color=C["d"], label="90° geometry, centre pixel")
    a.set_ylim(1e-3, 1e3)
    a.set_xlabel("fluorescein concentration (µM)")
    a.set_ylabel("signal, normalised to the dilute-limit slope\n(= concentration if no inner filter)")
    a.set_title("Inner-filter saturation", pad=34)
    a.legend(fontsize=7.5, loc="lower right")
    sec = a.secondary_xaxis("top", functions=(lambda c: c * 1e-6 * eps, lambda A: A / (1e-6 * eps)))
    sec.set_xlabel("absorbance of 1 cm at 488 nm")
    b = ax[1]
    b.semilogx(conc, Fff / Fff[0] / (conc / conc[0]), "-o", ms=3, color=C["a"], label="front-face")
    b.semilogx(conc, F90tot / F90tot[0] / (conc / conc[0]), "-s", ms=3, color=C["b"], label="90°, whole cuvette")
    b.semilogx(conc, F90c / F90c[0] / (conc / conc[0]), "-^", ms=3, color=C["d"], label="90°, centre pixel")
    ref = np.exp(-np.log(10) * A1cm * 0.5)
    b.semilogx(conc, ref, ":", color=C["k"], label="$10^{-A/2}$ (centre of 1 cm, excitation only)")
    b.set_ylim(1e-4, 1.3)
    b.set_yscale("log")
    b.set_xlabel("fluorescein concentration (µM)")
    b.set_ylabel("signal per unit concentration (relative)")
    b.set_title("Fractional loss from the inner-filter effect")
    b.legend(fontsize=7.5)
    save(fig, "fig05_inner_filter.png")

    # ---- 6: reabsorption red-shift ------------------------------------------------------------
    conc2 = np.logspace(-1, np.log10(500.0), 10 if QUICK else 14)
    peak_far, peak_near, peak_ff = [], [], []

    def peak_of(res):
        lam = res.emission_nm[res.excitations_nm[0]]
        s = _centre(res)
        k = int(np.argmax(s))
        sl = slice(max(k - 4, 0), k + 5)
        w = s[sl]
        return float((lam[sl] * w).sum() / w.sum())    # centroid of the top of the line (robust to the 2 nm grid)
    for c in conc2:
        peak_far.append(peak_of(cap.capture(exp_cuvette_90(c, beam_z=-4.0, fine=True), scale=0.25)))
        peak_near.append(peak_of(cap.capture(exp_cuvette_90(c, beam_z=4.0, fine=True), scale=0.25)))
        peak_ff.append(peak_of(cap.capture(exp_slab_front(c, fine=True), scale=0.25)))
    spec = {}
    for c in (1.0, 500.0):
        r = cap.capture(exp_cuvette_90(c, beam_z=-4.0, fine=True), scale=0.25)
        s = _centre(r)
        spec[c] = (r.emission_nm[r.excitations_nm[0]], s / s.max())
    NUM["reabsorption"] = dict(conc_uM=conc2.tolist(), peak_far_nm=peak_far, peak_near_nm=peak_near, peak_front_nm=peak_ff)
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.9))
    ax[0].semilogx(conc2, peak_far, "-o", ms=3, color=C["b"], label="90°, beam at the far wall (9 mm to exit)")
    ax[0].semilogx(conc2, peak_near, "-s", ms=3, color=C["a"], label="90°, beam at the near wall (1 mm to exit)")
    ax[0].semilogx(conc2, peak_ff, "-^", ms=3, color=C["c"], label="front-face slab")
    ax[0].axhline(512.0, color=C["g"], lw=0.8)
    ax[0].text(conc2[-1], 510.9, "intrinsic emission maximum 512 nm", fontsize=8, color=C["g"], ha="right")
    ax[0].set_xlabel("fluorescein concentration (µM)")
    ax[0].set_ylabel("apparent emission maximum (nm)")
    ax[0].set_title("Reabsorption (secondary inner filter) red-shift")
    ax[0].legend(fontsize=7.5)
    for c, col in ((1.0, C["a"]), (500.0, C["b"])):
        ax[1].plot(spec[c][0], spec[c][1], color=col, label=f"{c:g} µM")
    ax[1].set_xlabel("wavelength (nm)")
    ax[1].set_ylabel("normalised radiance (far-wall beam)")
    ax[1].set_title("Emission spectra: blue edge eaten by self-absorption")
    ax[1].legend(fontsize=8)
    save(fig, "fig06_reabsorption_redshift.png")


# ================================================================ 3. closed form vs volumetric quadrature
def fig_closed_vs_volume():
    from spectraforge.studio.tracer import trace_scene
    conc = np.logspace(-1, np.log10(1000.0), 12 if QUICK else 18)
    vol, clo = [], []

    def tracer_closed(exp, w, h, **kw):
        kw.pop("volume_samples", None)
        return trace_scene(exp, w, h, volume_samples=0, **kw)
    for c in conc:
        e = exp_slab_front(c, aperture=300.0)     # beam much wider than the slab column: laterally uniform lighting
        vol.append(_centre(cap.capture(e, scale=0.25)).sum())
        clo.append(_centre(cap.capture(e, scale=0.25, tracer=tracer_closed)).sum())
    vol, clo = np.array(vol), np.array(clo)
    NUM["closed_vs_volume"] = dict(conc_uM=conc.tolist(), volumetric=vol.tolist(), closed=clo.tolist(),
                                   max_abs_rel_diff=float(np.max(np.abs(vol / clo - 1))))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].loglog(conc, clo / clo[0] / (conc / conc[0]), "-", color=C["a"], lw=3, alpha=0.55, label="closed form  $T_{in}T_{out}(1-e^{-\\kappa L})/(4\\pi n^2\\cos\\theta_i'\\kappa)$")
    ax[0].loglog(conc, vol / vol[0] / (conc / conc[0]), "o", ms=4, color=C["b"], label="volume quadrature (tracer samples, exact cell integration)")
    ax[0].set_xlabel("fluorescein concentration (µM)")
    ax[0].set_ylabel("signal per unit concentration (relative)")
    ax[0].set_title("Front-lit slab, uniform illumination: two independent kernels")
    ax[0].legend(fontsize=8)
    ax[1].semilogx(conc, 100 * (vol / clo - 1), "-o", ms=3, color=C["d"])
    ax[1].set_xlabel("fluorescein concentration (µM)")
    ax[1].set_ylabel("quadrature / closed form − 1  (%)")
    ax[1].set_title("Difference")
    save(fig, "fig03_closed_vs_volume.png")


# ================================================================ 4. dilute limit vs legacy render
def fig_legacy():
    from spectraforge.acquisition import AcquisitionConfig
    from spectraforge.forward import render
    from spectraforge.library import load_builtin_library
    from spectraforge.material import Material
    from spectraforge.scene import Scene
    from spectraforge.studio.schema import FluorophoreSpec
    names = ["EGFP", "mCherry", "DAPI"]
    conc_uM = {"EGFP": 0.02, "mCherry": 0.05, "DAPI": 0.03}
    lib = load_builtin_library()
    specs = [FluorophoreSpec(name=n, ex_peak_nm=lib[n].ex_peak_nm, ex_fwhm_nm=lib[n].ex_fwhm_nm, em_peak_nm=lib[n].em_peak_nm,
                             em_fwhm_nm=lib[n].em_fwhm_nm, quantum_yield=lib[n].quantum_yield, extinction=lib[n].extinction,
                             em_skew=lib[n].em_skew, vibronic=lib[n].vibronic) for n in names]
    mat = MaterialSpec(id="d", name="d", dyes=[DyeAmount(fluorophore=n, concentration_uM=c) for n, c in conc_uM.items()],
                       refractive_index=1.333)
    fl = {s.name: s for s in specs}
    opt = optics_mod.material_optics(mat, fl, GRID)
    src = LightSourceSpec(id="L", name="L", kind="laser", line_fwhm_nm=1.0, lamp_curve="flat")
    exs = [365.0, 405.0, 450.0, 488.0, 532.0]
    one = np.array([1.0])
    ours = {}
    for ex in exs:
        k = T.clear_medium_kernel(opt, ex, one, one, np.array([1.0]), 1.0, np.array([1.0]))
        sp = cap.dye_spectra(opt, cap._line_shape(src, ex), 1.0, k.fluorescence_G[0])
        ours[ex] = sp
    acq = AcquisitionConfig(excitations=exs, em_min=300.0, em_max=900.0, em_step=5.0)
    scene = Scene(1, 1)
    for n in names:
        scene.paint_rect(Material(name=n, recipe={n: conc_uM[n]}), 0, 1, 0, 1)
    _, gt = render(scene, lib, acq, seed=0)
    em = np.arange(300.0, 900.0 + 1e-9, 5.0)
    xs, ys, cols = [], [], []
    peak_all = max(gt.per_fluorophore_spectra[m][e].max() for m in names for e in exs)
    palette = {"EGFP": C["c"], "mCherry": C["b"], "DAPI": C["a"]}
    for ex in exs:
        for n in names:
            leg = gt.per_fluorophore_spectra[n][ex]
            mine = np.interp(em, GRID, ours[ex][n]) * 5.0
            if leg.max() > 0.05 * peak_all:
                s = leg > 0.02 * leg.max()
                xs.append(leg[s]); ys.append(mine[s]); cols += [palette[n]] * int(s.sum())
    x, y = np.concatenate(xs), np.concatenate(ys)
    ratio = y / x
    NUM["legacy"] = dict(global_constant=float(ratio.mean()), cv=float(ratio.std() / ratio.mean()), n_points=int(len(x)))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax[0].loglog(x, y, ".", ms=3, alpha=0.6, color=C["a"])
    xx = np.array([x.min(), x.max()])
    ax[0].loglog(xx, xx * ratio.mean(), "-", color=C["k"], lw=1, label=f"one global constant {ratio.mean():.3g}")
    ax[0].set_xlabel("legacy spectraforge.render (per-dye spectra, a.u.)")
    ax[0].set_ylabel("studio engine, dilute limit (mW sr$^{-1}$mm$^{-2}$)")
    ax[0].set_title(f"{len(x)} (dye, excitation, emission-bin) points")
    ax[0].legend(fontsize=8)
    ax[1].hist(100 * (ratio / ratio.mean() - 1), bins=40, color=C["a"])
    ax[1].set_xlabel("deviation from the global constant (%)")
    ax[1].set_ylabel("count")
    ax[1].set_title(f"coefficient of variation {100 * ratio.std() / ratio.mean():.3f} %")
    save(fig, "fig04_dilute_vs_legacy.png")


# ================================================================ 7. photobleaching vs acquisition order
def fig_bleaching():
    n = 10
    seq = [ExcitationStep(wavelength_nm=488.0, power_mW=150.0, exposure_ms=500.0) for _ in range(n)]
    on = NO_NOISE
    off = NO_NOISE.model_copy(update={"photobleaching": False})
    r_on = cap.capture(exp_slab_front(1.0, steps=seq, toggles=on, aperture=6.0, normal=True, late=True), scale=0.25)
    r_off = cap.capture(exp_slab_front(1.0, steps=seq, toggles=off, aperture=6.0, normal=True, late=True), scale=0.25)
    # radiance dict is keyed by wavelength: identical keys collide, so use distinct nm offsets per step
    seq2 = [ExcitationStep(wavelength_nm=488.0 + 0.01 * i, power_mW=150.0, exposure_ms=500.0) for i in range(n)]
    r_on = cap.capture(exp_slab_front(1.0, steps=seq2, toggles=on, aperture=6.0, normal=True, late=True), scale=0.25)
    r_off = cap.capture(exp_slab_front(1.0, steps=seq2, toggles=off, aperture=6.0, normal=True, late=True), scale=0.25)
    s_on = np.array([_centre(r_on, e).sum() for e in r_on.excitations_nm])
    s_off = np.array([_centre(r_off, e).sum() for e in r_off.excitations_nm])
    frac = s_on / s_off
    surv = r_on.ground_truth["bleached:fluorescein"]
    cy, cx = surv.shape[0] // 2, surv.shape[1] // 2
    k_step = -np.log(frac[1] / frac[0])
    # order experiment: dose is the same, the measured amplitudes differ
    def two(p1, p2):
        steps = [ExcitationStep(wavelength_nm=488.0, power_mW=p1, exposure_ms=1500.0),
                 ExcitationStep(wavelength_nm=488.01, power_mW=p2, exposure_ms=1500.0)]
        a = cap.capture(exp_slab_front(1.0, steps=steps, toggles=on, aperture=6.0, normal=True, late=True), scale=0.25)
        b = cap.capture(exp_slab_front(1.0, steps=steps, toggles=off, aperture=6.0, normal=True, late=True), scale=0.25)
        return [_centre(a, e).sum() / _centre(b, e).sum() for e in a.excitations_nm], float(a.ground_truth["bleached:fluorescein"][cy, cx])
    hi_lo, s1 = two(300.0, 40.0)
    lo_hi, s2 = two(40.0, 300.0)
    NUM["bleaching"] = dict(per_step_fraction=frac.tolist(), final_survival_centre=float(surv[cy, cx]),
                            k_per_step=float(k_step), order_high_low=hi_lo, order_low_high=lo_hi,
                            final_survival_high_low=s1, final_survival_low_high=s2)
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.7), gridspec_kw={"width_ratios": [1.25, 1.1, 1]})
    i = np.arange(1, n + 1)
    ax[0].plot(i, frac, "o-", color=C["b"], label="engine (bleaching on / off)")
    ax[0].plot(i, np.exp(-k_step * (i - 1)) * frac[0], "--", color=C["k"], lw=1.2, label="$\\exp(-k\\,(i-1))$, k fitted from steps 1–2")
    ax[0].set_xlabel("acquisition step (same 488 nm line, 150 mW focused, 0.5 s/band)")
    ax[0].set_ylabel("signal relative to un-bleached")
    ax[0].set_ylim(0.0, 1.05)
    ax[0].set_title("Photobleaching accumulates step by step")
    ax[0].legend(fontsize=7.5)
    w = 0.35
    ax[1].bar(np.arange(2) - w / 2, hi_lo, w, color=C["b"], label="order: 300 mW → 40 mW")
    ax[1].bar(np.arange(2) + w / 2, lo_hi, w, color=C["a"], label="order: 40 mW → 300 mW")
    ax[1].set_xticks([0, 1], ["first step", "second step"])
    ax[1].set_ylim(0.0, 1.05)
    ax[1].set_ylabel("measured / un-bleached")
    ax[1].set_title("Same total dose, different order")
    ax[1].legend(fontsize=7.5, loc="lower left")
    im = ax[2].imshow(surv, cmap="viridis", vmin=max(0, surv.min() * 0.95), vmax=1)
    ax[2].set_title("surviving fraction (GT, after 10 steps)")
    ax[2].grid(False)
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    save(fig, "fig07_photobleaching.png")


# ================================================================ 8. noise statistics (photon transfer curve)
def fig_noise():
    sens = SensorSpec(width_px=256, height_px=256, binning=1, gain_e_per_adu=1.5, read_noise_e=8.0, dark_current_e_per_s=0.5,
                      offset_adu=100.0, full_well_e=1e9, bit_depth=16)
    tg = PhysicsToggles(fixed_pattern_noise=False, quantization=False, saturation=False)
    rng = np.random.default_rng(0)
    levels = np.logspace(0, 4.7, 24)
    mean_adu, var_adu = [], []
    for s in levels:
        a, _ = sensor.detect(np.full((256, 256, 1), s), 100.0, sens, tg, None, rng)
        a = a[:, :, 0].astype(np.float64)
        mean_adu.append(a.mean() - sens.offset_adu)
        var_adu.append(a.var())
    mean_adu, var_adu = np.array(mean_adu), np.array(var_adu)
    lowm = mean_adu < 400          # intercept (read noise) is best constrained at low signal
    A = np.vstack([mean_adu[lowm], np.ones(lowm.sum())]).T
    slope, icpt = np.linalg.lstsq(A, var_adu[lowm], rcond=None)[0]
    slope_all = np.polyfit(mean_adu, var_adu, 1)[0]
    NUM["ptc"] = dict(slope_ADU_inv=float(slope), slope_full_range=float(slope_all), expected_slope=float(1 / sens.gain_e_per_adu),
                      gain_e_per_ADU_fit=float(1 / slope), read_noise_e_fit=float(np.sqrt(max(icpt, 0)) * sens.gain_e_per_adu),
                      read_noise_e_true=sens.read_noise_e)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].loglog(mean_adu, var_adu, "o", ms=4, color=C["a"], label="simulated frames (256×256)")
    xx = np.logspace(0, np.log10(mean_adu.max()), 50)
    ax[0].loglog(xx, xx / sens.gain_e_per_adu + (sens.read_noise_e / sens.gain_e_per_adu) ** 2, "-", color=C["k"], lw=1,
                 label=f"theory: $\\bar S/g + (\\sigma_r/g)^2$, g={sens.gain_e_per_adu} e⁻/ADU")
    ax[0].set_xlabel("mean signal (ADU, offset removed)")
    ax[0].set_ylabel("variance (ADU²)")
    ax[0].set_title("Photon transfer curve")
    ax[0].legend(fontsize=7.5)
    ax[1].plot(mean_adu, var_adu, "o", ms=4, color=C["a"])
    ax[1].plot(mean_adu, slope * mean_adu + icpt, "-", color=C["b"],
               label=f"fit slope {slope:.4f} ADU$^{{-1}}$ (1/g = {1 / sens.gain_e_per_adu:.4f})\nintercept {icpt:.1f} ADU² → σ$_r$ = {NUM['ptc']['read_noise_e_fit']:.2f} e⁻")
    ax[1].set_xlim(0, 2500)
    ax[1].set_ylim(0, 2500 * slope * 1.1 + icpt)
    ax[1].set_xlabel("mean signal (ADU)")
    ax[1].set_ylabel("variance (ADU²)")
    ax[1].set_title("Linear view of the low-signal range")
    ax[1].legend(fontsize=7.5, loc="upper left")
    save(fig, "fig08_photon_transfer.png")


# ================================================================ 9. depth of field / defocus
def fig_dof():
    from spectraforge.studio.sensor import blur_cube, blur_sigma_px
    cam = presets.nuance_camera()
    z = np.linspace(280, 420, 300)
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for N, col in ((2.8, C["b"]), (4.0, C["a"]), (8.0, C["c"])):
        cm = cam.model_copy(update={"lens": cam.lens.model_copy(update={"f_number": N})})
        tg = PhysicsToggles()
        s = blur_sigma_px(z, cm, np.array([550.0]), tg, 330.0)[:, 0]
        ax[0].plot(z, s, color=col, label=f"f/{N:g}")
        t2 = tg.model_copy(update={"diffraction": False})
        ax[0].plot(z, blur_sigma_px(z, cm, np.array([550.0]), t2, 330.0)[:, 0], ":", color=col, lw=1)
    f, S1 = cam.lens.focal_length_mm, 330.0
    pitch = cam.sensor.pixel_pitch_um * cam.sensor.binning * 1e-3
    v1 = f * S1 / (S1 - f)
    coc = (f / 4.0) * np.abs(v1 - f * z / (z - f) - 0.05 * (550 - 550) / 100) / (f * z / (z - f))
    ax[0].plot(z, coc / 4 / pitch, "k--", lw=0.8, label="thin-lens CoC/4 (f/4)")
    ax[0].axvline(330, color=C["g"], lw=0.8)
    ax[0].set_xlabel("object distance (mm), lens focused at 330 mm")
    ax[0].set_ylabel("blur σ (binned pixels, 25.8 µm)")
    ax[0].set_title("Defocus blur vs depth (35 mm lens, 550 nm)")
    ax[0].text(0.03, 0.62, "dotted: without diffraction", transform=ax[0].transAxes, fontsize=7.5, color=C["g"])
    ax[0].legend(fontsize=7.5, loc="upper center")
    # edge profiles through blur_cube
    edge = np.zeros((8, 200, 1), dtype=np.float32)
    edge[:, 100:] = 1.0
    for zz, col in ((330.0, C["a"]), (345.0, C["c"]), (375.0, C["b"])):
        s = blur_sigma_px(np.array([zz]), cam, np.array([550.0]), PhysicsToggles(), 330.0)[0, 0]
        out = blur_cube(edge, np.full(edge.shape, s, dtype=np.float32))
        ax[1].plot(np.arange(200) - 100, out[4, :, 0], color=col, label=f"{zz:g} mm  (σ = {s:.2f} px)")
    ax[1].set_xlim(-25, 25)
    ax[1].set_xlabel("pixels across a sharp edge")
    ax[1].set_ylabel("recorded edge profile")
    ax[1].set_title("Edge spread at three depths")
    ax[1].legend(fontsize=8)
    NUM["dof"] = dict(sigma_at_focus_px=float(blur_sigma_px(np.array([330.0]), cam, np.array([550.0]), PhysicsToggles(), 330.0)[0, 0]),
                      sigma_at_375_px=float(blur_sigma_px(np.array([375.0]), cam, np.array([550.0]), PhysicsToggles(), 330.0)[0, 0]))
    save(fig, "fig09_depth_of_field.png")


# ================================================================ 10. filter angle blue shift
def fig_angle():
    from spectraforge.studio.tracer import trace_scene
    th = np.radians(np.linspace(0, 25, 100))
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9), gridspec_kw={"width_ratios": [1.1, 1, 1.25]}, layout="constrained")
    for n_eff, col in ((1.5, C["b"]), (2.0, C["a"]), (3.0, C["c"])):
        ax[0].plot(np.degrees(th), sensor.shifted_centre(600.0, th, n_eff) - 600.0, color=col, label=f"$n_{{eff}}$ = {n_eff}")
    ax[0].set_xlabel("field angle (°)")
    ax[0].set_ylabel("centre-wavelength shift of a 600 nm band (nm)")
    ax[0].set_title("$\\lambda(\\theta)=\\lambda_0\\sqrt{1-\\sin^2\\theta/n_{eff}^2}$")
    ax[0].legend(fontsize=8)
    exp = presets.preset("mehsi_lab_replica")
    hit = trace_scene(exp, 87, 64)
    fa = hit.field_angle_rad.reshape(64, 87)
    cam = exp.camera()
    dl = sensor.shifted_centre(600.0, fa, cam.filter.interference_n_eff) - 600.0
    im = ax[1].imshow(dl, cmap="viridis")
    ax[1].set_title(f"lab-replica field: 600 nm band shift\n(max field angle {np.degrees(fa.max()):.1f}°)")
    ax[1].grid(False)
    ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="nm")
    # effect on a measurement: radiance of a narrow-band emitter in a 20 nm band, centre vs corner
    lam = np.arange(540.0, 660.0, 1.0)
    line = np.exp(-4 * np.log(2) * ((lam - 590.0) / 8.0) ** 2)
    for ang, col in ((0.0, C["a"]), (float(fa.max()), C["b"])):
        c = sensor.shifted_centre(600.0, ang, cam.filter.interference_n_eff)
        ax[2].plot(lam, sensor.passband(lam, c, 20.0), color=col, label=f"band at {np.degrees(ang):.1f}°: centre {c:.1f} nm")
    ax[2].fill_between(lam, 0, line, color=C["g"], alpha=0.3, label="590 nm emission line (8 nm)")
    ax[2].set_xlabel("wavelength (nm)")
    ax[2].set_ylabel("transmission")
    ax[2].set_title("Corner pixels sample a bluer band")
    ax[2].legend(fontsize=7.5)
    NUM["angle"] = dict(max_field_angle_deg=float(np.degrees(fa.max())), max_shift_600nm=float(dl.min()), n_eff=cam.filter.interference_n_eff)
    save(fig, "fig10_filter_angle_shift.png")


# ================================================================ 11. laser_line demo (preset): inner-filter gradients
def fig_laser():
    exp = presets.preset("laser_line")
    r = cap.capture(exp)
    obj = r.ground_truth["object"]
    fl = optics_mod.merged_fluorophores(exp)
    labels = ["EGFP→mCherry\n(FRET pair)", "fluorescein\n10 µM", "fluorescein\n200 µM"]
    fig, ax = plt.subplots(2, 3, figsize=(12.6, 6.6), layout="constrained")
    yc = r.height // 2
    NUM["laser_line"] = {}
    for j, ex in enumerate([457.0, 488.0, 561.0]):
        tot = r.radiance[ex].sum(2)
        ax[0, j].imshow(tot, cmap="magma", vmax=np.percentile(tot, 99.9))
        ax[0, j].set_title(f"{ex:g} nm  ({r.power_mW[ex]:.0f} mW)")
        ax[0, j].axis("off")
        row = tot[yc]
        profs = {}
        for o in range(3):
            xs = np.nonzero(obj[yc] == o)[0]
            if len(xs) < 8:
                continue
            mm_per_px = 12.5 / len(xs)
            inner = xs[int(round(1.25 / mm_per_px)):len(xs) - int(round(1.25 / mm_per_px))]
            if row[inner].max() > 0:
                profs[o] = (row[inner], mm_per_px)
        # every curve in a panel is normalised to the single brightest cuvette of that panel, so relative
        # brightness is honest (a cuvette that does not absorb at this wavelength plots near zero, not at 1)
        pmax = max((p.max() for p, _ in profs.values()), default=1.0)
        for (o, (prof, mm_per_px)), col in zip(profs.items(), [{0: C["c"], 1: C["a"], 2: C["b"]}[k] for k in profs]):
            x_mm = (np.arange(len(prof)) + 0.5) * mm_per_px
            dim = prof.max() < 0.01 * pmax
            lab = labels[o].replace("\n", " ") + (" (≈ no absorption)" if dim else "")
            ax[1, j].semilogy(x_mm, prof / pmax, color=col, label=lab, ls=":" if dim else "-", alpha=0.6 if dim else 1.0)
            NUM["laser_line"][f"{ex:g}_{o}"] = float(prof.max())
            if o == 2 and ex in (457.0, 488.0):
                spec = fl["fluorescein"]
                mu = np.log(10) * float(optics_mod.dye_extinction_M_cm(spec, ex)) * 200e-6 / 10.0
                ax[1, j].semilogy(x_mm, np.exp(-mu * x_mm), "--", color=C["k"], lw=1, label=f"$e^{{-\\mu x}}$, µ = {mu:.2f} /mm")
        ax[1, j].set_xlabel("distance along the beam inside the cuvette (mm)")
        ax[1, j].set_ylabel("radiance at beam height (÷ brightest cuvette in panel)")
        ax[1, j].set_ylim(1e-4, 2)
        ax[1, j].legend(fontsize=7.5)
    fig.suptitle("laser_line preset: beam travels left → right through three cuvettes in series", fontsize=10)
    save(fig, "fig11_laser_line_demo.png")


# ================================================================ 12-17. layered blocks (3D Monte Carlo)
def _blk_geom(layers, fp):
    from spectraforge.studio import transport3d as T3
    from spectraforge.studio.contracts import BlockGeometry
    return BlockGeometry(0, tuple(map(float, fp)), [f"l{i}" for i in range(len(layers))], list(range(len(layers))),
                         T3.planes_from_layers(fp, layers), np.eye(4))


def _blk_opt(mua, mus, g, n, lam=500.0):
    from spectraforge.studio.contracts import RegionOptics
    a = lambda v: np.atleast_1d(np.asarray(v, float))
    return RegionOptics(lam, a(mua), a(mus), a(g), a(n))


def _blk_cfg(lateral, exit, sub=4, ang=12.0):
    from spectraforge.studio.schema import VolumeMCSettings
    return VolumeMCSettings(lateral_cell_mm=lateral, exit_cell_mm=exit, sublayers=sub, acceptance_half_angle_deg=ang)


def _blk_exp(name, ex_nm, photons, cell=0.5):
    """A layered-block preset reduced to one excitation, noise-free sensor, bleaching off, full photon
    fraction at the half-resolution scale used by the figures."""
    exp = presets.preset(name)
    src = exp.source()
    seq = [s for s in src.sequence if s.enabled and abs(s.wavelength_nm - ex_nm) < 1e-6]
    tg = exp.acquisition.physics.model_copy(update=dict(
        power_drift=False, shot_noise=False, read_noise=False, dark_current=False, fixed_pattern_noise=False,
        photobleaching=False))
    vs = exp.acquisition.volume_mc.model_copy(update=dict(
        excitation_photons=photons, emission_photons=photons, preview_fraction=1.0, lateral_cell_mm=cell,
        exit_cell_mm=cell))
    acq = exp.acquisition.model_copy(update=dict(physics=tg, volume_mc=vs))
    return exp.model_copy(update=dict(
        sources=[src.model_copy(update=dict(sequence=seq)) if s.id == src.id else s for s in exp.sources],
        acquisition=acq))


# ---------------------------------------------------------------- (a) 3D vs MCML / 1-D
def fig_blk_validation():
    from spectraforge.studio import transport3d as T3
    from spectraforge.studio.contracts import MaterialOptics
    from spectraforge.studio.schema import ThicknessProfile as TP
    npho = 600_000 if QUICK else 4_000_000
    rows = []
    # 1. homogeneous slab R/T/A vs MCML
    for lab, (mua, mus, g, n, d) in {"benchmark (n=1)": (1.0, 9.0, 0.75, 1.0, 0.2), "index mismatch n=1.4": (1.0, 9.0, 0.75, 1.4, 0.2),
                                    "tissue-like": (0.1, 20.0, 0.9, 1.38, 1.0)}.items():
        G = _blk_geom([TP(thickness_mm=d)], (200.0, 200.0))
        launch = T3.uniform_beam_launch(G, 500.0, 1.0, 200_000)
        r = T3.excitation_pass(G, _blk_opt(mua, mus, g, n), launch, [0, 1, 0], _blk_cfg(100, 100), n_photons=npho, seed=1)
        ref = mc.mcml_slab(mua, mus, g, n, 1.0, d, n_photons=npho, seed=3)
        for q, v, rf in (("R", r.diffuse_reflectance, ref["Rd"]), ("T", r.transmittance, ref["Tt"]), ("A", r.absorbed_fraction, ref["A"])):
            rows.append((f"slab {lab}", q, v, rf, "MCML (transport_mc.mcml_slab)"))
        rows.append((f"slab {lab}", "R+T+A+spec", r.info["energy_balance"], 1.0, "energy balance"))
    # 2. fluorescent layer vs the 1-D kernels
    EX, EM = 405.0, 520.0
    mua_ex, mus_ex, mua_em, mus_em, g, n = 0.30, 12.0, 0.10, 9.0, 0.8, 1.4
    mua_dye, phi, d = 0.20, 0.5, 2.0
    G = _blk_geom([TP(thickness_mm=d)], (60.0, 60.0))
    s = _blk_cfg(20, 20, sub=64)
    launch = T3.uniform_beam_launch(G, EX, 1.0, 400_000)
    ex = T3.excitation_pass(G, _blk_opt(mua_ex, mus_ex, g, n, EX), launch, [0, 1, 0], s, n_photons=npho * 2, seed=1)
    src = np.zeros((1,) + ex.absorbed_mW.shape)
    src[0, :, :, 0, :] = ex.absorbed_mW[:, :, 0, :] * (mua_dye / mua_ex) * phi
    em = T3.emission_pass(G, [_blk_opt(mua_em, mus_em, g, n, EM)], src, ["dye"], [0, 1, 0], s, n_photons=npho * 2, seed=2)
    lo = GRID <= EX + 1e-9
    m1 = MaterialOptics("m", GRID, np.where(lo, mua_ex, mua_em).astype(float), {}, np.where(lo, mus_ex, mus_em).astype(float), g, n, False, {}, {})
    k = T.get_transport_model().kernels(m1, EX, np.array([1.0]), np.array([1.0]), np.array([d]), 1.0)
    E = 1.0 / (60.0 * 60.0)
    L3, L1 = em.radiance["top"][1, 1, 0, 0], E * mua_dye * phi * k.fluorescence_G[0, int(np.argmin(abs(GRID - EM)))]
    rows += [("fluorescent layer, 2 mm, 60 mm footprint", "fluorescence radiance", L3, L1, "1-D kernel E mu_a,d Phi e G"),
             ("", "elastic radiance / E", ex.elastic_radiance["top"][1, 1] / E, k.elastic_rho[0], "1-D kernel rho")]
    # 3. stacked identical layers == one layer
    G1 = _blk_geom([TP(thickness_mm=1.0)], (100.0, 100.0))
    G2 = _blk_geom([TP(thickness_mm=0.4), TP(thickness_mm=0.6)], (100.0, 100.0))
    o1, o2 = _blk_opt(0.3, 12.0, 0.8, 1.4), _blk_opt([0.3, 0.3], [12.0, 12.0], [0.8, 0.8], [1.4, 1.4])
    l1 = T3.uniform_beam_launch(G1, 500.0, 1.0, 200_000)
    a = T3.excitation_pass(G1, o1, l1, [0, 1, 0], _blk_cfg(100, 100), n_photons=npho, seed=1)
    b = T3.excitation_pass(G2, o2, l1, [0, 1, 0], _blk_cfg(100, 100), n_photons=npho, seed=2)
    rows += [("two identical layers (0.4+0.6 mm) vs one (1.0 mm)", "R", b.diffuse_reflectance, a.diffuse_reflectance, "single layer"),
             ("", "T", b.transmittance, a.transmittance, ""), ("", "A", b.absorbed_fraction, a.absorbed_fraction, "")]
    NUM["blk_validation"] = [dict(case=c, quantity=q, engine=float(v), reference=float(rf), note=nt, rel_err=float(v / rf - 1))
                             for c, q, v, rf, nt in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 0.36 * len(rows) + 1.5), gridspec_kw=dict(width_ratios=[1.55, 1]))
    ax[0].axis("off")
    import textwrap
    cells = [[textwrap.fill(c, 24), textwrap.fill(q, 14), f"{v:.4g}", f"{rf:.4g}", f"{100 * (v / rf - 1):+.2f} %", textwrap.fill(nt, 24)] for c, q, v, rf, nt in rows]
    tb = ax[0].table(cellText=cells, colLabels=["case", "quantity", "3D", "reference", "diff.", "reference"],
                     loc="center", cellLoc="left", colWidths=[0.30, 0.15, 0.10, 0.11, 0.10, 0.24])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.0)
    tb.scale(1, 2.0)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if i == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(weight="bold")
    diffs = [100 * (v / rf - 1) for c, q, v, rf, nt in rows]
    ax[1].barh(np.arange(len(rows))[::-1], diffs, color=[C["a"] if abs(x) < 1.0 else C["b"] for x in diffs])
    ax[1].axvspan(-1, 1, color=C["g"], alpha=0.2)
    ax[1].set_yticks([])
    ax[1].set_xlabel("3D engine vs reference (%)   [grey band: ±1 %]")
    ax[1].set_title("deviation from the reference")
    save(fig, "fig12_blk_3d_vs_reference.png")


# ---------------------------------------------------------------- (b) bottom boundary
def _slab_capture_pair(th, photons):
    """Block vs 1-D box capture of the same fluorescent single-layer slab (secondary emission off in both)."""
    from spectraforge.studio.schema import BlockLayer, BoxShape, LayeredBlockShape, ThicknessProfile, Transform
    base = _blk_exp("layered_wedge_phantom", 490.0, photons, cell=1.5)
    tg = base.acquisition.physics.model_copy(update=dict(specular_glints=False, secondary_emission=False))
    mat = "fluorescein_agarose_layer"

    def build(shape, y):
        o = base.objects[0].model_copy(update=dict(shape=shape, material_id=mat, transform=Transform(position_mm=(0, y, 0), rotation_deg=(0, 0, 0))))
        return base.model_copy(update=dict(objects=[o], acquisition=base.acquisition.model_copy(update=dict(physics=tg))))

    blk = LayeredBlockShape(footprint_mm=(60, 60), layers=[BlockLayer(name="slab", material_id=mat, thickness=ThicknessProfile(thickness_mm=th))])
    rb = cap.capture(build(blk, 0.5), scale=0.5)          # slab standing above the stage plane (see the page text)
    rx = cap.capture(build(BoxShape(size_mm=(60, th, 60)), 0.5 + th / 2), scale=0.5)
    ex = rb.excitations_nm[0]
    H, W = rb.radiance[ex].shape[:2]
    sl = (slice(H // 2 - 6, H // 2 + 6), slice(W // 2 - 6, W // 2 + 6))
    return rb.emission_nm[ex], rb.radiance[ex][sl].mean(axis=(0, 1)), rx.radiance[ex][sl].mean(axis=(0, 1))


def fig_blk_bottom():
    from spectraforge.studio import transport3d as T3
    from spectraforge.studio.schema import ThicknessProfile as TP
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2), layout="constrained")
    # (i) clear slab: closed form with a Fresnel bottom (reflected part stays in the block)
    cases = [(1.33, 0.2, 2.0), (1.5, 0.5, 1.0), (2.0, 1.0, 0.5)]
    labels, mcT, anT, mcR, anR, sinkT = [], [], [], [], [], []
    G_cache = {}
    for n, mua, d in cases:
        G = _blk_geom([TP(thickness_mm=d)], (400.0, 400.0))
        launch = T3.uniform_beam_launch(G, 500.0, 1.0, 5000, beam_half_size_mm=(50.0, 50.0))
        r = T3.excitation_pass(G, _blk_opt(mua, 0.0, 0.0, n), launch, [0, 1, 0], _blk_cfg(200, 200, 2), n_photons=100_000, seed=2)
        R0 = ((n - 1) / (n + 1)) ** 2
        e = np.exp(-mua * d)
        den = 1 - R0 ** 2 * e ** 2
        labels.append(f"n={n}\nμa d={mua * d:.1f}")
        mcT.append(r.info["T_bottom"]); anT.append((1 - R0) ** 2 * e / den)
        mcR.append(r.info["R_top"]); anR.append((1 - R0) ** 2 * R0 * e ** 2 / den)
        sinkT.append((1 - R0) * e)
    x = np.arange(len(cases))
    ax[0].bar(x - 0.2, mcT, 0.38, color=C["a"], label="3D engine: light lost through the bottom, $T_{bot}$")
    ax[0].plot(x - 0.2, anT, "k_", ms=18, mew=2, label="closed form (Fresnel bottom)")
    ax[0].plot(x - 0.2, sinkT, "v", color=C["b"], label="perfect sink bottom (rejected)")
    ax[0].bar(x + 0.2, mcR, 0.38, color=C["c"], label="3D: diffuse reflectance, $R_{top}$")
    ax[0].plot(x + 0.2, anR, "k_", ms=18, mew=2)
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel("fraction of the incident power")
    ax[0].set_title("Clear slab, wide normal beam: bottom = Fresnel interface")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_ylim(0, 0.95)
    NUM["blk_bottom_clear"] = [dict(n=c[0], mua=c[1], d=c[2], T_mc=float(a), T_ref=float(b), R_mc=float(cc), R_ref=float(dd))
                               for c, a, b, cc, dd in zip(cases, mcT, anT, mcR, anR)]
    # (ii) 6 mm and 1 mm turbid fluorescent slab, block vs 1-D engine, capture level
    NUM["blk_bottom_slab"] = {}
    for a_, th in zip(ax[1:], (6.0, 1.0)):
        em, b, xx = _slab_capture_pair(th, 400_000 if QUICK else 2_000_000)
        sel = (em >= 495) & (em <= 625) & (xx > 1e-3 * xx.max())
        ratio = float(b[sel].sum() / xx[sel].sum())
        a_.plot(em, xx / xx.max(), color=C["k"], label="1-D engine (box)")
        a_.plot(em, b / xx.max(), color=C["b"], ls="--", label="3D block engine")
        a_.set_xlabel("emission (nm)")
        a_.set_ylabel("radiance (normalised to 1-D peak)")
        a_.set_title(f"{th:g} mm turbid fluorescent slab: block / 1-D = {ratio:.3f}")
        a_.legend(fontsize=8)
        NUM["blk_bottom_slab"][f"{th:g}mm"] = ratio
    fig.suptitle("Bottom-boundary consistency of the 3D block engine with the 1-D slab engine (secondary emission off in both)", fontsize=10)
    save(fig, "fig13_blk_bottom_boundary.png")


# ---------------------------------------------------------------- (c) peel-off vs cone
def fig_blk_peel():
    from spectraforge.studio import transport3d as T3
    from spectraforge.studio.schema import ThicknessProfile as TP
    WED = lambda t0, t1, ax_="x": TP(kind="linear", thickness_mm=t0, end_thickness_mm=t1, axis=ax_)
    cases = {
        "single layer, n=1.4": ([TP(thickness_mm=1.0)], _blk_opt(0.3, 12.0, 0.8, 1.4), _blk_opt(0.1, 9.0, 0.8, 1.4), (0, 1, 0), (0, -1, 0)),
        "single layer, n=1": ([TP(thickness_mm=1.0)], _blk_opt(0.3, 12.0, 0.8, 1.0), _blk_opt(0.1, 9.0, 0.8, 1.0), (0, 1, 0), (0, -1, 0)),
        "wedge, oblique cam+beam": ([TP(thickness_mm=1.5), WED(2.0, 0.0)], _blk_opt([0.2, 0.3], [10.0, 15.0], [0.8, 0.9], [1.4, 1.36]),
                                    _blk_opt([0.1, 0.2], [9.0, 12.0], [0.8, 0.9], [1.4, 1.36]), (0.5, 0.8, -0.3), (0.3, -0.9, 0.1)),
        "clear cover, mismatch": ([TP(thickness_mm=1.0), TP(thickness_mm=0.8)], _blk_opt([0.2, 0.05], [10.0, 0.0], [0.8, 0.9], [1.4, 1.5]),
                                  _blk_opt([0.1, 0.05], [9.0, 0.0], [0.8, 0.9], [1.4, 1.5]), (0.2, 1, 0.1), (0, -1, 0)),
    }
    npho = 500_000 if QUICK else 3_000_000
    s = _blk_cfg(5.0, 5.0, 2)
    rat = {}
    for name, (layers, ex_o, em_o, cam, dirn) in cases.items():
        G = _blk_geom(layers, (30.0, 30.0))
        cam = tuple(np.asarray(cam, float) / np.linalg.norm(cam))
        launch = T3.uniform_beam_launch(G, 500.0, 1.0, 50_000, direction=dirn, seed=1)
        vals = {}
        for peel in (False, True):
            vv = []
            for sd in (1, 2):
                e = T3.excitation_pass(G, ex_o, launch, cam, s, n_photons=npho, seed=sd, peel_off=peel)
                em = T3.emission_pass(G, [em_o], e.absorbed_mW[None].copy(), ["a"], cam, s, n_photons=npho, seed=sd + 10, peel_off=peel)
                vv.append((e.elastic_radiance["top"][1:5, 1:5].sum(), em.radiance["top"][1:5, 1:5, 0, 0].sum()))
            vals[peel] = np.array(vv)
        r_el = vals[True][:, 0] / vals[False][:, 0]
        r_fl = vals[True][:, 1] / vals[False][:, 1]
        rat[name] = (r_el.mean(), abs(np.diff(r_el))[0] / 2 + 1e-9, r_fl.mean(), abs(np.diff(r_fl))[0] / 2 + 1e-9)
    NUM["blk_peel_ratio"] = {k: dict(elastic=float(v[0]), fluorescence=float(v[2])) for k, v in rat.items()}
    # noise vs photons on a uniform field (1 mm exit cells) and a lateral profile across a wedge
    G = _blk_geom([TP(thickness_mm=1.0)], (30.0, 30.0))
    o = _blk_opt(0.3, 12.0, 0.8, 1.4)
    launch = T3.uniform_beam_launch(G, 500.0, 1.0, 50_000, seed=1)
    s1 = _blk_cfg(5.0, 1.0, 2)
    ns = [100_000, 300_000, 1_000_000] if QUICK else [100_000, 300_000, 1_000_000, 3_000_000]
    cv = {False: [], True: []}
    for peel in (False, True):
        for nn in ns:
            r = T3.excitation_pass(G, o, launch, [0, 1, 0], s1, n_photons=nn, seed=1, peel_off=peel)
            a_ = r.elastic_radiance["top"][8:22, 8:22]
            cv[peel].append(float(a_.std() / a_.mean()))
    NUM["blk_peel_noise"] = dict(photons=ns, cv_cone=cv[False], cv_peel=cv[True],
                                 std_reduction_at_equal_photons=[a / b for a, b in zip(cv[False], cv[True])])
    Gw = _blk_geom([TP(thickness_mm=1.5), WED(2.0, 0.0)], (30.0, 30.0))
    ow = _blk_opt([0.2, 0.3], [10.0, 15.0], [0.8, 0.9], [1.4, 1.36])
    lw = T3.uniform_beam_launch(Gw, 500.0, 1.0, 50_000, seed=1)
    sw = _blk_cfg(5.0, 0.75, 2)
    prof = {}
    for peel in (False, True):
        r = T3.excitation_pass(Gw, ow, lw, [0, 1, 0], sw, n_photons=ns[-2], seed=3, peel_off=peel)
        prof[peel] = r.elastic_radiance["top"]
    fg = r.faces["top"]
    xc = 0.5 * (fg.u_edges_mm[1:] + fg.u_edges_mm[:-1])
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5), layout="constrained")
    names = list(rat)
    yy = np.arange(len(names))
    ax[0].errorbar([rat[n][0] for n in names], yy + 0.12, xerr=[rat[n][1] for n in names], fmt="o", color=C["a"], label="elastic")
    ax[0].errorbar([rat[n][2] for n in names], yy - 0.12, xerr=[rat[n][3] for n in names], fmt="s", color=C["b"], label="fluorescence")
    ax[0].axvline(1, color=C["k"], lw=1)
    ax[0].axvspan(0.98, 1.02, color=C["g"], alpha=0.2)
    ax[0].set_yticks(yy, [n.replace(", oblique cam+beam", " (oblique)").replace("single layer", "slab") for n in names], fontsize=7)
    ax[0].set_xlabel("peel-off / cone (region sum)")
    ax[0].set_title("(a) same answer (grey: $\\pm$2 %)")
    ax[0].set_ylim(-0.5, len(names) - 0.1 + 0.6)
    ax[0].legend(fontsize=7, loc="upper center", ncol=2, borderaxespad=0.1, columnspacing=0.8, handletextpad=0.2)
    ax[1].plot(xc, prof[False][:, 20:30].mean(1), color=C["g"], label="cone")
    ax[1].plot(xc, prof[True][:, 20:30].mean(1), color=C["b"], label="peel-off")
    ax[1].set_xlabel("x across the wedge (mm)")
    ax[1].set_ylabel("elastic radiance (norm. units)")
    ax[1].set_title(f"(b) wedge profile, {ns[-2]:.0e} photons")
    ax[1].legend(fontsize=7, borderaxespad=0.1)
    ax[2].loglog(ns, cv[False], "o-", color=C["g"], label="cone")
    ax[2].loglog(ns, cv[True], "s-", color=C["b"], label="peel-off")
    ax[2].loglog(ns, cv[False][0] * np.sqrt(ns[0] / np.array(ns)), ":", color=C["k"], lw=1, label="$1/\\sqrt{N}$")
    ax[2].set_xlabel("photons")
    ax[2].set_ylabel("cell noise (CV)")
    ax[2].set_title(f"(c) noise, ×{cv[False][-1] / cv[True][-1]:.1f} lower")
    ax[2].legend(fontsize=7, loc="lower left", borderaxespad=0.1)
    save(fig, "fig14_blk_peel_vs_cone.png")


# ---------------------------------------------------------------- (d) wedge response and (f) collagen
def _bin_components(exp, res, gt_name, edges, keys, per_bin=8, margin=0.12, scale=0.5):
    """explain_pixel components of pixels in thickness bins (cached 3D Monte Carlo => fast). Returns per-bin
    {key-group: spectrum / irradiance} on the band axis, bin centres and counts. Pixels near the block edge are skipped."""
    ex = res.excitations_nm[0]
    gt = res.ground_truth[gt_name]
    fin = np.isfinite(gt)
    ys, xs = np.nonzero(fin)
    y0, y1 = np.percentile(ys, [25, 75])
    x0, x1 = np.percentile(xs, [margin * 100, 100 - margin * 100])
    ok = fin & (np.arange(gt.shape[0])[:, None] >= y0) & (np.arange(gt.shape[0])[:, None] <= y1) \
        & (np.arange(gt.shape[1])[None, :] >= x0) & (np.arange(gt.shape[1])[None, :] <= x1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        yy, xx = np.nonzero(ok & (gt >= a) & (gt < b))
        if yy.size == 0:
            out.append(None)
            continue
        pick = np.linspace(0, yy.size - 1, min(per_bin, yy.size)).astype(int)
        acc = {g: 0.0 for g in keys}
        n = 0
        for i in pick:
            e = cap.explain_pixel(exp, int(xx[i]), int(yy[i]), scale=scale)
            E = e["ground_truth"][f"irradiance@{ex:g}"]
            comp = e["components"][f"{ex:g}"]
            for g, pred in keys.items():
                tot = np.zeros(len(e["emission_nm"][f"{ex:g}"]))
                for k, v in comp.items():
                    if pred(k):
                        tot += np.array(v)
                acc[g] = acc[g] + tot / E
            n += 1
        out.append({g: v / n for g, v in acc.items()})
    return out, 0.5 * (edges[:-1] + edges[1:]), np.array(res.emission_nm[ex])


def _band_curves(res, gt_name, edges, margin=0.12, waves=None):
    """Band radiance binned by local cover thickness: raw and divided by the ground-truth surface irradiance,
    for the first band behind the excitation line (ex+20 nm, where the elastic leak competes) and for ex+40 nm."""
    ex = res.excitations_nm[0]
    em = np.array(res.emission_nm[ex])
    gt = res.ground_truth[gt_name]
    E = res.ground_truth[f"irradiance@{ex:g}"]
    img = res.radiance[ex]
    fin = np.isfinite(gt)
    ys, xs = np.nonzero(fin)
    rows = (np.arange(gt.shape[0])[:, None] > np.percentile(ys, 30)) & (np.arange(gt.shape[0])[:, None] < np.percentile(ys, 70))
    cols = (np.arange(gt.shape[1])[None, :] > np.percentile(xs, margin * 100)) & (np.arange(gt.shape[1])[None, :] < np.percentile(xs, 100 - margin * 100))
    ok = fin & rows & cols & (E > 0)
    out = {}
    for lab, w in zip(("first", "second"), waves or (ex + 20, ex + 40)):
        b = int(np.argmin(abs(em - w)))
        raw, nrm = [], []
        for a, c in zip(edges[:-1], edges[1:]):
            m = ok & (gt >= a) & (gt < c)
            raw.append(img[:, :, b][m].mean() if m.any() else np.nan)
            nrm.append((img[:, :, b] / np.where(E > 0, E, np.nan))[m].mean() if m.any() else np.nan)
        out[lab] = (float(em[b]), np.array(raw), np.array(nrm))
    return out


def _plot_band_curves(a_, ctr, cur, title):
    for lab, col in (("first", C["b"]), ("second", C["c"])):
        wl, raw, nrm = cur[lab]
        a_.plot(ctr, raw / raw[-1], "--", color=col, alpha=0.6, label=f"{wl:.0f} nm, raw")
        a_.plot(ctr, nrm / nrm[-1], "-o", ms=3, color=col, label=f"{wl:.0f} nm, / irradiance")
    a_.axhline(1, color=C["g"], lw=0.6)
    a_.set_xlabel("local cover thickness (mm)")
    a_.set_ylabel("band radiance (norm.)")
    a_.set_title(title, fontsize=8)
    a_.legend(fontsize=7, handlelength=1.6, borderaxespad=0.2)


def fig_blk_wedge():
    with plt.rc_context(PAPER_RC):
        _fig_blk_wedge()


def _fig_blk_wedge():
    photons = 1_000_000 if QUICK else 5_000_000
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 3.9), layout="constrained")
    NUM["blk_wedge"] = {}
    # ---- layered_wedge_phantom: buried fluorescein under a turbid cover (0 -> 2 mm)
    exp = _blk_exp("layered_wedge_phantom", 490.0, photons)
    res = cap.capture(exp, scale=0.5)
    edges = np.linspace(0.15, 1.85, 9)
    cur = _band_curves(res, "layer_thickness:cover", edges)
    _plot_band_curves(ax[0, 2], 0.5 * (edges[:-1] + edges[1:]), cur, "(c) band curves, wedge phantom")
    NUM["blk_wedge_band_curves"] = {k: dict(band_nm=v[0], raw=v[1].tolist(), per_irradiance=v[2].tolist()) for k, v in cur.items()}
    dat, ctr, em = _bin_components(exp, res, "layer_thickness:cover", edges, {"fl": lambda k: k.startswith("dye:fluorescein")})
    sig = np.array([d["fl"].sum() if d else np.nan for d in dat])
    blue, red = (em >= 495) & (em < 525), (em >= 535) & (em < 560)
    ratio = np.array([d["fl"][blue].sum() / d["fl"][red].sum() if d else np.nan for d in dat])
    ax[0, 0].semilogy(ctr, sig / sig[0], "o-", ms=3, color=C["c"], label="3D Monte Carlo")
    mu = -np.polyfit(ctr[np.isfinite(sig)], np.log(sig[np.isfinite(sig)] / sig[0]), 1)[0]
    ax[0, 0].semilogy(ctr, np.exp(-mu * (ctr - ctr[0])) , "--", color=C["k"], lw=1, label=f"fit, $\\mu$ = {mu:.2f} mm$^{{-1}}$")
    ax[0, 0].set_xlabel("local cover thickness (mm)")
    ax[0, 0].set_ylabel("signal / irradiance (norm.)")
    ax[0, 0].set_title("(a) wedge phantom, 490 nm")
    ax[0, 0].legend(fontsize=7, borderaxespad=0.2)
    ax[0, 1].plot(ctr, ratio, "o-", ms=3, color=C["d"])
    ax[0, 1].set_xlabel("local cover thickness (mm)")
    ax[0, 1].set_ylabel("blue / red band ratio")
    ax[0, 1].set_title("(b) spectral distortion")
    ax[0, 1].set_ylim(0, 1.25 * np.nanmax(ratio))        # full scale: a 2 % wobble must not look like a trend
    NUM["blk_wedge"]["wedge"] = dict(thickness=ctr.tolist(), signal=sig.tolist(), blue_red=ratio.tolist(), mu_eff_fit_per_mm=float(mu))
    # ---- tissue_under_collagen: buried NADH/FAD tissue under a collagen cover (1.5 -> 0.05 mm)
    exp2 = _blk_exp("tissue_under_collagen", 365.0, photons)
    res2 = cap.capture(exp2, scale=0.5)
    edges2 = np.linspace(0.12, 1.45, 9)
    dat2, ctr2, em2 = _bin_components(exp2, res2, "layer_thickness:collagen", edges2, {
        "tissue": lambda k: k.endswith("@tissue"), "collagen": lambda k: k.endswith("@collagen")})
    cur2 = _band_curves(res2, "layer_thickness:collagen", edges2, waves=(440.0, 520.0))
    _plot_band_curves(ax[1, 2], 0.5 * (edges2[:-1] + edges2[1:]), cur2, "(f) band curves, tissue under collagen")
    sig2 = np.array([d["tissue"].sum() if d else np.nan for d in dat2])
    b2, r2 = (em2 >= 420) & (em2 < 480), (em2 >= 520) & (em2 < 600)
    ratio2 = np.array([d["tissue"][b2].sum() / d["tissue"][r2].sum() if d else np.nan for d in dat2])
    ax[1, 0].semilogy(ctr2, sig2 / sig2[0], "o-", ms=3, color=C["c"])
    ax[1, 0].set_xlabel("local collagen thickness (mm)")
    ax[1, 0].set_ylabel("signal / irradiance (norm.)")
    ax[1, 0].set_title("(d) tissue under collagen, 365 nm")
    ax[1, 1].plot(ctr2, ratio2, "o-", ms=3, color=C["d"])
    ax[1, 1].set_xlabel("local collagen thickness (mm)")
    ax[1, 1].set_ylabel("blue / green-red band ratio")
    ax[1, 1].set_title("(e) spectral distortion")
    ax[1, 1].set_ylim(0, 1.25 * np.nanmax(ratio2))
    NUM["blk_wedge"]["tissue"] = dict(thickness=ctr2.tolist(), signal=sig2.tolist(), blue_red=ratio2.tolist())
    save(fig, "fig15_blk_wedge_response.png")
    # ---- (f) collagen autofluorescence vs collagen thickness
    col = np.array([d["collagen"].sum() if d else np.nan for d in dat2])
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), layout="constrained")
    ax[0].plot(ctr2, col / np.nanmax(col), "o-", ms=3, color=C["b"], label="collagen + elastin AF")
    ax[0].plot(ctr2, sig2 / np.nanmax(sig2), "s-", ms=3, color=C["a"], label="buried NADH + FAD")
    # saturating law  A (1 - exp(-t/ell))  for the cover's own emission (excitation attenuated both ways)
    ok = np.isfinite(col)
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(lambda t, A, ell: A * (1 - np.exp(-t / ell)), ctr2[ok], col[ok] / np.nanmax(col), p0=[1.5, 0.8])
        tt = np.linspace(ctr2[ok][0], ctr2[ok][-1], 50)
        ax[0].plot(tt, popt[0] * (1 - np.exp(-tt / popt[1])), ":", color=C["k"], label=f"saturating fit, $\\ell$ = {popt[1]:.2f} mm")
        NUM["blk_wedge"]["collagen_ell_mm"] = float(popt[1])
    except Exception:
        pass
    ax[0].set_xlabel("local collagen thickness (mm)")
    ax[0].set_ylabel("signal / irradiance (norm.)")
    ax[0].set_title("(a) cover autofluorescence vs buried signal")
    ax[0].legend(fontsize=7, loc="center right", borderaxespad=0.2)
    for d, c_, lab in ((dat2[1], C["a"], f"{ctr2[1]:.2f} mm"), (dat2[-2], C["b"], f"{ctr2[-2]:.2f} mm")):
        if d:
            ax[1].plot(em2, d["tissue"] + d["collagen"], color=c_, label=f"total, {lab}")
            ax[1].plot(em2, d["collagen"], color=c_, ls="--", lw=1.2, label=f"collagen AF, {lab}")
    ax[1].set_xlabel("emission (nm)")
    ax[1].set_ylabel("band radiance / irradiance")
    ax[1].set_title("(b) spectra at a thin and a thick cover")
    ax[1].legend(fontsize=7, borderaxespad=0.2)
    NUM["blk_wedge"]["collagen_af"] = dict(thickness=ctr2.tolist(), signal=col.tolist())
    save(fig, "fig17_blk_collagen_autofluorescence.png")


# ---------------------------------------------------------------- (e) light field
def fig_blk_field():
    from matplotlib.colors import LogNorm
    photons = 1_000_000 if QUICK else 8_000_000
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.4), layout="constrained")
    NUM["blk_field"] = {}
    for a_, (name, ex_nm) in zip(axs, (("layered_wedge_phantom", 490.0), ("tissue_under_collagen", 365.0))):
        exp = _blk_exp(name, ex_nm, photons)
        res = cap.capture(exp, scale=0.25)
        oid = res.blocks[0]["object_id"]
        vol = res.volumes[oid]
        fl = np.asarray(vol["fluence"][res.excitations_nm[0]], float)                  # (nx, nz, L, S)
        grid = vol["grid"]
        shape = exp.objects[0].shape
        pc = np.zeros((len(shape.layers) + 1, 3))
        pc[1:] = presets_planes(shape)
        nx, nz, L, S = fl.shape
        col = fl.sum(axis=(2, 3))
        iz = int(np.unravel_index(np.argmax(col), col.shape)[1])
        xe = grid.x_edges_mm
        zc = 0.5 * (grid.z_edges_mm[iz] + grid.z_edges_mm[iz + 1])
        H = pc[:, 0:1] + pc[:, 1:2] * xe[None, :] + pc[:, 2:3] * zc                          # (L+1, nx+1) plane heights
        rows_y, rows_c = [H[0]], []
        for l in range(L):
            for s in range(S):
                rows_y.append(H[l] + (H[l + 1] - H[l]) * (s + 1) / S)
                rows_c.append(fl[:, iz, l, s])
        Y = np.array(rows_y)
        Cm = np.array(rows_c)
        X = np.tile(xe, (Y.shape[0], 1))
        Cm = np.where(Cm > 0, Cm, np.nan)
        vmax = np.nanmax(Cm)
        im = a_.pcolormesh(X, Y, np.ma.masked_invalid(Cm), norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax), cmap="inferno", shading="flat")
        for l in range(L + 1):
            a_.plot(xe, H[l], color="white" if l else C["g"], lw=0.9 if l else 0.5)
        for l, nm in enumerate(vol["layer_names"]):
            a_.text(xe[0] + 0.6, 0.5 * (H[l, 0] + H[l + 1, 0]) if H[l + 1, 0] - H[l, 0] > 0.35 else H[l + 1, 0] + 0.25, nm, color="white" if l < L - 1 else "#222", fontsize=7.5,
                    va="center", bbox=dict(fc="#00000066", ec="none", pad=1.0))
        src = exp.source()
        pos, tgt = np.array(src.position_mm, float), np.array(src.target_mm, float)
        d = tgt - pos
        u = d / np.linalg.norm(d)
        top = float(H[-1].mean())
        hit = pos + d * ((top - pos[1]) / d[1])
        ang = np.degrees(np.arccos(abs(u[1])))
        uxy = u[[0, 1]] / np.hypot(u[0], u[1])
        a_.annotate("", xy=(hit[0], top + 0.1), xytext=(hit[0] - 6.0 * uxy[0], top + 0.1 - 6.0 * uxy[1]),
                    arrowprops=dict(arrowstyle="-|>", color=C["a"], lw=1.8))
        a_.set_ylim(-0.3, top + 6.5)
        a_.set_xlim(xe[0], xe[-1])
        a_.set_xlabel("x (mm)")
        a_.set_ylabel("height above the stage y (mm)")
        a_.set_aspect("equal")
        a_.set_title(f"{name}, {ex_nm:g} nm\nslice z = {zc:.1f} mm, beam {ang:.0f}° from the normal", fontsize=9)
        cb = fig.colorbar(im, ax=a_, orientation="horizontal", location="bottom", fraction=0.07, pad=0.14, aspect=40)
        cb.set_label("fluence (mW/mm², per step power)")
        NUM["blk_field"][name] = dict(ex_nm=ex_nm, slice_z_mm=float(zc), beam_angle_deg=float(ang),
                                      peak_fluence=float(vmax))
    fig.suptitle("Excitation light field inside the layered blocks (3D Monte Carlo, track-length fluence; log colour scale, "
                 "white lines: layer interfaces)", fontsize=9.5)
    save(fig, "fig16_blk_light_field.png")


def presets_planes(shape):
    from spectraforge.studio import transport3d as T3
    return T3.planes_from_layers(shape.footprint_mm, [l.thickness for l in shape.layers])


FIGS = {"mcml": fig_mcml, "diffusion": fig_diffusion, "closed": fig_closed_vs_volume, "legacy": fig_legacy,
        "inner": fig_inner_filter, "bleach": fig_bleaching, "noise": fig_noise, "dof": fig_dof, "angle": fig_angle, "laser": fig_laser,
        "blk_validation": fig_blk_validation, "blk_bottom": fig_blk_bottom, "blk_peel": fig_blk_peel,
        "blk_wedge": fig_blk_wedge, "blk_field": fig_blk_field}


def replot_blk_reference_from_json():
    """Paper-style fig12 from the stored ``blk_validation`` numbers (no simulation): case/quantity labels, values and
    the percent deviation in one full-width bar chart.  ``python -c "import studio_validation_figures as V; V.replot_blk_reference_from_json()"``."""
    rows = json.loads((OUT / "validation_numbers.json").read_text())["blk_validation"]
    short = {"slab benchmark (n=1)": "slab, n=1", "slab index mismatch n=1.4": "slab, n=1.4", "slab tissue-like": "slab, tissue-like",
             "fluorescent layer, 2 mm, 60 mm footprint": "fluor. layer 2 mm", "two identical layers (0.4+0.6 mm) vs one (1.0 mm)": "2 layers vs 1"}
    qn = {"R+T+A+spec": "R+T+A+spec.", "fluorescence radiance": "fluorescence", "elastic radiance / E": "elastic / E"}
    labels, cur = [], ""
    for r in rows:
        cur = short.get(r["case"], r["case"]) if r["case"] else cur
        labels.append(f"{cur}, {qn.get(r['quantity'], r['quantity'])}")
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.16, 0.17 * len(rows) + 0.75), layout="constrained")
        y = np.arange(len(rows))[::-1]
        d = [100 * r["rel_err"] for r in rows]
        ax.barh(y, d, height=0.7, color=[C["a"] if abs(x) < 1.0 else C["b"] for x in d])
        ax.axvspan(-1, 1, color=C["g"], alpha=0.2, lw=0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(-4.6, 4.2)
        for yi, r, x in zip(y, rows, d):
            ax.text(4.15, yi, f"{r['engine']:.4g} vs {r['reference']:.4g}", ha="right", va="center", fontsize=7)
            ax.text(x + (0.08 if x >= 0 else -0.08), yi, f"{x:+.2f}", ha="left" if x >= 0 else "right", va="center", fontsize=6.5, color="0.3")
        ax.set_xlabel("3D engine vs reference (%)   [grey band: $\\pm$1 %; right: engine vs reference value]")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        save(fig, "fig12_blk_3d_vs_reference.png")


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(FIGS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    QUICK = a.quick
    todo = [k for k in FIGS if not a.only or k in a.only.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "validation_numbers.json"
    if jf.exists():
        NUM.update(json.loads(jf.read_text()))
    for k in todo:
        t = time.perf_counter()
        FIGS[k]()
        print(f"[{k}] {time.perf_counter() - t:.1f} s", flush=True)
        jf.write_text(json.dumps(NUM, indent=1, default=float))


if __name__ == "__main__":
    main()
