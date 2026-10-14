"""Shared helpers of the C3 campaign experiments E5 (information-optimal acquisition), E6 (thickness mapping) and
E7 (depth-dependent unmixing failure) of the SpectralForge Studio paper.

Scene (``build_scene``): the E2 tissue-optics structure rebuilt on the literature tissue materials: 4 mm reticular dermis
substrate, 1 mm buried NADH/FAD/collagen/elastin layer (``dermis_papillary`` optics), dense-collagen cover
(``collagen_dense``, Jacques 2013 Mie+Rayleigh optics) of thickness t; the lab's full 8-line TLS protocol (310-430 nm) and
Nuance LCTF (31 bands, 420-720 nm): 8 x 31 = 248 channels.  Simulator evaluations are expensive (about 40 s per
8-excitation evaluation on a loaded machine), so every expensive object (surrogates, finite-difference Jacobians,
captures) is cached on disk (``~/.cache/spectralforge/c3``, plus the surrogate cache of ``surrogate.py``).

Systematic model error (see addendum in docs/superpowers/briefs/31-34-campaign-experiments.md): the noise model of
``surrogate.NoiseModel`` has an option ``model_error_rel`` (eps): per-channel variance += (eps S_c)^2.  eps is derived
from the sim-to-real residuals of the calibrated twin (``derive_eps``).
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraforge.studio import api  # noqa: E402
from spectraforge.studio import presets as P  # noqa: E402
from spectraforge.studio.schema import DyeAmount, Experiment, LayeredBlockShape  # noqa: E402
from spectraforge.studio.surrogate import ParamSpec, Surrogate  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
CACHE = Path.home() / ".cache" / "spectralforge" / "c3"
CACHE.mkdir(parents=True, exist_ok=True)

SCALE = 0.25                 # capture resolution (fast mode: all excitations, preview resolution)
PREVIEW_FRACTION = 0.15      # MC photon fraction (as in W4 first pass)
T_LAYER = "/objects/0/shape/layers/2/thickness/"
T_PATHS = (T_LAYER + "thickness_mm", T_LAYER + "end_thickness_mm")
UNIFORM = {T_LAYER + "kind": "uniform"}
NADH0, FAD0 = 60.0, 40.0     # buried-layer concentrations (uM)
EPS_GRID = [0.0, 0.005, 0.01, 0.02, 0.05]

# IEEE column widths
COL, DCOL = 3.5, 7.16
# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False})


def savefig(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ scene
def build_scene(t: float, t_end: float | None = None, nadh: float = NADH0, fad: float = FAD0, cover_mus500=None,
                seed: int = 0, exposure_scale: float = 1.0, footprint=(40.0, 40.0), excitations=None) -> Experiment:
    """Collagen-covered NADH/FAD tissue (E2 structure) with the lab's 8-line TLS protocol.  ``t_end`` makes the cover a
    linear wedge along x (t -> t_end)."""
    mats = {m.id: m for m in P.builtin_materials()}
    cover = mats["collagen_dense"].model_copy(update={"id": "cover"})
    if cover_mus500 is not None:
        cover = cover.model_copy(update={"scattering": cover.scattering.model_copy(
            update={"mus_prime_500_per_mm": float(cover_mus500)})})
    buried = mats["dermis_papillary"].model_copy(update={"id": "buried", "dyes": [
        DyeAmount(fluorophore="NADH", concentration_uM=nadh), DyeAmount(fluorophore="FAD", concentration_uM=fad),
        DyeAmount(fluorophore="collagen", concentration_uM=250.0), DyeAmount(fluorophore="elastin", concentration_uM=40.0)]})
    exp = P.tissue_under_collagen()
    block = LayeredBlockShape(footprint_mm=footprint, layers=[
        P._layer("substrate", "dermis_reticular", 4.0), P._layer("buried", "buried", 1.0),
        P._layer("cover", "cover", t, t_end)])
    src = exp.sources[0]
    seq = [s.model_copy(update={"exposure_ms": s.exposure_ms * exposure_scale}) for s in src.sequence
           if excitations is None or float(s.wavelength_nm) in excitations]
    exp = exp.model_copy(update={
        "name": "C3 tissue under collagen cover", "materials": list(mats.values()) + [cover, buried],
        "objects": [P._obj("block", "block", block, "cover", (0, 0, 0))],
        "sources": [src.model_copy(update={"sequence": seq})]})
    exp = api.set(exp, "/acquisition/seed", seed)
    return api.set(exp, "/acquisition/volume_mc/preview_fraction", PREVIEW_FRACTION)


def mat_index(exp: Experiment, mid: str) -> int:
    return [m.id for m in exp.materials].index(mid)


def tissue_params(exp: Experiment, which=("thickness",)) -> list[ParamSpec]:
    """Parameter specs of the scene: thickness (mm), buried NADH (uM), cover reduced scattering at 500 nm (1/mm)."""
    out = []
    if "thickness" in which:
        out.append(ParamSpec("cover_mm", T_PATHS, lo=0.05, hi=1.5))
    if "nadh" in which:
        out.append(ParamSpec("NADH_uM", (f"/materials/{mat_index(exp, 'buried')}/dyes/0/concentration_uM",),
                             lo=30.0, hi=120.0))
    if "mus" in which:
        out.append(ParamSpec("cover_mus500", (f"/materials/{mat_index(exp, 'cover')}/scattering/mus_prime_500_per_mm",),
                             lo=1.5, hi=4.5))
    return out


# ------------------------------------------------------------------------------------------------ caching
def cached(name: str, fn, *, force: bool = False):
    """Pickle cache of an expensive computation (name -> file in ~/.cache/spectralforge/c3)."""
    f = CACHE / f"{name}.pkl"
    if f.exists() and not force:
        try:
            return pickle.loads(f.read_bytes())
        except Exception:  # noqa: BLE001 - corrupt cache
            f.unlink(missing_ok=True)
    t0 = time.time()
    obj = fn()
    f.write_bytes(pickle.dumps(obj, protocol=4))
    print(f"[cache] built {name} in {time.time() - t0:.0f} s", flush=True)
    return obj


def scale_surrogate(sur: Surrogate, f: float) -> Surrogate:
    """Surrogate of the same scene acquired with exposures multiplied by ``f`` (signal and dark scale, read noise does not)."""
    nm = replace(sur.noise, dark_e=sur.noise.dark_e * f)
    return Surrogate(sur.params, sur.theta, sur.values_e * f, sur.channels, nm, mode=sur.mode, grid_axes=sur.grid_axes,
                     interp=sur.interp, smooth=sur.smooth, meta=sur.meta)


def with_eps(sur: Surrogate, eps: float) -> Surrogate:
    """Shallow copy of ``sur`` whose noise model carries the systematic relative error ``eps``."""
    s = Surrogate(sur.params, sur.theta, sur.values_e, sur.channels, sur.noise.with_model_error(eps), mode=sur.mode,
                  grid_axes=sur.grid_axes, interp=sur.interp, smooth=sur.smooth, meta=sur.meta)
    return s


# ------------------------------------------------------------------------------------------------ eps from sim-to-real
def derive_eps() -> dict:
    """Relative systematic model error eps from the sim-to-real residuals of the calibrated digital twin
    (``docs/studio/figures/sim2real_numbers.json``).

    * ``rms_rel``: RMS over the bands of (model - data)/data of the fitted spectra, per sample and excitation
      (band ripple, band-edge and calibration errors of the twin AFTER fitting its few free parameters).  We take the
      median over the 9 collagen fits (corrected K=1) and the 16 drops: this is an UPPER bound for the pure model
      error, because it still contains the camera noise of the real spectra.
    * ``sam``: the mean spectral angle theta of the same fits gives a shape-only relative error 2 sin(theta/2) (the
      relative L2 distance of two unit vectors at angle theta), the lower of the two estimates because it ignores
      a per-excitation amplitude error.
    """
    d = json.loads((REPO / "docs/studio/figures/sim2real_numbers.json").read_text())
    col = [v["corrected_K1"]["rms_rel"] for v in d["collagen_fits"].values()]
    col_sam = [v["corrected_K1"]["sam_mean_deg"] for v in d["collagen_fits"].values()]
    dr = [v["rms_rel"] for v in d["drops"]["per_drop"].values()]
    dr_sam = [v["sam_mean_deg"] for v in d["drops"]["per_drop"].values()]
    allrms = np.array(col + dr)
    allsam = np.array(col_sam + dr_sam)
    eps_rms = float(np.median(allrms))
    eps_sam = float(np.median(2.0 * np.sin(np.radians(allsam) / 2.0)))
    return {"eps_rms_rel_median": eps_rms, "eps_rms_rel_range": [float(allrms.min()), float(allrms.max())],
            "eps_sam_median": eps_sam, "sam_deg_median": float(np.median(allsam)),
            "sam_deg_range": [float(allsam.min()), float(allsam.max())], "n_fits": int(allrms.size),
            "eps_ref": round(eps_rms, 3),
            "note": "eps_ref = median RMS relative residual of calibrated (corrected K=1) fits of 9 collagen spectra + 16 drops; "
                    "upper bound (includes measurement noise); nominal uncalibrated model: 0.12-0.20"}


def provenance_block(extra=None) -> dict:
    from spectraforge.studio.provenance import provenance
    return provenance(extra)


def write_results(eid: str, obj: dict):
    d = RES / eid
    d.mkdir(parents=True, exist_ok=True)
    obj = dict(obj)
    obj["provenance"] = provenance_block()
    (d / "results.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    return d


# ------------------------------------------------------------------------------------------------ shared analysis objects
def tissue_surrogate(quick: bool = False):
    """1-D thickness surrogate of the tissue scene (8 excitations x 31 bands; quick: 2 excitations, 5 knots)."""
    from spectraforge.studio.surrogate import build_surrogate
    exc = [340.0, 365.0] if quick else None
    exp0 = build_scene(0.6, excitations=exc)
    sp = tissue_params(exp0, ("thickness",))[0]
    grid = [0.05, 0.2, 0.5, 1.0, 1.5] if quick else [0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5]
    sur = build_surrogate(exp0, [sp], {"cover_mm": grid}, scale=SCALE, smooth=None if quick else "gcv", excitations=exc)
    return exp0, sp, grid, sur


def augment(fi, mode: str):
    """Add nuisance gain parameters to a Fisher result: one global amplitude (``amp``) or one per excitation (``gain``).
    The nuisance Jacobian column of a gain g (nominal 1) is the mean signal of the channels it multiplies."""
    from spectraforge.studio import information as info
    mean = fi.mean_e
    if mode == "amp":
        cols, names = [mean], ["amp"]
    elif mode == "gain":
        exs = sorted(set(fi.channels.ex_nm.tolist()))
        cols = [np.where(np.abs(fi.channels.ex_nm - e) < 1e-6, mean, 0.0) for e in exs]
        names = [f"gain{e:g}" for e in exs]
    else:
        raise ValueError(mode)
    jac = np.hstack([fi.jac_e, np.stack(cols, axis=1)])
    th = np.concatenate([fi.theta, np.ones(len(cols))])
    return info.fisher_from_jacobian(list(fi.names) + names, th, jac, mean, fi.noise, fi.channels, fi.exposure_s, fi.meta)


def with_eps_fisher(fi, eps: float):
    from spectraforge.studio import information as info
    return info.fisher_from_jacobian(fi.names, fi.theta, fi.jac_e, fi.mean_e, fi.noise.with_model_error(eps),
                                     fi.channels, fi.exposure_s, fi.meta)


def build_phantom(t0: float, t1: float | None, level: float = 1.0, seed: int = 0, excitations=None) -> Experiment:
    """The generic layered wedge phantom (fluorescein layer under a turbid Intralipid-like cover), exposures x ``level``."""
    exp = P.layered_wedge_phantom()
    src = exp.sources[0]
    seq = [s.model_copy(update={"exposure_ms": s.exposure_ms * level}) for s in src.sequence
           if excitations is None or float(s.wavelength_nm) in excitations]
    exp = exp.model_copy(update={"sources": [src.model_copy(update={"sequence": seq})]})
    blk = exp.objects[0].shape
    layers = list(blk.layers)
    layers[2] = P._layer("cover", layers[2].material_id, t0, t1)
    exp = exp.model_copy(update={"objects": [exp.objects[0].model_copy(update={"shape": blk.model_copy(update={"layers": layers})})]})
    exp = api.set(exp, "/acquisition/seed", seed)
    return api.set(exp, "/acquisition/volume_mc/preview_fraction", PREVIEW_FRACTION)


def phantom_surrogate(quick: bool = False):
    from spectraforge.studio.surrogate import build_surrogate
    exc = [340.0, 380.0] if quick else None
    exp0 = build_phantom(0.5, 0.5, excitations=exc)
    sp = ParamSpec("cover_mm", T_PATHS, lo=0.05, hi=2.0)
    grid = [0.05, 0.4, 1.0, 2.0] if quick else [0.05, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0]
    sur = build_surrogate(exp0, [sp], {"cover_mm": grid}, scale=SCALE, smooth=None if quick else "gcv", excitations=exc)
    return exp0, sp, grid, sur


def wedge_experiment(kind: str, level: float, quick: bool = False, seed: int = 1) -> Experiment:
    if kind == "tissue":
        return build_scene(0.1, 1.4, exposure_scale=level, seed=seed, excitations=[340.0, 365.0] if quick else None)
    if kind == "phantom":
        return build_phantom(2.0, 0.0, level, seed=seed, excitations=[340.0, 380.0] if quick else None)
    raise ValueError(kind)


def get_capture(kind: str, level: float, quick: bool = False, seed: int = 1):
    """Noisy capture of a cover-thickness wedge (cached).  ``kind``: ``tissue`` (tissue-optics wedge, cover 0.1 -> 1.4 mm
    along x) or ``phantom`` (layered wedge phantom, cover 2 -> 0 mm); exposures multiplied by ``level``.  The cache holds
    our own pickles written by this script (local files, not untrusted input)."""
    exp = wedge_experiment(kind, level, quick, seed)
    return cached(f"cap_{kind}_{'q' if quick else 'f'}_L{level:g}_s{seed}", lambda: api.capture(exp, "fast", scale=SCALE))
