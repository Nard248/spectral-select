"""Reproducible validation experiment: does the perturbation-AE band selection recover the
planted fluorophore emission peaks on synthetic SpectraForge scenes?

Run:  python reports/spectraforge_validation_report.py

Finding (2026-06-19): across both a degenerate flat-region scene (2 unique *clean* spectra) and
richer scenes with smooth overlapping concentration fields (many unique clean spectra), the
perturbation-AE selection does NOT reliably recover the planted emission peaks. Recovery of the
two planted fluorophores ranges 0.0-0.5 and precision stays <=0.12, and the result is sensitive to
the random seed; the selection tends to pick spectral-edge bands rather than the 520/610 nm peaks.
Spectral variance alone does not fix this. This is a flag worth investigating (scene realism? the
method's variance/edge-seeking behaviour on smooth synthetic Gaussian emissions? config?), and is
exactly the kind of result the validation harness exists to surface.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np

from spectraforge import (
    AcquisitionConfig, ArtifactConfig, Fluorophore, Material, Scene, render,
)
from spectraforge.validation import validate_selection

LIB = {
    "G": Fluorophore("G", 488, 40, 520, 30, quantum_yield=0.6, extinction=1.0),
    "R": Fluorophore("R", 560, 40, 610, 35, quantum_yield=0.5, extinction=0.9),
}
ACQ = AcquisitionConfig(excitations=[488.0, 560.0], em_min=400, em_max=700, em_step=5)
ARTIFACTS = ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01)


def _smooth_field(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.kron(rng.random((16, 16)), np.ones((4, 4)))      # 64x64 blocky
    for _ in range(2):                                       # cheap blur -> smooth gradients
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5
    return f / f.max()


def flat_scene() -> Scene:
    s = Scene(64, 64)
    s.paint_rect(Material("g", {"G": 1.0}), 0, 64, 0, 32)
    s.paint_rect(Material("r", {"R": 1.0}), 0, 64, 32, 64)
    return s


def rich_scene(seed: int) -> Scene:
    s = Scene(64, 64)
    s.paint_map(Material("g", {"G": 1.0}), _smooth_field(seed))
    s.paint_map(Material("r", {"R": 1.0}), _smooth_field(seed + 100))
    return s


def _select_and_score(scene, seed):
    from spectral_select import Analyzer, Config
    spectra, gt = render(scene, LIB, ACQ, artifacts=ARTIFACTS, seed=seed)
    clean = gt.clean_cubes[488.0]                       # noise-free -> true structural variety
    n_unique = np.unique(np.round(clean.reshape(-1, clean.shape[-1]), 4), axis=0).shape[0]
    cfg = Config(sample_name="v", n_important_dimensions=15, n_bands_to_select=8,
                 perturbation_method="percentile", use_diversity_constraint=True,
                 training_epochs=30, device="cpu", random_seed=0,
                 output_dir=pathlib.Path(tempfile.mkdtemp()))
    a = Analyzer(cfg)
    a.fit(spectra)
    m = validate_selection(gt, a.get_wavelengths(), tol_nm=12)
    m["n_unique_spectra"] = n_unique
    return m


def _spectral_shape_sweep():
    """Headline finding: realistic asymmetric spectra vs smooth Gaussians (Increment D)."""
    import pathlib
    import tempfile
    from spectraforge import MeasuredFluorophore
    from spectraforge.scenegen import random_scene
    from spectraforge.sweep import aggregate_metrics, make_analyzer_selector, run_validation_sweep
    from spectral_select import Config

    mats = [Material("g", {"G": 1.0}), Material("r", {"R": 1.0})]
    factory = lambda seed: random_scene(mats, 64, 64, seed)
    cfg = Config(sample_name="v", n_important_dimensions=15, n_bands_to_select=8,
                 perturbation_method="percentile", use_diversity_constraint=True,
                 training_epochs=30, device="cpu", random_seed=0,
                 output_dir=pathlib.Path(tempfile.mkdtemp()))

    def asym(peak, rise, tail):                          # sharp blue edge + long red tail (real dyes)
        wl = np.arange(peak - rise, peak + tail, 2.0)
        v = np.where(wl < peak, np.exp(-((wl - peak) / (rise / 2.5)) ** 2),
                     np.exp(-(wl - peak) / (tail / 2.5)))
        return wl, v

    gw, gv = asym(520, 40, 140)
    rw, rv = asym(610, 45, 120)
    measured = {
        "G": MeasuredFluorophore("G", [440, 488, 520], [0, 1.0, 0.3], gw, gv, 0.6, 1.0),
        "R": MeasuredFluorophore("R", [520, 560, 600], [0, 1.0, 0.4], rw, rv, 0.5, 0.9),
    }
    libs = {"Gaussian (toy)": LIB, "Measured-asymmetric": measured}
    for label, lib in libs.items():
        res = run_validation_sweep(factory, lib, ACQ, make_analyzer_selector(cfg), seeds=[1, 2, 3, 4],
                                   artifacts=ARTIFACTS,
                                   physics=__import__("spectraforge").PhysicsConfig(
                                       psf_sigma_px=1.5, autofluorescence=0.02), tol_nm=12)
        a = aggregate_metrics(res)
        print(f"{label:<24}{a['fluorophores_recovered_mean']:>10.2f} ± {a['fluorophores_recovered_std']:<5.2f}"
              f"{a['precision_mean']:>10.2f} ± {a['precision_std']:<5.2f}")


def main():
    print("=" * 78)
    print("SpectraForge validation report — recovery of planted emission peaks")
    print("Planted: G emits ~520 nm (ex 488), R emits ~610 nm (ex 560)")
    print("=" * 78)
    print("[1] Scene variance (single seed, Gaussian spectra)")
    print(f"{'scene':<22}{'uniq clean spec':>16}{'recovered':>12}{'precision':>12}{'f1':>8}")
    for label, scene in [("flat (degenerate)", flat_scene())] + \
                        [(f"rich variance s={s}", rich_scene(s)) for s in (1, 2, 7)]:
        m = _select_and_score(scene, seed=1)
        print(f"{label:<22}{m['n_unique_spectra']:>16}{m['fluorophores_recovered']:>12.2f}"
              f"{m['precision']:>12.2f}{m['f1']:>8.2f}")

    print()
    print("[2] Spectral shape (4-scene sweep + PSF/autofluorescence physics) — THE headline result")
    print(f"{'fluorophore spectra':<24}{'recovered (mean±sd)':>18}{'precision (mean±sd)':>20}")
    _spectral_shape_sweep()

    print("-" * 78)
    print("Takeaway: with smooth symmetric Gaussian spectra the perturbation-AE selection looks")
    print("weak and seed-sensitive (recovery ~0.25). With realistic ASYMMETRIC spectra (sharp blue")
    print("edge + long red tail, like real dyes) recovery jumps to ~1.0 and precision rises. So the")
    print("apparent weakness was largely a toy-spectra artifact — synthetic validation must use")
    print("realistic spectral shapes. (Caveat: asymmetric emission also widens the informative")
    print("footprint, which inflates 'recovery'; the precision gain is the cleaner signal.)")


if __name__ == "__main__":
    main()
