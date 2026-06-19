"""SpectraForge validation report — perturbation-AE band selection vs a chance baseline.

CORRECTION NOTICE (2026-06-20): earlier versions of this report claimed the method "recovers the
planted emission bands", weakly with Gaussian spectra and strongly with realistic asymmetric/
measured spectra. Adversarial review showed those conclusions were ARTIFACTS of a near-saturated
ground-truth mask and the absence of a chance baseline. They are WITHDRAWN. The honest picture:

  * The broad informative-band mask covers most of the emission grid (mask% below), so broad-mask
    recovery/precision saturate and a uniformly-RANDOM selector matches or beats the AE on them.
  * On a TIGHT metric (hit a fluorophore's true emission PEAK within tol) the AE scores ~0 while
    random scores ~0.33 — the AE systematically selects edge/tail bands, not peaks.
  * BUT peaks may not be the right target: a discriminative selector can legitimately prefer
    non-peak bands. So this is NOT proof the method fails. Verdict: the harness, as configured, is
    INCONCLUSIVE; it needs a discriminability-grounded ground truth, not emission peaks.

See reports/fpbase_validation.py for the same comparison on real measured FPbase spectra.

Run:  python reports/spectraforge_validation_report.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np

from spectraforge import (
    AcquisitionConfig, ArtifactConfig, Fluorophore, Material, MeasuredFluorophore, PhysicsConfig,
)
from spectraforge.scenegen import random_scene
from spectraforge.sweep import (
    aggregate_metrics, make_analyzer_selector, make_random_selector, run_validation_sweep,
)

ACQ = AcquisitionConfig(excitations=[488.0, 560.0], em_min=400, em_max=700, em_step=5)
ARTIFACTS = ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01)
PHYSICS = PhysicsConfig(psf_sigma_px=1.5, autofluorescence=0.02)

GAUSSIAN = {
    "G": Fluorophore("G", 488, 40, 520, 30, quantum_yield=0.6, extinction=1.0),
    "R": Fluorophore("R", 560, 40, 610, 35, quantum_yield=0.5, extinction=0.9),
}


def _asym(peak, rise, tail):
    wl = np.arange(peak - rise, peak + tail, 2.0)
    v = np.where(wl < peak, np.exp(-((wl - peak) / (rise / 2.5)) ** 2),
                 np.exp(-(wl - peak) / (tail / 2.5)))
    return wl, v


def _measured_library():
    gw, gv = _asym(520, 40, 140)
    rw, rv = _asym(610, 45, 120)
    return {
        "G": MeasuredFluorophore("G", [440, 488, 520], [0, 1.0, 0.3], gw, gv, 0.6, 1.0),
        "R": MeasuredFluorophore("R", [520, 560, 600], [0, 1.0, 0.4], rw, rv, 0.5, 0.9),
    }


def main():
    mats = [Material("g", {"G": 1.0}), Material("r", {"R": 1.0})]
    factory = lambda seed: random_scene(mats, 64, 64, seed)

    def cfg():
        from spectral_select import Config
        return Config(sample_name="v", n_important_dimensions=15, n_bands_to_select=8,
                      perturbation_method="percentile", use_diversity_constraint=True,
                      training_epochs=30, device="cpu", random_seed=0,
                      output_dir=pathlib.Path(tempfile.mkdtemp()))

    print("=" * 84)
    print("SpectraForge: perturbation-AE vs RANDOM, on Gaussian vs measured-asymmetric spectra")
    print("peak_rec = hit the true emission peak (tight); broad_rec/precision use the wide mask.")
    print("=" * 84)
    print(f"{'selector / spectra':<26}{'peak_rec':>10}{'broad_rec':>11}{'precision':>11}{'mask%':>8}")

    libs = [("gaussian", GAUSSIAN), ("measured-asym", _measured_library())]
    for lib_label, lib in libs:
        for sel_label, selector in [("AE", make_analyzer_selector(cfg())),
                                    ("random", make_random_selector(k=8, seed=7))]:
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                res = run_validation_sweep(factory, lib, ACQ, selector, seeds=[1, 2, 3, 4],
                                           artifacts=ARTIFACTS, physics=PHYSICS, tol_nm=10)
            a = aggregate_metrics(res)
            print(f"{sel_label + ' / ' + lib_label:<26}{a['peak_recovery_mean']:>10.2f}"
                  f"{a['fluorophores_recovered_mean']:>11.2f}{a['precision_mean']:>11.2f}"
                  f"{a['mask_coverage_mean'] * 100:>7.0f}%")

    print("-" * 84)
    print("Takeaway: broad-mask metrics are saturated (random >= AE), so the earlier 'measured beats")
    print("Gaussian' headline was a mask-footprint artifact and is withdrawn. The tight peak metric")
    print("shows the AE does not target peaks (~0 vs random ~0.33), which is inconclusive about")
    print("method quality because peaks need not be the discriminative target. A discriminability-")
    print("based ground truth is required before any validation claim. No paper claim is made.")


if __name__ == "__main__":
    main()
