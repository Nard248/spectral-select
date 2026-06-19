"""Validation on REAL measured spectra (FPbase), WITH a chance baseline — corrected experiment.

History / correction (2026-06-20): an earlier version of this script claimed the perturbation-AE
selection "recovers the informative emission bands far better with measured spectra than with
Gaussian approximations." Adversarial review (and the checks below) showed that conclusion was an
ARTIFACT and is withdrawn:

  * The ground-truth "informative-band" mask, as defined, covers 83-93% of the emission grid
    (inflated by broad measured red tails + the autofluorescence floor + a loose global threshold),
    so broad-mask recovery/precision are near-saturated. A uniformly-RANDOM selector matches or
    beats the AE on them, so they cannot distinguish the method from chance.
  * On a TIGHT metric (a selected band must land within tol of a fluorophore's true emission PEAK)
    the AE scores ~0 while random scores ~0.33 — the AE systematically picks edge/tail bands.

IMPORTANT CAVEAT: hitting emission peaks is not necessarily the right objective. A discriminative
selector may legitimately prefer non-peak bands (shoulders/crossovers where fluorophores separate),
so peak_recovery ~ 0 is NOT proof the method fails. The defensible reading is that THIS harness, as
configured, is INCONCLUSIVE about the method's quality: it needs a discriminability-grounded ground
truth (e.g. a supervised band-importance oracle), not emission peaks. Both a positive and a naive
negative claim are unsupported. Always read these numbers next to the random baseline and mask%.

Run:  python reports/fpbase_validation.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import tempfile

import numpy as np

from spectraforge import (
    AcquisitionConfig, ArtifactConfig, Fluorophore, Material, PhysicsConfig, from_fpbase_payload,
)
from spectraforge.scenegen import random_scene
from spectraforge.sweep import (
    aggregate_metrics, make_analyzer_selector, make_random_selector, run_validation_sweep,
)

FIX = pathlib.Path(__file__).parent / "fpbase_spectra"
DYES = [("EBFP2", 385.0, 0.56, 1.0), ("EGFP", 489.0, 0.60, 1.0), ("mCherry", 587.0, 0.22, 0.9)]


def load_measured(name, qy, eps):
    payload = json.loads((FIX / f"{name}.json").read_text())
    return from_fpbase_payload(payload, quantum_yield=qy, extinction=eps)


def gaussian_like(mf, ex_fwhm=45.0, em_fwhm=50.0):
    return Fluorophore(mf.name, mf.ex_peak_nm, ex_fwhm, mf.em_peak_nm, em_fwhm,
                       mf.quantum_yield, mf.extinction)


def main():
    measured = {name: load_measured(name, qy, eps) for name, ex, qy, eps in DYES}
    gaussian = {name: gaussian_like(mf) for name, mf in measured.items()}
    acq = AcquisitionConfig(excitations=[ex for _, ex, _, _ in DYES], em_min=400, em_max=700, em_step=5)
    mats = [Material(name.lower(), {name: 1.0}) for name in measured]

    from spectral_select import Config

    def factory(seed):
        return random_scene(mats, 64, 64, seed)

    def cfg():
        return Config(sample_name="v", n_important_dimensions=18, n_bands_to_select=12,
                      perturbation_method="percentile", use_diversity_constraint=True,
                      training_epochs=30, device="cpu", random_seed=0,
                      output_dir=pathlib.Path(tempfile.mkdtemp()))

    print("=" * 84)
    print("Validation on REAL FPbase spectra — perturbation-AE vs a RANDOM chance baseline")
    print("Dyes:", ", ".join(f"{n} (ex{int(ex)}/em{int(measured[n].em_peak_nm)})" for n, ex, _, _ in DYES))
    print("peak_recovery = hit the TRUE emission peak (tight, discriminative); broad_rec/precision")
    print("use the wide informative mask (mask% of the grid) and are near-saturated.")
    print("=" * 84)
    print(f"{'selector / library':<24}{'peak_rec':>10}{'broad_rec':>11}{'precision':>11}{'mask%':>8}")

    artifacts = ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01)
    physics = PhysicsConfig(psf_sigma_px=1.5, autofluorescence=0.02)
    for lib_label, lib in [("measured", measured), ("gaussian", gaussian)]:
        for sel_label, selector in [("AE", make_analyzer_selector(cfg())),
                                    ("random", make_random_selector(k=12, seed=7))]:
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):  # mute training logs
                res = run_validation_sweep(factory, lib, acq, selector, seeds=[1, 2, 3, 4],
                                           artifacts=artifacts, physics=physics, tol_nm=10)
            a = aggregate_metrics(res)
            print(f"{sel_label + ' / ' + lib_label:<24}{a['peak_recovery_mean']:>10.2f}"
                  f"{a['fluorophores_recovered_mean']:>11.2f}{a['precision_mean']:>11.2f}"
                  f"{a['mask_coverage_mean'] * 100:>7.0f}%")

    print("-" * 84)
    print("Reading: the broad-mask metrics are near-saturated (random matches/beats the AE), so they")
    print("prove nothing. On the tight peak metric the AE hits ~0 of the true peaks vs random ~0.33.")
    print("This does NOT prove the method fails — peaks may not be the discriminative target — but it")
    print("does show this harness cannot validate the method as configured. Verdict: INCONCLUSIVE;")
    print("a discriminability-based ground truth (not emission peaks) is needed. No paper claim made.")


if __name__ == "__main__":
    main()
