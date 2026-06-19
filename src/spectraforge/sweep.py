"""Batch sweep + auto-validation: turn the harness into an experiment engine.

Render many scenes, run a band selector on each, score against ground truth, and aggregate —
so a single call answers "how well does this selection method recover planted bands, on average,
across N random scenes?". The selector is injected (dependency injection) so the sweep logic is
testable without heavy autoencoder training; ``make_analyzer_selector`` wires up the real one.
"""
from __future__ import annotations

import numpy as np

from spectraforge.forward import render
from spectraforge.validation import validate_selection

_AGG_KEYS = ("precision", "recall", "f1", "fluorophores_recovered", "peak_recovery", "mask_coverage")


def run_validation_sweep(scene_factory, library, acquisition, selector, seeds,
                         artifacts=None, physics=None, tol_nm: float = 10.0):
    """For each seed: render ``scene_factory(seed)`` -> ``selector(spectra)`` -> validate vs GT.

    Returns a list of metrics dicts (one per seed), each augmented with ``"seed"``.
    """
    results = []
    for seed in seeds:
        scene = scene_factory(seed)
        spectra, gt = render(scene, library, acquisition,
                             artifacts=artifacts, physics=physics, seed=seed)
        metrics = validate_selection(gt, selector(spectra), tol_nm=tol_nm)
        metrics["seed"] = seed
        results.append(metrics)
    return results


def aggregate_metrics(results) -> dict:
    """Mean/std of the core metrics across a sweep's results."""
    agg = {"n_runs": len(results)}
    for key in _AGG_KEYS:
        vals = [r[key] for r in results]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        agg[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return agg


def make_random_selector(k: int = 12, seed: int = 0):
    """Return a chance-baseline selector: ``k`` uniformly-random (excitation, emission) bands.

    Essential control — if a learned selector cannot beat this on the tight ``peak_recovery``
    metric, the harness has not shown the method recovers anything. Each call draws fresh bands
    (so a sweep's per-scene draws differ); a new selector with the same seed reproduces its draws.
    """
    state = {"n": 0}

    def selector(spectra):
        excitations = list(spectra.excitation_wavelengths)
        grid = list(spectra.get_excitation(excitations[0]).emission_wavelengths)
        state["n"] += 1
        rng = np.random.default_rng((seed + 1) * 100003 + state["n"])
        return [(float(rng.choice(excitations)), float(rng.choice(grid))) for _ in range(k)]

    return selector


def make_analyzer_selector(config):
    """Return a selector ``spectra -> [WavelengthBand]`` that trains an Analyzer per scene.

    Imported lazily so the sweep module does not hard-depend on spectral_select.
    """
    from spectral_select import Analyzer

    def selector(spectra):
        analyzer = Analyzer(config)
        analyzer.fit(spectra)
        return analyzer.get_wavelengths()

    return selector
