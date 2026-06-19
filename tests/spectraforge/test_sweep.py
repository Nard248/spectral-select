"""Increment E — rich-variance scene generation + batch sweep with auto-validation."""
import numpy as np

from spectraforge import AcquisitionConfig, Fluorophore, Material


def test_random_field_deterministic_varied_and_bounded():
    from spectraforge.scenegen import random_field
    a = random_field(32, 32, seed=5)
    b = random_field(32, 32, seed=5)
    assert np.array_equal(a, b)                 # deterministic by seed
    assert a.shape == (32, 32)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert a.std() > 0.0                        # carries real spatial variance (not flat)
    assert not np.array_equal(a, random_field(32, 32, seed=6))


def test_random_scene_resolves_to_fluorophores_with_variance():
    from spectraforge.scenegen import random_scene
    mats = [Material("m1", {"A": 1.0}), Material("m2", {"B": 0.5})]
    scene = random_scene(mats, 16, 16, seed=1)
    conc = scene.resolve()
    assert "A" in conc and "B" in conc
    assert conc["A"].std() > 0.0                # spatially varying -> non-degenerate spectra


def test_run_validation_sweep_and_aggregate():
    from spectraforge.scenegen import random_scene
    from spectraforge.sweep import aggregate_metrics, run_validation_sweep
    lib = {"A": Fluorophore("A", 480, 40, 520, 40, quantum_yield=0.6, extinction=1.0)}
    acq = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=10)
    mats = [Material("m", {"A": 1.0})]

    def factory(seed):
        return random_scene(mats, 16, 16, seed)

    def selector(spectra):                      # injected: always pick A's true peak
        return [(480.0, 520.0)]

    results = run_validation_sweep(factory, lib, acq, selector, seeds=[1, 2, 3], tol_nm=8)
    assert len(results) == 3
    assert [r["seed"] for r in results] == [1, 2, 3]
    assert all(r["fluorophores_recovered"] == 1.0 for r in results)   # peak picked -> recovered

    agg = aggregate_metrics(results)
    assert agg["n_runs"] == 3
    assert agg["fluorophores_recovered_mean"] == 1.0
    assert agg["fluorophores_recovered_std"] == 0.0
