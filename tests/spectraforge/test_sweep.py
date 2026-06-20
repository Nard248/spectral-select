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


def test_make_labeled_scene_is_balanced_and_labels_match_dominant():
    import numpy as np
    from spectraforge.scenegen import make_labeled_scene
    mats = [Material("a", {"A": 1.0}), Material("b", {"B": 1.0}), Material("c", {"C": 1.0})]
    scene, labels = make_labeled_scene(mats, 48, 48, seed=1)
    assert labels.shape == (48, 48)
    counts = np.bincount(labels.ravel(), minlength=3)
    # balanced: every class present and none dominates wildly (each within 0.5x..2x of even split)
    even = labels.size / 3
    assert counts.min() > 0.5 * even and counts.max() < 2.0 * even
    # the label is the argmax material by concentration at that pixel
    conc = scene.resolve()
    stack = np.stack([conc["A"], conc["B"], conc["C"]])      # (3, H, W)
    assert np.array_equal(stack.argmax(0), labels)


def test_make_random_selector_is_a_chance_baseline():
    from spectraforge import Fluorophore, Scene, render
    from spectraforge.sweep import make_random_selector
    lib = {"A": Fluorophore("A", 480, 40, 520, 40)}
    acq = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=10)
    s = Scene(8, 8)
    s.paint_rect(Material("m", {"A": 1.0}), 0, 8, 0, 8)
    spectra, _ = render(s, lib, acq)
    sel = make_random_selector(k=6, seed=1)
    bands = sel(spectra)
    assert len(bands) == 6
    grid = list(acq.emission_grid())
    for ex, em in bands:
        assert ex == 480.0 and em in grid          # valid (excitation, emission) pairs on the grid
    # deterministic by seed across the two calls a sweep would make is NOT required, but a fresh
    # selector with the same seed reproduces its own first draw:
    assert make_random_selector(k=6, seed=1)(spectra) == bands


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
