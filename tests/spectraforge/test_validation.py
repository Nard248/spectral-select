"""Increment B — validation harness: score a band selection against ground truth."""
import numpy as np

from spectraforge.groundtruth import GroundTruth
from spectraforge.validation import validate_selection


def _gt_single_informative():
    grid = np.array([400.0, 450.0, 500.0, 550.0])
    cube = np.zeros((2, 2, 4))
    cube[..., 2] = 5.0  # band 2 (500 nm) is the only informative band at ex 488
    return GroundTruth(
        concentration_maps={"A": np.ones((2, 2))},
        clean_cubes={488.0: cube},
        emission_grid=grid,
        excitations=[488.0],
        per_fluorophore_spectra={"A": {488.0: np.array([0.0, 0.0, 5.0, 0.0])}},
    )


def test_validate_precision_recall():
    gt = _gt_single_informative()
    m = validate_selection(gt, [(488.0, 500.0), (488.0, 400.0)], tol_nm=5)
    assert m["n_selected"] == 2
    assert m["hits"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0       # the single informative band was selected


def test_validate_per_fluorophore_recovery():
    gt = _gt_single_informative()
    m = validate_selection(gt, [(488.0, 500.0)], tol_nm=5)
    assert m["per_fluorophore"]["A"] is True
    assert m["fluorophores_recovered"] == 1.0
    m2 = validate_selection(gt, [(488.0, 400.0)], tol_nm=5)
    assert m2["per_fluorophore"]["A"] is False
    assert m2["fluorophores_recovered"] == 0.0


def test_validate_peak_recovery_and_mask_coverage():
    # peak_recovery is TIGHT: it requires hitting the fluorophore's true emission peak (500 nm),
    # not merely landing anywhere on the broad informative mask. mask_coverage exposes saturation.
    gt = _gt_single_informative()
    m = validate_selection(gt, [(488.0, 500.0)], tol_nm=5)
    assert m["peak_recovery"] == 1.0
    assert m["peak_hits"]["A"] is True
    assert m["mask_coverage"] == 0.25       # only 1 of 4 bands is informative
    m2 = validate_selection(gt, [(488.0, 520.0)], tol_nm=5)   # 20 nm off the 500 nm peak
    assert m2["peak_recovery"] == 0.0
    assert m2["peak_hits"]["A"] is False


def test_render_populates_per_fluorophore_spectra():
    from spectraforge import Fluorophore, Material, Scene, AcquisitionConfig, render
    lib = {"A": Fluorophore("A", 480, 40, 520, 30, quantum_yield=0.5, extinction=1.0)}
    acq = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=5)
    s = Scene(5, 5)
    s.paint_rect(Material("m", {"A": 1.0}), 0, 5, 0, 5)
    _, gt = render(s, lib, acq)
    assert "A" in gt.per_fluorophore_spectra
    per = gt.informative_bands_per_fluorophore()
    grid = acq.emission_grid()
    band520 = int(np.argmin(np.abs(grid - 520)))
    assert per["A"][480.0][band520]


def test_end_to_end_validation_contract():
    """Integration contract: render -> Analyzer.fit -> validate_selection yields a well-formed
    metrics dict over the real selection. (Whether the method *recovers* planted bands is a
    research question studied via reports/spectraforge_validation_report.py, not a CI assertion —
    recovery is scene/seed-dependent; see the report for the finding.)"""
    import pathlib
    import tempfile
    from spectraforge import Fluorophore, Material, Scene, AcquisitionConfig, ArtifactConfig, render
    from spectral_select import Analyzer, Config
    lib = {
        "G": Fluorophore("G", 488, 40, 520, 30, quantum_yield=0.6, extinction=1.0),
        "R": Fluorophore("R", 560, 40, 610, 35, quantum_yield=0.5, extinction=0.9),
    }
    acq = AcquisitionConfig(excitations=[488.0, 560.0], em_min=400, em_max=700, em_step=5)
    s = Scene(64, 64)
    s.paint_rect(Material("g", {"G": 1.0}), 0, 64, 0, 32)
    s.paint_rect(Material("r", {"R": 1.0}), 0, 64, 32, 64)
    spectra, gt = render(s, lib, acq, artifacts=ArtifactConfig(photon_scale=500, read_sigma=0.01), seed=1)
    cfg = Config(sample_name="v", n_important_dimensions=6, n_bands_to_select=8,
                 perturbation_method="standard_deviation", use_diversity_constraint=False,
                 training_epochs=3, device="cpu", output_dir=pathlib.Path(tempfile.mkdtemp()))
    a = Analyzer(cfg)
    a.fit(spectra)
    m = validate_selection(gt, a.get_wavelengths(), tol_nm=10)
    assert m["n_selected"] == 8
    for key in ("precision", "recall", "f1", "fluorophores_recovered"):
        assert 0.0 <= m[key] <= 1.0
    assert set(m["per_fluorophore"]) == {"G", "R"}      # both planted fluorophores are scored
    assert m["hits"] <= m["n_selected"]
