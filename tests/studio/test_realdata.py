"""Real-data adapter tests: a synthetic SpectraData fixture covers the logic, the lab data (Data/processed)
are used only if present (skipped otherwise)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spectraforge.studio import analysis
from spectraforge.studio import calibration as cal
from spectraforge.studio import realdata as rd

ROOT = Path(__file__).resolve().parents[2]
PEPSIN = ROOT / "Data" / "processed" / "Collagen Pepsin"


def _fake_spectra_data(h=12, w=16, seed=0):
    """Two excitations with different band lists (the Nuance protocol), a mask and metadata."""
    rng = np.random.default_rng(seed)
    exs = {}
    for ex, start, n in ((340.0, 420.0, 8), (400.0, 430.0, 6)):
        em = list(start + 10.0 * np.arange(n))
        cube = rng.uniform(1, 2, (h, w, n)).astype(np.float32)
        cube[:, :4, :] += 10.0                                          # left strip is brighter
        exs[ex] = SimpleNamespace(excitation_nm=ex, cube=cube, emission_wavelengths=em,
                                  exposure_time=100.0 + ex, laser_power=1e-3)
    mask = np.ones((h, w), np.uint8)
    mask[:, -2:] = 0
    return SimpleNamespace(excitations=exs, mask=mask, sample_name="fake", metadata={"k": 1},
                           spatial_shape=(h, w)), exs


def test_from_spectra_data_adapts_and_keeps_metadata():
    sd, exs = _fake_spectra_data()
    c = rd.from_spectra_data(sd, {"left": np.arange(16)[None, :].repeat(12, 0) < 4})
    assert c.excitations_nm == [340.0, 400.0]
    assert c.n_bands == {340.0: 8, 400.0: 6}
    assert c.height == 12 and c.width == 16
    assert c.exposure_ms[340.0] == 440.0 and c.power[400.0] == 1e-3
    assert c.mask.dtype == bool and c.mask[:, -2:].sum() == 0
    assert c.roi_names() == ["left"]
    # per-excitation band lists survive
    assert c.emission_nm[400.0][0] == 430.0
    back = c.as_spectra_data()
    assert sorted(back.excitations) == [340.0, 400.0]


def test_roi_statistics_apply_mask_and_nan_handling():
    sd, exs = _fake_spectra_data()
    c = rd.from_spectra_data(sd)
    c.cubes[340.0][0, 0, 0] = np.nan
    full = np.ones((12, 16), bool)
    sp = c.roi_mean_spectra(full)
    # mask excludes the last two columns, NaN pixel ignored
    cube = exs[340.0].cube.astype(float).copy()
    cube[0, 0, 0] = np.nan
    ref = np.nanmean(cube[:, :-2][...], axis=(0, 1))
    assert np.allclose(sp[340.0], ref)
    with pytest.raises(ValueError):
        c.roi_mean_spectra(np.zeros((12, 16), bool))
    with pytest.raises(KeyError):
        c.roi_pixels("nope")


def test_load_real_cube_from_folder_with_roi_json(tmp_path):
    sd, exs = _fake_spectra_data()
    pk = {"data": {ex: {"cube": e.cube, "wavelengths": e.emission_wavelengths} for ex, e in exs.items()},
          "excitation_wavelengths": list(exs), "mask": sd.mask,
          "exposure_times": {ex: e.exposure_time for ex, e in exs.items()},
          "laser_powers": {ex: e.laser_power for ex, e in exs.items()}}
    with open(tmp_path / "spectra_masked.pkl", "wb") as f:
        pickle.dump(pk, f)
    (tmp_path / "roi_regions.json").write_text(json.dumps({"regions": [
        {"class_id": 2, "class_name": "A", "rect": {"row_min": 0, "row_max": 6, "col_min": 0, "col_max": 4}},
        {"class_id": 2, "class_name": "A", "rect": {"row_min": 6, "row_max": 12, "col_min": 0, "col_max": 4}},
        {"class_id": 3, "class_name": "B", "rect": {"row_min": 0, "row_max": 12, "col_min": 8, "col_max": 14}}]}))
    c = rd.load_real_cube(tmp_path)
    assert c.name == tmp_path.name
    assert c.roi_names() == ["2", "3"]
    assert c.roi_pixels("2").sum() == 12 * 4          # two rectangles of the same class are merged
    # labels path
    lab = np.zeros((12, 16), int)
    lab[2:4, 2:4] = 7
    c2 = rd.load_real_cube(tmp_path / "spectra_masked.pkl", labels=lab)
    assert c2.roi_names() == ["7"]
    with pytest.raises(ValueError):
        rd.load_real_cube(tmp_path, labels=np.zeros((3, 3), int))


def test_real_roi_eem_has_analysis_eem_structure_and_nan_outside_span():
    sd, exs = _fake_spectra_data()
    c = rd.from_spectra_data(sd, {"left": np.arange(16)[None, :].repeat(12, 0) < 4})
    e = rd.real_roi_eem(c, "left")
    assert isinstance(e, analysis.EemResult)
    assert e.excitations_nm == [340.0, 400.0]
    assert e.grid_emission_nm[0] == 420.0 and e.grid_emission_nm[-1] == 490.0
    assert e.grid_value.shape == (2, e.grid_emission_nm.size)
    # 400 nm excitation: bands 430..480 -> NaN outside the span
    assert np.isnan(e.grid_value[1][e.grid_emission_nm < 430]).all()
    assert np.isnan(e.grid_value[1][e.grid_emission_nm > 480]).all()
    assert np.isfinite(e.grid_value[1][(e.grid_emission_nm >= 430) & (e.grid_emission_nm <= 480)]).all()
    assert e.n_pixels == 12 * 4 - 0
    # the left strip is brighter by ~10 over the background
    assert np.nanmean(e.value[340.0]) > 10
    d = e.to_dict()                                    # JSON-ready like the analysis API
    assert "grid_value" in d


def test_real_and_simulated_eem_are_directly_comparable():
    """A noise-free simulated capture and the same cubes presented as a RealCube give identical EEMs."""
    from spectraforge.studio import capture as cap
    from spectraforge.studio.schema import FluorophoreSpec

    spec = FluorophoreSpec(name="x", ex_peak_nm=350, ex_fwhm_nm=60, em_peak_nm=470, em_fwhm_nm=70)
    res = cap.capture(cal.thin_layer_experiment(spec, excitations=(340.0, 400.0), sensor_px=(24, 18)))
    roi = {"shape": "disk", "x": 12, "y": 9, "radius_px": 5}
    sim = analysis.eem(res, roi, {"mode": "adu"})
    sd = SimpleNamespace(excitations={ex: SimpleNamespace(cube=res.adu[ex], emission_wavelengths=list(res.emission_nm[ex]),
                                                          exposure_time=None, laser_power=None)
                                      for ex in res.excitations_nm}, mask=None, sample_name="sim", metadata={})
    cube = rd.from_spectra_data(sd, {"roi": analysis.roi_mask(analysis.coerce_roi(roi), res.height, res.width)})
    real = rd.real_roi_eem(cube, "roi")
    assert np.array_equal(real.grid_emission_nm, sim.grid_emission_nm)
    both = np.isfinite(real.grid_value) & np.isfinite(sim.grid_value)
    assert np.array_equal(np.isnan(real.grid_value), np.isnan(sim.grid_value))
    assert np.allclose(real.grid_value[both], sim.grid_value[both], rtol=1e-6)
    # explicit shared grid
    r2 = rd.real_roi_eem(cube, "roi", grid=sim.grid_emission_nm[::2])
    assert r2.grid_value.shape[1] == sim.grid_emission_nm[::2].size


def test_subset_and_unknown_excitation():
    sd, _ = _fake_spectra_data()
    c = rd.from_spectra_data(sd)
    s = c.subset([400.0])
    assert s.excitations_nm == [400.0] and list(s.cubes) == [400.0]
    with pytest.raises(KeyError):
        c.subset([500.0])


@pytest.mark.skipif(not PEPSIN.exists(), reason="Data/processed not available")
def test_load_real_lab_data_pepsin():
    c = rd.load_real_cube(PEPSIN)
    assert c.excitations_nm == [310.0, 325.0, 340.0, 365.0, 385.0, 400.0]
    assert (c.height, c.width) == (256, 348)
    assert c.roi_names() == ["2", "3", "4"]
    e = rd.real_roi_eem(c, "2")
    assert e.n_pixels > 1000
    assert np.isfinite(e.grid_value[0]).sum() > 100
