"""Calibration tests: synthetic ground truth from the engine itself (fast model and full noisy captures)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectraforge.studio import analysis
from spectraforge.studio import calibration as cal
from spectraforge.studio import capture as cap
from spectraforge.studio import optics as optics_mod
from spectraforge.studio import realdata as rd
from spectraforge.studio.schema import FluorophoreSpec

ROOT = Path(__file__).resolve().parents[2]
PEPSIN = ROOT / "Data" / "processed" / "Collagen Pepsin"

EXS = (310.0, 325.0, 340.0, 365.0, 385.0, 400.0)
TRUTH = FluorophoreSpec(name="truth", ex_peak_nm=350.0, ex_fwhm_nm=70.0, em_peak_nm=470.0, em_fwhm_nm=80.0,
                        em_skew=0.3, vibronic=0.0)
START = TRUTH.model_copy(update=dict(ex_peak_nm=330.0, ex_fwhm_nm=50.0, em_peak_nm=510.0, em_fwhm_nm=60.0,
                                     em_skew=0.0))


def _bands(exs=EXS, step=10.0, n=31):
    return {ex: max(420.0, ex + 20.0) + step * np.arange(n) for ex in exs}


def _model_eem(specs, amps=None, exs=EXS, gains=None, noise=0.0, seed=0, bands=None):
    inst = cal.TwinInstrument.lab_replica()
    bands = bands or _bands(exs)
    m = cal.ThinLayerModel(inst, bands)
    e = m.as_eem(specs, amps, gains=gains)
    if noise:
        rng = np.random.default_rng(seed)
        val = {ex: v * (1 + noise * rng.standard_normal(v.size)) for ex, v in e.value.items()}
        e = cal.eem_from_arrays(exs, bands, val)
    return e


# --------------------------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------------------------
def test_excitation_overlap_matches_gaussian_for_a_narrow_line():
    inst = cal.TwinInstrument.lab_replica()
    m = cal.ThinLayerModel(inst, _bands((340.0,)))
    # 10 nm Gaussian line on a Gaussian absorption (sigma_a): overlap = exp(-0.5 d^2 / (sa^2 + sl^2)) * sa/sqrt(sa^2+sl^2)
    sa = TRUTH.ex_fwhm_nm / 2.3548200450309493
    sl = inst.source.line_fwhm_nm / 2.3548200450309493
    d = 340.0 - TRUTH.ex_peak_nm
    want = np.exp(-0.5 * d ** 2 / (sa ** 2 + sl ** 2)) * sa / np.sqrt(sa ** 2 + sl ** 2)
    assert m.excitation_overlap(TRUTH, 340.0) == pytest.approx(want, rel=2e-3)


def test_emission_projection_conserves_unit_integral_for_a_flat_response():
    """With unit response the band projection of the unit-integral emission density is the in-grid fraction."""
    inst = cal.TwinInstrument.lab_replica()
    m = cal.ThinLayerModel(inst, _bands((340.0,)))
    m.R[340.0] = np.ones_like(m.R[340.0])
    assert m.emission_projection(TRUTH, 340.0)[0] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("spec", [
    TRUTH,
    FluorophoreSpec(name="v", ex_peak_nm=380.0, ex_fwhm_nm=50.0, em_peak_nm=520.0, em_fwhm_nm=60.0,
                    em_skew=0.5, vibronic=0.2)])
def test_fast_model_matches_full_capture_within_3_percent(spec):
    v = cal.verify_fast_model(spec, excitations=(325.0, 365.0, 400.0))
    assert v["max_rel_dev"] < 0.03                    # the brief's tolerance
    assert v["max_rel_dev"] < 1e-3                    # in practice: same formalism, only float32 + grid


# --------------------------------------------------------------------------------------------
# fit: recovery
# --------------------------------------------------------------------------------------------
def test_global_gain_recovers_all_parameters_noise_free():
    e = _model_eem([TRUTH], gains={ex: 1e-3 for ex in EXS})
    fr = cal.fit_fluorophores(e, gain="global", prior=START, n_starts=6, seed=1)
    s = fr.specs[0]
    assert s.em_peak_nm == pytest.approx(470.0, abs=0.05)
    assert s.em_fwhm_nm == pytest.approx(80.0, abs=0.1)
    assert s.em_skew == pytest.approx(0.3, abs=0.01)
    assert s.ex_peak_nm == pytest.approx(350.0, abs=0.1)
    assert s.ex_fwhm_nm == pytest.approx(70.0, abs=0.2)
    assert fr.diagnostics["r2"] > 0.99999
    assert fr.diagnostics["sam_mean_deg"] < 0.05
    assert fr.relative_brightness == [1.0]
    # the global gain is recovered (any common factor)
    assert np.allclose(list(fr.gains.values()), 1e-3, rtol=1e-3)


def test_per_excitation_gain_recovers_emission_and_the_gains():
    g = {ex: 10.0 ** (i - 2) for i, ex in enumerate(EXS)}          # units differ by 5 decades between lines
    e = _model_eem([TRUTH], gains=g)
    fr = cal.fit_fluorophores(e, gain="per_excitation", prior=START, n_starts=6, seed=1)
    s = fr.specs[0]
    assert s.em_peak_nm == pytest.approx(470.0, abs=0.05)
    assert s.em_fwhm_nm == pytest.approx(80.0, abs=0.1)
    assert "c0.ex_peak_nm" not in fr.param_names                   # not identifiable -> not fitted
    assert s.ex_peak_nm == START.ex_peak_nm                        # held at the prior
    # gains are relative to the (pinned, wrong) excitation prior: ratios between lines follow the true overlaps
    inst = cal.TwinInstrument.lab_replica()
    m = cal.ThinLayerModel(inst, _bands())
    ratio = {ex: fr.gains[ex] / g[ex] * m.excitation_overlap(START, ex) / m.excitation_overlap(TRUTH, ex) for ex in EXS}
    assert np.allclose(list(ratio.values()), ratio[340.0], rtol=1e-3)


def test_confidence_intervals_cover_the_truth_for_noisy_data():
    covered = []
    for seed in range(12):
        e = _model_eem([TRUTH], gains={ex: 1.0 for ex in EXS}, noise=0.02, seed=seed)
        fr = cal.fit_fluorophores(e, gain="global", prior=START, n_starts=3, seed=seed, loss="normalized")
        i = fr.param_names.index("c0.em_peak_nm")
        covered.append(abs(fr.param_values[i] - 470.0) <= fr.ci95[i])
        assert fr.ci95[i] > 0 and np.isfinite(fr.ci95[i])
    assert np.mean(covered) >= 0.75            # nominal 95 %; 12 draws


def test_two_component_fit_describes_a_mixture_and_bic_prefers_it():
    a = TRUTH
    b = FluorophoreSpec(name="b", ex_peak_nm=400.0, ex_fwhm_nm=50.0, em_peak_nm=600.0, em_fwhm_nm=60.0)
    e = _model_eem([a, b], amps=[1.0, 0.6], gains={ex: 1.0 for ex in EXS})
    f1 = cal.fit_fluorophores(e, gain="global", n_components=1, n_starts=6, seed=0)
    f2 = cal.fit_fluorophores(e, gain="global", n_components=2, n_starts=10, seed=0,
                              prior=[TRUTH, b.model_copy(update=dict(em_peak_nm=580.0, ex_peak_nm=380.0))])
    assert f2.diagnostics["bic"] < f1.diagnostics["bic"]
    assert f2.diagnostics["sam_mean_deg"] < 1.0
    assert sum(f2.relative_brightness) == pytest.approx(1.0)


def test_determinism_same_seed_same_result():
    e = _model_eem([TRUTH], gains={ex: 1.0 for ex in EXS}, noise=0.03, seed=3)
    a = cal.fit_fluorophores(e, gain="per_excitation", n_starts=5, seed=9)
    b = cal.fit_fluorophores(e, gain="per_excitation", n_starts=5, seed=9)
    assert a.param_values == b.param_values
    assert a.ci95 == b.ci95


def test_response_correction_terms_are_fitted_and_applied_in_prediction():
    """Smooth throughput correction: parameters exist, the gains are exposed and ``predict_eem`` applies them.

    (A smooth tilt is partly degenerate with the emission peak / skew -- a wrong throughput curve biases the peak
    instead of being "absorbed"; that systematic is quantified in docs/studio/SIM2REAL.md.)"""
    inst = cal.TwinInstrument.lab_replica()
    bands = _bands()
    m = cal.ThinLayerModel(inst, bands)
    tilt = {ex: np.exp(1.2 * (bands[ex] - 550.0) / 150.0 - 0.4 * ((bands[ex] - 550.0) / 150.0) ** 2) for ex in EXS}
    pred = m.predict([TRUTH])
    e = cal.eem_from_arrays(EXS, bands, {ex: pred[ex] * tilt[ex] for ex in EXS})
    f2 = cal.fit_fluorophores(e, gain="per_excitation", prior=START, n_starts=6, response_degree=2, response_sigma=5.0)
    assert [n for n in f2.param_names if n.startswith("instrument[")] == ["instrument[0]", "instrument[1]"]
    assert len(f2.instrument_gain) > 20 and all(v > 0 for v in f2.instrument_gain.values())
    assert f2.diagnostics["sam_mean_deg"] < 0.6
    out = cal.predict_eem(f2, bands)
    for ex in EXS:
        y = e.value[ex]
        c = np.dot(y, out[ex]) / (np.linalg.norm(y) * np.linalg.norm(out[ex]))
        assert np.degrees(np.arccos(min(c, 1.0))) < 1.0
    f0 = cal.fit_fluorophores(e, gain="per_excitation", prior=START, n_starts=6, response_degree=0)
    assert not f0.instrument_gain and not any(n.startswith("instrument[") for n in f0.param_names)


def test_ripple_basis_is_orthogonal_to_smooth_polynomials():
    wl = np.arange(420.0, 720.0, 10.0)
    B = cal._ripple_basis(wl)
    t = (wl - wl.mean()) / np.ptp(wl)
    for deg in range(4):
        assert np.max(np.abs(B.T @ t ** deg)) < 1e-9


def test_recovery_from_noisy_full_capture_exposure_power_path():
    """Twin -> twin through the whole engine with detector noise on: unbiased to a fraction of a nm."""
    exs = (310.0, 340.0, 365.0, 385.0, 400.0, 415.0)
    exp = cal.thin_layer_experiment(TRUTH, excitations=exs, concentration_uM=0.5, noise=True, sensor_px=(48, 36),
                                    exposure_ms=100.0, seed=5)
    res = cap.capture(exp)
    dark = cap.capture(exp, mode="dark")
    e = analysis.eem(res, {"shape": "disk", "x": 24, "y": 18, "radius_px": 12}, {"mode": "dark_subtracted"}, dark=dark)
    fr = cal.fit_fluorophores(e, gain="global", prior=START, n_starts=4, seed=0)
    s = fr.specs[0]
    assert s.em_peak_nm == pytest.approx(470.0, abs=0.5)
    assert s.em_fwhm_nm == pytest.approx(80.0, abs=1.5)
    assert s.ex_peak_nm == pytest.approx(350.0, abs=0.5)
    assert fr.diagnostics["r2"] > 0.995


# --------------------------------------------------------------------------------------------
# prediction / hold-out
# --------------------------------------------------------------------------------------------
def test_evaluate_prediction_on_training_data_equals_fit_diagnostics():
    e = _model_eem([TRUTH], gains={ex: 1.0 for ex in EXS}, noise=0.03, seed=2)
    fr = cal.fit_fluorophores(e, gain="per_excitation", n_starts=4, seed=0)
    ev = cal.evaluate_prediction(fr, e, refit_gains=True)
    assert ev["sam_mean_deg"] == pytest.approx(fr.diagnostics["sam_mean_deg"], rel=1e-6)
    assert ev["r2"] == pytest.approx(fr.diagnostics["r2"], rel=1e-6)
    ev2 = cal.evaluate_prediction(fr, e, refit_gains=False)
    assert ev2["sam_mean_deg"] == pytest.approx(fr.diagnostics["sam_mean_deg"], rel=1e-6)


def test_global_model_predicts_absolute_amplitude_at_held_out_excitations():
    e = _model_eem([TRUTH], gains={ex: 7.0 for ex in EXS}, noise=0.01, seed=4)
    from dataclasses import replace
    tr = replace(e, excitations_nm=[310.0, 340.0, 385.0], emission_nm={x: e.emission_nm[x] for x in (310.0, 340.0, 385.0)},
                 value={x: e.value[x] for x in (310.0, 340.0, 385.0)}, std={x: e.std[x] for x in (310.0, 340.0, 385.0)},
                 grid_value=e.grid_value[[0, 2, 4]])
    te_ex = (325.0, 365.0, 400.0)
    te = replace(e, excitations_nm=list(te_ex), emission_nm={x: e.emission_nm[x] for x in te_ex},
                 value={x: e.value[x] for x in te_ex}, std={x: e.std[x] for x in te_ex}, grid_value=e.grid_value[[1, 3, 5]])
    fr = cal.fit_fluorophores(tr, gain="global", prior=START, n_starts=6, seed=0)
    ev = cal.evaluate_prediction(fr, te, refit_gains=False)
    assert ev["r2_abs"] > 0.98
    for ratio in ev["peak_ratio_pred_over_meas"].values():
        assert ratio == pytest.approx(1.0, abs=0.05)


def test_baseline_scores_are_worse_than_a_matching_model_and_flat_is_worst():
    e = _model_eem([TRUTH], gains={ex: 1.0 for ex in EXS}, noise=0.01, seed=0)
    b = cal.baseline_scores(e, e)
    assert b["empirical_sam_deg"] < 2.0
    assert b["flat_sam_deg"] > 10.0


def test_pixel_sam_map_is_small_on_model_pixels_and_large_elsewhere():
    inst = cal.TwinInstrument.lab_replica()
    bands = _bands((340.0, 365.0))
    m = cal.ThinLayerModel(inst, bands)
    other = TRUTH.model_copy(update=dict(em_peak_nm=600.0, em_fwhm_nm=40.0))
    h, w = 8, 10
    cubes = {}
    p_t, p_o = m.predict([TRUTH]), m.predict([other])
    for ex in bands:
        cb = np.empty((h, w, bands[ex].size), np.float32)
        cb[:, :5] = p_t[ex]
        cb[:, 5:] = p_o[ex]
        cubes[ex] = cb
    from types import SimpleNamespace
    sd = SimpleNamespace(excitations={ex: SimpleNamespace(cube=cubes[ex], emission_wavelengths=list(bands[ex]),
                                                          exposure_time=None, laser_power=None) for ex in bands},
                         mask=None, sample_name="s", metadata={})
    cube = rd.from_spectra_data(sd)
    e = rd.real_roi_eem(cube, np.arange(w)[None, :].repeat(h, 0) < 5)
    fr = cal.fit_fluorophores(e, gain="per_excitation", prior=START, n_starts=6)
    sam = cal.pixel_sam_map(cube, fr)
    assert np.nanmean(sam[:, :5]) < 1.0
    assert np.nanmean(sam[:, 5:]) > 20.0


# --------------------------------------------------------------------------------------------
# summaries, scattering, library
# --------------------------------------------------------------------------------------------
def test_emission_summary_of_a_gaussian():
    s = FluorophoreSpec(name="g", ex_peak_nm=350, ex_fwhm_nm=50, em_peak_nm=500.0, em_fwhm_nm=60.0)
    d = cal.emission_summary(s)
    assert d["peak_nm"] == pytest.approx(500.0, abs=1.0)
    assert d["red_half_max_nm"] == pytest.approx(530.0, abs=1.5)          # peak + FWHM/2
    assert d["fraction_in_window"] > 0.99
    blue = s.model_copy(update={"em_peak_nm": 390.0})
    assert cal.emission_summary(blue)["fraction_in_window"] < 0.4


def test_calibrate_scattering_reports_not_identifiable_with_rayleigh_cutoff():
    from types import SimpleNamespace
    bands = _bands((340.0, 400.0))
    cube = rd.from_spectra_data(SimpleNamespace(
        excitations={ex: SimpleNamespace(cube=np.ones((4, 4, b.size), np.float32), emission_wavelengths=list(b),
                                         exposure_time=None, laser_power=None) for ex, b in bands.items()},
        mask=None, sample_name="s", metadata={}))
    sc = cal.calibrate_scattering(cube, np.ones((4, 4), bool))
    assert not sc.identifiable and sc.mus_prime_500_per_mm is None
    assert "cut off" in sc.reason
    # elastic bands present but no white reference: still not identifiable, but says why
    near = {340.0: np.array([335.0, 345.0] + list(np.arange(360.0, 440.0, 10.0)))}
    cube2 = rd.from_spectra_data(SimpleNamespace(
        excitations={340.0: SimpleNamespace(cube=np.ones((4, 4, near[340.0].size), np.float32),
                                            emission_wavelengths=list(near[340.0]), exposure_time=None,
                                            laser_power=None)}, mask=None, sample_name="s", metadata={}))
    sc2 = cal.calibrate_scattering(cube2, np.ones((4, 4), bool))
    assert not sc2.identifiable and sc2.elastic_bands == {340.0: 2}


def test_calibrated_fluorophore_json_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    cal.write_calibrated_fluorophores([(TRUTH, {"dataset": "x", "r2": 0.9})], p, note="t")
    specs = cal.calibrated_fluorophores(path=p)
    assert specs == [TRUTH]
    sp, prov = cal.calibrated_fluorophores(with_provenance=True, path=p)[0]
    assert prov["dataset"] == "x" and sp.name == "truth"
    assert cal.calibrated_fluorophores(path=tmp_path / "missing.json") == []


def test_shipped_calibrated_fluorophores_file_is_valid_if_present():
    entries = cal.calibrated_fluorophores(with_provenance=True)
    for spec, prov in entries:
        assert isinstance(spec, FluorophoreSpec)
        assert {"dataset", "roi", "method", "r2"} <= set(prov)
        assert 0.0 < prov["r2"] <= 1.0
        # usable in the engine
        optics_mod.to_legacy(spec).emission(np.arange(400.0, 700.0))


def test_input_validation():
    e = _model_eem([TRUTH], gains={ex: 1.0 for ex in EXS})
    with pytest.raises(ValueError):
        cal.fit_fluorophores(e, gain="nope")
    with pytest.raises(ValueError):
        cal.fit_fluorophores(e, n_components=4)
    with pytest.raises(ValueError):
        cal.fit_fluorophores(e, loss="nope")


# --------------------------------------------------------------------------------------------
# real lab data (skipped if Data/ is absent)
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not PEPSIN.exists(), reason="Data/processed not available")
def test_real_collagen_pepsin_calibration_beats_the_nominal_instrument():
    c = rd.load_real_cube(PEPSIN)
    e = rd.real_roi_eem(c, "2")
    nom = cal.fit_fluorophores(e, gain="per_excitation", response_degree=0, n_starts=6, seed=0)
    cor = cal.fit_fluorophores(e, gain="per_excitation", response_degree=2, n_starts=6, seed=0)
    assert cor.diagnostics["r2"] > 0.85
    assert cor.diagnostics["sam_mean_deg"] < 10.0
    assert cor.diagnostics["r2"] > nom.diagnostics["r2"]
    d = cal.emission_summary(cor.specs[0])
    assert 470 < d["centroid_window_nm"] < 530            # collagen emission centroid in the Nuance window
    # fitted spec is a valid FluorophoreSpec usable by the engine
    assert isinstance(cor.specs[0], FluorophoreSpec)
