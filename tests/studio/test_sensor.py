"""Camera: bands, radiometry, blur, detector."""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge.studio import presets
from spectraforge.studio import sensor as sm
from spectraforge.studio.contracts import SPECTRAL_GRID_NM as GRID
from spectraforge.studio.schema import (
    CameraSpec,
    LensSpec,
    PhysicsToggles,
    SensorSpec,
    SpectralFilterSpec,
)

H_PLANCK, C_LIGHT = 6.62607015e-34, 2.99792458e8


def flat_camera(**lens_kw):
    lens = LensSpec(focal_length_mm=35.0, f_number=4.0, focus_distance_mm=300.0, transmission=0.9, **lens_kw)
    return CameraSpec(id="c", name="c", lens=lens,
                      sensor=SensorSpec(qe_curve="flat", pixel_pitch_um=6.45, binning=4),
                      filter=SpectralFilterSpec(peak_transmission_curve="flat", emission_longpass_nm=None,
                                                band_start_policy="fixed", band_start_nm=550.0, n_bands=1,
                                                fwhm_nm=20.0))


def test_band_centres_policies():
    f = SpectralFilterSpec()
    assert sm.band_centres(f, 430.0)[0] == pytest.approx(450.0)       # Nuance: ex + 20
    assert sm.band_centres(f, 365.0)[0] == pytest.approx(420.0)       # band_start wins
    c = sm.band_centres(f, 430.0)
    assert len(c) == 31 and np.allclose(np.diff(c), 10.0)
    fixed = SpectralFilterSpec(band_start_policy="fixed", band_start_nm=500.0)
    assert sm.band_centres(fixed, 480.0)[0] == 500.0


def test_longpass_floor_and_edge():
    lp = sm.longpass(GRID, 415.0, 5.0)
    assert lp[GRID == 300][0] == pytest.approx(1e-5, rel=1e-3)
    assert lp[GRID == 500][0] == pytest.approx(1.0, abs=1e-6)
    assert 0.4 < lp[GRID == 415][0] < 0.6
    assert sm.longpass(GRID, None, 5.0).min() == 1.0


def test_etendue_hand_calculation():
    """Electrons from a spectrally flat radiance match the closed-form chain."""
    cam = flat_camera()
    resp = sm.band_responses(cam, np.array([550.0]), PhysicsToggles())
    L0 = 1e-6                                              # mW / (mm^2 sr nm)
    spectral = np.full((1, int(resp.window.sum())), L0, dtype=np.float32)
    rad, el = sm.project_bands(spectral, resp, np.zeros(1, int))
    t_ms = 100.0
    copt = sm.CameraOptics.from_camera(cam)
    e = float(el[0, 0]) * copt.electron_scale(t_ms)
    # independent hand calculation
    f, S, N, T = 35.0, 300.0, 4.0, 0.9
    m = f / (S - f)
    k0 = np.pi * T / (4 * N ** 2 * (1 + m) ** 2)
    area = (6.45e-3 * 4) ** 2
    band_integral = 20.0 * np.sqrt(np.pi / (4 * np.log(2)))            # integral of the Gaussian pass-band
    photon_per_mW_s = 550e-9 / (H_PLANCK * C_LIGHT) * 1e-3             # photons / (mW s) at 550 nm
    expected = k0 * area * 0.1 * L0 * band_integral * 0.9 * 0.8 * photon_per_mW_s * 550.0 / 550.0
    assert e == pytest.approx(expected, rel=2e-3)          # lambda varies slightly across the band
    assert float(rad[0, 0]) == pytest.approx(L0 * band_integral, rel=1e-4)
    assert copt.magnification == pytest.approx(m)


def test_off_axis_blue_shift_sign_and_magnitude():
    cam = flat_camera()
    resp = sm.band_responses(cam, np.array([600.0]), PhysicsToggles(), max_field_angle_rad=np.radians(20.0))
    assert resp.ring_angles_rad.size > 1
    centroid = lambda r: (resp.rad[r, 0] * GRID).sum() / resp.rad[r, 0].sum()     # noqa: E731
    assert centroid(0) == pytest.approx(600.0, abs=0.1)
    assert centroid(len(resp.ring_angles_rad) - 1) < centroid(0) - 1.0            # shifts to shorter lambda
    expected = 600.0 * np.sqrt(1 - np.sin(np.radians(20.0)) ** 2 / 4.0)
    assert centroid(len(resp.ring_angles_rad) - 1) == pytest.approx(expected, abs=0.2)
    off = sm.band_responses(cam, np.array([600.0]), PhysicsToggles(filter_angle_shift=False),
                            max_field_angle_rad=np.radians(20.0))
    assert off.ring_angles_rad.size == 1
    idx = resp.ring_index(np.array([0.0, np.radians(20.0)]))
    assert idx[0] == 0 and idx[1] == len(resp.ring_angles_rad) - 1


def test_lctf_transmission_applied_per_band():
    cam = flat_camera()
    cam = cam.model_copy(update={"filter": cam.filter.model_copy(update={"peak_transmission_curve": "lctf_vis"})})
    centres = np.array([430.0, 650.0])
    resp = sm.band_responses(cam, centres, PhysicsToggles())
    ratio = resp.el[0, 1].sum() / resp.el[0, 0].sum()
    # passband and QE (flat) cancel, remaining: T(650)/T(430) x photon energy (650/430)
    from spectraforge.studio import spectra
    t = spectra.filter_peak_transmission("lctf_vis", centres)
    assert ratio == pytest.approx(t[1] / t[0] * 650.0 / 430.0, rel=0.02)


def test_vignetting_cos4():
    th = np.radians([0.0, 10.0, 20.0])
    np.testing.assert_allclose(sm.vignetting(th), np.cos(th) ** 4)
    assert (sm.vignetting(th, False) == 1).all()


def test_defocus_sigma_zero_at_focus_grows_away_and_chromatic():
    cam = flat_camera(axial_chromatic_mm_per_100nm=0.0)
    tg = PhysicsToggles(diffraction=False)
    c = np.array([550.0])
    d = np.array([300.0, 320.0, 360.0, np.inf])
    s = sm.blur_sigma_px(d, cam, c, tg)[:, 0]
    assert s[0] < 1e-6 and s[1] > 0 and s[2] > s[1] and s[3] < 1e-6
    # hand calc: c = A |v1 - v2| / v2 ; sigma = c/4/pitch
    f = 35.0
    v1, v2 = f * 300 / 265, f * 320 / 285
    coc = (f / 4.0) * abs(v1 - v2) / v2
    assert s[1] == pytest.approx(coc / 4 / (6.45e-3 * 4), rel=1e-6)
    off = sm.blur_sigma_px(d, cam, c, PhysicsToggles(diffraction=False, defocus=False))[:, 0]
    assert np.allclose(off, 0.0)
    chrom = flat_camera(axial_chromatic_mm_per_100nm=0.5)
    sc = sm.blur_sigma_px(np.array([300.0]), chrom, np.array([450.0, 550.0, 700.0]), tg)[0]
    assert sc[1] < sc[0] and sc[1] < sc[2]
    assert sm.blur_sigma_px(np.array([300.0]), chrom, np.array([450.0]),
                            PhysicsToggles(diffraction=False, chromatic_aberration=False))[0, 0] < 1e-6


def test_diffraction_sigma_value():
    cam = flat_camera()
    s = sm.blur_sigma_px(np.array([300.0]), cam, np.array([550.0]), PhysicsToggles(defocus=False,
                                                                                    chromatic_aberration=False))
    m = 35.0 / 265.0
    assert s[0, 0] == pytest.approx(0.42 * 550e-6 * 4.0 * (1 + m) / (6.45e-3 * 4), rel=1e-6)


def test_blur_cube_conserves_flux_and_smooths():
    rng = np.random.default_rng(0)
    cube = np.zeros((40, 40, 2), dtype=np.float32)
    cube[20, 20, :] = 1.0
    sig = np.zeros((40, 40, 2), dtype=np.float32)
    sig[:, :, 0] = 2.0
    out = sm.blur_cube(cube, sig)
    assert out[:, :, 0].sum() == pytest.approx(1.0, rel=1e-3)
    assert out[20, 20, 0] < 0.1 and out[20, 20, 1] == pytest.approx(1.0)       # band 1 sharp
    # spatially varying: left half sharp, right half blurred
    cube2 = rng.random((32, 32, 1)).astype(np.float32)
    sig2 = np.zeros((32, 32, 1), dtype=np.float32)
    sig2[:, 16:] = 3.0
    o2 = sm.blur_cube(cube2, sig2)
    np.testing.assert_allclose(o2[:, :8], cube2[:, :8], atol=1e-6)
    assert o2[:, 24:].std() < 0.5 * cube2[:, 24:].std()
    assert sm.blur_cube(cube2, np.zeros_like(sig2)) is cube2


def test_shot_and_read_noise_statistics():
    sensor = SensorSpec(read_noise_e=8.0, dark_current_e_per_s=2.0, gain_e_per_adu=1.5, offset_adu=100.0,
                        full_well_e=1e6, bit_depth=16)
    tg = PhysicsToggles(fixed_pattern_noise=False, quantization=False)
    S, t_ms = 4000.0, 500.0
    sig = np.full((128, 128, 2), S)
    rng = np.random.default_rng(1)
    adu, sat = sm.detect(sig, t_ms, sensor, tg, None, rng)
    dark = 2.0 * 0.5
    assert sat == 0.0
    assert adu.mean() == pytest.approx((S + dark) / 1.5 + 100.0, rel=2e-3)
    assert adu.var() == pytest.approx((S + dark + 64.0) / 1.5 ** 2, rel=0.03)
    # shot noise off, read noise off -> deterministic
    tg2 = PhysicsToggles(fixed_pattern_noise=False, quantization=False, shot_noise=False, read_noise=False)
    a2, _ = sm.detect(sig, t_ms, sensor, tg2, None, rng)
    assert a2.std() == 0 and a2.mean() == pytest.approx((S + dark) / 1.5 + 100.0, rel=1e-6)
    # read-noise only variance
    tg3 = PhysicsToggles(fixed_pattern_noise=False, quantization=False, shot_noise=False)
    a3, _ = sm.detect(sig, t_ms, sensor, tg3, None, rng)
    assert a3.var() == pytest.approx(64.0 / 1.5 ** 2, rel=0.05)


def test_quantisation_and_saturation():
    sensor = SensorSpec(full_well_e=1000.0, gain_e_per_adu=1.0, offset_adu=10.0, bit_depth=12, read_noise_e=0.0,
                        dark_current_e_per_s=0.0, prnu_pct=0.0, dsnu_e=0.0, hot_pixel_fraction=0.0)
    tg = PhysicsToggles(shot_noise=False, read_noise=False)
    sig = np.full((8, 8, 1), 5000.0)
    adu, sat = sm.detect(sig, 100.0, sensor, tg, None, np.random.default_rng(0))
    assert sat == 1.0 and adu.max() == pytest.approx(1010.0)                 # full-well clip, below ADC max
    sig2 = np.full((8, 8, 1), 123.4)
    adu2, _ = sm.detect(sig2, 100.0, sensor, tg, None, np.random.default_rng(0))
    assert (adu2 == np.round(adu2)).all() and adu2[0, 0, 0] == 133.0
    # saturation toggle off lets the signal run past the full well
    adu3, _ = sm.detect(sig, 100.0, sensor, tg.model_copy(update={"saturation": False}), None,
                        np.random.default_rng(0))
    assert adu3.max() > 4000
    # ADC clip below full well
    s2 = sensor.model_copy(update={"full_well_e": 1e6})
    adu4, sat4 = sm.detect(np.full((4, 4, 1), 1e5), 100.0, s2, tg, None, np.random.default_rng(0))
    assert adu4.max() == 4095.0 and sat4 == 1.0


def test_fixed_pattern_deterministic_and_statistics():
    sensor = SensorSpec(prnu_pct=2.0, dsnu_e=3.0, hot_pixel_fraction=0.01)
    a = sm.make_fixed_pattern(sensor, 64, 64, seed=5)
    b = sm.make_fixed_pattern(sensor, 64, 64, seed=5)
    c = sm.make_fixed_pattern(sensor, 64, 64, seed=6)
    np.testing.assert_array_equal(a.prnu, b.prnu)
    assert not np.array_equal(a.prnu, c.prnu)
    assert a.prnu.std() == pytest.approx(0.02, rel=0.15) and a.dsnu_e.std() == pytest.approx(3.0, rel=0.15)
    assert (a.hot_e_per_s > 0).sum() == round(0.01 * 64 * 64)
    # flat-field response shows PRNU when enabled and is flat when disabled
    tg_on = PhysicsToggles(shot_noise=False, read_noise=False, dark_current=False, quantization=False)
    sig = np.full((64, 64, 1), 10000.0)
    on, _ = sm.detect(sig, 10.0, sensor.model_copy(update={"dsnu_e": 0.0, "hot_pixel_fraction": 0.0}), tg_on,
                      sm.make_fixed_pattern(sensor.model_copy(update={"dsnu_e": 0.0, "hot_pixel_fraction": 0.0}),
                                            64, 64, 1), np.random.default_rng(0))
    assert on.std() / on.mean() > 0.01
    off, _ = sm.detect(sig, 10.0, sensor, tg_on.model_copy(update={"fixed_pattern_noise": False}), a,
                       np.random.default_rng(0))
    assert off.std() < 1e-2


def test_hot_pixels_appear_in_long_exposure_dark_frame():
    sensor = SensorSpec(hot_pixel_fraction=0.01, dark_current_e_per_s=0.0, dsnu_e=0.0, read_noise_e=0.0,
                        offset_adu=0.0, gain_e_per_adu=1.0)
    fp = sm.make_fixed_pattern(sensor, 64, 64, 3)
    tg = PhysicsToggles(shot_noise=False, read_noise=False, quantization=False)
    adu, _ = sm.detect(np.zeros((64, 64, 1)), 1000.0, sensor, tg, fp, np.random.default_rng(0))
    assert (adu > 50).sum() == (fp.hot_e_per_s > 0).sum()


def test_snr_estimate():
    sensor = SensorSpec(read_noise_e=10.0, dark_current_e_per_s=0.0)
    tg = PhysicsToggles()
    s = np.array([100.0, 10000.0])
    snr = sm.snr_estimate(s, 100.0, sensor, tg)
    np.testing.assert_allclose(snr, s / np.sqrt(s + 100.0))
