"""Beam irradiance, line shapes, leakage, drift."""
import numpy as np
import pytest

from spectraforge.studio import source as so
from spectraforge.studio.schema import BeamSpec, ExcitationStep, LightSourceSpec


def make(kind="tunable_lamp", profile="gaussian", D=8.0, alpha=6.0, **kw):
    return LightSourceSpec(id="s", name="s", kind=kind, position_mm=(0, 100, 0), target_mm=(0, 0, 0),
                           beam=BeamSpec(profile=profile, aperture_diameter_mm=D, divergence_half_angle_deg=alpha),
                           **kw)


def plane_power(src, z, half_width, n=1201):
    xs = np.linspace(-half_width, half_width, n)
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([X.ravel(), np.full(X.size, 100.0 - z), Z.ravel()], axis=1)
    nrm = np.tile([0.0, 1.0, 0.0], (len(pts), 1))               # plane perpendicular to the axis
    E = so.irradiance_per_mW(src, pts, nrm)
    dA = (xs[1] - xs[0]) ** 2
    return float(E.sum() * dA)


@pytest.mark.parametrize("profile", ["gaussian", "flat_top"])
@pytest.mark.parametrize("z", [5.0, 100.0, 300.0])
def test_beam_power_conservation(profile, z):
    src = make(profile=profile)
    w = 4.0 + z * np.tan(np.radians(6.0))
    assert plane_power(src, z, 4.0 * w, n=1201) == pytest.approx(1.0, rel=2e-3)


def test_collimated_beam_power_and_constant_irradiance():
    src = make(alpha=0.0)
    assert plane_power(src, 50.0, 14.0) == pytest.approx(1.0, rel=2e-3)
    e1 = so.irradiance_per_mW(src, np.array([[0.0, 50.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))[0]
    e2 = so.irradiance_per_mW(src, np.array([[0.0, -150.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))[0]
    assert e1 == pytest.approx(e2)
    assert e1 == pytest.approx(2.0 / (np.pi * 16.0), rel=1e-9)


def test_inverse_square_far_field():
    src = make()
    e = lambda z: so.irradiance_per_mW(src, np.array([[0.0, 100.0 - z, 0.0]]), np.array([[0.0, 1.0, 0.0]]))[0]
    z1, z2 = 20_000.0, 40_000.0
    assert e(z1) / e(z2) == pytest.approx(4.0, rel=0.01)
    # exact: w(z) = D/2 + z tan(alpha)
    w = 4.0 + 1000.0 * np.tan(np.radians(6.0))
    assert e(1000.0) == pytest.approx(2.0 / (np.pi * w * w), rel=1e-9)


def test_behind_aperture_and_back_facing_are_dark():
    src = make()
    up = np.array([[0.0, 1.0, 0.0]])
    assert so.irradiance_per_mW(src, np.array([[0.0, 110.0, 0.0]]), up)[0] == 0.0       # behind source
    down = np.array([[0.0, -1.0, 0.0]])
    assert so.irradiance_per_mW(src, np.array([[0.0, 10.0, 0.0]]), down)[0] == 0.0      # faces away


def test_incidence_cosine():
    src = make(alpha=0.0)
    p = np.array([[0.0, 0.0, 0.0]])
    n45 = np.array([[np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0]])
    e0 = so.irradiance_per_mW(src, p, np.array([[0.0, 1.0, 0.0]]))[0]
    assert so.irradiance_per_mW(src, p, n45)[0] == pytest.approx(e0 * np.cos(np.pi / 4))


def test_beam_samples_direction_points_to_source():
    src = make()
    bs = so.beam_samples(src, np.array([[30.0, 0.0, 0.0]]))
    u = bs.u_to_source[0]
    assert np.linalg.norm(u) == pytest.approx(1.0)
    # following u for dist reaches the aperture plane y = 100
    assert (30.0 * np.array([1, 0, 0]) + u * bs.dist_mm[0])[1] == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("kind,fwhm", [("tunable_lamp", 10.0), ("led_array", 18.0), ("laser", 1.0), ("laser", 0.3)])
def test_line_shapes_integrate_to_one(kind, fwhm):
    src = make(kind=kind, line_fwhm_nm=fwhm)
    grid = np.arange(250.0, 900.0 + 1e-9, 1.0)
    for center in (310.0, 405.0, 532.0):
        f = so.line_shape(src, center, grid)
        assert float(f.sum() * 1.0) == pytest.approx(1.0, abs=1e-9)
        assert (f >= 0).all()


def test_laser_one_nm_bin_exact_vs_point_sampling():
    src = make(kind="laser", line_fwhm_nm=1.0)
    grid = np.arange(250.0, 900.0 + 1e-9, 1.0)
    f = so.line_shape(src, 488.0, grid)
    # exact bin integral of a Gaussian centred on a bin: erf(0.5/(√2σ)) with σ = 1/2.3548
    from math import erf, sqrt
    sigma = 1.0 / (2 * sqrt(2 * np.log(2)))
    assert f[grid == 488.0][0] == pytest.approx(erf(0.5 / (sqrt(2) * sigma)), rel=1e-12)
    assert f.argmax() == np.searchsorted(grid, 488.0)


def test_lamp_fwhm_and_led_asymmetry():
    grid = np.arange(380.0, 520.0, 0.05)
    lamp = so.line_shape(make(line_fwhm_nm=10.0), 450.0, grid)
    half = grid[lamp >= lamp.max() / 2]
    assert half.max() - half.min() == pytest.approx(10.0, abs=0.2)
    led = so.line_shape(make(kind="led_array", line_fwhm_nm=18.0), 450.0, grid)
    half = grid[led >= led.max() / 2]
    assert half.max() - half.min() == pytest.approx(18.0, abs=0.3)
    dx = 0.05
    red = led[grid > 450.0].sum() * dx
    blue = led[grid < 450.0].sum() * dx
    assert red > blue * 1.2                         # longer red tail
    assert 0.0 < led.sum() * dx <= 1.0 + 1e-9


def test_leakage_od_scaling_and_kinds():
    grid = np.arange(250.0, 900.0 + 1e-9, 1.0)
    l4 = so.leakage_spectrum(make(excitation_filter_od=4.0), 365.0, grid)
    l6 = so.leakage_spectrum(make(excitation_filter_od=6.0), 365.0, grid)
    assert np.allclose(l6 * 100.0, l4)
    assert l4[grid == 365.0][0] < 1e-3 * l4.max() + 1e-12                  # notch at the line
    assert 0 < l4.sum() < 0.02                       # ~1% of the line (OD 4, broadband vs a 10 nm line)
    laser = so.leakage_spectrum(make(kind="laser", excitation_filter_od=6.0), 488.0, grid)
    assert laser.sum() < 1e-8
    led = so.leakage_spectrum(make(kind="led_array", excitation_filter_od=3.0), 450.0, grid)
    assert led.sum() > laser.sum()
    sp = so.excitation_spectrum(make(), 365.0, grid)
    assert np.allclose(sp.total, sp.line + sp.leakage)


def test_xenon_spectrum_shape():
    nm = np.arange(250.0, 1000.0)
    x = so.xenon_relative_spectrum(nm)
    assert (x >= 0).all() and np.isfinite(x).all()
    at = lambda w: float(so.xenon_relative_spectrum(np.array([w]))[0])
    assert at(260) < 0.1 * at(500)                 # UV roll-off
    assert at(300) < at(400) < at(500)
    assert at(881.9) > 2.0 * at(870)               # NIR line
    assert at(828.0) > 1.5 * at(850)


def test_drifted_powers():
    steps = [ExcitationStep(wavelength_nm=400.0 + (i % 500), power_mW=2.0) for i in range(20000)]
    src = make(power_drift_pct=2.0)
    p = np.array(so.drifted_powers(src, steps, np.random.default_rng(0), True))
    assert p.mean() == pytest.approx(2.0, rel=2e-3)
    assert p.std() == pytest.approx(0.04, rel=0.05)
    again = np.array(so.drifted_powers(src, steps, np.random.default_rng(0), True))
    assert np.array_equal(p, again)
    off = so.drifted_powers(src, steps[:5], np.random.default_rng(0), False)
    assert off == [2.0] * 5
    none = so.drifted_powers(make(power_drift_pct=0.0), steps[:5], np.random.default_rng(0), True)
    assert none == [2.0] * 5
