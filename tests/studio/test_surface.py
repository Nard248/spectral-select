"""Fresnel / Snell / GGX identities."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from spectraforge.studio import surface as sf


def test_normal_incidence_reflectance():
    for n1, n2 in [(1.0, 1.5), (1.5, 1.0), (1.333, 1.52), (1.0, 2.4)]:
        assert sf.fresnel_reflectance(1.0, n1, n2) == pytest.approx(((n1 - n2) / (n1 + n2)) ** 2, rel=1e-12)


@given(st.floats(0.0, 1.0), st.floats(1.0, 2.5), st.floats(1.0, 2.5))
@settings(max_examples=200, deadline=None)
def test_r_plus_t_is_one_and_bounded(c, n1, n2):
    r = float(sf.fresnel_reflectance(c, n1, n2))
    t = float(sf.fresnel_transmittance(c, n1, n2))
    assert 0.0 <= r <= 1.0
    assert r + t == pytest.approx(1.0, abs=1e-12)


def test_total_internal_reflection_above_critical_angle():
    n1, n2 = 1.5, 1.0
    cc = sf.critical_angle_cos(n1, n2)                  # cos of critical angle
    below = np.cos(np.arccos(cc) - 0.01)                # smaller angle than critical
    above = np.cos(np.arccos(cc) + 0.01)
    assert float(sf.fresnel_reflectance(above, n1, n2)) == 1.0
    assert float(sf.fresnel_reflectance(below, n1, n2)) < 1.0
    _, tir = sf.snell_cos_t(above, n1, n2)
    assert bool(tir)
    # near the critical angle R rises steeply to 1
    assert float(sf.fresnel_reflectance(np.cos(np.arccos(cc) - 1e-4), n1, n2)) > 0.9


def test_grazing_incidence_reflects_everything():
    assert float(sf.fresnel_reflectance(0.0, 1.0, 1.5)) == pytest.approx(1.0)


@given(st.floats(0.01, 1.0), st.floats(1.0, 2.0), st.floats(1.0, 2.0))
@settings(max_examples=100, deadline=None)
def test_fresnel_reciprocity(c, n1, n2):
    """R(θ1; n1→n2) == R(θ2; n2→n1) with θ2 the refracted angle (Stokes)."""
    ct, tir = sf.snell_cos_t(c, n1, n2)
    if bool(tir) or float(ct) < 1e-6:
        return
    assert float(sf.fresnel_reflectance(c, n1, n2)) == pytest.approx(float(sf.fresnel_reflectance(ct, n2, n1)), abs=1e-9)


def test_refract_vector_obeys_snell_and_is_unit():
    rng = np.random.default_rng(0)
    n = np.array([0.0, 1.0, 0.0])
    d = rng.normal(size=(200, 3))
    d[:, 1] = -np.abs(d[:, 1]) - 0.05
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t, tir = sf.refract(d, np.tile(n, (200, 1)), 1.0, 1.5)
    assert not tir.any()
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0)
    sin_i = np.linalg.norm(d[:, [0, 2]], axis=1)
    sin_t = np.linalg.norm(t[:, [0, 2]], axis=1)
    assert np.allclose(1.0 * sin_i, 1.5 * sin_t, atol=1e-12)
    assert (t[:, 1] < 0).all()
    # tangential direction preserved
    assert np.allclose(np.cross(d, t)[:, 1], 0.0, atol=1e-12)
    # normal orientation does not matter
    t2, _ = sf.refract(d, -np.tile(n, (200, 1)), 1.0, 1.5)
    assert np.allclose(t, t2)


def test_refract_tir_returns_mirror_reflection():
    n = np.array([[0.0, 1.0, 0.0]])
    th = np.radians(70)
    d = np.array([[np.sin(th), -np.cos(th), 0.0]])
    t, tir = sf.refract(d, n, 1.5, 1.0)
    assert bool(tir[0])
    assert np.allclose(t, [[np.sin(th), np.cos(th), 0.0]])


def test_ggx_D_is_normalised():
    for alpha in (0.05, 0.2, 0.6):
        th = np.linspace(0, np.pi / 2, 400001)
        integrand = sf.ggx_D(np.cos(th), alpha) * np.cos(th) * 2 * np.pi * np.sin(th)
        val = np.trapezoid(integrand, th)
        assert val == pytest.approx(1.0, abs=2e-3)


def test_ggx_brdf_reciprocal_and_albedo():
    rng = np.random.default_rng(1)
    n = np.tile([0.0, 0.0, 1.0], (500, 1))

    def rand_dirs():
        v = rng.normal(size=(500, 3))
        v[:, 2] = np.abs(v[:, 2]) + 0.1
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    wi, wo = rand_dirs(), rand_dirs()
    f1 = sf.ggx_brdf(n, wi, wo, 0.3, 1.5)
    f2 = sf.ggx_brdf(n, wo, wi, 0.3, 1.5)
    assert np.allclose(f1, f2, rtol=1e-9)
    # directional albedo for a nearly smooth surface at normal incidence ≈ Fresnel R(0)
    alpha = 0.05
    th = np.linspace(1e-4, np.pi / 2 - 1e-4, 4001)
    ph = np.linspace(0, 2 * np.pi, 361)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    wo = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    wi = np.tile([0.0, 0.0, 1.0], (len(wo), 1))
    f = sf.ggx_brdf(np.tile([0.0, 0.0, 1.0], (len(wo), 1)), wi, wo, alpha, 1.5).reshape(TH.shape)
    albedo = np.trapezoid(np.trapezoid(f * np.cos(TH) * np.sin(TH), ph, axis=1), th)
    assert albedo == pytest.approx(0.04, rel=0.15)
    assert albedo < 1.0


def test_ggx_zero_below_horizon():
    n = np.array([[0.0, 0.0, 1.0]])
    assert sf.ggx_brdf(n, np.array([[0.0, 0.6, -0.8]]), np.array([[0.0, 0.0, 1.0]]), 0.2, 1.5)[0] == 0.0
