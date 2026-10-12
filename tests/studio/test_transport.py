"""Verification of the SpectralForge Studio transport engine (physics credibility is the product).

Every test prints / asserts a number against an independent reference:

* MCML benchmark (Wang & Jacques 1995 / Prahl): mu_a=1, mu_s=9, g=0.75, n=1, d=0.2 mm
  -> Rd = 0.09734, Tt = 0.66096.
* exact analytic multiple-reflection transmittance of an index-mismatched, non-scattering slab;
* diffusion theory (exp(-mu_eff z), Farrell/Patterson-type albedo formula with Egan-Hilgeman
  internal reflection);
* energy conservation R + T + A = 1;
* reciprocity normalisation vs a brute-force forward MC that launches isotropic emitters;
* turbid -> clear closed form as mu_s -> 0.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from spectraforge.studio import transport as T
from spectraforge.studio import transport_mc as mc
from spectraforge.studio.contracts import SPECTRAL_GRID_NM, MaterialOptics

WL = SPECTRAL_GRID_NM


def make_optics(mua=0.1, mus=10.0, g=0.8, n=1.4, clear=False, mua_fn=None):
    mua_arr = np.full_like(WL, mua) if mua_fn is None else mua_fn(WL)
    mus_arr = np.zeros_like(WL) if clear else np.full_like(WL, mus)
    return MaterialOptics("m", WL, mua_arr.astype(float), {}, mus_arr, g, n, clear, {}, {})


def two_band_optics(mua_ex, mus_ex, mua_em, mus_em, g, n, ex=405.0):
    """Constant (mua, mus) up to and including ``ex`` and a second constant set above it."""
    lo = WL <= ex + 1e-9
    return MaterialOptics("m", WL, np.where(lo, mua_ex, mua_em).astype(float), {},
                          np.where(lo, mus_ex, mus_em).astype(float), g, n, False, {}, {})


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return T.get_transport_model(n_photons=120_000, seed=0,
                                 cache_dir=tmp_path_factory.mktemp("mccache"))


# ============================================================================================
# Fresnel + clear closed form
# ============================================================================================
def test_fresnel_normal_incidence_and_tir():
    r = T.fresnel_reflectance(1.0, 1.5, 1.0)
    assert r == pytest.approx(0.04, abs=1e-12)                       # ((n-1)/(n+1))^2
    assert T.fresnel_reflectance(1.5, 1.0, 0.5) == 1.0               # beyond critical angle (41.8 deg)
    # reciprocity of T at conjugate angles
    c_ext = 0.7
    c_int = np.sqrt(1 - (1 / 1.5) ** 2 * (1 - c_ext ** 2))
    assert 1 - T.fresnel_reflectance(1.0, 1.5, c_ext) == pytest.approx(
        1 - T.fresnel_reflectance(1.5, 1.0, c_int), abs=1e-12)


def _clear_ref(n, no, mua_ex, mua_em, ci_ext, co_ext, d, path=None):
    """Independent re-implementation by numerical depth integration."""
    ci = np.sqrt(1 - (no / n) ** 2 * (1 - ci_ext ** 2))
    co = np.sqrt(1 - (no / n) ** 2 * (1 - co_ext ** 2))

    def R(n1, n2, c1):
        s2 = (n1 / n2) ** 2 * (1 - c1 ** 2)
        c2 = np.sqrt(1 - s2)
        return 0.5 * (((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)) ** 2
                      + ((n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)) ** 2)
    t_in = 1 - R(no, n, ci_ext)
    t_out = 1 - R(n, no, co)
    L = path if path is not None else d / co
    s = np.linspace(0, L, 200001)                                     # along the view chord
    integrand = np.exp(-mua_ex * s * co / ci) * np.exp(-mua_em * s)
    integral = np.trapezoid(integrand, s)
    return t_in * t_out / (4 * np.pi * n * n * ci) * integral


def test_clear_closed_form_vs_numerical_integration():
    opt = make_optics(mua=0.0, clear=True, n=1.4,
                      mua_fn=lambda w: 0.5 + 2.0 * np.exp(-0.5 * ((w - 520) / 30) ** 2))
    i405 = int(np.where(WL == 405)[0][0])
    mua_ex = opt.mua_per_mm[i405]
    ci, co, d = 0.8, 0.6, 1.3
    k = T.clear_medium_kernel(opt, 405.0, [ci], [co], [d])
    for lam in (450, 520, 600):
        i = int(np.where(WL == lam)[0][0])
        ref = _clear_ref(1.4, 1.0, mua_ex, opt.mua_per_mm[i], ci, co, d)
        assert k.fluorescence_G[0, i] == pytest.approx(ref, rel=2e-4)
    assert k.elastic_rho[0] == 0.0
    assert 0 < k.mean_depth_mm[0] < d


def test_clear_dilute_limit_and_path_override():
    opt = make_optics(mua=1e-9, clear=True, n=1.33)
    ci, co = 0.9, 0.7
    L = 2.5
    k = T.clear_medium_kernel(opt, 405.0, [ci], [co], [99.0], path_mm=[L])
    cip = np.sqrt(1 - (1 / 1.33) ** 2 * (1 - ci ** 2))
    cop = np.sqrt(1 - (1 / 1.33) ** 2 * (1 - co ** 2))
    t_in = 1 - T.fresnel_reflectance(1.0, 1.33, ci)
    t_out = 1 - T.fresnel_reflectance(1.33, 1.0, cop)
    expected = t_in * t_out * L / (4 * np.pi * 1.33 ** 2 * cip)       # G -> T_in T_out L / (4 pi n^2 cos_i')
    assert k.fluorescence_G[0, 100] == pytest.approx(expected, rel=1e-5)
    # path_mm=None -> L = d / cos(theta_o')
    k2 = T.clear_medium_kernel(opt, 405.0, [ci], [co], [L * cop])
    assert k2.fluorescence_G[0, 100] == pytest.approx(expected, rel=1e-5)


def test_clear_beer_lambert_saturation_and_n_outside():
    # strong absorption -> G ~ T T /(4 pi n^2 cos_i') / kappa (inner-filter limit, independent of L)
    opt = make_optics(mua=3.0, clear=True, n=1.45)
    ci, co = 0.75, 0.75
    kk = T.clear_medium_kernel(opt, 405.0, [ci], [co], [50.0])
    cip = np.sqrt(1 - (1 / 1.45) ** 2 * (1 - ci ** 2))
    kappa = 3.0 * (co_i := np.sqrt(1 - (1 / 1.45) ** 2 * (1 - co ** 2))) / cip + 3.0
    t = (1 - T.fresnel_reflectance(1.0, 1.45, ci)) * (1 - T.fresnel_reflectance(1.45, 1.0, co_i))
    assert kk.fluorescence_G[0, 10] == pytest.approx(t / (4 * np.pi * 1.45 ** 2 * cip) / kappa, rel=1e-5)
    # glass in front (n_outside = 1.5 > n) can make the excitation/view ray impossible -> G = 0
    g = T.clear_medium_kernel(make_optics(mua=0.1, clear=True, n=1.2), 405.0, [0.2], [0.9], [1.0],
                              n_outside=[1.6])
    assert g.fluorescence_G[0].max() == 0.0
    # semi-infinite non-absorbing path stays finite
    assert np.isfinite(T.clear_medium_kernel(opt, 405.0, [1.0], [1.0], [np.inf]).fluorescence_G).all()


# ============================================================================================
# Engine verification: MCML benchmark, analytic slab, energy conservation
# ============================================================================================
def test_mcml_benchmark_wang_jacques():
    r = mc.mcml_slab(1.0, 9.0, 0.75, 1.0, 1.0, 0.2, n_photons=16_000_000, seed=3)
    print(f"\nMCML benchmark: Rd={r['Rd']:.5f} (ref 0.09734)  Tt={r['Tt']:.5f} (ref 0.66096)  "
          f"R+T+A={r['Rd'] + r['Tt'] + r['A']:.6f}")
    assert r["Rd"] == pytest.approx(0.09734, rel=0.01)
    assert r["Tt"] == pytest.approx(0.66096, rel=0.01)
    assert r["Rd"] + r["Tt"] + r["A"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.slow
def test_mcml_benchmark_tight():
    r = mc.mcml_slab(1.0, 9.0, 0.75, 1.0, 1.0, 0.2, n_photons=200_000_000, seed=4)
    print(f"\nMCML tight: Rd={r['Rd']:.5f} Tt={r['Tt']:.5f}")
    assert r["Rd"] == pytest.approx(0.09734, rel=0.003)
    assert r["Tt"] == pytest.approx(0.66096, rel=0.003)


def test_index_mismatched_nonscattering_slab_exact():
    """Normal incidence, no scattering: T = (1-R)^2 e^{-mu d} / (1 - R^2 e^{-2 mu d}),
    Rtot = R + (1-R)^2 R e^{-2 mu d} / (1 - R^2 e^{-2 mu d}) (incoherent multiple reflections)."""
    n, mua, d = 1.5, 0.8, 1.0
    R = ((n - 1) / (n + 1)) ** 2
    e = np.exp(-mua * d)
    t_ref = (1 - R) ** 2 * e / (1 - R ** 2 * e ** 2)
    rd_ref = (1 - R) ** 2 * R * e ** 2 / (1 - R ** 2 * e ** 2)         # excl. specular R
    r = mc.mcml_slab(mua, 1e-9, 0.0, n, 1.0, d, n_photons=4_000_000, seed=2)
    print(f"\nmismatched clear slab: Tt={r['Tt']:.5f} (exact {t_ref:.5f})  Rd={r['Rd']:.5f} (exact {rd_ref:.5f})")
    # mcml_slab results are per unit INCIDENT power (already multiplied by 1 - Rsp)
    assert r["Tt"] == pytest.approx(t_ref, rel=0.01)
    assert r["Rd"] == pytest.approx(rd_ref, rel=0.02)


def test_energy_conservation_mismatched_turbid():
    r = mc.mcml_slab(0.3, 12.0, 0.85, 1.4, 1.0, 0.7, mu0_ext=0.6, n_photons=1_000_000, seed=1)
    # specular reflection + diffuse reflection + transmission + absorption = incident power
    assert r["Rsp"] + r["Rd"] + r["Tt"] + r["A"] == pytest.approx(1.0, abs=1e-9)
    assert r["Rsp"] == pytest.approx(float(T.fresnel_reflectance(1.0, 1.4, 0.6)), abs=1e-12)


def _white_rtA(g, n, no, mu_a, mu_s, d, mu_int=1.0, n_photons=600_000):
    tau = np.array([mu_s * d])
    NU = 40
    u = np.sin(np.pi * np.arange(NU + 1) / (2 * NU)) ** 2
    o = mc.bake_arrays(g, n, no, np.array([mu_int]), tau, u, T.ZINF_EDGES, T.RAD_EDGES, n_photons, 7)
    e = T._bin_weights(np.array([mu_a / mu_s]))[:, 0]
    with np.errstate(all="ignore"):
        rd = float((o["rtop_fin"][0, 0].astype(float) @ e).sum())
        tt = float(o["tt_fin"][0, 0].astype(float) @ e)
        absorbed = (mu_a / mu_s) * float((o["m_fin"][0, 0].astype(float) @ e).sum())
    return rd, tt, absorbed


@pytest.mark.slow
def test_white_mc_matches_mcml_benchmark_and_conserves_energy():
    rd, tt, a = _white_rtA(0.75, 1.0, 1.0, 1.0, 9.0, 0.2, n_photons=2_000_000)
    print(f"\nwhite-MC benchmark: Rd={rd:.5f} (0.09734) Tt={tt:.5f} (0.66096) A={a:.5f} sum={rd + tt + a:.5f}")
    assert rd == pytest.approx(0.09734, rel=0.015)
    assert tt == pytest.approx(0.66096, rel=0.01)
    assert rd + tt + a == pytest.approx(1.0, abs=0.01)


@pytest.mark.slow
def test_white_mc_vs_conventional_mcml_index_mismatch():
    """Independent code paths agree for an index-mismatched, oblique, turbid slab."""
    n, g, mua, mus, d = 1.4, 0.8, 0.5, 10.0, 0.6
    mu_int = 0.8
    rd, tt, a = _white_rtA(g, n, 1.0, mua, mus, d, mu_int=mu_int, n_photons=2_000_000)
    mu_ext = np.sqrt(1 - (n / 1.0) ** 2 * (1 - mu_int ** 2))
    ref = mc.mcml_slab(mua, mus, g, n, 1.0, d, mu0_ext=mu_ext, n_photons=8_000_000, seed=9)
    f = 1 - ref["Rsp"]
    print(f"\nwhite vs MCML (n=1.4): Rd {rd:.5f}/{ref['Rd'] / f:.5f}  Tt {tt:.5f}/{ref['Tt'] / f:.5f}  "
          f"A {a:.5f}/{ref['A'] / f:.5f}")
    assert rd == pytest.approx(ref["Rd"] / f, rel=0.02)
    assert tt == pytest.approx(ref["Tt"] / f, rel=0.015)
    assert a == pytest.approx(ref["A"] / f, rel=0.01)


# ============================================================================================
# Diffusion limit
# ============================================================================================
def _semi_infinite_white(g, n, mu_a, mu_s, n_photons=1_500_000):
    NU = 40
    u = np.sin(np.pi * np.arange(NU + 1) / (2 * NU)) ** 2
    o = mc.bake_arrays(g, n, 1.0, np.array([1.0]), T.TAU_NODES[:1], u, T.ZINF_EDGES, T.RAD_EDGES,
                       n_photons, 3)
    e = T._bin_weights(np.array([mu_a / mu_s]))[:, 0]
    with np.errstate(all="ignore"):
        mass = o["m_inf"][0].astype(float) @ e
        rd = float((o["rtop_inf"][0].astype(float) @ e).sum())
    return mass, rd


@pytest.mark.slow
def test_diffusion_limit_decay_constant():
    g, n, mua, musp = 0.9, 1.0, 0.01, 1.0                    # mu_s = 10, albedo' = 0.990
    mus = musp / (1 - g)
    mass, _ = _semi_infinite_white(g, n, mua, mus)
    mu_eff = np.sqrt(3 * mua * (mua + musp))                  # 1/mm
    # fit the bin-integrated exponential beyond 5 transport mfp
    edges = T.ZINF_EDGES / mus                                # mm
    sel = (edges[:-1] > 5.0 / musp) & (edges[1:] < 5.0 / mu_eff)
    lo, hi, m = edges[:-1][sel], edges[1:][sel], mass[sel]
    from scipy.optimize import least_squares

    def resid(p):
        a, mu = p
        return (a * (np.exp(-mu * lo) - np.exp(-mu * hi)) / mu - m) / np.sqrt(np.maximum(m, 1e-12))
    sol = least_squares(resid, [m.sum() * mu_eff, mu_eff])
    print(f"\ndiffusion decay: fitted mu_eff={sol.x[1]:.4f}/mm vs diffusion {mu_eff:.4f}/mm")
    assert sol.x[1] == pytest.approx(mu_eff, rel=0.04)


@pytest.mark.slow
@pytest.mark.parametrize("n,albedo_pairs,tol", [(1.0, [(0.9, 0.9), (0.9, 0.99), (0.9, 0.999)], 0.03),
                                                (1.4, [(0.9, 0.99), (0.9, 0.999)], 0.06)])
def test_diffuse_reflectance_vs_dipole(n, albedo_pairs, tol):
    """Semi-infinite total diffuse reflectance vs the Farrell/Patterson albedo formula
    R = (a'/2)(1 + exp(-(4/3) A sqrt(3(1-a')))) exp(-sqrt(3(1-a'))), A from Egan-Hilgeman."""
    for g, ap in albedo_pairs:
        musp = 1.0
        mus = musp / (1 - g)
        mua = musp * (1 - ap) / ap
        _, rd = _semi_infinite_white(g, n, mua, mus, n_photons=800_000)
        rdi = -1.440 / n ** 2 + 0.710 / n + 0.668 + 0.0636 * n if n != 1.0 else 0.0
        A = (1 + rdi) / (1 - rdi)
        s3 = np.sqrt(3 * (1 - ap))
        ref = 0.5 * ap * (1 + np.exp(-4.0 / 3.0 * A * s3)) * np.exp(-s3)
        print(f"\nn={n} a'={ap}: MC Rd={rd:.4f}  dipole {ref:.4f}  (ratio {rd / ref:.3f})")
        assert rd == pytest.approx(ref, rel=tol)


# ============================================================================================
# Reciprocity normalisation vs brute-force forward MC (the key verification of G)
# ============================================================================================
def _brute_force_G(a_ex, s_ex, a_em, s_em, g, n, no, d, ci_ext, co_ext, nz=40):
    """G by two conventional MCs: excitation fluence F_ex(z) (absorption scoring) and radiance
    escaping towards co_ext from isotropic emitters at each depth bin centre (forward emitters)."""
    r = mc.mcml_slab(a_ex, s_ex, g, n, no, d, mode="collimated", mu0_ext=ci_ext, n_photons=3_000_000,
                     nz_bins=nz, zmax=d, seed=5)
    dz = d / nz
    f_ex = r["absz"] / (a_ex * dz)                                    # per unit incident irradiance
    b = int(co_ext ** 2 * mc.N_COSBIN)
    lum = np.zeros(nz)
    for j in range(nz):
        rr = mc.mcml_slab(a_em, s_em, g, n, no, d, mode="point", z0=(j + 0.5) * dz, n_photons=300_000,
                          seed=11 + j, nz_bins=4, zmax=d)
        lum[j] = rr["hist_cos2"][b] / (np.pi / mc.N_COSBIN) / no ** 2  # air-equivalent radiance
    return float((f_ex * lum * dz).sum())


RECIPROCITY_CASES = [
    # (mua_ex, mus_ex, mua_em, mus_em, g, n, n_out, d, cos_in, thickness label)
    (0.1, 10.0, 0.3, 8.0, 0.8, 1.4, 1.0, 1.0, 0.8),
    (0.5, 20.0, 0.05, 15.0, 0.9, 1.5, 1.0, 6.0, 0.9),
    (0.2, 12.0, 0.1, 10.0, 0.85, 1.4, 1.5, 2.0, 0.7),
]


@pytest.mark.slow
@pytest.mark.parametrize("case", RECIPROCITY_CASES)
def test_reciprocity_G_vs_forward_emitter_mc(model, case):
    a, s, a2, s2, g, n, no, d, ci = case
    co = np.sqrt(5.5 / mc.N_COSBIN)                                   # centre of a cos^2 bin
    opt = two_band_optics(a, s, a2, s2, g, n)
    k = model.kernels(opt, 405.0, [ci], [co], [d], n_outside=[no])
    i = int(np.where(WL == 520)[0][0])
    G = float(k.fluorescence_G[0, i])
    ref = _brute_force_G(a, s, a2, s2, g, n, no, d, ci, co)
    print(f"\nreciprocity {case}: table G={G:.5f}  brute-force forward MC={ref:.5f}  ratio={G / ref:.4f}")
    assert G == pytest.approx(ref, rel=0.035)


# ============================================================================================
# Kernel behaviour: limits, monotonicity, elastic, radial, speed, determinism
# ============================================================================================
def test_turbid_tends_to_clear_as_mus_to_zero(model):
    """mu_s -> 0 reproduces the closed form exactly when there is no index mismatch (the closed form
    ignores internal reflections off the far face; the MC includes them)."""
    d, ci, co = 1.5, 0.8, 0.7
    for mus in (0.003, 0.02):
        opt = make_optics(mua=0.4, mus=mus, g=0.8, n=1.0)
        kt = model.kernels(opt, 405.0, [ci], [co], [d])
        kc = T.clear_medium_kernel(make_optics(mua=0.4, clear=True, n=1.0), 405.0, [ci], [co], [d])
        ratio = float(kt.fluorescence_G[0, 100] / kc.fluorescence_G[0, 100])
        print(f"\nn=1, mu_s={mus}/mm: turbid/clear G = {ratio:.4f}")
        assert ratio == pytest.approx(1.0, abs=0.04)


@pytest.mark.slow
def test_turbid_to_clear_with_index_mismatch_matches_internal_reflection_mc(model):
    """n = 1.4: the clear closed form omits far-face internal reflections (a few %); the turbid
    kernel at mu_s -> 0 must equal a brute-force MC of the same (practically clear) slab."""
    d, ci, co = 1.5, 0.8, np.sqrt(5.5 / mc.N_COSBIN)
    opt = make_optics(mua=0.4, mus=0.003, g=0.8, n=1.4)
    kt = model.kernels(opt, 405.0, [ci], [co], [d])
    kc = T.clear_medium_kernel(make_optics(mua=0.4, clear=True, n=1.4), 405.0, [ci], [co], [d])
    ref = _brute_force_G(0.4, 0.003, 0.4, 0.003, 0.8, 1.4, 1.0, d, ci, co)
    g_t, g_c = float(kt.fluorescence_G[0, 100]), float(kc.fluorescence_G[0, 100])
    print(f"\nn=1.4: turbid {g_t:.5f}  clear closed form {g_c:.5f}  brute-force {ref:.5f}")
    assert g_t == pytest.approx(ref, rel=0.035)
    assert 1.0 <= g_t / g_c < 1.15


def test_monotonic_absorption_and_inner_filter(model):
    ci = co = np.array([0.8])
    gs = []
    for mua in (0.02, 0.1, 0.5, 2.0):
        k = model.kernels(make_optics(mua=mua, mus=10.0), 405.0, ci, co, [np.inf])
        gs.append(float(k.fluorescence_G[0, 300]))
    assert all(a > b for a, b in zip(gs, gs[1:])), gs             # more absorption -> less G
    opt = make_optics(mua=0.3, mus=10.0)
    ds = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0, np.inf])
    k = model.kernels(opt, 405.0, np.full_like(ds, 0.8), np.full_like(ds, 0.8), ds)
    g = k.fluorescence_G[:, 300].astype(float)
    assert np.all(np.diff(g) > -0.03 * g.max()), g                # thicker -> larger, saturating (MC noise ~1.5%)
    inc = np.diff(g)
    assert abs(inc[-2]) < 0.04 * g[-1] and inc[2] > inc[-3]       # saturation = inner filter
    assert 0 < k.mean_depth_mm[-1] < 10.0


def test_elastic_hemispherical_reflectance_vs_mcml(model):
    """int rho cos(theta) dOmega = T_in-weighted diffuse reflectance (compare to conventional MCML)."""
    mua, mus, g, n, d = 0.3, 12.0, 0.8, 1.4, 1.0
    opt = make_optics(mua=mua, mus=mus, g=g, n=n)
    ci_ext = 0.85
    cb = np.sqrt((np.arange(mc.N_COSBIN) + 0.5) / mc.N_COSBIN)         # bin-centre exit cosines
    k = model.kernels(opt, 405.0, np.full(mc.N_COSBIN, ci_ext), cb, np.full(mc.N_COSBIN, d))
    rd_model = float(np.sum(k.elastic_rho) * np.pi / mc.N_COSBIN)
    ref = mc.mcml_slab(mua, mus, g, n, 1.0, d, mu0_ext=ci_ext, n_photons=6_000_000, seed=2)
    print(f"\nelastic hemispherical reflectance: model {rd_model:.4f}  MCML Rd {ref['Rd']:.4f}")
    assert rd_model == pytest.approx(ref["Rd"], rel=0.05)
    assert np.all(np.isfinite(k.elastic_rho)) and np.all(k.elastic_rho > 0)


def test_radial_profile_normalised_and_absorption_narrows(model):
    wide = model.kernels(make_optics(mua=0.01, mus=10.0), 405.0, [1.0], [1.0], [np.inf])
    narrow = model.kernels(make_optics(mua=2.0, mus=10.0), 405.0, [1.0], [1.0], [np.inf])
    for k in (wide, narrow):
        assert k.radial_profile.shape == (len(k.radial_lambda_nm), len(k.radial_r_mm))
        for p in k.radial_profile:
            assert np.trapezoid(2 * np.pi * k.radial_r_mm * p, k.radial_r_mm) == pytest.approx(1.0, rel=1e-6)
    def mean_r(k):
        r, p = k.radial_r_mm, k.radial_profile[0]
        return np.trapezoid(2 * np.pi * r * r * p, r)
    print(f"\nmean subsurface spread: low absorption {mean_r(wide):.3f} mm, high {mean_r(narrow):.3f} mm")
    assert mean_r(narrow) < mean_r(wide)


@pytest.mark.slow
def test_speed_determinism_and_cache(tmp_path):
    opt = make_optics(mua=0.1, mus=10.0, g=0.9, n=1.4)
    rng = np.random.default_rng(1)
    P = 90_000
    ci, co = rng.uniform(0.3, 1, P), rng.uniform(0.3, 1, P)
    th = np.where(rng.random(P) < 0.3, np.inf, rng.uniform(0.05, 5, P))
    t0 = time.time()
    m1 = T.get_transport_model(n_photons=24_000, seed=0, cache_dir=tmp_path)
    m1.kernels(opt, 405.0, ci[:10], co[:10], th[:10])
    t_bake = time.time() - t0
    m1.kernels(opt, 405.0, ci, co, th)
    t0 = time.time()
    k1 = m1.kernels(opt, 405.0, ci, co, th)
    t_k = time.time() - t0
    print(f"\nbake (incl. JIT, n_photons=24k) {t_bake:.1f}s; kernels for P=90k, W=651: {t_k * 1e3:.0f} ms")
    assert k1.fluorescence_G.shape == (P, len(WL)) and np.isfinite(k1.fluorescence_G).all()
    assert t_k < 1.0
    # cache hit: a new model with the same cache dir loads identical tables quickly
    t0 = time.time()
    m2 = T.get_transport_model(n_photons=24_000, seed=0, cache_dir=tmp_path)
    k2 = m2.kernels(opt, 405.0, ci[:2000], co[:2000], th[:2000])
    assert time.time() - t0 < 5.0 and len(list(tmp_path.glob("tables_*.npz"))) == 1
    np.testing.assert_array_equal(k1.fluorescence_G[:2000], k2.fluorescence_G)
    # determinism with a fresh cache dir (re-bake)
    m3 = T.get_transport_model(n_photons=24_000, seed=0, cache_dir=tmp_path / "other")
    k3 = m3.kernels(opt, 405.0, ci[:2000], co[:2000], th[:2000])
    np.testing.assert_array_equal(k1.fluorescence_G[:2000], k3.fluorescence_G)


def test_dispatch_on_clear_and_invalid_pixels(model):
    clear = make_optics(mua=0.2, clear=True, n=1.4)
    k = model.kernels(clear, 405.0, [0.9], [0.8], [1.0])
    ref = T.clear_medium_kernel(clear, 405.0, [0.9], [0.8], [1.0])
    np.testing.assert_array_equal(k.fluorescence_G, ref.fluorescence_G)
    turb = make_optics(mua=0.1, mus=10.0)
    k = model.kernels(turb, 405.0, [0.9, -0.1, 0.9], [0.8, 0.8, 0.0], [1.0, 1.0, 1.0])
    assert k.fluorescence_G[0].max() > 0 and k.fluorescence_G[1].max() == 0 and k.fluorescence_G[2].max() == 0
    assert k.elastic_rho[1] == 0


def test_thickness_axis_is_continuous(model):
    opt = make_optics(mua=0.1, mus=10.0)
    ds = np.geomspace(1e-4, 500.0, 400)
    k = model.kernels(opt, 405.0, np.full_like(ds, 0.7), np.full_like(ds, 0.7), ds)
    g = k.fluorescence_G[:, 200].astype(float)
    jumps = np.abs(np.diff(np.log(g)))
    assert jumps.max() < 0.12, jumps.max()                         # no discontinuity across nodes / thin / inf
    g_inf = model.kernels(opt, 405.0, [0.7], [0.7], [np.inf]).fluorescence_G[0, 200]
    assert g[-1] == pytest.approx(g_inf, rel=0.03)


def test_reabsorbed_fraction(model):
    lo = model.reabsorbed_fraction(make_optics(mua=0.0, mus=10.0), 520.0, 405.0, thickness_mm=1.0)
    mid = model.reabsorbed_fraction(make_optics(mua=0.5, mus=10.0), 520.0, 405.0, thickness_mm=1.0)
    hi = model.reabsorbed_fraction(make_optics(mua=5.0, mus=10.0), 520.0, 405.0, thickness_mm=1.0)
    print(f"\nreabsorbed fraction: mu_a 0 -> {lo:.4f}, 0.5 -> {mid:.4f}, 5 -> {hi:.4f}")
    assert lo == pytest.approx(0.0, abs=1e-6) and 0.05 < mid < hi < 1.0


# ============================================================================================
# Performance paths: em_index, shared model, reproducibility, warm-up
# ============================================================================================
def test_em_index_columns_match_full_result(model):
    sel = np.array([0, 5, 77, 300, 650])
    rng = np.random.default_rng(5)
    ci, co = rng.uniform(0.3, 1, 500), rng.uniform(0.3, 1, 500)
    th = np.where(rng.random(500) < 0.3, np.inf, rng.uniform(0.05, 5, 500))
    for opt in (make_optics(mua=0.1, mus=10.0, g=0.9, n=1.4), make_optics(mua=0.2, clear=True, n=1.4)):
        full = model.kernels(opt, 405.0, ci, co, th)
        part = model.kernels(opt, 405.0, ci, co, th, em_index=sel)
        assert part.fluorescence_G.shape == (500, sel.size)
        np.testing.assert_array_equal(part.fluorescence_G, full.fluorescence_G[:, sel])
        np.testing.assert_array_equal(part.elastic_rho, full.elastic_rho)
        if not opt.clear:                                   # clear: depth is weighted over the selected columns
            np.testing.assert_array_equal(part.mean_depth_mm, full.mean_depth_mm)
    with pytest.raises(IndexError):
        model.kernels(opt, 405.0, ci, co, th, em_index=[0, 651])


def test_kernel_equals_table_at_nodes(model):
    """With n_outside == n the Fresnel/refraction steps are identities, so on the table nodes
    G = exp(ln_j) / (4 pi n^2 mu_i mu_o) exactly (independent check of the fused interpolation kernel)."""
    opt = make_optics(mua=0.1, mus=10.0, g=0.9, n=1.4)
    gt = model.gtable(opt, 405.0, 1.4)
    ia, ib, k = 3, 5, 7
    k_ = model.kernels(opt, 405.0, [T.MU_NODES[ia]], [T.MU_NODES[ib]], [gt.d_nodes[k]], n_outside=1.4)
    w = np.arange(0, 651, 13)
    lin = (1 - gt.w_wgt[w]) * gt.ln_j[ia, ib, k, gt.w_idx[w]] + gt.w_wgt[w] * gt.ln_j[ia, ib, k, gt.w_idx[w] + 1]
    expect = np.exp(lin) / (4 * np.pi * 1.4 ** 2 * T.MU_NODES[ia] * T.MU_NODES[ib])
    np.testing.assert_allclose(k_.fluorescence_G[0, w], expect, rtol=2e-6)


def test_shared_model_and_table_reuse(tmp_path):
    m1 = T.get_transport_model(n_photons=24_000, seed=3, cache_dir=tmp_path)
    assert T.get_transport_model(n_photons=24_000, seed=3, cache_dir=tmp_path) is m1
    assert T.get_transport_model(n_photons=24_000, seed=4, cache_dir=tmp_path) is not m1
    opt = make_optics(mua=0.1, mus=10.0, g=0.9, n=1.4)
    g1 = m1.gtable(opt, 405.0, 1.0)
    assert m1.gtable(opt, 405.0, 1.0) is g1                 # J assembly cached across calls
    assert m1.baked(0.9, 1.4, 1.0) is m1.baked(0.9, 1.4, 1.0)


def test_bake_and_kernels_are_bit_identical_across_runs_and_thread_counts(tmp_path):
    import numba
    rng = np.random.default_rng(2)
    ci, co = rng.uniform(0.3, 1, 800), rng.uniform(0.3, 1, 800)
    th = np.where(rng.random(800) < 0.3, np.inf, rng.uniform(0.05, 5, 800))
    opt = make_optics(mua=0.1, mus=10.0, g=0.9, n=1.4)
    outs = []
    nmax = numba.get_num_threads()
    try:
        for i, nt in enumerate((nmax, 1, max(2, nmax // 2))):
            numba.set_num_threads(nt)
            bt = T.bake_tables(0.9, 1.4, 1.0, n_photons=8_000, seed=7, cache_dir=tmp_path / f"r{i}")
            m = T.MCTransportModel(n_photons=8_000, seed=7, cache_dir=tmp_path / f"r{i}")
            k = m.kernels(opt, 405.0, ci, co, th)
            outs.append((bt.arrays, k))
    finally:
        numba.set_num_threads(nmax)
    for arrays, k in outs[1:]:
        for name in arrays:
            assert np.array_equal(arrays[name], outs[0][0][name]), name
        assert np.array_equal(k.fluorescence_G, outs[0][1].fluorescence_G)
        assert np.array_equal(k.elastic_rho, outs[0][1].elastic_rho)
        assert np.array_equal(k.mean_depth_mm, outs[0][1].mean_depth_mm)


def test_warmup_jit_stage_and_status(tmp_path):
    r = T.warmup(["laser_line"], bake=False, gtables=False, cache_dir=tmp_path)
    assert r["errors"] == [] and r["jit_s"] >= 0
    plan = T.builtin_bake_plan(["laser_line"])
    assert plan[0]["preset"] == "laser_line" and isinstance(plan[0]["combos"], list)
    th = T.start_warmup(presets=["laser_line"], bake=False, gtables=False, cache_dir=tmp_path)
    th.join(120)
    st = T.warmup_status()
    assert st["state"] == "done" and st["fraction"] == 1.0 and st["result"]["errors"] == []
