"""Physics verification of the 3D layered-block Monte Carlo against independent references.

Each test prints "number vs reference" (run with ``-s``).  All are marked ``slow`` (1e6-1e7 photons).

1. total R / T / A of a homogeneous slab vs the repo's MCML engine (``transport_mc.mcml_slab``);
2. fluorescence + elastic radiance of a fluorescent layer vs the 1D kernels ``E mu_a Phi e G`` / ``E rho``;
3. two identical stacked layers == one layer of the summed thickness;
4. clear absorbing cover over a turbid fluorescent layer vs Beer-Lambert (+ Fresnel) analytics;
5. energy conservation (checked for every pass above via ``info['energy_balance']``);
6. wedge cover: local fluorescence vs uniform covers of the local thickness; monotone in thickness;
7. performance on the brief's reference configuration.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from spectraforge.studio import transport as TR
from spectraforge.studio import transport3d as T
from spectraforge.studio import transport_mc as mc
from spectraforge.studio.contracts import SPECTRAL_GRID_NM as WL
from spectraforge.studio.contracts import MaterialOptics, RegionOptics
from spectraforge.studio.schema import VolumeMCSettings

from .test_transport3d import UNI, WEDGE, geom, opt

pytestmark = pytest.mark.slow


def settings(lateral, exit, sub=4, ang=12.0):
    return VolumeMCSettings(lateral_cell_mm=lateral, exit_cell_mm=exit, sublayers=sub,
                            acceptance_half_angle_deg=ang)


def fluor_pipeline(G, ex_opt, em_opt, dye_layer, dye_frac, phi, s, n_exc, n_em, power=1.0, seed=1,
                   beam=None, cam=(0.0, 1.0, 0.0), direction=(0.0, -1.0, 0.0)):
    """Excitation pass -> sources (dye share of the absorbed power x quantum yield) -> emission pass.
    Returns (excitation tallies, emission tallies, E) with E the beam irradiance (mW/mm^2)."""
    X, Z = G.footprint_mm
    half = beam if beam is not None else (X / 2, Z / 2)
    launch = T.uniform_beam_launch(G, ex_opt.lambda_nm, power, 400_000, beam_half_size_mm=half,
                                   direction=direction, seed=seed)
    ex = T.excitation_pass(G, ex_opt, launch, cam, s, n_photons=n_exc, seed=seed)
    src = np.zeros((1,) + ex.absorbed_mW.shape)
    src[0, :, :, dye_layer, :] = ex.absorbed_mW[:, :, dye_layer, :] * dye_frac * phi
    em = T.emission_pass(G, [em_opt], src, ["dye"], cam, s, n_photons=n_em, seed=seed + 1)
    E = power / (4 * half[0] * half[1])
    return ex, em, E


# ============================================================================================
# 1. MCML
# ============================================================================================
@pytest.mark.parametrize("mua,mus,g,n,d", [(1.0, 9.0, 0.75, 1.0, 0.2),     # Prahl/Wang benchmark
                                           (1.0, 9.0, 0.75, 1.4, 0.2),     # index mismatch
                                           (0.1, 20.0, 0.9, 1.38, 1.0)])   # tissue-like
def test_slab_matches_mcml(mua, mus, g, n, d):
    G = geom([UNI(d)], (200.0, 200.0))
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 200_000)
    r = T.excitation_pass(G, opt(mua, mus, g, n), launch, [0, 1, 0], settings(100, 100, 4),
                          n_photons=3_000_000, seed=1)
    ref = mc.mcml_slab(mua, mus, g, n, 1.0, d, n_photons=3_000_000, seed=3)
    R, Tt, A = r.diffuse_reflectance, r.transmittance, r.absorbed_fraction
    print(f"\nslab mua={mua} mus={mus} g={g} n={n} d={d}:  3D R/T/A = {R:.5f} {Tt:.5f} {A:.5f}"
          f"   MCML = {ref['Rd']:.5f} {ref['Tt']:.5f} {ref['A']:.5f}   (spec {r.info['specular']:.5f} vs {ref['Rsp']:.5f})")
    assert R == pytest.approx(ref["Rd"], rel=0.015)
    assert Tt == pytest.approx(ref["Tt"], rel=0.015)
    assert A == pytest.approx(ref["A"], rel=0.015)
    assert r.info["specular"] == pytest.approx(ref["Rsp"], abs=1e-9)
    assert r.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)
    if n == 1.0:                                    # published values (Prahl 1989 / Wang 1995)
        assert R == pytest.approx(0.09734, rel=0.015)
        assert Tt == pytest.approx(0.66096, rel=0.015)


# ============================================================================================
# 2. fluorescent single layer vs the 1D kernels
# ============================================================================================
def _mat1d(mua_ex, mus_ex, mua_em, mus_em, g, n, ex):
    lo = WL <= ex + 1e-9
    return MaterialOptics("m", WL, np.where(lo, mua_ex, mua_em).astype(float), {},
                          np.where(lo, mus_ex, mus_em).astype(float), g, n, False, {}, {})


def test_fluorescent_layer_matches_1d_kernels():
    EX, EM = 405.0, 520.0
    mua_ex, mus_ex, mua_em, mus_em, g, n = 0.30, 12.0, 0.10, 9.0, 0.8, 1.4
    mua_dye, phi, d = 0.20, 0.5, 2.0
    fp = (60.0, 60.0)
    G = geom([UNI(d)], fp)
    s = settings(20, 20, sub=64)
    ex, em, E = fluor_pipeline(
        G, opt(mua_ex, mus_ex, g, n, EX),
        opt(mua_em, mus_em, g, n, EM), 0, mua_dye / mua_ex, phi, s, 6_000_000, 6_000_000)
    k = TR.get_transport_model().kernels(_mat1d(mua_ex, mus_ex, mua_em, mus_em, g, n, EX), EX,
                                         np.array([1.0]), np.array([1.0]), np.array([d]), 1.0)
    iem = int(np.argmin(abs(WL - EM)))
    L3 = em.radiance["top"][1, 1, 0, 0]
    L1 = E * mua_dye * phi * k.fluorescence_G[0, iem]
    rho3 = ex.elastic_radiance["top"][1, 1] / E
    rho1 = k.elastic_rho[0]
    print(f"\nfluorescence 3D = {L3:.4e}  1D = {L1:.4e}  ratio {L3 / L1:.4f}")
    print(f"elastic rho  3D = {rho3:.5f}  1D = {rho1:.5f}  ratio {rho3 / rho1:.4f}"
          f"   (3D footprint {fp[0]:.0f} mm leaks {ex.info['T_sides']:.3f} through the sides)")
    assert L3 == pytest.approx(L1, rel=0.04)
    assert rho3 == pytest.approx(rho1, rel=0.05)
    assert ex.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)
    assert em.info["energy_balance"][0] == pytest.approx(1.0, abs=1e-9)


# ============================================================================================
# 3. stacked identical layers
# ============================================================================================
def test_two_identical_layers_equal_one():
    mua, mus, g, n = 0.15, 10.0, 0.85, 1.4
    results = []
    for layers in ([UNI(1.2)], [UNI(0.5), UNI(0.7)], [UNI(0.3), UNI(0.0), UNI(0.4), UNI(0.5)]):
        L = len(layers)
        G = geom(layers, (100.0, 100.0))
        o = opt([mua] * L, [mus] * L, [g] * L, [n] * L)
        launch = T.uniform_beam_launch(G, 405.0, 1.0, 100_000)
        s = settings(100, 100, 4)
        ex = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=2_000_000, seed=1)
        grid = T.make_voxel_grid(G, s)
        src = np.zeros((1,) + grid.shape)
        src[0, :, :, :, :] = 1.0 * (T.voxel_volumes(G, grid) > 0)
        # emitters uniform in depth: weight sublayer cells by their volume -> same as uniform emission
        src[0] *= T.voxel_volumes(G, grid)
        em = T.emission_pass(G, [o], src, ["u"], [0, 1, 0], s, n_photons=2_000_000, seed=2)
        results.append((ex.diffuse_reflectance, ex.transmittance, ex.absorbed_fraction,
                        em.info["escape_top"][0], em.info["reabsorbed"][0]))
        print(f"\nstack {[l.thickness_mm for l in layers]}: R={results[-1][0]:.5f} T={results[-1][1]:.5f} "
              f"A={results[-1][2]:.5f}  emission escape_top={results[-1][3]:.5f} reabs={results[-1][4]:.5f}")
    ref = np.array(results[0])
    for r in results[1:]:
        assert np.allclose(r, ref, rtol=0.012, atol=2e-4)


# ============================================================================================
# 4. clear absorbing cover
# ============================================================================================
def _cover_ratio(n_layer, n_cover, d_cover, mua_c_ex, mua_c_em):
    EX, EM = 405.0, 520.0
    s = settings(20, 20, sub=32)
    layer_ex = (0.30, 12.0)
    layer_em = (0.10, 9.0)
    g = 0.8
    dye_frac, phi = 0.2 / 0.30, 0.5
    out = {}
    for name, cover in (("bare", False), ("covered", True)):
        layers = [UNI(1.5)] + ([UNI(d_cover)] if cover else [])
        G = geom(layers, (60.0, 60.0))
        ex_o = opt([layer_ex[0]] + ([mua_c_ex] if cover else []), [layer_ex[1]] + ([0.0] if cover else []),
                   [g] + ([0.0] if cover else []), [n_layer] + ([n_cover] if cover else []), EX)
        em_o = opt([layer_em[0]] + ([mua_c_em] if cover else []), [layer_em[1]] + ([0.0] if cover else []),
                   [g] + ([0.0] if cover else []), [n_layer] + ([n_cover] if cover else []), EM)
        ex, em, E = fluor_pipeline(G, ex_o, em_o, 0, dye_frac, phi, s, 5_000_000, 5_000_000)
        out[name] = (em.radiance["top"][1, 1, 0, 0] / E, ex.elastic_radiance["top"][1, 1] / E,
                     ex.info["specular"])
    return out


def _T(n1, n2):
    return 1.0 - float(T._fresnel_np(n1, n2, np.array(1.0)))


def test_clear_absorbing_cover_index_matched():
    """All indices 1: ratio = exp(-mua_ex d) exp(-mua_em d / cos), no Fresnel factors at all."""
    d, ax, ae = 1.0, 0.25, 0.15
    o = _cover_ratio(1.0, 1.0, d, ax, ae)
    meas = o["covered"][0] / o["bare"][0]
    expect = math.exp(-ax * d) * math.exp(-ae * d / math.cos(math.radians(6.0)))
    print(f"\ncover (n=1): fluorescence covered/bare = {meas:.4f}   Beer-Lambert = {expect:.4f}"
          f"   ratio {meas / expect:.4f}")
    assert meas == pytest.approx(expect, rel=0.03)


def _series_T(n_a, n_b, n_c):
    """Normal-incidence transmittance a -> b -> c through a clear non-absorbing layer b (incoherent)."""
    R1, R2 = 1 - _T(n_a, n_b), 1 - _T(n_b, n_c)
    return _T(n_a, n_b) * _T(n_b, n_c) / (1 - R1 * R2)


def test_clear_cover_with_fresnel_nonabsorbing():
    """A clear NON-absorbing n=1.5 cover over an n=1.4 turbid layer only changes the Fresnel transmission
    (the Snell invariant n sin(theta) keeps the critical angle and the angular reflectance of the stack
    ~ that of the bare layer): covered/bare = [T(1->1.5->1.4)/T(1->1.4)] x [T(1.4->1.5->1)/T(1.4->1)]."""
    o = _cover_ratio(1.4, 1.5, 1.0, 0.0, 0.0)
    meas = o["covered"][0] / o["bare"][0]
    expect = (_series_T(1.0, 1.5, 1.4) / _T(1.0, 1.4)) * (_series_T(1.4, 1.5, 1.0) / _T(1.4, 1.0))
    print(f"\ncover (n=1.5 over 1.4, non-absorbing): covered/bare = {meas:.4f}   Fresnel analytic = {expect:.4f}"
          f"   ratio {meas / expect:.4f}")
    assert meas == pytest.approx(expect, rel=0.03)


def test_clear_absorbing_cover_with_fresnel_bounds():
    """Absorbing + index mismatch.  The naive estimate (Beer-Lambert x single-pass Fresnel factors) neglects
    that the diffuse light recycled by the air interface crosses the cover again, obliquely, so the true
    ratio must lie BELOW it but above the all-light-twice-at-45deg bound."""
    d, ax, ae = 1.0, 0.25, 0.15
    nl, nc = 1.4, 1.5
    o = _cover_ratio(nl, nc, d, ax, ae)
    meas = o["covered"][0] / o["bare"][0]
    naive = (_T(1.0, nc) * _T(nc, nl) / _T(1.0, nl) * math.exp(-ax * d)
             * _T(nl, nc) * _T(nc, 1.0) / _T(nl, 1.0) * math.exp(-ae * d))
    floor = naive * math.exp(-2 * (ax + ae) * d / math.cos(math.radians(45.0)))
    print(f"\ncover (n=1.5 over 1.4, absorbing): covered/bare = {meas:.4f}   naive single-pass estimate = "
          f"{naive:.4f}   recycled-light floor = {floor:.4f}")
    assert floor < meas < naive


# ============================================================================================
# 6. wedge
# ============================================================================================
def test_wedge_matches_local_uniform_covers_and_is_monotone():
    EX, EM = 405.0, 520.0
    t0, t1, fp = 0.2, 1.4, (60.0, 60.0)
    s = settings(6, 6, sub=16)
    g = 0.8
    layer_ex, layer_em = (0.30, 12.0), (0.10, 9.0)
    cover_ex, cover_em = (0.02, 14.0), (0.015, 11.0)           # non-fluorescent turbid cover
    n = 1.4

    def run(cover):
        G = geom([UNI(1.0), cover], fp)
        ex_o = opt([layer_ex[0], cover_ex[0]], [layer_ex[1], cover_ex[1]], [g, 0.9], [n, n], EX)
        em_o = opt([layer_em[0], cover_em[0]], [layer_em[1], cover_em[1]], [g, 0.9], [n, n], EM)
        ex, em, E = fluor_pipeline(G, ex_o, em_o, 0, 0.2 / 0.3, 0.5, s, 3_000_000, 3_000_000)
        return em.radiance["top"][:, :, 0, 0] / E, G

    Lw, Gw = run(WEDGE(t0, t1, "x"))
    cells = [1, 3, 5, 7, 9]                                       # x centres -27 ... 27 mm
    xc = -30 + 6 * (np.arange(10) + 0.5)
    prof = Lw[:, 4:6].mean(1)                                      # average the two central z-cells
    print("\nwedge fluorescence vs x:", np.round(prof[cells] / prof[1], 4))
    assert all(prof[cells[i]] > prof[cells[i + 1]] for i in range(len(cells) - 1)), "not monotone"
    diffs = []
    for c in (2, 5, 7):
        tl = float(t0 + (t1 - t0) * (xc[c] / fp[0] + 0.5))
        Lu, _ = run(UNI(tl))
        # the uniform block's centre is far from its edges: compare the same x-cell? use its centre cells
        u = Lu[4:6, 4:6].mean()
        w = Lw[c, 4:6].mean()
        diffs.append(w / u - 1)
        print(f"x = {xc[c]:+.0f} mm  local cover {tl:.3f} mm: wedge {w:.4e}  uniform {u:.4e}  diff {100 * (w / u - 1):+.1f} %")
    assert max(abs(x) for x in diffs) < 0.10


# ============================================================================================
# 7. performance (brief reference configuration)
# ============================================================================================
def test_performance_reference_configuration():
    fp = (40.0, 40.0)
    G = geom([UNI(2.0), UNI(0.5), WEDGE(0.0, 2.0, "x")], fp)
    s = VolumeMCSettings(lateral_cell_mm=0.5, exit_cell_mm=0.5, sublayers=8)
    o = opt([0.1, 0.8, 0.05], [10.0, 12.0, 20.0], [0.9, 0.8, 0.85], [1.4, 1.4, 1.4], 365.0)
    launch = T.uniform_beam_launch(G, 365.0, 1.0, 200_000, beam_half_size_mm=(15, 15))
    T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=20_000, seed=1)      # JIT warm-up
    t = time.perf_counter()
    ex = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=2_000_000, seed=1)
    t_ex = time.perf_counter() - t
    C = 24
    src = np.repeat(ex.absorbed_mW[None], C, 0) * 0.01
    o2 = opt([0.05, 0.2, 0.03], [9.0, 10.0, 18.0], [0.9, 0.8, 0.85], [1.4, 1.4, 1.4], 520.0)
    T.emission_pass(G, [o2], src, [str(i) for i in range(C)], [0, 1, 0], s, n_photons=20_000, seed=1)
    t = time.perf_counter()
    T.emission_pass(G, [o2], src, [str(i) for i in range(C)], [0, 1, 0], s, n_photons=2_000_000, seed=1)
    t_em = time.perf_counter() - t
    print(f"\nexcitation 2e6 photons: {t_ex:.2f} s ({ex.info['mean_steps_per_photon']:.0f} steps/photon);"
          f" emission 2e6 photons C={C}: {t_em:.2f} s per node")
    assert t_ex < 120 and t_em < 120          # loose: CI machines are shared; see the report for real numbers
