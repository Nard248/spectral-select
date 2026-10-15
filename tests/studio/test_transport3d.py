"""Fast tests of the 3D layered-block Monte Carlo (geometry, bookkeeping, determinism, robustness).

Slow physics references (MCML, 1D kernels, wedge) live in ``test_transport3d_validation.py``.
"""
from __future__ import annotations

import math

import numba
import numpy as np
import pytest

from spectraforge.studio import transport3d as T
from spectraforge.studio.contracts import BlockGeometry, RegionOptics
from spectraforge.studio.schema import PhysicsToggles, ThicknessProfile, VolumeMCSettings

UNI = lambda t: ThicknessProfile(thickness_mm=t)
WEDGE = lambda t0, t1, axis="x": ThicknessProfile(kind="linear", thickness_mm=t0, end_thickness_mm=t1,
                                                  axis=axis)


def geom(layers, fp=(20.0, 20.0)) -> BlockGeometry:
    return BlockGeometry(0, tuple(map(float, fp)), [f"l{i}" for i in range(len(layers))],
                         list(range(len(layers))), T.planes_from_layers(fp, layers), np.eye(4))


def opt(mua, mus, g, n, lam=500.0) -> RegionOptics:
    a = lambda v: np.atleast_1d(np.asarray(v, float))
    return RegionOptics(lam, a(mua), a(mus), a(g), a(n))


def cfg(lateral=5.0, exit=5.0, sub=4, ang=12.0, **kw) -> VolumeMCSettings:
    return VolumeMCSettings(lateral_cell_mm=lateral, exit_cell_mm=exit, sublayers=sub,
                            acceptance_half_angle_deg=ang, **kw)


# ============================================================================================
# Geometry
# ============================================================================================
def test_planes_match_thickness_profiles():
    fp = (30.0, 24.0)
    layers = [UNI(1.0), WEDGE(0.2, 1.4, "x"), WEDGE(2.0, 0.0, "z"), UNI(0.5)]
    G = geom(layers, fp)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, z = rng.uniform(-fp[0] / 2, fp[0] / 2), rng.uniform(-fp[1] / 2, fp[1] / 2)
        acc = 0.0
        for i, prof in enumerate(layers):
            acc += prof.at(x, z, fp)
            assert G.top_height(i, x, z) == pytest.approx(acc, abs=1e-12)
            assert float(T.layer_thickness(G, i, x, z)) == pytest.approx(prof.at(x, z, fp), abs=1e-12)


def test_build_block_geometry_from_preset():
    from spectraforge.studio.presets import preset

    exp = preset("layered_wedge_phantom")
    G = T.build_block_geometry(exp, 0)
    assert G.layer_names == ["substrate", "fluorescent layer", "cover"]
    ids = [m.id for m in exp.materials]
    assert [ids[i] for i in G.layer_material] == ["tissue_substrate", "fluorescein_agarose_layer",
                                                  "intralipid_cover"]
    assert G.top_height(2, -20.0, 0.0) == pytest.approx(4.0 + 0.5 + 2.0)
    assert G.top_height(2, +20.0, 0.0) == pytest.approx(4.5)
    with pytest.raises(TypeError):
        T.build_block_geometry(preset("generic_macro_bench"), 0)


def test_voxel_volumes_are_exact():
    G = geom([UNI(1.0), WEDGE(0.0, 2.0, "x"), UNI(0.5)], (20.0, 12.0))
    grid = T.make_voxel_grid(G, cfg(lateral=1.0, sub=5))
    vol = T.voxel_volumes(G, grid)
    assert vol.shape == grid.shape == (20, 12, 3, 5)
    expect = [1.0 * 240, 1.0 * 240, 0.5 * 240]          # mean thickness x footprint
    for l in range(3):
        assert vol[:, :, l, :].sum() == pytest.approx(expect[l], rel=1e-12)


def test_face_grids_cover_block():
    G = geom([UNI(1.0), WEDGE(0.0, 3.0, "x")], (20.0, 10.0))
    f = T.make_face_grids(G, cfg(exit=2.0))
    assert set(f) == {"top", "+x", "-x", "+z", "-z"}
    assert f["top"].u_edges_mm[0] == -10 and f["top"].u_edges_mm[-1] == 10
    assert f["+x"].v_edges_mm[-1] == pytest.approx(1.0 + 3.0 / 1.0 * 1.0 + 0.0, abs=3.0)  # >= max height
    assert f["+x"].v_edges_mm[-1] >= 1.0 + 3.0 - 1e-9


# ============================================================================================
# Conservation, determinism, robustness
# ============================================================================================
CONFIGS = {
    "matched": ([UNI(1.0)], opt(0.1, 8.0, 0.8, 1.0)),
    "mismatch_two_layers": ([UNI(0.6), UNI(0.8)], opt([0.05, 0.2], [6.0, 12.0], [0.7, 0.9], [1.33, 1.5])),
    "wedge_cover": ([UNI(1.0), UNI(0.5), WEDGE(0.0, 2.0)],
                    opt([0.05, 0.3, 0.02], [8.0, 10.0, 15.0], [0.8, 0.8, 0.9], [1.38, 1.38, 1.41])),
    "clear_absorbing_cover": ([UNI(1.0), UNI(0.8)], opt([0.1, 0.4], [10.0, 0.0], [0.8, 0.0], [1.4, 1.5])),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_energy_conservation_excitation(name):
    layers, o = CONFIGS[name]
    G = geom(layers)
    launch = T.uniform_beam_launch(G, 405.0, 2.0, 20000, beam_half_size_mm=(8, 8),
                                   direction=(0.2, -1.0, 0.1))
    r = T.excitation_pass(G, o, launch, [0, 1, 0], cfg(), n_photons=150_000, seed=3)
    i = r.info
    total = i["R_top"] + i["T_bottom"] + i["T_sides"] + r.absorbed_fraction + i["specular"] + i["lost_steps"]
    print(f"\n[{name}] R={i['R_top']:.4f} T={r.transmittance:.4f} A={r.absorbed_fraction:.4f} "
          f"spec={i['specular']:.4f}  sum-roulette={i['energy_balance']:.12f}")
    assert i["energy_balance"] == pytest.approx(1.0, abs=1e-9)
    assert total - i["roulette_net"] == pytest.approx(1.0, abs=1e-9)
    assert i["lost_steps"] == 0.0
    # voxel absorption tally equals the scalar absorbed power
    assert r.absorbed_mW.sum() / 2.0 == pytest.approx(r.absorbed_fraction, rel=1e-9)
    assert np.isfinite(r.fluence_mW_mm2).all() and (r.fluence_mW_mm2 >= 0).all()
    for f in T.FACES:
        assert np.isfinite(r.elastic_radiance[f]).all()


def test_fluence_consistent_with_absorption():
    """Absorbed power of a voxel = mu_a * fluence * volume (two independent estimators agree)."""
    layers = [UNI(1.0), UNI(1.0)]
    G = geom(layers, (12.0, 12.0))
    o = opt([0.5, 0.15], [8.0, 6.0], [0.8, 0.85], [1.4, 1.4])
    launch = T.uniform_beam_launch(G, 405.0, 1.0, 20000, beam_half_size_mm=(4, 4))
    s = cfg(lateral=12.0, sub=4)
    r = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=600_000, seed=2)
    vol = T.voxel_volumes(G, r.grid)
    for l in range(2):
        a_track = (o.mua_per_mm[l] * r.fluence_mW_mm2[:, :, l, :] * vol[:, :, l, :]).sum()
        a_coll = r.absorbed_mW[:, :, l, :].sum()
        print(f"\nlayer {l}: mu_a*Phi*V = {a_track:.5f}  collision tally = {a_coll:.5f}")
        assert a_track == pytest.approx(a_coll, rel=0.02)


def test_energy_conservation_emission():
    G = geom([UNI(1.0), UNI(0.7)])
    o = opt([0.05, 0.1], [8.0, 12.0], [0.8, 0.85], [1.4, 1.38], lam=520.0)
    s = cfg()
    grid = T.make_voxel_grid(G, s)
    src = np.zeros((2,) + grid.shape)
    src[0, :, :, 0, :] = 1.0
    src[1, :, :, 1, :] = 2.0
    r = T.emission_pass(G, [o], src, ["a", "b"], [0, 1, 0], s, n_photons=100_000, seed=4)
    assert r.info["energy_balance"][0] == pytest.approx(1.0, abs=1e-9)
    esc = r.info["escape_top"][0] + r.info["escape_bottom"][0] + r.info["escape_sides"][0] + r.info["reabsorbed"][0]
    assert esc == pytest.approx(1.0, abs=2e-3)            # differs from 1 only by the roulette gain/loss
    # origin diagnostics
    for f in ("top",):
        frac = r.origin_layer_fraction[f]
        ok = np.isfinite(frac).all(-1)
        assert np.allclose(frac[ok].sum(-1), 1.0, atol=1e-9)
        assert (r.origin_depth_mm[f][ok] > 0).all()


def test_emission_channels_are_exactly_linear():
    G = geom([UNI(1.0), UNI(0.7)])
    o = opt([0.05, 0.1], [8.0, 12.0], [0.8, 0.85], [1.4, 1.38], lam=520.0)
    s = cfg()
    grid = T.make_voxel_grid(G, s)
    src = np.zeros((3,) + grid.shape)
    src[0, :, :, 1, :] = 1.0
    src[1] = 3.0 * src[0]
    src[2, :, :, 0, :] = 0.5
    r = T.emission_pass(G, [o, opt([0.05, 0.1], [7.0, 11.0], [0.8, 0.85], [1.4, 1.38], lam=540.0)],
                        src, ["a", "b", "c"], [0, 1, 0], s, n_photons=60_000, seed=1)
    R = r.radiance["top"]
    assert R.shape == (4, 4, 2, 3)
    assert np.allclose(R[..., 1], 3.0 * R[..., 0], rtol=1e-12, atol=0)
    assert (R[..., 2].sum() > 0) and (R >= 0).all()
    assert list(r.em_nodes_nm) == [520.0, 540.0]


def test_determinism_and_thread_invariance():
    layers, o = CONFIGS["wedge_cover"]
    G = geom(layers)
    s = cfg(lateral=2.0, exit=2.0, sub=3)
    launch = T.uniform_beam_launch(G, 405.0, 1.0, 5000, beam_half_size_mm=(6, 6), seed=1)
    grid = T.make_voxel_grid(G, s)
    src = np.zeros((2,) + grid.shape)
    src[0, :, :, 1, :] = 1.0
    src[1, :, :, 2, :] = 1.0
    nodes = [opt([0.05, 0.1, 0.03], [8, 10, 15], [0.8, 0.8, 0.9], [1.38, 1.38, 1.41], lam=500.0 + 20 * k)
             for k in range(2)]
    maxthreads = numba.get_num_threads()

    def run():
        e = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=40_000, seed=11)
        m = T.emission_pass(G, nodes, src, ["a", "b"], [0, 1, 0], s, n_photons=40_000, seed=12)
        return e, m

    try:
        numba.set_num_threads(1)
        e1, m1 = run()
        numba.set_num_threads(min(4, maxthreads))
        e2, m2 = run()
    finally:
        numba.set_num_threads(maxthreads)
    e3, m3 = run()
    for e in (e2, e3):
        assert np.array_equal(e1.absorbed_mW, e.absorbed_mW)
        assert np.array_equal(e1.fluence_mW_mm2, e.fluence_mW_mm2)
        for f in T.FACES:
            assert np.array_equal(e1.elastic_radiance[f], e.elastic_radiance[f])
    for m in (m2, m3):
        for f in T.FACES:
            assert np.array_equal(m1.radiance[f], m.radiance[f])
    e4 = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=40_000, seed=99)
    assert not np.array_equal(e1.absorbed_mW, e4.absorbed_mW)       # the seed matters


@pytest.mark.parametrize("layers", [
    [UNI(1.0), WEDGE(0.0, 2.0)],                                   # cover -> 0 exactly at the -x edge
    [UNI(1.0), UNI(0.0), UNI(0.5)],                                # a zero-thickness layer everywhere
    [WEDGE(0.0, 1.0), WEDGE(1.0, 0.0), UNI(0.3)],                  # two layers that vanish at opposite edges
])
def test_zero_thickness_regions_are_robust(layers):
    G = geom(layers)
    L = len(layers)
    o = opt([0.1] * L, [8.0] * L, [0.8] * L, [1.4, 1.5, 1.4][:L])
    s = cfg(lateral=2.0, exit=2.0, sub=3)
    launch = T.uniform_beam_launch(G, 405.0, 1.0, 5000, direction=(0.1, -1.0, 0.05))
    r = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=60_000, seed=5)
    assert np.isfinite(r.absorbed_mW).all() and np.isfinite(r.fluence_mW_mm2).all()
    assert r.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)
    assert r.info["lost_steps"] == 0.0
    # nothing is absorbed in a layer that has no volume
    vol = T.voxel_volumes(G, r.grid)
    assert r.absorbed_mW[vol <= 0].sum() == 0.0
    sources = np.where(vol > 0, 1.0, 0.0)[None]
    e = T.emission_pass(G, [o], sources, ["a"], [0, 1, 0], s, n_photons=40_000, seed=6)
    for f in T.FACES:
        assert np.isfinite(e.radiance[f]).all()
    assert e.info["energy_balance"][0] == pytest.approx(1.0, abs=1e-9)


def test_wedge_mirror_symmetry():
    """Reversing the wedge direction mirrors the result (checks the signs of the plane normals)."""
    o = opt([0.1, 0.2], [8.0, 10.0], [0.8, 0.8], [1.4, 1.5])
    s = cfg(lateral=5.0, exit=5.0, sub=4)
    out = []
    for cover in (WEDGE(0.2, 2.0, "x"), WEDGE(2.0, 0.2, "x")):
        G = geom([UNI(0.8), cover], (20.0, 20.0))
        launch = T.uniform_beam_launch(G, 405.0, 1.0, 20000, direction=(0.0, -1.0, 0.0))
        out.append(T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=600_000, seed=1))
    a, b = out
    pa, pb = a.absorbed_mW.sum(axis=(1, 3)), b.absorbed_mW.sum(axis=(1, 3))[::-1]      # (nx, L), x mirrored
    print("\nmirror absorbed per x column (cover):", np.round(pa[:, 1], 4), np.round(pb[:, 1], 4))
    assert a.diffuse_reflectance == pytest.approx(b.diffuse_reflectance, rel=0.02)
    assert a.absorbed_fraction == pytest.approx(b.absorbed_fraction, rel=0.02)
    assert np.allclose(pa, pb, rtol=0.06)
    assert pa[0, 1] < 0.6 * pa[-1, 1]          # the wedge really is asymmetric (thin side absorbs less)


def test_side_face_radiance_and_visibility():
    """A camera looking along +x sees the +x face only; faces seen edge-on are flagged not visible."""
    G = geom([UNI(2.0)], (12.0, 12.0))
    o = opt(0.05, 8.0, 0.8, 1.4)
    s = cfg(lateral=12.0, exit=2.0, sub=2)
    launch = T.uniform_beam_launch(G, 405.0, 1.0, 5000)
    r = T.excitation_pass(G, o, launch, [1, 0, 0], s, n_photons=200_000, seed=1)
    assert r.info["visible_faces"] == {"top": False, "+x": True, "-x": False, "+z": False, "-z": False}
    assert r.elastic_radiance["+x"].sum() > 0
    assert r.elastic_radiance["top"].sum() == 0 and r.elastic_radiance["-x"].sum() == 0
    # block is 2 mm tall: cells above the top surface (cell height 2 mm -> exactly covered) are finite
    assert np.isfinite(r.elastic_radiance["+x"]).all()


def test_toggles_and_validation():
    G = geom([UNI(1.0)])
    o = opt(0.1, 8.0, 0.8, 1.4)
    launch = T.uniform_beam_launch(G, 405.0, 1.0, 2000)
    r = T.excitation_pass(G, o, launch, [0, 1, 0], cfg(), n_photons=20_000, seed=1,
                          toggles=PhysicsToggles(fresnel=False))
    assert r.info["specular"] == 0.0
    r2 = T.excitation_pass(G, o, launch, [0, 1, 0], cfg(), n_photons=20_000, seed=1,
                           toggles=PhysicsToggles(scattering=False))
    assert r2.diffuse_reflectance < 0.3 * r.diffuse_reflectance       # no scattering -> (almost) no remission
    with pytest.raises(ValueError):
        T.excitation_pass(G, opt([0.1, 0.1], [1, 1], [0, 0], [1.4, 1.4]), launch, [0, 1, 0], cfg(),
                          n_photons=1000, seed=1)
    with pytest.raises(ValueError):
        T.excitation_pass(G, o, launch, [0, 0, 0], cfg(), n_photons=1000, seed=1)
    with pytest.raises(ValueError):
        grid = T.make_voxel_grid(G, cfg())
        T.emission_pass(G, [o], np.ones((1, 2, 2, 1, 4)), ["a"], [0, 1, 0], cfg(), n_photons=1000, seed=1)
    with pytest.raises(ValueError):
        T.emission_pass(G, [o], -np.ones((1,) + grid.shape), ["a"], [0, 1, 0], cfg(), n_photons=1000, seed=1)


def test_clear_prism_refraction_angle():
    """A normal beam enters a clear n=1.5 wedge through its flat bottom and leaves through the tilted top
    plane; Snell's law with the PLANE normal predicts the exit direction (tests the tilted-interface
    Fresnel/refraction code)."""
    n_glass, X = 1.5, 20.0
    b = (2.0 - 1.0) / X                                   # slope of the top plane: thickness 1 -> 2 mm
    alpha = math.atan(b)
    theta_t = math.asin(n_glass * math.sin(alpha))
    delta = theta_t - alpha                               # exit direction from vertical, toward +x
    G = geom([WEDGE(1.0, 2.0, "x")], (X, 10.0))
    o = opt(0.0, 0.0, 0.0, n_glass)
    launch = T.BlockLaunch(405.0, np.array([[x, 0.0, z] for x in np.linspace(-8, 8, 9) for z in (-2.0, 0.0, 2.0)]),
                           np.tile([0.0, 1.0, 0.0], (27, 1)), np.full(27, 1.0 / 27))
    s = cfg(lateral=20.0, exit=20.0, sub=1, ang=1.0)
    hit = T.excitation_pass(G, o, launch, [math.sin(delta), math.cos(delta), 0.0], s, n_photons=40_000, seed=1)
    miss_vertical = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=40_000, seed=1)
    miss_mirror = T.excitation_pass(G, o, launch, [-math.sin(delta), math.cos(delta), 0.0], s,
                                    n_photons=40_000, seed=1)
    print(f"\nprism: predicted exit {math.degrees(delta):.3f} deg from vertical;"
          f" radiance sum  at +delta {hit.elastic_radiance['top'].sum():.3e}"
          f"  vertical {miss_vertical.elastic_radiance['top'].sum():.3e}"
          f"  -delta {miss_mirror.elastic_radiance['top'].sum():.3e}")
    assert hit.elastic_radiance["top"].sum() > 0
    assert miss_vertical.elastic_radiance["top"].sum() == 0.0      # 1.43 deg off axis, cone is 1 deg
    assert miss_mirror.elastic_radiance["top"].sum() == 0.0
    # throughput into the cone: (1-Rin)(1-Rout) of the beam (+ internal bounces), clear: A = 0
    assert hit.absorbed_fraction == 0.0


# ============================================================================================
# v1.1 QA: bottom face = Fresnel interface to air (consistent with the 1-D slab engine)
# ============================================================================================
@pytest.mark.parametrize("n,mua,d", [(1.5, 0.5, 1.0), (1.33, 0.2, 2.0), (2.0, 1.0, 0.5)])
def test_bottom_face_is_fresnel_interface_to_air(n, mua, d):
    """Clear (mu_s = 0) absorbing slab, wide normal beam: the light is bounced between the top and the bottom
    interface.  With a Fresnel-to-air bottom (transmitted part lost, reflected part kept) the diffuse
    reflectance and the bottom transmittance are the closed-form geometric series::

        T_bot = (1-R0)^2 e^-tau / (1 - R0^2 e^-2tau),   R_top = (1-R0)^2 R0 e^-2tau / (1 - R0^2 e^-2tau)

    with tau = mu_a d, R0 = ((n-1)/(n+1))^2 (a perfect sink would give T_bot = (1-R0) e^-tau, R_top = 0)."""
    G = geom([UNI(d)], (400.0, 400.0))
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 5000, beam_half_size_mm=(50.0, 50.0))
    r = T.excitation_pass(G, opt(mua, 0.0, 0.0, n), launch, [0, 1, 0], cfg(lateral=200, exit=200, sub=2),
                          n_photons=100_000, seed=2)
    R0 = ((n - 1) / (n + 1)) ** 2
    e = math.exp(-mua * d)
    den = 1 - R0 ** 2 * e ** 2
    t_ref, r_ref = (1 - R0) ** 2 * e / den, (1 - R0) ** 2 * R0 * e ** 2 / den
    print(f"\nn={n} mua={mua} d={d}: T_bot {r.info['T_bottom']:.6f} (ref {t_ref:.6f})  "
          f"R_top {r.info['R_top']:.6f} (ref {r_ref:.6f})  sink would give {(1 - R0) * e:.6f} / 0")
    assert r.info["T_bottom"] == pytest.approx(t_ref, rel=2e-3)
    assert r.info["R_top"] == pytest.approx(r_ref, rel=5e-3)
    assert r.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)
    # the same slab through the repo's 1-D (MCML) engine: n_outside on both faces
    from spectraforge.studio import transport_mc as mc
    ref = mc.mcml_slab(mua, 0.0, 0.0, n, 1.0, d, n_photons=100_000, seed=1)
    assert r.info["T_bottom"] == pytest.approx(ref["Tt"], rel=5e-3)


def test_no_fresnel_toggle_removes_bottom_reflection():
    G = geom([UNI(1.0)], (400.0, 400.0))
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 5000, beam_half_size_mm=(50.0, 50.0))
    r = T.excitation_pass(G, opt(0.5, 0.0, 0.0, 1.5), launch, [0, 1, 0], cfg(lateral=200, exit=200, sub=2),
                          n_photons=20_000, seed=2, toggles=PhysicsToggles(fresnel=False))
    assert r.info["T_bottom"] == pytest.approx(math.exp(-0.5), rel=1e-9)
    assert r.info["R_top"] == 0.0


# ============================================================================================
# v1.1 QA: peel-off (next-event) estimator
# ============================================================================================
def _peel_call(G, o, x, y, z, i, cam, fres=True):
    pl, pn = T._plane_arrays(G)
    L = len(G.layer_names)
    lay = np.ascontiguousarray(np.stack([o.mua_per_mm, o.mus_per_mm, o.g, o.n]).astype(float))
    from spectraforge.studio import transport3d_mc as k
    cam = np.asarray(cam, float) / np.linalg.norm(cam)
    res = np.zeros(7)   # _peel_top result slots 0..6 (F, d_xyz, x_exit, z_exit, J)
    k._peel_top(x, y, z, i, pl, pn, lay, fres, cam[0], cam[1], cam[2], G.footprint_mm[0] / 2,
                G.footprint_mm[1] / 2, L, np.zeros(L + 2, np.int64), np.zeros(L + 2, np.int64),
                np.zeros((L + 2, 3)), np.zeros(L + 2), np.zeros(L + 2), np.zeros(L + 2, np.int64), res,
                T._pack_inclusions(None), False)
    return res


@pytest.mark.parametrize("theta_deg", [0.0, 25.0, 50.0])
def test_peel_factor_single_interface_is_textbook(theta_deg):
    """One layer n=1.5 (mu_a = 0.2, mu_s = 3) under air: F = (1-R(theta_i)) exp(-mu_t s) cos(theta_a)/(n^2 cos(theta_i))
    with Snell's law, s = depth/cos(theta_i); the exit point is displaced by depth*tan(theta_i)."""
    n, mua, mus, thick, y0 = 1.5, 0.2, 3.0, 1.0, 0.4
    G = geom([UNI(thick)], (40.0, 40.0))
    th = math.radians(theta_deg)
    res = _peel_call(G, opt(mua, mus, 0.5, n), 0.0, y0, 0.0, 0, [math.sin(th), math.cos(th), 0.0])
    ti = math.asin(math.sin(th) / n)
    from spectraforge.studio.transport_mc import fresnel_R
    R = fresnel_R(n, 1.0, math.cos(ti))
    s = (thick - y0) / math.cos(ti)
    expect = (1 - R) * math.exp(-(mua + mus) * s) * math.cos(th) / (n * n * math.cos(ti))
    assert res[0] == pytest.approx(expect, rel=1e-9)
    assert res[4] == pytest.approx((thick - y0) * math.tan(ti), abs=1e-9)
    assert res[1] == pytest.approx(math.sin(ti), abs=1e-9) and res[2] == pytest.approx(math.cos(ti), abs=1e-9)


def test_peel_factor_chain_through_wedge_layers_conserves_etendue():
    """Two layers (n=1.4 under n=1.5, tilted top): with ALL Fresnel losses and absorption removed the
    peel-off factor is the pure solid-angle ratio, which must equal the Jacobian computed numerically from
    the forward refraction of a small bundle of directions."""
    G = geom([UNI(1.0), WEDGE(1.0, 0.2, "x")], (30.0, 30.0))
    o = opt([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.4, 1.5])
    cam = np.array([0.2, 1.0, -0.1])
    cam /= np.linalg.norm(cam)
    res = _peel_call(G, o, 1.0, 0.5, 0.0, 0, cam)
    F = res[0]
    # F = T * J; recover T from the forward chain and compare J with a finite-difference solid-angle ratio
    pl, pn = T._plane_arrays(G)

    def forward(d0):
        """direction in air for a start direction d0 in layer 0 (refract at plane 1 then plane 2)"""
        u = np.asarray(d0, float)
        for p, (n1, n2) in ((1, (1.4, 1.5)), (2, (1.5, 1.0))):
            m = pn[p]
            c1 = u @ m
            eta = n1 / n2
            c2 = math.sqrt(1 - eta * eta * (1 - c1 * c1))
            u = eta * u + (c2 - eta * c1) * m
        return u / np.linalg.norm(u)

    d0 = res[1:4]
    assert np.allclose(forward(d0), cam, atol=1e-9)             # the found direction really reaches the camera
    e1 = np.cross(d0, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d0, e1)
    h = 1e-6
    a1 = (forward(d0 + h * e1) - forward(d0 - h * e1)) / (2 * h)
    a2 = (forward(d0 + h * e2) - forward(d0 - h * e2)) / (2 * h)
    dOmega_air_per_inside = np.linalg.norm(np.cross(a1, a2))      # |a1 x a2| for unit-area input bundle
    J = 1.0 / dOmega_air_per_inside
    from spectraforge.studio.transport_mc import fresnel_R
    # transmittances along the chain
    u = d0
    Tprod = 1.0
    for p, (n1, n2) in ((1, (1.4, 1.5)), (2, (1.5, 1.0))):
        m = pn[p]
        c1 = u @ m
        Tprod *= 1 - fresnel_R(n1, n2, c1)
        eta = n1 / n2
        c2 = math.sqrt(1 - eta * eta * (1 - c1 * c1))
        u = eta * u + (c2 - eta * c1) * m
    assert F == pytest.approx(Tprod * J, rel=1e-5)


def test_peel_camera_below_the_surface_gives_no_path():
    """The line of sight from inside to a camera that does not see the top plane does not exist."""
    G = geom([UNI(1.0), UNI(1.0)], (30.0, 30.0))
    o = opt([0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.2, 2.0])
    assert _peel_call(G, o, 0.0, 0.5, 0.0, 0, [1.0, -0.2, 0.0])[0] == 0.0
    assert _peel_call(G, o, 0.0, 0.5, 0.0, 0, [0.0, 1.0, 0.0])[0] > 0.0


def _region_sum(a, lo, hi):
    return a[lo:hi, lo:hi].sum()


PEEL_CASES = {
    "mismatched_single": ([UNI(1.0)], opt(0.3, 12.0, 0.8, 1.4), opt(0.1, 9.0, 0.8, 1.4), (0, 1, 0), (0, -1, 0)),
    "index_matched": ([UNI(1.0)], opt(0.3, 12.0, 0.8, 1.0), opt(0.1, 9.0, 0.8, 1.0), (0, 1, 0), (0, -1, 0)),
    "wedge_oblique": ([UNI(1.5), WEDGE(2.0, 0.0, "x")], opt([0.2, 0.3], [10.0, 15.0], [0.8, 0.9], [1.4, 1.36]),
                      opt([0.1, 0.2], [9.0, 12.0], [0.8, 0.9], [1.4, 1.36]), (0.5, 0.8, -0.3), (0.3, -0.9, 0.1)),
    "clear_cover_mismatch": ([UNI(1.0), UNI(0.8)], opt([0.2, 0.05], [10.0, 0.0], [0.8, 0.9], [1.4, 1.5]),
                             opt([0.1, 0.05], [9.0, 0.0], [0.8, 0.9], [1.4, 1.5]), (0.2, 1, 0.1), (0, -1, 0)),
}


@pytest.mark.parametrize("name", list(PEEL_CASES))
def test_peel_off_agrees_with_cone_estimator(name):
    """Both estimators are unbiased: their elastic and fluorescence face radiance agree within Monte-Carlo
    noise (region sums over the interior, 5 mm exit cells)."""
    layers, ex_o, em_o, cam, dirn = PEEL_CASES[name]
    G = geom(layers, (30.0, 30.0))
    s = cfg(lateral=5.0, exit=5.0, sub=2, ang=12.0)
    cam = tuple(np.asarray(cam, float) / np.linalg.norm(cam))
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 50_000, direction=dirn, seed=1)
    out = {}
    for peel in (False, True):
        ex = T.excitation_pass(G, ex_o, launch, cam, s, n_photons=900_000, seed=1, peel_off=peel)
        src = ex.absorbed_mW[None].copy()
        em = T.emission_pass(G, [em_o], src, ["a"], cam, s, n_photons=900_000, seed=2, peel_off=peel)
        assert ex.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)
        assert em.info["energy_balance"][0] == pytest.approx(1.0, abs=1e-9)
        out[peel] = (ex.elastic_radiance["top"], em.radiance["top"][:, :, 0, 0])
    for k, lab in ((0, "elastic"), (1, "fluorescence")):
        a, b = _region_sum(out[False][k], 1, 5), _region_sum(out[True][k], 1, 5)
        print(f"\n{name} {lab}: cone {a:.4e}  peel-off {b:.4e}  ratio {b / a:.4f}")
        assert b == pytest.approx(a, rel=0.04)


def test_peel_off_reduces_noise_at_equal_photons():
    G = geom([UNI(1.0)], (30.0, 30.0))
    s = cfg(lateral=5.0, exit=1.0, sub=2, ang=12.0)
    o = opt(0.3, 12.0, 0.8, 1.4)
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 50_000, seed=1)
    cv = {}
    for peel in (False, True):
        r = T.excitation_pass(G, o, launch, [0, 1, 0], s, n_photons=600_000, seed=1, peel_off=peel)
        a = r.elastic_radiance["top"][8:22, 8:22]
        cv[peel] = a.std() / a.mean()
    print(f"\ncell-to-cell CV of a uniform field: cone {cv[False]:.3f}  peel-off {cv[True]:.3f}  "
          f"(x{cv[False] / cv[True]:.1f})")
    assert cv[True] < cv[False] / 1.8


def test_peel_off_is_deterministic_and_thread_invariant():
    G = geom([UNI(1.0), WEDGE(1.0, 0.3, "x")], (16.0, 16.0))
    o = opt([0.2, 0.3], [10.0, 12.0], [0.8, 0.9], [1.4, 1.36])
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 5_000, seed=1)
    s = cfg(lateral=4.0, exit=2.0, sub=2)
    res = []
    old = numba.get_num_threads()
    try:
        for nt in (1, min(4, old)):
            numba.set_num_threads(nt)
            res.append(T.excitation_pass(G, o, launch, [0.1, 1, 0], s, n_photons=60_000, seed=5).elastic_radiance["top"])
    finally:
        numba.set_num_threads(old)
    assert np.array_equal(res[0], res[1])
