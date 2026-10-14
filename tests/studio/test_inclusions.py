"""Fast tests of the heterogeneous (inclusion) 3D Monte Carlo of layered blocks (SpectralForge Studio v1.2).

Slow physics references (null / filling inclusion, first-order perturbation theory, spatially resolved reflectance
against an independent MCML-type code and Farrell diffusion theory, buried-sphere contrast) live in
``test_inclusions_validation.py``.
"""
from __future__ import annotations

import math

import numba
import numpy as np
import pytest

from spectraforge.studio import inclusions as incl_geo
from spectraforge.studio import transport3d as T
from spectraforge.studio.contracts import BlockGeometry, RegionOptics
from spectraforge.studio.schema import Inclusion, PhysicsToggles, ThicknessProfile, VolumeMCSettings

UNI = lambda t: ThicknessProfile(thickness_mm=t)
WEDGE = lambda t0, t1, axis="x": ThicknessProfile(kind="linear", thickness_mm=t0, end_thickness_mm=t1, axis=axis)


def sph(name="s", c=(0.0, 1.0, 0.0), r=1.0, mat="m"):
    return Inclusion(name=name, material_id=mat, kind="sphere", center_mm=c, radius_mm=r)


def igeom(layers, incs, fp=(20.0, 20.0)) -> BlockGeometry:
    n = len(layers)
    return BlockGeometry(0, tuple(map(float, fp)), [f"l{i}" for i in range(n)], list(range(n)),
                         T.planes_from_layers(fp, layers), np.eye(4), [i.name for i in incs],
                         [0] * len(incs), list(incs))


def iopt(layers, incl=(), lam=500.0) -> RegionOptics:
    """``layers``: list of (mua, mus, g, n); ``incl``: list of (mua, mus, g)."""
    a = lambda k, rows: np.array([r[k] for r in rows], float)
    kw = {}
    if incl:
        kw = dict(incl_mua=a(0, incl), incl_mus=a(1, incl), incl_g=a(2, incl))
    return RegionOptics(lam, a(0, layers), a(1, layers), a(2, layers), a(3, layers), **kw)


def cfg(vox=0.25, lateral=2.0, exit=2.0, sub=4, ang=12.0):
    return VolumeMCSettings(lateral_cell_mm=lateral, exit_cell_mm=exit, sublayers=sub, acceptance_half_angle_deg=ang,
                            inclusion_voxel_mm=vox)


HOST = (0.05, 10.0, 0.9, 1.4)


# ============================================================================================
# analytic shapes
# ============================================================================================
@pytest.mark.parametrize("inc", [
    Inclusion(name="a", material_id="m", kind="sphere", center_mm=(1, 2, -1), radius_mm=1.2),
    Inclusion(name="a", material_id="m", kind="ellipsoid", center_mm=(1, 2, -1), semi_axes_mm=(1.5, 0.7, 1.1)),
    Inclusion(name="a", material_id="m", kind="cylinder", center_mm=(1, 2, -1), radius_mm=0.8, axis="z", length_mm=4.0),
    Inclusion(name="a", material_id="m", kind="cylinder", center_mm=(1, 2, -1), radius_mm=0.8, axis="x", length_mm=4.0),
    Inclusion(name="a", material_id="m", kind="box", center_mm=(1, 2, -1), size_mm=(2.0, 1.0, 3.0)),
])
def test_vertical_extent_matches_brute_force_membership(inc):
    rng = np.random.default_rng(0)
    x, z = rng.uniform(-3, 5, 200), rng.uniform(-5, 3, 200)
    ylo, yhi, valid = incl_geo.vertical_extent(inc, x, z)
    ys = np.linspace(-1, 5, 1201)
    for i in range(0, 200, 7):
        m = incl_geo.inside(inc, x[i], ys, z[i])
        if not valid[i]:
            assert not m.any()
        else:
            assert ys[m].min() == pytest.approx(ylo[i], abs=0.006)
            assert ys[m].max() == pytest.approx(yhi[i], abs=0.006)
    lo, hi = incl_geo.bbox(inc)
    pts = rng.uniform(lo - 1, hi + 1, (4000, 3))
    m = incl_geo.inside(inc, pts[:, 0], pts[:, 1], pts[:, 2])
    assert np.all((pts[m] >= lo - 1e-12) & (pts[m] <= hi + 1e-12))
    # Monte Carlo volume of the shape against the analytic volume
    box = rng.uniform(lo, hi, (200_000, 3))
    frac = incl_geo.inside(inc, box[:, 0], box[:, 1], box[:, 2]).mean()
    assert frac * np.prod(hi - lo) == pytest.approx(incl_geo.analytic_volume(inc), rel=0.03)


def test_footprint_depth_clips_to_block():
    inc = sph(c=(0, 4.2, 0), r=1.0)                    # sticks out of a 5 mm block: top at 5.2 > 5
    mask, depth = incl_geo.footprint_depth(inc, np.array([0.0, 0.6, 2.0]), np.array([0.0, 0.0, 0.0]), 5.0)
    assert mask.tolist() == [True, True, False]
    assert depth[0] == pytest.approx(0.0)              # the top of the inclusion is clipped at the surface
    assert depth[1] == pytest.approx(5.0 - (4.2 + 0.8))  # sqrt(1 - .36) = .8 -> top at 5.0
    assert np.isnan(depth[2])
    inc2 = sph(c=(0, 2.0, 0), r=1.0)
    m2, d2 = incl_geo.footprint_depth(inc2, np.array([0.0]), np.array([0.0]), 5.0)
    assert d2[0] == pytest.approx(2.0)                 # top at y = 3, surface at 5
    # a sphere completely above the surface does not exist inside the block
    m3, _ = incl_geo.footprint_depth(sph(c=(0, 8.0, 0), r=1.0), np.array([0.0]), np.array([0.0]), 5.0)
    assert not m3[0]


# ============================================================================================
# voxelisation
# ============================================================================================
def test_sphere_voxel_volume_and_symmetry():
    G = igeom([UNI(5.0)], [sph(c=(0, 2.5, 0), r=1.5)])
    m = T.make_inclusion_map(G, cfg(vox=0.1))
    vol = m.occ_volume_mm3.sum()
    exact = 4 / 3 * math.pi * 1.5 ** 3
    print(f"\nsphere r=1.5 @0.1 mm voxels: {vol:.4f} vs {exact:.4f} ({vol / exact - 1:+.2%})")
    assert vol == pytest.approx(exact, rel=0.04)
    assert m.region.dtype == np.uint8 and set(np.unique(m.region)) == {0, 1}
    assert np.array_equal(m.region, m.region[::-1, :, ::-1])                # symmetric about the centre axis
    assert m.shape == (200, 50, 200) and m.occ_index.max() == m.n_occupied - 1
    assert m.dense(np.ones(m.n_occupied)).sum() == m.n_occupied


def test_box_aligned_with_voxels_is_exact_and_later_inclusion_wins():
    a = Inclusion(name="a", material_id="m", kind="box", center_mm=(0, 2.0, 0), size_mm=(4.0, 2.0, 4.0))
    b = Inclusion(name="b", material_id="m", kind="box", center_mm=(1, 2.0, 0), size_mm=(2.0, 2.0, 2.0))
    G = igeom([UNI(5.0)], [a, b])
    m = T.make_inclusion_map(G, cfg(vox=0.25))
    assert m.volume_of(1) + m.volume_of(2) == pytest.approx(4 * 2 * 4)         # union = a
    assert m.volume_of(2) == pytest.approx(2 * 2 * 2)                            # b fully inside a, and wins
    assert m.volume_of(1) == pytest.approx(32 - 8)
    ix = int((1.1 - m.origin_mm[0]) / m.voxel_mm)
    iy = int(2.0 / m.voxel_mm)
    iz = int((0.1 - m.origin_mm[2]) / m.voxel_mm)
    assert m.region[ix, iy, iz] == 2 and m.region[int((-1.9 - m.origin_mm[0]) / m.voxel_mm), iy, iz] == 1


def test_inclusion_outside_or_clipped_by_block():
    top = igeom([UNI(5.0)], [sph(c=(0, 5.0, 0), r=1.0)])               # half above the surface
    m = T.make_inclusion_map(top, cfg(vox=0.1))
    assert m.volume_of(1) == pytest.approx(0.5 * 4 / 3 * math.pi, rel=0.05)
    away = igeom([UNI(5.0)], [sph(c=(0, 9.0, 0), r=1.0)])
    assert T.make_inclusion_map(away, cfg()).n_occupied == 0
    edge = igeom([UNI(5.0)], [sph(c=(10.0, 2.0, 0), r=1.5)], fp=(20.0, 20.0))   # straddles the side face
    me = T.make_inclusion_map(edge, cfg(vox=0.1))
    assert me.volume_of(1) == pytest.approx(0.5 * 4 / 3 * math.pi * 1.5 ** 3, rel=0.05)


def test_map_is_coarsened_to_fit_the_memory_cap(monkeypatch):
    monkeypatch.setattr(T, "INCLUSION_MAX_VOXELS", 50_000)
    G = igeom([UNI(5.0)], [sph(c=(0, 2.5, 0), r=1.5)])
    m = T.make_inclusion_map(G, cfg(vox=0.05))
    assert m.voxel_mm > m.requested_voxel_mm == 0.05
    assert int(np.prod(m.shape)) <= 50_000
    assert m.volume_of(1) == pytest.approx(4 / 3 * math.pi * 1.5 ** 3, rel=0.2)


def test_covered_volume_matches_inclusion_volume():
    G = igeom([UNI(2.0), WEDGE(0.5, 2.5)], [sph(c=(0, 2.4, 0), r=1.2)])
    s = cfg(vox=0.1, lateral=1.0)
    m = T.make_inclusion_map(G, s)
    grid = T.make_voxel_grid(G, s)
    cov = T.inclusion_covered_volume(G, grid, m)
    assert cov.sum() == pytest.approx(m.occ_volume_mm3.sum(), rel=1e-9)
    vol = T.voxel_volumes(G, grid)
    assert np.all(cov <= vol * (1 + 1e-9) + 1e-12)
    assert (T.host_voxel_volumes(G, grid, m) >= 0).all()


# ============================================================================================
# bit-identity without inclusions
# ============================================================================================
def test_blocks_without_inclusions_are_bit_identical_to_v11():
    """Golden values computed with the unmodified v1.1 engine (``git show HEAD:transport3d*.py``) for the same call:
    wedge + clear + zero-thickness layers, Fresnel mismatch at every interface."""
    fp = (12.0, 10.0)
    layers = [UNI(1.0), WEDGE(0.2, 1.2), UNI(0.0), UNI(0.4)]
    G = BlockGeometry(0, fp, [f"l{i}" for i in range(4)], list(range(4)), T.planes_from_layers(fp, layers), np.eye(4))
    opt = RegionOptics(500.0, np.array([0.05, 0.4, 0.1, 0.02]), np.array([8.0, 12.0, 0.0, 5.0]),
                       np.array([0.8, 0.9, 0.0, 0.5]), np.array([1.4, 1.33, 1.5, 1.4]))
    s = VolumeMCSettings(lateral_cell_mm=1.0, exit_cell_mm=1.0, sublayers=4)
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 5000)
    cam = [0.2, 1.0, 0.1]
    a = T.excitation_pass(G, opt, launch, cam, s, n_photons=200_000, seed=3, peel_off=False)
    assert a.diffuse_reflectance == 0.28551617326592205
    assert a.transmittance == 0.22745448170950383
    assert a.absorbed_fraction == 0.4592434066905169
    assert float(a.absorbed_mW.sum()) == 0.45924340669052166
    assert float(a.fluence_mW_mm2.sum()) == 22.135832590110677
    assert float(a.elastic_radiance["top"].sum()) == 0.09119772837046573
    src = np.zeros((2,) + a.absorbed_mW.shape)
    src[0, :, :, 1, :] = a.absorbed_mW[:, :, 1, :]
    src[1, :, :, 3, :] = a.absorbed_mW[:, :, 3, :]
    c = T.emission_pass(G, [opt], src, ["a", "b"], cam, s, n_photons=100_000, seed=4, peel_off=False)
    assert float(c.radiance["top"].sum()) == 0.030376846919995853
    assert c.info["escape_top"][0] == 0.2144279173285886
    assert float(np.nansum(c.origin_depth_mm["top"])) == 73.39378626447635


# ============================================================================================
# conservation, tallies, robustness
# ============================================================================================
SCENES = {
    "sphere": ([UNI(5.0)], [sph(c=(0, 2.5, 0), r=1.2)], [HOST], [(0.5, 8.0, 0.8)]),
    "absorber_clear": ([UNI(5.0)], [sph(c=(0, 2.5, 0), r=1.2)], [HOST], [(2.0, 0.0, 0.0)]),     # mu_s = 0 inclusion
    "straddles_edge": ([UNI(4.0)], [sph(c=(10.0, 2.0, 0), r=1.5)], [HOST], [(0.3, 15.0, 0.7)]),
    "crosses_wedge_interface": ([UNI(1.0), WEDGE(0.2, 3.0)], [sph(c=(0, 1.5, 0), r=1.0)],
                                [HOST, (0.2, 14.0, 0.85, 1.33)], [(0.6, 9.0, 0.8)]),
    "clear_host_turbid_incl": ([UNI(1.5), UNI(2.0)], [sph(c=(0, 1.5, 0), r=0.9)],
                               [(0.02, 0.0, 0.0, 1.45), HOST], [(0.4, 12.0, 0.8)]),
    "vessel_and_box": ([UNI(4.0)], [Inclusion(name="v", material_id="m", kind="cylinder", center_mm=(0, 3.0, 0),
                                              radius_mm=0.6, axis="x"),
                                    Inclusion(name="b", material_id="m", kind="box", center_mm=(4, 1.5, 4),
                                              size_mm=(3, 1, 3))],
                      [HOST], [(1.0, 5.0, 0.5), (0.1, 30.0, 0.95)]),
}


@pytest.mark.parametrize("name", list(SCENES))
def test_energy_conservation_and_tally_split(name):
    layers, incs, lprops, iprops = SCENES[name]
    G = igeom(layers, incs)
    if len(lprops) == 1:
        lprops = lprops * len(layers)
    opt = iopt(lprops, iprops)
    launch = T.uniform_beam_launch(G, 500.0, 2.0, 20_000)
    r = T.excitation_pass(G, opt, launch, [0.1, 1.0, 0.0], cfg(), n_photons=60_000, seed=5)
    assert r.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)
    tot = r.diffuse_reflectance + r.transmittance + r.absorbed_fraction + r.info["specular"]
    assert tot + r.info["lost_steps"] - r.info["roulette_net"] == pytest.approx(1.0, abs=1e-9)
    norm = 1.0 / 2.0
    A_layers = r.absorbed_mW.sum() * norm
    A_incl = r.inclusion_absorbed_mW.sum() * norm
    assert A_layers + A_incl == pytest.approx(r.absorbed_fraction, abs=1e-9)
    assert A_incl == pytest.approx(sum(r.info["inclusion_absorbed_fraction"]), abs=1e-12)
    for arr in (r.absorbed_mW, r.fluence_mW_mm2, r.inclusion_absorbed_mW, r.inclusion_fluence_mW_mm2,
                r.elastic_radiance["top"], r.top_exit_mW):
        assert np.all(np.isfinite(arr)) and arr.min() >= 0.0
    assert A_incl > 0
    assert r.top_exit_mW.sum() == pytest.approx(r.info["R_top"] * 2.0, rel=1e-9)   # per-cell top exit power sums to R


def test_inclusion_voxels_hold_no_layer_tally():
    """A box that exactly covers one tally sublayer (y in [2.5, 3.75]) over |x|, |z| < 3: the layer (host) tallies of
    those voxels are exactly zero, the inclusion tallies carry the absorption."""
    G = igeom([UNI(5.0)], [Inclusion(name="b", material_id="m", kind="box", center_mm=(0, 3.125, 0), size_mm=(6, 1.25, 6))])
    s = cfg(vox=0.25, lateral=1.0, sub=4)
    r = T.excitation_pass(G, iopt([HOST], [(0.3, 10.0, 0.9)]), T.uniform_beam_launch(G, 500.0, 1.0, 20_000),
                          [0, 1, 0], s, n_photons=60_000, seed=2)
    xc = 0.5 * (r.grid.x_edges_mm[1:] + r.grid.x_edges_mm[:-1])
    inside = np.abs(xc) < 3.0
    cov = r.absorbed_mW[np.ix_(inside, inside)][:, :, 0, 2]
    assert np.all(cov == 0.0)
    assert np.all(r.fluence_mW_mm2[np.ix_(inside, inside)][:, :, 0, 2] == 0.0)
    assert r.absorbed_mW[np.ix_(inside, inside)][:, :, 0, 3].max() > 0           # the sublayer above is host material
    assert r.inclusion_absorbed_mW.sum() > 0 and r.inclusion_fluence_mW_mm2.max() > 0


def test_determinism_and_thread_invariance_with_inclusions():
    layers, incs, lprops, iprops = SCENES["vessel_and_box"]
    G = igeom(layers, incs)
    opt = iopt(lprops, iprops)
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 5000)
    ref = None
    old = numba.get_num_threads()
    try:
        for nt in (1, min(4, old), old, old):
            numba.set_num_threads(nt)
            r = T.excitation_pass(G, opt, launch, [0.1, 1.0, 0.0], cfg(), n_photons=40_000, seed=11)
            sig = (r.diffuse_reflectance, r.absorbed_mW.tobytes(), r.inclusion_absorbed_mW.tobytes(),
                   r.inclusion_fluence_mW_mm2.tobytes(), r.elastic_radiance["top"].tobytes())
            if ref is None:
                ref = sig
            assert sig == ref, f"{nt} threads differ"
    finally:
        numba.set_num_threads(old)
    r2 = T.excitation_pass(G, opt, launch, [0.1, 1.0, 0.0], cfg(), n_photons=40_000, seed=12)
    assert r2.diffuse_reflectance != ref[0]


@pytest.mark.parametrize("peel", [True, False])
def test_emission_from_an_inclusion_source(peel):
    layers, incs, lprops, iprops = SCENES["sphere"]
    G = igeom(layers, incs)
    s = cfg()
    m = T.make_inclusion_map(G, s)
    isrc = np.zeros((1,) + m.shape)
    isrc[0].reshape(-1)[m.occ_flat] = 0.01 * m.occ_volume_mm3
    layers_src = np.zeros((1,) + T.make_voxel_grid(G, s).shape)
    em = T.emission_pass(G, [iopt(lprops, iprops)], layers_src, ["x"], [0.0, 1.0, 0.0], s, n_photons=40_000, seed=3,
                         peel_off=peel, inclusion_sources=isrc)
    assert em.info["energy_balance"][0] == pytest.approx(1.0, abs=1e-9)
    assert em.info["inclusion_emitted_mW"][0] == pytest.approx(0.01 * m.occ_volume_mm3.sum())
    assert em.info["layer_emitted_mW"] == 0.0
    f = em.origin_inclusion_fraction["top"]
    ol = em.origin_layer_fraction["top"]
    ok = np.isfinite(f[..., 0])
    assert ok.any()
    assert np.allclose(f[ok][:, 0], 1.0) and np.allclose(ol[ok][:, 0], 0.0)    # everything that escapes came from the sphere
    assert np.nanmax(em.radiance["top"]) > 0 and np.isfinite(em.radiance["top"]).all()
    # sources are emitted in the sphere: the origin depth is the sphere depth (5 - 2.5 = 2.5 mm, +- 1.2)
    d = em.origin_depth_mm["top"][ok]
    assert np.all(np.abs(d - 2.5) < 1.3)


def test_host_sources_are_not_emitted_inside_inclusions():
    """Host source only in the lower half (y in [0, 2.5]) of a 5 mm layer, a box inclusion over y in [1.5, 3.5]: every
    packet must start at y < 1.5, i.e. at a depth below the surface of at least 3.5 mm."""
    G = igeom([UNI(5.0)], [Inclusion(name="b", material_id="m", kind="box", center_mm=(0, 2.5, 0), size_mm=(20, 2, 20))])
    s = cfg(sub=2, lateral=5.0)
    grid = T.make_voxel_grid(G, s)
    src = np.zeros((1,) + grid.shape)
    src[0, :, :, 0, 0] = 1.0
    em = T.emission_pass(G, [iopt([HOST], [HOST[:3]])], src, ["x"], [0, 1, 0], s, n_photons=30_000, seed=1)
    ok = np.isfinite(em.origin_depth_mm["top"])
    assert ok.any()
    assert np.nanmin(em.origin_depth_mm["top"]) >= 3.5 - 1e-6
    assert em.info["energy_balance"][0] == pytest.approx(1.0, abs=1e-9)
    assert em.origin_inclusion_fraction["top"][ok][:, 0].max() == 0.0


def test_toggle_scattering_off_applies_to_inclusions():
    layers, incs, lprops, iprops = SCENES["sphere"]
    G = igeom(layers, incs)
    r = T.excitation_pass(G, iopt(lprops, iprops), T.uniform_beam_launch(G, 500.0, 1.0, 5000), [0, 1, 0], cfg(),
                          n_photons=20_000, seed=1, toggles=PhysicsToggles(scattering=False))
    assert r.diffuse_reflectance < 0.05                         # no scattering: only (negligible) Fresnel returns
    assert r.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)


def test_optics_validation_for_inclusions():
    G = igeom([UNI(2.0)], [sph(c=(0, 1, 0))])
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 100)
    with pytest.raises(ValueError, match="incl_mua"):
        T.excitation_pass(G, iopt([HOST]), launch, [0, 1, 0], cfg(), n_photons=1000, seed=1)
    with pytest.raises(ValueError, match="inclusion optical"):
        T.excitation_pass(G, iopt([HOST], [(-1.0, 1.0, 0.0)]), launch, [0, 1, 0], cfg(), n_photons=1000, seed=1)
