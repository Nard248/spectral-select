"""Layered blocks: geometry / tracer support and the 3D-Monte-Carlo capture orchestration.

The orchestration tests run against a FAKE ``transport3d`` (analytic exponential-attenuation model, same API
as the real module) so they test the capture side only: source construction, bleaching order, channel
bookkeeping, pixel sampling, ground truth, explain_pixel consistency, caching. ``test_real_*`` tests use the
real ``transport3d`` when it exists (marked slow-ish, small photon counts)."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spectraforge.studio import block_capture, capture, geometry
from spectraforge.studio import optics as optics_mod
from spectraforge.studio.contracts import (BlockGeometry, EmissionTallies, ExcitationTallies, FaceGrid,
                                           VoxelGrid)
from spectraforge.studio.presets import PRESETS
from spectraforge.studio.schema import PhysicsToggles
from spectraforge.studio.tracer import trace_scene


# ============================================================================================
# fake transport3d
# ============================================================================================
def _fake_module() -> types.ModuleType:
    m = types.ModuleType("spectraforge.studio.transport3d")
    m.calls = {"exc": 0, "em": 0}

    def build_block_geometry(exp, oi):
        o = exp.objects[oi]
        sh = o.shape
        R, p = geometry.object_transform(o.transform)
        W = np.eye(4)
        W[:3, :3], W[:3, 3] = R, p
        ids = [mm.id for mm in exp.materials]
        return BlockGeometry(oi, tuple(sh.footprint_mm), [l.name for l in sh.layers],
                             [ids.index(l.material_id) for l in sh.layers],
                             geometry.block_plane_coeffs(sh), W)

    def make_voxel_grid(geom, s):
        X, Z = geom.footprint_mm
        nx, nz = max(2, int(round(X / s.lateral_cell_mm))), max(2, int(round(Z / s.lateral_cell_mm)))
        return VoxelGrid(nx, nz, len(geom.layer_names), s.sublayers, np.linspace(-X / 2, X / 2, nx + 1),
                         np.linspace(-Z / 2, Z / 2, nz + 1))

    def make_face_grids(geom, s):
        X, Z = geom.footprint_mm
        ymax = float(np.max(geom.plane_coeffs[-1, 0] + np.abs(geom.plane_coeffs[-1, 1]) * X / 2
                            + np.abs(geom.plane_coeffs[-1, 2]) * Z / 2))
        n = lambda L: max(2, int(round(L / s.exit_cell_mm)))   # noqa: E731
        return {"top": FaceGrid("top", np.linspace(-X / 2, X / 2, n(X) + 1), np.linspace(-Z / 2, Z / 2, n(Z) + 1)),
                "+x": FaceGrid("+x", np.linspace(-Z / 2, Z / 2, n(Z) + 1), np.linspace(0, ymax, n(ymax) + 1)),
                "-x": FaceGrid("-x", np.linspace(-Z / 2, Z / 2, n(Z) + 1), np.linspace(0, ymax, n(ymax) + 1)),
                "+z": FaceGrid("+z", np.linspace(-X / 2, X / 2, n(X) + 1), np.linspace(0, ymax, n(ymax) + 1)),
                "-z": FaceGrid("-z", np.linspace(-X / 2, X / 2, n(X) + 1), np.linspace(0, ymax, n(ymax) + 1))}

    def _vox_geometry(geom, grid):
        xc = 0.5 * (grid.x_edges_mm[1:] + grid.x_edges_mm[:-1])
        zc = 0.5 * (grid.z_edges_mm[1:] + grid.z_edges_mm[:-1])
        X, Z = np.meshgrid(xc, zc, indexing="ij")
        pc = geom.plane_coeffs
        tops = pc[:, 0, None, None] + pc[:, 1, None, None] * X + pc[:, 2, None, None] * Z      # (L,nx,nz)
        lows = np.concatenate([np.zeros((1,) + X.shape), tops[:-1]])
        sub = (np.arange(grid.sublayers) + 0.5) / grid.sublayers
        y = lows[..., None] + (tops - lows)[..., None] * sub                                  # (L,nx,nz,S)
        depth = tops[-1][None, ..., None] - y                                                 # below top
        return np.moveaxis(y, 0, 2), np.moveaxis(depth, 0, 2), np.moveaxis(tops - lows, 0, 2)  # (nx,nz,L,S) ...

    def _tau_above(geom, grid, mueff):
        """cumulative effective optical depth from the top surface to the voxel centre (nx,nz,L,S)."""
        _y, depth, thick = _vox_geometry(geom, grid)             # thick: (nx,nz,L)
        L = len(geom.layer_names)
        tau = np.zeros_like(depth)
        cum_above = np.zeros(depth.shape[:2])
        for li in range(L - 1, -1, -1):
            d_in = depth[:, :, li, :] - cum_above[..., None]       # depth within layer li (0..thick)
            tau[:, :, li, :] = (cum_above * 0)[..., None]
            tau[:, :, li, :] = tau_acc(li, cum_above, d_in, mueff)
            cum_above = cum_above + thick[:, :, li]
        return tau

    def tau_acc(li, cum_above, d_in, mueff):
        return _TAU_CTX["above"][li][..., None] + mueff[li] * np.maximum(d_in, 0.0)

    _TAU_CTX: dict = {}

    def _with_tau(geom, grid, mueff):
        _y, depth, thick = _vox_geometry(geom, grid)
        L = len(geom.layer_names)
        above = [None] * L
        acc = np.zeros(thick.shape[:2])
        for li in range(L - 1, -1, -1):
            above[li] = acc.copy()
            acc = acc + mueff[li] * thick[:, :, li]
        _TAU_CTX["above"] = above
        return _tau_above(geom, grid, mueff)

    def excitation_pass(geom, optics, launch, camera_dir_local, settings, *, n_photons, seed, toggles=None,
                        peel_off=True):
        m.calls["exc"] += 1
        grid = m.make_voxel_grid(geom, settings)
        faces = m.make_face_grids(geom, settings)
        mua, mus, g = optics.mua_per_mm, optics.mus_per_mm, optics.g
        mueff = np.sqrt(3 * mua * (mua + mus * (1 - g)))
        tau = _with_tau(geom, grid, mueff)
        X, Z = geom.footprint_mm
        P = float(launch.power_mW.sum())
        E0 = P / (X * Z)
        fl = E0 * 2.0 * np.exp(-tau)
        _y, _d, thick = _vox_geometry(geom, grid)
        dx = np.diff(grid.x_edges_mm)[:, None, None, None]
        dz = np.diff(grid.z_edges_mm)[None, :, None, None]
        vol = dx * dz * thick[..., None] / grid.sublayers
        ab = fl * mua[None, None, :, None] * vol
        el = {f: np.zeros((len(fg.u_edges_mm) - 1, len(fg.v_edges_mm) - 1)) for f, fg in faces.items()}
        el["top"][:] = E0 * 0.3 / np.pi
        return ExcitationTallies(optics.lambda_nm, grid, ab, fl, el, faces, 0.3, 0.0, 0.7, n_photons)

    def emission_pass(geom, optics_nodes, sources, channels, camera_dir_local, settings, *, n_photons, seed,
                      toggles=None, peel_off=True):
        m.calls["em"] += 1
        grid = m.make_voxel_grid(geom, settings)
        faces = m.make_face_grids(geom, settings)
        Ne, C = len(optics_nodes), len(channels)
        out = {f: np.zeros((len(fg.u_edges_mm) - 1, len(fg.v_edges_mm) - 1, Ne, C)) for f, fg in faces.items()}
        fg = faces["top"]
        xi = np.clip(np.searchsorted(grid.x_edges_mm, 0.5 * (fg.u_edges_mm[1:] + fg.u_edges_mm[:-1])) - 1, 0, grid.nx - 1)
        zi = np.clip(np.searchsorted(grid.z_edges_mm, 0.5 * (fg.v_edges_mm[1:] + fg.v_edges_mm[:-1])) - 1, 0, grid.nz - 1)
        cell = (np.diff(fg.u_edges_mm)[:, None] * np.diff(fg.v_edges_mm)[None, :])
        for j, ro in enumerate(optics_nodes):
            mueff = np.sqrt(3 * ro.mua_per_mm * (ro.mua_per_mm + ro.mus_per_mm * (1 - ro.g)))
            tau = _with_tau(geom, grid, mueff)
            att = np.exp(-tau)
            for c in range(C):
                col = (sources[c] * att).sum(axis=(2, 3))                     # (nx,nz)
                out["top"][:, :, j, c] = col[np.ix_(xi, zi)] / (np.pi * cell)
        return EmissionTallies(np.array([ro.lambda_nm for ro in optics_nodes]), list(channels), out, faces,
                               n_photons=n_photons)

    m.build_block_geometry, m.make_voxel_grid, m.make_face_grids = build_block_geometry, make_voxel_grid, make_face_grids
    m.excitation_pass, m.emission_pass = excitation_pass, emission_pass
    return m


@pytest.fixture
def fake3d(monkeypatch):
    mod = _fake_module()
    monkeypatch.setitem(sys.modules, "spectraforge.studio.transport3d", mod)
    import spectraforge.studio as pkg
    monkeypatch.setattr(pkg, "transport3d", mod, raising=False)
    block_capture.clear_cache()
    yield mod
    block_capture.clear_cache()


def _small(exp, photons=20_000, nodes=40.0, cell=1.0):
    """Cheap settings: coarse grid, few nodes, two excitations."""
    vs = exp.acquisition.volume_mc.model_copy(update={
        "excitation_photons": photons, "emission_photons": photons, "emission_node_step_nm": nodes,
        "lateral_cell_mm": cell, "exit_cell_mm": cell, "sublayers": 4})
    acq = exp.acquisition.model_copy(update={"volume_mc": vs})
    return exp.model_copy(update={"acquisition": acq})


def _steps(exp, n):
    src = exp.source()
    seq = [s for s in src.sequence if s.enabled][:n]
    return exp.model_copy(update={"sources": [src.model_copy(update={"sequence": seq}) if s.id == src.id else s
                                              for s in exp.sources]})


def _wedge(n_steps=2, **kw):
    return _steps(_small(PRESETS["layered_wedge_phantom"](), **kw), n_steps)


# ============================================================================================
# geometry / tracer
# ============================================================================================
@pytest.mark.parametrize("name", ["layered_wedge_phantom", "tissue_under_collagen"])
def test_block_presets_trace(name):
    h = trace_scene(PRESETS[name](), 87, 64)
    bh = h.info["block_hits"]
    assert 0 in bh and len(bh[0]["pixel"]) > 500
    assert np.all(np.isfinite(h.irradiance_per_mW)) and np.all(np.isfinite(h.glint_per_mW))
    m = h.material[0, bh[0]["pixel"]]
    exp = PRESETS[name]()
    assert np.all(m == [mm.id for mm in exp.materials].index(exp.objects[0].shape.layers[-1].material_id))
    assert set(np.unique(bh[0]["face"])) <= {"top", "+x", "-x", "+z", "-z"}


def test_block_plane_coeffs_match_profiles():
    shape = PRESETS["layered_wedge_phantom"]().objects[0].shape
    pc = geometry.block_plane_coeffs(shape)
    for x, z in ((-17.0, 3.0), (0.0, 0.0), (12.5, -8.0)):
        acc = 0.0
        for i, layer in enumerate(shape.layers):
            acc += layer.thickness.at(x, z, shape.footprint_mm)
            assert pc[i, 0] + pc[i, 1] * x + pc[i, 2] * z == pytest.approx(acc, abs=1e-9)


@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_block_ray_hits_top_plane(backend, monkeypatch):
    monkeypatch.setattr(geometry.Scene, "backend", backend)
    exp = PRESETS["layered_wedge_phantom"]()
    sc = geometry.build_scene(exp)
    shape = exp.objects[0].shape
    pc = geometry.block_plane_coeffs(shape)[-1]
    xs = np.array([-15.0, -5.0, 0.0, 9.0, 18.0])
    o = np.stack([xs, np.full(5, 50.0), np.zeros(5)], 1)
    h = sc.first_hit(o, np.tile([0.0, -1.0, 0.0], (5, 1)))
    expect = pc[0] + pc[1] * xs
    assert np.allclose(50.0 - h.t, expect, atol=1e-6)
    assert np.all(h.entering)
    nrm = np.array([-pc[1], 1.0, -pc[2]]) / np.linalg.norm([-pc[1], 1.0, -pc[2]])
    assert np.allclose(h.normal, nrm, atol=1e-9)


def test_block_is_opaque_to_shadow_rays_and_ends_chain():
    exp = PRESETS["layered_wedge_phantom"]()
    exp.materials  # top layer (intralipid cover) is turbid; make a clear-topped block too
    from spectraforge.studio.schema import BlockLayer, LayeredBlockShape, ThicknessProfile
    shape = LayeredBlockShape(footprint_mm=(20, 20), layers=[
        BlockLayer(name="a", material_id="tissue_substrate", thickness=ThicknessProfile(thickness_mm=3)),
        BlockLayer(name="b", material_id="clear_cover", thickness=ThicknessProfile(thickness_mm=1))])
    obj = exp.objects[0].model_copy(update={"shape": shape})
    h = trace_scene(exp.model_copy(update={"objects": [obj]}), 64, 48)
    assert (h.material[1] >= 0).sum() == 0 or np.all(h.object_index[1][h.object_index[0] == 0] != 0)
    assert np.all(h.object_index[1, h.object_index[0] == 0] == -1)      # chain ends at the block


# ============================================================================================
# capture orchestration (fake 3D)
# ============================================================================================
def test_presets_capture_no_nan(fake3d):
    for name in ("layered_wedge_phantom", "tissue_under_collagen"):
        exp = _steps(_small(PRESETS[name]()), 2)
        r = capture.capture(exp, scale=0.25)
        for ex in r.excitations_nm:
            assert np.all(np.isfinite(r.adu[ex])) and np.all(np.isfinite(r.radiance[ex]))
            assert r.radiance[ex].max() > 0
        assert r.blocks and r.blocks[0]["layer_names"]
        v = r.volumes[r.blocks[0]["object_id"]]
        assert set(v["fluence"]) == set(r.excitations_nm) and v["source"]
        for k in ("layer_thickness:cover" if name == "layered_wedge_phantom" else "layer_thickness:collagen",
                  "block_face", "origin_depth_mm"):
            assert k in r.ground_truth


def test_progress_messages(fake3d):
    msgs = []
    capture.capture(_wedge(2), scale=0.25, progress=lambda f, m: msgs.append((f, m)))
    text = [m for _, m in msgs]
    assert any("· 3D excitation" in t for t in text) and any(t.startswith("emission node") and t.endswith("· 3D") for t in text)
    assert text[0] == "tracing rays" and text[-1] == "done"
    fr = [f for f, _ in msgs]
    assert all(b >= a - 1e-12 for a, b in zip(fr, fr[1:]))


def test_gt_layer_thickness_matches_wedge(fake3d):
    exp = _wedge(1)
    r = capture.capture(exp, scale=0.25)
    gt = r.ground_truth["layer_thickness:cover"]
    h = trace_scene(exp, r.width, r.height)
    d = h.info["block_hits"][0]
    top = d["face"] == "top"
    sh = exp.objects[0].shape
    expect = np.array([sh.layers[2].thickness.at(x, z, sh.footprint_mm) for x, z in zip(d["x"][top], d["z"][top])])
    got = gt.reshape(-1)[d["pixel"][top]]
    assert np.allclose(got, expect, atol=1e-4)
    assert np.all(np.isnan(gt.reshape(-1)[h.object_index[0] == -2]))
    depth = r.ground_truth["layer_depth:fluorescent layer"].reshape(-1)[d["pixel"][top]]
    assert np.allclose(depth, expect, atol=1e-4)               # depth of the fluor layer top = cover thickness


def _roi_mean(r, ex, band, xs, per_irradiance=False):
    gt = r.ground_truth["layer_thickness:cover"]
    img = r.radiance[ex][:, :, band]
    if per_irradiance:    # the lamp is off-axis (E falls along +x): yield per unit local surface irradiance
        img = img / r.ground_truth[f"irradiance@{ex:g}"]
    out = []
    for a, b in xs:
        m = (gt >= a) & (gt < b)
        out.append(img[m].mean() if m.any() else np.nan)
    return np.array(out)


def test_buried_signal_increases_as_cover_thins(fake3d):
    exp = _wedge(1)
    r = capture.capture(exp, scale=0.5)
    ex = r.excitations_nm[0]
    em = r.emission_nm[ex]
    band = int(np.argmin(np.abs(em - 520)))
    v = _roi_mean(r, ex, band, [(1.6, 2.0), (1.2, 1.6), (0.8, 1.2), (0.4, 0.8), (0.0, 0.4)])
    assert np.all(np.isfinite(v)) and np.all(np.diff(v) > 0)


def test_dark_mode_has_no_block_fluorescence(fake3d):
    r = capture.capture(_wedge(1), mode="dark", scale=0.25)
    ex = r.excitations_nm[0]
    gt = r.ground_truth["layer_thickness:cover"]
    assert r.radiance[ex][np.isfinite(gt)].max() == pytest.approx(0.0, abs=1e-12)


def test_bleaching_order_matters_for_blocks(fake3d):
    base = _small(PRESETS["layered_wedge_phantom"]())
    base = base.model_copy(update={"acquisition": base.acquisition.model_copy(update={
        "physics": base.acquisition.physics.model_copy(update={"power_drift": False, "shot_noise": False})})})
    src = base.source()
    seq = [s.model_copy(update={"power_mW": 300.0, "exposure_ms": 20_000.0}) for s in src.sequence if s.enabled][-2:]
    def run(order):
        e = base.model_copy(update={"sources": [src.model_copy(update={"sequence": order}) if s.id == src.id else s
                                                for s in base.sources]})
        r = capture.capture(e, scale=0.25)
        return {ex: r.radiance[ex] for ex in r.excitations_nm}
    a, b = run(seq), run(seq[::-1])
    ex = seq[1].wavelength_nm              # second step in `a`, first in `b`
    # same step, different bleaching history -> different signal; the later one is dimmer
    assert a[ex].sum() < b[ex].sum() * 0.999
    tg_off = base.acquisition.physics.model_copy(update={"photobleaching": False})
    e0 = base.model_copy(update={"sources": [src.model_copy(update={"sequence": seq}) if s.id == src.id else s
                                             for s in base.sources],
                                 "acquisition": base.acquisition.model_copy(update={"physics": tg_off})})
    r0 = capture.capture(e0, scale=0.25)
    assert r0.radiance[ex].sum() > a[ex].sum()


def test_no_fluorescence_toggle_leaves_only_elastic(fake3d):
    e = _wedge(1)
    tg = e.acquisition.physics.model_copy(update={"fluorescence": False, "raman": False})
    e = e.model_copy(update={"acquisition": e.acquisition.model_copy(update={"physics": tg})})
    r = capture.capture(e, scale=0.25)
    assert not r.volumes["block"]["source"]


def test_explain_pixel_sums_to_radiance(fake3d):
    exp = _wedge(2)
    r = capture.capture(exp, scale=0.5)
    gt = r.ground_truth["layer_thickness:cover"]
    ys, xs = np.nonzero(np.isfinite(gt))
    # interior pixel of the block
    k = len(ys) // 2
    y, x = int(ys[k]), int(xs[k])
    calls = dict(fake3d.calls)
    e = capture.explain_pixel(exp, x, y, scale=0.5)
    assert fake3d.calls == calls                                  # served from the MC cache
    for key, comps in e["components"].items():
        names = set(comps)
        assert any(n.startswith("dye:") and "@" in n for n in names) and "elastic" in names
        tot = np.sum([np.array(v) for v in comps.values()], axis=0)
        assert np.allclose(tot, e["radiance"][key], rtol=1e-4, atol=1e-12)
        assert np.allclose(e["radiance"][key], r.radiance[float(key)][y, x], rtol=0.01, atol=1e-9)
    assert e["ground_truth"]["layer_thickness:cover"] == pytest.approx(gt[y, x], abs=1e-4)


def test_mc_cache_reused_between_captures(fake3d):
    exp = _wedge(1)
    capture.capture(exp, scale=0.25)
    n = dict(fake3d.calls)
    capture.capture(exp, scale=0.25)
    assert fake3d.calls == n
    capture.capture(exp, scale=1.0 if False else 0.26)           # different scale, same photon fraction
    assert fake3d.calls == n


def test_tissue_collagen_autofluorescence_grows_with_thickness(fake3d):
    exp = _steps(_small(PRESETS["tissue_under_collagen"]()), 1)
    r = capture.capture(exp, scale=0.5)
    gt = r.ground_truth["layer_thickness:collagen"]
    ex = r.excitations_nm[0]
    band = int(np.argmin(np.abs(r.emission_nm[ex] - 400)))
    # the collagen-cover component is the only one that grows with its own thickness: check the source volume
    src = r.volumes["block"]["source"]
    key = next(k for k in src if k[1] == "collagen|collagen")
    col = src[key].sum(axis=(2, 3))
    xs = r.volumes["block"]["grid"].x_edges_mm
    assert col[:2].sum() > col[-2:].sum()                          # cover is thickest at -x
    assert gt[np.isfinite(gt)].max() > 1.2


def test_voxel_volumes_sum_to_block_volume(fake3d):
    exp = _wedge(1)
    r = capture.capture(exp, scale=0.25)
    run = block_capture.get_runs.__globals__["_CACHE"]
    runs = next(iter(run.values()))[1]
    vol = runs[0].voxel_volumes()
    sh = exp.objects[0].shape
    expect = 40.0 * 40.0 * (4.0 + 0.5 + 1.0)                      # wedge averages 1 mm
    assert vol.sum() == pytest.approx(expect, rel=1e-6)


# ============================================================================================
# real transport3d (skipped until the module exists)
# ============================================================================================
def _real():
    return pytest.importorskip("spectraforge.studio.transport3d")


def _quiet(exp):
    tg = exp.acquisition.physics.model_copy(update={"power_drift": False, "shot_noise": False, "read_noise": False,
                                                    "dark_current": False, "fixed_pattern_noise": False})
    return exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"physics": tg})})


def _only_step(exp, wl):
    src = exp.source()
    seq = [s for s in src.sequence if s.enabled and abs(s.wavelength_nm - wl) < 1e-6]
    return exp.model_copy(update={"sources": [src.model_copy(update={"sequence": seq}) if s.id == src.id else s
                                              for s in exp.sources]})


@pytest.fixture(scope="module")
def real_wedge():
    _real()
    block_capture.clear_cache()
    exp = _quiet(_small(_only_step(PRESETS["layered_wedge_phantom"](), 490.0), photons=600_000, nodes=30.0, cell=1.0))
    exp = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={
        "physics": exp.acquisition.physics.model_copy(update={"photobleaching": False})})})
    return exp, capture.capture(exp, scale=0.5)


def test_real_wedge_no_nan_and_monotone(real_wedge):
    exp, r = real_wedge
    ex = r.excitations_nm[0]
    assert np.all(np.isfinite(r.adu[ex])) and np.all(np.isfinite(r.radiance[ex]))
    band = int(np.argmin(np.abs(r.emission_nm[ex] - 530)))
    # bins stay >= ~3 mm away from the block edges (lateral light loss through the side faces is real)
    v = _roi_mean(r, ex, band, [(1.4, 1.7), (1.1, 1.4), (0.8, 1.1), (0.5, 0.8), (0.3, 0.5)],
                  per_irradiance=True)
    assert np.all(np.isfinite(v)) and np.all(np.diff(v) > 0), v


def test_real_wedge_explain_matches_capture(real_wedge):
    exp, r = real_wedge
    gt = r.ground_truth["layer_thickness:cover"]
    ys, xs = np.nonzero(np.isfinite(gt))
    y, x = int(np.median(ys)), int(np.median(xs))
    e = capture.explain_pixel(exp, x, y, scale=0.5)
    key = f"{r.excitations_nm[0]:g}"
    tot = np.sum([np.array(v) for v in e["components"][key].values()], axis=0)
    assert np.allclose(tot, e["radiance"][key], rtol=1e-3, atol=1e-12)
    assert np.allclose(e["radiance"][key], r.radiance[r.excitations_nm[0]][y, x], rtol=0.01, atol=1e-12)
    names = set(e["components"][key])
    assert "dye:fluorescein@fluorescent layer" in names and "elastic" in names


def test_real_dark_mode_has_no_block_fluorescence(real_wedge):
    exp, _ = real_wedge
    r = capture.capture(exp, mode="dark", scale=0.25)
    gt = r.ground_truth["layer_thickness:cover"]
    assert r.radiance[r.excitations_nm[0]][np.isfinite(gt)].max() == 0.0


@pytest.fixture(scope="module")
def real_tissue():
    _real()
    block_capture.clear_cache()
    exp = _quiet(_small(_only_step(PRESETS["tissue_under_collagen"](), 365.0), photons=600_000, nodes=30.0, cell=1.0))
    return exp, capture.capture(exp, scale=0.5)


def _strip_components(exp, r, lo, hi, n=6):
    """Mean noise-free component spectra over ``n`` block pixels whose cover thickness is in [lo, hi)."""
    gt = r.ground_truth["layer_thickness:collagen"]
    ys, xs = np.nonzero((gt >= lo) & (gt < hi))
    assert ys.size >= n
    pick = np.linspace(0, ys.size - 1, n).astype(int)
    key = f"{r.excitations_nm[0]:g}"
    acc: dict[str, np.ndarray] = {}
    for i in pick:
        e = capture.explain_pixel(exp, int(xs[i]), int(ys[i]), scale=0.5)
        for k, v in e["components"][key].items():
            acc[k] = acc.get(k, 0.0) + np.array(v) / n
    return acc, r.emission_nm[r.excitations_nm[0]]


def test_real_collagen_autofluorescence_grows_with_thickness(real_tissue):
    exp, r = real_tissue
    thin, em = _strip_components(exp, r, 0.0, 0.5)
    thick, _ = _strip_components(exp, r, 1.0, 1.6)
    k = "dye:collagen@collagen"
    assert thick[k].sum() > 1.5 * thin[k].sum()


def test_real_buried_tissue_spectrum_distorted_by_cover(real_tissue):
    exp, r = real_tissue
    thin, em = _strip_components(exp, r, 0.0, 0.5)
    thick, _ = _strip_components(exp, r, 1.0, 1.6)
    blue, red = em < 480, (em >= 520) & (em < 640)

    def ratio(c):
        t = sum(v for k, v in c.items() if k.endswith("@tissue"))
        return t[blue].sum() / t[red].sum()

    rt, rk = ratio(thin), ratio(thick)
    assert abs(rk / rt - 1.0) > 0.03, (rt, rk)


@pytest.mark.parametrize("th", [6.0, 3.0, 1.0])
def test_real_block_matches_1d_slab(th):
    """Bottom-boundary consistency (turbid fluorescent single-layer slab, oblique lamp, secondary emission off
    in both engines): the block engine (Fresnel bottom face, transmitted light lost) and the 1-D engine
    (n_outside below the slab) agree within 6 % on the band-integrated fluorescence.  Thicker slabs (~20 mm)
    deviate by up to ~20 % in the red tail: the 1-D engine assumes laterally uniform irradiance, the 3D block
    sees the finite ~10 mm lamp spot and lets the diffuse light escape sideways (a real 3D effect)."""
    _real()
    from spectraforge.studio.schema import BlockLayer, BoxShape, LayeredBlockShape, ThicknessProfile, Transform
    base = _quiet(_only_step(PRESETS["layered_wedge_phantom"](), 490.0))
    tg = base.acquisition.physics.model_copy(update={"photobleaching": False, "specular_glints": False,
                                                     "secondary_emission": False})
    mat = "fluorescein_agarose_layer"

    def build(shape, y):
        o = base.objects[0].model_copy(update={"shape": shape, "material_id": mat,
                                               "transform": Transform(position_mm=(0, y, 0), rotation_deg=(0, 0, 0))})
        e = _small(base.model_copy(update={"objects": [o]}), photons=2_000_000, nodes=20.0, cell=1.5)
        vs = e.acquisition.volume_mc.model_copy(update={"preview_fraction": 1.0})     # full photons at scale 0.5
        return e.model_copy(update={"acquisition": e.acquisition.model_copy(update={"physics": tg, "volume_mc": vs})})

    blk = LayeredBlockShape(footprint_mm=(60, 60), layers=[BlockLayer(
        name="slab", material_id=mat, thickness=ThicknessProfile(thickness_mm=th))])
    # the slab must stand ABOVE the stage plane (y = 0): a slab straddling it makes the 1-D tracer measure its
    # thickness only down to the plane (th/2) -- that, not the bottom boundary, was the 25-50 % 'disagreement'
    rb = capture.capture(build(blk, 0.5), scale=0.5)
    rx = capture.capture(build(BoxShape(size_mm=(60, th, 60)), 0.5 + th / 2), scale=0.5)
    ex = rb.excitations_nm[0]
    H, W = rb.radiance[ex].shape[:2]
    sl = (slice(H // 2 - 6, H // 2 + 6), slice(W // 2 - 6, W // 2 + 6))
    b, x = rb.radiance[ex][sl].mean(axis=(0, 1)), rx.radiance[ex][sl].mean(axis=(0, 1))
    sel = (rb.emission_nm[ex] >= 500) & (rb.emission_nm[ex] <= 620) & (x > 1e-3 * x.max())
    ratio = float(b[sel].sum() / x[sel].sum())
    print(f"\n{th:g} mm slab: block / 1-D band-integrated fluorescence = {ratio:.4f}")
    assert ratio == pytest.approx(1.0, abs=0.06)


# ============================================================================================
# v1.1 QA: face tally -> spectrum conversion must not multiply by the channel power again
# ============================================================================================
def _film_exp(power_mW: float, thickness=0.2, dye_uM=3.0, photons=300_000):
    """A thin, clear (mu_s = 0), index-1 fluorescein film: every number is hand-computable.
    Emission is isotropic from a layer that does not attenuate it, so the radiance towards the camera
    (angle theta_c from the film normal) above an exit cell is  L = S_col / (4 pi A_cell cos theta_c)."""
    from spectraforge.studio.schema import (BlockLayer, DyeAmount, LayeredBlockShape, MaterialSpec,
                                            ThicknessProfile)
    base = _quiet(_only_step(PRESETS["layered_wedge_phantom"](), 490.0))
    mat = MaterialSpec(id="clear_film", name="clear film", dyes=[DyeAmount(fluorophore="fluorescein",
                                                                          concentration_uM=dye_uM)],
                       refractive_index=1.0)
    shape = LayeredBlockShape(footprint_mm=(20.0, 20.0), layers=[
        BlockLayer(name="film", material_id="clear_film", thickness=ThicknessProfile(thickness_mm=thickness))])
    obj = base.objects[0].model_copy(update={"shape": shape, "material_id": "clear_film"})
    src = base.source()
    src = src.model_copy(update={"sequence": [s.model_copy(update={"power_mW": power_mW}) for s in src.sequence]})
    tg = base.acquisition.physics.model_copy(update={"photobleaching": False, "specular_glints": False,
                                                     "raman": False, "elastic_leak": False})
    vs = base.acquisition.volume_mc.model_copy(update={
        "excitation_photons": photons, "emission_photons": photons, "preview_fraction": 1.0,
        "lateral_cell_mm": 1.0, "exit_cell_mm": 1.0, "sublayers": 2, "emission_node_step_nm": 30.0,
        "acceptance_half_angle_deg": 8.0})
    acq = base.acquisition.model_copy(update={"physics": tg, "volume_mc": vs})
    return base.model_copy(update={"materials": list(base.materials) + [mat], "objects": [obj], "sources": [src],
                                   "acquisition": acq})


def _last_run():
    return list(block_capture._CACHE.values())[-1][1][0]


def test_thin_film_radiance_matches_hand_computation_and_scales_linearly_with_power():
    _real()
    from spectraforge.studio.contracts import SPECTRAL_GRID_NM as GRID
    out = {}
    for P in (2.0, 4.0):
        block_capture.clear_cache()
        res = capture.capture(_film_exp(P), scale=0.2)
        run = _last_run()
        ex = run.ex_list[0]
        (key, src), = [(k, v) for k, v in run.sources.items()]
        fg = run.faces["top"]
        nu, nv = len(fg.u_edges_mm) - 1, len(fg.v_edges_mm) - 1
        # sources and exit cells share the 1 mm grid: sum the 5 x 5 columns around the beam centre
        sl = (slice(nu // 2 - 2, nu // 2 + 3), slice(nv // 2 - 2, nv // 2 + 3))
        S_col = src.sum(axis=(2, 3))[sl]
        cosc = float(run.camera_dir_local[1])
        area = float(np.diff(fg.u_edges_mm)[0] * np.diff(fg.v_edges_mm)[0])
        T = run.rad["top"][:, :, :, 0][sl]                                  # (5, 5, Ne) mW/(mm2 sr)
        T_mid = T[:, :, T.shape[2] // 2]
        expect = S_col / (4 * np.pi * area * cosc)
        ratio = T_mid.sum() / expect.sum()
        # spectrum from the face tally: integral of e_k(lambda) is 1 => band-integrated radiance == T
        u = np.array([0.0]) + 0.01
        comps = run.components(ex, np.zeros(1, np.int8), u, u, np.arange(len(GRID)), np.zeros(len(GRID)),
                               np.zeros(len(GRID)), True, True, False)
        (name, spec, is_fl), = comps
        integ = float(spec[0].sum() * (GRID[1] - GRID[0]))
        Tc = float(_interp_at_centre(run, 0))
        print(f"\nP={P}: sum T / hand = {ratio:.4f}   band integral {integ:.4e} vs T {Tc:.4e}")
        H, W = res.radiance[ex].shape[:2]
        pix = float(res.radiance[ex][H // 2 - 1:H // 2 + 2, W // 2 - 1:W // 2 + 2].sum())
        out[P] = (ratio, integ, run.launch_power[ex], float(src.sum()), pix)
        assert name == "dye:fluorescein@film" and is_fl
        assert ratio == pytest.approx(1.0, abs=0.05)
        assert integ == pytest.approx(Tc, rel=0.03)
    # linear in the source power (a second multiplication by the channel power would give x4)
    assert out[4.0][2] / out[2.0][2] == pytest.approx(2.0, rel=1e-6)          # launched power
    assert out[4.0][3] / out[2.0][3] == pytest.approx(2.0, rel=0.02)          # emitted power
    assert out[4.0][1] / out[2.0][1] == pytest.approx(2.0, rel=0.05)          # band-integrated radiance
    assert out[4.0][4] / out[2.0][4] == pytest.approx(2.0, rel=1e-6)          # captured pixel radiance


def _interp_at_centre(run, ch):
    """Face tally of channel ``ch`` at the film centre, linearly interpolated in wavelength to the band mean."""
    fg = run.faces["top"]
    from spectraforge.studio.block_capture import _bilinear
    t = _bilinear(run.rad["top"][:, :, :, ch], fg.u_edges_mm, fg.v_edges_mm, np.array([0.01]), np.array([0.01]))[0]
    e = run.em_density[ch].astype(float)
    from spectraforge.studio.contracts import SPECTRAL_GRID_NM as GRID
    return float(np.sum(np.interp(GRID, run.nodes, t) * e) * (GRID[1] - GRID[0]))


# ============================================================================================
# v1.1 QA: adaptive emission nodes
# ============================================================================================
def _bump_arrays(lam):
    """Optics of a two-layer block vs wavelength: a hemoglobin-like absorption bump at 545 nm in the cover."""
    lam = np.asarray(lam, float)
    bump = 1.5 * np.exp(-0.5 * ((lam - 545.0) / 10.0) ** 2)
    mua = np.stack([0.08 + 0.1 * (lam - 450.0) / 200.0, 0.05 + bump])
    mus = np.stack([9.0 * (lam / 500.0) ** -1.2, 12.0 * (lam / 500.0) ** -1.2])
    return mua, mus


def _bump_optics(lam: float):
    from spectraforge.studio.contracts import RegionOptics
    mua, mus = _bump_arrays(lam)
    return RegionOptics(float(lam), mua, mus, np.array([0.85, 0.85]), np.array([1.38, 1.4]))


def test_choose_emission_nodes_refines_only_where_the_transport_bends():
    from spectraforge.studio.contracts import SPECTRAL_GRID_NM as GRID
    f = np.exp(-(0.4 + 0.5 * np.exp(-0.5 * ((GRID - 545.0) / 10.0) ** 2)) * 2.0)
    uniform = int(np.ceil((700 - 450) / 20.0)) + 1
    nodes = block_capture.choose_emission_nodes(f, 450.0, 700.0, 20.0)
    assert nodes[0] == 450.0 and nodes[-1] == 700.0 and np.all(np.diff(nodes) > 0)
    d = np.diff(nodes)
    assert len(nodes) <= uniform                                       # never more nodes than the uniform grid
    assert d.max() <= 40.0 + 1e-9 and d.min() >= 2.5 - 1e-9
    near = (nodes > 515) & (nodes < 575)
    assert near.sum() >= 6                                              # resolved at the bump ...
    assert np.diff(nodes[nodes >= 575]).min() >= 30.0                  # ... coarse away from it
    # with enough budget the 3 % interpolation criterion is met everywhere
    many = block_capture.choose_emission_nodes(f, 450.0, 700.0, 20.0, budget=60)
    err = np.abs(np.interp(GRID, many, np.interp(many, GRID, f)) - f) / np.maximum(f, 0.05 * f.max())
    assert err[(GRID >= 450) & (GRID <= 700)].max() < 0.031
    # and the placement beats the uniform grid of the same size
    uni = np.linspace(450.0, 700.0, uniform)
    err_u = np.abs(np.interp(GRID, uni, np.interp(uni, GRID, f)) - f) / np.maximum(f, 0.05 * f.max())
    err_a = np.abs(np.interp(GRID, nodes, np.interp(nodes, GRID, f)) - f) / np.maximum(f, 0.05 * f.max())
    m = (GRID >= 450) & (GRID <= 700)
    assert err_a[m].max() < err_u[m].max()
    flat = block_capture.choose_emission_nodes(np.ones_like(GRID), 450.0, 700.0, 20.0)
    assert len(flat) <= 9 < uniform and np.diff(flat).max() <= 40.0 + 1e-9   # nothing to resolve -> coarsest allowed grid


@pytest.mark.slow
def test_adaptive_nodes_interpolation_error_vs_dense_monte_carlo():
    """The Monte-Carlo transport factor interpolated from the adaptive nodes matches a 10 nm reference."""
    t3 = _real()
    from spectraforge.studio import transport3d as T3
    from spectraforge.studio.contracts import BlockGeometry, SPECTRAL_GRID_NM as GRID
    from spectraforge.studio.schema import ThicknessProfile, VolumeMCSettings
    planes = T3.planes_from_layers((12.0, 12.0), [ThicknessProfile(thickness_mm=1.5),
                                                 ThicknessProfile(kind="linear", thickness_mm=1.0,
                                                                  end_thickness_mm=0.2, axis="x")])
    G = BlockGeometry(0, (12.0, 12.0), ["fl", "cover"], [0, 1], planes, np.eye(4))
    s = VolumeMCSettings(lateral_cell_mm=12.0, exit_cell_mm=12.0, sublayers=3, acceptance_half_angle_deg=12.0)
    grid = T3.make_voxel_grid(G, s)
    src = np.zeros((1,) + grid.shape)
    src[0, :, :, 0, :] = 1.0
    mua, mus = _bump_arrays(GRID)
    mua = np.maximum(mua, 0.0)
    f = np.exp(-np.sum(np.sqrt(3 * mua * (mua + 0.15 * mus)) * np.array([[1.5], [0.6]]), axis=0))
    nodes = block_capture.choose_emission_nodes(f, 450.0, 700.0, 20.0, budget=24)
    dense = np.arange(450.0, 700.1, 10.0)
    lam_all = np.unique(np.concatenate([dense, nodes]))
    em = T3.emission_pass(G, [_bump_optics(l) for l in lam_all], src, ["a"], [0, 1, 0], s, n_photons=300_000, seed=3)
    T_all = em.radiance["top"][0, 0, :, 0]
    T_dense = np.interp(dense, lam_all, T_all)
    approx = np.interp(dense, nodes, np.interp(nodes, lam_all, T_all))
    rel = np.abs(approx - T_dense) / T_dense
    print(f"\nadaptive nodes: {len(nodes)} nodes vs {len(dense)} at 10 nm -> max interpolation error {rel.max():.4f}, "
          f"rms {np.sqrt(np.mean(rel ** 2)):.4f}; uniform 20 nm would use {len(np.arange(450, 700.1, 20))}")
    assert rel.max() < 0.05          # 3 % interpolation criterion + Monte-Carlo noise of the reference


# ============================================================================================
# v1.1 QA: explain / ROI components of block pixels reuse the cached 3D Monte Carlo
# ============================================================================================
def _block_pixels(r, n):
    gt = r.ground_truth["layer_thickness:cover"]
    ys, xs = np.nonzero(np.isfinite(gt))
    pick = np.linspace(0, ys.size - 1, n).astype(int)
    return [(int(xs[i]), int(ys[i])) for i in pick]


def test_explain_after_preview_reuses_the_cached_block_run(fake3d):
    """The studio explains the previewed excitation of the FULL experiment: no new Monte Carlo is run."""
    import time
    exp = _wedge(3)
    sc = exp.acquisition.preview_scale
    r = capture.preview(exp, scale=sc)
    n = dict(fake3d.calls)
    assert n["exc"] == 1
    (x, y), = _block_pixels(r, 1)
    t0 = time.perf_counter()
    e = capture.explain_pixel(exp, x, y, scale=sc)
    dt = time.perf_counter() - t0
    assert fake3d.calls == n, "explain re-ran the block Monte Carlo"
    assert list(e["components"]) == [f"{r.excitations_nm[0]:g}"]
    key = f"{r.excitations_nm[0]:g}"
    tot = np.sum([np.array(v) for v in e["components"][key].values()], axis=0)
    assert np.allclose(tot, r.radiance[r.excitations_nm[0]][y, x], rtol=0.01, atol=1e-12)
    assert dt < 1.0                                                   # ~30 ms unloaded (budget 0.3 s)
    # an explicit excitation list also works, and a second pixel is as cheap
    e2 = capture.explain_pixel(exp, x, y, scale=sc, excitations=[r.excitations_nm[0]])
    assert e2["radiance"] == e["radiance"] and fake3d.calls == n


def test_roi_components_of_64_block_pixels_need_no_new_monte_carlo(fake3d):
    import time
    from spectraforge.studio import analysis
    exp = _wedge(2)
    sc = exp.acquisition.preview_scale
    r = capture.preview(exp, scale=sc)
    n = dict(fake3d.calls)
    px = _block_pixels(r, 64)
    t0, c0 = time.perf_counter(), time.process_time()
    comps = [capture.explain_pixel(exp, x, y, scale=sc)["components"] for x, y in px]
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    assert fake3d.calls == n
    assert all(len(c) == 1 for c in comps)
    print(f"\n64 block pixels explained: {wall:.2f} s wall, {cpu:.2f} s cpu")
    assert cpu < 10.0 and wall < 20.0                                  # ~1-2 s unloaded (budget 2 s)


def test_cached_runs_are_not_mutated_by_pixel_lookups(fake3d):
    """explain_pixel attaches a 1-pixel look-up to the engine's COPY of the shared run, so a capture that
    follows still sees its own full-frame look-up and gives identical results."""
    exp = _wedge(1)
    sc = exp.acquisition.preview_scale
    a = capture.preview(exp, scale=sc)
    (x, y), = _block_pixels(a, 1)
    capture.explain_pixel(exp, x, y, scale=sc)
    b = capture.preview(exp, scale=sc)
    ex = a.excitations_nm[0]
    assert np.array_equal(a.radiance[ex], b.radiance[ex])
    run = block_capture._CACHE_LOCK and list(block_capture._CACHE.values())[-1][1][0]
    assert run.mask is None                                            # the cached object was never attached to


def test_block_cache_entries_cover_the_union_of_requested_blocks(fake3d):
    exp = _wedge(1)
    sc = exp.acquisition.preview_scale
    a = capture.preview(exp, scale=sc)
    (x, y), = _block_pixels(a, 1)
    capture.explain_pixel(exp, x, y, scale=sc)                        # pixel covers block 0 only
    assert block_capture.cached_excitations(exp, "sample", exp.acquisition.volume_mc.preview_fraction) == [a.excitations_nm[0]]
    other = exp.model_copy(update={"name": "renamed"})                 # the name does not change the physics
    assert block_capture.cached_excitations(other, "sample", exp.acquisition.volume_mc.preview_fraction) is not None
    assert block_capture.cached_excitations(exp, "dark", 0.1) is None


def test_real_wedge_first_band_is_elastic_contaminated_but_second_band_is_clean(real_wedge):
    """Lead's QA question (ex 490): in the first band behind the line (510 nm) the elastic leak competes with the
    fluorescein; at ex+40 nm the fluorescence is >99 % of the signal and, per unit surface irradiance, falls
    monotonically with the local cover thickness."""
    exp, r = real_wedge
    ex = r.excitations_nm[0]
    gt = r.ground_truth["layer_thickness:cover"]
    ys, xs = np.nonzero(np.isfinite(gt))
    k = len(ys) // 2
    e = capture.explain_pixel(exp, int(xs[k]), int(ys[k]), scale=0.5)
    comp = e["components"][f"{ex:g}"]
    em = np.array(e["emission_nm"][f"{ex:g}"])
    b1, b2 = int(np.argmin(abs(em - (ex + 20)))), int(np.argmin(abs(em - (ex + 40))))
    fl = np.array(comp["dye:fluorescein@fluorescent layer"])
    el = np.array(comp["elastic"]) + np.array(comp.get("leak_source", np.zeros_like(fl)))
    assert el[b1] > 0.3 * fl[b1]                       # first band: elastic is comparable
    assert fl[b2] > 100 * el[b2]                       # second band: fluorescence only
    band = int(np.argmin(abs(em - (ex + 40))))
    v = _roi_mean(r, ex, band, [(1.4, 1.7), (1.1, 1.4), (0.8, 1.1), (0.5, 0.8), (0.3, 0.5)], per_irradiance=True)
    assert np.all(np.diff(v) > 0), v


# ============================================================================================
# ground truth with several blocks sharing a layer name
# ============================================================================================
def test_gt_layer_thickness_merged_across_blocks_with_shared_layer_name(fake3d):
    from spectraforge.studio.schema import BlockLayer, LayeredBlockShape, ThicknessProfile, Transform
    exp = _wedge(1)
    base = exp.objects[0]
    ths = (1.0, 2.0, 3.0)
    xs = (-14.0, 0.0, 14.0)
    objs = []
    for i, (th, x) in enumerate(zip(ths, xs)):
        shp = LayeredBlockShape(footprint_mm=(10, 10), layers=[
            BlockLayer(name="cover", material_id="clear_cover", thickness=ThicknessProfile(thickness_mm=th)),
            BlockLayer(name="fl", material_id="tissue_substrate", thickness=ThicknessProfile(thickness_mm=1.0))])
        objs.append(base.model_copy(update={"id": f"blk{i}", "name": f"block{i}", "shape": shp,
                                            "transform": Transform(position_mm=(x, base.transform.position_mm[1], 0.0),
                                                                   rotation_deg=(0, 0, 0))}))
    exp = exp.model_copy(update={"objects": objs})
    r = capture.capture(exp, scale=0.25)
    gt = r.ground_truth
    merged = gt["layer_thickness:cover"]
    h = trace_scene(exp, r.width, r.height)
    seen = set()
    for bi, th in enumerate(ths):
        q = gt[f"layer_thickness:block{bi}/cover"]
        own = np.isfinite(q)
        assert own.sum() > 0
        assert np.allclose(q[own], th, atol=1e-4)
        assert np.allclose(merged[own], th, atol=1e-4)            # every block's pixels survive in the merged map
        seen.add(bi)
    assert seen == {0, 1, 2}
    own_total = sum(int(np.isfinite(gt[f"layer_thickness:block{bi}/cover"]).sum()) for bi in range(3))
    assert int(np.isfinite(merged).sum()) == own_total             # disjoint blocks: nothing lost, nothing added
    assert np.isclose(np.nanmin(merged), 1.0, atol=1e-4) and np.isclose(np.nanmax(merged), 3.0, atol=1e-4)
    # explain_pixel at a pixel of the 3rd block reports that block's thickness
    q = gt["layer_thickness:block2/cover"]
    yy, xx = np.argwhere(np.isfinite(q))[0]
    ex = capture.explain_pixel(exp, int(xx), int(yy), scale=0.25)
    assert ex["ground_truth"]["layer_thickness:cover"] == pytest.approx(3.0, abs=1e-4)
