"""Capture-level behaviour of block inclusions with the REAL 3D engine (SpectralForge Studio v1.2):
ground-truth maps, explain components, bleaching / FRET / clear inclusions, metadata, presets."""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge.studio import block_capture, capture
from spectraforge.studio.presets import PRESETS
from spectraforge.studio.schema import Inclusion

BEAD = dict(name="bead", kind="sphere", center_mm=(-5.0, 2.5, 0.0), radius_mm=2.0)


def _quiet(exp):
    tg = exp.acquisition.physics.model_copy(update={"power_drift": False, "shot_noise": False, "read_noise": False,
                                                    "dark_current": False, "fixed_pattern_noise": False,
                                                    "photobleaching": False})
    return exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"physics": tg})})


def _setup(material="fluorescein_agarose_layer", wl=490.0, photons=250_000, **inc_kw):
    exp = PRESETS["layered_wedge_phantom"]()
    sh = exp.objects[0].shape
    inc = Inclusion(material_id=material, **{**BEAD, **inc_kw})
    sh = sh.model_copy(update={"inclusions": [inc]})
    exp = exp.model_copy(update={"objects": [exp.objects[0].model_copy(update={"shape": sh})]})
    src = exp.source()
    seq = [s for s in src.sequence if s.enabled and abs(s.wavelength_nm - wl) < 1e-6]
    exp = exp.model_copy(update={"sources": [src.model_copy(update={"sequence": seq}) if s.id == src.id else s
                                             for s in exp.sources]})
    vs = exp.acquisition.volume_mc.model_copy(update={
        "excitation_photons": photons, "emission_photons": photons, "emission_node_step_nm": 40.0,
        "lateral_cell_mm": 1.0, "exit_cell_mm": 1.0, "sublayers": 4, "inclusion_voxel_mm": 0.25})
    return _quiet(exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"volume_mc": vs})}))


@pytest.fixture(scope="module")
def res():
    block_capture.clear_cache()
    exp = _setup()
    return exp, capture.capture(exp, scale=0.5)


def test_ground_truth_maps_of_an_inclusion(res):
    exp, r = res
    gt = r.ground_truth
    for k in ("inclusion_mask:bead", "inclusion_depth_mm:bead", "origin_fraction:bead", "origin_layer_fraction:cover"):
        assert k in gt
    mask = gt["inclusion_mask:bead"]
    block = np.isfinite(gt["layer_thickness:cover"])
    assert np.all(np.isnan(mask[~block])) and np.isnan(gt["origin_fraction:bead"][~block]).all()
    assert set(np.unique(mask[block])) <= {0.0, 1.0}
    n_in = int((mask == 1).sum())
    area_px = block.sum()
    # footprint area fraction of the 40 x 40 block: pi r^2 / 1600 = 0.785 %
    assert n_in > 0 and n_in / area_px == pytest.approx(np.pi * 4.0 / 1600.0, rel=0.35)
    depth = gt["inclusion_depth_mm:bead"]
    assert np.isnan(depth[mask != 1]).all() and np.isfinite(depth[mask == 1]).all()
    # the top of the bead is at y = 4.5; the surface above x = -5 is 4.5 + 1.25 = 5.75 (wedge cover 2 -> 0 along x)
    ys, xs = np.nonzero(mask == 1)
    assert depth[mask == 1].min() >= 1.0 and depth[mask == 1].max() < 4.0
    of = gt["origin_fraction:bead"]
    assert np.nanmax(of) > 0.2 and np.nanmin(of) >= 0.0 and np.nanmax(of) <= 1.0 + 1e-9
    # the bead is the only emitter of fluorescein next to the 0.5 mm film -> origin fraction highest over the bead
    assert np.nanmean(of[mask == 1]) > 2.0 * np.nanmean(of[(mask == 0) & block])


def test_blocks_entry_and_volumes_describe_the_inclusion(res):
    exp, r = res
    b = r.blocks[0]
    assert b["inclusions"] and b["inclusions"][0]["name"] == "bead" and b["inclusions"][0]["kind"] == "sphere"
    assert b["inclusions"][0]["center_mm"] == [-5.0, 2.5, 0.0] and b["inclusions"][0]["region"] == 1
    v = r.volumes[b["object_id"]]
    inc = v["inclusions"]
    assert inc["names"] == ["bead"] and inc["region"].dtype == np.uint8 and inc["shape"] == list(inc["region"].shape)
    ex = r.excitations_nm[0]
    assert inc["fluence"][ex].shape == inc["region"].shape and inc["absorbed"][ex].sum() > 0
    ((key, src),) = [(k, a) for k, a in inc["source"].items() if k[1] == "bead|fluorescein"]
    assert key[1] == "bead|fluorescein" and src.shape == inc["region"].shape and src.sum() > 0
    assert np.all(src[inc["region"] == 0] == 0)
    # layer sources of the fluorescent layer are present too, and the layer grid keeps its v1.1 shape
    assert any(k[1] == "fluorescent layer|fluorescein" for k in v["source"])


def test_explain_pixel_has_the_inclusion_component_and_sums(res):
    exp, r = res
    ex = r.excitations_nm[0]
    mask = r.ground_truth["inclusion_mask:bead"]
    ys, xs = np.nonzero(mask == 1)
    k = len(ys) // 2
    e = capture.explain_pixel(exp, int(xs[k]), int(ys[k]), scale=0.5)
    comp = e["components"][f"{ex:g}"]
    assert "dye:fluorescein@bead" in comp and "dye:fluorescein@fluorescent layer" in comp
    tot = np.sum([np.array(v) for v in comp.values()], axis=0)
    assert np.allclose(tot, e["radiance"][f"{ex:g}"], rtol=1e-3, atol=1e-12)
    assert np.allclose(e["radiance"][f"{ex:g}"], r.radiance[ex][ys[k], xs[k]], rtol=0.02, atol=1e-12)
    assert e["ground_truth"]["inclusion_mask:bead"] == 1.0
    assert 1.0 <= e["ground_truth"]["inclusion_depth_mm:bead"] < 4.0
    assert e["ground_truth"]["origin_fraction:bead"] > 0.0
    bead_sig = np.array(comp["dye:fluorescein@bead"]).sum()
    film_sig = np.array(comp["dye:fluorescein@fluorescent layer"]).sum()
    assert bead_sig > 0 and film_sig > 0


def test_bead_signal_is_visible_in_the_image(res):
    exp, r = res
    ex = r.excitations_nm[0]
    mask = r.ground_truth["inclusion_mask:bead"]
    band = int(np.argmin(np.abs(r.emission_nm[ex] - (ex + 40.0))))
    img = r.radiance[ex][..., band]
    block = np.isfinite(r.ground_truth["layer_thickness:cover"])
    on = img[mask == 1].mean()
    # a ring of block pixels at the same cover thickness (same x), away from the bead
    ys, xs = np.nonzero(mask == 1)
    off = img[(mask == 0) & block & (np.abs(np.arange(img.shape[1])[None, :] - xs.mean()) < 3)
              & (np.abs(np.arange(img.shape[0])[:, None] - ys.mean()) > 10)]
    assert np.isfinite(img[block]).all() and off.size > 5
    assert on > 1.15 * off.mean(), (on, off.mean())            # the bead adds fluorescein under the same cover


def test_photobleaching_applies_to_inclusion_dyes():
    block_capture.clear_cache()
    exp = _setup(wl=470.0, material="fret_egfp_mcherry", photons=150_000)
    sq = exp.source().sequence
    seq = [sq[0].model_copy(update={"power_mW": 400.0, "exposure_ms": 200_000.0})] * 1
    src = exp.source().model_copy(update={"sequence": seq * 2})
    exp = exp.model_copy(update={"sources": [src]})
    on = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={
        "physics": exp.acquisition.physics.model_copy(update={"photobleaching": True})})})
    r_off = capture.capture(exp, scale=0.5)
    block_capture.clear_cache()
    r_on = capture.capture(on, scale=0.5)
    mask = r_on.ground_truth["inclusion_mask:bead"]
    assert (mask == 1).any()
    ex = r_on.excitations_nm[0]
    a, b = r_off.radiance[ex][mask == 1].sum(), r_on.radiance[ex][mask == 1].sum()
    assert b < 0.9 * a                                            # bleached inside the bead


def test_fret_in_a_clear_inclusion():
    """FRET pair EGFP -> mCherry in a (clear, mu_s = 0) inclusion: FRET moves emission from the donor to the acceptor."""
    block_capture.clear_cache()
    exp = _setup(wl=470.0, material="fret_egfp_mcherry", photons=150_000)
    ex_on = exp
    tg = exp.acquisition.physics.model_copy(update={"fret": False})
    ex_off = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"physics": tg})})
    runs = {}
    for name, e in (("on", ex_on), ("off", ex_off)):
        block_capture.clear_cache()
        r = capture.capture(e, scale=0.5)
        mask = r.ground_truth["inclusion_mask:bead"]
        ys, xs = np.nonzero(mask == 1)
        k = len(ys) // 2
        em = capture.explain_pixel(e, int(xs[k]), int(ys[k]), scale=0.5)
        runs[name] = em["components"][f"{r.excitations_nm[0]:g}"]
    don = lambda c: np.array(c["dye:EGFP@bead"]).sum()
    acc = lambda c: np.array(c["dye:mCherry@bead"]).sum()
    assert don(runs["on"]) < 0.95 * don(runs["off"])
    assert acc(runs["on"]) > 1.05 * acc(runs["off"])


@pytest.mark.parametrize("name", ["mucosa_dysplasia_inclusion", "vessel_under_collagen", "skin_three_layer"])
def test_tissue_presets_capture_with_the_inclusion_engine(name):
    block_capture.clear_cache()
    exp = _quiet(PRESETS[name]())
    vs = exp.acquisition.volume_mc.model_copy(update={
        "excitation_photons": 80_000, "emission_photons": 80_000, "emission_node_step_nm": 60.0,
        "lateral_cell_mm": 1.0, "exit_cell_mm": 1.0, "sublayers": 4})
    exp = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"volume_mc": vs})})
    src = exp.source()
    exp = exp.model_copy(update={"sources": [src.model_copy(update={"sequence": [s for s in src.sequence if s.enabled][:2]})
                                             if s.id == src.id else s for s in exp.sources]})
    r = capture.capture(exp, scale=0.25)
    for ex in r.excitations_nm:
        assert np.isfinite(r.adu[ex]).all() and np.isfinite(r.radiance[ex]).all() and r.radiance[ex].max() > 0
    sh = exp.objects[0].shape
    assert [i["name"] for i in r.blocks[0]["inclusions"]] == [i.name for i in sh.inclusions]
    for i in sh.inclusions:
        assert f"inclusion_mask:{i.name}" in r.ground_truth
