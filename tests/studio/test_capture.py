"""Capture orchestration tests (stub transport; integration tests auto-skip until transport exists)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from spectraforge.studio import capture as cap
from spectraforge.studio import optics as optics_mod
from spectraforge.studio import presets
from spectraforge.studio.contracts import SPECTRAL_GRID_NM as GRID
from spectraforge.studio.contracts import TransportKernels
from spectraforge.studio.schema import (
    Background,
    BoxShape,
    DyeAmount,
    ExcitationStep,
    Experiment,
    FretPair,
    LightSourceSpec,
    MaterialSpec,
    PhysicsToggles,
    RoomLight,
    ScatteringSpec,
    SceneObject,
    SensorSpec,
    SpectralFilterSpec,
    Transform,
)


# --------------------------------------------------------------------------------------------
# stub transport: clear-medium closed form (contracts.py) and a crude turbid semi-infinite model
# --------------------------------------------------------------------------------------------
def _fresnel_T(c1, n1, n2):
    s2 = (n1 / n2) ** 2 * (1 - c1 ** 2)
    c2 = np.sqrt(np.clip(1 - s2, 0, None))
    rs = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)) ** 2
    rp = ((n1 * c2 - n2 * c1) / (n1 * c2 + n2 * c1)) ** 2
    return 1 - 0.5 * (rs + rp), c2


class StubTransport:
    """Minimal ``TransportModel``: closed-form clear medium; turbid = crude semi-infinite."""

    def __init__(self):
        self.calls = 0

    def kernels(self, optics, ex_nm, cos_in, cos_out, thickness_mm, n_outside=1.0, path_mm=None):
        self.calls += 1
        n = optics.n
        cos_in = np.asarray(cos_in, float)
        cos_out = np.asarray(cos_out, float)
        n_out = np.broadcast_to(np.asarray(n_outside, float), cos_in.shape)
        T_in, ci = _fresnel_T(np.clip(cos_in, 1e-3, 1), n_out, n)
        T_out, co = _fresnel_T(np.clip(cos_out, 1e-3, 1), n_out, n)
        ci = np.maximum(ci, 0.05)
        co = np.maximum(co, 0.05)
        W = optics.wavelengths_nm
        mut = optics.mua_per_mm + optics.mus_per_mm
        mut_ex = float(np.interp(ex_nm, W, mut))
        kappa = mut_ex * co[:, None] / ci[:, None] + mut[None, :]
        if optics.clear:
            L = (np.asarray(thickness_mm, float) / co) if path_mm is None else np.asarray(path_mm, float)
            L = np.where(np.isfinite(L), L, 1e6)
            shape = -np.expm1(-kappa * L[:, None]) / kappa
            rho = np.zeros(len(cos_in))
        else:
            shape = 1.0 / kappa
            mus = float(np.interp(ex_nm, W, optics.mus_per_mm))
            mua = float(np.interp(ex_nm, W, optics.mua_per_mm))
            rho = np.full(len(cos_in), 0.3 * mus / (mus + mua + 1e-12) / np.pi)
        G = (T_in * T_out / (4 * np.pi * n ** 2 * ci))[:, None] * shape
        return TransportKernels(fluorescence_G=G, elastic_rho=rho)


# --------------------------------------------------------------------------------------------
# helpers: a minimal flat-slab experiment
# --------------------------------------------------------------------------------------------
def slab_experiment(dyes, *, conc=1.0, steps=((365, 5.0, 100.0), (405, 5.0, 100.0)), toggles=None,
                    extra_mat=None, sensor=None, filt=None, source=None, objects_mat="slab",
                    clear=True, room=None, fret=()):
    mats = presets.builtin_materials()
    sc = ScatteringSpec() if clear else ScatteringSpec(model="power_law", mus_prime_500_per_mm=10.0,
                                                       power_b=1.0, g=0.8)
    pairs = [(d, conc) if isinstance(d, str) else d for d in dyes]
    slab = MaterialSpec(id="slab", name="slab", dyes=[DyeAmount(fluorophore=d, concentration_uM=c) for d, c in pairs],
                        refractive_index=1.33, scattering=sc, surface_roughness=0.0, fret=list(fret))
    mats = mats + [slab] + ([extra_mat] if extra_mat else [])
    cam = presets.nuance_camera(position=(0.0, 300.0, 0.0))
    if sensor is not None:
        cam = cam.model_copy(update={"sensor": sensor})
    if filt is not None:
        cam = cam.model_copy(update={"filter": filt})
    src = source or presets.tunable_lamp(sequence=list(steps))
    src = src.model_copy(update={"power_drift_pct": 0.0})
    exp = Experiment(
        materials=mats,
        objects=[SceneObject(id="o", name="slab", shape=BoxShape(size_mm=(30.0, 4.0, 30.0)),
                             material_id=objects_mat, transform=Transform(position_mm=(0.0, 2.0, 0.0)))],
        background=Background(kind="plane", material_id="black_anodized",
                              room_light=room or RoomLight()),
        sources=[src], cameras=[cam], active_source_id=src.id, active_camera_id=cam.id)
    tg = toggles or PhysicsToggles()
    exp = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"physics": tg})})
    return exp


def no_noise():
    return PhysicsToggles(shot_noise=False, read_noise=False, dark_current=False, fixed_pattern_noise=False,
                          quantization=False, saturation=False, power_drift=False, defocus=False,
                          diffraction=False, chromatic_aberration=False)


def small_sensor(**kw):
    """48x36 sensor with the same physical size as the default 348x256 Nuance sensor."""
    kw.setdefault("pixel_pitch_um", 6.45 * 348 / 48 / 4 * 4)
    return SensorSpec(width_px=48, height_px=36, binning=4, **kw)


# --------------------------------------------------------------------------------------------
# physics building blocks
# --------------------------------------------------------------------------------------------
def test_raman_position_488_is_584():
    c = cap.raman_center_nm(488.0)
    assert 583.0 < c < 586.0
    shape = cap._raman_shape(488.0)
    assert abs(GRID[np.argmax(shape)] - c) < 2.0
    assert abs(shape.sum() - 1.0) < 1e-6 and 10 < 3 * np.sqrt(((GRID - c) ** 2 * shape).sum()) < 60


def test_raman_mu_magnitude_and_lambda_scaling():
    m488 = cap.raman_mu_phi_per_mm(488.0)
    assert 1e-7 < m488 < 1e-6
    # lambda^-4 cross-section (and the energy factor) -> much stronger in the blue
    assert cap.raman_mu_phi_per_mm(405.0) > 1.8 * m488


def test_dilute_limit_matches_legacy_render():
    """Per-dye emission shape and cross-excitation amplitude ratios match spectraforge.render
    up to one global geometry constant (flat slab, normal incidence, tiny concentrations)."""
    from spectraforge.acquisition import AcquisitionConfig
    from spectraforge.library import load_builtin_library
    from spectraforge.material import Material
    from spectraforge.scene import Scene
    from spectraforge.forward import render

    names = ["EGFP", "mCherry", "DAPI"]
    conc_uM = {"EGFP": 0.02, "mCherry": 0.05, "DAPI": 0.03}
    mat = MaterialSpec(id="d", name="d", dyes=[DyeAmount(fluorophore=n, concentration_uM=c) for n, c in conc_uM.items()],
                       refractive_index=1.33)
    exp = slab_experiment([])
    narrow = exp.source().model_copy(update={"line_fwhm_nm": 1.0, "lamp_curve": "flat"})
    from spectraforge.studio.schema import FluorophoreSpec
    lib = load_builtin_library()
    # the studio presets carry literature extinction coefficients; the legacy engine uses the library's
    # relative values -> compare like with like by overriding the studio dyes with the library ones
    lib_specs = [FluorophoreSpec(name=n, ex_peak_nm=lib[n].ex_peak_nm, ex_fwhm_nm=lib[n].ex_fwhm_nm,
                                 em_peak_nm=lib[n].em_peak_nm, em_fwhm_nm=lib[n].em_fwhm_nm,
                                 quantum_yield=lib[n].quantum_yield, extinction=lib[n].extinction,
                                 em_skew=lib[n].em_skew, vibronic=lib[n].vibronic) for n in names]
    exp = exp.model_copy(update={"sources": [narrow], "fluorophores": lib_specs})
    fl = optics_mod.merged_fluorophores(exp)
    opt = optics_mod.material_optics(mat, fl, GRID)
    st = StubTransport()
    cos = np.array([1.0])
    exs = [365.0, 405.0, 450.0, 488.0, 532.0]
    src = exp.source()
    ours = {}
    for ex in exs:
        line = cap._line_shape(src, ex)
        k = st.kernels(opt, ex, cos, cos, np.array([1.0]), path_mm=np.array([1.0]))
        sp = cap.dye_spectra(opt, line, 1.0, k.fluorescence_G[0])
        ours[ex] = {n: sp[n] for n in names}       # (W,) mW/(mm^2 sr nm) per dye, E = 1
    # legacy: concentrations in the same ratio, one pixel
    lib = load_builtin_library()
    em = np.arange(300.0, 900.0 + 1e-9, 5.0)      # wide enough that no emission mass is clipped
    acq = AcquisitionConfig(excitations=exs, em_min=300.0, em_max=900.0, em_step=5.0)
    scene = Scene(1, 1)
    for n in names:
        scene.paint_rect(Material(name=n, recipe={n: conc_uM[n]}), 0, 1, 0, 1)
    _, gt = render(scene, lib, acq, seed=0)
    ratios = []
    for ex in exs:
        for n in names:
            leg = gt.per_fluorophore_spectra[n][ex]            # (n_em,), unit-sum emission x amplitude
            mine = np.interp(em, GRID, ours[ex][n]) * 5.0       # per-nm density -> per-5nm-bin sum
            peak_all = max(gt.per_fluorophore_spectra[m][e].max() for m in names for e in exs)
            if leg.max() > 0.05 * peak_all:           # skip dye/excitation pairs that barely absorb
                sel = leg > 0.02 * leg.max()
                ratios.append(mine[sel] / leg[sel])
    allr = np.concatenate(ratios)
    assert len(allr) > 20
    assert np.std(allr) / np.mean(allr) < 0.02           # one global constant, within 2 %


def test_fret_quenches_donor_and_sensitises_acceptor():
    pair = FretPair(donor="EGFP", acceptor="mCherry", k_per_uM=0.2)
    base = slab_experiment([("EGFP", 5.0), ("mCherry", 5.0)], toggles=no_noise(), sensor=small_sensor(),
                           steps=((470, 5.0, 100.0),), fret=[pair])
    off = base.model_copy(update={"acquisition": base.acquisition.model_copy(update={
        "physics": no_noise().model_copy(update={"fret": False})})})
    tr = StubTransport()
    on_res = cap.explain_pixel(base, 24, 18, scale=1.0, transport=tr)
    off_res = cap.explain_pixel(off, 24, 18, scale=1.0, transport=tr)
    k = "470"
    don_on = np.array(on_res["components"][k]["dye:EGFP"]).sum()
    don_off = np.array(off_res["components"][k]["dye:EGFP"]).sum()
    acc_on = np.array(on_res["components"][k]["dye:mCherry"]).sum()
    acc_off = np.array(off_res["components"][k]["dye:mCherry"]).sum()
    assert don_on < don_off * 0.75          # donor quenched by (1 - E), E = kc/(1+kc) = 0.5
    assert acc_on > acc_off                 # acceptor sensitised
    assert abs(don_on / don_off - 0.5) < 0.02


def test_nuance_band_start_and_angle_shift_sign():
    from spectraforge.studio import sensor as sm
    f = SpectralFilterSpec()
    assert sm.band_centres(f, 430.0)[0] == pytest.approx(450.0)
    assert sm.band_centres(f, 310.0)[0] == pytest.approx(420.0)
    assert len(sm.band_centres(f, 430.0)) == f.n_bands
    assert sm.shifted_centre(600.0, np.radians(15.0), 2.0) < 600.0     # blue shift off axis


def test_capture_shapes_ground_truth_and_determinism():
    exp = slab_experiment(["EGFP"], conc=2.0, sensor=small_sensor())
    tr = StubTransport()
    a = cap.capture(exp, transport=tr)
    b = cap.capture(exp, transport=StubTransport())
    assert a.width == 48 and a.height == 36 and a.excitations_nm == [365.0, 405.0]
    B = exp.camera().filter.n_bands
    for ex in a.excitations_nm:
        assert a.adu[ex].shape == (36, 48, B) and a.adu[ex].dtype == np.float32
        assert a.radiance[ex].shape == a.electrons_mean[ex].shape == (36, 48, B)
        np.testing.assert_array_equal(a.adu[ex], b.adu[ex])            # bit-identical repeats
    keys = set(a.ground_truth)
    assert set(cap.GROUND_TRUTH_KEYS) <= keys
    assert {"irradiance@365", "irradiance@405", "conc:EGFP", "bleached:EGFP"} <= keys
    assert a.ground_truth["conc:EGFP"].max() == pytest.approx(2.0)
    c = cap.capture(exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"seed": 7})}),
                    transport=tr)
    assert not np.array_equal(a.adu[365.0], c.adu[365.0])


def test_dark_and_white_modes():
    exp = slab_experiment(["EGFP"], sensor=small_sensor())
    tr = StubTransport()
    dark = cap.capture(exp, mode="dark", transport=tr)
    sample = cap.capture(exp, transport=tr)
    assert dark.mode == "dark" and all(v == 0 for v in dark.power_mW.values())
    off = exp.camera().sensor.offset_adu
    assert abs(dark.adu[365.0].mean() - off) < 5.0
    assert sample.adu[365.0].mean() > dark.adu[365.0].mean() + 1.0
    white = cap.capture(exp, mode="white_reference", transport=tr)
    assert white.mode == "white_reference"
    assert white.excitations_nm == sample.excitations_nm
    np.testing.assert_allclose(white.emission_nm[365.0], sample.emission_nm[365.0])
    assert white.radiance[365.0].mean() > 0
    assert white.ground_truth["material"].min() >= 0       # plate fills the whole frame


def test_room_light_adds_signal_in_dark_mode():
    room = RoomLight(enabled=True, spectrum="fluorescent_tube", illuminance_lux=500.0)
    exp = slab_experiment(["EGFP"], sensor=small_sensor(), room=room, toggles=no_noise())
    dark = cap.capture(exp, mode="dark", transport=StubTransport())
    off = exp.model_copy(update={"background": exp.background.model_copy(update={"room_light": RoomLight()})})
    dark_off = cap.capture(off, mode="dark", transport=StubTransport())
    assert dark.radiance[365.0].sum() > 0 and dark_off.radiance[365.0].sum() == 0
    ex = cap.explain_pixel(exp, 24, 18, mode="dark", transport=StubTransport())
    assert "room_light" in ex["components"]["365"]


def test_bleaching_order_effect():
    """Strong bleaching: reversing the sequence order changes later-step signals; off -> identical."""
    def exp_for(order, bleach_on):
        e = slab_experiment(["EGFP"], conc=5.0, sensor=small_sensor(), toggles=no_noise().model_copy(
            update={"photobleaching": bleach_on}), steps=order)
        fl = presets.builtin_fluorophores()
        egfp = next(f for f in fl if f.name == "EGFP").model_copy(update={"bleach_quantum_yield": 0.3})
        return e.model_copy(update={"fluorophores": [egfp]})
    A, Bs = (450, 20.0, 2000.0), (488, 20.0, 2000.0)
    fwd, rev = exp_for((A, Bs), True), exp_for((Bs, A), True)
    tr = StubTransport
    r_f = cap.capture(fwd, transport=tr())
    r_r = cap.capture(rev, transport=tr())
    # signal at 488 is lower when it comes second (the 450 step bleached the dye first)
    assert r_f.radiance[488.0].sum() < r_r.radiance[488.0].sum() * 0.95
    assert r_f.ground_truth["bleached:EGFP"].min() < 0.9
    o_f = cap.capture(exp_for((A, Bs), False), transport=tr())
    o_r = cap.capture(exp_for((Bs, A), False), transport=tr())
    np.testing.assert_allclose(o_f.radiance[488.0], o_r.radiance[488.0], rtol=1e-6)
    np.testing.assert_allclose(o_f.ground_truth["bleached:EGFP"], 1.0)


def test_explain_pixel_sums_to_capture_radiance():
    exp = slab_experiment(["EGFP", "mCherry"], conc=1.0, sensor=small_sensor(), toggles=no_noise())
    tr = StubTransport()
    res = cap.capture(exp, transport=tr)
    d = cap.explain_pixel(exp, 24, 18, transport=tr)
    assert d["x"] == 24 and d["y"] == 18
    for ex in res.excitations_nm:
        key = f"{ex:g}"
        tot = np.sum([np.array(v) for v in d["components"][key].values()], axis=0)
        np.testing.assert_allclose(tot, d["radiance"][key], rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(d["radiance"][key], res.radiance[ex][18, 24], rtol=1e-3, atol=1e-9)
    assert any(k.startswith("dye:") for k in d["components"]["365"])
    assert d["ground_truth"]["material"] == "slab" and d["ground_truth"]["conc:EGFP"] == 1.0


def test_explain_background_pixel_component_name():
    exp = slab_experiment(["EGFP"], sensor=small_sensor())
    d = cap.explain_pixel(exp, 1, 1, transport=StubTransport())
    names = set(d["components"]["365"])
    assert any(n.startswith("background:") for n in names) or names <= {"glint", "leak_source"} | names
    assert d["ground_truth"]["object"] == "background"


def test_cancel_returns_partial_with_warning():
    exp = slab_experiment(["EGFP"], sensor=small_sensor())
    res = cap.capture(exp, transport=StubTransport(), cancel=lambda: True)
    assert res.excitations_nm == [] and any("cancel" in w for w in res.warnings)


def test_progress_is_monotonic_and_ends_at_one():
    exp = slab_experiment(["EGFP"], sensor=small_sensor())
    seen = []
    cap.capture(exp, transport=StubTransport(), progress=lambda f, m: seen.append(f))
    assert seen[-1] == 1.0 and all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))


def test_preview_single_excitation_and_scale():
    exp = slab_experiment(["EGFP"], sensor=SensorSpec(width_px=96, height_px=64))
    r = cap.preview(exp, 405.0, scale=0.5, transport=StubTransport())
    assert r.excitations_nm == [405.0] and r.width == 48 and r.height == 32
    r2 = cap.preview(exp, 380.0, scale=0.25, transport=StubTransport())
    assert r2.excitations_nm == [380.0]


def test_raman_only_in_aqueous():
    aq = MaterialSpec(id="aq", name="aq", aqueous=True, refractive_index=1.333, surface_roughness=0.0)
    exp = slab_experiment([], extra_mat=aq, objects_mat="aq", sensor=small_sensor(), toggles=no_noise(),
                          steps=((488, 5.0, 100.0),),
                          filt=SpectralFilterSpec(band_start_nm=520.0, band_step_nm=10.0, n_bands=12,
                                                  emission_longpass_nm=None))
    d = cap.explain_pixel(exp, 24, 18, transport=StubTransport())
    raman = np.array(d["components"]["488"]["raman"])
    assert raman.sum() > 0
    centres = np.array(d["emission_nm"]["488"])
    assert abs(centres[np.argmax(raman)] - 584.0) <= 10.0
    dry = slab_experiment([], objects_mat="slab", sensor=small_sensor(), steps=((488, 5.0, 100.0),))
    d2 = cap.explain_pixel(dry, 24, 18, transport=StubTransport())
    assert "raman" not in d2["components"]["488"]


def test_saturation_warning_and_clip():
    exp = slab_experiment(["EGFP"], conc=50.0, sensor=small_sensor(full_well_e=500.0),
                          steps=((470, 5.0, 3000.0),))
    r = cap.capture(exp, transport=StubTransport())
    assert any("saturat" in w for w in r.warnings)
    assert r.adu[470.0].max() <= 2 ** 16 - 1


def test_turbid_material_and_subsurface_spread_run():
    exp = slab_experiment(["EGFP"], conc=2.0, clear=False, sensor=small_sensor())
    r = cap.capture(exp, transport=StubTransport())
    assert np.isfinite(r.adu[365.0]).all()


def test_physics_toggles_remove_components():
    base = slab_experiment(["EGFP"], conc=3.0, sensor=small_sensor(), toggles=no_noise(), steps=((470, 5.0, 100.0),))

    def comps(**tog):
        e = base.model_copy(update={"acquisition": base.acquisition.model_copy(
            update={"physics": no_noise().model_copy(update=tog)})})
        return cap.explain_pixel(e, 24, 18, transport=StubTransport())["components"]["470"]
    assert "dye:EGFP" in comps()
    assert "dye:EGFP" not in comps(fluorescence=False)
    assert "glint" not in comps(specular_glints=False)
    assert "secondary:EGFP" not in comps(secondary_emission=False)


def test_secondary_emission_grows_with_concentration():
    def frac(c):
        e = slab_experiment([("fluorescein", c)], sensor=small_sensor(), toggles=no_noise(), steps=((470, 5.0, 100.0),),
                            filt=SpectralFilterSpec(band_start_nm=480.0, band_step_nm=10.0, n_bands=20,
                                                    emission_longpass_nm=None))
        d = cap.explain_pixel(e, 24, 18, transport=StubTransport())["components"]["470"]
        return np.sum(d["secondary:fluorescein"]) / np.sum(d["dye:fluorescein"])
    lo, hi = frac(1.0), frac(2000.0)
    assert 0 <= lo < 0.05 and hi > 3 * lo and hi < 10.0


def test_blur_is_applied_and_toggles():
    sharp = slab_experiment(["EGFP"], conc=3.0, sensor=small_sensor(), toggles=no_noise(), steps=((470, 5.0, 100.0),))
    cam = sharp.camera()
    lens = cam.lens.model_copy(update={"axial_chromatic_mm_per_100nm": 3.0})
    blurry = sharp.model_copy(update={"cameras": [cam.model_copy(update={"lens": lens})]})
    on = blurry.model_copy(update={"acquisition": blurry.acquisition.model_copy(update={
        "physics": no_noise().model_copy(update={"chromatic_aberration": True, "diffraction": True})})})
    e_sharp = cap.capture(sharp, transport=StubTransport()).electrons_mean[470.0]
    e_blur = cap.capture(on, transport=StubTransport()).electrons_mean[470.0]
    tv = lambda a: np.abs(np.diff(a, axis=1)).sum()      # noqa: E731
    assert tv(e_blur) < 0.98 * tv(e_sharp)
    assert e_blur.sum() == pytest.approx(e_sharp.sum(), rel=0.03)         # blur conserves flux


def test_vignetting_and_angle_effects_on_flat_field():
    exp = slab_experiment(["EGFP"], sensor=small_sensor(), toggles=no_noise())
    white = cap.capture(exp, mode="white_reference", transport=StubTransport())
    e = white.electrons_mean[365.0].sum(axis=2)
    assert e[18, 24] > e[0, 0]          # centre brighter than corner (cos^4 + beam profile)


def test_dilute_limit_with_real_clear_kernel():
    tr = pytest.importorskip("spectraforge.studio.transport")
    exp = slab_experiment([])
    fl = optics_mod.merged_fluorophores(exp)
    mat = MaterialSpec(id="d", name="d", dyes=[DyeAmount(fluorophore="EGFP", concentration_uM=0.02),
                                              DyeAmount(fluorophore="mCherry", concentration_uM=0.05)],
                       refractive_index=1.33)
    opt = optics_mod.material_optics(mat, fl, GRID)
    src = exp.source().model_copy(update={"line_fwhm_nm": 1.0, "lamp_curve": "flat"})
    one = np.array([1.0])
    amps = {}
    for ex in (450.0, 488.0, 532.0):
        k = tr.clear_medium_kernel(opt, ex, one, one, np.array([1.0]), 1.0, np.array([1.0]))
        sp = cap.dye_spectra(opt, cap._line_shape(src, ex), 1.0, k.fluorescence_G[0])
        amps[ex] = {d: sp[d].sum() for d in sp}
    # dilute: amplitude ratio between two excitations equals the legacy extinction*exc ratio
    f_e, f_m = fl["EGFP"], fl["mCherry"]
    from spectraforge.studio.optics import to_legacy
    for d, f in (("EGFP", f_e), ("mCherry", f_m)):
        got = amps[450.0][d] / amps[488.0][d]
        want = float(to_legacy(f).excitation(450.0) / to_legacy(f).excitation(488.0))
        assert got == pytest.approx(want, rel=0.03)


@pytest.mark.skipif(importlib.util.find_spec("spectraforge.studio.transport") is None,
                    reason="real transport module not available yet")
@pytest.mark.parametrize("name", ["mehsi_lab_replica", "generic_macro_bench", "laser_line", "led_multispectral"])
def test_integration_presets(name):
    exp = presets.preset(name)
    r = cap.preview(exp)
    assert np.isfinite(r.adu[r.excitations_nm[0]]).all()


def test_volumetric_path_matches_clear_kernel_on_same_face_slab():
    """Synthetic volume quadrature (same-face illumination) reproduces the clear closed form."""
    from spectraforge.studio.tracer import trace_scene
    Q = 64

    def vol_tracer(experiment, width, height, **kw):
        h = trace_scene(experiment, width, height, **kw)
        K, P = h.material.shape
        L = h.path_mm
        s = (np.arange(Q) + 0.5) / Q
        h.vol_s_mm = L[:, :, None] * s[None, None, :]
        h.vol_weight_mm = np.repeat((L / Q)[:, :, None], Q, axis=2)
        mat = np.maximum(h.material, 0)
        n = np.array([m.refractive_index for m in experiment.materials])[mat]
        n_out = h.n_outside
        cos_i = np.clip(h.cos_light, 1e-3, 1)
        T_in, ci = _fresnel_T(cos_i, n_out, n)
        T_out, _ = _fresnel_T(np.clip(h.cos_view, 1e-3, 1), n_out, n)
        ci = np.maximum(ci, 0.05)
        F = h.irradiance_per_mW * T_in / ci
        h.vol_fluence_per_mW = np.repeat(F[:, :, None], Q, axis=2) * (h.material >= 0)[:, :, None]
        z = h.vol_s_mm * h.cos_view_internal[:, :, None]
        h.vol_light_path_mm = z / ci[:, :, None]
        h.vol_T_out = T_out * (h.material >= 0)
        return h

    def kernel_tracer(experiment, width, height, **kw):
        h = trace_scene(experiment, width, height, **kw)
        h.vol_weight_mm = None
        return h

    exp = slab_experiment([("fluorescein", 20.0)], sensor=small_sensor(), toggles=no_noise(),
                          steps=((470, 5.0, 100.0),))
    for e in (exp,):
        a = cap.capture(e, transport=StubTransport(), tracer=vol_tracer)
        b = cap.capture(e, transport=StubTransport(), tracer=kernel_tracer)
    obj = a.ground_truth["object"] == 0
    ra, rb = a.radiance[470.0].sum(axis=2)[obj], b.radiance[470.0].sum(axis=2)[obj]
    assert np.median(np.abs(ra / rb - 1.0)) < 0.01


# --------------------------------------------------------------------------------------------
# QA additions: progress format, thick-limit volume integration, prior-occluder Beer, glint statistics
# --------------------------------------------------------------------------------------------
def test_progress_messages_have_the_documented_format():
    import re
    exp = slab_experiment(["EGFP"], steps=((405, 5.0, 50.0), (488, 5.0, 50.0)), toggles=no_noise(),
                          sensor=small_sensor())
    msgs = []
    cap.capture(exp, transport=StubTransport(), progress=lambda f, m: msgs.append((f, m)))
    texts = [m for _, m in msgs]
    rad = [m for m in texts if m.startswith("simulating")]
    sen = [m for m in texts if m.startswith("sensor")]
    assert all(re.fullmatch(r"simulating ([\d.]+) nm \((\d+)/(\d+)\) · radiance", m) for m in rad)
    assert all(re.fullmatch(r"sensor ([\d.]+) nm \((\d+)/(\d+)\)", m) for m in sen)
    assert {"simulating 405 nm (1/2) · radiance", "simulating 488 nm (2/2) · radiance",
            "sensor 405 nm (1/2)", "sensor 488 nm (2/2)"} <= set(texts)
    fr = [f for f, _ in msgs]
    assert fr == sorted(fr) and fr[-1] == 1.0 and texts[-1] == "done"


def _front_slab(c_uM, aperture):
    from spectraforge.studio.schema import BeamSpec, LensSpec
    exp = slab_experiment([("fluorescein", c_uM)], steps=((488, 5.0, 50.0),), toggles=no_noise(),
                          sensor=small_sensor())
    src = exp.source().model_copy(update={
        "kind": "laser", "line_fwhm_nm": 1.0, "lamp_curve": "flat", "excitation_filter_od": 6.0,
        "position_mm": (-60.0, 160.0, 0.0), "target_mm": (0.0, 4.0, 0.0),
        "beam": BeamSpec(profile="flat_top", aperture_diameter_mm=aperture, divergence_half_angle_deg=0.05)})
    cam = exp.camera().model_copy(update={"filter": SpectralFilterSpec(
        kind="lctf", band_start_nm=530.0, band_step_nm=10.0, n_bands=8, band_start_policy="fixed",
        fwhm_nm=20.0, peak_transmission_curve="flat", emission_longpass_nm=None)})
    return exp.model_copy(update={"sources": [src], "cameras": [cam]})


@pytest.mark.parametrize("c_uM", [1.0, 100.0, 1000.0])
def test_volume_quadrature_matches_closed_form_even_when_optically_thick(c_uM):
    """Front-lit, laterally uniform illumination: tracer volume samples (exact cell integration) vs the
    closed-form kernel, including mm-penetration media where a point rule over 16 nodes fails."""
    from spectraforge.studio.tracer import trace_scene
    tr = pytest.importorskip("spectraforge.studio.transport")
    exp = _front_slab(c_uM, aperture=400.0)
    closed = lambda e, w, h, **kw: trace_scene(e, w, h, volume_samples=0, **{k: v for k, v in kw.items()  # noqa: E731
                                                                              if k != "volume_samples"})
    a = cap.capture(exp, scale=0.25, transport=tr.get_transport_model(), tracer=closed)
    b = cap.capture(exp, scale=0.25, transport=tr.get_transport_model())
    ca = a.radiance[488.0][a.height // 2, a.width // 2].sum()
    cb = b.radiance[488.0][b.height // 2, b.width // 2].sum()
    assert ca > 0
    assert cb == pytest.approx(ca, rel=0.01)


def test_beam_is_attenuated_by_a_preceding_cuvette_in_a_row():
    """Beer-Lambert absorption by clear media crossed *before* a side-lit cuvette (tracer omits it)."""
    pytest.importorskip("spectraforge.studio.transport")
    exp = presets.preset("laser_line")
    nfl = [i for i, o in enumerate(exp.objects) if o.id == "cuv_fluo"][0]
    only = exp.model_copy(update={"objects": [o for o in exp.objects if o.id == "cuv_fluo"]})
    both = exp.model_copy(update={"objects": [o for o in exp.objects if o.id in ("cuv_fret", "cuv_fluo")]})
    kw = dict(scale=0.25, excitations=[488.0])
    a = cap.capture(only, **kw)
    b = cap.capture(both, **kw)
    sel = a.ground_truth["object"] == 0
    sel_b = b.ground_truth["object"] == 1
    ra = a.radiance[488.0].sum(2)[sel].max()
    rb = b.radiance[488.0].sum(2)[sel_b].max()
    fl = optics_mod.merged_fluorophores(exp)
    tau = np.log(10) * float(optics_mod.dye_extinction_M_cm(fl["EGFP"], 488.0)) * 8e-6 * 1.0     # 1 cm of 8 uM EGFP
    assert rb < ra * np.exp(-0.8 * tau)                  # clearly attenuated ...
    assert rb > ra * np.exp(-1.2 * tau) * 0.7            # ... by about the Beer factor (+ Fresnel of 4 interfaces)


def test_glint_statistics_report_fraction_and_magnitude():
    P, B = 400, 5
    st = cap._Step(index=0, ex_nm=405.0, power_mW=1.0, exposure_ms=1.0, centres=np.zeros(B), responses=None,
                   wsel=None, line=None, leak=None, raman=None, beer_idx=None, beer_w=None)
    st.rad_s = np.zeros((P, B), np.float32)
    st.rad_d = np.ones((P, B), np.float32)
    st.fl_b = np.ones((P, B), np.float32)
    st.rad_s[:200, 0] = 10.0                               # half of the pixels: glint 10x the fluorescence
    msg = cap._glint_stats(st, np.ones(P, bool))
    assert msg is not None and "50%" in msg and "200 px" in msg and "median 10x" in msg
    st.rad_s[:] = 0.01
    assert cap._glint_stats(st, np.ones(P, bool)) is None
