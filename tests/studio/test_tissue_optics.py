"""Tissue optics: literature anchors, unit consistency, Jacques 2013 scattering, Intralipid, tissue library."""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge.studio import optics, presets_tissue, spectra, tissue_optics as to
from spectraforge.studio.presets import builtin_fluorophores
from spectraforge.studio.schema import Experiment, MaterialSpec, ScatteringSpec, TissueComposition

FL = {f.name: f for f in builtin_fluorophores()}
GRID = np.arange(380.0, 900.0 + 1e-9, 2.0)


# ---- literature anchors ------------------------------------------------------------------
@pytest.mark.parametrize("nm,oxy,ref", [(415, True, 524_280), (542, True, 53_236), (576, True, 55_540),
                                        (555, False, 54_540)])
def test_hemoglobin_anchor_values(nm, oxy, ref):
    assert to.hemoglobin_extinction(float(nm), oxy) == pytest.approx(ref, rel=0.01)


def test_hemoglobin_isosbestic_points():
    w = np.arange(480.0, 620.0, 0.25)
    d = to.hemoglobin_extinction(w, True) - to.hemoglobin_extinction(w, False)
    crossings = w[np.where(np.diff(np.sign(d)) != 0)[0]]
    for iso in (500, 529, 545, 570, 584):
        assert np.min(np.abs(crossings - iso)) <= 1.5, (iso, crossings)


def test_water_980nm():
    assert to.water_mua_per_cm(980.0) == pytest.approx(0.45, rel=0.05)     # 0.4385 tabulated
    assert to.water_mua_per_cm(700.0) == pytest.approx(0.006, rel=0.05)
    # Pope & Fry minimum near 420 nm
    assert to.water_mua_per_cm(420.0) < 1e-4


def test_lipid_930nm_peak_and_melanosome_formula():
    assert to.lipid_mua_per_cm(930.0) == pytest.approx(0.131, rel=0.02)
    assert to.melanosome_mua_per_cm(500.0) == pytest.approx(519.0)
    assert to.melanosome_mua_per_cm(1000.0) == pytest.approx(519.0 * 2.0 ** -3.5)


def test_pigment_peaks():
    assert to.bilirubin_extinction(450.0) == pytest.approx(55_000, rel=0.1)   # PhotochemCAD scaled to 55 000 @ 450.8
    assert to.bilirubin_extinction(800.0) == 0.0
    assert 120_000 < float(to.beta_carotene_extinction(450.0)) < 160_000      # ~139 000 (Mohamed lit.)


# ---- unit consistency --------------------------------------------------------------------
def test_whole_blood_mua_415nm():
    # mua = ln10 * 524 280 / 1/M/cm * 150/64500 M = 2807 /cm = 280.7 /mm (HbO2)
    expected = np.log(10) * 524_280 * 150.0 / 64_500.0 / 10.0
    assert to.whole_blood_mua_per_mm(415.0, 1.0) == pytest.approx(expected, rel=0.01)
    assert expected == pytest.approx(280.7, rel=1e-3)
    # saturation blends linearly
    lo, hi = to.whole_blood_mua_per_mm(600.0, 0.0), to.whole_blood_mua_per_mm(600.0, 1.0)
    assert to.whole_blood_mua_per_mm(600.0, 0.4) == pytest.approx(0.6 * lo + 0.4 * hi)


def test_chromophore_amount_units_and_error():
    assert to.chromophore_mua_per_mm("water", 700.0, 0.5) == pytest.approx(0.5 * 0.0006, rel=0.05)
    # 10 uM bilirubin at 450 nm: ln10 * 55 000 * 1e-5 M = 1.27 /cm
    assert to.chromophore_mua_per_mm("bilirubin", 450.0, 10.0) == pytest.approx(0.127, rel=0.1)
    with pytest.raises(ValueError):
        to.chromophore_mua_per_mm("nope", 500.0)


def test_tissue_mua_is_sum_of_chromophores_and_additive_in_optics():
    comp = TissueComposition(blood_volume_fraction=0.02, oxygen_saturation=0.7, water_fraction=0.6,
                             lipid_fraction=0.1, melanosome_fraction=0.05, bilirubin_uM=5.0,
                             beta_carotene_uM=2.0)
    nm = np.array([450.0, 550.0, 650.0, 850.0])
    by_hand = (0.02 * to.whole_blood_mua_per_mm(nm, 0.7) + 0.6 * to.water_mua_per_cm(nm) / 10
               + 0.1 * to.lipid_mua_per_cm(nm) / 10 + 0.05 * to.melanosome_mua_per_cm(nm) / 10
               + np.log(10) * (5e-6 * to.bilirubin_extinction(nm) + 2e-6 * to.beta_carotene_extinction(nm)) / 10)
    np.testing.assert_allclose(to.tissue_mua_per_mm(comp, nm), by_hand, rtol=1e-12)
    assert to.tissue_mua_per_mm(None, nm).sum() == 0
    base = MaterialSpec(id="m", name="m")
    with_t = base.model_copy(update={"tissue": comp})
    o0, o1 = optics.material_optics(base, FL, GRID), optics.material_optics(with_t, FL, GRID)
    np.testing.assert_allclose(o1.mua_per_mm - o0.mua_per_mm, to.tissue_mua_per_mm(comp, GRID), rtol=1e-6, atol=2e-9)


def test_hemoglobin_shape_unchanged_in_character():
    assert spectra.hemoglobin_shape(500.0, 0.5) == pytest.approx(1.0)
    assert spectra.hemoglobin_shape(415.0, 1.0) == pytest.approx(25.0, rel=0.05)   # Soret peak vs 500 nm
    assert spectra.water_mua_per_mm(700.0) == pytest.approx(6.0e-4, rel=0.05)


# ---- scattering --------------------------------------------------------------------------
def test_mie_rayleigh_reduces_to_power_law_at_zero_rayleigh():
    pl = ScatteringSpec(model="power_law", mus_prime_500_per_mm=2.0, power_b=1.3, g=0.9)
    mr = ScatteringSpec(model="mie_rayleigh", mus_prime_500_per_mm=2.0, power_b=1.3, f_rayleigh=0.0, g=0.9)
    np.testing.assert_allclose(optics.reduced_scattering_per_mm(mr, GRID), optics.reduced_scattering_per_mm(pl, GRID))
    m = lambda s: optics.material_optics(MaterialSpec(id="a", name="a", scattering=s), FL, GRID)   # noqa: E731
    np.testing.assert_allclose(m(mr).mus_per_mm, m(pl).mus_per_mm)
    # pure Rayleigh: lambda^-4
    r = ScatteringSpec(model="mie_rayleigh", mus_prime_500_per_mm=1.0, power_b=1.0, f_rayleigh=1.0, g=0.5)
    assert optics.reduced_scattering_per_mm(r, 1000.0) == pytest.approx(2.0 ** -4)


def test_jacques_generic_tissue_value_and_shape():
    # Table 2 'skin': a' = 48.0 /cm -> mus'(500) = 4.8 /mm exactly, and eq. 2 ratio at 400 / 800 nm
    s = to.jacques_generic_mus_prime_per_mm("skin", np.array([400.0, 500.0, 800.0]))
    a, f, b = to.JACQUES2013_TABLE2["skin"]
    assert s[1] == pytest.approx(4.8)
    assert s[0] / s[1] == pytest.approx(f * 1.25 ** 4 + (1 - f) * 1.25 ** b)
    assert s[2] / s[1] == pytest.approx(f * 1.6 ** -4 + (1 - f) * 1.6 ** -b)
    assert 1.0 < s[0] / s[2] < 8.0                      # monotone decrease, sane contrast
    # the library skin material reproduces it through the full optics chain
    mat = next(m for m in presets_tissue.tissue_materials() if m.id == "dermis_papillary")
    o = optics.material_optics(mat, FL, GRID)
    np.testing.assert_allclose(o.mus_per_mm * (1 - o.g), to.jacques_generic_mus_prime_per_mm("skin", GRID),
                               rtol=1e-9)


def test_intralipid_van_staveren_633nm():
    # mus = 2.54e9 * 633^-2.4 /cm, g = 1.1 - 0.58e-3 * 633 = 0.7329, mus' = mus(1-g)
    mus_cm = 2.54e9 * 633.0 ** -2.4
    assert to.intralipid_mus_per_mm(633.0, 10.0) == pytest.approx(mus_cm / 10.0, rel=1e-12)
    assert to.intralipid_g(633.0) == pytest.approx(0.7329, abs=1e-4)
    assert to.intralipid_mus_prime_per_mm(633.0, 10.0) == pytest.approx(mus_cm * (1 - 0.7329) / 10, rel=1e-3)
    assert to.intralipid_mus_prime_per_mm(633.0, 1.0) == pytest.approx(
        0.1 * to.intralipid_mus_prime_per_mm(633.0, 10.0))
    # the library phantom (power-law fit over 400-900 nm) reproduces mus'(633) to a few %
    mat = next(m for m in presets_tissue.tissue_materials() if m.id == "intralipid_1pct")
    o = optics.material_optics(mat, FL, np.array([633.0]))
    ref = float(to.intralipid_mus_prime_per_mm(633.0, 1.0))
    assert float(o.mus_per_mm[0] * (1 - o.g)) == pytest.approx(ref, rel=0.08)
    # mus'(633) of 1 % Intralipid is about 1.3 /mm (literature 1.0-1.4 /mm)
    assert 0.9 < ref < 1.5


def test_fit_power_law_recovers_exact_power_law():
    nm = np.linspace(400, 900, 40)
    a, b = to.fit_power_law(nm, 3.0 * (nm / 500.0) ** -1.7)
    assert a == pytest.approx(3.0) and b == pytest.approx(1.7)


# ---- refractive index dispersion ---------------------------------------------------------
def test_water_index_and_cauchy():
    assert to.water_refractive_index(589.3) == pytest.approx(1.3330, abs=7e-4)
    assert to.water_refractive_index(633.0) == pytest.approx(1.3317, abs=7e-4)
    n = to.cauchy_index(np.array([450.0, 589.0, 800.0]), 1.38)
    assert n[1] == pytest.approx(1.38) and n[0] > n[1] > n[2]
    assert 0.005 < n[0] - n[2] < 0.02


# ---- library -----------------------------------------------------------------------------
def test_tissue_library_valid_unique_and_physical():
    mats = presets_tissue.tissue_materials()
    ids = [m.id for m in mats]
    assert len(set(ids)) == len(ids) >= 15
    from spectraforge.studio.presets import builtin_materials
    from spectraforge.studio.presets import _base_materials, _layered_materials
    assert not set(ids) & {m.id for m in _base_materials() + _layered_materials()}
    assert set(ids) <= {m.id for m in builtin_materials()}   # library is wired into the builtins
    for m in mats:
        o = optics.material_optics(m, FL, GRID)
        assert np.all(np.isfinite(o.mua_per_mm)) and np.all(o.mua_per_mm > 0)
    mel = {m.id: optics.material_optics(m, FL, np.array([500.0])).mua_per_mm[0] for m in mats}
    assert mel["epidermis_dark"] > mel["epidermis_medium"] > mel["epidermis_light"]
    # blood is the strongest absorber at the Soret band
    ob = optics.material_optics(next(m for m in mats if m.id == "blood_oxygenated"), FL, np.array([415.0]))
    assert 250 < ob.mua_per_mm[0] < 300


def test_tissue_experiments_validate_build_and_preview():
    exps = presets_tissue.tissue_presets()
    assert len(exps) >= 3
    from spectraforge.studio import capture, geometry
    for e in exps:
        Experiment.model_validate(e.model_dump())
        for lay_or_inc in e.objects[0].shape.layers:
            assert any(m.id == lay_or_inc.material_id for m in e.materials)
        geometry.build_scene(e)
    small = exps[0].model_copy(deep=True)
    small.acquisition.volume_mc.excitation_photons = 20_000
    small.acquisition.volume_mc.emission_photons = 20_000
    res = capture.preview(small, scale=0.1)
    assert res is not None
