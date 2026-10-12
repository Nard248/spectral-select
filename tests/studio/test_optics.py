"""MaterialSpec -> MaterialOptics."""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge.studio import optics
from spectraforge.studio import presets
from spectraforge.studio.contracts import SPECTRAL_GRID_NM as GRID
from spectraforge.studio.schema import (
    AbsorberSpec,
    DyeAmount,
    Experiment,
    FluorophoreSpec,
    MaterialSpec,
    PhysicsToggles,
    ScatteringSpec,
)

BLANK = Experiment(materials=presets.builtin_materials())
FL = optics.merged_fluorophores(BLANK)


def mat(**kw):
    kw.setdefault("id", "m")
    kw.setdefault("name", "m")
    return MaterialSpec(**kw)


def test_merged_fluorophores_override_and_extend():
    custom = FluorophoreSpec(name="EGFP", ex_peak_nm=400, ex_fwhm_nm=10, em_peak_nm=500, em_fwhm_nm=20)
    extra = FluorophoreSpec(name="mine", ex_peak_nm=500, ex_fwhm_nm=30, em_peak_nm=600, em_fwhm_nm=30)
    m = optics.merged_fluorophores(BLANK.model_copy(update={"fluorophores": [custom, extra]}))
    assert m["EGFP"].ex_peak_nm == 400 and "mine" in m and "DAPI" in m


def test_dye_absorption_hand_calculation():
    # mua = ln10 * eps * c[M] / 10 ; eps = extinction*55000*excitation(lam)
    m = mat(dyes=[DyeAmount(fluorophore="EGFP", concentration_uM=10.0)])
    o = optics.material_optics(m, FL, GRID)
    spec = FL["EGFP"]
    lam = spec.ex_peak_nm
    expected = np.log(10) * spec.extinction * 55000.0 * 10e-6 / 10.0
    assert o.interp(o.mua_dyes_per_mm["EGFP"], lam) == pytest.approx(expected, rel=1e-6)
    assert o.interp(o.mua_per_mm, lam) == pytest.approx(expected, rel=1e-6)    # no other absorber


def test_background_absorber_adds():
    m = mat(dyes=[DyeAmount(fluorophore="EGFP", concentration_uM=1.0)],
            background_absorption=AbsorberSpec(preset="melanin", scale=2.0))
    o = optics.material_optics(m, FL, GRID)
    np.testing.assert_allclose(o.mua_per_mm, o.mua_dyes_per_mm["EGFP"] + 2.0 * (GRID / 500.0) ** -3.46)


def test_emission_density_unit_area():
    o = optics.material_optics(mat(dyes=[DyeAmount(fluorophore="mCherry", concentration_uM=1.0)]), FL, GRID)
    e = o.emission_density["mCherry"]
    assert e.sum() * 1.0 == pytest.approx(1.0, rel=1e-9)
    assert GRID[np.argmax(e)] == pytest.approx(FL["mCherry"].em_peak_nm, abs=2.0)


def test_stern_volmer_quenching_and_toggle():
    m = mat(dyes=[DyeAmount(fluorophore="fluorescein", concentration_uM=200.0)], quenching_ksv_per_uM=0.002)
    o = optics.material_optics(m, FL, GRID)
    assert o.quantum_yields["fluorescein"] == pytest.approx(FL["fluorescein"].quantum_yield / 1.4)
    o2 = optics.material_optics(m, FL, GRID, toggles=PhysicsToggles(quenching=False))
    assert o2.quantum_yields["fluorescein"] == pytest.approx(FL["fluorescein"].quantum_yield)


def test_concentration_scale():
    m = mat(dyes=[DyeAmount(fluorophore="EGFP", concentration_uM=4.0)])
    a = optics.material_optics(m, FL, GRID)
    b = optics.material_optics(m, FL, GRID, concentration_scale={"EGFP": 0.5})
    np.testing.assert_allclose(b.mua_dyes_per_mm["EGFP"], 0.5 * a.mua_dyes_per_mm["EGFP"])


def test_scattering_power_law_and_hg():
    m = mat(scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=10.0, power_b=1.5, g=0.8))
    o = optics.material_optics(m, FL, GRID)
    assert not o.clear and o.g == 0.8
    assert o.interp(o.mus_per_mm, 500.0) == pytest.approx(10.0 / 0.2)
    assert o.interp(o.mus_per_mm, 800.0) == pytest.approx(10.0 / 0.2 * 1.6 ** -1.5)


def test_clear_rule_matches_tracer():
    assert optics.is_clear_material(mat())
    assert optics.is_clear_material(mat(scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=0.0)))
    turbid = mat(scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=1.0))
    assert not optics.is_clear_material(turbid)
    assert optics.material_optics(turbid, FL, GRID, toggles=PhysicsToggles(scattering=False)).clear


def test_absorption_toggle_removes_inner_filter_but_keeps_dye_absorption():
    m = mat(dyes=[DyeAmount(fluorophore="EGFP", concentration_uM=100.0)],
            background_absorption=AbsorberSpec(preset="water", scale=1.0))
    o = optics.material_optics(m, FL, GRID, toggles=PhysicsToggles(absorption=False))
    assert o.mua_per_mm.max() < 1e-6
    assert o.mua_dyes_per_mm["EGFP"].max() > 0.1


def test_unknown_fluorophore_is_loud():
    with pytest.raises(KeyError):
        optics.material_optics(mat(dyes=[DyeAmount(fluorophore="nope", concentration_uM=1.0)]), FL, GRID)


def test_all_builtin_materials_build():
    for m in presets.builtin_materials():
        o = optics.material_optics(m, FL, GRID)
        assert np.isfinite(o.mua_per_mm).all() and (o.mua_per_mm > 0).all()
        assert o.cache_key()


def test_average_over_line():
    line = np.exp(-0.5 * ((GRID - 500.0) / 5.0) ** 2)
    arr = GRID.copy()
    assert optics.average_over_line(arr, line) == pytest.approx(500.0, abs=1e-6)
    two = np.stack([GRID, 2 * GRID])
    np.testing.assert_allclose(optics.average_over_line(two, line), [500.0, 1000.0], atol=1e-5)
