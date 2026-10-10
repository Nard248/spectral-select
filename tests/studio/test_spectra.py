"""Tabulated spectra: values, normalisations and smoothness."""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge.studio import spectra as sp

NM = np.arange(250.0, 900.0 + 1e-9, 1.0)


def test_water_absorption_reference_values():
    w = sp.water_mua_per_mm
    assert w(420.0) < 1e-5                         # tiny in the blue
    assert w(700.0) == pytest.approx(6.2e-4, rel=0.1)
    assert w(800.0) == pytest.approx(2.0e-3, rel=0.15)
    assert w(NM).min() > 0 and w(900.0) > w(800.0) > w(700.0) > w(600.0) > w(500.0)


def test_hemoglobin_normalised_at_500_and_soret_peak():
    for ox in (0.0, 0.5, 1.0):
        assert sp.hemoglobin_shape(500.0, ox) == pytest.approx(1.0)
    assert sp.hemoglobin_shape(415.0, 1.0) > 15 * sp.hemoglobin_shape(650.0, 1.0)
    # oxy vs deoxy differ at 560 / 576 nm (the classic double peak of HbO2)
    assert sp.hemoglobin_shape(576.0, 1.0) > sp.hemoglobin_shape(576.0, 0.0)
    assert sp.hemoglobin_shape(556.0, 0.0) > sp.hemoglobin_shape(556.0, 1.0)


def test_melanin_and_carbon_power_laws():
    assert sp.melanin_shape(500.0) == pytest.approx(1.0)
    assert sp.melanin_shape(400.0) / sp.melanin_shape(800.0) == pytest.approx(2.0 ** 3.46, rel=1e-9)
    assert sp.carbon_black_shape(250.0) == pytest.approx(2.0)


def test_absorber_scale_conventions():
    assert sp.absorber_mua_per_mm("melanin", 3.0, 500.0) == pytest.approx(3.0)
    assert sp.absorber_mua_per_mm("water", 2.0, 700.0) == pytest.approx(2.0 * sp.water_mua_per_mm(700.0))
    assert sp.absorber_mua_per_mm("none", 5.0, NM).sum() == 0
    c = sp.absorber_mua_per_mm("custom", 2.0, np.array([400.0, 500.0, 600.0]), [(400, 1.0), (600, 3.0)])
    np.testing.assert_allclose(c, [2.0, 4.0, 6.0])
    with pytest.raises(ValueError):
        sp.absorber_mua_per_mm("bogus", 1.0, 500.0)


@pytest.mark.parametrize("curve", ["ccd_bv", "ccd_uv", "scmos", "flat"])
def test_qe_in_unit_range_and_smooth(curve):
    q = sp.qe(curve, NM)
    assert q.min() >= 0 and q.max() <= 1
    assert np.abs(np.diff(q)).max() < 0.02          # no discontinuities on a 1 nm grid


def test_qe_ordering():
    assert sp.qe("ccd_uv", 300.0) > 3 * sp.qe("ccd_bv", 300.0)
    assert sp.qe("scmos", 560.0) > sp.qe("ccd_bv", 560.0)


def test_lctf_vis_shape():
    t = sp.filter_peak_transmission("lctf_vis", NM)
    assert sp.filter_peak_transmission("lctf_vis", 420.0) < 0.06
    assert 0.40 <= sp.filter_peak_transmission("lctf_vis", 650.0) <= 0.52
    assert sp.filter_peak_transmission("lctf_vis", 720.0) > 0.35
    assert t.max() <= 0.55 and np.abs(np.diff(t)).max() < 0.02
    assert sp.filter_peak_transmission("flat", 500.0) == sp.FILTER_FLAT


@pytest.mark.parametrize("kind", list(sp.ROOM_LIGHT_KINDS))
def test_room_light_lux_normalisation(kind):
    for lux in (5.0, 300.0):
        e = sp.room_light_irradiance_mW_mm2_nm(kind, lux, NM)          # mW/mm^2/nm
        lum = 683.0 * np.trapezoid(sp.cie_v_lambda(NM) * e * 1e3, NM)    # W/m^2/nm -> lux
        assert lum == pytest.approx(lux, rel=0.01)


def test_fluorescent_tube_has_mercury_lines_and_red_phosphor():
    s = sp.room_light_relative_spectrum("fluorescent_tube", NM)
    for line in (405, 436, 546, 578, 611):
        win = (NM > line - 3) & (NM < line + 3)
        local = (NM > line - 15) & (NM < line + 15) & ~win
        assert s[win].max() > 1.5 * np.median(s[local])
    # incandescent is red-rich, daylight blue-rich
    inc = sp.room_light_relative_spectrum("incandescent", NM)
    day = sp.room_light_relative_spectrum("daylight", NM)
    assert inc[NM == 700][0] / inc[NM == 450][0] > 5 * day[NM == 700][0] / day[NM == 450][0]


def test_cie_v_lambda_peak():
    v = sp.cie_v_lambda(NM)
    assert NM[np.argmax(v)] == pytest.approx(555.0, abs=3.0)
    assert v.max() == pytest.approx(1.0, abs=0.02)
