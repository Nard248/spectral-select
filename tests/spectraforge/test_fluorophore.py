import numpy as np
from spectraforge.fluorophore import Fluorophore


def test_excitation_peaks_at_one():
    f = Fluorophore("X", ex_peak_nm=480, ex_fwhm_nm=40, em_peak_nm=520, em_fwhm_nm=40)
    assert np.isclose(f.excitation(480.0), 1.0)
    assert f.excitation(480.0) > f.excitation(520.0)


def test_emission_unit_area_on_grid():
    f = Fluorophore("X", ex_peak_nm=480, ex_fwhm_nm=40, em_peak_nm=520, em_fwhm_nm=40)
    grid = np.arange(400, 700, 2.0)
    em = f.emission(grid)
    assert np.isclose(em.sum(), 1.0)
    assert grid[int(np.argmax(em))] == 520.0


def test_fwhm_to_sigma_width():
    f = Fluorophore("X", ex_peak_nm=500, ex_fwhm_nm=50, em_peak_nm=560, em_fwhm_nm=50)
    half = 500 + 25  # at +FWHM/2 the excitation should be ~0.5
    assert np.isclose(f.excitation(half), 0.5, atol=1e-3)
