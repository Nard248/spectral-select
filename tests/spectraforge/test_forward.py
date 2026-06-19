import numpy as np
from spectral_select.types import SpectraData
from spectraforge.fluorophore import Fluorophore
from spectraforge.material import Material
from spectraforge.scene import Scene
from spectraforge.acquisition import AcquisitionConfig
from spectraforge.forward import render

LIB = {"A": Fluorophore("A", 480, 40, 520, 30, quantum_yield=0.5, extinction=1.0)}
ACQ = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=5)


def _scene(conc):
    s = Scene(6, 6)
    s.paint_rect(Material("m", {"A": conc}), 0, 6, 0, 6)
    return s


def test_render_emits_spectradata_with_right_shape():
    spectra, gt = render(_scene(1.0), LIB, ACQ)
    assert isinstance(spectra, SpectraData)
    ex = spectra.get_excitation(480.0)
    assert ex.cube.shape == (6, 6, len(ACQ.emission_grid()))
    assert len(ex.emission_wavelengths) == ex.cube.shape[2]


def test_emission_peak_at_fluorophore_em_peak():
    spectra, _ = render(_scene(1.0), LIB, ACQ)
    cube = spectra.get_excitation(480.0).cube
    grid = ACQ.emission_grid()
    peak_band = int(np.argmax(cube[0, 0]))
    assert grid[peak_band] == 520.0


def test_amplitude_scales_with_concentration():
    s1, _ = render(_scene(1.0), LIB, ACQ)
    s2, _ = render(_scene(2.0), LIB, ACQ)
    assert np.allclose(2 * s1.get_excitation(480.0).cube, s2.get_excitation(480.0).cube)


def test_linearity_invariant_clean():
    a, b = _scene(1.0), _scene(0.7)
    ra, _ = render(a, LIB, ACQ)
    rb, _ = render(b, LIB, ACQ)
    rab, _ = render(a + b, LIB, ACQ)
    assert np.allclose(rab.get_excitation(480.0).cube,
                       ra.get_excitation(480.0).cube + rb.get_excitation(480.0).cube)


def test_exposure_power_scaling_applied_and_recorded():
    acq = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=5,
                            exposure={480.0: 3.0}, power={480.0: 2.0})
    base, _ = render(_scene(1.0), LIB, ACQ)
    scaled, _ = render(_scene(1.0), LIB, acq)
    assert np.allclose(scaled.get_excitation(480.0).cube, 6.0 * base.get_excitation(480.0).cube)
    assert scaled.get_excitation(480.0).exposure_time == 3.0
    assert scaled.get_excitation(480.0).laser_power == 2.0
