"""Increment C — richer (opt-in) physics: PSF blur, inner-filter, autofluorescence."""
import numpy as np

from spectraforge import AcquisitionConfig, Fluorophore, Material, Scene, render
from spectraforge.physics import PhysicsConfig

LIB = {"A": Fluorophore("A", 480, 40, 520, 40, quantum_yield=0.6, extinction=1.0)}
ACQ = AcquisitionConfig(excitations=[480.0], em_min=400, em_max=700, em_step=10)
BAND = int(np.argmin(np.abs(ACQ.emission_grid() - 520)))


def _scene_rect(conc=1.0):
    s = Scene(16, 16)
    s.paint_rect(Material("m", {"A": conc}), 4, 12, 4, 12)
    return s


def _cube(spectra):
    return spectra.get_excitation(480.0).cube


def test_physics_off_preserves_linearity():
    with_off = _cube(render(_scene_rect(), LIB, ACQ, physics=PhysicsConfig())[0])
    no_physics = _cube(render(_scene_rect(), LIB, ACQ)[0])
    assert np.allclose(with_off, no_physics)


def test_psf_blur_spreads_signal_to_neighbors():
    s = Scene(16, 16)
    s.paint_rect(Material("m", {"A": 1.0}), 7, 9, 7, 9)   # tiny 2x2 block
    sharp = _cube(render(s, LIB, ACQ)[0])
    blur = _cube(render(s, LIB, ACQ, physics=PhysicsConfig(psf_sigma_px=2.0))[0])
    assert sharp[5, 8, BAND] == 0.0          # pixel outside the block: dark when sharp
    assert blur[5, 8, BAND] > 0.0            # PSF leaks signal into it


def test_inner_filter_attenuates_and_breaks_linearity():
    phys = PhysicsConfig(inner_filter=True, inner_filter_strength=2.0)
    base = _cube(render(_scene_rect(1.0), LIB, ACQ)[0])
    ife1 = _cube(render(_scene_rect(1.0), LIB, ACQ, physics=phys)[0])
    ife2 = _cube(render(_scene_rect(2.0), LIB, ACQ, physics=phys)[0])
    assert ife1[8, 8, BAND] < base[8, 8, BAND]                  # attenuation
    assert not np.allclose(ife2[8, 8, BAND], 2 * ife1[8, 8, BAND])  # nonlinear in concentration


def test_autofluorescence_adds_uniform_background():
    base = _cube(render(_scene_rect(), LIB, ACQ)[0])
    af = _cube(render(_scene_rect(), LIB, ACQ, physics=PhysicsConfig(autofluorescence=0.05))[0])
    assert base[0, 0].sum() == 0.0           # unpainted pixel dark without autofluorescence
    assert af[0, 0].sum() > 0.0              # autofluorescence lights it up everywhere
