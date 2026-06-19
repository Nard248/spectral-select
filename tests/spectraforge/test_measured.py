"""Increment D — measured (real) spectra via interpolation; drop-in for Fluorophore."""
import numpy as np

from spectraforge import AcquisitionConfig, Material, Scene, render


def test_measured_fluorophore_interpolates_and_normalizes():
    from spectraforge.measured import MeasuredFluorophore
    f = MeasuredFluorophore(
        "m",
        ex_wavelengths=[400, 500, 600], ex_values=[0, 1, 0],
        em_wavelengths=[500, 550, 600], em_values=[0, 2, 0],
    )
    assert f.excitation(500) == 1.0          # peak-normalized
    assert f.excitation(450) == 0.5          # linear interpolation
    assert f.excitation(700) == 0.0          # outside support -> 0
    em = f.emission(np.array([500.0, 550.0, 600.0]))
    assert np.isclose(em.sum(), 1.0)         # area-normalized on the grid (matches Fluorophore)
    assert em[1] > em[0]                      # peak at 550


def test_measured_fluorophore_is_render_drop_in():
    from spectraforge.measured import MeasuredFluorophore
    f = MeasuredFluorophore("m", [480, 520, 560], [0, 1, 0], [560, 600, 640], [0, 1, 0],
                            quantum_yield=0.6, extinction=1.0)
    acq = AcquisitionConfig(excitations=[520.0], em_min=500, em_max=700, em_step=10)
    s = Scene(8, 8)
    s.paint_rect(Material("mat", {"m": 1.0}), 0, 8, 0, 8)
    spectra, _ = render(s, {"m": f}, acq)
    cube = spectra.get_excitation(520.0).cube
    grid = acq.emission_grid()
    peak_band = int(np.argmin(np.abs(grid - 600)))   # measured emission peak
    assert cube[4, 4, peak_band] > 0
    assert cube[4, 4, peak_band] == cube.max()       # the measured peak is the brightest band


def test_from_fpbase_payload():
    from spectraforge.measured import MeasuredFluorophore, from_fpbase_payload
    payload = {
        "name": "EGFP", "qy": 0.6,
        "spectra": [
            {"subtype": "EX", "data": [[450, 0.0], [488, 1.0], [510, 0.2]]},
            {"subtype": "EM", "data": [[500, 0.0], [507, 1.0], [550, 0.3]]},
        ],
    }
    f = from_fpbase_payload(payload)
    assert isinstance(f, MeasuredFluorophore)
    assert f.name == "EGFP"
    assert f.quantum_yield == 0.6
    assert f.excitation(488) == 1.0
    assert f.emission(np.array([507.0])).sum() > 0
