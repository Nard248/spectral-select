"""End-to-end: SpectraForge render -> SpectraData -> spectral_select.Analyzer."""
import numpy as np

from spectral_select.types import SpectraData
from spectraforge.fluorophore import Fluorophore
from spectraforge.material import Material
from spectraforge.scene import Scene
from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.forward import render

LIB = {
    "G": Fluorophore("G", 488, 40, 520, 30, quantum_yield=0.6, extinction=1.0),
    "R": Fluorophore("R", 560, 40, 610, 35, quantum_yield=0.4, extinction=0.9),
}
ACQ = AcquisitionConfig(excitations=[488.0, 560.0], em_min=400, em_max=700, em_step=5)


def _two_material_scene():
    # 64x64 so the Analyzer extracts multiple baseline patches (meaningful selection).
    s = Scene(64, 64)
    s.paint_rect(Material("g", {"G": 1.0}), 0, 64, 0, 32)
    s.paint_rect(Material("r", {"R": 1.0}), 0, 64, 32, 64)
    return s


def test_render_roundtrips_through_pickle(tmp_path):
    spectra, gt = render(
        _two_material_scene(), LIB, ACQ,
        artifacts=ArtifactConfig(rayleigh_strength=0.2, photon_scale=200, read_sigma=0.01),
        seed=0, sample_name="synth",
    )
    p = tmp_path / "synth.pkl"
    spectra.to_pickle(p)
    reloaded = SpectraData.from_pickle(p)
    assert reloaded.n_excitations == 2
    assert reloaded.get_excitation(488.0).cube.shape == spectra.get_excitation(488.0).cube.shape


def test_groundtruth_informative_bands_near_emission_peaks():
    spectra, gt = render(_two_material_scene(), LIB, ACQ, seed=0)
    info = gt.informative_bands(threshold=0.05)
    grid = ACQ.emission_grid()
    band_520 = int(np.argmin(np.abs(grid - 520)))  # G emission under 488 excitation
    assert info[488.0][band_520]


def test_analyzer_runs_on_synthetic(tmp_path):
    from spectral_select import Analyzer, Config

    spectra, _ = render(
        _two_material_scene(), LIB, ACQ,
        artifacts=ArtifactConfig(photon_scale=500, read_sigma=0.01), seed=1,
    )
    cfg = Config(
        sample_name="synth", n_important_dimensions=4, n_bands_to_select=4,
        perturbation_method="standard_deviation", use_diversity_constraint=False,
        training_epochs=2, device="cpu", output_dir=tmp_path,
    )
    analyzer = Analyzer(cfg)
    analyzer.fit(spectra)
    bands = analyzer.get_wavelengths()
    assert len(bands) == 4
