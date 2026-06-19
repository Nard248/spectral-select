from spectraforge.demo import build_demo
from spectral_select.types import SpectraData


def test_build_demo_produces_valid_dataset():
    spectra, gt = build_demo()
    assert isinstance(spectra, SpectraData)
    assert spectra.n_excitations == 3
    assert set(gt.concentration_maps) >= {"collagen", "NADH", "FAD", "EGFP"}
