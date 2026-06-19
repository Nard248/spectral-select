from spectral_select.types import SpectraData
from spectraforge.material import Material
from spectraforge.gui.state import ForgeState
from spectraforge.gui.render_ops import render_state, export_dataset


def _state():
    st = ForgeState(height=12, width=12)
    layer = st.add_layer("g", Material("g", {"EGFP": 1.0}))
    layer.amount_map[:, :6] = 1.0
    return st


def test_render_state_returns_spectradata():
    spectra, gt = render_state(_state())
    assert isinstance(spectra, SpectraData)
    assert spectra.n_excitations == 3
    assert "EGFP" in gt.concentration_maps


def test_export_dataset_writes_files(tmp_path):
    spectra, gt = render_state(_state())
    export_dataset(spectra, gt, tmp_path)
    assert (tmp_path / "spectra_unmasked.pkl").exists()
    assert (tmp_path / "groundtruth.npz").exists()
