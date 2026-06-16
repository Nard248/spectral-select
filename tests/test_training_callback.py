"""train_with_masking should invoke an optional per-epoch progress callback."""
import numpy as np
import torch

from spectral_select.models.autoencoder import HyperspectralCAEWithMasking
from spectral_select.models.dataset import MaskedHyperspectralDataset
from spectral_select.models.training import train_with_masking


def _tiny_model_and_dataset():
    np.random.seed(0)
    cube = np.random.rand(8, 8, 4).astype(np.float32)
    wavelengths = [500.0, 510.0, 520.0, 530.0]
    mask = np.ones((8, 8), dtype=bool)
    # Model takes {ex_float: cube}; dataset takes the nested production format.
    model = HyperspectralCAEWithMasking(excitations_data={365.0: cube}, k1=4, k3=2, filter_size=3)
    data_dict = {
        "data": {"365.0": {"cube": cube, "wavelengths": wavelengths}},
        "excitation_wavelengths": [365.0],
    }
    dataset = MaskedHyperspectralDataset(data_dict=data_dict, mask=mask, normalize=True)
    return model, dataset, mask


def test_progress_callback_fires_once_per_epoch(tmp_path):
    torch.manual_seed(0)
    model, dataset, mask = _tiny_model_and_dataset()
    calls = []
    train_with_masking(
        model=model, dataset=dataset, num_epochs=3, learning_rate=1e-3,
        chunk_size=8, chunk_overlap=0, device="cpu",
        early_stopping_patience=None, scheduler_patience=999, mask=mask,
        output_dir=str(tmp_path), verbose=False,
        progress_callback=lambda epoch, total, loss: calls.append((epoch, total, loss)),
    )
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(c[1] == 3 for c in calls)
    assert all(isinstance(c[2], float) for c in calls)
