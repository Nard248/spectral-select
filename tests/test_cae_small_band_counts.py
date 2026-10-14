import pytest
import torch

from spectral_select.models.autoencoder import HyperspectralCAEWithMasking
from spectral_select.architectures.deep_cae import DeepSpectralCAE


@pytest.mark.parametrize("cls", [HyperspectralCAEWithMasking, DeepSpectralCAE])
@pytest.mark.parametrize("n_bands", [1, 2, 3, 4, 5, 31])
def test_forward_shape_small_band_counts(cls, n_bands):
    torch.manual_seed(0)
    data = {405.0: torch.rand(1, 8, 8, n_bands), 450.0: torch.rand(1, 8, 8, n_bands)}
    model = cls({ex: t[0] for ex, t in data.items()}, k1=4, k3=4, filter_size=3).eval()
    with torch.no_grad():
        out = model(data)
    for ex, t in data.items():
        assert out[ex].shape == t.shape
