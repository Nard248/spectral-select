import torch
from channel_select.engine import run_selection
from channel_select.protocols import SelectionConfig


class TinyModel:
    groups = ["g"]
    channels_per_group = {"g": 4}

    def __init__(self):
        self._latent = torch.randn(6, 3)  # (batch, latent dims)

    def encode(self, batch):
        return self._latent

    def decode(self, latent):
        # channel c reconstructed from latent dim (c % 3), so perturbing a dim
        # influences a deterministic channel.
        b = latent.shape[0]
        recon = torch.zeros(b, 5, 4)  # (batch, time, channel)
        for c in range(4):
            recon[..., c] = latent[:, c % 3:c % 3 + 1]
        return {"g": recon}


def test_run_selection_returns_k_channels():
    model = TinyModel()
    data = {"g": torch.randn(6, 5, 4)}
    cfg = SelectionConfig(
        n_important_dimensions=3, n_channels_to_select=2,
        normalization_method="max_per_group", diversity_method="none",
        perturbation_method="standard_deviation", perturbation_magnitudes=[50],
    )
    result = run_selection(model, data, cfg)
    assert len(result.selected) == 2
    assert all(g == "g" for g, _ in result.selected)
    assert set(result.influence.keys()) == {"g"}
