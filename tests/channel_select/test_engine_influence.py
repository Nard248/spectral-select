import numpy as np
import torch
from channel_select.engine import measure_channel_influence


class FakeModel:
    """decode() returns latent[:, :1] broadcast so influence = |Δlatent| per channel."""
    groups = ["g"]
    channels_per_group = {"g": 2}

    def decode(self, latent):
        # latent (batch, 2). Build recon (batch, time=3, channel=2):
        # channel 0 tracks latent dim 0, channel 1 tracks latent dim 1.
        b = latent.shape[0]
        recon = torch.zeros(b, 3, 2)
        recon[..., 0] = latent[:, 0:1]
        recon[..., 1] = latent[:, 1:2]
        return {"g": recon}


def test_reduces_all_axes_except_channel():
    model = FakeModel()
    base_latent = torch.zeros(1, 2)
    baseline_recon = model.decode(base_latent)
    pert = base_latent.clone()
    pert[:, 1] += 4.0  # perturb only channel-1's driver
    infl = measure_channel_influence(model, pert, baseline_recon, weight=1.0)
    assert infl["g"].shape == (2,)
    assert np.isclose(infl["g"][0], 0.0)
    assert np.isclose(infl["g"][1], 4.0)  # mean|Δ| over batch×time = 4
