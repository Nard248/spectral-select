import numpy as np
import torch
from channel_select.engine import latent_statistics, perturbation_amount


def test_standard_deviation_amount():
    latent = torch.zeros(4, 2, 2)
    latent[:, 0, 1] = torch.tensor([0.0, 2.0, 4.0, 6.0])  # std ~2.236
    stats = latent_statistics(latent.reshape(4, -1))
    amt = perturbation_amount(
        coord=(0, 1), latent_dims=(2, 2), magnitude=100.0, sign=1,
        stats=stats, baseline_latent=latent, method="standard_deviation",
    )
    # torch.std defaults to unbiased (N-1), so match with ddof=1
    expected = 1.0 * np.std(latent[:, 0, 1].numpy(), ddof=1)  # magnitude/100 * std
    assert np.isclose(amt, expected, atol=1e-4)
