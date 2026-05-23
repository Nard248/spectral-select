import torch
from channel_select.engine import select_important_dimensions


def test_variance_ranks_high_variance_dims_first():
    # latent (batch=3, d0=2, d1=2). Make coord (1,1) highest variance.
    latent = torch.zeros(3, 2, 2)
    latent[:, 1, 1] = torch.tensor([0.0, 5.0, 10.0])   # high variance
    latent[:, 0, 0] = torch.tensor([1.0, 1.1, 0.9])    # low variance
    top = select_important_dimensions(latent, method="variance", n=1)
    assert len(top) == 1
    score, coord = top[0]
    assert coord == (1, 1)
    assert score > 0


def test_returns_n_sorted_descending():
    latent = torch.randn(5, 3, 4)
    top = select_important_dimensions(latent, method="variance", n=4)
    assert len(top) == 4
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True)
