"""Tests for the minimal fixes to src/spectral_select/me_smartattention (review bugs B2, B4, B5, B6)."""
import sys

import numpy as np
import pytest

from spectral_select.me_smartattention.splits import make_specimen_splits
from tests.me_attention.conftest import make_me


@pytest.mark.parametrize("counts", [[3, 3, 3], [4, 4, 4, 4], [2, 5, 3]])
def test_specimen_splits_cover_classes(counts):
    d = make_me(counts)
    for seed in range(20):
        splits = make_specimen_splits(d.X, d.y, d.groups, seed=seed)
        assert len(splits) == min(counts)
        for tr, te in splits:
            assert set(np.unique(d.y[te])) == set(range(len(counts)))
            assert set(d.groups[tr]).isdisjoint(d.groups[te])


def test_no_einops_and_edge_mode_default():
    import inspect
    from spectral_select.me_smartattention import smart_attention, training
    assert "einops" not in sys.modules
    assert inspect.signature(smart_attention.MESmartAttention.__init__).parameters["edge_mode"].default == "zero"
    assert inspect.signature(training.run_smart_attention).parameters["edge_mode"].default == "zero"
    from spectral_select.me_smartattention import baseline
    assert training.specimen_vote_accuracy is baseline.specimen_vote_accuracy


@pytest.mark.parametrize("edge", ["zero", "discard", "circular"])
def test_model_forward(edge):
    import torch
    from spectral_select.me_smartattention.smart_attention import MESmartAttention
    m = MESmartAttention(n_excitations=3, n_emissions=8, n_classes=3, dim=16, heads=2, dim_head=8,
                         edge_mode=edge).eval()
    out = m(torch.randn(5, 3, 8))
    assert out.shape == (5, 3) and torch.isfinite(out).all()


def test_training_smoke_and_fixed_f1():
    from spectral_select.me_smartattention.training import run_smart_attention
    from spectral_select.me_smartattention.baseline import run_logistic_regression
    rng = np.random.default_rng(0)
    d = make_me([3, 3, 3], px=30)
    X_eem = rng.normal(size=(len(d.y), 2, 5)).astype(np.float32) + d.y[:, None, None]
    mask = np.ones((2, 5), bool)
    data = dict(X_eem=X_eem, eem_mask=mask, y=d.y, groups=d.groups)
    sp = make_specimen_splits(X_eem.reshape(len(d.y), -1), d.y, d.groups, seed=0)
    r = run_smart_attention(data, sp, epochs=1, samples_per_specimen=64, batch_size=32, dim=8, heads=1,
                            dim_head=4, device="cpu")
    assert r["specimen_total"] == 9 and 0 <= r["macro_f1"] <= 1
    l = run_logistic_regression(X_eem.reshape(len(d.y), -1), d.y, d.groups, sp)
    assert l["specimen_total"] == 9
