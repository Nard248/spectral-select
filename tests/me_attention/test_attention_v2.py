import numpy as np
import torch

from me_attention.models import attention_v2 as V
from me_attention.models.attention import synthetic_meta


def _meta():
    return synthetic_meta(n_features=40, n_ex=4, n_classes=3)


def test_hybrid_forward_grad():
    meta = _meta()
    m = V.hybrid_factory(meta, dict(dim=32, depth=1, heads=2, n_chunks=4))
    x = torch.randn(5, meta["n_features"], requires_grad=True)
    out = m(x)
    assert out.shape == (5, 3) and torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_centered_brightness_invariance():
    meta = _meta()
    inner = V.hybrid_factory(meta, dict(dim=32, depth=1, heads=2, n_chunks=4, dropout=0.0, token_drop=0.0))
    for mode in ("pixel", "ex", "both"):
        m = V.Centered(inner, meta, mode).eval()
        x = torch.randn(6, meta["n_features"])
        shift = x + 2.5  # multiplicative drift in log space (global)
        assert torch.allclose(m(x), m(shift), atol=1e-4)


def test_v2_estimator_all_objectives_run():
    meta = _meta()
    rng = np.random.default_rng(0)
    F = meta["n_features"]
    y = np.repeat(np.arange(3), 80)
    g = np.repeat(np.arange(6), 40)           # 2 specimens per class
    X = (rng.random((240, F)) + 1 + y[:, None] * 0.3).astype(np.float32)
    cfg = dict(dim=16, depth=1, heads=2, n_chunks=3, epochs=2, samples_per_specimen=64, batch_size=32, log1p=True,
               standardize=False, center="pixel", aug="scale+noise", aug_ex_scale=0.4, aug_smooth=0.3, ex_drop=0.2,
               mixup=0.4, supcon_w=0.2, adv_w=1.0, fixed_epochs=True)
    est = V.TorchEstimatorV2(V.hybrid_factory, cfg, meta, 0, "cpu").fit(X, y, g)
    P = est.predict_proba(X)
    assert P.shape == (240, 3) and np.isfinite(P).all()
    assert not est.degenerate_


def test_adversary_degenerate_flag_when_one_specimen_per_class():
    meta = _meta()
    X = (np.random.default_rng(0).random((90, meta["n_features"])) + 1).astype(np.float32)
    y = np.repeat(np.arange(3), 30)
    est = V.TorchEstimatorV2(V.hybrid_factory, dict(dim=16, depth=1, heads=2, epochs=1, samples_per_specimen=32,
                                                    adv_w=1.0, supcon_w=0.1), meta, 0, "cpu").fit(X, y, y.copy())
    assert est.degenerate_
