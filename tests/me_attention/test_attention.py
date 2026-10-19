"""Tests for me_attention.models.attention (cpu only, fast)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from me_attention.models import attention as A

META = A.synthetic_meta(n_features=60, n_ex=6, n_classes=3, seed=0)
F = META["n_features"]

SPECTRAL = {
    "v0_zero": (A.smartattention_v0_factory, dict(edge_mode="zero", dim=16, depth=3)),
    "v0_discard": (A.smartattention_v0_factory, dict(edge_mode="discard", dim=16, depth=4)),
    "meas_cls": (A.meas_tokens_factory, dict(dim=32, depth=2, pool="cls", token_drop=0.2)),
    "meas_mean_learned": (A.meas_tokens_factory, dict(dim=32, depth=2, pool="mean", pos="learned", val_embed="mlp")),
    "meas_attn_both": (A.meas_tokens_factory, dict(dim=32, depth=2, pool="attn", pos="both", token_drop=0.2)),
    "meas_relbias": (A.meas_tokens_factory, dict(dim=32, depth=2, relbias=True)),
    "meas_gated": (A.meas_tokens_factory, dict(dim=32, depth=2, gated=True, gate_l1=0.1, pool="mean")),
    "axial": (A.axial_factory, dict(dim=32, depth=2, token_drop=0.2)),
    "axial_attn_rel": (A.axial_factory, dict(dim=32, depth=1, pool="attn", relbias=True)),
    "ex_linear": (A.ex_tokens_factory, dict(dim=32, depth=2)),
    "ex_conv_sub": (A.ex_tokens_factory, dict(dim=32, depth=2, ex_embed="conv", sub_tokens=2, pool="mean", token_drop=0.2)),
    "ex_linear_sub4": (A.ex_tokens_factory, dict(dim=32, depth=2, sub_tokens=4)),
}


def _x(b=8, seed=0):
    return torch.randn(b, F, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", list(SPECTRAL))
def test_forward_backward(name):
    fac, cfg = SPECTRAL[name]
    m = fac(META, cfg)
    x = _x()
    m.train()
    out = m(x)
    assert out.shape == (8, 3) and torch.isfinite(out).all()
    (out.sum() + m.extra_loss()).backward()
    nograd = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    # relbias output layer / unused conv heads may legitimately have zero grads but must exist and be finite
    assert not nograd, nograd
    m.eval()
    assert torch.isfinite(m(x)).all()


@pytest.mark.parametrize("name", list(SPECTRAL))
def test_attention_importance(name):
    fac, cfg = SPECTRAL[name]
    m = fac(META, cfg)
    for mode in ("rollout", "last"):
        s = m.attention_importance(_x(), mode)
        assert s.shape == (F,) and np.isfinite(s).all()


def test_axial_masked_rows_cols_no_nan():
    meta = dict(META)
    gi = np.asarray(meta["grid_index"])
    keep = gi[:, 0] != 1  # excitation 1 fully missing -> fully masked row
    keep &= gi[:, 1] != gi[0, 1]  # one emission column removed for the rest
    keep |= np.arange(len(gi)) < 3
    for k in ("ex_nm", "em_nm", "grid_index"):
        meta[k] = np.asarray(meta[k])[keep]
    meta["n_features"] = int(keep.sum())
    for kw in (dict(pool="mean"), dict(pool="attn", relbias=True, token_drop=0.5)):
        m = A.axial_factory(meta, dict(dim=32, depth=2, **kw)).train()
        out = m(torch.randn(4, meta["n_features"]))
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_ex_tokens_empty_excitation_no_nan():
    meta = dict(META)
    gi = np.asarray(meta["grid_index"])
    keep = gi[:, 0] != 2
    for k in ("ex_nm", "em_nm", "grid_index"):
        meta[k] = np.asarray(meta[k])[keep]
    meta["n_features"] = int(keep.sum())
    out = A.ex_tokens_factory(meta, dict(dim=32, depth=2, sub_tokens=2, token_drop=0.5)).train()(torch.randn(4, meta["n_features"]))
    assert torch.isfinite(out).all()


def test_meas_tokens_permutation_equivariance():
    """Without learned positions the encoder is equivariant if wavelengths are permuted with the features."""
    m = A.MeasTokens(META, 3, dim=32, depth=2, pos="fourier", pool="mean").eval()
    x = _x(4)
    perm = torch.randperm(F, generator=torch.Generator().manual_seed(1))
    m2 = A.MeasTokens(META, 3, dim=32, depth=2, pos="fourier", pool="mean").eval()
    m2.load_state_dict(m.state_dict())
    for k in ("ex_nm", "em_nm"):
        pass
    meta_p = dict(META)
    for k in ("ex_nm", "em_nm", "grid_index"):
        meta_p[k] = np.asarray(META[k])[perm.numpy()]
    mp = A.MeasTokens(meta_p, 3, dim=32, depth=2, pos="fourier", pool="mean").eval()
    mp.load_state_dict(m.state_dict())
    assert torch.allclose(m(x), mp(x[:, perm]), atol=1e-5)


def test_gate_paths():
    m = A.MeasTokens(META, 3, dim=32, depth=2, gated=True, gate_l1=0.5)
    g = m.feature_importance()
    assert g.shape == (F,) and ((g > 0) & (g < 1)).all()
    assert m.extra_loss().item() > 0
    assert A.MeasTokens(META, 3, dim=32, depth=2).extra_loss().item() == 0
    # l1 pushes gates down
    opt = torch.optim.Adam(m.parameters(), 0.05)
    for _ in range(20):
        opt.zero_grad()
        m.extra_loss().backward()
        opt.step()
    assert m.feature_importance().mean() < g.mean()


def test_relbias_changes_output_after_update():
    m = A.MeasTokens(META, 3, dim=32, depth=2, relbias=True).eval()
    x = _x(4)
    a = m(x)
    with torch.no_grad():
        m.relbias.mlp[2].weight.add_(torch.randn_like(m.relbias.mlp[2].weight))
    assert not torch.allclose(a, m(x))
    assert m.relbias().shape == (1, 4, F, F)


def test_token_drop_only_in_training():
    m = A.MeasTokens(META, 3, dim=32, depth=2, token_drop=0.5).eval()
    x = _x(4)
    assert torch.allclose(m(x), m(x))


# ------------------------------------------------------------------ spatial
@pytest.mark.parametrize("cfg", [
    dict(spec_encoder="mlp", dim=32),
    dict(spec_encoder="ex", dim=32, centre_weight=True, token_drop=0.3),
    dict(spec_encoder="meas", dim=32, readout="centre", centre_weight=True, depth=1),
    dict(spec_encoder="ex", dim=32, readout="centre"),
])
def test_spatial(cfg):
    m = A.spatial_factory(META, cfg)
    P = 5
    x = torch.randn(6, P, P, F)
    x[0, :, :] = 0
    x[0, P // 2, P // 2] = torch.randn(F)  # only the centre is valid
    x[1, :2, :] = 0  # whole rows masked
    x[2:4, 0, 0] = 0
    m.train()
    out = m(x)
    assert out.shape == (6, 3) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    s = m.attention_importance(x)
    assert s.shape == (P, P) and np.isfinite(s).all()
    # smaller patch works too
    assert m.eval()(x[:, 1:4, 1:4]).shape == (6, 3)


def test_spatial_masked_pixels_do_not_matter():
    m = A.spatial_factory(META, dict(spec_encoder="ex", dim=32)).eval()
    x = torch.randn(3, 5, 5, F)
    x[:, 0] = 0
    y = x.clone()
    assert torch.allclose(m(x), m(y))
    assert torch.isfinite(m(torch.zeros(2, 5, 5, F))).all()  # everything masked incl. centre


# ------------------------------------------------------------------ learnability
def _train_toy(model, steps=160, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(1200, F, generator=g)
    y = (X[:, 17] > 0.4).long() + (X[:, 17] > -0.4).long()  # class determined by one (ex, em) feature
    X = X + 0.0
    opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=1e-4)
    model.train()
    for i in range(steps):
        idx = torch.randint(0, 900, (64,), generator=g)
        loss = torch.nn.functional.cross_entropy(model(X[idx]), y[idx]) + model.extra_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return (model(X[900:]).argmax(1) == y[900:]).float().mean().item()


@pytest.mark.parametrize("name,fac,cfg", [
    ("meas", A.meas_tokens_factory, dict(dim=32, depth=2, pos="both", token_drop=0.0, dropout=0.0)),
    ("meas_gated", A.meas_tokens_factory, dict(dim=32, depth=2, gated=True, gate_l1=0.01, pool="attn", dropout=0.0, token_drop=0.0)),
    ("axial", A.axial_factory, dict(dim=16, depth=1, heads=2, dropout=0.0, token_drop=0.0)),
    ("ex", A.ex_tokens_factory, dict(dim=32, depth=2, dropout=0.0, token_drop=0.0, sub_tokens=2)),
    ("v0", A.smartattention_v0_factory, dict(dim=32, depth=2, dropout=0.0, embedding_dropout=0.0)),
])
def test_toy_learnable(name, fac, cfg):
    torch.manual_seed(0)
    acc = _train_toy(fac(META, cfg), steps=450 if name == "v0" else 160)
    assert acc > 0.9, (name, acc)


def test_registry():
    for n, spec in A.ATTENTION_MODELS.items():
        assert spec.family == "ours" and spec.track == ("spatial" if n == "spatial_attn" else "spectral")
        assert set(A.MODEL_SPACES[n]) <= set(spec.space)
        assert set(A.MODEL_SPACES[n]) <= set(spec.default)  # every model hyper-param has a default
        for k, v in A.MODEL_SPACES[n].items():
            assert spec.default[k] in v or k in ("dim_head",), (n, k)
    # sampled configs build and run
    rng = np.random.default_rng(0)
    for n, spec in A.ATTENTION_MODELS.items():
        fac = A._FACTORIES[n]
        for _ in range(6):
            cfg = {k: v[rng.integers(len(v))] for k, v in A.MODEL_SPACES[n].items()}
            m = fac(META, cfg)
            x = torch.randn(2, 3, 3, F) if n == "spatial_attn" else torch.randn(2, F)
            assert torch.isfinite(m.eval()(x)).all(), (n, cfg)
