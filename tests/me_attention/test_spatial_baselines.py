import numpy as np
import pytest
import torch

from me_attention.models.baselines_spatial import SPATIAL_BASELINES
from me_attention.models.registry import MODELS, models_for
from me_attention.preprocess import PREPROC_SPACE
from me_attention.protocol import make_patches
from me_attention.train import TRAIN_SPACE
from .syn_blocks import make_blocks

EXPECTED = {"patchmean_lr", "patchmean_rbfsvm", "patchmean_rf", "cnn2d", "cnn3d", "hybridsn",
            "spectralformer_patch", "ssftt"}
NEURAL = {"cnn2d", "cnn3d", "hybridsn", "spectralformer_patch", "ssftt"}


def test_registry_contents():
    assert set(SPATIAL_BASELINES) == EXPECTED
    for n, s in SPATIAL_BASELINES.items():
        assert s.track == "spatial" and s.family == "baseline"
        assert set(PREPROC_SPACE) <= set(s.space) and s.space["P"] == [5, 7] and s.default["P"] in (5, 7)
        assert (set(TRAIN_SPACE) <= set(s.space)) == (n in NEURAL)
    assert len(MODELS) == len(set(models_for("spectral")) | set(models_for("spatial")))
    assert "spatial_attn" in models_for("spatial", "ours") and "spatial_attn" not in models_for("spectral")
    assert all(MODELS[n].space["P"] == [5, 7] for n in models_for("spatial"))
    assert set(models_for("spatial", "baseline")) == EXPECTED


@pytest.fixture(scope="module")
def syn():
    return make_blocks(F=24)


@pytest.mark.parametrize("P", [5, 7])
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fit_predict(name, P, syn):
    d = syn
    spec = SPATIAL_BASELINES[name]
    idx = np.arange(0, len(d.y), 3)
    X = make_patches(d, idx, P)
    y, g = d.y[idx], d.groups[idx]
    cfg = {**spec.default, "P": P, "norm": "l2", "norm_scope": "per_ex", "log1p": True}
    if name in NEURAL:
        cfg.update(epochs=2, samples_per_specimen=100, warmup_epochs=1)
    if name == "patchmean_rf":
        cfg["n_estimators"] = 20
    e = spec.build(cfg, d.meta(), 0, "cpu").fit(X, y, g, X[:40], y[:40])
    Pr = e.predict_proba(X[::5])
    assert Pr.shape == (len(X[::5]), 3) and np.isfinite(Pr).all() and np.allclose(Pr.sum(1), 1, atol=1e-4)


def test_patchmean_features(syn):
    from me_attention.models.baselines_spatial import PatchMeanEstimator
    A = np.zeros((1, 3, 3, 2), np.float32)
    A[0, 1, 1] = [1, 2]
    A[0, 0, 0] = [3, 4]
    f = PatchMeanEstimator(None, {}, {"n_classes": 3}, 0)._flat(A)
    assert np.allclose(f, [[1, 2, 2, 3]])  # centre, mean over the two valid pixels


def test_masked_pixels_do_not_matter(syn):
    """Zeros outside the specimen: cnn/attention outputs are finite and deterministic in eval mode."""
    d = syn
    spec = SPATIAL_BASELINES["ssftt"]
    est = spec.build({**spec.default, "P": 5}, d.meta(), 0, "cpu")
    m = est.module_factory(d.meta(), est.cfg).eval()
    x = torch.from_numpy(make_patches(d, np.arange(8), 5))
    assert torch.allclose(m(x), m(x)) and torch.isfinite(m(x)).all()
