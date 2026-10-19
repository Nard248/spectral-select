import numpy as np
import pytest

from me_attention.models.baselines_spectral import SPECTRAL_BASELINES
from me_attention.preprocess import PREPROC_SPACE
from me_attention.train import TRAIN_SPACE
from .toy import meta, toy

EXPECTED = {"lr", "linsvm", "rbfsvm", "rf", "et", "hgb", "knn", "mlp", "cnn1d", "spectralformer", "bandvit"}
NEURAL = {"mlp", "cnn1d", "spectralformer", "bandvit"}


def test_registry_contents():
    assert set(SPECTRAL_BASELINES) == EXPECTED
    for n, s in SPECTRAL_BASELINES.items():
        assert s.track == "spectral" and s.family == "baseline"
        assert set(PREPROC_SPACE) <= set(s.space)
        assert (set(TRAIN_SPACE) <= set(s.space)) == (n in NEURAL)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fit_predict(name):
    spec = SPECTRAL_BASELINES[name]
    X, y, g = toy(n=60)
    cfg = dict(spec.default)
    if name in NEURAL:
        cfg.update(epochs=2, samples_per_specimen=100, warmup_epochs=1)
    if name in ("rf", "et"):
        cfg["n_estimators"] = 20
    if name == "hgb":
        cfg["max_iter"] = 10
    cfg.update(norm="l2", norm_scope="per_ex", log1p=True)
    e = spec.build(cfg, meta(), 0, "cpu").fit(X, y, g, X[:30], y[:30])
    P = e.predict_proba(X[::7])
    assert P.shape == (len(X[::7]), 3) and np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-4)


def test_random_space_samples_build():
    rng = np.random.default_rng(0)
    X, y, g = toy(n=30)
    for name in NEURAL:
        spec = SPECTRAL_BASELINES[name]
        cfg = {k: v[rng.integers(len(v))] for k, v in spec.space.items()}
        cfg = {**spec.default, **cfg, "epochs": 1, "samples_per_specimen": 50, "warmup_epochs": 0}
        spec.build(cfg, meta(), 0, "cpu").fit(X, y, g).predict_proba(X[:5])
