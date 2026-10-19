import numpy as np
import torch
from torch import nn

from me_attention.train import SklearnEstimator, TorchEstimator, balanced_accuracy
from .toy import meta, toy

CFG = dict(epochs=8, samples_per_specimen=300, batch_size=128, warmup_epochs=1, lr=3e-3)


def _factory(m, c):
    return nn.Sequential(nn.Linear(m["n_features"], 32), nn.GELU(), nn.Linear(32, m["n_classes"]))


def _split():
    X, y, g = toy()
    te = np.isin(g, [2, 5, 8])  # one specimen per class held out
    return X[~te], y[~te], g[~te], X[te], y[te]


def _est(cfg=None, seed=0):
    return TorchEstimator(_factory, {**CFG, **(cfg or {})}, meta(), seed, "cpu")


def test_learns_separable():
    X, y, g, Xt, yt = _split()
    e = _est(dict(norm="l2", log1p=True)).fit(X, y, g)
    P = e.predict_proba(Xt)
    assert P.shape == (len(Xt), 3) and np.allclose(P.sum(1), 1, atol=1e-5)
    assert balanced_accuracy(yt, P.argmax(1), 3) > 0.9
    assert len(e.history_) == 8 and e.best_epoch_ == 8


def test_aug_modes_run():
    X, y, g, Xt, yt = _split()
    for aug in ["scale", "scale+noise"]:
        e = _est(dict(aug=aug, epochs=3)).fit(X, y, g)
        assert np.isfinite(e.predict_proba(Xt)).all()


def test_early_stopping_restores_best():
    X, y, g, Xt, yt = _split()
    e = _est(dict(epochs=30, patience=3)).fit(X, y, g, Xt, yt)
    vals = [h["val_bacc"] for h in e.history_]
    assert e.best_epoch_ == int(np.argmax(vals)) + 1
    assert len(vals) <= 30
    pred = e.predict_proba(Xt).argmax(1)
    assert abs(balanced_accuracy(yt, pred, 3) - max(vals)) < 1e-6


def test_determinism():
    X, y, g, Xt, yt = _split()
    a = _est(seed=3).fit(X, y, g).predict_proba(Xt)
    b = _est(seed=3).fit(X, y, g).predict_proba(Xt)
    c = _est(seed=4).fit(X, y, g).predict_proba(Xt)
    assert np.allclose(a, b, atol=1e-6) and not np.allclose(a, c, atol=1e-4)


def test_spatial_input():
    X, y, g, Xt, yt = _split()
    rng = np.random.default_rng(0)
    def patches(A):
        P = np.zeros((len(A), 3, 3, 40), np.float32)
        P[:, 1, 1] = A
        P[:, 0, 1] = A * 0.9
        return P
    class Mod(nn.Module):
        def __init__(self):
            super().__init__(); self.l = nn.Linear(40, 3)
        def forward(self, x):
            return self.l(x[:, 1, 1])
    e = TorchEstimator(lambda m, c: Mod(), {**CFG, "aug": "scale+noise", "epochs": 3}, meta(), 0, "cpu",
                       input_kind="spatial").fit(patches(X[::5]), y[::5], g[::5])
    assert e.predict_proba(patches(Xt[:50])).shape == (50, 3)


def test_sklearn_estimator():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    X, y, g, Xt, yt = _split()
    for mk, kw in [(lambda c, m, s: LogisticRegression(max_iter=200), {}),
                   (lambda c, m, s: LinearSVC(dual="auto"), dict(max_train=500))]:
        e = SklearnEstimator(mk, {"max_train": kw.get("max_train"), "norm": "l2"}, meta(), 0).fit(X, y, g)
        P = e.predict_proba(Xt)
        assert np.allclose(P.sum(1), 1) and P.shape[1] == 3
        assert balanced_accuracy(yt, P.argmax(1), 3) > 0.8


def test_ondevice_preproc_matches_numpy_and_ex_drop():
    import numpy as np
    import torch
    from me_attention.models.attention import synthetic_meta
    from me_attention.train import TorchEstimator
    meta = synthetic_meta(n_features=40, n_ex=4)
    rng = np.random.default_rng(0)
    X = rng.gamma(2.0, 100.0, size=(300, 40)).astype(np.float32)
    est = TorchEstimator(lambda m, c: torch.nn.Linear(40, 3), dict(norm="none", log1p=True), meta, 0, "cpu")
    from me_attention.preprocess import make_preprocessor
    est.pre = make_preprocessor(est.cfg, meta).fit(X)
    est._t_mean, est._t_std = torch.from_numpy(est.pre.mean_), torch.from_numpy(est.pre.std_)
    np.testing.assert_allclose(est._tpre_none(torch.from_numpy(X)).numpy(), est.pre.transform(X), atol=1e-4)
    ids = np.unique(meta["ex_nm"], return_inverse=True)[1]
    est._n_ex, est._ex_ids_t = int(ids.max() + 1), torch.from_numpy(ids)
    out = est._t_ex_drop(torch.ones(500, 40), 0.5).numpy()
    assert (out.sum(1) > 0).all()                      # at least one excitation survives
    for e in range(est._n_ex):                         # whole blocks are dropped together
        blk = out[:, ids == e]
        assert ((blk.min(1) == blk.max(1))).all()
    assert 0.2 < (out[:, 0] == 0).mean() < 0.8
