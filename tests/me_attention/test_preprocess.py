import numpy as np
import pytest

from me_attention.preprocess import PREPROC_SPACE, Preprocessor, make_preprocessor

EX = np.repeat([300.0, 350, 400, 450, 500], 8)


def _X(n=200, seed=0):
    return np.abs(np.random.default_rng(seed).normal(5, 2, (n, 40))).astype(np.float32) + 0.1


def test_l2_global_unit_norm():
    X = _X()
    p = Preprocessor(norm="l2", standardize=False)
    out = p.fit_transform(X)
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)


def test_per_ex_blocks():
    X = _X()
    out = Preprocessor(norm="l2", norm_scope="per_ex", standardize=False, ex_nm=EX).fit_transform(X)
    for e in np.unique(EX):
        assert np.allclose(np.linalg.norm(out[:, EX == e], axis=1), 1, atol=1e-5)
    out = Preprocessor(norm="area", norm_scope="per_ex", standardize=False, ex_nm=EX).fit_transform(X)
    assert np.allclose(out[:, EX == 300].sum(1), 1, atol=1e-5)
    out = Preprocessor(norm="max", standardize=False).fit_transform(X)
    assert np.allclose(out.max(1), 1, atol=1e-6)


def test_standardize_train_only_and_guard():
    X = _X()
    X[:, 3] = 7.0  # constant feature
    p = Preprocessor().fit(X)
    out = p.transform(X)
    assert np.allclose(out.mean(0)[[0, 1]], 0, atol=1e-4) and np.isfinite(out).all()
    assert np.allclose(out[:, 3], 0)
    Xt = _X(seed=1) + 100
    assert np.abs(p.transform(Xt).mean()) > 10  # test data uses train statistics, not refit


def test_log1p_clips_negatives():
    X = _X()
    X[0, 0] = -5
    out = Preprocessor(log1p=True).fit_transform(X)
    assert np.isfinite(out).all()


def test_patch_masked_pixels_stay_zero():
    rng = np.random.default_rng(0)
    P = np.abs(rng.normal(5, 2, (30, 3, 3, 40))).astype(np.float32) + 0.1
    P[:, 0, :, :] = 0
    P[5, 1, 1, :] = 0
    for kw in [dict(), dict(norm="l2", log1p=True), dict(norm="area", norm_scope="per_ex", ex_nm=EX)]:
        p = Preprocessor(**kw)
        out = p.fit_transform(P)
        assert out.shape == P.shape and out.dtype == np.float32
        assert (out[:, 0] == 0).all() and (out[5, 1, 1] == 0).all()
        assert abs(out[:, 1:][np.any(P[:, 1:] != 0, -1)].mean()) < 0.1


def test_space_and_factory():
    assert set(PREPROC_SPACE) == {"norm", "norm_scope", "log1p"}
    p = make_preprocessor(dict(norm="l2", norm_scope="per_ex"), dict(ex_nm=EX))
    assert p.norm == "l2" and len(p._blocks) == 5
    with pytest.raises(ValueError):
        Preprocessor(norm="bad")
