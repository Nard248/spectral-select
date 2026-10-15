"""Regression: Sobol' S1 must be invariant to an additive offset / scale of the output (centred estimator)."""
import numpy as np
import pytest

from spectraforge.studio import sensitivity as sens


TOL = 0.15  # finite-n (512) Saltelli error on Ishigami ST is ~0.1; same tolerance with and without offset


def _Y(n, offset, scale, k=3, seed=0):
    A, B, ABs = sens.saltelli_matrices(n, k, seed)
    f = lambda X: offset + scale * sens.ishigami(X)  # noqa: E731
    return np.concatenate([f(A), f(B)] + [f(M) for M in ABs]), k


def _old_s1(Y, n, k):
    fA, fB, fAB = Y[:n], Y[n:2 * n], Y[2 * n:].reshape(k, n)
    return np.mean(fB * (fAB - fA), axis=1) / np.var(np.concatenate([fA, fB]), ddof=1)


@pytest.mark.parametrize("offset,scale", [(0.0, 1.0), (1e4, 1.0), (1e4, 1e-3), (-3e3, 1e3)])
def test_s1_invariant_to_offset_and_scale(offset, scale):
    n = 512
    Y0, k = _Y(n, 0.0, 1.0)
    Y1, _ = _Y(n, offset, scale)
    r0 = sens.sobol_indices(Y0, n, k, n_boot=20)
    r1 = sens.sobol_indices(Y1, n, k, n_boot=20)
    s1, st = sens.ishigami_analytic()
    np.testing.assert_allclose(r1["S1"], r0["S1"], atol=1e-6)
    np.testing.assert_allclose(r1["ST"], r0["ST"], atol=1e-6)
    assert r1["S1"] == pytest.approx(s1, abs=TOL)
    assert r1["ST"] == pytest.approx(st, abs=TOL)


def test_old_uncentred_formula_fails_with_offset():
    n = 512
    Y, k = _Y(n, 1e4, 1.0)
    old = _old_s1(Y, n, k)
    s1, _ = sens.ishigami_analytic()
    assert np.max(np.abs(old - s1)) > 1.0          # garbage (|error| >> 1)
    assert np.max(np.abs(sens.sobol_indices(Y, n, k, n_boot=5)["S1"] - s1)) < TOL
