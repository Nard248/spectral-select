"""Objectives J(S) (docs/rl_select/SPEC.md section 4): JSup (labels), JRecon / JReconPC (label-free)."""
from __future__ import annotations

import functools
from typing import Sequence

import numpy as np

from .data import inner_folds
from .types import Task

RIDGE = 1e-6


def _quiet(fn):
    """Silence spurious FP flags raised by Apple Accelerate matmul (results are finite)."""
    @functools.wraps(fn)
    def w(*a, **k):
        with np.errstate(all="ignore"):
            return fn(*a, **k)
    return w


def _as_set(S: Sequence[int], F: int) -> np.ndarray:
    """Validate S (unique, in range) and return it as a sorted int array."""
    a = np.asarray(list(S), dtype=np.int64)
    if a.ndim != 1:
        raise ValueError("S must be a 1-D sequence of indices")
    if a.size and (a.min() < 0 or a.max() >= F):
        raise ValueError(f"index out of range [0,{F})")
    if len(np.unique(a)) != a.size:
        raise ValueError("S contains duplicate indices")
    return np.sort(a)


def correlation_matrix(task: Task) -> np.ndarray:
    """Pearson correlation of the F features over all training pixels (float64, cached on the task)."""
    if "corr" not in task.info:
        X = task.Xtr
        F = X.shape[1]
        G = np.zeros((F, F))
        for s in range(0, len(X), 20000):
            c = X[s:s + 20000].astype(np.float64)
            with np.errstate(all="ignore"):  # Accelerate raises spurious FP flags in matmul
                G += c.T @ c
        G /= len(X)  # features are z-scored with training stats -> mean 0, var 1 (0 for constant)
        d = np.sqrt(np.maximum(np.diag(G), 1e-12))
        task.info["corr"] = np.clip(G / d[:, None] / d[None, :], -1.0, 1.0)
    return task.info["corr"]


def pca_targets(task: Task, q: int = 20) -> np.ndarray:
    """Cross-covariance [F, q] between features and the top-q whitened PCA scores: sqrt(lam_j) * V[:, j]."""
    key = f"pca_{q}"
    if key not in task.info:
        w, V = np.linalg.eigh(correlation_matrix(task))
        q = min(q, task.n_features)
        order = np.argsort(w)[::-1][:q]
        task.info[key] = V[:, order] * np.sqrt(np.maximum(w[order], 0.0))[None, :]
    return task.info[key]


@_quiet
def residual_stats(Sigma: np.ndarray, S: Sequence[int], C: np.ndarray | None = None,
                   total_var: float | None = None) -> dict[str, np.ndarray]:
    """Residual variance and R2 gain/loss of adding/removing every candidate (vectorised).

    Targets t have feature-target cross-covariance ``C`` [F, T] (default ``Sigma``: reconstruct all
    features) and total variance ``total_var`` (default trace(Sigma)). Returns resid_var [F]
    (Sigma_cc - Sigma_cS Sigma_SS^-1 Sigma_Sc), gain_add [F] (R2 gain if c added; 0 for c in S) and
    loss_remove [F] (R2 loss if c removed; 0 for c not in S).
    """
    F = Sigma.shape[0]
    C = Sigma if C is None else C
    tv = float(np.trace(Sigma)) if total_var is None else float(total_var)
    S = _as_set(S, F)
    inS = np.zeros(F, dtype=bool)
    inS[S] = True
    if S.size == 0:
        rv = np.diag(Sigma).copy()
        Rc = C
        loss = np.zeros(F)
    else:
        P = np.linalg.inv(Sigma[np.ix_(S, S)] + RIDGE * np.eye(S.size))
        B = Sigma[:, S] @ P                                  # [F, s] regression weights of each c on S
        rv = np.diag(Sigma) - np.einsum("fs,fs->f", B, Sigma[:, S])
        Rc = C - B @ C[S]                                    # residual cross-cov with targets [F, T]
        loss = np.zeros(F)
        A = C[S].T @ P                                       # [T, s]; column j = (C_S^T P) e_j
        loss[S] = (A ** 2).sum(0) / np.diag(P) / tv
    rv_safe = np.maximum(rv, 1e-9)
    gain = (Rc ** 2).sum(1) / rv_safe / tv
    gain[inS] = 0.0
    gain[rv < 1e-9] = 0.0
    return {"resid_var": rv, "gain_add": gain, "loss_remove": loss}


class _Base:
    """Shared caching, counting and optional penalty terms (lam_corr * R_corr + mu_hw * #excitations)."""

    name = "base"
    uses_labels = False

    def __init__(self, task: Task, lam_corr: float = 0.0, mu_hw: float = 0.0):
        self.F = task.n_features
        self.lam_corr, self.mu_hw = float(lam_corr), float(mu_hw)
        self.ex_nm = np.asarray(task.ex_nm)
        self._corr = np.abs(correlation_matrix(task)) if lam_corr else None
        self.n_evals = 0
        self.use_cache = True
        self._cache: dict[frozenset, float] = {}

    def _base(self, S: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    def penalty(self, S: np.ndarray) -> float:
        p = 0.0
        if self.lam_corr and S.size > 1:
            sub = self._corr[np.ix_(S, S)]
            p += self.lam_corr * (sub.sum() - S.size) / (S.size * (S.size - 1))
        if self.mu_hw:
            p += self.mu_hw * len(np.unique(self.ex_nm[S]))
        return p

    def __call__(self, S: Sequence[int]) -> float:
        key = frozenset(int(i) for i in S)
        if self.use_cache and key in self._cache:
            return self._cache[key]
        a = _as_set(S, self.F)
        v = float(self._base(a) - self.penalty(a))
        self.n_evals += 1
        if self.use_cache:
            self._cache[key] = v
        return v

    def reset_counter(self) -> None:
        self.n_evals = 0

    def clear_cache(self) -> None:
        self._cache.clear()


class JSup(_Base):
    """Mean inner-fold balanced accuracy of shrinkage LDA (equal priors) with precomputed scatter."""

    name = "jsup"
    uses_labels = True

    def __init__(self, task: Task, alpha: float = 0.1, lam_corr: float = 0.0, mu_hw: float = 0.0):
        super().__init__(task, lam_corr, mu_hw)
        self.alpha, self.C = float(alpha), task.n_classes
        X = task.Xtr[task.sub_idx].astype(np.float64)
        y = task.ytr[task.sub_idx]
        folds = inner_folds(task)
        K, C, F = len(folds), self.C, self.F
        self.sums = np.zeros((K, C, F)); self.counts = np.zeros((K, C))
        self.scatter = np.zeros((K, F, F)); self.dof = np.zeros(K)
        self.val: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for k, (tr, va) in enumerate(folds):
            Xt, yt = X[tr], y[tr]
            for c in range(C):
                Xc = Xt[yt == c]
                if len(Xc) == 0:
                    raise ValueError("inner-fold train part misses a class")
                self.counts[k, c] = len(Xc)
                self.sums[k, c] = Xc.sum(0)
                Z = Xc - Xc.mean(0)
                with np.errstate(all="ignore"):
                    self.scatter[k] += Z.T @ Z
            self.dof[k] = len(Xt) - C
            yv = y[va]
            cnt = np.bincount(yv, minlength=C).astype(float)
            if (cnt == 0).any():
                raise ValueError("inner-fold val part misses a class")
            self.val.append((X[va], yv, cnt))
        self.means = self.sums / self.counts[:, :, None]

    def _base(self, S: np.ndarray) -> float:
        C = self.C
        if S.size == 0:
            return 1.0 / C
        s = S.size
        Sw = self.scatter[:, S][:, :, S] / self.dof[:, None, None]
        tr = np.trace(Sw, axis1=1, axis2=2) / s
        Sw = (1 - self.alpha) * Sw + (self.alpha * tr)[:, None, None] * np.eye(s)
        M = np.transpose(self.means[:, :, S], (0, 2, 1))                 # [K, s, C]
        with np.errstate(all="ignore"):
            A = np.linalg.solve(Sw, M)                                   # [K, s, C]
        if not np.isfinite(A).all():
            raise FloatingPointError("non-finite LDA solution")
        bias = 0.5 * (M * A).sum(1)                                      # [K, C]
        tot = 0.0
        for k, (Xv, yv, cnt) in enumerate(self.val):
            with np.errstate(all="ignore"):
                pred = (Xv[:, S] @ A[k] - bias[k]).argmax(1)
            ok = np.bincount(yv[pred == yv], minlength=C)
            tot += float((ok / cnt).mean())
        return tot / len(self.val)


class _Recon(_Base):
    """Linear reconstruction R2 of targets with cross-covariance Cxt from the features in S."""

    def __init__(self, task: Task, Cxt: np.ndarray, total_var: float, lam_corr: float = 0.0, mu_hw: float = 0.0):
        super().__init__(task, lam_corr, mu_hw)
        self.Sigma = correlation_matrix(task)
        self.Cxt, self.total_var = Cxt, float(total_var)

    @_quiet
    def _base(self, S: np.ndarray) -> float:
        if S.size == 0:
            return 0.0
        A = self.Sigma[np.ix_(S, S)] + RIDGE * np.eye(S.size)
        Cs = self.Cxt[S]
        return float((Cs * np.linalg.solve(A, Cs)).sum() / self.total_var)

    def stats(self, S: Sequence[int]) -> dict[str, np.ndarray]:
        """residual_stats for this objective's targets (not counted as evaluations)."""
        return residual_stats(self.Sigma, S, self.Cxt, self.total_var)


class JRecon(_Recon):
    """Label-free R2 of reconstructing all F standardized features from S."""

    name = "jrecon"

    def __init__(self, task: Task, **kw):
        Sig = correlation_matrix(task)
        super().__init__(task, Sig, float(np.trace(Sig)), **kw)


class JReconPC(_Recon):
    """Label-free R2 of reconstructing the top-q whitened PCA scores (q = min(20, F)) from S."""

    name = "jreconpc"

    def __init__(self, task: Task, q: int = 20, **kw):
        Ct = pca_targets(task, q)
        super().__init__(task, Ct, float(Ct.shape[1]), **kw)


OBJECTIVES = {"jsup": JSup, "jrecon": JRecon, "jreconpc": JReconPC}


def make_objective(name: str, task: Task, **kw):
    """Factory: name in {'jsup','jrecon','jreconpc'}; kw forwarded (alpha, lam_corr, mu_hw, q)."""
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective {name!r}; expected {list(OBJECTIVES)}")
    return OBJECTIVES[name](task, **kw)
