"""Per-task precomputation and label-free / supervised candidate features for the RL selector.

``TaskFeatures`` holds the training correlation matrix, PCA targets, static per-candidate features and
(optionally) supervised features (Fisher rank, oracle priors from exact n=2 enumeration). ``state(S)``
returns the cached ``[F, d]`` candidate feature matrix plus JRecon / JReconPC of S. Everything is
normalised within the task so a policy transfers across datasets with different F.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .objective import correlation_matrix, pca_targets
from .types import Objective, Task

SUP_LEVELS = ("none", "fisher", "oracle")
BASE_NAMES = ["ex", "em", "stokes", "logmean", "logstd_rank",
              "in_S", "gain_rec", "gain_rec_rel", "loss_rec", "loss_rec_rel",
              "gain_pc", "gain_pc_rel", "loss_pc", "loss_pc_rel", "rvar",
              "maxcorr", "meancorr", "n_same_ex", "toggle_ex", "dem"]
GLOBAL_NAMES = ["size_frac", "n_over_20", "phase_add", "phase_rem", "cycles_left", "jrecon", "jpc"]
N_GLOBAL = len(GLOBAL_NAMES)
DEFAULT_ORACLE_DIR = Path("results/rl_select/oracle")


def residual_stats(Sigma: np.ndarray, C: np.ndarray, S: Sequence[int], ridge: float = 1e-6) -> dict:
    """Closed-form linear-reconstruction statistics for set S (private version, O(F^2|S|)).

    Sigma: [F,F] correlation of all features (targets of JRecon). C: [F,q] cross-covariance between
    features and the q whitened PCA scores (targets of JReconPC). Returns arrays over all F candidates:
    rvar (residual variance), gain_rec/gain_pc (R2 gain if added; meaningful for c not in S),
    loss_rec/loss_pc (R2 loss if removed; meaningful for c in S), and scalars jrec, jpc.
    """
    with np.errstate(all="ignore"):   # Accelerate raises spurious FP flags in matmul
        return _residual_stats(Sigma, C, S, ridge)


def _residual_stats(Sigma, C, S, ridge):
    F = Sigma.shape[0]
    q = C.shape[1]
    S = np.asarray(sorted(set(int(s) for s in S)), dtype=np.int64)
    eps = 1e-9
    in_S = np.zeros(F, bool)
    in_S[S] = True
    loss_rec = np.zeros(F)
    loss_pc = np.zeros(F)
    if len(S) == 0:
        rvar = np.diag(Sigma).copy()
        gain_rec = (Sigma ** 2).sum(0) / (F * np.maximum(rvar, eps))
        gain_pc = (C ** 2).sum(1) / (q * np.maximum(rvar, eps))
        return dict(rvar=rvar, gain_rec=gain_rec, gain_pc=gain_pc, loss_rec=loss_rec, loss_pc=loss_pc,
                    jrec=0.0, jpc=0.0)
    A = Sigma[np.ix_(S, S)] + ridge * np.eye(len(S))
    P = np.linalg.inv(A)
    B = Sigma[S, :]                     # [s,F]
    PB = P @ B                          # [s,F]
    Rm = Sigma - Sigma[:, S] @ PB       # residual covariance [F,F]
    rvar = np.diag(Rm).copy()
    gain_rec = (Rm ** 2).sum(0) / (F * np.maximum(rvar, eps))
    Cs = C[S, :]
    PC = P @ Cs                         # [s,q]
    D = C - Sigma[:, S] @ PC            # [F,q]
    gain_pc = (D ** 2).sum(1) / (q * np.maximum(rvar, eps))
    pdiag = np.diag(P)
    loss_rec[S] = (PB ** 2).sum(1) / pdiag / F
    loss_pc[S] = (PC ** 2).sum(1) / pdiag / q
    gain_rec[in_S] = 0.0
    gain_pc[in_S] = 0.0
    return dict(rvar=rvar, gain_rec=gain_rec, gain_pc=gain_pc, loss_rec=loss_rec, loss_pc=loss_pc,
                jrec=float((B * PB).sum() / F), jpc=float((Cs * PC).sum() / q))


@dataclass
class State:
    feats: np.ndarray   # [F, d] float32
    jrec: float
    jpc: float


def _rank01(x: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0,1] (ties averaged via stable argsort order)."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(0) - mu) ** 2
        den += len(Xc) * Xc.var(0)
    return num / np.maximum(den, 1e-12)


class TaskFeatures:
    """Precomputed per-task quantities + cached ``state(S)``."""

    def __init__(self, task: Task, objective: Objective | None = None, sup: str = "none",
                 oracle_dir: str | Path | None = None, cache_size: int = 2000, q: int = 20,
                 ridge: float = 1e-6):
        if sup not in SUP_LEVELS:
            raise ValueError(f"sup must be one of {SUP_LEVELS}, got {sup!r}")
        if sup == "oracle" and objective is None:
            raise ValueError("sup='oracle' needs an objective for the exact n=2 enumeration")
        self.task, self.sup, self.ridge = task, sup, ridge
        self.F = task.n_features
        self.valid = np.asarray(task.valid, bool)
        X = task.Xtr[task.sub_idx].astype(np.float64)
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)       # used only for supervised Fisher scores
        self.Sigma = correlation_matrix(task)                  # shared with objective.py (training pixels)
        self.absSigma = np.abs(self.Sigma)
        self.q = min(q, self.F)
        self.C = pca_targets(task, self.q)                     # cov(x_j, whitened PC score k)
        # static features
        z = lambda v: (v - v.mean()) / max(v.std(), 1e-8)
        ex, em = task.ex_nm.astype(float), task.em_nm.astype(float)
        self.static = np.stack([z(ex), z(em), z(em - ex), z(task.raw_logmean.astype(float)),
                                _rank01(task.raw_logstd.astype(float))], 1)
        self.ex = ex
        self.em = em
        _, self.ex_id = np.unique(ex, return_inverse=True)
        self.n_ex = int(self.ex_id.max()) + 1
        # per-task gain scale = best single-feature gain from the empty set
        st0 = residual_stats(self.Sigma, self.C, [], ridge)
        self.gain_scale = max(float(st0["gain_rec"][self.valid].max()), 1e-9)
        self.gain_scale_pc = max(float(st0["gain_pc"][self.valid].max()), 1e-9)
        # supervised
        self.fisher_rank = None
        self.pair_norm = None
        self.p_band = None
        self.oracle_cost = 0
        if sup in ("fisher", "oracle"):
            self.fisher_rank = _rank01(fisher_scores(X, task.ytr[task.sub_idx]))
        if sup == "oracle":
            self._build_oracle(objective, Path(oracle_dir) if oracle_dir else DEFAULT_ORACLE_DIR)
        self.d = len(BASE_NAMES) + {"none": 0, "fisher": 1, "oracle": 3}[sup]
        self._cache: dict[frozenset, State] = {}
        self._cache_size = cache_size
        self.names = BASE_NAMES + {"none": [], "fisher": ["fisher_rank"],
                                   "oracle": ["fisher_rank", "p_band", "p_pair"]}[sup]

    # ---- oracle priors -------------------------------------------------------------------------
    def _build_oracle(self, objective: Objective, odir: Path) -> None:
        F = self.F
        fname = f"{self.task.name.replace('/', '_')}__{getattr(objective, 'name', 'obj')}.npz"
        path = odir / fname
        self.oracle_cost = F * (F - 1) // 2
        if path.exists():
            M = np.load(path)["M"]
            if M.shape != (F, F):
                raise ValueError(f"oracle cache {path} has shape {M.shape}, expected {(F, F)}")
        else:
            M = np.full((F, F), np.nan)
            for i in range(F):
                for j in range(i + 1, F):
                    if self.valid[i] and self.valid[j]:
                        M[i, j] = M[j, i] = objective([i, j])
            odir.mkdir(parents=True, exist_ok=True)
            np.savez(path, M=M)
        iu = np.triu_indices(F, 1)
        vals = M[iu]
        ok = np.isfinite(vals)
        lo, hi = vals[ok].min(), vals[ok].max()
        Mn = np.where(np.isfinite(M), (M - lo) / max(hi - lo, 1e-12), 0.0)
        np.fill_diagonal(Mn, 0.0)
        self.pair_norm = Mn
        k = max(1, int(np.ceil(0.01 * ok.sum())))
        top = np.argsort(np.where(ok, vals, -np.inf))[::-1][:k]
        cnt = np.bincount(np.concatenate([iu[0][top], iu[1][top]]), minlength=F).astype(float)
        self.p_band = cnt / max(cnt.max(), 1.0)
        self.oracle_matrix = M

    # ---- core ----------------------------------------------------------------------------------
    def jrecon(self, S: Sequence[int]) -> float:
        return self.state(S).jrec

    def jpc(self, S: Sequence[int]) -> float:
        return self.state(S).jpc

    def state(self, S: Sequence[int]) -> State:
        key = frozenset(int(s) for s in S)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        idx = np.array(sorted(key), dtype=np.int64)
        if len(idx) and (idx.min() < 0 or idx.max() >= self.F):
            raise ValueError("selected index out of range")
        st = residual_stats(self.Sigma, self.C, idx, self.ridge)
        F = self.F
        in_S = np.zeros(F, bool)
        in_S[idx] = True
        nS, S_ = ~in_S & self.valid, in_S

        def rel(v, m):
            mx = v[m].max() if m.any() else 0.0
            return v / mx if mx > 1e-12 else np.zeros_like(v)

        g_r, g_p = st["gain_rec"] / self.gain_scale, st["gain_pc"] / self.gain_scale_pc
        l_r, l_p = st["loss_rec"] / self.gain_scale, st["loss_pc"] / self.gain_scale_pc
        gr_rel = np.where(nS, rel(st["gain_rec"], nS), 0.0)
        gp_rel = np.where(nS, rel(st["gain_pc"], nS), 0.0)
        lr_rel = np.where(S_, rel(st["loss_rec"], S_), 0.0)
        lp_rel = np.where(S_, rel(st["loss_pc"], S_), 0.0)
        if len(idx):
            ac = self.absSigma[:, idx]
            maxc = ac.max(1)
            meanc = ac.mean(1)
            cnt = np.bincount(self.ex_id[idx], minlength=self.n_ex)[self.ex_id]
            D = np.abs(self.em[:, None] - self.em[idx][None, :])
            D = np.where((self.ex_id[:, None] == self.ex_id[idx][None, :]) & (np.arange(F)[:, None] != idx[None, :]),
                         D, np.inf)
            dmin = D.min(1)
            dem = np.where(np.isfinite(dmin), np.minimum(dmin / 100.0, 3.0) / 3.0, 1.0)
        else:
            maxc = meanc = np.zeros(F)
            cnt = np.zeros(F, np.int64)
            dem = np.ones(F)
        toggle = np.where(in_S, cnt == 1, cnt == 0).astype(float)
        same = (cnt - in_S) / 4.0
        cols = [self.static, in_S[:, None].astype(float)]
        cols += [a[:, None] for a in (g_r, gr_rel, l_r, lr_rel, g_p, gp_rel, l_p, lp_rel, st["rvar"],
                                      maxc, meanc, same, toggle, dem)]
        if self.sup != "none":
            cols.append(self.fisher_rank[:, None])
        if self.sup == "oracle":
            pp = self.pair_norm[:, idx].mean(1) if len(idx) else np.zeros(F)
            cols += [self.p_band[:, None], pp[:, None]]
        feats = np.concatenate(cols, 1).astype(np.float32)
        assert feats.shape == (F, self.d)
        out = State(feats, st["jrec"], st["jpc"])
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = out
        return out


def candidate_features(tf: TaskFeatures, S_mask: np.ndarray, phase: int | None = None) -> np.ndarray:
    """[F, d] candidate features for selected-mask S_mask. Features do not depend on phase (the
    in_S flag, gains and losses are all present); ``phase`` only enters the global features."""
    S_mask = np.asarray(S_mask, bool)
    if S_mask.shape != (tf.F,):
        raise ValueError(f"S_mask must have shape ({tf.F},)")
    return tf.state(np.flatnonzero(S_mask)).feats


def global_features(tf: TaskFeatures, S_mask: np.ndarray, phase: int, n: int, cycle: int = 0, T: int = 0
                    ) -> np.ndarray:
    """[g] global context: |S|/n, n/20, phase one-hot (add, remove), cycles left / T, JRecon, JReconPC."""
    if phase not in (0, 1):
        raise ValueError("phase must be 0 (add) or 1 (remove)")
    st = tf.state(np.flatnonzero(np.asarray(S_mask, bool)))
    left = max(T - cycle, 0) / T if T > 0 else 0.0
    return np.array([np.sum(S_mask) / n, n / 20.0, float(phase == 0), float(phase == 1), left, st.jrec, st.jpc],
                    dtype=np.float32)
