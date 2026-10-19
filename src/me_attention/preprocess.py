"""Shared preprocessing for every ME-attention model (fit on training data only).

Pipeline (last axis = features F; works for spectral ``[n, F]`` and patch ``[n, P, P, F]``):

1. per-pixel normalisation ``norm`` in {none, l2, area, max}, over all features (``norm_scope="global"``) or
   within each excitation block (``"per_ex"``, blocks = unique values of ``ex_nm``);
2. optional ``log1p``: ``log1p(max(v, 0) / s)`` where ``s`` is the median positive value of the normalised
   training data (fitted scalar; makes the log non-trivial for unit-scale normalised data);
3. optional per-feature standardisation (statistics fitted on non-zero pixels only).

All-zero pixels (patch pixels masked outside the specimen) stay exactly zero after ``transform``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

NORMS = ["none", "l2", "area", "max"]
# FROZEN (dev findings): per_ex scope lost 4-8 points for every model family -> global only; 'max' untested -> not searched
# (still accepted by Preprocessor). Default for ALL models = dev-best preprocessing.
PREPROC_SPACE: dict[str, list[Any]] = dict(norm=["none", "l2", "area"], norm_scope=["global"], log1p=[False, True])
PREPROC_DEFAULT: dict[str, Any] = dict(norm="area", norm_scope="global", log1p=True)
_EPS = 1e-8
_MAX_STAT_ROWS = 200_000


class Preprocessor:
    def __init__(self, norm: str = "none", norm_scope: str = "global", log1p: bool = False,
                 standardize: bool = True, ex_nm: np.ndarray | None = None):
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        if norm_scope not in ("global", "per_ex"):
            raise ValueError(f"norm_scope must be global|per_ex, got {norm_scope!r}")
        self.norm, self.norm_scope, self.log1p, self.standardize = norm, norm_scope, bool(log1p), standardize
        self.ex_nm = None if ex_nm is None else np.asarray(ex_nm, dtype=float)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.scale_: float = 1.0
        self._blocks: list[np.ndarray] = self._make_blocks()

    def _make_blocks(self) -> list[np.ndarray]:
        if self.norm_scope == "per_ex" and self.ex_nm is not None:
            return [np.where(self.ex_nm == e)[0] for e in np.unique(self.ex_nm)]
        return []

    # --- internals (operate on 2-D [m, F] float32) ---
    def _normalise(self, A: np.ndarray) -> np.ndarray:
        if self.norm == "none":
            return A
        def den(B):
            if self.norm == "l2":
                return np.sqrt((B * B).sum(1, keepdims=True))
            if self.norm == "area":
                return np.abs(B).sum(1, keepdims=True)
            return np.abs(B).max(1, keepdims=True)
        if not self._blocks:
            return A / np.maximum(den(A), _EPS)
        out = np.empty_like(A)
        for idx in self._blocks:
            B = A[:, idx]
            out[:, idx] = B / np.maximum(den(B), _EPS)
        return out

    def _base(self, A: np.ndarray) -> np.ndarray:
        A = self._normalise(A)
        if self.log1p:
            A = np.log1p(np.maximum(A, 0.0) / self.scale_)
        return A.astype(np.float32, copy=False)

    @staticmethod
    def _rows(X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, X.shape[-1])

    # --- public ---
    def fit(self, X: np.ndarray) -> "Preprocessor":
        A = self._rows(np.asarray(X, dtype=np.float32))
        A = A[np.any(A != 0, axis=1)]
        if len(A) > _MAX_STAT_ROWS:
            A = A[np.random.default_rng(0).choice(len(A), _MAX_STAT_ROWS, replace=False)]
        if A.shape[1] and self.ex_nm is not None and len(self.ex_nm) != A.shape[1]:
            raise ValueError("ex_nm length does not match feature dimension")
        self.scale_ = 1.0
        if self.log1p:
            N = self._normalise(A)
            pos = N[N > 0]
            self.scale_ = float(np.median(pos)) if pos.size else 1.0
            self.scale_ = max(self.scale_, _EPS)
        B = self._base(A)
        if self.standardize and len(B):
            self.mean_ = B.mean(0).astype(np.float32)
            self.std_ = B.std(0).astype(np.float32)
            self.std_ = np.where(self.std_ < _EPS, 1.0, self.std_).astype(np.float32)
        else:
            F = A.shape[1]
            self.mean_ = np.zeros(F, np.float32)
            self.std_ = np.ones(F, np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Preprocessor not fitted")
        X = np.asarray(X, dtype=np.float32)
        A = self._rows(X)
        valid = np.any(A != 0, axis=1)
        out = np.zeros(A.shape, np.float32)
        if valid.any():
            B = self._base(A[valid])
            out[valid] = (B - self.mean_) / self.std_ if self.standardize else B
        return out.reshape(X.shape)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def make_preprocessor(cfg: dict[str, Any], meta: dict[str, Any]) -> Preprocessor:
    return Preprocessor(norm=cfg.get("norm", "none"), norm_scope=cfg.get("norm_scope", "global"),
                        log1p=cfg.get("log1p", False), standardize=cfg.get("standardize", True),
                        ex_nm=meta.get("ex_nm"))
