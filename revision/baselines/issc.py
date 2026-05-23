"""ISSC — Improved Sparse Subspace Clustering for band selection.

Treats each spectral band as a data point in pixel-space, learns sparse
self-expression coefficients between bands, and clusters them. Selects
one representative band per cluster.

Reference family:
    Elhamifar & Vidal 2013 "Sparse Subspace Clustering: Algorithm, Theory,
    and Applications" (TPAMI); ISSC variant for HSI band selection in e.g.
    Wang et al. "Optimal Clustering Framework for HSI Band Selection" (2018).

This is a simplified, dependency-light implementation:
    1. For each band b, regress its column onto the other bands' columns
       with L1 penalty (lasso). Gives a sparse coefficient vector c_b.
    2. Affinity W[a, b] = |c_b[a]| + |c_a[b]| (symmetric).
    3. Spectral clustering on W with k=K clusters.
    4. Within each cluster, pick the band with highest variance as its
       representative.

Heavy-lifting (the lasso per band) is the slow part; we keep n_alphas low.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Lasso


def _self_expression(features: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Solve for each band b: c_b = argmin ||F[:, b] - F[:, ~b] c||^2 + alpha*|c|_1.

    Returns C: (B, B) with C[a, b] = coefficient of band a in expressing b
    (C[b, b] = 0 by construction).
    """
    N, B = features.shape
    C = np.zeros((B, B), dtype=np.float64)
    # Standardize columns (band-wise) so the lasso treats all bands on the
    # same scale. Center pixels too.
    Fc = features - features.mean(axis=0, keepdims=True)
    col_norms = np.linalg.norm(Fc, axis=0) + 1e-12
    Fn = Fc / col_norms

    for b in range(B):
        mask = np.ones(B, dtype=bool)
        mask[b] = False
        # Use a quick lasso; small max_iter to keep this tractable
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=400,
                      selection="random", random_state=0, tol=1e-3)
        try:
            lasso.fit(Fn[:, mask], Fn[:, b])
            coef = lasso.coef_
        except Exception:
            coef = np.zeros(B - 1)
        # Insert zero at b's own position
        full = np.zeros(B)
        full[mask] = coef
        C[:, b] = full
    return C


def select_issc(features: np.ndarray, K: int, *, alpha: float = 0.01,
                seed: int = 0, **_):
    """ISSC band selection.

    For large N (pixels), subsample to keep the lasso tractable.
    """
    N, B = features.shape
    # Subsample pixels for speed (lasso is the bottleneck)
    rng = np.random.default_rng(seed)
    n_sub = min(N, 5000)
    if N > n_sub:
        idx = rng.choice(N, size=n_sub, replace=False)
        F = features[idx]
    else:
        F = features

    C = _self_expression(F, alpha=alpha)
    W = np.abs(C) + np.abs(C.T)
    np.fill_diagonal(W, 0.0)
    # Spectral clustering on band affinity
    sc = SpectralClustering(
        n_clusters=min(K, B),
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
        n_init=10,
    )
    labels = sc.fit_predict(W)
    # Choose representative band per cluster: highest variance
    band_var = features.var(axis=0)
    chosen = []
    for c in range(int(labels.max()) + 1):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        best = members[np.argmax(band_var[members])]
        chosen.append(int(best))
    if len(chosen) < K:
        # Pad with next-best variance bands not in chosen
        remain = sorted(set(range(B)) - set(chosen),
                        key=lambda b: -band_var[b])
        chosen.extend(remain[:K - len(chosen)])
    return np.asarray(chosen[:K], dtype=int)
