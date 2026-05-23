"""Classical HSI band-selection baselines.

All take (features: (n_pix, n_bands), K: int, ...) -> (K,) band indices.

These are ports/refactors of the existing functions in
`experiments/drop_data_selection_sweep.py`, lifted to a unified signature
and with the K-truncation done here rather than at the caller.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
def select_variance(features: np.ndarray, K: int, *, seed: int = 0, **_):
    """Top-K bands by per-band variance across pixels."""
    ranking = np.argsort(-features.var(axis=0))
    return np.asarray(ranking[:K], dtype=int)


# ---------------------------------------------------------------------------
def select_pca_loading(features: np.ndarray, K: int, *,
                       n_components: int = 5, seed: int = 0, **_):
    """Top-K bands by absolute PCA loading magnitude.

    Score = sum_c (var_ratio_c * |loading_{c, b}|) over the top n_components.
    """
    n_bands = features.shape[1]
    n_components = min(n_components, n_bands - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(features)
    weight = (pca.explained_variance_ratio_[:, None] *
              np.abs(pca.components_)).sum(axis=0)
    ranking = np.argsort(-weight)
    return np.asarray(ranking[:K], dtype=int)


# ---------------------------------------------------------------------------
def select_sam_greedy(features: np.ndarray, K: int, *, seed: int = 0, **_):
    """SAM-style greedy selection: each new band maximizes minimum spectral
    angle (in absolute-cosine sense) to the already-chosen set."""
    rng = np.random.default_rng(seed)
    F = features
    norms = np.linalg.norm(F, axis=0) + 1e-12
    Fn = F / norms
    n_bands = F.shape[1]
    chosen = [int(np.argmax(F.var(axis=0)))]
    while len(chosen) < K:
        cos = Fn.T @ Fn[:, chosen]
        cos = np.clip(cos, -1.0, 1.0)
        angles = np.arccos(np.abs(cos))
        min_angle = angles.min(axis=1)
        min_angle[chosen] = -np.inf
        noise = rng.uniform(-1e-9, 1e-9, size=min_angle.shape)
        chosen.append(int(np.argmax(min_angle + noise)))
        if len(chosen) >= n_bands:
            break
    return np.asarray(chosen[:K], dtype=int)


# ---------------------------------------------------------------------------
def select_spa(features: np.ndarray, K: int, *, seed: int = 0, **_):
    """Successive Projections Algorithm (SPA).

    Repeatedly select the band most orthogonal to the span of already-chosen
    bands. Standard SPA.
    """
    F = features.astype(np.float64, copy=True)
    n_bands = F.shape[1]
    norms = np.linalg.norm(F, axis=0)
    chosen = [int(np.argmax(norms))]
    R = F.copy()
    while len(chosen) < K:
        v = R[:, chosen[-1]:chosen[-1] + 1]
        denom = float(v.T @ v) + 1e-12
        proj_coef = (v.T @ R) / denom
        R = R - v @ proj_coef
        norms = np.linalg.norm(R, axis=0)
        norms[chosen] = -np.inf
        chosen.append(int(np.argmax(norms)))
        if len(chosen) >= n_bands:
            break
    return np.asarray(chosen[:K], dtype=int)


# ---------------------------------------------------------------------------
def select_mcuve(features: np.ndarray, K: int, *,
                 n_iter: int = 50, subset_frac: float = 0.5,
                 seed: int = 0, **_):
    """Monte Carlo Uninformative Variable Elimination.

    Score = |mean_coef| / std_coef over bootstrap regressions of bands
    against the first principal component (an unsupervised target).
    """
    rng = np.random.default_rng(seed)
    N, B = features.shape
    target = PCA(n_components=1, random_state=seed).fit_transform(features)[:, 0]
    coefs = np.zeros((n_iter, B), dtype=np.float64)
    for i in range(n_iter):
        idx = rng.choice(N, size=max(int(subset_frac * N), 100), replace=False)
        X = features[idx]
        y = target[idx]
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        denom = (Xc * Xc).sum(axis=0) + 1e-12
        coefs[i] = (Xc * yc[:, None]).sum(axis=0) / denom
    mean = coefs.mean(axis=0)
    std = coefs.std(axis=0) + 1e-12
    stability = np.abs(mean) / std
    ranking = np.argsort(-stability)
    return np.asarray(ranking[:K], dtype=int)


# ---------------------------------------------------------------------------
def select_random(features: np.ndarray, K: int, *, seed: int = 0, **_):
    """Uniform random K-subset (no replacement)."""
    rng = np.random.default_rng(seed)
    n_bands = features.shape[1]
    chosen = rng.choice(n_bands, size=min(K, n_bands), replace=False)
    return np.asarray(chosen, dtype=int)
