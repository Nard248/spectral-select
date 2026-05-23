"""Sparse-LASSO — supervised band selection via L1-penalized regression.

For multi-class data, we use L1-penalized logistic regression
(`LogisticRegression(penalty='l1', solver='saga')`). The importance of
each band is the maximum absolute coefficient across classes:

    importance[b] = max_c |W[c, b]|

Top-K bands by importance are selected. The strength of L1 (`C` parameter)
controls sparsity; we sweep a small grid and pick the value that yields
at least K non-zero coefficients.

This is a supervised baseline. For Drop Data (unlabeled) we fall back to
selecting the K bands with the highest absolute correlation to the first
principal component of the data — but the proper application is on
Lichens and Pepsin, where labels exist.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def select_sparse_lasso(features: np.ndarray, K: int, *,
                        labels: np.ndarray | None = None,
                        seed: int = 0,
                        C_grid: tuple = (0.001, 0.01, 0.1, 1.0, 10.0),
                        max_iter: int = 500,
                        **_):
    """L1-penalized multinomial logistic regression band selection.

    If `labels` is None, returns top-K by variance as a safe fallback
    (sparse-LASSO is supervised — this case shouldn't be evaluated on
    unlabeled datasets).
    """
    if labels is None:
        return np.asarray(np.argsort(-features.var(axis=0))[:K], dtype=int)

    # Standardize features per band
    X = StandardScaler().fit_transform(features)
    y = np.asarray(labels)

    # Sweep C-grid from sparsest to densest; stop when at least K bands
    # have non-zero importance
    chosen_importance = None
    chosen_C = None
    for C in C_grid:
        try:
            clf = LogisticRegression(
                penalty="l1", solver="saga", multi_class="auto",
                C=C, max_iter=max_iter, random_state=seed, n_jobs=1,
                tol=1e-3,
            )
            clf.fit(X, y)
            coef = clf.coef_  # (n_classes, n_bands) or (1, n_bands) for binary
            importance = np.max(np.abs(coef), axis=0)
            n_nonzero = int((importance > 1e-10).sum())
            chosen_importance = importance
            chosen_C = C
            if n_nonzero >= K:
                break
        except Exception:
            continue

    if chosen_importance is None:
        # Fallback if all fits failed
        return np.asarray(np.argsort(-features.var(axis=0))[:K], dtype=int)

    ranking = np.argsort(-chosen_importance)
    return np.asarray(ranking[:K], dtype=int)
