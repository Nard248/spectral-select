"""Outer-test evaluation of selected sets (docs/rl_select/SPEC.md section 5). Only module that touches Xte."""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .objective import JRecon, JReconPC, correlation_matrix
from .stats import metrics
from .types import Task

CAPS = {"knn": 1000, "lda": 1000, "rf": 1000, "svm": 300}


def _make_clf(name: str, seed: int):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", C=10, gamma="scale")
    raise ValueError(f"unknown classifier {name!r}")


def _cap_idx(task: Task, cap: int, seed: int) -> np.ndarray:
    """Seeded cap of ``cap`` pixels per training specimen."""
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(task.gtr):
        idx = np.flatnonzero(task.gtr == s)
        out.append(rng.choice(idx, size=cap, replace=False) if len(idx) > cap else idx)
    return np.sort(np.concatenate(out))


def _check_set(task: Task, S: Sequence[int]) -> np.ndarray:
    S = np.asarray(list(S), dtype=np.int64)
    if S.size == 0 or len(np.unique(S)) != S.size or S.min() < 0 or S.max() >= task.n_features:
        raise ValueError("S must be a non-empty set of valid unique indices")
    return S


def evaluate(task: Task, S: Sequence[int], classifiers: Sequence[str] = ("knn", "lda", "rf", "svm"),
             seed: int = 0) -> dict[str, dict]:
    """Train each classifier on Xtr[:, S] (capped per specimen) and score all of Xte[:, S]."""
    S = _check_set(task, S)
    Xte = task.Xte[:, S]
    out = {}
    for name in classifiers:
        idx = _cap_idx(task, CAPS[name], seed)
        clf = _make_clf(name, seed).fit(task.Xtr[idx][:, S], task.ytr[idx])
        out[name] = metrics(task.yte, clf.predict(Xte), task.gte, task.n_classes)
    return out


def all_pairs_reference(task: Task, classifiers: Sequence[str] = ("knn", "lda", "rf", "svm"),
                        seed: int = 0) -> dict[str, dict]:
    """Reference scores using all F candidate pairs."""
    return evaluate(task, np.flatnonzero(task.valid), classifiers, seed)


def set_metrics(task: Task, S: Sequence[int]) -> dict[str, float]:
    """Label-free set descriptors: #distinct excitations, mean |corr|, JRecon, JReconPC."""
    S = _check_set(task, S)
    corr = np.abs(correlation_matrix(task))[np.ix_(S, S)]
    k = len(S)
    return {
        "n_distinct_excitations": int(len(np.unique(task.ex_nm[S]))),
        "mean_abs_corr": float((corr.sum() - k) / (k * (k - 1))) if k > 1 else 0.0,
        "jrecon": JRecon(task)(S),
        "jreconpc": JReconPC(task)(S),
    }


def stability(sets: Sequence[Sequence[int]], F: int) -> dict[str, float]:
    """Mean pairwise Jaccard and Kuncheva consistency index over equal-size sets."""
    sets = [frozenset(int(i) for i in s) for s in sets]
    if len(sets) < 2:
        raise ValueError("need at least 2 sets")
    k = len(sets[0])
    if any(len(s) != k for s in sets) or not 0 < k < F:
        raise ValueError("sets must have equal size 0 < k < F")
    jac, kun = [], []
    for a, b in combinations(sets, 2):
        r = len(a & b)
        jac.append(r / len(a | b))
        kun.append((r * F - k * k) / (k * (F - k)))
    return {"jaccard": float(np.mean(jac)), "kuncheva": float(np.mean(kun))}
