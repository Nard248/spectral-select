"""Core tests for the rl_select study: folds, objectives, residual stats, baselines."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

from rl_select import baselines as B
from rl_select.data import DATASETS, all_tasks, inner_folds, load_dataset, make_task, specimen_folds
from rl_select.evaluate import evaluate, set_metrics, stability
from rl_select.objective import JRecon, JReconPC, JSup, correlation_matrix, make_objective, residual_stats


@pytest.fixture(scope="module")
def task():
    return make_task("collagen_pepsin", 0, 0)


@pytest.fixture(scope="module")
def jsup(task):
    return JSup(task)


@pytest.mark.parametrize("key", DATASETS)
def test_fold_invariants(key):
    d = load_dataset(key)
    folds = specimen_folds(key, 0)
    flat = [s for f in folds for s in f]
    assert sorted(flat) == list(range(len(d["specimen_classes"])))          # partition
    for f in folds:
        assert set(d["specimen_classes"][f]) == set(range(d["n_classes"]))   # every class in test
    t = make_task(key, 0, 0)
    assert set(np.unique(t.gtr)).isdisjoint(set(np.unique(t.gte)))
    assert set(np.unique(t.yte)) == set(range(t.n_classes))
    # preprocessing is fit on training pixels only
    assert abs(float(t.Xtr.mean())) < 1e-3
    folds_in = inner_folds(t)
    ysub = t.ytr[t.sub_idx]
    for tr, va in folds_in:
        assert len(np.intersect1d(tr, va)) == 0
        assert set(ysub[tr]) == set(range(t.n_classes)) == set(ysub[va])     # class-covering


def test_lichens2_pseudo_specimens():
    for f in range(2):
        t = make_task("lichens2", 0, f)
        n_cls4 = len(np.unique(t.gtr[t.ytr == 4]))
        assert n_cls4 == 1                                                   # single training specimen
        for tr, va in inner_folds(t):
            ysub = t.ytr[t.sub_idx]
            assert 4 in ysub[tr] and 4 in ysub[va]


def test_all_tasks_count():
    assert len(all_tasks("collagen_acetic", 0)) == 3


def _ref_lda_ba(task, S):
    """Plain-numpy reference LDA (shrinkage 0.1, equal priors) over the inner folds."""
    X = task.Xtr[task.sub_idx].astype(np.float64)[:, S]
    y = task.ytr[task.sub_idx]
    C = task.n_classes
    bas = []
    for tr, va in inner_folds(task):
        Xt, yt = X[tr], y[tr]
        mus = np.stack([Xt[yt == c].mean(0) for c in range(C)])
        Sw = sum(((Xt[yt == c] - mus[c]).T @ (Xt[yt == c] - mus[c])) for c in range(C)) / (len(Xt) - C)
        Sw = 0.9 * Sw + 0.1 * np.trace(Sw) / len(S) * np.eye(len(S))
        W = np.linalg.inv(Sw)
        sc = X[va] @ W @ mus.T - 0.5 * np.einsum("cs,st,ct->c", mus, W, mus)
        pred = sc.argmax(1)
        bas.append(np.mean([(pred[y[va] == c] == c).mean() for c in range(C)]))
    return float(np.mean(bas))


def test_jsup_matches_reference_and_counts(task, jsup):
    jsup.reset_counter(); jsup.clear_cache()
    for S in ([3, 40, 90], [1, 2, 3, 4, 5, 100, 150, 20]):
        assert abs(jsup(S) - _ref_lda_ba(task, S)) < 1e-6
    assert jsup.n_evals == 2
    jsup([90, 40, 3])                       # permutation hits the cache
    assert jsup.n_evals == 2
    assert jsup([]) == pytest.approx(1 / task.n_classes)
    with pytest.raises(ValueError):
        jsup([1, 1, 2])
    with pytest.raises(ValueError):
        jsup([10_000])


def test_jsup_penalties(task):
    base = JSup(task)
    pen = JSup(task, lam_corr=0.5, mu_hw=0.01)
    S = [3, 40, 90]
    corr = np.abs(correlation_matrix(task))[np.ix_(S, S)]
    rc = (corr.sum() - 3) / 6
    n_ex = len(np.unique(task.ex_nm[S]))
    assert pen(S) == pytest.approx(base(S) - 0.5 * rc - 0.01 * n_ex)


def test_jrecon_range_and_monotone(task):
    for J in (JRecon(task), JReconPC(task)):
        rng = np.random.default_rng(0)
        S = [int(i) for i in rng.choice(task.n_features, 6, replace=False)]
        prev = J(S[:1])
        for k in range(2, 7):
            cur = J(S[:k])
            assert 0 <= prev <= 1 and 0 <= cur <= 1 + 1e-9
            assert cur >= prev - 1e-9
            prev = cur
        assert J(list(range(task.n_features))) > 0.999 or isinstance(J, JReconPC)


def test_residual_stats_vs_bruteforce(task):
    for J in (JRecon(task), JReconPC(task)):
        S = [5, 60, 120, 33]
        st = J.stats(S)
        base = J(S)
        for c in range(0, task.n_features, 7):
            if c in S:
                assert st["gain_add"][c] == 0
                assert st["loss_remove"][c] == pytest.approx(base - J([s for s in S if s != c]), abs=1e-4)
            else:
                assert st["loss_remove"][c] == 0
                assert st["gain_add"][c] == pytest.approx(J(S + [c]) - base, abs=1e-4)
    Sig = correlation_matrix(task)
    rv = residual_stats(Sig, [5, 60])["resid_var"]
    A = Sig[np.ix_([5, 60], [5, 60])]
    assert rv[10] == pytest.approx(Sig[10, 10] - Sig[10, [5, 60]] @ np.linalg.solve(A, Sig[[5, 60], 10]), abs=1e-4)


def test_objective_picklable_and_factory(task):
    import pickle
    J = make_objective("jsup", task)
    J2 = pickle.loads(pickle.dumps(J))
    assert J([1, 2, 3]) == pytest.approx(J2([1, 2, 3]))
    with pytest.raises(ValueError):
        make_objective("nope", task)


@pytest.mark.parametrize("name", [k for k in B.BASELINES if k != "exact"])
@pytest.mark.parametrize("n", [3, 8])
def test_baselines_valid_and_budget(task, name, n):
    obj = JSup(task)
    r = B.BASELINES[name](task, n, obj=obj, budget=150, seed=1)
    assert len(r.selected) == n and len(set(r.selected)) == n
    assert all(0 <= i < task.n_features and task.valid[i] for i in r.selected)
    if name in B.WRAPPERS:
        assert r.n_evals <= 150
        assert r.inner_J is not None
        if r.trace:
            assert r.trace[-1][0] <= 150
            assert all(b[1] >= a[1] for a, b in zip(r.trace, r.trace[1:]))
    else:
        assert r.n_evals == 0


def test_wrapper_budget_independent_of_warm_cache(task):
    obj = JSup(task)
    r1 = B.random_search(task, 4, obj=obj, budget=60, seed=3)
    r2 = B.random_search(task, 4, obj=obj, budget=60, seed=3)   # objective cache is warm now
    assert r1.n_evals == r2.n_evals == 60 and r1.selected == r2.selected


def test_exact_n2_is_optimal(task):
    obj = JSup(task)
    r = B.exact(task, 2, obj, n_jobs=4)
    F = task.n_features
    assert r.n_evals == F * (F - 1) // 2
    # spot-check optimality against random pairs and the chosen pair's value
    assert r.inner_J == pytest.approx(JSup(task)(r.selected))
    rng = np.random.default_rng(0)
    ref = JSup(task)
    for _ in range(200):
        p = rng.choice(F, 2, replace=False)
        assert ref(p) <= r.inner_J + 1e-12


def test_evaluate_and_metrics(task):
    S = [3, 40, 90, 120]
    out = evaluate(task, S, classifiers=("knn", "lda"))
    for k in ("knn", "lda"):
        assert 0 <= out[k]["balanced_acc"] <= 1 and "specimen_vote_acc" in out[k]
    m = set_metrics(task, S)
    assert 0 <= m["jrecon"] <= 1 and m["n_distinct_excitations"] >= 1
    s = stability([[1, 2, 3], [1, 2, 3]], 20)
    assert s["jaccard"] == 1 and s["kuncheva"] == pytest.approx(1.0)
