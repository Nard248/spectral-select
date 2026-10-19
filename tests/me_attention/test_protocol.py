import numpy as np
import pytest

from me_attention import protocol as P
from tests.me_attention.conftest import make_me


@pytest.mark.parametrize("counts", [[3, 3, 3], [4, 4, 4, 4], [4, 10, 7, 5, 2], [2, 5, 3], [6, 2]])
def test_outer_folds_cover_all_classes_and_disjoint(counts):
    d = make_me(counts)
    k = min(counts)
    for rep in range(5):
        folds = P.outer_folds(d, rep)
        assert len(folds) == k
        seen = []
        for tr, te in folds:
            assert set(np.unique(d.y[te])) == set(range(len(counts)))
            assert set(d.groups[tr]).isdisjoint(d.groups[te])
            assert len(tr) + len(te) == len(d.y)
            seen.append(set(d.groups[te]))
        assert set().union(*seen) == set(range(sum(counts)))      # every specimen tested exactly once
        assert sum(len(s) for s in seen) == sum(counts)


def test_outer_folds_seeded_and_vary():
    d = make_me([4, 4, 4, 4])
    a = [te.tolist() for _, te in P.outer_folds(d, 0)]
    assert a == [te.tolist() for _, te in P.outer_folds(d, 0)]
    assert any(a != [te.tolist() for _, te in P.outer_folds(d, r)] for r in range(1, 5))


def test_inner_split_whole_specimens():
    d = make_me([3, 3, 3])
    tr, _ = P.outer_folds(d, 0)[0]
    itr, iva = P.inner_split(d, tr, seed=1)
    assert set(itr).isdisjoint(iva) and set(itr) | set(iva) == set(tr)
    assert set(d.groups[itr]).isdisjoint(d.groups[iva])
    assert sorted(set(d.y[iva])) == [0, 1, 2]
    assert len(set(d.groups[iva])) == 3                       # one specimen per class
    assert sorted(set(d.y[itr])) == [0, 1, 2]


def test_inner_split_single_specimen_block():
    d = make_me([2, 2], px=20)       # k=2 => 1 train specimen per class
    tr, _ = P.outer_folds(d, 0)[0]
    itr, iva = P.inner_split(d, tr, seed=3)
    assert set(itr).isdisjoint(iva) and set(itr) | set(iva) == set(tr)
    for c in range(2):
        s = d.groups[tr][d.y[tr] == c][0]
        v = iva[d.groups[iva] == s]
        assert len(v) == 5                                     # 25% of 20
        cols = np.sort(d.coords[v, 1])
        assert (np.diff(cols) == 1).all()                      # contiguous along the longer (column) axis
        assert (d.groups[itr] == s).any()                      # specimen still partly in inner-train


def test_make_patches_zero_outside_specimen():
    d = make_me([2, 2], px=6)
    # put two specimens side by side: make specimen 1 adjacent to specimen 0 by rewriting coords
    d.coords[d.groups == 1] = d.coords[d.groups == 0] + np.array([1, 0])
    d.specimen_map[:] = -1
    d.specimen_map[d.coords[:, 0], d.coords[:, 1]] = d.groups
    P.clear_patch_cache()
    idx = np.flatnonzero(d.groups == 0)[:3]
    out = P.make_patches(d, idx, 3)
    assert out.shape == (3, 3, 3, d.n_features) and out.dtype == np.float32
    assert np.allclose(out[:, 1, 1], d.X[idx])                 # centre = pixel itself
    assert (out[:, 2, :] == 0).all()                           # row below belongs to specimen 1 -> zeroed
    assert (out[:, 0, :] == 0).all()                           # image/background above
    # a neighbour inside the same specimen is carried
    assert np.allclose(out[1, 1, 0], d.X[idx[0]]) and np.allclose(out[1, 1, 2], d.X[idx[2]])
    P.clear_patch_cache()


def test_metrics_missing_predicted_class():
    y = np.array([0, 0, 1, 1, 2, 2])
    g = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 0, 1, 1])                        # class 2 never predicted
    m = P.metrics(y, pred, g, 3)
    assert m["per_class_recall"] == [1.0, 0.5, 0.0]
    assert m["balanced_acc"] == pytest.approx(0.5)
    assert m["pixel_acc"] == pytest.approx(3 / 6)
    from sklearn.metrics import f1_score
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))
    assert m["specimen_vote_acc"] == pytest.approx(1 / 3)  # specimen 1 ties -> lowest class
    assert m["confusion"][2] == [0, 2, 0]
    assert P.metrics(y, np.eye(3)[pred], g, 3)["pixel_acc"] == m["pixel_acc"]
    assert P.metrics(y, pred, g, 3, whole_specimen=False)["specimen_vote_acc"] is None


def test_stats_and_holm():
    a = np.array([0.9, 0.8, 0.85, 0.95, 0.9, 0.88])
    b = a - np.array([0.05, 0.02, 0.04, 0.03, 0.06, 0.01])
    s = P.stats_paired(a, b, 1 / 3)
    assert s["mean_diff"] == pytest.approx(0.035) and s["wins"] == 6
    assert 0 < s["nb_p"] < 0.05 and s["ci95"][0] > 0
    assert s["wilcoxon_p"] == pytest.approx(2 / 64)
    z = P.stats_paired(a, a, 1 / 3)
    assert z["mean_diff"] == 0 and z["wilcoxon_p"] == 1.0
    assert P.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
