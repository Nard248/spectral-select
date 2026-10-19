import numpy as np
import pytest

from rl_select.augment import MixObjective, build_pool
from rl_select.data import CACHE_DIR


@pytest.mark.skipif(not (CACHE_DIR / "collagen_acetic.npz").exists(), reason="no cache")
def test_pool_collagen():
    pool = build_pool(["collagen_acetic"], mix=True, n_dropout=2)
    # 3 folds x (1 original + 2 dropout + 3 class subsets)
    assert len(pool) == 3 * 6
    names = [t.name for t, _ in pool]
    assert any("#c01" in n for n in names) and any("#xd" in n for n in names)
    for t, o in pool:
        assert t.Xte.shape[0] == 0
        assert t.sub_idx.max() < len(t.Xtr)
        if "#c" in t.name:
            assert t.n_classes == 2 and set(np.unique(t.ytr)) == {0, 1}
            assert abs(t.Xtr.mean()) < 1e-3
        if "#xd" in t.name:
            assert (~t.valid).sum() > 0
        assert isinstance(o, MixObjective)
        S = list(np.flatnonzero(t.valid)[:4])
        v = o(S)
        assert np.isfinite(v) and v > 0.0
        assert o.empty_value == 1.0 / o.sup.C
