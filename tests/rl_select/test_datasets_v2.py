"""v2 rerun configuration: dataset list, results root, lichens2_sub3 cache, recorded-wavelength labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rl_select import paths
from rl_select.data import DATASETS, KNOWN_DATASETS, all_tasks, load_dataset, n_outer_folds, specimen_folds

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "rl_select"))

needs_cache = pytest.mark.skipif(not (paths.cache_dir() / "lichens2_sub3.npz").exists(), reason="no recwl cache")


def test_default_dataset_list(monkeypatch):
    monkeypatch.delenv("RL_SELECT_DATASETS", raising=False)
    assert paths.datasets() == ("lichens1", "lichens2_sub3", "collagen_acetic", "collagen_pepsin")
    assert "lichens2" in KNOWN_DATASETS and "lichens2" not in paths.DEFAULT_DATASETS


def test_dataset_list_override(monkeypatch):
    monkeypatch.setenv("RL_SELECT_DATASETS", "lichens1,lichens2,collagen_pepsin")
    assert paths.datasets() == ("lichens1", "lichens2", "collagen_pepsin")
    monkeypatch.setenv("RL_SELECT_DATASETS", "lichens1,nonsense")
    with pytest.raises(ValueError):
        paths.datasets()


def test_cross_pools_follow_the_list():
    p = paths.cross_pools(paths.DEFAULT_DATASETS)
    assert p["C2L"] == (("collagen_acetic", "collagen_pepsin"), ("lichens1", "lichens2_sub3"))
    assert p["L2C"] == (("lichens1", "lichens2_sub3"), ("collagen_acetic", "collagen_pepsin"))
    assert paths.cross_of(paths.DEFAULT_DATASETS)["lichens2_sub3"] == "C2L"
    assert paths.cross_pools(("lichens1", "lichens2")) == {}      # no second family -> no cross-family pools


def test_results_root_and_paper_out(monkeypatch):
    monkeypatch.delenv("RL_SELECT_RESULTS", raising=False)
    monkeypatch.delenv("RL_SELECT_PAPER_OUT", raising=False)
    assert paths.results_root() == paths.LEGACY_ROOT
    assert paths.paper_out() == paths.ROOT / "publications" / "rl_band_selection"
    monkeypatch.setenv("RL_SELECT_RESULTS", "results/rl_select_v2")
    assert paths.results_root() == paths.ROOT / "results" / "rl_select_v2"
    assert paths.paper_out() == paths.ROOT / "results" / "rl_select_v2" / "paper"   # never the publication folder


def test_seeds_and_workers(monkeypatch):
    monkeypatch.delenv("RL_SELECT_SEEDS", raising=False)
    assert paths.seeds() == (0, 1, 2)
    monkeypatch.setenv("RL_SELECT_SEEDS", "0,3")
    assert paths.seeds() == (0, 3)
    monkeypatch.setenv("RL_SELECT_WORKERS", "5")
    assert paths.workers() == 5


def test_s2_order_trains_every_variant_with_every_seed():
    import run_rl as R
    order = R.s2_order((0, 1, 2))
    got = {(src, v, sd) for _, src, v, sd in order}
    assert len(got) == len(order)
    want = {(s, v, sd) for s in R.POOLS for v in R.VARIANTS for sd in (0, 1, 2)}
    assert got == want
    assert set(R.POOLS) == {"C2L", "L2C"} | {f"lodo-{d}" for d in R.DATASETS}


@needs_cache
def test_lichens2_sub3_cache():
    d = load_dataset("lichens2_sub3")
    assert d["X"].shape == (13530, 210)
    assert d["n_classes"] == 3
    assert np.bincount(d["specimen_classes"]).tolist() == [5, 5, 5]
    assert np.bincount(d["y"]).tolist() == [5385, 3608, 4537]
    assert n_outer_folds("lichens2_sub3") == 5
    for rep in (0, 1):
        folds = specimen_folds("lichens2_sub3", rep)
        assert len(folds) == 5 and all(len(f) == 3 for f in folds)
        assert sorted(s for f in folds for s in f) == list(range(15))
    tasks = all_tasks("lichens2_sub3", 0)
    assert len(tasks) == 5 and all(t.n_classes == 3 for t in tasks)


# first retained emission band per excitation under the RECORDED axis (after Rayleigh filtering); the legacy labels were
# lower by RECORDED_WL_OFFSET (lichens1 385: 430, 400: 440, 415: 455, 430: 470; collagen 400: 430/440; lichens2 415/430: 455/470)
@needs_cache
@pytest.mark.parametrize("key,ex,first_em", [
    ("lichens1", 385.0, 450.0), ("lichens1", 400.0, 480.0), ("lichens1", 415.0, 490.0), ("lichens1", 430.0, 510.0),
    ("collagen_pepsin", 400.0, 460.0), ("collagen_acetic", 400.0, 470.0),
    ("lichens2_sub3", 415.0, 440.0), ("lichens2_sub3", 430.0, 440.0), ("lichens2", 430.0, 440.0)])
def test_recorded_wavelength_labels(key, ex, first_em):
    d = load_dataset(key)
    assert json.loads(str(d["meta"]))["info"]["recorded_wavelengths"] is True
    assert d["em_nm"][d["ex_nm"] == ex].min() == first_em


@needs_cache
@pytest.mark.skipif(not (paths.LEGACY_ROOT / "cache" / "lichens1.npz").exists(), reason="no legacy cache")
@pytest.mark.parametrize("key,legacy,wl_key", [("lichens1", "lichens1", "lichens1"), ("collagen_acetic", "collagen_acetic", "collagen_acetic"),
                                             ("collagen_pepsin", "collagen_pepsin", "collagen_pepsin"), ("lichens2", "lichens2", "lichens2"),
                                             ("lichens2_sub3", "lichens2", "lichens2")])
def test_labels_equal_legacy_plus_recorded_offset(key, legacy, wl_key):
    from me_attention.data import RECORDED_WL_OFFSET
    old = np.load(paths.LEGACY_ROOT / "cache" / f"{legacy}.npz", allow_pickle=True)
    d = load_dataset(key)
    assert np.array_equal(old["ex_nm"], d["ex_nm"])
    off = np.array([RECORDED_WL_OFFSET[wl_key].get(float(e), 0.0) for e in old["ex_nm"]])
    assert np.array_equal(old["em_nm"] + off, d["em_nm"])
    assert (off != 0).any()


@needs_cache
def test_caches_have_legacy_keys():
    need = {"X", "ex_nm", "em_nm", "y", "groups", "coords", "specimen_classes", "excitations", "emissions",
            "grid_index", "specimen_map", "meta"}
    for k in DATASETS:
        assert need <= set(load_dataset(k))


@needs_cache
@pytest.mark.skipif(not (paths.LEGACY_ROOT / "cache" / "lichens1.npz").exists(), reason="no legacy cache")
@pytest.mark.parametrize("key", ["lichens1", "collagen_acetic", "collagen_pepsin", "lichens2", "lichens2_sub3"])
def test_X_identical_to_legacy_cache(key):
    import build_caches
    msg = build_caches.verify(key, paths.cache_dir(), paths.LEGACY_ROOT / "cache")   # raises on mismatch
    assert "X" in msg
