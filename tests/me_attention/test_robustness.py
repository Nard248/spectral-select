"""Fast CPU unit tests for me_attention.robustness (toy data only, no GPU, no real dataset)."""
import numpy as np
import pytest

from me_attention import harness as H
from me_attention import robustness as R
from me_attention.preprocess import Preprocessor
from me_attention.protocol import outer_folds
from .conftest import make_me


# ============================================================================================= perturb_seed
def test_perturb_seed_deterministic_and_model_free():
    a = R.perturb_seed("lichens2_sub3", 0, 1, "T1", 2)
    b = R.perturb_seed("lichens2_sub3", 0, 1, "T1", 2)
    c = R.perturb_seed("lichens2_sub3", 0, 1, "T1", 3)
    assert isinstance(a, int) and a == b and a != c
    with pytest.raises(TypeError):
        R.perturb_seed("ds", 0, 0, "T1", 1, model="lr")  # must not accept a model argument


# ============================================================================================= T1
def test_t1_subsets_exhaustive_vs_sampled():
    exhaustive = R.t1_subsets(n_ex=6, m=1, seed=123)
    assert len(exhaustive) == 6 and exhaustive == sorted(exhaustive)
    assert set(exhaustive) == {(i,) for i in range(6)}

    sampled = R.t1_subsets(n_ex=8, m=3, seed=123)
    assert len(sampled) == 10 and len(set(sampled)) == 10
    for sub in sampled:
        assert len(sub) == 3 and len(set(sub)) == 3 and all(0 <= i < 8 for i in sub)
    # same seed -> same draw
    assert R.t1_subsets(8, 3, seed=123) == sampled
    assert R.t1_subsets(8, 3, seed=7) != sampled


def test_apply_t1_level0_is_clean_and_writes_impute():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 10, size=(5, 4)).astype(np.float32)
    blocks = [np.array([0, 1]), np.array([2, 3])]
    impute = np.array([100.0, 200.0, 300.0, 400.0], np.float32)
    clean = R.apply_t1(X, blocks, (), impute)
    assert np.array_equal(clean, X)
    dropped = R.apply_t1(X, blocks, (1,), impute)
    assert np.array_equal(dropped[:, :2], X[:, :2])
    assert np.all(dropped[:, 2] == 300.0) and np.all(dropped[:, 3] == 400.0)


def test_t1_impute_maps_to_zero_after_preprocessing():
    # median of positive normalised values == 1.0 for every feature (norm="none" => normalised == raw)
    X_train = np.array([[0.3, 0.3], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    pre = Preprocessor(norm="none", log1p=True, standardize=True).fit(X_train)
    assert pre.scale_ == 1.0  # premise of Deviation 2 (PHASEB_IMPL_SPEC.md section 2)
    impute = R.impute_values(X_train)
    row = pre.transform(impute[None, :])
    assert np.allclose(row, 0.0, atol=1e-5)


# ============================================================================================= T2
def test_apply_t2_sigma0_is_clean():
    X = np.full((6, 4), 5.0, np.float32)
    blocks = [np.array([0, 1]), np.array([2, 3])]
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = R.apply_t2(X, blocks, groups, sigma=0.0, seed=1)
    assert np.array_equal(out, X)


def test_apply_t2_constant_within_specimen_excitation_block():
    X = np.full((6, 4), 5.0, np.float32)
    blocks = [np.array([0, 1]), np.array([2, 3])]
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = R.apply_t2(X, blocks, groups, sigma=0.4, seed=99)
    ratio = out / X
    ratios_by_block = []
    for s in (0, 1):
        rows = groups == s
        for idx in blocks:
            block_ratio = ratio[np.ix_(rows, idx)]
            assert np.allclose(block_ratio, block_ratio[0, 0])  # constant within (specimen, excitation)
            ratios_by_block.append(float(block_ratio[0, 0]))
    assert len(set(np.round(ratios_by_block, 9))) > 1  # differs between blocks


def test_model_independence_t1_t2_t3():
    """apply_t1 / apply_t2 / apply_t3 never see a model name; the same (dataset, repeat, fold, family, level)
    must produce byte-identical output no matter which model is nominally being evaluated."""
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 10, size=(6, 4)).astype(np.float32)
    blocks = [np.array([0, 1]), np.array([2, 3])]
    groups = np.array([0, 0, 0, 1, 1, 1])
    std_train = X.std(0)
    impute = R.impute_values(X)

    seed_a = R.perturb_seed("ds", 0, 0, "T1", 1)
    seed_b = R.perturb_seed("ds", 0, 0, "T1", 1)  # "model lr" and "model rbfsvm" would compute the exact same call
    assert seed_a == seed_b
    assert np.array_equal(R.apply_t1(X, blocks, (0,), impute), R.apply_t1(X, blocks, (0,), impute))

    seed_t2 = R.perturb_seed("ds", 0, 0, "T2", 0.25)
    assert np.array_equal(R.apply_t2(X, blocks, groups, 0.25, seed_t2), R.apply_t2(X, blocks, groups, 0.25, seed_t2))

    seed_t3 = R.perturb_seed("ds", 0, 0, "T3", 0.5)
    assert np.array_equal(R.apply_t3(X, std_train, 0.5, seed_t3), R.apply_t3(X, std_train, 0.5, seed_t3))


# ============================================================================================= T3
def test_apply_t3_s0_is_clean_and_clips_at_zero():
    X = np.array([[1.0, 2.0, 0.0, 5.0]], np.float32)
    std_train = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
    assert np.array_equal(R.apply_t3(X, std_train, 0.0, seed=1), X)
    out = R.apply_t3(X, std_train * 50, 2.0, seed=1)
    assert np.all(out >= 0.0)


# ============================================================================================= L
@pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4])
def test_flip_labels(p):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=200)
    seed = R.perturb_seed("ds", 0, 0, "L", p)
    out = R.flip_labels(y, n_classes=3, p=p, seed=seed)
    changed = out != y
    assert changed.sum() == round(p * len(y))
    assert np.all(out[changed] != y[changed])
    assert set(np.unique(out)) <= {0, 1, 2}
    assert np.array_equal(R.flip_labels(y, 3, p, seed), out)  # deterministic


# ============================================================================================= C
def test_subsample_pixels_respects_per_specimen_counts():
    idx = np.arange(30)
    groups = np.repeat([0, 1, 2], 10)
    out = R.subsample_pixels(idx, groups, n=4, seed=5)
    counts = {s: int((groups[out] == s).sum()) for s in (0, 1, 2)}
    assert counts == {0: 4, 1: 4, 2: 4}
    assert len(out) == len(set(out.tolist()))
    big = R.subsample_pixels(idx, groups, n=1000, seed=5)
    assert np.array_equal(np.sort(big), idx)  # min(n, count_s) == count_s when n is large


def test_subsample_specimens_respects_avail_and_keeps_every_class():
    idx = np.arange(60)
    groups = np.repeat(np.arange(6), 10)  # 6 specimens, 10 px each
    specimen_classes = np.array([0, 0, 1, 1, 2, 2])
    out = R.subsample_specimens(idx, groups, specimen_classes, s=1, seed=3)
    kept_specimens = np.unique(groups[out])
    assert len(kept_specimens) == 3  # one per class, no class dropped
    assert set(specimen_classes[kept_specimens]) == {0, 1, 2}
    for sp in kept_specimens:  # every pixel of a kept specimen is kept
        assert (groups[out] == sp).sum() == 10

    out2 = R.subsample_specimens(idx, groups, specimen_classes, s=5, seed=3)
    assert set(np.unique(groups[out2])) == set(range(6))  # min(5, 2) == 2 -> keep both per class


def test_levels_for_cb_skipping_avail_2_2_2():
    data = make_me([2, 2, 2])  # 2 specimens/class, matches collagen_pepsin's avail
    train_idx = np.arange(len(data.y))
    levels = R.levels_for("C", data, train_idx)
    a_levels = sorted(v for sub, v in levels if sub == "a")
    b_levels = sorted(v for sub, v in levels if sub == "b")
    assert a_levels == [25, 100, 400]
    assert b_levels == [1]  # s=2 removes nothing from any class -> skipped (Deviation 3)


def test_levels_for_t1_excludes_m_ge_n_ex():
    data = make_me([2, 2, 2])  # 2 excitations in the toy fixture
    assert R.levels_for("T1", data, np.arange(len(data.y))) == [1]


def test_levels_for_other_families_match_prereg():
    data = make_me([2, 2, 2])
    tr = np.arange(len(data.y))
    assert R.levels_for("T2", data, tr) == [0.1, 0.25, 0.5]
    assert R.levels_for("T3", data, tr) == [0.1, 0.25, 0.5, 1.0]
    assert R.levels_for("L", data, tr) == [0.1, 0.2, 0.3, 0.4]


# ============================================================================================= clip_config
def test_clip_config_knn_clips_k_to_n_train():
    cfg, notes = R.clip_config({"k": 5, "weights": "uniform", "max_train": 20000}, "knn", n_train=3)
    assert cfg["k"] == 3 and len(notes) >= 1
    assert cfg["max_train"] == 3
    cfg2, notes2 = R.clip_config({"k": 2}, "knn", n_train=10)
    assert cfg2["k"] == 2 and notes2 == []  # nothing to clip


def test_clip_config_batch_size_for_neural_model():
    cfg, notes = R.clip_config({"batch_size": 256}, "mlp", n_train=4)
    assert cfg["batch_size"] == 8  # max(8, n_train)
    assert any("batch_size" in n for n in notes)


# ============================================================================================= run_unit (toy end-to-end)
@pytest.fixture
def toy_unit_fixture(monkeypatch, tmp_path):
    data = make_me([3, 3, 3], px=10)
    data.name = "synth_robust"
    monkeypatch.setattr(R, "load_dataset", lambda key: data)
    R._DATA_CACHE.clear()
    tr, te = outer_folds(data, 0)[0]
    clean_path = H.result_path(tmp_path, "cleanfake", "synth_robust", "lr")
    seed = H.job_seed("synth_robust", "lr", 0, 0)
    H.append_record(clean_path, dict(dataset="synth_robust", model="lr", repeat=0, fold=0, seed=seed,
                                     n_train=int(len(tr)), n_test=int(len(te)), best_config={"C": 1.0}))
    yield data, tr, te, tmp_path
    R._DATA_CACHE.clear()


def test_run_unit_emits_exactly_the_expected_keys(toy_unit_fixture):
    data, tr, te, tmp_path = toy_unit_fixture
    job = dict(dataset="synth_robust", model="lr", repeat=0, fold=0, device="cpu", threads=1,
              families=list(R.ALL_FAMILIES), clean_root=tmp_path, clean_sweep="cleanfake")
    recs = R.run_unit(job)
    assert all("error" not in r for r in recs), [r.get("error") for r in recs if "error" in r]
    got_keys = {R._record_key(r) for r in recs}
    expected_keys = R._expected_keys(data, tr, list(R.ALL_FAMILIES))
    assert got_keys == expected_keys

    clean = [r for r in recs if r["family"] == "clean"]
    assert len(clean) == 1 and clean[0]["level"] == 0 and clean[0]["n_subsets"] is None
    for r in recs:
        assert set(r["test"]) >= {"balanced_acc", "macro_f1", "pixel_acc", "specimen_avg_acc",
                                  "specimen_vote_acc", "per_class_recall", "confusion", "n", "n_specimens", "labels"}
        assert 0.0 <= r["test"]["balanced_acc"] <= 1.0
        assert r["n_train"] == len(tr) and r["n_test"] == len(te)
    t1 = [r for r in recs if r["family"] == "T1"]
    assert len(t1) == 1 and t1[0]["level"] == 1  # toy data has only 2 excitations -> only m=1 valid
    assert t1[0]["n_subsets"] == 2 and len(t1[0]["subset_baccs"]) == 2
    assert len([r for r in recs if r["family"] == "T2"]) == 3
    assert len([r for r in recs if r["family"] == "T3"]) == 4
    assert len([r for r in recs if r["family"] == "L"]) == 4
    c_recs = [r for r in recs if r["family"] == "C"]
    assert {r["subset"] for r in c_recs} <= {"a", "b"}
    assert all(r["subset"] is None for r in recs if r["family"] != "C")


def test_run_unit_missing_clean_record_becomes_error(monkeypatch, tmp_path):
    data = make_me([3, 3, 3], px=10)
    monkeypatch.setattr(R, "load_dataset", lambda key: data)
    R._DATA_CACHE.clear()
    job = dict(dataset="synth_robust", model="lr", repeat=0, fold=0, device="cpu", threads=1,
              families=["clean"], clean_root=tmp_path, clean_sweep="nosuchsweep")
    recs = R.run_unit(job)
    assert len(recs) == 1 and "error" in recs[0]
    R._DATA_CACHE.clear()


# ============================================================================================= build_jobs
def test_build_jobs_resumability(monkeypatch, tmp_path):
    data = make_me([3, 3, 3], px=10)
    monkeypatch.setattr(R, "load_dataset", lambda key: data)
    R._DATA_CACHE.clear()
    jobs, skipped = R.build_jobs(tmp_path, ["synth_robust"], ["lr"], {"synth_robust": 1},
                                 families=["clean", "T2"], only_repeat=0, only_fold=0)
    assert len(jobs) == 1 and skipped == 0

    tr, _ = outer_folds(data, 0)[0]
    path = R.result_path(tmp_path, "synth_robust", "lr")
    H.append_record(path, dict(dataset="synth_robust", model="lr", repeat=0, fold=0, family="clean", level=0,
                               subset=None, test={}))
    for sigma in R.levels_for("T2", data, tr):
        H.append_record(path, dict(dataset="synth_robust", model="lr", repeat=0, fold=0, family="T2", level=sigma,
                                   subset=None, test={}))
    jobs2, skipped2 = R.build_jobs(tmp_path, ["synth_robust"], ["lr"], {"synth_robust": 1},
                                   families=["clean", "T2"], only_repeat=0, only_fold=0)
    assert len(jobs2) == 0 and skipped2 == 1
    R._DATA_CACHE.clear()
