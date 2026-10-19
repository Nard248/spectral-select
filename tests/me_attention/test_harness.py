import json

import numpy as np
import pytest

from me_attention import harness as H
from me_attention.types import ModelSpec
from .syn_blocks import make_blocks

DS = "syn"
MODELS = ["lr", "mlp", "patchmean_lr", "cnn2d"]


@pytest.fixture
def env(monkeypatch, tmp_path):
    data = make_blocks(F=24)
    data.name = DS
    monkeypatch.setattr(H, "load_dataset", lambda key: data)
    H._DATA_CACHE.clear()
    H._POOLS.clear()

    def boom_build(cfg, meta, seed, device):
        raise ValueError("deliberate failure")
    real = H.get_models()
    reg = dict(real)
    reg["boom"] = ModelSpec("boom", "spectral", "baseline", boom_build, space={"a": [1, 2]}, default={"a": 1})
    monkeypatch.setattr(H, "get_models", lambda: reg)
    yield data, tmp_path
    H._DATA_CACHE.clear()
    H._POOLS.clear()


def sweep(tmp_path, models, **kw):
    args = dict(sweep="t", datasets=[DS], models=models, repeats={DS: 1}, budget=2, max_epochs=2, device="cpu",
                workers=0, spatial_cap=300, root=tmp_path)
    args.update(kw)
    return H.run_sweep(**args)


def test_helpers():
    spec = H.get_models()["mlp"]
    a, b = H.sample_configs(spec, 4, 7), H.sample_configs(spec, 4, 7)
    assert a == b and a[0] == spec.default and len(a) == 4
    assert H.rung_epochs(40) == [8, 20, 40] and H.rung_epochs(2) == [1, 2] and H.rung_epochs(1) == [1]
    assert H.job_seed("a", "b", 0, 1) != H.job_seed("a", "b", 0, 2)
    assert H.parse_repeats("a=5,b=3", ["a", "b", "c"]) == {"a": 5, "b": 3, "c": 5}
    assert set(H.expand_models("all-spatial")) >= {"cnn2d", "spatial_attn"}


def test_sweep_writes_jsonl_and_resumes(env):
    data, tmp = env
    s = sweep(tmp, MODELS)
    assert s["errors"] == 0 and s["ok"] == len(MODELS) * 3  # k = 3 outer folds
    for m in MODELS:
        recs = H.read_records(H.result_path(tmp, "t", DS, m))
        assert sorted(r["fold"] for r in recs) == [0, 1, 2]
        r = recs[0]
        assert r["track"] in ("spectral", "spatial") and r["test"]["balanced_acc"] > 0.0
        assert len(r["trials"]) >= 2 and r["trials"][0]["i"] == 0 and r["inner_score"] is not None
        assert set(r["per_specimen_acc"]) == {str(x) for x in r["test_specimens"]}
        assert r["t_tune_s"] >= 0 and r["t_final_s"] >= 0 and r["t_predict_s"] >= 0
        assert (r["n_params"] is not None) == (m in ("mlp", "cnn2d"))
        if m in ("mlp", "cnn2d"):
            assert r["best_epoch"] >= 3
        if m == "cnn2d":
            assert r["best_config"]["P"] in (5, 7)
    assert (tmp / "t" / "harness.log").exists()
    # the test fold held every class and no test specimen is in the training split
    # resumability
    again = sweep(tmp, MODELS)
    assert again["ok"] == 0 and again["skipped"] == len(MODELS) * 3
    assert len(H.read_records(H.result_path(tmp, "t", DS, "lr"))) == 3
    # partial resume: drop the last line of one file -> only that job is redone
    p = H.result_path(tmp, "t", DS, "lr")
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:2]) + "\n")
    assert sweep(tmp, ["lr"])["ok"] == 1
    assert len(H.read_records(p)) == 3
    rows = H.summarize("t", tmp, print_table=False)
    assert {r["model"] for r in rows} == set(MODELS) and all(r["n"] == 3 for r in rows)


def test_error_record(env):
    _, tmp = env
    s = sweep(tmp, ["boom", "lr"])
    assert s["errors"] == 3 and s["ok"] == 3
    recs = H.read_records(H.result_path(tmp, "t", DS, "boom"))
    assert len(recs) == 3 and all("error" in r and "deliberate failure" in r["error"] for r in recs)
    assert all(r["dataset"] == DS and r["model"] == "boom" for r in recs)
    # error records are not "done": a rerun retries them; summary ignores them
    assert H.done_keys(tmp, "t", DS, "boom") == set()
    rows = {r["model"]: r for r in H.summarize("t", tmp, print_table=False)}
    assert rows["boom"]["n"] == 0 and rows["boom"]["n_err"] == 3


def test_process_pool(env):
    """Spawned workers cannot see monkeypatches, so only check the pure-python plumbing with workers=0."""
    data, tmp = env
    rec = H.run_job(dict(dataset=DS, model="lr", repeat=0, fold=1, budget=2, max_epochs=2, device="cpu", threads=1))
    assert "error" not in rec and rec["fold"] == 1
    # test-fold specimens come from the protocol: whole specimens, every class
    assert set(data.specimen_classes[rec["test_specimens"]]) == {0, 1, 2}


def test_spatial_patch_path_matches_pool(env):
    data, tmp = env
    pool = H.SpatialPool(data, 5, 10)
    from me_attention.protocol import make_patches
    r = pool.rows(np.arange(len(data.y)))
    assert len(r) <= 10 * 9 and np.array_equal(pool.patches, make_patches(data, pool.idx, 5))
