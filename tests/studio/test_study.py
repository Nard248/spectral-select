import json

import numpy as np
import pandas as pd
import pytest

from spectraforge.studio import api
from spectraforge.studio.study import (StudyError, compute_metric, make_design, param_unit, param_value, plan_runs,
                                       run_study, validate_spec)
from spectraforge.studio.study_schema import MetricSpec, ParameterSpec, StudySpec

from .fakes_study import COVER, FLUOR, base_experiment, fake_capture

ROI = {"shape": "rect", "x0": 0.4, "y0": 0.4, "x1": 0.9, "y1": 0.9}
ROI_B = {"shape": "rect", "x0": 0.0, "y0": 0.0, "x1": 0.2, "y1": 0.2}      # the bright corner of the fake


def params():
    return [{"pointer": COVER, "label": "cover", "low": 0.1, "high": 1.0,
             "also_set": ["/objects/0/shape/layers/2/thickness/end_thickness_mm"]},
            {"pointer": FLUOR, "label": "fluor", "low": 0.2, "high": 1.0}]


def spec(kind="lhs", n=6, reps=2, **kw):
    return StudySpec(name="t", base_experiment=base_experiment(), parameters=params(),
                     design={"kind": kind, "n": n, "seed": 1, "levels": 4}, replicates=reps, mode="preview",
                     metrics=[{"name": "signal", "kind": "band_sum", "roi": ROI}], **kw)


# ------------------------------------------------------------------ schema
def test_schema_validation():
    with pytest.raises(ValueError):
        ParameterSpec(pointer="x", low=0, high=1)
    with pytest.raises(ValueError):
        ParameterSpec(pointer="/a", low=1, high=0)
    with pytest.raises(ValueError):
        ParameterSpec(pointer="/a", low=0, high=1, scale="log")
    with pytest.raises(ValueError):
        ParameterSpec(pointer="/a", kind="choice", values=[1])
    with pytest.raises(ValueError):
        MetricSpec(name="c", kind="contrast", roi=ROI)             # needs roi_b
    with pytest.raises(ValueError):
        StudySpec(base_experiment="layered_wedge_phantom", parameters=params() + params(),
                  metrics=[{"name": "a", "kind": "band_sum"}])    # duplicate labels
    s = StudySpec.model_validate_json(spec().model_dump_json())
    assert s.parameters[1].label == "fluor"
    assert StudySpec(base_experiment="layered_wedge_phantom", parameters=params(),
                     metrics=[{"name": "a", "kind": "band_sum"}]).resolve_base().name


def test_parameter_value_mapping():
    p = ParameterSpec(pointer="/a", low=1.0, high=100.0, scale="log")
    assert param_value(p, 0) == pytest.approx(1.0) and param_value(p, 1) == pytest.approx(100.0)
    assert param_value(p, 0.5) == pytest.approx(10.0)
    assert param_unit(p, 10.0) == pytest.approx(0.5)
    q = ParameterSpec(pointer="/a", kind="integer", low=2, high=5)
    assert {param_value(q, u) for u in np.linspace(0, 1, 50)} == {2, 3, 4, 5}
    c = ParameterSpec(pointer="/a", kind="choice", values=["x", "y", "z"])
    assert [param_value(c, u) for u in (0, 0.4, 0.99, 1.0)] == ["x", "y", "z", "z"]
    assert param_unit(c, "y") == pytest.approx(0.5)


# ------------------------------------------------------------------ designs
@pytest.mark.parametrize("kind,n,rows", [("random", 7, 7), ("lhs", 7, 7), ("sobol", 8, 8), ("grid", 0, 16),
                                         ("saltelli", 8, 8 * 4), ("morris", 5, 5 * 3)])
def test_design_shapes_bounds_determinism(kind, n, rows):
    s = spec(kind, n=max(n, 1))
    d1, d2 = make_design(s), make_design(s)
    assert len(d1) == rows and np.array_equal(d1.U, d2.U)
    assert d1.U.min() >= 0 and d1.U.max() <= 1
    lo, hi = zip(*[(p.low, p.high) for p in s.parameters])
    v = np.array(d1.values, dtype=float)
    assert (v >= np.array(lo) - 1e-12).all() and (v <= np.array(hi) + 1e-12).all()
    s2 = spec(kind, n=max(n, 1))
    s2.design.seed = 99
    if kind not in ("grid",):
        assert not np.array_equal(make_design(s2).U, d1.U)


def test_lhs_is_stratified():
    d = make_design(spec("lhs", n=10))
    for j in range(2):
        assert sorted((d.U[:, j] * 10).astype(int)) == list(range(10))


def test_saltelli_structure():
    s = spec("saltelli", n=16)
    d = make_design(s)
    n, k = 16, 2
    A, B = d.U[:n], d.U[n:2 * n]
    for i in range(k):
        AB = d.U[(2 + i) * n:(3 + i) * n]
        assert np.array_equal(AB[:, i], B[:, i])
        assert np.array_equal(np.delete(AB, i, 1), np.delete(A, i, 1))
    assert list(d.table["block"].unique()) == ["A", "B", "AB:0", "AB:1"]


def test_morris_trajectories_are_oat_with_delta():
    s = spec("morris", n=6)
    s.design.levels = 4
    d = make_design(s)
    delta = 4 / (2 * 3)
    for t in range(6):
        idx = np.where(d.table["traj"] == t)[0]
        assert len(idx) == 3
        seen = set()
        for a, b in zip(idx[:-1], idx[1:]):
            diff = d.U[b] - d.U[a]
            i = int(d.table["step_param"].iloc[b])
            seen.add(i)
            assert np.count_nonzero(np.abs(diff) > 1e-12) == 1
            assert diff[i] == pytest.approx(delta * d.table["direction"].iloc[b])
        assert seen == {0, 1}
    with pytest.raises(StudyError):
        s.design.levels = 5
        make_design(s)


def test_grid_with_choice_and_integer():
    s = StudySpec(base_experiment=base_experiment(), design={"kind": "grid", "levels": 3},
                  parameters=[{"pointer": "/acquisition/seed", "kind": "integer", "low": 0, "high": 2},
                              {"pointer": "/acquisition/preview_scale", "kind": "choice", "values": [0.2, 0.3]}],
                  metrics=[{"name": "a", "kind": "band_sum"}])
    d = make_design(s)
    assert len(d) == 6 and {tuple(v) for v in d.values} == {(a, b) for a in (0, 1, 2) for b in (0.2, 0.3)}


def test_validate_spec_catches_bad_pointer_and_range():
    bad = spec()
    bad.parameters[0].pointer = "/objects/0/shape/layers/2/thickness/nope"
    with pytest.raises(StudyError):
        validate_spec(bad)
    bad2 = spec()
    bad2.parameters[1].low = -5.0                      # thickness must be >= 0
    with pytest.raises(StudyError):
        validate_spec(bad2)


def test_max_runs_guard():
    s = spec("lhs", n=50, reps=10)
    s.max_runs = 100
    with pytest.raises(StudyError):
        plan_runs(s)


# ------------------------------------------------------------------ metrics
def test_metrics_on_fake_capture():
    res = fake_capture(base_experiment())
    ex = res.excitations_nm[0]
    M = lambda **kw: MetricSpec(name="m", **kw)           # noqa: E731
    v = compute_metric(res, M(kind="roi_band", roi=ROI, band_nm=460.0))
    assert 100 < v < 1000
    s = compute_metric(res, M(kind="band_sum", roi=ROI))
    assert s > v
    assert compute_metric(res, M(kind="peak_wavelength", roi=ROI)) == pytest.approx(460.0, abs=3.0)
    r = compute_metric(res, M(kind="band_ratio", roi=ROI, a={"ex": ex, "band_nm": 440.0}, b={"ex": ex, "band_nm": 460.0}))
    assert 0 < r < 1
    assert compute_metric(res, M(kind="gt_stat", gt="layer_thickness:cover")) == pytest.approx(
        base_experiment().objects[0].shape.layers[2].thickness.thickness_mm, rel=1e-5)
    assert compute_metric(res, M(kind="eem_integral", roi=ROI)) > 0
    # corner patch is 3x brighter: Michelson contrast (3-1)/(3+1)=0.5 on the noise-free part
    c = compute_metric(res, M(kind="contrast", roi=ROI_B, roi_b=ROI, norm={"mode": "radiance"},
                                  inner={"kind": "band", "ex": ex, "band_nm": 460.0}))
    assert 0.45 < c < 0.51
    w = compute_metric(res, M(kind="contrast", contrast="weber", roi=ROI_B, roi_b=ROI, norm={"mode": "radiance"},
                              inner={"kind": "band", "ex": ex, "band_nm": 460.0}))
    assert 1.8 < w < 2.2
    assert compute_metric(res, M(kind="cnr", roi=ROI_B, roi_b=ROI, norm={"mode": "radiance"}, inner={"kind": "band", "ex": ex, "band_nm": 460.0})) > 5
    # pixel units: same ROI expressed in pixels gives the same number
    px = {"shape": "rect", "x0": 0.4 * 19, "y0": 0.4 * 15, "x1": 0.9 * 19, "y1": 0.9 * 15}
    assert compute_metric(res, M(kind="band_sum", roi=px, roi_units="pixels")) == pytest.approx(s)


# ------------------------------------------------------------------ runner
def test_run_study_fake_end_to_end_and_resume(tmp_path):
    s = spec("lhs", n=6, reps=2)
    calls = []

    def cap(exp, mode, ex):
        calls.append(exp.acquisition.seed)
        return fake_capture(exp, mode, ex)

    out = tmp_path / "st"
    r = run_study(s, out, capture_fn=cap)
    assert r.state == "done" and len(r.runs) == 12 and len(calls) == 12
    assert (out / "runs.parquet").exists() and (out / "runs.csv").exists() and (out / "manifest.json").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_completed"] == 12 and man["provenance"]["numpy"] and man["spec_hash"]
    df = pd.read_parquet(out / "runs.parquet")
    assert {"p:cover", "p:fluor", "u:cover", "m:signal", "seed", "exp_hash", "t_capture_s", "status"} <= set(df.columns)
    assert sorted(df["seed"].unique()) == [0, 1]                       # replicate seeds
    # the analytic fake: signal decreases with cover thickness
    mean = df.groupby("point")[["p:cover", "m:signal"]].mean()
    assert np.corrcoef(mean["p:cover"], mean["m:signal"])[0, 1] < -0.5
    # resume: nothing re-run, identical table
    calls.clear()
    r2 = run_study(s, out, capture_fn=cap)
    assert calls == [] and r2.manifest["n_resumed"] == 12
    pd.testing.assert_frame_equal(r2.runs.drop(columns=["t_capture_s", "t_metrics_s"]),
                                  r.runs.drop(columns=["t_capture_s", "t_metrics_s"]))
    # determinism across fresh directories
    r3 = run_study(s, tmp_path / "other", capture_fn=cap)
    pd.testing.assert_frame_equal(r3.runs.drop(columns=["t_capture_s", "t_metrics_s"]),
                                  r.runs.drop(columns=["t_capture_s", "t_metrics_s"]))


def test_cancel_then_resume_completes(tmp_path):
    s = spec("random", n=5, reps=1)
    done = []

    def cap(exp, mode, ex):
        done.append(1)
        return fake_capture(exp, mode, ex)

    r = run_study(s, tmp_path, capture_fn=cap, cancel=lambda: len(done) >= 2)
    assert r.state == "cancelled" and len(r.runs) == 2
    r2 = run_study(s, tmp_path, capture_fn=cap)
    assert r2.state == "done" and len(r2.runs) == 5 and len(done) == 5 and r2.manifest["n_resumed"] == 2


def test_failed_run_is_recorded_not_fatal(tmp_path):
    s = spec("random", n=3, reps=1)

    def cap(exp, mode, ex):
        if exp.objects[0].shape.layers[1].thickness.thickness_mm > 0.0 and cap.n == 1:
            cap.n += 1
            raise RuntimeError("boom")
        cap.n += 1
        return fake_capture(exp, mode, ex)
    cap.n = 0
    r = run_study(s, tmp_path, capture_fn=cap)
    assert r.state == "done" and r.manifest["n_failed"] == 1
    assert (r.runs["status"] == "error").sum() == 1 and "boom" in r.runs["error"].iloc[1]
    # resume retries only the failed run
    r2 = run_study(s, tmp_path, capture_fn=lambda e, m, x: fake_capture(e, m, x))
    assert r2.manifest["n_failed"] == 0 and r2.manifest["n_resumed"] == 2


def test_keep_captures(tmp_path):
    r = run_study(spec("random", n=2, reps=1), tmp_path, capture_fn=fake_capture, keep_captures=True)
    files = sorted((tmp_path / "captures").glob("*.npz"))
    assert len(files) == 2
    assert api.load_capture(files[0]).meta["provenance"]["experiment_hash"] in set(r.runs["exp_hash"])


def test_progress_callback(tmp_path):
    seen = []
    run_study(spec("random", n=3, reps=1), tmp_path, capture_fn=fake_capture, progress=lambda f, m: seen.append(f))
    assert seen[-1] == 1.0 and seen == sorted(seen)


# ------------------------------------------------------------------ end to end on the real engine
def test_wedge_phantom_real_engine_study(tmp_path):
    """2-parameter LHS of 6 points x 2 replicates on layered_wedge_phantom (preview), < 60 s."""
    import time
    base = api.set_many(api.preset("layered_wedge_phantom"), {"/acquisition/volume_mc/excitation_photons": 100_000,
                                                              "/acquisition/volume_mc/emission_photons": 100_000})
    s = StudySpec(name="wedge", base_experiment=base, parameters=params(), design={"kind": "lhs", "n": 6, "seed": 1},
                  replicates=2, mode="preview",
                  metrics=[{"name": "signal", "kind": "band_sum", "roi": ROI},
                           {"name": "cover_gt", "kind": "gt_stat", "gt": "layer_thickness:cover", "roi": ROI}])
    t0 = time.time()
    r = run_study(s, tmp_path)
    assert time.time() - t0 < 180      # ~25-35 s on an idle machine (incl. numba JIT); generous for shared CI
    assert r.state == "done" and len(r.runs) == 12 and r.manifest["n_failed"] == 0
    assert np.isfinite(r.runs["m:signal"]).all()
    # the gt_stat metric must reproduce the parameter exactly (end thickness is linked)
    assert np.allclose(r.runs["m:cover_gt"], r.runs["p:cover"], atol=1e-3)
    assert r.runs.groupby("point")["m:signal"].std().max() / r.runs["m:signal"].mean() < 0.05   # replicates agree


def test_process_pool_gives_identical_results(tmp_path):
    """workers=2 (spawn pool, real engine) reproduces the serial table exactly."""
    s = StudySpec(name="w", base_experiment="generic_macro_bench", replicates=1,
                  parameters=[{"pointer": "/acquisition/preview_scale", "label": "s", "low": 0.1, "high": 0.2}],
                  design={"kind": "random", "n": 3, "seed": 1}, metrics=[{"name": "sig", "kind": "band_sum"}])
    a = run_study(s, tmp_path / "a", workers=1)
    b = run_study(s, tmp_path / "b", workers=2)
    drop = ["t_capture_s", "t_metrics_s"]
    pd.testing.assert_frame_equal(a.runs.drop(columns=drop), b.runs.drop(columns=drop))
    assert b.manifest["workers"] == 2
