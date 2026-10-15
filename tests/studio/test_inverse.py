"""Surrogate construction, ML retrieval, profile CIs, CRLB efficiency and thickness maps (v1.2 W4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fakes_inverse as fk  # noqa: E402
from spectraforge.studio import information as info  # noqa: E402
from spectraforge.studio import inverse as inv  # noqa: E402
from spectraforge.studio import sensor as sensor_mod  # noqa: E402
from spectraforge.studio.schema import Experiment  # noqa: E402
from spectraforge.studio.surrogate import (ChannelTable, NoiseModel, ParamSpec, Surrogate,  # noqa: E402
                                            apply_params, build_surrogate, simulate_channels)
from spectraforge.studio.sweeps import set_pointer  # noqa: E402

SP = ParamSpec("cover", (fk.COVER_T, fk.COVER_T_END))


def _exp(t=0.8, kind="uniform", end=None):
    d = fk.base_experiment().model_dump(mode="json")
    for p, v in ((fk.COVER_KIND, kind), (fk.COVER_T, t), (fk.COVER_T_END, t if end is None else end)):
        set_pointer(d, p, v)
    return Experiment.model_validate(d)


@pytest.fixture(scope="module")
def sur():
    return build_surrogate(_exp(), [SP], {"cover": np.linspace(0.1, 1.6, 16)}, capture_fn=fk.fake_capture, cache=False)


def test_noise_model_matches_the_detector():
    """Variance of ROI-mean ADU from ``sensor.detect`` vs the Poisson-Gaussian NoiseModel (shot+dark+read+quantisation)."""
    exp = _exp()
    cm = simulate_channels(exp, None, scale=0.5, capture_fn=fk.fake_capture)
    res = fk.fake_capture(exp, scale=0.5)
    cam, tg = exp.camera(), exp.acquisition.physics
    ex = 365.0
    e = res.electrons_mean[ex]
    rng = np.random.default_rng(0)
    roi = (slice(30, 60), slice(50, 120))
    n = e[roi].shape[0] * e[roi].shape[1]
    frames = np.stack([sensor_mod.detect(e, res.exposure_ms[ex], cam.sensor, tg, None, rng)[0][roi].mean(axis=(0, 1))
                       for _ in range(400)])                                  # (frames, bands)
    nm = cm.noise.with_n_pix(n)
    sel = np.abs(cm.channels.ex_nm - ex) < 1e-6
    pred = nm.subset(sel).variance_adu(e[roi].mean(axis=(0, 1)))
    ratio = frames.var(axis=0, ddof=1) / pred
    assert np.all(np.abs(ratio - 1) < 0.25), ratio                              # 400 frames -> 7 % sd of a variance
    assert abs(ratio.mean() - 1) < 0.06


def test_surrogate_interpolates_and_roundtrips(tmp_path, sur):
    th = 0.43
    truth = simulate_channels(_exp(th), None, capture_fn=fk.fake_capture).mean_e
    assert np.max(np.abs(sur.predict_e([th]) - truth) / truth) < 2e-3          # cubic spline, 16 points
    p = sur.save(tmp_path / "s.npz")
    s2 = Surrogate.load(p)
    assert np.allclose(s2.predict_e([th]), sur.predict_e([th]))
    assert s2.names == ["cover"] and s2.noise.n_pix == sur.noise.n_pix


def test_cache_hit_and_determinism(tmp_path):
    kw = dict(capture_fn=fk.fake_capture, cache=tmp_path)
    a = build_surrogate(_exp(), [SP], {"cover": np.linspace(0.1, 1.0, 6)}, **kw)
    b = build_surrogate(_exp(), [SP], {"cover": np.linspace(0.1, 1.0, 6)}, **kw)
    assert a.meta["cache"] == "miss" and b.meta["cache"] == "hit"
    assert np.array_equal(a.values_e, b.values_e)


def test_two_parameter_grid_and_sobol_surrogates():
    exp = _exp()
    sp2 = ParamSpec("end", (fk.COVER_T_END,), lo=0.1, hi=1.6)
    sp1 = ParamSpec("start", (fk.COVER_T,), lo=0.1, hi=1.6)
    g = build_surrogate(exp, [sp1, sp2], {"start": np.linspace(0.1, 1.6, 7), "end": np.linspace(0.1, 1.6, 7)},
                        capture_fn=fk.fake_capture, cache=False)
    assert g.values_e.shape[0] == 49
    sob = build_surrogate(exp, [sp1, sp2], {"start": (0.1, 1.6), "end": (0.1, 1.6)}, mode="sobol", n_points=64,
                          seed=3, capture_fn=fk.fake_capture, cache=False)
    # the fake model depends on the MEAN thickness at the centre (wedge mean), check interpolation at an interior point
    th = [0.6, 1.1]
    truth = simulate_channels(apply_params(exp, [sp1, sp2], th), None, capture_fn=fk.fake_capture).mean_e
    assert np.max(np.abs(g.predict_e(th) - truth) / truth) < 0.05
    assert np.max(np.abs(sob.predict_e(th) - truth) / truth) < 0.05


def test_param_shift_mode_keeps_wedge_slope():
    exp = _exp(1.5, "linear", 0.05)
    p = ParamSpec("t0", (fk.COVER_T, fk.COVER_T_END), mode="shift")
    prof = apply_params(exp, [p], [1.8]).objects[0].shape.layers[2].thickness
    assert prof.thickness_mm == pytest.approx(1.8)
    assert prof.thickness_mm - prof.end_thickness_mm == pytest.approx(1.45, abs=1e-9)      # slope unchanged
    with pytest.raises(Exception, match="invalid experiment"):
        apply_params(exp, [p], [1.0])                                                       # end would be negative


def test_fit_recovers_truth_and_chi2_is_one(sur):
    rng = np.random.default_rng(1)
    th = 0.9
    ests, chis = [], []
    for _ in range(40):
        y = inv.simulate_observation(sur.predict_e([th]), sur.noise, rng)
        r = inv.fit(y, sur)
        ests.append(r.theta[0])
        chis.append(r.reduced_chi2)
    assert abs(np.mean(ests) - th) < 3 * np.std(ests) / np.sqrt(40) + 1e-4
    assert 0.7 < np.mean(chis) < 1.3


def test_crlb_efficiency_200_realisations(sur):
    """Empirical SD of the ML estimator over 200 noisy realisations agrees with the CRLB within 20 %."""
    for th in (0.3, 0.9, 1.4):
        mc = inv.monte_carlo_fit(sur, [th], n_real=200, seed=7)
        assert abs(mc["ratio"][0] - 1) < 0.2, (th, mc["ratio"])
        assert abs(mc["mean_fit_se"][0] / mc["crlb_sd"][0] - 1) < 0.05            # fit covariance = (J^T W J)^-1


def test_profile_likelihood_ci_matches_wald_and_covers(sur):
    rng = np.random.default_rng(5)
    th = 0.7
    cover = 0
    for _ in range(60):
        y = inv.simulate_observation(sur.predict_e([th]), sur.noise, rng)
        r = inv.fit(y, sur, profile=True, level=0.95)
        ci = r.ci["cover"]
        cover += ci["lo"] <= th <= ci["hi"]
        assert not ci["open_lo"] and not ci["open_hi"]
        assert (ci["hi"] - ci["lo"]) / (2 * 1.96 * r.se[0]) == pytest.approx(1.0, abs=0.1)
    assert 0.85 <= cover / 60 <= 1.0


def _two_param_surrogate():
    rng = np.random.default_rng(4)
    C = 24
    A, k, B = rng.uniform(300, 1500, C), rng.uniform(0.3, 2.0, C), rng.uniform(100, 800, C)
    ga, gb = np.linspace(0.1, 2.0, 10), np.linspace(0.5, 3.0, 10)
    pts = np.array([(a, b) for a in ga for b in gb])
    V = A[None] * np.exp(-k[None] * pts[:, :1]) + B[None] * np.log1p(pts[:, 1:2])
    nm = NoiseModel(1.5, 8.0, np.full(C, 1.0), n_pix=40.0)
    return Surrogate([ParamSpec("a", ("/x",)), ParamSpec("b", ("/y",))], pts, V,
                     ChannelTable(np.repeat([365.0, 430.0], C // 2), np.tile(np.arange(C // 2) * 10.0 + 450, 2)), nm,
                     grid_axes=[ga, gb], interp="cubic")


def test_two_parameter_fit_efficiency_and_profile_ci():
    s = _two_param_surrogate()
    th = [0.9, 1.7]
    mc = inv.monte_carlo_fit(s, th, n_real=200, seed=2)
    assert np.all(np.abs(mc["ratio"] - 1) < 0.2), mc["ratio"]
    assert np.all(np.abs(mc["bias"]) < 3 * mc["empirical_sd"] / np.sqrt(200))
    y = inv.simulate_observation(s.predict_e(th), s.noise, np.random.default_rng(0))
    r = inv.fit(y, s, profile=True)
    for j, nm in enumerate(["a", "b"]):
        ci = r.ci[nm]
        assert ci["lo"] < th[j] < ci["hi"] and not ci["open_lo"] and not ci["open_hi"]
        assert (ci["hi"] - ci["lo"]) / (2 * 1.96 * r.se[j]) == pytest.approx(1.0, abs=0.15)


def test_open_interval_when_unidentifiable_signal():
    """Flat signal in the upper part of the range: the profile never crosses the threshold -> open interval flagged."""
    g = np.linspace(0, 2, 15)
    C = 6
    base = np.linspace(100, 300, C)
    V = base[None, :] * (1 + 0.3 * np.tanh(6 * (0.5 - g))[:, None] * np.linspace(0.5, 1, C)[None, :])
    nm = NoiseModel(1.5, 8.0, np.zeros(C), n_pix=20.0)
    s = Surrogate([ParamSpec("a", ("/x",))], g[:, None], V, ChannelTable(np.full(C, 400.0), np.arange(C) * 10.0 + 450),
                  nm, grid_axes=[g])
    y = s.predict_e([1.8])
    r = inv.fit(y, s, profile=True)
    assert r.ci["a"]["open_hi"] or r.ci["a"]["hi"] >= 1.99


def test_retrieve_map_against_ground_truth_wedge(sur):
    exp = _exp(1.4, "linear", 0.2)
    res = fk.fake_capture(exp, scale=0.25, noisy=True, seed=11)
    # surrogate for the uniform-cover model, retrieval on the wedge (model = local flat cover)
    m = inv.retrieve_map(res, sur, tile=4, gt_name="layer_thickness:collagen")
    st = m.stats
    assert st["n_scored"] > 50
    assert abs(st["bias"]) < 0.02
    assert st["rmse"] < 0.05
    assert 0.8 < st["coverage_1p96se"] <= 1.0                                   # predicted SE is calibrated


def test_observation_from_capture_roundtrip(sur):
    res = fk.fake_capture(_exp(0.5), scale=0.25, noisy=False)
    y, n = inv.observation_from_capture(res, sur)
    truth = sur.predict_e([0.5])
    # noise-free ADU -> electrons (dark removed): equals the noise-free signal up to ADU rounding (quantisation off? on)
    assert np.max(np.abs(y - truth) / truth) < 0.02
    assert n > 10


# --------------------------------------------------------------------------------------------
# HTTP router (fake simulator, real job worker)
# --------------------------------------------------------------------------------------------
def _client():
    import time

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from spectraforge.studio.jobs import JobManager
    from spectraforge.studio.routes_inverse import router
    from spectraforge.studio.server import CaptureStore, StoredCapture

    app = FastAPI()
    app.include_router(router)
    app.state.jobs = JobManager()
    app.state.store = CaptureStore(4)
    app.state.sweeps = None
    app.state.inverse_capture_fn = fk.fake_capture
    cl = TestClient(app)

    def wait(job_id, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = app.state.jobs.get(job_id)
            if j.state in ("done", "error", "cancelled"):
                assert j.state == "done", j.error
                return j.capture_id
            time.sleep(0.02)
        raise AssertionError("job timeout")
    return app, cl, wait, StoredCapture


def test_router_surrogate_fit_fisher_design_flow():
    app, cl, wait, StoredCapture = _client()
    exp = _exp(0.8).model_dump(mode="json")
    params = [{"name": "cover", "paths": [fk.COVER_T, fk.COVER_T_END]}]
    r = cl.post("/api/surrogates", json={"experiment": exp, "parameters": params, "cache": False,
                                         "grid": {"cover": list(np.linspace(0.1, 1.6, 12))}})
    assert r.status_code == 200, r.text
    sid = wait(r.json()["job_id"])
    s = cl.get(f"/api/surrogates/{sid}").json()
    assert s["names"] == ["cover"] and s["n_channels"] == 16 and s["loo"]["loo_rel_rms_max"] < 0.01
    # fit a noisy capture
    res = fk.fake_capture(_exp(0.9), scale=0.25, noisy=True, seed=3)
    app.state.store.put(StoredCapture(meta={"id": "cap1"}, result=res, experiment=_exp(0.9), scale=0.25))
    f = cl.post("/api/fit", json={"surrogate_id": sid, "observation": {"capture_id": "cap1"}, "profile": True})
    assert f.status_code == 200, f.text
    fj = f.json()
    assert abs(fj["theta"][0] - 0.9) < 4 * fj["se"][0] + 0.01
    assert fj["ci"]["cover"]["lo"] < fj["theta"][0] < fj["ci"]["cover"]["hi"]
    # thickness map
    m = cl.post("/api/fit", json={"surrogate_id": sid, "observation": {"capture_id": "cap1"},
                                  "map": {"tile": 8, "gt_name": "layer_thickness:collagen"}})
    assert m.status_code == 200 and m.json()["stats"]["n_scored"] > 10
    # Fisher from the surrogate -> design
    fi = cl.post("/api/fisher", json={"surrogate_id": sid, "theta": [0.9]}).json()
    assert fi["crlb_sd"][0] > 0 and len(fi["info_per_channel"]) == 16
    d = cl.post("/api/design", json={"fisher_id": fi["fisher_id"], "k": 3, "criterion": "D",
                                     "budget": "exposure_time"}).json()
    assert len(d["selected"]) == 3 and len(d["crlb_sd_vs_k"]) == 3 and d["exposure"]["objective_opt"] is not None
    # Fisher from the simulator is a job
    j = cl.post("/api/fisher", json={"experiment": exp, "parameters": params, "step": 0.02}).json()
    rid = wait(j["job_id"])
    fr = cl.get(f"/api/inverse/results/{rid}").json()
    assert fr["crlb_sd"][0] == pytest.approx(fi["crlb_sd"][0], rel=0.15)
    j2 = cl.post("/api/design", json={"experiment": exp, "parameters": params, "k": 2, "step": 0.02}).json()
    dr = cl.get(f"/api/inverse/results/{wait(j2['job_id'])}").json()
    assert dr["k"] == 2


def test_router_errors_are_client_errors():
    app, cl, wait, _ = _client()
    assert cl.post("/api/unmix", json={"capture_id": "nope"}).status_code == 404
    assert cl.post("/api/fit", json={"surrogate_id": "sur-x"}).status_code == 404
    assert cl.post("/api/surrogates", json={"parameters": []}).status_code == 422
    exp = _exp(0.8).model_dump(mode="json")
    r = cl.post("/api/surrogates", json={"experiment": exp, "parameters": [{"path": "/nope/x"}],
                                         "grid": {"x": [1, 2, 3]}})
    assert r.status_code in (200, 422)
    r = cl.post("/api/surrogates", json={"experiment": exp, "parameters": [{"path": fk.COVER_T}],
                                         "grid": {"thickness_mm": list(range(1, 400))}})
    assert r.status_code == 422                                 # exceeds the simulator-run limit
    assert cl.post("/api/design", json={"fisher_id": "fis-x", "k": 0}).status_code == 422


def test_router_rejects_non_boolean_cache_path():
    app, cl, wait, _ = _client()
    exp = _exp(0.8).model_dump(mode="json")
    r = cl.post("/api/surrogates", json={"experiment": exp, "parameters": [{"name": "c", "paths": [fk.COVER_T]}],
                                         "grid": {"c": [0.1, 0.5, 1.0]}, "cache": "../../x"})
    assert r.status_code == 422 and "boolean" in r.text


def test_fit_scaled_is_invariant_to_illumination_scale(sur):
    rng = np.random.default_rng(2)
    th = 0.8
    ests = []
    for s in (0.4, 1.0, 2.5):
        y = inv.simulate_observation(s * sur.predict_e([th]), sur.noise, rng)
        r = inv.fit_scaled(y, sur)
        ests.append(r.theta[0])
        assert r.scale == pytest.approx(s, rel=0.05)
    assert max(ests) - min(ests) < 0.1
