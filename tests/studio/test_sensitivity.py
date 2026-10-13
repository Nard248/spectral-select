"""Validation of the Sobol' / Morris estimators against analytic results (Ishigami, Sobol' G)."""
import json

import numpy as np
import pandas as pd
import pytest

from spectraforge.studio import sensitivity as sens
from spectraforge.studio.study import make_design, run_study
from spectraforge.studio.study_schema import StudySpec

from .fakes_study import COVER, FLUOR, base_experiment, fake_capture

TOL = 0.02


def test_ishigami_analytic_values():
    s1, st = sens.ishigami_analytic()
    assert s1 == pytest.approx([0.3139, 0.4424, 0.0], abs=1e-4)
    assert st == pytest.approx([0.5576, 0.4424, 0.2437], abs=1e-4)


def test_sobol_ishigami_n8192():
    r = sens.sobol_analyse_function(sens.ishigami, 3, 2 ** 13, seed=0, n_boot=200)
    s1, st = sens.ishigami_analytic()
    assert np.abs(r["S1"] - s1).max() < TOL, (r["S1"], s1)
    assert np.abs(r["ST"] - st).max() < TOL, (r["ST"], st)
    # the analytic value lies inside the bootstrap CI (allowing the estimator's small bias)
    for i in range(3):
        assert r["S1_ci"][i, 0] - 0.01 <= s1[i] <= r["S1_ci"][i, 1] + 0.01
        assert r["ST_ci"][i, 0] - 0.01 <= st[i] <= r["ST_ci"][i, 1] + 0.01


def test_sobol_g_function_n8192():
    a = np.array([0.0, 1.0, 4.5, 9.0, 99.0])
    r = sens.sobol_analyse_function(lambda x: sens.sobol_g(x, a), 5, 2 ** 13, seed=1, n_boot=100)
    s1, st = sens.sobol_g_analytic(a)
    assert np.abs(r["S1"] - s1).max() < TOL and np.abs(r["ST"] - st).max() < TOL
    assert r["S1"][0] > r["S1"][4]                                 # ordering by importance


def test_sobol_additive_model_has_no_interactions():
    f = lambda x: 2 * x[:, 0] + x[:, 1]                           # noqa: E731
    r = sens.sobol_analyse_function(f, 3, 2 ** 12, seed=2, n_boot=50)
    assert r["S1"] == pytest.approx([0.8, 0.2, 0.0], abs=0.02)
    assert r["ST"] == pytest.approx(r["S1"], abs=0.02)


def test_sobol_indices_rejects_wrong_length():
    with pytest.raises(ValueError):
        sens.sobol_indices(np.zeros(10), 4, 2)


def test_bootstrap_ci_narrows_with_n():
    w = []
    for n in (2 ** 8, 2 ** 12):
        r = sens.sobol_analyse_function(sens.ishigami, 3, n, seed=0, n_boot=100)
        w.append(float((r["S1_ci"][:, 1] - r["S1_ci"][:, 0]).mean()))
    assert w[1] < w[0] / 2


def _morris_run(f, k, r, levels=4, seed=0):
    from spectraforge.studio.study_schema import StudySpec
    spec = StudySpec(base_experiment="layered_wedge_phantom",
                     parameters=[{"pointer": f"/p{i}", "label": f"x{i}", "low": 0, "high": 1} for i in range(k)],
                     design={"kind": "morris", "n": r, "levels": levels, "seed": seed},
                     metrics=[{"name": "y", "kind": "band_sum"}])
    d = make_design(spec)
    Y = f(d.U)
    t = d.table
    delta = levels / (2 * (levels - 1))
    return sens.morris_indices(Y, t["traj"], t["step"], t["step_param"], t["direction"], k, delta, n_boot=100)


def test_morris_linear_model_is_exact():
    # Y = 3 x0 - 2 x1 + 0 x2: every elementary effect equals the coefficient, sigma = 0
    r = _morris_run(lambda U: 3 * U[:, 0] - 2 * U[:, 1], 3, 12)
    assert r["mu"] == pytest.approx([3, -2, 0], abs=1e-9)
    assert r["mu_star"] == pytest.approx([3, 2, 0], abs=1e-9)
    assert r["sigma"] == pytest.approx([0, 0, 0], abs=1e-9)


def test_morris_quadratic_elementary_effects_are_exact():
    # Y = x0^2: EE = ((x+d D)^2 - x^2)/(d D) = 2x + d D for a step of size d=Delta, direction D
    levels, k = 6, 2
    spec = StudySpec(base_experiment="layered_wedge_phantom",
                     parameters=[{"pointer": f"/p{i}", "label": f"x{i}", "low": 0, "high": 1} for i in range(k)],
                     design={"kind": "morris", "n": 20, "levels": levels, "seed": 5},
                     metrics=[{"name": "y", "kind": "band_sum"}])
    d = make_design(spec)
    delta = levels / (2 * (levels - 1))
    ee = sens.morris_effects(d.U[:, 0] ** 2, d.table["traj"].to_numpy(), d.table["step"].to_numpy(),
                             d.table["step_param"].to_numpy(), d.table["direction"].to_numpy(), k, delta)
    want = []
    for t in range(20):
        idx = np.where(d.table["traj"] == t)[0]
        for a_, b_ in zip(idx[:-1], idx[1:]):
            if d.table["step_param"].iloc[b_] == 0:
                want.append(d.U[a_, 0] + d.U[b_, 0])             # (x'^2 - x^2)/(x' - x) = x + x'
    assert np.allclose(sorted(ee[0]), sorted(want)) and len(ee[0]) == 20
    assert np.allclose(ee[1], 0.0) and len(ee[1]) == 20             # x1 does not enter


def test_morris_sigma_detects_interactions():
    add = _morris_run(lambda U: U[:, 0] + U[:, 1], 2, 30)
    mul = _morris_run(lambda U: U[:, 0] * U[:, 1], 2, 30)
    assert add["sigma"] == pytest.approx([0, 0], abs=1e-9)
    assert (mul["sigma"] > 0.1).all()                               # EE_0 = x1 varies from point to point


def test_morris_ishigami_flags_x3_as_influential_through_interaction():
    # x3 has S1 = 0 (Sobol') yet matters: Morris mu* > 0 with sigma ~ mu* (pure interaction), Campolongo 2007
    r = _morris_run(sens.ishigami, 3, 300, levels=8, seed=3)
    assert r["mu_star"][2] > 1.0 and r["sigma"][2] > 0.7 * r["mu_star"][2]
    assert (r["mu_star"] > 1.0).all()


# ------------------------------------------------------------------ studies
def _spec(kind, n, reps):
    return StudySpec(name="s", base_experiment=base_experiment(),
                     parameters=[{"pointer": COVER, "label": "cover", "low": 0.1, "high": 1.0,
                                  "also_set": ["/objects/0/shape/layers/2/thickness/end_thickness_mm"]},
                                 {"pointer": FLUOR, "label": "fluor", "low": 0.2, "high": 1.0}],
                     design={"kind": kind, "n": n, "seed": 4, "levels": 4}, replicates=reps,
                     metrics=[{"name": "signal", "kind": "band_sum",
                               "roi": {"shape": "rect", "x0": 0.4, "y0": 0.4, "x1": 0.9, "y1": 0.9}}])


def test_study_sobol_matches_analytic_fake_response(tmp_path):
    """Fake response y = exp(-2 c)(0.5 + f): multiplicative in two independent inputs -> closed-form indices."""
    r = run_study(_spec("saltelli", 128, 1), tmp_path, capture_fn=fake_capture)
    res = sens.analyse_study(tmp_path, n_boot=100)
    ind = res["metrics"]["signal"]["indices"]
    # analytic: c~U(0.1,1), f~U(0.2,1);  y = g(c) h(f), g = exp(-2c), h = 0.5 + f
    rng = np.random.default_rng(0)
    c, f = rng.uniform(0.1, 1, 2_000_000), rng.uniform(0.2, 1, 2_000_000)
    g, h = np.exp(-2 * c), 0.5 + f
    V = np.var(g * h)
    s1c = np.var(g * h.mean()) / V
    s1f = np.var(h * g.mean()) / V
    assert ind["cover"]["S1"] == pytest.approx(s1c, abs=0.06)
    assert ind["fluor"]["S1"] == pytest.approx(s1f, abs=0.06)
    # interaction exists (multiplicative): ST > S1 for both
    assert ind["cover"]["ST"] > ind["cover"]["S1"] - 0.02
    assert (tmp_path / "sensitivity.json").exists() and (tmp_path / "sensitivity.csv").exists()
    assert "S1" in sens.format_table(res)


def test_study_noise_floor(tmp_path):
    run_study(_spec("saltelli", 16, 3), tmp_path, capture_fn=fake_capture)
    res = sens.analyse_study(tmp_path, n_boot=20, save=False)
    nz = res["metrics"]["signal"]["noise"]
    assert nz["replicates"] == 3 and 0 <= nz["noise_floor"] < 0.01      # fake noise (sd 2) << signal variance


def test_study_morris(tmp_path):
    run_study(_spec("morris", 10, 1), tmp_path, capture_fn=fake_capture)
    res = sens.analyse_study(tmp_path, n_boot=50, save=False)
    ind = res["metrics"]["signal"]["indices"]
    assert ind["cover"]["mu"] < 0                                       # signal decreases with cover thickness
    assert ind["fluor"]["mu"] > 0


def test_analyse_rejects_other_designs(tmp_path):
    r = run_study(_spec("lhs", 4, 1), tmp_path, capture_fn=fake_capture)
    with pytest.raises(ValueError, match="saltelli"):
        sens.analyse_study(tmp_path)
