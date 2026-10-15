"""Fisher information, CRLB and optimal design (v1.2 W4): analytic checks, exhaustive-search check, FD vs analytic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fakes_inverse as fk  # noqa: E402
from spectraforge.studio import information as info  # noqa: E402
from spectraforge.studio.surrogate import (ChannelTable, NoiseModel, ParamSpec, Surrogate,  # noqa: E402
                                            build_surrogate)

SP = ParamSpec("cover", (fk.COVER_T, fk.COVER_T_END))


def _exp(t=0.8):
    from spectraforge.studio.schema import Experiment
    from spectraforge.studio.sweeps import set_pointer
    d = fk.base_experiment().model_dump(mode="json")
    for p, v in ((fk.COVER_KIND, "uniform"), (fk.COVER_T, t), (fk.COVER_T_END, t)):
        set_pointer(d, p, v)
    return Experiment.model_validate(d)


def test_linear_gaussian_fisher_is_exact():
    """mu = X theta with constant Gaussian noise: F = X^T X / var exactly (Kay 1993, eq. 3.31), CRLB = F^-1."""
    rng = np.random.default_rng(0)
    C, P = 30, 2
    X = rng.normal(size=(C, P)) * 5 + 40.0
    nm = NoiseModel(1.0, 3.0, np.zeros(C), n_pix=1.0, shot=False, read=True, quantization=False)   # var = 9
    g1, g2 = np.linspace(0, 1, 5), np.linspace(0, 2, 6)
    pts = np.array([(a, b) for a in g1 for b in g2])
    sur = Surrogate([ParamSpec("a", ("/x",)), ParamSpec("b", ("/y",))], pts, pts @ X.T,
                    ChannelTable(np.full(C, 400.0), np.arange(C) * 10.0 + 450), nm, grid_axes=[g1, g2],
                    interp="multilinear")
    fi = info.fisher_from_surrogate(sur, [0.4, 0.9])
    assert np.allclose(fi.fisher, X.T @ X / 9.0, rtol=1e-6)
    assert np.allclose(fi.covariance, np.linalg.inv(X.T @ X / 9.0), rtol=1e-6)
    assert fi.identifiable


def test_poisson_limit_matches_analytic_information():
    """No read noise, no dark: var = mu/n and F = n * mu'^2 / mu (Poisson information)."""
    A, k = 500.0, 0.9
    g = np.linspace(0.1, 2.0, 12)
    nm = NoiseModel(1.0, 0.0, np.zeros(1), n_pix=25.0, read=False, quantization=False)
    sur = Surrogate([SP], g[:, None], (A * np.exp(-k * g))[:, None], ChannelTable(np.array([400.0]), np.array([500.0])),
                    nm, grid_axes=[g])
    th = 0.7
    fi = info.fisher_from_surrogate(sur, [th])
    mu = A * np.exp(-k * th)
    assert fi.fisher[0, 0] == pytest.approx(25.0 * (k * mu) ** 2 / mu, rel=2e-4)


def test_fd_simulator_fisher_matches_surrogate_fisher():
    exp = _exp(0.8)
    fd = info.fisher(exp, [SP], capture_fn=fk.fake_capture, step=0.02)
    sur = build_surrogate(exp, [SP], {"cover": np.linspace(0.2, 1.5, 14)}, capture_fn=fk.fake_capture, cache=False)
    fs = info.fisher_from_surrogate(sur, [0.8])
    assert fd.sd[0] == pytest.approx(fs.sd[0], rel=2e-3)
    assert fd.channels.n == 16 and fd.meta["n_runs"] == 3


def test_band_and_excitation_selection():
    exp = _exp(0.8)
    full = info.fisher(exp, [SP], capture_fn=fk.fake_capture, step=0.02)
    one = info.fisher(exp, [SP], excitations=[430.0], capture_fn=fk.fake_capture, step=0.02)
    assert one.channels.n == 8 and np.all(one.channels.ex_nm == 430.0)
    assert one.sd[0] > full.sd[0]                        # fewer channels -> weaker bound
    span = info.fisher(exp, [SP], bands=(450.0, 480.0), capture_fn=fk.fake_capture, step=0.02)
    assert np.all((span.channels.em_nm >= 450) & (span.channels.em_nm <= 480))


def _two_param_fisher(C=14, seed=1):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(C, 2))
    J[:, 1] += 0.6 * J[:, 0]
    mean = rng.uniform(100, 1000, C)
    nm = NoiseModel(1.5, 8.0, np.full(C, 1.0), n_pix=10.0)
    return info.fisher_from_jacobian(["a", "b"], [1.0, 2.0], J * 40, mean, nm,
                                     ChannelTable(np.zeros(C), np.arange(C, dtype=float)),
                                     exposure_s=np.full(C, 0.1))


@pytest.mark.parametrize("criterion", ["D", "A", "target:a", "target:b"])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_greedy_matches_exhaustive_on_small_case(criterion, k):
    fi = _two_param_fisher()
    d = info.optimal_design(fi, k, criterion)
    ex = d.exhaustive
    assert ex is not None and ex["n_combinations"] > 0
    # D-optimal plain greedy equals the exhaustive optimum on this case; with the exchange refinement the returned
    # design reaches the exhaustive optimum for the A and target criteria as well
    if criterion == "D":
        assert ex["greedy_same_set"] and abs(ex["greedy_gap"]) < 1e-9
    assert ex["same_set"] and abs(ex["refined_gap"]) < 1e-9


def test_crlb_decreases_with_k_and_beats_random():
    fi = _two_param_fisher(C=40, seed=3)
    d = info.optimal_design(fi, 8, "D", exhaustive_check=False)
    assert d.crlb_sd.shape == (8, 2)
    assert np.all(np.diff(d.crlb_sd[1:, 0]) <= 1e-12) and np.all(np.diff(d.crlb_sd[1:, 1]) <= 1e-12)
    rng = np.random.default_rng(0)
    rand = [np.prod(info.crlb_for_channels(fi, rng.choice(40, 8, replace=False))) for _ in range(100)]
    assert np.prod(d.crlb_sd[-1]) <= min(rand) * (1 + 1e-9)
    # all-channel bound is the floor
    assert np.all(fi.sd <= d.crlb_sd[-1] + 1e-12)


def test_unidentifiable_parameters_are_flagged():
    C = 10
    j = np.random.default_rng(0).normal(size=(C, 1))
    J = np.hstack([j, 2 * j])                           # collinear columns
    nm = NoiseModel(1.0, 5.0, np.zeros(C))
    fi = info.fisher_from_jacobian(["a", "b"], [0, 0], J, np.full(C, 100.0), nm)
    assert not fi.identifiable and np.all(np.isinf(fi.sd))


def test_exposure_allocation_beats_reference_and_equal():
    fi = _two_param_fisher(C=12, seed=5)
    d = info.optimal_design(fi, 5, "D", budget="exposure_time", exhaustive_check=False)
    ex = d.exposure
    assert sum(ex["t_opt_s"]) == pytest.approx(ex["total_time_s"], rel=1e-6)
    assert ex["objective_opt"] >= ex["objective_reference"] - 1e-9
    assert ex["objective_opt"] >= ex["objective_equal"] - 1e-9
    assert all(t >= 0 for t in ex["t_opt_s"])


def test_exposure_shot_limited_single_parameter_concentrates_on_best_channel():
    """Single parameter in the shot-limited regime: information per unit time is linear in t, so the whole budget goes
    to the channel with the highest rate (Pukelsheim ch. 7)."""
    C = 6
    rng = np.random.default_rng(2)
    J = rng.uniform(1, 5, size=(C, 1))
    mean = rng.uniform(1e3, 5e3, C)
    nm = NoiseModel(1.0, 0.0, np.zeros(C), n_pix=1.0, read=False, quantization=False, full_well_e=1e12)
    fi = info.fisher_from_jacobian(["a"], [1.0], J, mean, nm, exposure_s=np.full(C, 0.1))
    ex = info.allocate_exposure(fi, list(range(C)), "D", total_time_s=1.0)
    best = int(np.argmax(J[:, 0] ** 2 / mean))
    assert np.argmax(ex["t_opt_s"]) == best and ex["t_opt_s"][best] > 0.95


def test_zero_noise_free_channel_selection_prefers_informative_band():
    """A band that does not depend on theta carries no information and is never selected first."""
    C = 5
    J = np.array([[0.0], [0.0], [3.0], [0.0], [1.0]])
    nm = NoiseModel(1.0, 4.0, np.zeros(C))
    fi = info.fisher_from_jacobian(["a"], [1.0], J, np.full(C, 200.0), nm)
    d = info.optimal_design(fi, 2, "D")
    assert d.selected[0] == 2 and set(d.selected) == {2, 4}


def test_compare_with_channel_selection_runs_on_simulated_dataset():
    pytest.importorskip("torch")
    exp = _exp(0.8)
    fi = info.fisher(exp, [ParamSpec("cover", (fk.COVER_T, fk.COVER_T_END))], capture_fn=fk.fake_capture, step=0.02)
    sd = info.spectradata_from_capture(fk.fake_capture(exp, noisy=True))
    out = info.compare_with_channel_selection(fi, sd, k=4, epochs=15)
    u, o = out["unsupervised"]["crlb_sd"][0], out["information_optimal"]["crlb_sd"][0]
    assert o <= u * (1 + 1e-9) and o >= out["all_channels"]["crlb_sd"][0] * (1 - 1e-9)


def test_model_error_adds_relative_variance_and_caps_precision():
    """var_c += (eps S_c)^2: analytic Fisher, default eps=0 unchanged, CRLB floor ~ eps/|dlnS/dtheta| for many channels."""
    A, k = 500.0, 0.9
    g = np.linspace(0.1, 2.0, 12)
    nm = NoiseModel(1.0, 0.0, np.zeros(1), n_pix=25.0, read=False, quantization=False)
    sur = Surrogate([SP], g[:, None], (A * np.exp(-k * g))[:, None], ChannelTable(np.array([400.0]), np.array([500.0])),
                    nm, grid_axes=[g])
    th = 0.7
    mu = A * np.exp(-k * th)
    f0 = info.fisher_from_surrogate(sur, [th])
    assert f0.noise.model_error_rel == 0.0
    eps = 0.02
    fe = info.fisher_from_surrogate(sur, [th], model_error_rel=eps)
    mu = float(fe.mean_e[0])
    var = mu / 25.0 + (eps * mu) ** 2
    assert fe.var_e[0] == pytest.approx(var, rel=1e-12)
    assert fe.fisher[0, 0] == pytest.approx((k * mu) ** 2 / var, rel=2e-4)
    # large signal: photon term vanishes, CRLB sd -> eps / k (relative-sensitivity floor)
    assert fe.sd[0] == pytest.approx(np.sqrt(var) / abs(fe.jac_e[0, 0]), rel=1e-9)
    assert fe.sd[0] > f0.sd[0]
    assert NoiseModel.from_json(fe.noise.to_json()).model_error_rel == eps
    assert nm.with_model_error(eps).subset(np.array([True])).model_error_rel == eps
    from dataclasses import replace
    fx = replace(fe, exposure_s=np.array([0.5]))
    i1 = info.information_at_exposure(fx, [0], np.array([0.5]))[0, 0, 0]       # reference exposure
    assert i1 == pytest.approx(fe.fisher[0, 0], rel=1e-9)
    i2 = info.information_at_exposure(fx, [0], np.array([50.0]))[0, 0, 0]      # 100x exposure: photon term vanishes
    assert i2 == pytest.approx((100 * fe.jac_e[0, 0]) ** 2 / (eps * 100 * mu) ** 2, rel=0.03)
