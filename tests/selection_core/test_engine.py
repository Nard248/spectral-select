"""Unit tests for the shared selection_core engine.

These pin the domain-agnostic perturbation -> influence -> normalize primitives
that both spectral_select.Analyzer (HSI) and channel_select.engine delegate to.
"""
import numpy as np
import torch

from selection_core import (
    select_important_dimensions,
    latent_statistics,
    perturbation_amount,
    measure_influence,
    accumulate_influence,
    normalize_influence,
)


def test_select_important_dimensions_ranks_highest_variance_first():
    # latent (batch, 3): coord (2,) varies most, coord (0,) is constant.
    latent = torch.tensor(
        [[1.0, 0.0, -5.0], [1.0, 1.0, 5.0], [1.0, 2.0, 50.0]]
    )
    top = select_important_dimensions(latent, method="variance", n=2)
    assert len(top) == 2
    # highest variance is dim 2, then dim 1; dim 0 (constant) excluded
    assert top[0][1] == (2,)
    assert top[1][1] == (1,)
    assert top[0][0] >= top[1][0]


def test_select_important_dimensions_multidim_coords():
    latent = torch.randn(4, 2, 3)  # coords are 2-tuples
    top = select_important_dimensions(latent, method="activation", n=3)
    assert len(top) == 3
    assert all(len(coord) == 2 for _, coord in top)


def test_perturbation_amount_standard_deviation():
    latent = torch.tensor([[0.0], [2.0], [4.0]])  # one coord, std known
    stats = latent_statistics(latent.reshape(latent.shape[0], -1))
    # method standard_deviation: sign * (magnitude/100) * std
    amt = perturbation_amount(
        coord=(0,), latent_dims=(1,), magnitude=100, sign=1,
        stats=stats, baseline_latent=latent, method="standard_deviation",
    )
    expected = 1.0 * (100 / 100.0) * float(torch.std(latent.reshape(3, -1), dim=0)[0])
    assert abs(amt - expected) < 1e-6


def test_measure_influence_reduces_all_but_last_axis():
    # decode returns a tensor (batch=2, spatial=2, channels=3); influence is per-channel
    def decode_fn(latent):
        return {"g": latent}  # identity for the test
    baseline = {"g": torch.zeros(2, 2, 3)}
    perturbed = torch.ones(2, 2, 3)
    infl = measure_influence(decode_fn, groups=["g"], perturbed_latent=perturbed,
                             baseline_recon=baseline, weight=1.0)
    assert infl["g"].shape == (3,)
    assert np.allclose(infl["g"], 1.0)  # mean|1-0| over batch+spatial = 1


def test_measure_influence_applies_weight():
    def decode_fn(latent):
        return {"g": latent}
    baseline = {"g": torch.zeros(1, 3)}
    infl = measure_influence(decode_fn, ["g"], torch.full((1, 3), 2.0), baseline, weight=0.5)
    assert np.allclose(infl["g"], 1.0)  # 0.5 * mean|2-0| = 1.0


def test_normalize_influence_variance_dtype_param_changes_precision():
    influence = {"g": np.array([1.0, 1.0, 1.0])}
    data = {"g": torch.rand(4, 3)}
    out64 = normalize_influence(influence, data, "variance", variance_float64=True)
    out32 = normalize_influence(influence, data, "variance", variance_float64=False)
    assert out64["g"].shape == (3,)
    # both divide by per-channel variance; values close but the dtype path differs
    assert np.allclose(out64["g"], out32["g"], rtol=1e-3)


def test_normalize_influence_max_per_group():
    influence = {"g": np.array([2.0, 4.0, 1.0])}
    out = normalize_influence(influence, data={}, method="max_per_group")
    assert np.isclose(out["g"].max(), 1.0)
    assert np.allclose(out["g"], np.array([0.5, 1.0, 0.25]))


def test_accumulate_influence_runs_end_to_end():
    torch.manual_seed(0)
    latent = torch.randn(3, 2, 2)  # (batch, d1, d2)

    def decode_fn(lat):
        # deterministic: channel vector per group from latent sums
        return {"g": lat.reshape(lat.shape[0], -1, 1).repeat(1, 1, 3)}
    baseline = decode_fn(latent)
    important = select_important_dimensions(latent, "variance", 2)
    infl = accumulate_influence(
        decode_fn, groups=["g"], channels_per_group={"g": 3},
        latent=latent, baseline_recon=baseline, important_dims=important,
        magnitudes=[20], directions=["bidirectional"], perturbation_method="standard_deviation",
    )
    assert infl["g"].shape == (3,)
    assert np.all(infl["g"] >= 0)


def test_latent_statistics_single_sample_is_finite():
    # Degenerate batch=1 (e.g. a tiny image yields one baseline patch) must not produce
    # NaN std (torch.std with correction=1 divides by n-1=0). Surfaced by SpectraForge.
    latent = torch.randn(1, 5)
    stats = latent_statistics(latent)
    assert torch.isfinite(stats["std"]).all()
    assert torch.allclose(stats["std"], torch.zeros(5))


def test_latent_statistics_multisample_unchanged():
    # batch >= 2 keeps sample std (correction=1) so the Phase-7 equivalence holds.
    latent = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    stats = latent_statistics(latent)
    assert torch.allclose(stats["std"], torch.std(latent, dim=0))
