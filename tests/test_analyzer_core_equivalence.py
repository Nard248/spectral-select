"""Equivalence guard for the Phase 7 engine unification.

Proves that spectral_select.Analyzer's perturbation math is byte-identical to
selection_core, using an injected deterministic baseline state + a fake HSI model
(so no stochastic training is involved). Run BEFORE the Analyzer is refactored to
delegate, this proves the extraction is faithful; it must keep passing after.
"""
import numpy as np
import torch

from spectral_select import Analyzer, Config
import selection_core


N_BANDS = 4
EXCITATIONS = [365.0, 405.0]


class FakeHSIModel:
    """Minimal HSI model: deterministic linear decode so perturbations move the output."""

    def __init__(self, latent_numel: int, seed: int = 0):
        self.excitation_wavelengths = list(EXCITATIONS)
        self.emission_bands = {ex: N_BANDS for ex in EXCITATIONS}
        g = torch.Generator().manual_seed(seed)
        self._W = {
            ex: torch.randn(latent_numel, 2 * 2 * N_BANDS, generator=g, dtype=torch.float32)
            for ex in EXCITATIONS
        }

    @property
    def groups(self):
        return self.excitation_wavelengths

    @property
    def channels_per_group(self):
        return self.emission_bands

    def decode(self, latent: torch.Tensor):
        b = latent.shape[0]
        flat = latent.reshape(b, -1)
        return {ex: (flat @ self._W[ex]).reshape(b, 2, 2, N_BANDS) for ex in EXCITATIONS}


class FakeDataset:
    def __init__(self, seed: int = 1):
        g = torch.Generator().manual_seed(seed)
        self._data = {ex: torch.rand(5, 5, N_BANDS, generator=g) for ex in EXCITATIONS}
        self.emission_wavelengths = {ex: [ex + 50 + 10 * i for i in range(N_BANDS)] for ex in EXCITATIONS}

    def get_all_data(self):
        return self._data


def _make_fitted_analyzer(method="variance", n_dims=5,
                          perturbation_method="standard_deviation"):
    config = Config(
        sample_name="equiv",
        dimension_selection_method=method,
        n_important_dimensions=n_dims,
        perturbation_method=perturbation_method,
        perturbation_magnitudes=[10, 20],
        perturbation_directions=["bidirectional"],
        normalization_method="variance",
        device="cpu",
    )
    a = Analyzer(config)
    torch.manual_seed(7)
    latent = torch.randn(2, 3, 2, 2, 2)  # (batch, n_channels, n_latent, h, w)
    model = FakeHSIModel(latent_numel=3 * 2 * 2 * 2)
    a._model = model
    a._dataset = FakeDataset()
    a._baseline_latent = latent
    with torch.no_grad():
        a._baseline_reconstruction = model.decode(latent)
    return a, latent, model


def test_select_important_dimensions_matches_core():
    for method in ("variance", "activation"):
        a, latent, _ = _make_fitted_analyzer(method=method)
        a._select_important_dimensions()
        core = selection_core.select_important_dimensions(latent, method, a.config.n_important_dimensions)
        assert [c for _, c in a._important_dims] == [c for _, c in core]
        assert np.allclose([s for s, _ in a._important_dims], [s for s, _ in core], rtol=0, atol=0)


def test_compute_influence_matches_core_accumulate():
    a, latent, model = _make_fitted_analyzer(perturbation_method="standard_deviation")
    a._select_important_dimensions()
    a._compute_influence_scores()

    core_infl = selection_core.accumulate_influence(
        model.decode, model.excitation_wavelengths, model.emission_bands,
        latent, a._baseline_reconstruction, a._important_dims,
        magnitudes=a.config.perturbation_magnitudes,
        directions=a.config.perturbation_directions,
        perturbation_method=a.config.perturbation_method,
    )
    for ex in EXCITATIONS:
        assert np.array_equal(a._influence_matrix[ex], core_infl[ex])


def test_compute_influence_matches_core_percentile():
    a, latent, model = _make_fitted_analyzer(perturbation_method="percentile")
    a._select_important_dimensions()
    a._compute_influence_scores()
    core_infl = selection_core.accumulate_influence(
        model.decode, model.excitation_wavelengths, model.emission_bands,
        latent, a._baseline_reconstruction, a._important_dims,
        magnitudes=a.config.perturbation_magnitudes,
        directions=a.config.perturbation_directions,
        perturbation_method=a.config.perturbation_method,
    )
    for ex in EXCITATIONS:
        assert np.array_equal(a._influence_matrix[ex], core_infl[ex])


def test_normalize_influences_matches_core_float32_path():
    a, latent, model = _make_fitted_analyzer()
    a._select_important_dimensions()
    a._compute_influence_scores()
    raw = {ex: a._influence_matrix[ex].copy() for ex in EXCITATIONS}

    a._normalize_influences()  # analyzer mutates in place (variance, float32)

    core_norm = selection_core.normalize_influence(
        raw, a._dataset.get_all_data(), "variance", variance_float64=False
    )
    for ex in EXCITATIONS:
        assert np.array_equal(a._influence_matrix[ex], core_norm[ex])
