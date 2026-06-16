"""Characterization guard for the prepare()/select() split.

Proves Analyzer.select() reproduces the legacy inline fit() math sequence (so the
split changes nothing numerically) and that select() is re-runnable without re-prepare.
Uses an injected deterministic baseline + a fake HSI model (no training).
"""
import torch

from spectral_select import Analyzer, Config

N_BANDS = 4
EXCITATIONS = [365.0, 405.0]


class FakeHSIModel:
    def __init__(self, latent_numel: int, seed: int = 0):
        self.excitation_wavelengths = list(EXCITATIONS)
        self.emission_bands = {ex: N_BANDS for ex in EXCITATIONS}
        g = torch.Generator().manual_seed(seed)
        self._W = {
            ex: torch.randn(latent_numel, 2 * 2 * N_BANDS, generator=g, dtype=torch.float32)
            for ex in EXCITATIONS
        }

    def decode(self, latent: torch.Tensor):
        b = latent.shape[0]
        flat = latent.reshape(b, -1)
        return {ex: (flat @ self._W[ex]).reshape(b, 2, 2, N_BANDS) for ex in EXCITATIONS}


class FakeDataset:
    def __init__(self, seed: int = 1):
        g = torch.Generator().manual_seed(seed)
        self._data = {ex: torch.rand(5, 5, N_BANDS, generator=g) for ex in EXCITATIONS}
        self.emission_wavelengths = {
            ex: [ex + 50 + 10 * i for i in range(N_BANDS)] for ex in EXCITATIONS
        }

    def get_all_data(self):
        return self._data


def _config():
    return Config(
        sample_name="ps", dimension_selection_method="variance",
        n_important_dimensions=5, perturbation_method="standard_deviation",
        perturbation_magnitudes=[10, 20], perturbation_directions=["bidirectional"],
        normalization_method="variance", n_bands_to_select=4,
        use_diversity_constraint=False, device="cpu",
    )


def _injected(latent, model):
    a = Analyzer(_config())
    a._model = model
    a._dataset = FakeDataset()
    a._baseline_latent = latent
    with torch.no_grad():
        a._baseline_reconstruction = model.decode(latent)
    a._is_prepared = True
    return a


def _bands_key(bands):
    return [(b.excitation_nm, b.emission_band_index, b.influence_score) for b in bands]


def test_select_matches_legacy_private_sequence():
    torch.manual_seed(7)
    latent = torch.randn(2, 3, 2, 2, 2)
    model = FakeHSIModel(latent_numel=3 * 2 * 2 * 2)

    a = _injected(latent, model)
    a._select_important_dimensions()
    a._compute_influence_scores()
    a._normalize_influences()
    legacy = a._select_top_bands()

    b = _injected(latent, model)
    result = b.select(b.config)

    assert _bands_key(result.selected_bands) == _bands_key(legacy)


def test_select_is_rerunnable_without_reprepare():
    torch.manual_seed(7)
    latent = torch.randn(2, 3, 2, 2, 2)
    model = FakeHSIModel(latent_numel=3 * 2 * 2 * 2)
    a = _injected(latent, model)

    r1 = a.select(a.config)
    a.config.n_bands_to_select = 2
    r2 = a.select(a.config)

    assert len(r1.selected_bands) == 4
    assert len(r2.selected_bands) == 2


def test_select_before_prepare_raises():
    a = Analyzer(_config())
    try:
        a.select()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "prepare" in str(exc).lower()
