"""Tests for the train/select PipelineState fields and the headless worker job."""
from mehsi_preprocessor.state import PipelineState, STEP_TRAIN, STEP_SELECT


def test_state_has_train_select_fields_and_invalidation():
    s = PipelineState()
    assert s.analyzer is None
    assert s.training_losses == []
    assert s.selection_result is None

    # Leaving the Train step invalidates the (downstream) selection, keeps the model.
    s.analyzer = object()
    s.selection_result = object()
    s.invalidate_from(STEP_TRAIN)
    assert s.selection_result is None
    assert s.analyzer is not None

    # Re-doing an earlier step (ROI = 7) invalidates both model and selection.
    s.selection_result = object()
    s.invalidate_from(7)
    assert s.analyzer is None
    assert s.selection_result is None


def test_step_constants():
    assert STEP_TRAIN == 9
    assert STEP_SELECT == 10


# --- headless worker-logic test -------------------------------------------------
import torch
from spectral_select import Analyzer, Config
from mehsi_preprocessor.workers import run_selection_job

_N_BANDS = 4
_EX = [365.0, 405.0]


class _FakeModel:
    def __init__(self, numel, seed=0):
        self.excitation_wavelengths = list(_EX)
        self.emission_bands = {e: _N_BANDS for e in _EX}
        g = torch.Generator().manual_seed(seed)
        self._W = {e: torch.randn(numel, 2 * 2 * _N_BANDS, generator=g) for e in _EX}

    def decode(self, latent):
        b = latent.shape[0]
        flat = latent.reshape(b, -1)
        return {e: (flat @ self._W[e]).reshape(b, 2, 2, _N_BANDS) for e in _EX}


class _FakeDataset:
    def __init__(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        self._d = {e: torch.rand(5, 5, _N_BANDS, generator=g) for e in _EX}
        self.emission_wavelengths = {e: [e + 50 + 10 * i for i in range(_N_BANDS)] for e in _EX}

    def get_all_data(self):
        return self._d


def _prepared():
    cfg = Config(sample_name="w", n_important_dimensions=4, n_bands_to_select=3,
                 perturbation_method="standard_deviation", use_diversity_constraint=False,
                 device="cpu")
    a = Analyzer(cfg)
    torch.manual_seed(1)
    a._model = _FakeModel(24)
    a._dataset = _FakeDataset()
    a._baseline_latent = torch.randn(2, 3, 2, 2, 2)
    a._baseline_reconstruction = a._model.decode(a._baseline_latent)
    a._is_prepared = True
    return a


def test_run_selection_job_returns_result():
    a = _prepared()
    result = run_selection_job(a, a.config)
    assert len(result.selected_bands) == 3
