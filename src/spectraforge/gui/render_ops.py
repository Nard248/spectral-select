"""Pure render/export helpers shared by the worker and tests (no Qt)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from spectraforge.forward import render
from spectraforge.gui.layer import build_scene


def render_state(state):
    """Build the scene from ``state`` and render -> (SpectraData, GroundTruth)."""
    scene = build_scene(state)
    return render(
        scene, state.library, state.acquisition,
        artifacts=state.artifacts, seed=state.seed, sample_name="forge",
    )


def _default_validate_config():
    from spectral_select import Config
    return Config(
        sample_name="forge_validate", n_important_dimensions=15, n_bands_to_select=12,
        perturbation_method="percentile", use_diversity_constraint=True,
        training_epochs=30, device="cpu", random_seed=0,
        output_dir=Path(tempfile.mkdtemp()),
    )


def validate_state(state, config=None, tol_nm: float = 12.0):
    """Render the current scene, select bands with the Analyzer, and score vs ground truth.

    Renders fresh into a LOCAL (never assigns ``state.last_render``) so this is safe to run on a
    worker thread without racing the GUI thread, and always reflects the current scene rather than a
    stale render. Returns the ``validate_selection`` metrics dict (incl. the tight ``peak_recovery``
    and ``mask_coverage`` — read these next to a chance baseline; the broad-mask metrics saturate).
    """
    from spectral_select import Analyzer
    from spectraforge.validation import validate_selection

    spectra, ground_truth = render_state(state)
    analyzer = Analyzer(config or _default_validate_config())
    analyzer.fit(spectra)
    return validate_selection(ground_truth, analyzer.get_wavelengths(), tol_nm=tol_nm)


def export_dataset(spectra, ground_truth, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spectra.to_pickle(out / "spectra_unmasked.pkl")
    ground_truth.save(out)
