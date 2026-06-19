"""Pure render/export helpers shared by the worker and tests (no Qt)."""
from __future__ import annotations

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


def export_dataset(spectra, ground_truth, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spectra.to_pickle(out / "spectra_unmasked.pkl")
    ground_truth.save(out)
