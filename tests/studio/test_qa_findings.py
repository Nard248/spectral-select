"""Findings of the physics QA pass for modules owned by other engineers (tracer / geometry).

Minimal reproductions of QA findings; the remaining xfail is an optional tracer enhancement.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge.studio import presets
from spectraforge.studio.tracer import trace_scene


def test_third_cuvette_in_a_beam_row_is_lit():
    """laser_line: the beam crosses cuvette 1, 2 and then lights cuvette 3.  A shadow ray from a sample in
    cuvette 3 crosses 10 interfaces; the tracer gives up after 8 and returns zero fluence."""
    exp = presets.preset("laser_line")
    hit = trace_scene(exp, 87, 64, toggles=exp.acquisition.physics)
    fluence = {}
    for oi in range(3):
        sel = np.nonzero(hit.object_index[0] == oi)[0]
        fluence[oi] = float((hit.vol_weight_mm[:, sel] * hit.vol_fluence_per_mW[:, sel]).sum(axis=2).max())
    assert fluence[0] > 0.1 and fluence[1] > 0.1
    assert fluence[2] > 0.1, f"third cuvette fluence {fluence[2]:.2e} (first two: {fluence[0]:.2f}, {fluence[1]:.2f})"


@pytest.mark.xfail(reason="vol_fluence_per_mW omits Beer-Lambert absorption by clear media crossed before the "
                          "sample's own medium; capture.py compensates with a ray march of its own "
                          "(_prior_occluder_paths).  A tracer-provided per-material path would be cheaper.",
                   strict=False)
def test_volume_samples_expose_prior_clear_path():
    exp = presets.preset("laser_line")
    hit = trace_scene(exp, 43, 32, toggles=exp.acquisition.physics)
    assert getattr(hit, "vol_prior_path_mm", None) is not None
