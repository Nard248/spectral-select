"""Cheap deterministic stand-in for ``api.capture`` used by the study / dataset / route tests.

The fake signal depends analytically on two layered-wedge parameters so studies have a known response:
``y = 1000 * exp(-2 * cover) * (0.5 + fluor) + noise(seed)`` where ``cover`` = start thickness of layer 2 and
``fluor`` = thickness of layer 1 (``layered_wedge_phantom`` pointers)."""
from __future__ import annotations

import numpy as np

from spectraforge.studio.contracts import CaptureResult
from spectraforge.studio.presets import preset
from spectraforge.studio.provenance import attach

H, W, B = 16, 20, 5
COVER = "/objects/0/shape/layers/2/thickness/thickness_mm"
FLUOR = "/objects/0/shape/layers/1/thickness/thickness_mm"


def base_experiment():
    return preset("layered_wedge_phantom")


def fake_capture(exp, mode="preview", excitations=None) -> CaptureResult:
    shape = exp.objects[0].shape
    cover = shape.layers[2].thickness.thickness_mm
    fluor = shape.layers[1].thickness.thickness_mm
    seed = int(exp.acquisition.seed)
    rng = np.random.default_rng(seed)
    ex = float(excitations[0]) if excitations else 340.0
    yy, xx = np.mgrid[0:H, 0:W]
    inclusion = (xx < 5) & (yy < 5)                                  # bright corner patch
    y = 1000.0 * np.exp(-2.0 * cover) * (0.5 + fluor)
    em = 420.0 + 10.0 * np.arange(B)
    shapeb = np.exp(-0.5 * ((em - 460.0) / 25.0) ** 2)
    cube = (100 + y * shapeb[None, None, :] * np.where(inclusion, 3.0, 1.0)[..., None]
            + rng.normal(0, 2.0, (H, W, B))).astype(np.float32)
    names = [layer.name for layer in shape.layers]
    gt = {"material": np.ones((H, W), np.float32), "object": np.zeros((H, W), np.float32),
          "layer_thickness:cover": np.full((H, W), cover, np.float32),
          "layer_thickness:substrate": np.full((H, W), 4.0, np.float32),
          "layer_thickness:fluorescent layer": np.full((H, W), fluor, np.float32)}
    frac = {n: np.zeros((H, W), np.float32) for n in names}
    frac[names[1]][:] = 0.7
    frac[names[0]][:] = 0.3
    frac[names[1]][inclusion] = 1.0
    frac[names[0]][inclusion] = 0.0
    for n in names:
        gt[f"origin_layer_fraction:{n}"] = frac[n]
    res = CaptureResult(
        experiment_json=exp.model_dump_json(), mode="sample", width=W, height=H, excitations_nm=[ex],
        emission_nm={ex: em}, adu={ex: cube}, radiance={ex: (cube - 100) * 1e-3}, electrons_mean={ex: cube * 1.5},
        exposure_ms={ex: 100.0}, power_mW={ex: 2.0}, ground_truth=gt, timings_s={"total": 0.01})
    attach(res, exp, requested_mode=mode, seed=seed)
    return res
