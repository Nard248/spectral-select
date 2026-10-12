"""Fake v1.1 CaptureResults for the analysis / sweep / server tests (block engine not required)."""
from __future__ import annotations

import numpy as np

from spectraforge.studio.contracts import CaptureResult, VoxelGrid
from spectraforge.studio.presets import preset

H, W, B = 20, 30, 6
OFFSET = 100.0   # generic_macro_bench camera offset_adu


def fake_block_result(mode: str = "sample", scale_adu: float = 1.0, exposure_ms: float = 50.0,
                      power_mW: float = 2.0) -> CaptureResult:
    """adu[365] = offset + s*(col+1)*(band+1)*10 ; adu[405] = offset + s*(row+1)*(band+1)*10 (bands differ per ex).
    Ground truth: cover thickness = 2 mm * col/(W-1) (NaN for cols < 5, i.e. 'not on the block')."""
    exp = preset("generic_macro_bench")
    rr, cc = np.mgrid[0:H, 0:W]
    bi = np.arange(B)
    exs = [365.0, 405.0]
    em = {365.0: 420.0 + 10.0 * bi, 405.0: 440.0 + 10.0 * bi}
    adu = {365.0: (OFFSET + scale_adu * (cc[..., None] + 1) * (bi + 1) * 10.0).astype(np.float32),
           405.0: (OFFSET + scale_adu * (rr[..., None] + 1) * (bi + 1) * 10.0).astype(np.float32)}
    rad = {e: ((a - OFFSET) * 1e-3).astype(np.float32) for e, a in adu.items()}
    el = {e: ((a - OFFSET) * 2.0).astype(np.float32) for e, a in adu.items()}
    thick = np.where(cc >= 5, 2.0 * cc / (W - 1), np.nan)
    gt = {"layer_thickness:cover": thick, "depth_mm": np.full((H, W), 300.0),
          "material": (cc > W // 2).astype(np.int32), "origin_depth_mm": np.where(cc >= 5, 0.4, np.nan),
          "label": np.full((H, W), "x", dtype="U1")}
    grid = VoxelGrid(nx=4, nz=3, n_layers=3, sublayers=2, x_edges_mm=np.linspace(-2, 2, 5),
                     z_edges_mm=np.linspace(-1.5, 1.5, 4))
    rng = np.random.default_rng(0)
    vol = {"block1": {"grid": grid, "layer_names": ["substrate", "tissue", "cover"],
                      "fluence": {e: rng.random((4, 3, 3, 2)).astype(np.float32) for e in exs},
                      "absorbed": {e: rng.random((4, 3, 3, 2)).astype(np.float32) for e in exs},
                      "source": {(365.0, "tissue|NADH"): rng.random((4, 3, 3, 2)).astype(np.float32),
                                 (365.0, "tissue|FAD"): rng.random((4, 3, 3, 2)).astype(np.float32)},
                      "units": {"fluence": "mW/mm^2", "absorbed": "mW", "source": "mW/mm^3"}}}
    blocks = [{"object_id": "block1", "layer_names": ["substrate", "tissue", "cover"], "footprint_mm": (4.0, 3.0),
               "world_from_local": np.eye(4).tolist()}]
    return CaptureResult(
        experiment_json=exp.model_dump_json(), mode=mode, width=W, height=H, excitations_nm=exs, emission_nm=em,
        adu=adu, radiance=rad, electrons_mean=el, exposure_ms={e: exposure_ms for e in exs},
        power_mW={e: power_mW for e in exs}, ground_truth=gt, warnings=["w"], timings_s={"total": 0.5},
        blocks=blocks, volumes=vol)
