"""Cheap analytic stand-ins for the simulator used by the inverse-problem tests (no Monte Carlo).

``fake_capture`` mimics ``capture.capture``: ``electrons_mean[ex] = A_c exp(-k_c * thickness(x))`` for the cover
thickness read from the experiment (``/objects/0/shape/layers/2/thickness``), noisy ``adu`` drawn with the REAL
detector model ``sensor.detect`` (so the noise-model tests are tests against ``sensor.py``)."""
from __future__ import annotations

import numpy as np

from spectraforge.studio import sensor as sensor_mod
from spectraforge.studio.contracts import CaptureResult
from spectraforge.studio.presets import preset

COVER_T = "/objects/0/shape/layers/2/thickness/thickness_mm"
COVER_T_END = "/objects/0/shape/layers/2/thickness/end_thickness_mm"
COVER_KIND = "/objects/0/shape/layers/2/thickness/kind"
EXS = (365.0, 430.0)
NB = 8


def base_experiment(photons: float = 1.0):
    """tissue_under_collagen with a short sequence (365, 430 nm) and bright signals."""
    e = preset("tissue_under_collagen")
    d = e.model_dump(mode="json")
    d["sources"][0]["sequence"] = [s for s in d["sources"][0]["sequence"] if s["wavelength_nm"] in EXS]
    d["cameras"][0]["filter"]["n_bands"] = NB
    from spectraforge.studio.schema import Experiment
    return Experiment.model_validate(d)


def _amp(c: int, ex_i: int) -> tuple[float, float]:
    rng = np.random.default_rng(100 + ex_i)
    A = rng.uniform(300.0, 3000.0, NB)
    k = rng.uniform(0.3, 2.5, NB)
    return A[c], k[c]


def thickness_map(exp, width: int, height: int) -> np.ndarray:
    prof = exp.objects[0].shape.layers[2].thickness
    x = (np.arange(width) + 0.5) / width
    t = prof.thickness_mm + (prof.end_thickness_mm - prof.thickness_mm) * x if prof.kind == "linear" \
        else np.full(width, prof.thickness_mm)
    return np.broadcast_to(t[None, :], (height, width)).copy()


def fake_capture(exp, *, scale: float = 0.25, excitations=None, noisy: bool = False, seed: int = 0, mode=None,
                 **_ignored) -> CaptureResult:
    cam, tg = exp.camera(), exp.acquisition.physics
    w, h = max(8, int(round(cam.sensor.width_px * scale))), max(8, int(round(cam.sensor.height_px * scale)))
    th = thickness_map(exp, w, h)
    seq = {s.wavelength_nm: s for s in exp.source().sequence if s.enabled}
    exs = [e for e in EXS if e in seq and (excitations is None or any(abs(e - x) < 1e-6 for x in excitations))]
    res = CaptureResult(experiment_json=exp.model_dump_json(), mode="sample", width=w, height=h, excitations_nm=[],
                        emission_nm={}, adu={}, radiance={}, electrons_mean={}, exposure_ms={}, power_mW={},
                        ground_truth={})
    rng = np.random.default_rng(seed)
    for i, ex in enumerate(exs):
        e_cube = np.empty((h, w, NB), np.float32)
        for c in range(NB):
            A, k = _amp(c, i)
            e_cube[:, :, c] = A * np.exp(-k * th) * seq[ex].exposure_ms / 100.0
        adu, _ = sensor_mod.detect(e_cube, seq[ex].exposure_ms, cam.sensor, tg, None, rng if noisy else
                                   np.random.default_rng(1))
        res.excitations_nm.append(ex)
        res.emission_nm[ex] = 450.0 + 10.0 * np.arange(NB) + 5.0 * i
        res.electrons_mean[ex] = e_cube
        res.adu[ex] = adu
        res.radiance[ex] = e_cube / 7.0
        res.exposure_ms[ex] = seq[ex].exposure_ms
        res.power_mW[ex] = seq[ex].power_mW
    res.ground_truth = {"layer_thickness:collagen": th.astype(np.float32), "object": np.zeros((h, w), np.int32)}
    return res
