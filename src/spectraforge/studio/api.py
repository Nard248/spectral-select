"""Headless Python API of SpectralForge Studio: a notebook-first facade over the engine.

Everything the web UI does is available here without a server.  The functions are thin wrappers over
:mod:`.capture`, :mod:`.analysis`, :mod:`.exports` and :mod:`.sweeps` (nothing is duplicated), plus
provenance, so a script is reproducible and citable.

Complete example::

    from spectraforge.studio import api

    exp = api.preset("layered_wedge_phantom")                     # or api.load("exp.json")
    exp = api.set(exp, "/objects/0/shape/layers/2/thickness/thickness_mm", 0.3)   # RFC 6901, validated copy
    res = api.capture(exp, mode="preview", seed=0)                # CaptureResult (+ res.meta['provenance'])

    roi = {"shape": "rect", "x0": 20, "y0": 20, "x1": 60, "y1": 40}
    e = api.eem(res, roi=roi, norm={"mode": "exposure_power"})     # EemResult: grid, excitations, values
    r = api.response(res, "layer_thickness:cover",
                     {"kind": "band_sum", "ex": res.excitations_nm[0]}, roi=roi)   # signal vs. thickness
    api.save(res, "out.npz")                                      # npz incl. provenance
    sd = api.to_spectradata(res)                                  # spectral_select.types.SpectraData
    print(api.methods_text(exp))                                  # publication-ready Methods paragraph

Capture modes: ``"preview"`` (first enabled excitation, ``acquisition.preview_scale`` resolution, fast),
``"full"`` (all enabled excitations, native resolution, ``acquisition.mode`` sample) and the engine modes
``"sample" | "dark" | "white_reference"`` (full resolution).  Everything is deterministic given the seed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from . import analysis, exports
from .provenance import attach, experiment_hash, provenance  # noqa: F401  (re-exported)
from .schema import Experiment
from .sweeps import SweepError, apply_value, set_pointer

__all__ = ["preset", "presets", "load", "dump", "set", "set_many", "get", "capture", "eem", "roi_stats",
           "response", "line_profile", "save", "load_capture", "to_spectradata", "methods_text",
           "experiment_hash", "provenance", "validate", "SweepError"]

_pyset = set   # the module defines ``set`` (public API name); keep the builtin reachable


# ------------------------------------------------------------------------------------------ experiments
def presets() -> list[str]:
    """Names of the built-in experiment presets."""
    from .presets import PRESETS
    return list(PRESETS)


def preset(name: str) -> Experiment:
    """A fresh copy of a built-in preset (see :func:`presets`)."""
    from .presets import PRESETS
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]()


def load(path: str | Path) -> Experiment:
    """Read and validate an experiment JSON file."""
    return Experiment.model_validate_json(Path(path).read_text())


def dump(exp: Experiment, path: str | Path) -> Path:
    """Write an experiment as indented JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(exp.model_dump(mode="json"), indent=2))
    return p


def validate(exp: Experiment | dict) -> Experiment:
    """Validate (and return) an experiment given as model or JSON dict."""
    return exp if isinstance(exp, Experiment) else Experiment.model_validate(exp)


def get(exp: Experiment, pointer: str) -> Any:
    """Value at an RFC 6901 pointer of the experiment's JSON tree."""
    cur: Any = exp.model_dump(mode="json")
    from .sweeps import parse_pointer
    for t in parse_pointer(pointer):
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


def set(exp: Experiment, pointer: str, value: Any) -> Experiment:  # noqa: A001 - public API name
    """Return a validated copy of ``exp`` with ``value`` at the RFC 6901 ``pointer``
    (e.g. ``/objects/0/shape/layers/2/thickness/thickness_mm``).  Never mutates ``exp``; raises
    :class:`SweepError` for a missing path or a value that no longer validates."""
    return apply_value(exp.model_dump(mode="json"), pointer, value)


def set_many(exp: Experiment, values: dict[str, Any]) -> Experiment:
    """Like :func:`set` for several ``{pointer: value}`` pairs, validated once."""
    doc = copy.deepcopy(exp.model_dump(mode="json"))
    for p, v in values.items():
        set_pointer(doc, p, v)
    from pydantic import ValidationError
    try:
        return Experiment.model_validate(doc)
    except ValidationError as e:
        err = e.errors()[0]
        raise SweepError(f"{'/'.join(str(x) for x in err['loc'])}: {err['msg']}") from e


# ------------------------------------------------------------------------------------------ capture
def capture(exp: Experiment, mode: str = "preview", seed: int | None = None, *, excitations=None,
            scale: float | None = None, progress=None, cancel=None, transport=None, tracer=None):
    """Run the simulation.  ``mode``: ``preview`` | ``full`` | ``sample`` | ``dark`` | ``white_reference``.

    ``seed`` overrides ``acquisition.seed`` (same seed -> same result).  The returned ``CaptureResult`` carries
    ``res.meta['provenance']`` (versions, git commit, experiment hash, wall time)."""
    from . import capture as cap
    exp = validate(exp)
    if seed is not None:
        exp = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"seed": int(seed)})})
    kw = dict(progress=progress, cancel=cancel, transport=transport, tracer=tracer)
    if mode == "preview":
        ex = None if excitations is None else list(excitations)[0]
        res = cap.preview(exp, excitation_nm=ex, scale=scale, **kw)
    elif mode in ("full", "sample", "dark", "white_reference"):
        res = cap.capture(exp, mode="sample" if mode == "full" else mode, scale=scale or 1.0,
                          excitations=excitations, **kw)
    else:
        raise ValueError(f"mode must be preview|full|sample|dark|white_reference, got {mode!r}")
    attach(res, exp, requested_mode=mode, seed=int(exp.acquisition.seed))
    return res


# ------------------------------------------------------------------------------------------ analysis
def eem(res, roi, norm=None, stat: str = "mean", **kw):
    """Excitation-emission matrix of a ROI (:func:`analysis.eem`)."""
    return analysis.eem(res, roi, norm, stat, **kw)


def roi_stats(res, roi, norm=None, stat: str = "mean", **kw):
    return analysis.roi_stats(res, roi, norm, stat, **kw)


def response(res, x_ground_truth: str, metric, roi=None, bins: int = 24, norm=None, **kw):
    """Binned metric-vs-ground-truth response curve (:func:`analysis.response_curve`)."""
    return analysis.response_curve(res, x_ground_truth, metric, roi, bins, norm, **kw)


def line_profile(res, p0, p1, **kw):
    return analysis.line_profile(res, p0, p1, **kw)


# ------------------------------------------------------------------------------------------ I/O
def save(res, path: str | Path, fmt: str | None = None) -> Path:
    """Save a capture.  ``fmt`` (default from the suffix): ``npz`` | ``spectradata_pkl`` | ``tiff_zip`` |
    ``ground_truth_zip`` | ``experiment_json``.  npz files embed the provenance and reload with
    :func:`load_capture`."""
    p = Path(path)
    fmt = fmt or {".npz": "npz", ".pkl": "spectradata_pkl", ".json": "experiment_json",
                  ".zip": "tiff_zip"}.get(p.suffix.lower(), "npz")
    payload, _name, _media = exports.build_export(res, "api", p.stem, fmt)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


def load_capture(path: str | Path):
    """Reload an npz written by :func:`save` (ground truth, volumes and ``meta`` included)."""
    return analysis.load_npz_capture(path)


def to_spectradata(res, sample_name: str | None = None, labels: dict | None = None):
    """Convert a capture to :class:`spectral_select.types.SpectraData` (the ML pipeline input).

    Cubes are dark-offset subtracted ADU (as in the pickle export), ``exposure_time`` in seconds and
    ``laser_power`` in mW per excitation.  ``labels`` (optional ``{name: 2-D array}``) and the experiment JSON /
    provenance are stored in ``SpectraData.metadata``."""
    import numpy as np
    from spectral_select.types import ExcitationData, SpectraData
    name = sample_name or json.loads(res.experiment_json).get("name", "sample")
    pl = exports.spectradata_payload(res, name)
    exs = {}
    for k, d in pl["data"].items():
        ex = float(k)
        exs[ex] = ExcitationData(excitation_nm=ex, cube=d["cube"], emission_wavelengths=list(d["wavelengths"]),
                                 exposure_time=pl["exposure_times"][k], laser_power=pl["laser_powers"][k])
    meta = {k: v for k, v in pl.items() if k not in ("data", "excitation_wavelengths", "mask",
                                                      "exposure_times", "laser_powers")}
    if labels:
        meta["labels"] = {str(k): np.asarray(v) for k, v in labels.items()}
    return SpectraData(excitations=exs, mask=pl["mask"], sample_name=name, metadata=meta)


def methods_text(exp: Experiment, study=None, *, style: str = "markdown") -> str:
    """Publication-ready Methods paragraph (see :mod:`.methods_text`)."""
    from . import methods_text as mt
    return mt.methods_text(exp, study, style=style)
