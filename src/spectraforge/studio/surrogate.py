r"""Surrogate models of the simulator for fast parameter retrieval and experiment design (v1.2, W4).

A *surrogate* replaces the expensive forward model :math:`\mu(\theta)` (a 3-D Monte Carlo capture, seconds)
by an interpolant of the **noise-free ROI-mean signal** of every (excitation, emission band) channel, sampled
on a grid (or a Sobol set) of 1-3 experiment parameters.  Because the interpolant is cheap (microseconds) it
can be called thousands of times by the maximum-likelihood fit (:mod:`spectraforge.studio.inverse`) and by the
Fisher-information machinery (:mod:`spectraforge.studio.information`).

What is stored
--------------
The simulator is run with its *noise-free* output ``CaptureResult.electrons_mean`` (expected signal
photo-electrons per pixel after optical blur, **before** shot / read / dark noise, PRNU, quantisation); the
detector noise is therefore OFF by construction and does not have to be toggled.  The Monte Carlo seed of the
experiment (``acquisition.seed`` and the 3-D MC streams derived from it) is **fixed** across the grid, i.e.
all grid points use *common random numbers* (CRN): the MC noise of neighbouring parameter values is positively
correlated, so differences (and finite-difference Jacobians) are far smoother than with independent seeds.
The residual MC noise is removed by an optional smoothing spline (``smooth="gcv"``).

Noise model (derived from ``sensor.detect``)
--------------------------------------------
For one pixel and one channel ``c`` the detector chain is ``S`` signal electrons ``->`` Poisson(``S + D``)
``->`` ``+ N(0, sigma_r^2)`` ``->`` ``/ g`` ``+`` offset ``->`` round (ADC).  The variance of a single pixel
in electrons^2 is (shot + dark + read + quantisation, the last being the variance ``1/12`` ADU^2 of uniform
rounding error times ``g^2``)::

    var_e(S) = (S + D) + sigma_r^2 + g^2 / 12                          [e-^2]
    var_ADU  = var_e / g^2 = (S + D)/g^2 + sigma_r^2/g^2 + 1/12        [ADU^2]

and the mean over the ``n`` independent pixels of an ROI divides it by ``n``.  Each term follows the
corresponding toggle of :class:`~spectraforge.studio.schema.PhysicsToggles`.  The fixed-pattern terms
(PRNU, DSNU, hot pixels) are *systematic* (identical in every frame) and are not noise in this sense; they are
not modelled (documented limitation).  References: Janesick, *Photon Transfer* (SPIE Press 2007), ch. 3-5;
Healey & Kondepudy, IEEE TPAMI 16(3), 267-276 (1994), doi:10.1109/34.276126 (CCD noise model).

Public API
----------
:class:`ParamSpec`, :func:`apply_params`, :class:`NoiseModel`, :class:`Surrogate`, :func:`build_surrogate`,
:func:`simulate_channels` (one simulator evaluation), :func:`roi_signal`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy import interpolate as sci

from .schema import Experiment
from .sweeps import SweepError, parse_pointer, set_pointer

ENGINE_TAG = "surrogate-1"
DEFAULT_ROI = {"shape": "rect", "normalized": True, "x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6}
MAX_PARAMS = 3


class InverseError(ValueError):
    """Invalid inverse-problem request (HTTP 422 in the router)."""


# --------------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------------
def get_pointer(doc: Any, path: str) -> Any:
    """Read an existing location of a dict / list tree by RFC 6901 pointer."""
    cur = doc
    for t in parse_pointer(path):
        if isinstance(cur, dict):
            if t not in cur:
                raise SweepError(f"path {path!r}: no key '{t}'")
            cur = cur[t]
        elif isinstance(cur, list) and t.isdigit() and int(t) < len(cur):
            cur = cur[int(t)]
        else:
            raise SweepError(f"path {path!r}: cannot resolve '{t}'")
    return cur


@dataclass(frozen=True)
class ParamSpec:
    """One scalar unknown of the inverse problem.

    ``paths``: one or several JSON pointers into the experiment JSON.  ``mode='absolute'`` sets every path to
    the parameter value; ``mode='shift'`` makes the parameter the value of the FIRST path and moves the others
    by the same amount (e.g. ``thickness_mm`` and ``end_thickness_mm`` of a wedge: the parameter is the left-edge
    thickness, the wedge keeps its slope).  ``lo`` / ``hi``: bounds used by the fit and the default grid span.
    """

    name: str
    paths: tuple[str, ...]
    mode: str = "absolute"
    lo: float | None = None
    hi: float | None = None

    @staticmethod
    def coerce(x: "ParamSpec | str | dict") -> "ParamSpec":
        if isinstance(x, ParamSpec):
            return x
        if isinstance(x, str):
            return ParamSpec(name=x.rstrip("/").split("/")[-1] or x, paths=(x,))
        if isinstance(x, dict):
            paths = x.get("paths") or ([x["path"]] if "path" in x else None)
            if not paths:
                raise InverseError("a parameter needs 'path' or 'paths'")
            if x.get("mode", "absolute") not in ("absolute", "shift"):
                raise InverseError("parameter mode must be 'absolute' or 'shift'")
            nm = x.get("name") or str(paths[0]).rstrip("/").split("/")[-1]
            return ParamSpec(nm, tuple(str(p) for p in paths), x.get("mode", "absolute"),
                             None if x.get("lo") is None else float(x["lo"]),
                             None if x.get("hi") is None else float(x["hi"]))
        raise InverseError(f"cannot interpret parameter {x!r}")

    def to_dict(self) -> dict:
        return {"name": self.name, "paths": list(self.paths), "mode": self.mode, "lo": self.lo, "hi": self.hi}

    def reference(self, doc: dict) -> float:
        return float(get_pointer(doc, self.paths[0]))

    def apply(self, doc: dict, value: float, ref_doc: dict) -> None:
        v = float(value)
        if self.mode == "absolute":
            for p in self.paths:
                set_pointer(doc, p, v)
        else:
            delta = v - self.reference(ref_doc)
            for p in self.paths:
                set_pointer(doc, p, float(get_pointer(ref_doc, p)) + delta)


def coerce_params(params: Sequence) -> list[ParamSpec]:
    specs = [ParamSpec.coerce(p) for p in params]
    if not 1 <= len(specs) <= MAX_PARAMS:
        raise InverseError(f"1 to {MAX_PARAMS} parameters are supported (got {len(specs)})")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise InverseError(f"parameter names must be unique, got {names}")
    return specs


def apply_params(base: Experiment | dict, specs: Sequence[ParamSpec], theta: Sequence[float],
                 overrides: dict | None = None) -> Experiment:
    """Deep-copy ``base``, apply ``overrides`` ({pointer: value}) and the parameter values, re-validate."""
    ref = base if isinstance(base, dict) else base.model_dump(mode="json")
    doc = copy.deepcopy(ref)
    for p, v in (overrides or {}).items():
        set_pointer(doc, p, v)
    ref2 = copy.deepcopy(doc)
    for s, v in zip(specs, theta):
        s.apply(doc, float(v), ref2)
    try:
        return Experiment.model_validate(doc)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError -> client error
        raise InverseError(f"parameter values {list(map(float, theta))} give an invalid experiment: {exc}") from exc


# --------------------------------------------------------------------------------------------
# channels, ROI signal, noise model
# --------------------------------------------------------------------------------------------
@dataclass
class ChannelTable:
    """The (excitation, emission band) channels of a capture, flattened excitation-major."""

    ex_nm: np.ndarray        # (C,)
    em_nm: np.ndarray        # (C,)

    @property
    def n(self) -> int:
        return int(self.ex_nm.size)

    @staticmethod
    def from_result(result, excitations: Sequence[float] | None = None) -> "ChannelTable":
        exs = [float(e) for e in result.excitations_nm if excitations is None or
               any(abs(e - x) < 1e-6 for x in excitations)]
        ex = np.concatenate([np.full(len(result.emission_nm[e]), e) for e in exs]) if exs else np.zeros(0)
        em = np.concatenate([np.asarray(result.emission_nm[e], float) for e in exs]) if exs else np.zeros(0)
        return ChannelTable(ex, em)

    def select(self, excitations="all", bands="all") -> np.ndarray:
        """Boolean mask.  ``excitations``: 'all' or a list of nm; ``bands``: 'all', ``(lo, hi)`` nm span, or
        a list of emission-band centres (nm, matched to the nearest band with 0.5 nm tolerance)."""
        m = np.ones(self.n, dtype=bool)
        if not (isinstance(excitations, str) and excitations == "all"):
            m &= np.any([np.abs(self.ex_nm - float(e)) < 1e-6 for e in excitations], axis=0)
        if not (isinstance(bands, str) and bands == "all"):
            b = list(bands)
            if len(b) == 2 and all(isinstance(v, (int, float)) for v in b) and isinstance(bands, tuple):
                m &= (self.em_nm >= b[0] - 1e-9) & (self.em_nm <= b[1] + 1e-9)
            else:
                m &= np.any([np.abs(self.em_nm - float(v)) < 0.5 for v in b], axis=0)
        return m

    def labels(self) -> list[str]:
        return [f"{e:g}->{b:g}" for e, b in zip(self.ex_nm, self.em_nm)]

    def to_dict(self) -> dict:
        return {"ex_nm": self.ex_nm.tolist(), "em_nm": self.em_nm.tolist()}


def resolve_roi(roi: dict | None, width: int, height: int) -> dict:
    """ROI spec in pixel coordinates.  ``{"normalized": true, ...}`` coordinates (fractions of the image width /
    height; ``radius_px`` and ``rx`` as fractions of the width, ``ry`` of the height) are scaled to the capture
    resolution, so the same ROI works at every preview scale."""
    r = dict(roi or DEFAULT_ROI)
    if r.pop("normalized", False):
        for k in ("x", "x0", "x1", "cx", "rx"):
            if k in r:
                r[k] = float(r[k]) * width
        for k in ("y", "y0", "y1", "cy", "ry"):
            if k in r:
                r[k] = float(r[k]) * height
        if "radius_px" in r:
            r["radius_px"] = float(r["radius_px"]) * width
        if "points" in r:
            r["points"] = [(float(a) * width, float(b) * height) for a, b in r["points"]]
    return r


def roi_signal(result, roi: dict | None, *, excitations=None, field: str = "electrons_mean"
               ) -> tuple[np.ndarray, int, ChannelTable]:
    """Mean over the ROI pixels of ``result.<field>`` for every channel -> ``(values (C,), n_pixels, channels)``."""
    from . import analysis
    spec = resolve_roi(roi, result.width, result.height)
    mask = analysis.roi_mask(spec, result.height, result.width)
    n = int(mask.sum())
    if n == 0:
        raise InverseError("the ROI contains no pixels at this resolution")
    table = ChannelTable.from_result(result, excitations)
    out = []
    src = getattr(result, field)
    for e in dict.fromkeys(table.ex_nm.tolist()):
        cube = np.asarray(src[e], dtype=np.float64)
        out.append(cube[mask].mean(axis=0))
    return np.concatenate(out), n, table


@dataclass
class NoiseModel:
    """Poisson-Gaussian noise of the ROI-mean signal of every channel (see the module docstring).

    All arrays are per channel; signals are expected *signal electrons* ``S`` (dark ``D`` is added here)."""

    gain_e_per_adu: float
    read_e: float
    dark_e: np.ndarray                   # (C,) expected dark electrons per pixel and exposure
    n_pix: float = 1.0                   # independent pixels averaged (effective, full detector resolution)
    shot: bool = True
    read: bool = True
    quantization: bool = True
    full_well_e: float = 1e30
    #: systematic model / calibration error, relative to the signal: adds ``(eps * S)^2`` to every channel variance
    #: (not reduced by pixel averaging, it is common to all pixels of the ROI).  0 = photon / read noise only.
    model_error_rel: float = 0.0

    def with_model_error(self, eps: float) -> "NoiseModel":
        return replace(self, model_error_rel=float(eps))

    def variance_e(self, signal_e: np.ndarray) -> np.ndarray:
        """Variance of the ROI mean in e-^2 (``signal_e`` may be (C,) or (N, C)); includes the optional systematic
        term ``(model_error_rel * S)^2``."""
        s = np.maximum(np.asarray(signal_e, dtype=float), 0.0)
        v = (s + self.dark_e) if self.shot else np.zeros_like(s)
        if self.read:
            v = v + self.read_e ** 2
        if self.quantization:
            v = v + self.gain_e_per_adu ** 2 / 12.0
        v = v / self.n_pix
        if self.model_error_rel:
            v = v + (self.model_error_rel * s) ** 2
        return v

    def variance_adu(self, signal_e: np.ndarray) -> np.ndarray:
        return self.variance_e(signal_e) / self.gain_e_per_adu ** 2

    def subset(self, mask: np.ndarray) -> "NoiseModel":
        return replace(self, dark_e=self.dark_e[mask])

    def with_n_pix(self, n: float) -> "NoiseModel":
        return replace(self, n_pix=float(n))

    def to_json(self) -> dict:
        return {"gain_e_per_adu": self.gain_e_per_adu, "read_e": self.read_e, "dark_e": self.dark_e.tolist(),
                "n_pix": self.n_pix, "shot": self.shot, "read": self.read, "quantization": self.quantization,
                "full_well_e": self.full_well_e, "model_error_rel": self.model_error_rel}

    @staticmethod
    def from_json(d: dict) -> "NoiseModel":
        return NoiseModel(float(d["gain_e_per_adu"]), float(d["read_e"]), np.asarray(d["dark_e"], float),
                          float(d["n_pix"]), bool(d["shot"]), bool(d["read"]), bool(d["quantization"]),
                          float(d.get("full_well_e", 1e30)), float(d.get("model_error_rel", 0.0)))


def noise_model_for(exp: Experiment, table: ChannelTable, exposure_ms: dict[float, float],
                    n_pix: float) -> NoiseModel:
    """Build the :class:`NoiseModel` of ``exp``'s active camera for the channels ``table``.

    ``exposure_ms``: excitation -> exposure of that step (``CaptureResult.exposure_ms``)."""
    cam, tg = exp.camera(), exp.acquisition.physics
    s = cam.sensor
    dark = np.array([s.dark_current_e_per_s * exposure_ms[float(e)] * 1e-3 if tg.dark_current else 0.0
                     for e in table.ex_nm])
    return NoiseModel(float(s.gain_e_per_adu), float(s.read_noise_e), dark, float(n_pix), bool(tg.shot_noise),
                      bool(tg.read_noise and s.read_noise_e > 0), bool(tg.quantization), float(s.full_well_e))


def effective_n_pixels(n_roi_pixels: int, scale: float) -> float:
    """Independent full-resolution detector pixels represented by ``n_roi_pixels`` pixels of a capture at
    resolution ``scale`` (a coarser capture samples the same detector area with fewer, per-pixel-equivalent
    samples, ``1/scale^2`` full pixels each)."""
    return float(n_roi_pixels) / float(scale) ** 2


@dataclass
class ChannelMean:
    """Result of one simulator evaluation: ROI-mean noise-free signal electrons per channel + noise model."""

    mean_e: np.ndarray
    channels: ChannelTable
    noise: NoiseModel
    n_roi_pixels: int
    scale: float
    exposure_ms: dict
    timings_s: dict = field(default_factory=dict)
    result: Any = None            # the CaptureResult (kept only on request)


def simulate_channels(exp: Experiment, roi: dict | None = None, *, scale: float = 0.25, excitations=None,
                      capture_fn: Callable | None = None, n_pixels: float | None = None,
                      keep_result: bool = False) -> ChannelMean:
    """Run the simulator once and return the noise-free ROI-mean signal of every channel.

    ``capture_fn(exp, scale=, excitations=) -> CaptureResult`` defaults to :func:`spectraforge.studio.capture.capture`
    (injectable for tests)."""
    if capture_fn is None:
        from .capture import capture as capture_fn  # noqa: PLC0415 - heavy import, deferred
    t0 = time.perf_counter()
    res = capture_fn(exp, scale=scale, excitations=excitations)
    if not res.excitations_nm:
        raise InverseError("the simulator returned an empty capture (cancelled or no enabled excitation)")
    vals, n, table = roi_signal(res, roi, excitations=excitations)
    nm = noise_model_for(exp, table, res.exposure_ms,
                         float(n_pixels) if n_pixels else effective_n_pixels(n, scale))
    return ChannelMean(vals, table, nm, n, scale, dict(res.exposure_ms),
                       {"capture_s": time.perf_counter() - t0, **dict(res.timings_s)},
                       res if keep_result else None)


# --------------------------------------------------------------------------------------------
# the surrogate
# --------------------------------------------------------------------------------------------
class Surrogate:
    """Interpolant ``theta -> expected signal electrons of every channel`` plus the instrument noise model.

    ``interp``: ``'cubic'`` (1-D: not-a-knot cubic spline, or a GCV smoothing spline when ``smooth='gcv'``),
    ``'multilinear'`` / ``'cubic'`` on tensor grids (RegularGridInterpolator), ``'rbf'`` for scattered (Sobol)
    points (thin-plate-spline RBF with optional ``smooth`` ridge)."""

    def __init__(self, params: Sequence[ParamSpec], theta: np.ndarray, values_e: np.ndarray,
                 channels: ChannelTable, noise: NoiseModel | None, *, mode: str = "grid",
                 grid_axes: Sequence[np.ndarray] | None = None, interp: str = "auto",
                 smooth: float | str | None = None, meta: dict | None = None):
        self.params = list(params)
        self.theta = np.atleast_2d(np.asarray(theta, float))
        self.values_e = np.asarray(values_e, float)
        self.channels = channels
        self.noise = noise
        self.mode = mode
        self.grid_axes = [np.asarray(a, float) for a in grid_axes] if grid_axes is not None else None
        self.smooth = smooth
        self.meta = dict(meta or {})
        P = len(self.params)
        if self.theta.shape[1] != P:
            raise InverseError("theta must have one column per parameter")
        if self.values_e.shape != (self.theta.shape[0], channels.n):
            raise InverseError(f"values_e must be (n_points, n_channels) = {(self.theta.shape[0], channels.n)}, "
                               f"got {self.values_e.shape}")
        self.bounds = np.stack([self.theta.min(axis=0), self.theta.max(axis=0)], axis=1)     # (P, 2)
        self.interp = self._choose_interp(interp)
        self._f = self._build()

    # ------------------------------------------------------------------ construction
    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def _choose_interp(self, interp: str) -> str:
        if interp == "auto":
            if self.mode == "sobol" or self.grid_axes is None:
                return "rbf"
            return "cubic" if (self.n_params == 1 or min(len(a) for a in self.grid_axes) >= 4) else "multilinear"
        if interp not in ("cubic", "multilinear", "rbf"):
            raise InverseError("interp must be 'auto', 'cubic', 'multilinear' or 'rbf'")
        if interp != "rbf" and self.grid_axes is None:
            raise InverseError(f"interp '{interp}' needs a tensor grid (mode='grid')")
        return interp

    def _build(self) -> Callable[[np.ndarray], np.ndarray]:
        V = self.values_e
        if self.interp == "rbf":
            sm = 0.0 if self.smooth in (None, "gcv") else float(self.smooth)
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            span = np.where(hi > lo, hi - lo, 1.0)
            rbf = sci.RBFInterpolator((self.theta - lo) / span, V, kernel="thin_plate_spline", smoothing=sm, degree=1)
            P = self.n_params
            return lambda x: rbf((np.asarray(x, float).reshape(-1, P) - lo) / span)
        if self.n_params == 1:
            x = self.grid_axes[0]
            order = np.argsort(x)
            xs, Vs = x[order], V[order]
            if self.smooth == "gcv" or isinstance(self.smooth, float) and self.smooth > 0:
                lam = None if self.smooth == "gcv" else float(self.smooth)
                if xs.size < 4:
                    raise InverseError("smoothing needs at least 4 grid points")
                splines = [sci.make_smoothing_spline(xs, Vs[:, c], lam=lam) for c in range(Vs.shape[1])]
                return lambda t: np.stack([s(np.atleast_1d(np.asarray(t, float)).reshape(-1)) for s in splines],
                                          axis=-1)
            cs = sci.CubicSpline(xs, Vs, axis=0, bc_type="not-a-knot") if (self.interp == "cubic" and xs.size >= 4) \
                else sci.interp1d(xs, Vs, axis=0, kind="linear")
            return lambda t: np.atleast_2d(cs(np.atleast_1d(np.asarray(t, float)).reshape(-1))).reshape(-1, V.shape[1])
        shape = tuple(len(a) for a in self.grid_axes) + (V.shape[1],)
        # V rows are ordered with the LAST parameter fastest (itertools.product order)
        rgi = sci.RegularGridInterpolator(self.grid_axes, V.reshape(shape),
                                          method="cubic" if self.interp == "cubic" else "linear",
                                          bounds_error=False, fill_value=None)
        return lambda x: rgi(np.asarray(x, float).reshape(-1, len(self.grid_axes)))

    # ------------------------------------------------------------------ evaluation
    def _as2d(self, theta) -> np.ndarray:
        t = np.asarray(theta, float)
        if t.ndim == 0:
            return t.reshape(1, 1)
        if t.ndim == 1:
            return t.reshape(1, -1) if (self.n_params > 1 or t.size == 1) else t.reshape(-1, 1)
        return t

    def clip(self, theta) -> np.ndarray:
        return np.clip(self._as2d(theta), self.bounds[:, 0], self.bounds[:, 1])

    def predict_e(self, theta, clip: bool = True) -> np.ndarray:
        """Expected signal electrons: ``(C,)`` for one ``theta`` of ``n_params`` entries, ``(N, C)`` for ``(N, P)``."""
        single = np.ndim(theta) <= 1 and np.size(theta) == self.n_params
        t = self.clip(theta) if clip else self._as2d(theta)
        out = np.asarray(self._f(t[:, 0] if self.n_params == 1 else t)).reshape(t.shape[0], -1)
        return out[0] if single else out

    def jac_e(self, theta, rel_step: float = 1e-3) -> np.ndarray:
        """Jacobian ``(C, P)`` d(signal electrons)/d(theta) by central differences of the (smooth) interpolant."""
        th = np.asarray(theta, float).reshape(-1)
        J = np.empty((self.channels.n, self.n_params))
        for j in range(self.n_params):
            lo, hi = self.bounds[j]
            h = rel_step * (hi - lo if hi > lo else 1.0)
            a, b = th.copy(), th.copy()
            a[j], b[j] = max(th[j] - h, lo), min(th[j] + h, hi)
            if b[j] - a[j] <= 0:
                J[:, j] = 0.0
                continue
            J[:, j] = (self.predict_e(b) - self.predict_e(a)) / (b[j] - a[j])
        return J

    def dense_table(self, n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``(theta (n,P), predicted electrons (n,C))`` on a dense grid (1-D) or Sobol set; cached.  Used as the
        starting-point scan of the fit."""
        key = (n, seed)
        cache = self.__dict__.setdefault("_dense", {})
        if key not in cache:
            if self.n_params == 1:
                t = np.linspace(self.bounds[0, 0], self.bounds[0, 1], n).reshape(-1, 1)
            else:
                from scipy.stats import qmc
                u = qmc.Sobol(self.n_params, seed=seed).random(n)
                t = self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])
            cache[key] = (t, self.predict_e(t))
        return cache[key]

    def subset(self, mask: np.ndarray) -> "Surrogate":
        """Surrogate restricted to the channels ``mask`` (shares the training data)."""
        ch = ChannelTable(self.channels.ex_nm[mask], self.channels.em_nm[mask])
        nm = self.noise.subset(mask) if self.noise is not None else None
        return Surrogate(self.params, self.theta, self.values_e[:, mask], ch, nm, mode=self.mode,
                         grid_axes=self.grid_axes, interp=self.interp, smooth=self.smooth, meta=self.meta)

    def loo_error(self) -> dict:
        """Hold-out check: leave-one-out relative error of interior points (1-D grids only), RMS over channels
        weighted by signal.  Returns ``{}`` when not applicable."""
        if self.n_params != 1 or self.grid_axes is None or self.theta.shape[0] < 6:
            return {}
        x, V = self.grid_axes[0], self.values_e
        errs = []
        for i in range(1, x.size - 1):
            keep = np.arange(x.size) != i
            if self.smooth == "gcv":
                pred = np.array([sci.make_smoothing_spline(x[keep], V[keep, c])(x[i]) for c in range(V.shape[1])])
            else:
                pred = sci.CubicSpline(x[keep], V[keep], axis=0)(x[i])
            errs.append(np.sqrt(np.sum((pred - V[i]) ** 2) / max(np.sum(V[i] ** 2), 1e-30)))
        return {"loo_rel_rms_mean": float(np.mean(errs)), "loo_rel_rms_max": float(np.max(errs)), "n": len(errs)}

    # ------------------------------------------------------------------ persistence
    def to_dict(self, include_values: bool = True) -> dict:
        d = {"names": self.names, "n_channels": self.channels.n, "params": [p.to_dict() for p in self.params],
             "mode": self.mode, "interp": self.interp, "smooth": self.smooth, "bounds": self.bounds.tolist(), "n_points": int(self.theta.shape[0]),
             "channels": self.channels.to_dict(), "meta": self.meta,
             "noise": self.noise.to_json() if self.noise is not None else None}
        if include_values:
            d["theta"], d["values_e"] = self.theta.tolist(), self.values_e.tolist()
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"params": [p.to_dict() for p in self.params], "mode": self.mode, "interp": self.interp,
                "smooth": self.smooth, "meta": self.meta,
                "noise": self.noise.to_json() if self.noise is not None else None}
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, theta=self.theta, values_e=self.values_e, ex_nm=self.channels.ex_nm,
                            em_nm=self.channels.em_nm, meta=np.array(json.dumps(meta)),
                            **({f"axis{i}": a for i, a in enumerate(self.grid_axes)} if self.grid_axes else {}))
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "Surrogate":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            axes = [z[f"axis{i}"] for i in range(len(meta["params"])) if f"axis{i}" in z.files] or None
            params = [ParamSpec.coerce(p) for p in meta["params"]]
            return Surrogate(params, z["theta"], z["values_e"], ChannelTable(z["ex_nm"], z["em_nm"]),
                             NoiseModel.from_json(meta["noise"]) if meta["noise"] else None, mode=meta["mode"],
                             grid_axes=axes, interp=meta["interp"], smooth=meta["smooth"], meta=meta["meta"])


# --------------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------------
def cache_dir() -> Path:
    d = Path(os.environ.get("SPECTRALFORGE_SURROGATE_CACHE", Path.home() / ".cache" / "spectralforge" / "surrogates"))
    return d


def _stable(o: Any) -> str:
    return json.dumps(o, sort_keys=True, default=lambda v: np.asarray(v).tolist() if hasattr(v, "__array__")
                      else str(v))


def surrogate_key(base: Experiment, specs, grid_desc, mode, seed, roi, scale, excitations, overrides, smooth, interp,
                  n_pixels) -> str:
    blob = _stable({"exp": base.model_dump(mode="json"), "params": [s.to_dict() for s in specs], "grid": grid_desc,
                    "mode": mode, "seed": seed, "roi": roi, "scale": scale, "ex": excitations, "ov": overrides,
                    "smooth": smooth, "interp": interp, "npix": n_pixels, "tag": ENGINE_TAG})
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _axes_from_grid(specs: Sequence[ParamSpec], grid) -> list[np.ndarray]:
    if isinstance(grid, dict):
        axes = [grid[s.name] if s.name in grid else grid.get(s.paths[0]) for s in specs]
        if any(a is None for a in axes):
            raise InverseError(f"grid must give values for every parameter {[s.name for s in specs]}")
    else:
        axes = list(grid)
        if len(axes) != len(specs):
            raise InverseError("grid must have one axis per parameter")
    out = []
    for s, a in zip(specs, axes):
        a = np.unique(np.asarray(a, dtype=float).reshape(-1))
        if a.size < 2:
            raise InverseError(f"grid axis of '{s.name}' needs at least 2 distinct values")
        if not np.all(np.isfinite(a)):
            raise InverseError(f"grid axis of '{s.name}' has non-finite values")
        out.append(a)
    return out


def build_surrogate(base_exp: Experiment, params: Sequence, grid, metric_fn: Callable | str = "roi_eem",
                    mode: str = "grid", seed: int = 0, *, roi: dict | None = None, scale: float = 0.25,
                    excitations: Sequence[float] | None = None, overrides: dict | None = None,
                    n_points: int | None = None, interp: str = "auto", smooth: float | str | None = None,
                    n_pixels: float | None = None, cache: bool | str | Path = True,
                    progress: Callable[[float, str], None] | None = None, cancel: Callable[[], bool] | None = None,
                    capture_fn: Callable | None = None) -> Surrogate:
    """Sample the simulator on a grid (``mode='grid'``) or Sobol set (``mode='sobol'``) and return a :class:`Surrogate`.

    ``params``: 1-3 :class:`ParamSpec` / JSON pointers / dicts.  ``grid``: dict ``name -> values`` (grid mode) or
    ``name -> (lo, hi)`` (Sobol mode with ``n_points`` samples).  ``metric_fn``: ``"roi_eem"`` (the ROI-mean signal
    electrons of every channel) or a callable ``f(CaptureResult) -> 1-D array`` (then the surrogate has no noise
    model and the caller must give one to the fit).  ``overrides``: ``{pointer: value}`` applied to the base experiment
    before the parameters (e.g. make a wedge uniform: ``{".../thickness/kind": "uniform"}``).  ``seed`` seeds the
    Sobol sequence only: the simulator's own seed (``acquisition.seed``) is left unchanged for all points (common
    random numbers).  ``cache`` writes / reads ``<cache_dir>/<key>.npz`` (``False`` disables, a path overrides the
    directory).  ``smooth``: ``None`` interpolate, ``'gcv'`` GCV smoothing spline (1-D) to filter MC noise."""
    specs = coerce_params(params)
    base_doc = base_exp.model_dump(mode="json")
    if mode not in ("grid", "sobol"):
        raise InverseError("mode must be 'grid' or 'sobol'")
    if mode == "grid":
        axes = _axes_from_grid(specs, grid)
        import itertools
        pts = np.array(list(itertools.product(*axes)), dtype=float)
        grid_desc = [a.tolist() for a in axes]
    else:
        from scipy.stats import qmc
        n = int(n_points or 0)
        if n < 4:
            raise InverseError("sobol mode needs n_points >= 4")
        spans = []
        for s in specs:
            lo_hi = grid.get(s.name) if isinstance(grid, dict) else None
            if lo_hi is None or len(lo_hi) != 2 or not lo_hi[1] > lo_hi[0]:
                raise InverseError(f"sobol mode needs a (lo, hi) span for '{s.name}'")
            spans.append((float(lo_hi[0]), float(lo_hi[1])))
        u = qmc.Sobol(len(specs), seed=int(seed)).random(n)
        pts = np.array([[lo + ui * (hi - lo) for ui, (lo, hi) in zip(row, spans)] for row in u])
        axes = None
        grid_desc = spans
    if len(pts) > 4096:
        raise InverseError(f"{len(pts)} simulator evaluations requested; the limit is 4096")
    custom = callable(metric_fn)
    ckey = None
    cdir = None
    if cache and not custom:
        cdir = Path(cache) if isinstance(cache, (str, Path)) else cache_dir()
        ckey = surrogate_key(base_exp, specs, grid_desc, mode, seed, roi, scale,
                             None if excitations is None else [float(e) for e in excitations], overrides, smooth,
                             interp, n_pixels)
        f = cdir / f"{ckey}.npz"
        if f.exists():
            try:
                s = Surrogate.load(f)
                s.meta["cache"] = "hit"
                return s
            except Exception:  # noqa: BLE001 - corrupt cache file: rebuild
                f.unlink(missing_ok=True)

    vals, table, noise, nroi, t0, expo = [], None, None, 0, time.perf_counter(), {}
    timings = []
    for i, th in enumerate(pts):
        if cancel is not None and cancel():
            from .jobs import Cancelled
            raise Cancelled()
        exp = apply_params(base_doc, specs, th, overrides)
        if custom:
            if capture_fn is None:
                from .capture import capture as capture_fn  # noqa: PLC0415
            res = capture_fn(exp, scale=scale, excitations=excitations)
            vals.append(np.asarray(metric_fn(res), float).reshape(-1))
        else:
            cm = simulate_channels(exp, roi, scale=scale, excitations=excitations, capture_fn=capture_fn,
                                   n_pixels=n_pixels)
            if table is None:
                table, noise, nroi = cm.channels, cm.noise, cm.n_roi_pixels
                expo = {f"{k:g}": float(v) for k, v in cm.exposure_ms.items()}
            elif cm.channels.n != table.n:
                raise InverseError("the channel set changed across the grid (excitation sequence is parameter-dependent)")
            vals.append(cm.mean_e)
            timings.append(cm.timings_s.get("capture_s", 0.0))
        if progress is not None:
            progress((i + 1) / len(pts), f"simulator evaluation {i + 1}/{len(pts)}")
    V = np.stack(vals)
    if custom:
        table = ChannelTable(np.full(V.shape[1], np.nan), np.arange(V.shape[1], dtype=float))
    sur = Surrogate(specs, pts, V, table, noise, mode=mode, grid_axes=axes, interp=interp, smooth=smooth,
                    meta={"scale": scale, "roi": roi or DEFAULT_ROI, "n_roi_pixels": nroi, "seed": int(seed),
                          "overrides": overrides or {}, "exposure_ms": expo, "build_s": time.perf_counter() - t0,
                          "mean_capture_s": float(np.mean(timings)) if timings else None, "cache": "miss",
                          "base_experiment_name": base_exp.name, "engine_tag": ENGINE_TAG,
                          "excitations": None if excitations is None else [float(e) for e in excitations]})
    if cdir is not None and ckey is not None:
        try:
            sur.save(cdir / f"{ckey}.npz")
        except OSError:
            sur.meta["cache"] = "unwritable"
    return sur
