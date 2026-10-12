"""Scientist-facing analysis of SpectralForge Studio captures (pure functions on a ``CaptureResult``).

Everything here works on an in-memory :class:`~spectraforge.studio.contracts.CaptureResult` *or* on one
re-loaded offline from an ``npz`` export with :func:`load_npz_capture`, so the same numbers the web UI shows
can be reproduced in Jupyter.  The HTTP layer (``server.py``) is a thin wrapper around these functions.

Notebook example (download the capture as ``npz`` from the UI, or ``exports.to_npz(result)``)::

    from spectraforge.studio import analysis as an
    cap = an.load_npz_capture("my_capture.npz")                       # CaptureResult, no server needed
    norm = {"mode": "exposure_power"}                                 # ADU/(s*mW), offset removed
    roi = {"shape": "disk", "x": 120, "y": 80, "radius_px": 6}
    eem = an.eem(cap, roi, norm)                                      # mean EEM of the disk
    print(eem.unit, eem.value[365.0].argmax(), eem.grid_value.shape)
    st = an.roi_stats(cap, roi, norm)                                 # + ground-truth statistics
    print(st.ground_truth["layer_thickness:cover"]["mean"])
    prof = an.line_profile(cap, (10, 80), (230, 80), width_px=5, ground_truth=["layer_thickness:cover"])
    curve = an.response_curve(cap, "layer_thickness:cover",
                              {"kind": "band", "ex": 365, "band_nm": 460}, bins=16, norm=norm)
    print(curve.bin_centres, curve.mean)                              # signal vs local cover thickness
    vol = an.volume(cap, "block1", "fluence", 365)                    # (nx, nz, layers*sublayers) float32

Conventions
-----------
* Pixel coordinates: ``x`` = column, ``y`` = row, origin top-left; the centre of pixel (col c, row r) is the
  point ``(x=c, y=r)`` (so integer coordinates address pixel centres).  Rectangles are inclusive of both corner
  pixels; a disk of radius r contains the pixel centres at distance <= r.
* Excitations are addressed by wavelength (nearest within 0.5 nm); dict keys of results are floats in Python and
  ``f"{ex:g}"`` strings in the JSON form (``to_dict``).  Non-finite numbers become ``null`` in JSON.

Normalisation modes (:class:`NormSpec`), per pixel and band, with ``S`` the sample ADU
``D`` the dark and ``W`` the white-reference ADU::

    adu               S                                     [ADU]
    radiance          noise-free band radiance              [mW/(mm^2*sr)]
    electrons         expected signal electrons             [e-]
    dark_subtracted   S - D                                 [ADU]            (needs a dark capture)
    exposure_power    (S - offset) / (t_exp[s] * P[mW])     [ADU/(s*mW)]     offset = sensor offset_adu
    white_normalized  (S - D) / (W - D)                     [ratio]          D = dark capture if given,
                                                                            else the sensor offset; NaN where
                                                                            |W - D| ~ 0
    irradiance_normalized
                      (S - offset) / (t_exp[s] * E_ex)      [ADU/(s*mW/mm^2)]  E_ex = ground-truth map
                                                                            ``irradiance@<ex>`` (mW/mm^2) at the
                                                                            pixel; NaN where E_ex = 0 or below
                                                                            ``IRRADIANCE_FLOOR`` (1e-3) of its maximum
                                                                            (a narrow beam's vanishing tail would
                                                                            otherwise divide noise by ~0).  Removes the
                                                                            lamp's spatial gradient.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Union

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .contracts import CaptureResult, VoxelGrid

GRID_STEP_NM = 2.0
IRRADIANCE_FLOOR = 1e-3        # irradiance_normalized: pixels with E_ex below this fraction of max(E_ex) are NaN
MAX_COMPONENT_PIXELS = 64
MAX_SCATTER_POINTS = 4000
MAX_POLYGON_POINTS = 10000
VOLUME_KINDS = ("fluence", "absorbed", "source")


class AnalysisError(ValueError):
    """A request the analysis library cannot satisfy (bad ROI, mismatched dark/white, unknown layer, ...).
    ``status`` is the HTTP status the server maps it to."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------------------------
# Specs (pydantic mirrors of api/types.ts)
# --------------------------------------------------------------------------------------------
F = Annotated[float, Field(allow_inf_nan=False)]


class NormSpec(BaseModel):
    mode: Literal["adu", "radiance", "electrons", "dark_subtracted", "exposure_power", "white_normalized",
                  "irradiance_normalized"] = "adu"
    dark_capture_id: str | None = None
    white_capture_id: str | None = None


class PointRoi(BaseModel):
    shape: Literal["point"]
    x: F
    y: F


class DiskRoi(BaseModel):
    shape: Literal["disk"]
    x: F
    y: F
    radius_px: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class RectRoi(BaseModel):
    shape: Literal["rect"]
    x0: F
    y0: F
    x1: F
    y1: F


class EllipseRoi(BaseModel):
    shape: Literal["ellipse"]
    cx: F
    cy: F
    rx: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    ry: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class PolygonRoi(BaseModel):
    shape: Literal["polygon"]
    points: Annotated[list[tuple[F, F]], Field(min_length=3, max_length=MAX_POLYGON_POINTS)]


RoiSpec = Annotated[Union[PointRoi, DiskRoi, RectRoi, EllipseRoi, PolygonRoi], Field(discriminator="shape")]


class BandRef(BaseModel):
    ex: F
    band_nm: F


class MetricBand(BaseModel):
    kind: Literal["band"]
    ex: F
    band_nm: F


class MetricBandRatio(BaseModel):
    kind: Literal["band_ratio"]
    a: BandRef
    b: BandRef


class MetricBandSum(BaseModel):
    kind: Literal["band_sum"]
    ex: F
    from_nm: F | None = None
    to_nm: F | None = None


class MetricPeak(BaseModel):
    kind: Literal["peak_wavelength"]
    ex: F


MetricSpec = Annotated[Union[MetricBand, MetricBandRatio, MetricBandSum, MetricPeak], Field(discriminator="kind")]

_roi_adapter: TypeAdapter = TypeAdapter(RoiSpec)
_metric_adapter: TypeAdapter = TypeAdapter(MetricSpec)


def _err_text(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())


def coerce_roi(roi) -> BaseModel:
    try:
        return _roi_adapter.validate_python(roi)
    except ValidationError as exc:
        raise AnalysisError(f"invalid ROI: {_err_text(exc)}") from exc


def coerce_metric(metric) -> BaseModel:
    try:
        return _metric_adapter.validate_python(metric)
    except ValidationError as exc:
        raise AnalysisError(f"invalid metric: {_err_text(exc)}") from exc


def coerce_norm(norm) -> NormSpec:
    if norm is None:
        return NormSpec()
    if isinstance(norm, NormSpec):
        return norm
    try:
        return NormSpec.model_validate(norm)
    except ValidationError as exc:
        raise AnalysisError(f"invalid norm: {_err_text(exc)}") from exc


# --------------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------------
def to_jsonable(x: Any) -> Any:
    """Recursively convert numpy / NaN-bearing structures to JSON-safe Python (non-finite -> None,
    float dict keys -> ``f"{k:g}"``, pydantic models -> dicts)."""
    if isinstance(x, BaseModel):
        return to_jsonable(x.model_dump())
    if isinstance(x, dict):
        return {(f"{k:g}" if isinstance(k, (float, np.floating)) else str(k)): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _key(ex: float) -> str:
    return f"{ex:g}"


# --------------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------------
UNITS = {"adu": "ADU", "radiance": "mW/(mm²·sr)", "electrons": "e⁻", "dark_subtracted": "ADU",
         "exposure_power": "ADU/(s·mW)", "white_normalized": "ratio",
         "irradiance_normalized": "ADU/(s·mW/mm²)"}


def find_ex(result, ex: float) -> float:
    """The capture's excitation nearest to ``ex`` (within 0.5 nm) or an :class:`AnalysisError`."""
    exs = list(result.excitations_nm)
    if not exs:
        raise AnalysisError("the capture has no excitation frames")
    best = min(exs, key=lambda e: abs(e - ex))
    if abs(best - ex) > 0.5:
        raise AnalysisError(f"excitation {ex:g} nm is not in the capture ({[_key(e) for e in exs]})", 404)
    return best


def sensor_offset_adu(result) -> float:
    """Sensor ``offset_adu`` of the active camera in ``result.experiment_json``."""
    try:
        exp = json.loads(result.experiment_json)
        cams = exp["cameras"]
        cam = next((c for c in cams if c["id"] == exp.get("active_camera_id")), cams[0])
        return float(cam["sensor"]["offset_adu"])
    except Exception as exc:  # noqa: BLE001
        raise AnalysisError(f"cannot read the sensor offset from the capture's experiment: {exc}") from exc


def check_compatible(sample, other, role: str) -> None:
    """Raise unless ``other`` (a dark / white capture) has the sample's pixel grid, excitations and bands."""
    if (other.width, other.height) != (sample.width, sample.height):
        raise AnalysisError(f"{role} capture is {other.width}x{other.height} px but the sample is "
                            f"{sample.width}x{sample.height} px")
    a = sorted(float(e) for e in sample.excitations_nm)
    b = sorted(float(e) for e in other.excitations_nm)
    if len(a) != len(b) or any(abs(x - y) > 0.5 for x, y in zip(a, b)):
        raise AnalysisError(f"{role} capture excitations {[_key(e) for e in b]} nm differ from the sample's "
                            f"{[_key(e) for e in a]} nm")
    for ex in sample.excitations_nm:
        o = find_ex(other, ex)
        ea, eb = np.asarray(sample.emission_nm[ex]), np.asarray(other.emission_nm[o])
        if ea.shape != eb.shape or np.max(np.abs(ea - eb), initial=0.0) > 0.5:
            raise AnalysisError(f"{role} capture bands at {_key(ex)} nm do not match the sample's bands "
                                f"({eb.size} vs {ea.size} bands, or different centres)")


def _take(cube: np.ndarray, sel) -> np.ndarray:
    a = np.asarray(cube)
    return a if sel is None else a[sel]


def _irradiance_map(result, ex: float) -> np.ndarray:
    """Ground-truth surface irradiance map ``irradiance@<ex>`` (H, W) in mW/mm^2."""
    gt = result.ground_truth or {}
    for key in (f"irradiance@{ex:g}", f"irradiance@{int(round(ex))}"):
        if key in gt:
            return np.asarray(gt[key])
    raise AnalysisError(f"irradiance_normalized needs the ground-truth map 'irradiance@{ex:g}' "
                        f"(this capture has none; re-run it with ground truth enabled)")


def _norm_ex(result, norm: NormSpec, ex: float, dark, white, sel) -> np.ndarray:
    """Normalised values (float64) of excitation ``ex``; ``sel = (rows, cols)`` index arrays -> shape (N, B),
    ``None`` -> (H, W, B)."""
    mode = norm.mode
    if mode == "adu":
        return _take(result.adu[ex], sel).astype(np.float64)
    if mode == "radiance":
        return _take(result.radiance[ex], sel).astype(np.float64)
    if mode == "electrons":
        em = getattr(result, "electrons_mean", None) or {}
        if ex not in em:
            raise AnalysisError("this capture carries no electron counts (re-export it with a newer engine)")
        return _take(em[ex], sel).astype(np.float64)
    if mode == "exposure_power":
        try:
            t_s = float(result.exposure_ms[ex]) / 1000.0
            p_mw = float(result.power_mW[ex])
        except KeyError as exc:
            raise AnalysisError("this capture carries no exposure / power metadata") from exc
        if not (t_s > 0 and p_mw > 0):
            raise AnalysisError(f"exposure_power needs exposure > 0 and power > 0 "
                                f"(got {t_s * 1000:g} ms, {p_mw:g} mW at {_key(ex)} nm); is this a dark frame?")
        return (_take(result.adu[ex], sel).astype(np.float64) - sensor_offset_adu(result)) / (t_s * p_mw)
    if mode == "irradiance_normalized":
        try:
            t_s = float(result.exposure_ms[ex]) / 1000.0
        except KeyError as exc:
            raise AnalysisError("this capture carries no exposure metadata") from exc
        if not t_s > 0:
            raise AnalysisError(f"irradiance_normalized needs exposure > 0 (got {t_s * 1000:g} ms at {_key(ex)} nm)")
        e_ex = _irradiance_map(result, ex)
        e_sel = np.asarray(e_ex, dtype=np.float64) if sel is None else np.asarray(e_ex, dtype=np.float64)[sel]
        sig = _take(result.adu[ex], sel).astype(np.float64) - sensor_offset_adu(result)
        out = np.full(sig.shape, np.nan)
        e_all = np.asarray(e_ex, dtype=np.float64)
        e_max = np.nanmax(e_all) if np.isfinite(e_all).any() else 0.0
        ok = (e_sel > IRRADIANCE_FLOOR * e_max) & np.isfinite(e_sel)
        np.divide(sig, (t_s * e_sel)[..., None], out=out, where=ok[..., None])
        return out
    if mode in ("dark_subtracted", "white_normalized"):
        s = _take(result.adu[ex], sel).astype(np.float64)
        if dark is None and mode == "dark_subtracted":
            raise AnalysisError("dark_subtracted needs a dark capture (norm.dark_capture_id)")
        d: np.ndarray | float
        if dark is not None:
            d = _take(dark.adu[find_ex(dark, ex)], sel).astype(np.float64)
        else:
            d = sensor_offset_adu(result)
        if mode == "dark_subtracted":
            return s - d
        if white is None:
            raise AnalysisError("white_normalized needs a white-reference capture (norm.white_capture_id)")
        w = _take(white.adu[find_ex(white, ex)], sel).astype(np.float64) - d
        num = s - d
        out = np.full(num.shape, np.nan)
        ok = np.abs(w) > 1e-6
        np.divide(num, w, out=out, where=ok)
        return out
    raise AnalysisError(f"unknown normalisation mode '{mode}'")


def _validate_norm_inputs(result, norm: NormSpec, dark, white) -> None:
    if norm.mode in ("dark_subtracted", "white_normalized") and dark is not None:
        check_compatible(result, dark, "dark")
    if norm.mode == "white_normalized":
        if white is None:
            raise AnalysisError("white_normalized needs a white-reference capture (norm.white_capture_id)")
        check_compatible(result, white, "white")
    if norm.mode == "dark_subtracted" and dark is None:
        raise AnalysisError("dark_subtracted needs a dark capture (norm.dark_capture_id)")


def normalize(result, norm=None, *, dark=None, white=None) -> tuple[dict[float, np.ndarray], str]:
    """Normalised cubes for every excitation -> ``({ex: (H, W, B) float32}, unit)`` (see module docstring).

    ``dark`` / ``white`` are CaptureResults; they must match the sample's size, excitations and bands."""
    norm = coerce_norm(norm)
    _validate_norm_inputs(result, norm, dark, white)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {ex: _norm_ex(result, norm, ex, dark, white, None).astype(np.float32) for ex in result.excitations_nm}
    return out, UNITS[norm.mode]


# --------------------------------------------------------------------------------------------
# ROI masks
# --------------------------------------------------------------------------------------------
def roi_mask(spec, height: int, width: int) -> np.ndarray:
    """Boolean (H, W) mask of the pixels whose centres lie in the ROI (may be empty)."""
    roi = coerce_roi(spec)
    mask = np.zeros((height, width), dtype=bool)

    def box(x0: float, x1: float, y0: float, y1: float):
        c0, c1 = max(int(np.floor(x0)), 0), min(int(np.ceil(x1)), width - 1)
        r0, r1 = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)), height - 1)
        if c0 > c1 or r0 > r1:
            return None
        yy, xx = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        return (slice(r0, r1 + 1), slice(c0, c1 + 1)), xx.astype(np.float64), yy.astype(np.float64)

    if isinstance(roi, PointRoi):
        c, r = int(np.floor(roi.x + 0.5)), int(np.floor(roi.y + 0.5))
        if 0 <= c < width and 0 <= r < height:
            mask[r, c] = True
        return mask
    if isinstance(roi, DiskRoi):
        b = box(roi.x - roi.radius_px, roi.x + roi.radius_px, roi.y - roi.radius_px, roi.y + roi.radius_px)
        if b:
            sl, xx, yy = b
            mask[sl] = (xx - roi.x) ** 2 + (yy - roi.y) ** 2 <= roi.radius_px ** 2 + 1e-9
    elif isinstance(roi, RectRoi):
        xa, xb = sorted((roi.x0, roi.x1))
        ya, yb = sorted((roi.y0, roi.y1))
        b = box(xa, xb, ya, yb)
        if b:
            sl, xx, yy = b
            mask[sl] = (xx >= xa - 1e-9) & (xx <= xb + 1e-9) & (yy >= ya - 1e-9) & (yy <= yb + 1e-9)
    elif isinstance(roi, EllipseRoi):
        b = box(roi.cx - roi.rx, roi.cx + roi.rx, roi.cy - roi.ry, roi.cy + roi.ry)
        if b:
            sl, xx, yy = b
            mask[sl] = ((xx - roi.cx) / roi.rx) ** 2 + ((yy - roi.cy) / roi.ry) ** 2 <= 1.0 + 1e-9
    else:  # polygon, even-odd rule on pixel centres
        pts = np.asarray(roi.points, dtype=np.float64)
        b = box(pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        if b:
            sl, xx, yy = b
            inside = np.zeros(xx.shape, dtype=bool)
            xj, yj = pts[-1]
            for xi, yi in pts:
                cross = (yi > yy) != (yj > yy)
                with np.errstate(divide="ignore", invalid="ignore"):
                    x_int = (xj - xi) * (yy - yi) / (yj - yi) + xi
                inside ^= cross & (xx < x_int)
                xj, yj = xi, yi
            mask[sl] = inside
    return mask


def _roi_pixels(result, roi) -> tuple[BaseModel, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    spec = coerce_roi(roi)
    mask = roi_mask(spec, result.height, result.width)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise AnalysisError(f"the ROI contains no pixels of the {result.width}x{result.height} image "
                            "(outside the image, or smaller than a pixel)")
    return spec, mask, (rows, cols)


# --------------------------------------------------------------------------------------------
# EEM / ROI statistics
# --------------------------------------------------------------------------------------------
def _spec_dict(spec: BaseModel) -> dict:
    return spec.model_dump(mode="python")


@dataclass
class EemResult:
    roi: dict
    n_pixels: int
    norm: NormSpec
    unit: str
    excitations_nm: list[float]
    emission_nm: dict[float, np.ndarray]
    value: dict[float, np.ndarray]
    std: dict[float, np.ndarray]
    grid_emission_nm: np.ndarray
    grid_value: np.ndarray                      # (n_ex, n_grid), NaN where a band is absent

    def to_dict(self) -> dict:
        return to_jsonable({
            "roi": self.roi, "n_pixels": self.n_pixels, "norm": self.norm.model_dump(), "unit": self.unit,
            "excitations_nm": self.excitations_nm, "emission_nm": self.emission_nm, "value": self.value,
            "std": self.std, "grid_emission_nm": self.grid_emission_nm, "grid_value": self.grid_value})


@dataclass
class RoiStats(EemResult):
    ground_truth: dict[str, dict[str, float]] = field(default_factory=dict)
    components: dict[str, dict[str, np.ndarray]] | None = None

    def to_dict(self) -> dict:
        d = EemResult.to_dict(self)
        d["ground_truth"] = to_jsonable(self.ground_truth)
        d["components"] = to_jsonable(self.components)
        return d


def _stat(a: np.ndarray, stat: str) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mid = np.nanmedian(a, axis=0) if stat == "median" else np.nanmean(a, axis=0)
        return mid, np.nanstd(a, axis=0)


def emission_grid(result, step: float = GRID_STEP_NM) -> np.ndarray:
    """Union-span emission grid of all excitations of ``result`` (``step`` nm, starting at the first band)."""
    lo = min(float(np.min(result.emission_nm[ex])) for ex in result.excitations_nm)
    hi = max(float(np.max(result.emission_nm[ex])) for ex in result.excitations_nm)
    return lo + step * np.arange(int(np.floor((hi - lo) / step + 1e-9)) + 1)


def eem(result, roi, norm=None, stat: str = "mean", *, dark=None, white=None) -> EemResult:
    """Excitation-emission matrix of a pixel / disk / ROI: per excitation and band the ``stat`` ('mean' or
    'median') and std over the ROI pixels, plus a common 2 nm emission grid for heatmaps (linear interpolation of
    each excitation's bands, NaN outside its band span)."""
    if stat not in ("mean", "median"):
        raise AnalysisError("stat must be 'mean' or 'median'")
    spec, _mask, sel = _roi_pixels(result, roi)
    nspec = coerce_norm(norm)
    _validate_norm_inputs(result, nspec, dark, white)
    grid = emission_grid(result)
    value, std, gval = {}, {}, np.full((len(result.excitations_nm), grid.size), np.nan)
    emission = {}
    for i, ex in enumerate(result.excitations_nm):
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = _norm_ex(result, nspec, ex, dark, white, sel)             # (N, B)
        value[ex], std[ex] = _stat(vals, stat)
        em = np.asarray(result.emission_nm[ex], dtype=np.float64)
        emission[ex] = em
        ok = np.isfinite(value[ex])
        if ok.sum() >= 1:
            inside = (grid >= em.min() - 1e-9) & (grid <= em.max() + 1e-9)
            if ok.sum() == 1:
                gval[i, inside] = value[ex][ok][0] if em.size == 1 else np.nan
            else:
                gval[i, inside] = np.interp(grid[inside], em[ok], value[ex][ok])
    return EemResult(roi=_spec_dict(spec), n_pixels=int(sel[0].size), norm=nspec, unit=UNITS[nspec.mode],
                     excitations_nm=[float(e) for e in result.excitations_nm], emission_nm=emission,
                     value=value, std=std, grid_emission_nm=grid, grid_value=gval)


def numeric_ground_truth(result) -> dict[str, np.ndarray]:
    """The (H, W) numeric ground-truth maps of ``result`` as float64 (strings / objects are skipped)."""
    out = {}
    for name, g in result.ground_truth.items():
        a = np.asarray(g)
        if a.shape == (result.height, result.width) and a.dtype.kind in "fiub":
            out[name] = a.astype(np.float64)
    return out


def _gt_map(result, name: str) -> np.ndarray:
    if name not in result.ground_truth:
        raise AnalysisError(f"unknown ground-truth layer '{name}' (available: {sorted(result.ground_truth)})")
    g = numeric_ground_truth(result).get(name)
    if g is None:
        raise AnalysisError(f"ground-truth layer '{name}' is not a numeric map")
    return g


def roi_stats(result, roi, norm=None, stat: str = "mean", *, components: bool = False,
              explain: Callable[[int, int], dict] | None = None, cache: dict | None = None,
              dark=None, white=None) -> RoiStats:
    """:func:`eem` plus ground-truth statistics inside the ROI (numeric layers; NaN-aware: layers with no finite
    value in the ROI are omitted) and, with ``components=True``, the decomposition averaged over at most
    ``MAX_COMPONENT_PIXELS`` evenly-subsampled ROI pixels.  ``explain(x, y)`` must return the
    ``PixelExplain``-like dict of ``capture.explain_pixel`` (its ``components``: ``{ex: {name: per-band}}``);
    ``cache`` (any dict) memoises it per pixel.  Components stay in radiance units (not normalised) and an
    absent component counts as zero for that pixel."""
    base = eem(result, roi, norm, stat, dark=dark, white=white)
    _spec, _mask, (rows, cols) = _roi_pixels(result, roi)
    gt = {}
    for name, g in numeric_ground_truth(result).items():
        v = g[rows, cols]
        v = v[np.isfinite(v)]
        if v.size:
            gt[name] = {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
    comps = None
    if components:
        if explain is None:
            raise AnalysisError("component decomposition is unavailable for this capture", 501)
        n = rows.size
        pick = np.unique(np.linspace(0, n - 1, min(n, MAX_COMPONENT_PIXELS)).round().astype(int))
        acc: dict[str, dict[str, np.ndarray]] = {}
        for k in pick:
            x, y = int(cols[k]), int(rows[k])
            if cache is not None and (x, y) in cache:
                c = cache[(x, y)]
            else:
                c = _normalise_components(explain(x, y).get("components") or {})
                if cache is not None:
                    cache[(x, y)] = c
            for exk, byname in c.items():
                for cname, arr in byname.items():
                    slot = acc.setdefault(exk, {})
                    slot[cname] = slot[cname] + arr if cname in slot else arr.copy()
        comps = {exk: {cname: arr / pick.size for cname, arr in byname.items()} for exk, byname in acc.items()}
    return RoiStats(**{f: getattr(base, f) for f in EemResult.__dataclass_fields__}, ground_truth=gt, components=comps)


def _normalise_components(raw: dict) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for exk, byname in raw.items():
        try:
            key = _key(float(exk))
        except (TypeError, ValueError):
            key = str(exk)
        out[key] = {str(name): np.asarray(arr, dtype=np.float64) for name, arr in byname.items()}
    return out


# --------------------------------------------------------------------------------------------
# Line profile
# --------------------------------------------------------------------------------------------
@dataclass
class LineProfile:
    distance_px: np.ndarray
    distance_mm: np.ndarray | None
    value: dict[float, np.ndarray]              # ex -> (n_points, n_bands)
    emission_nm: dict[float, np.ndarray]
    ground_truth: dict[str, np.ndarray]
    unit: str

    def to_dict(self) -> dict:
        return to_jsonable({"distance_px": self.distance_px, "distance_mm": self.distance_mm, "value": self.value,
                            "emission_nm": self.emission_nm, "ground_truth": self.ground_truth, "unit": self.unit})


def _bilinear_taps(x: np.ndarray, y: np.ndarray, h: int, w: int):
    """Corner indices / weights of bilinear sampling at (x, y); ``valid`` marks points inside the image."""
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    xc, yc = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(xc).astype(int), np.floor(yc).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = xc - x0, yc - y0
    rows = np.stack([y0, y0, y1, y1], axis=-1)
    cols = np.stack([x0, x1, x0, x1], axis=-1)
    wts = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=-1)
    return rows, cols, wts, valid


def _combine(vals: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """vals (..., 4, B), wts (..., 4): weighted sum where zero-weight taps are ignored even if NaN."""
    w = wts[..., None]
    with np.errstate(invalid="ignore"):
        return np.where(w > 0, vals * w, 0.0).sum(axis=-2)


def _camera_geometry(result) -> tuple[float, float] | None:
    """(effective pixel pitch [mm] of this capture, focal length [mm]) or None."""
    try:
        exp = json.loads(result.experiment_json)
        cams = exp["cameras"]
        cam = next((c for c in cams if c["id"] == exp.get("active_camera_id")), cams[0])
        s = cam["sensor"]
        pitch = float(s["pixel_pitch_um"]) * float(s["binning"]) * 1e-3 * float(s["width_px"]) / result.width
        return pitch, float(cam["lens"]["focal_length_mm"])
    except Exception:  # noqa: BLE001 - distance in mm is optional
        return None


def line_profile(result, p0, p1, *, width_px: float = 1.0, norm=None, excitations_nm=None, ground_truth=None,
                 dark=None, white=None) -> LineProfile:
    """Bilinear samples along ``p0 -> p1`` (pixel coordinates), spacing <= 1 px, averaged over ``round(width_px)``
    parallel lines spread across ``width_px`` perpendicular to the line.  Samples outside the image are NaN.
    ``ground_truth`` layers are sampled along the centre line (NaN if any contributing neighbour is NaN).
    ``distance_mm``: cumulative distance on the first surface = sum of ``depth_mm x pixel_pitch / focal_length``
    per step (small-angle pixel footprint; missing depths are interpolated); ``None`` if no depth is finite."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    if p0.shape != (2,) or p1.shape != (2,) or not np.all(np.isfinite([*p0, *p1])):
        raise AnalysisError("p0 and p1 must be finite [x, y] pairs")
    if not width_px >= 1e-9 or width_px > max(result.width, result.height):
        raise AnalysisError("width_px must be positive and no larger than the image")
    length = float(np.hypot(*(p1 - p0)))
    if length < 1e-9:
        raise AnalysisError("p0 and p1 coincide")
    nspec = coerce_norm(norm)
    _validate_norm_inputs(result, nspec, dark, white)
    exs = [find_ex(result, e) for e in excitations_nm] if excitations_nm else list(result.excitations_nm)
    n = max(2, int(np.ceil(length)) + 1)
    t = np.linspace(0.0, 1.0, n)
    ux, uy = (p1 - p0) / length
    nw = max(1, int(round(width_px)))
    offs = np.arange(nw) - (nw - 1) / 2.0 if nw > 1 else np.zeros(1)
    offs = offs * (width_px / nw) if nw > 1 else offs
    x = p0[0] + t[None, :] * (p1[0] - p0[0]) - uy * offs[:, None]          # (nw, n)
    y = p0[1] + t[None, :] * (p1[1] - p0[1]) + ux * offs[:, None]
    rows, cols, wts, valid = _bilinear_taps(x, y, result.height, result.width)

    value, emission = {}, {}
    for ex in exs:
        with np.errstate(invalid="ignore", divide="ignore"):
            v = _norm_ex(result, nspec, ex, dark, white, (rows.ravel(), cols.ravel()))     # (nw*n*4, B)
        v = v.reshape(nw, n, 4, -1)
        samp = _combine(v, wts)                                            # (nw, n, B)
        samp = np.where(valid[..., None], samp, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value[ex] = np.nanmean(samp, axis=0)
        emission[ex] = np.asarray(result.emission_nm[ex], dtype=np.float64)

    cr, cc, cw, cv = _bilinear_taps(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]),
                                    result.height, result.width)

    def along(g: np.ndarray) -> np.ndarray:
        s = _combine(g[cr, cc][..., None], cw)[..., 0]
        return np.where(cv, s, np.nan)

    gt = {name: along(_gt_map(result, name)) for name in (ground_truth or [])}
    dist_px = t * length
    dist_mm = None
    geo = _camera_geometry(result)
    depth = result.ground_truth.get("depth_mm")
    if geo is not None and depth is not None and np.asarray(depth).shape == (result.height, result.width):
        d = along(np.asarray(depth, dtype=np.float64))
        d = np.where(np.isfinite(d), d, np.nan)
        if np.isfinite(d).any():
            idx = np.arange(n)
            d = np.interp(idx, idx[np.isfinite(d)], d[np.isfinite(d)])
            step_mm = 0.5 * (d[1:] + d[:-1]) * geo[0] / geo[1] * np.diff(dist_px)
            dist_mm = np.concatenate([[0.0], np.cumsum(step_mm)])
    return LineProfile(distance_px=dist_px, distance_mm=dist_mm, value=value, emission_nm=emission,
                       ground_truth=gt, unit=UNITS[nspec.mode])


# --------------------------------------------------------------------------------------------
# Response curve
# --------------------------------------------------------------------------------------------
@dataclass
class ResponseCurve:
    x_ground_truth: str
    metric: dict
    bin_centres: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    scatter_x: np.ndarray
    scatter_y: np.ndarray
    unit: str

    def to_dict(self) -> dict:
        return to_jsonable({"x_ground_truth": self.x_ground_truth, "metric": self.metric,
                            "bin_centres": self.bin_centres, "mean": self.mean, "std": self.std,
                            "count": self.count, "scatter_x": self.scatter_x, "scatter_y": self.scatter_y,
                            "unit": self.unit})


def _nearest_band(result, ex: float, band_nm: float) -> int:
    em = np.asarray(result.emission_nm[ex], dtype=np.float64)
    return int(np.argmin(np.abs(em - band_nm)))


def peak_wavelength(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Per-row peak position of (N, B) spectra: argmax refined by a parabola through the 3 points around it
    (offset clamped to +-0.5 band; end bands are not refined).  Rows with no finite value give NaN."""
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(centres, dtype=np.float64)
    filled = np.where(np.isfinite(v), v, -np.inf)
    k = filled.argmax(axis=1)
    n = np.arange(v.shape[0])
    out = c[k].astype(np.float64)
    if c.size >= 3:
        inner = (k > 0) & (k < c.size - 1)
        km, kp = np.clip(k - 1, 0, c.size - 1), np.clip(k + 1, 0, c.size - 1)
        ym, y0, yp = v[n, km], v[n, k], v[n, kp]
        den = ym - 2 * y0 + yp
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = np.where(inner & np.isfinite(den) & (den < 0), 0.5 * (ym - yp) / den, 0.0)
        delta = np.clip(np.nan_to_num(delta), -0.5, 0.5)
        step = np.where(delta >= 0, c[kp] - c[k], c[k] - c[km])
        out = out + delta * step
    out[~np.isfinite(v).any(axis=1)] = np.nan
    return out


def metric_values(result, metric, norm: NormSpec, sel, dark=None, white=None) -> tuple[np.ndarray, str]:
    """Per-pixel metric at the pixels ``sel`` -> ((N,) float64, unit)."""
    m = coerce_metric(metric)
    unit = UNITS[norm.mode]

    def cube(ex):
        with np.errstate(invalid="ignore", divide="ignore"):
            return _norm_ex(result, norm, ex, dark, white, sel)

    if isinstance(m, MetricBand):
        ex = find_ex(result, m.ex)
        return cube(ex)[:, _nearest_band(result, ex, m.band_nm)], unit
    if isinstance(m, MetricBandRatio):
        ea, eb = find_ex(result, m.a.ex), find_ex(result, m.b.ex)
        a = cube(ea)[:, _nearest_band(result, ea, m.a.band_nm)]
        b = cube(eb)[:, _nearest_band(result, eb, m.b.band_nm)]
        out = np.full(a.shape, np.nan)
        np.divide(a, b, out=out, where=np.isfinite(b) & (b != 0))
        return out, "ratio"
    if isinstance(m, MetricBandSum):
        ex = find_ex(result, m.ex)
        em = np.asarray(result.emission_nm[ex], dtype=np.float64)
        lo = -np.inf if m.from_nm is None else m.from_nm
        hi = np.inf if m.to_nm is None else m.to_nm
        if lo > hi:
            raise AnalysisError("band_sum: from_nm must be <= to_nm")
        inside = (em >= lo) & (em <= hi)
        if not inside.any():
            raise AnalysisError(f"band_sum: no band of the {_key(ex)} nm excitation lies in "
                                f"[{lo:g}, {hi:g}] nm (bands span {em.min():g}-{em.max():g} nm)")
        return cube(ex)[:, inside].sum(axis=1), unit
    ex = find_ex(result, m.ex)
    return peak_wavelength(cube(ex), result.emission_nm[ex]), "nm"


def response_curve(result, x_ground_truth: str, metric, roi=None, bins: int = 24, norm=None, *,
                   dark=None, white=None) -> ResponseCurve:
    """Binned mean / std / count of a per-pixel ``metric`` against a ground-truth map (e.g.
    ``layer_thickness:cover``), over the ROI (or every pixel where the map is finite), plus a deterministic
    scatter subsample of at most 4000 points."""
    if not 1 <= int(bins) <= 1000:
        raise AnalysisError("bins must be between 1 and 1000")
    bins = int(bins)
    nspec = coerce_norm(norm)
    _validate_norm_inputs(result, nspec, dark, white)
    mspec = coerce_metric(metric)
    xmap = _gt_map(result, x_ground_truth)
    if roi is None:
        keep = np.isfinite(xmap)
    else:
        _s, rmask, _sel = _roi_pixels(result, roi)
        keep = rmask & np.isfinite(xmap)
    rows, cols = np.nonzero(keep)
    if rows.size == 0:
        raise AnalysisError(f"no pixel with a finite '{x_ground_truth}' value in the region")
    y, unit = metric_values(result, mspec, nspec, (rows, cols), dark, white)
    x = xmap[rows, cols]
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        raise AnalysisError("the metric is not finite at any pixel of the region")
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    count = np.bincount(which, minlength=bins).astype(np.int64)
    s1 = np.bincount(which, weights=y, minlength=bins)
    s2 = np.bincount(which, weights=y * y, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, s1 / count, np.nan)
        var = np.where(count > 0, s2 / count - mean ** 2, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    if x.size > MAX_SCATTER_POINTS:
        pick = np.sort(np.random.default_rng(0).choice(x.size, MAX_SCATTER_POINTS, replace=False))
        sx, sy = x[pick], y[pick]
    else:
        sx, sy = x, y
    return ResponseCurve(x_ground_truth=x_ground_truth, metric=mspec.model_dump(), bin_centres=0.5 * (edges[1:] + edges[:-1]),
                         mean=mean, std=std, count=count, scatter_x=sx, scatter_y=sy, unit=unit)


# --------------------------------------------------------------------------------------------
# Volumes (3D light fields of layered blocks)
# --------------------------------------------------------------------------------------------
def _block_names(result) -> dict[str, dict]:
    return {str(b["object_id"]): b for b in (getattr(result, "blocks", None) or [])}


def volume_infos(result) -> list[dict]:
    """``VolumeInfo`` dicts (api/types.ts) for every tallied volume of the capture."""
    out = []
    for oid, vol in (getattr(result, "volumes", None) or {}).items():
        grid = vol["grid"]
        names = list(vol.get("layer_names") or _block_names(result).get(str(oid), {}).get("layer_names", []))
        units = vol.get("units", {}) or {}
        for kind in VOLUME_KINDS:
            tallies = vol.get(kind) or {}
            if not tallies:
                continue
            if kind == "source":
                exs = sorted({float(k[0]) for k in tallies})
                channels = sorted({str(k[1]) for k in tallies})
                shape = np.asarray(next(iter(tallies.values()))).shape
            else:
                exs = sorted(float(k) for k in tallies)
                channels = []
                shape = np.asarray(next(iter(tallies.values()))).shape
            nx, nz = int(shape[0]), int(shape[1])
            nd = int(np.prod(shape[2:]))
            out.append({"object_id": str(oid), "kind": kind, "excitations_nm": exs, "channels": channels,
                        "dims": [nx, nz, nd], "x_edges_mm": [float(v) for v in np.asarray(grid.x_edges_mm)],
                        "z_edges_mm": [float(v) for v in np.asarray(grid.z_edges_mm)], "layer_names": names,
                        "sublayers": int(grid.sublayers), "unit": str(units.get(kind, ""))})
    return out


def volume(result, object_id: str, kind: str, ex: float | None = None, channel: str | None = None) -> np.ndarray:
    """One tally volume as float32 ``(nx, nz, n_layers*sublayers)`` C-order (layer-major, sublayer fastest;
    layers bottom -> top).  ``ex`` defaults to the first available excitation; ``kind='source'`` also needs
    ``channel`` (defaults to the first)."""
    vols = getattr(result, "volumes", None) or {}
    if object_id not in vols:
        raise AnalysisError(f"no 3D volumes for object '{object_id}' (have {sorted(vols)})", 404)
    if kind not in VOLUME_KINDS:
        raise AnalysisError(f"kind must be one of {VOLUME_KINDS}, got '{kind}'")
    tallies = vols[object_id].get(kind) or {}
    if not tallies:
        raise AnalysisError(f"object '{object_id}' has no '{kind}' volume", 404)
    if kind == "source":
        avail = sorted({(float(k[0]), str(k[1])) for k in tallies})
        exs = sorted({a for a, _ in avail})
        e = exs[0] if ex is None else min(exs, key=lambda v: abs(v - ex))
        if ex is not None and abs(e - ex) > 0.5:
            raise AnalysisError(f"excitation {ex:g} nm not available for the source volume ({exs})", 404)
        chans = sorted({c for a, c in avail if abs(a - e) < 1e-6})
        ch = chans[0] if channel is None else channel
        if ch not in chans:
            raise AnalysisError(f"unknown source channel '{ch}' at {e:g} nm (have {chans})", 404)
        key = next(k for k in tallies if abs(float(k[0]) - e) < 1e-6 and str(k[1]) == ch)
    else:
        exs = sorted(float(k) for k in tallies)
        e = exs[0] if ex is None else min(exs, key=lambda v: abs(v - ex))
        if ex is not None and abs(e - ex) > 0.5:
            raise AnalysisError(f"excitation {ex:g} nm not available for the {kind} volume ({exs})", 404)
        key = next(k for k in tallies if abs(float(k) - e) < 1e-6)
    a = np.asarray(tallies[key], dtype=np.float32)
    return np.ascontiguousarray(a.reshape(a.shape[0], a.shape[1], -1))


# --------------------------------------------------------------------------------------------
# Offline loading of the npz export
# --------------------------------------------------------------------------------------------
_NPZ_PER_EX = {"adu": "adu", "radiance": "radiance", "electrons": "electrons_mean", "emission_nm": "emission_nm",
               "exposure_ms": "exposure_ms", "power_mW": "power_mW"}
_NPZ_RE = re.compile(r"^(adu|radiance|electrons|emission_nm|exposure_ms|power_mW)_(-?[0-9.eE+]+)$")


def load_npz_capture(path) -> CaptureResult:
    """Re-create a :class:`CaptureResult` from an ``npz`` export (``exports.to_npz``), including ground-truth maps
    and layered-block volumes, so every function of this module works offline."""
    with np.load(path, allow_pickle=False) as z:
        files = list(z.files)
        if "experiment_json" not in files or "excitations_nm" not in files:
            raise AnalysisError("not a SpectralForge Studio npz export (missing experiment_json / excitations_nm)")
        meta = json.loads(str(z["meta_json"])) if "meta_json" in files else {}
        exs = [float(e) for e in z["excitations_nm"]]
        per: dict[str, dict[float, Any]] = {k: {} for k in _NPZ_PER_EX}
        for name in files:
            m = _NPZ_RE.match(name)
            if m:
                per[m.group(1)][float(m.group(2))] = z[name]
        if not per["adu"]:
            raise AnalysisError("npz export contains no adu cubes")
        h, w = per["adu"][exs[0]].shape[:2]
        gt_names = meta.get("ground_truth_names", {})
        gt = {gt_names.get(n[3:], n[3:]): z[n] for n in files if n.startswith("gt_")}
        volumes = _load_volumes(z, files)
        blocks = json.loads(str(z["blocks_json"])) if "blocks_json" in files else []
        return CaptureResult(
            experiment_json=str(z["experiment_json"]), mode=meta.get("mode", "sample"), width=int(w), height=int(h),
            excitations_nm=exs, emission_nm={e: np.asarray(per["emission_nm"][e], dtype=np.float64) for e in exs},
            adu={e: per["adu"][e] for e in exs}, radiance={e: per["radiance"][e] for e in exs if e in per["radiance"]},
            electrons_mean={e: per["electrons"][e] for e in exs if e in per["electrons"]},
            exposure_ms={e: float(per["exposure_ms"][e]) for e in exs if e in per["exposure_ms"]},
            power_mW={e: float(per["power_mW"][e]) for e in exs if e in per["power_mW"]},
            ground_truth=gt, warnings=list(meta.get("warnings", [])), timings_s=dict(meta.get("timings_s", {})),
            engine_version=meta.get("engine_version", "studio-1.1"), blocks=blocks, volumes=volumes)


def _load_volumes(z, files: list[str]) -> dict[str, dict]:
    vols: dict[str, dict] = {}
    for name in files:
        if not name.startswith("volume_meta/"):
            continue
        m = json.loads(str(z[name]))
        oid, safe = m["object_id"], m["safe_name"]
        x_e, z_e = np.asarray(m["x_edges_mm"]), np.asarray(m["z_edges_mm"])
        entry: dict[str, Any] = {"layer_names": m["layer_names"], "units": m.get("units", {}),
                                 "fluence": {}, "absorbed": {}, "source": {}}
        n_layers = len(m["layer_names"])
        entry["grid"] = VoxelGrid(nx=len(x_e) - 1, nz=len(z_e) - 1, n_layers=n_layers, sublayers=int(m["sublayers"]),
                                  x_edges_mm=x_e, z_edges_mm=z_e)
        prefix = f"volume/{safe}/"
        for fn in files:
            if not fn.startswith(prefix):
                continue
            parts = fn[len(prefix):].split("/")
            kind, ex = parts[0], float(parts[1])
            if kind == "source":
                entry["source"][(ex, m.get("channels", {}).get(parts[2], parts[2]))] = z[fn]
            else:
                entry[kind][ex] = z[fn]
        vols[oid] = entry
    return vols
