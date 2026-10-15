"""Real ME-HSI measurements as a first-class object comparable to a simulated capture.

The lab stores each sample as a pickled :class:`spectral_select.types.SpectraData`: one cube
``(H, W, B_ex)`` per excitation line (each with its *own* emission-band list, because the Nuance
protocol starts the band list at ``max(420 nm, excitation + offset)`` and drops bands inside the
Rayleigh cut-off), optional exposure / lamp-power metadata, a 0/1 mask of the valid sample pixels and
ROI annotations (``roi_regions.json`` rectangles or ``class_mask.png`` colour classes, or an integer
label image for the droplet data).

:class:`RealCube` holds exactly that and :func:`real_roi_eem` reduces it to the *same*
:class:`spectraforge.studio.analysis.EemResult` structure that :func:`analysis.eem` returns for a simulated
:class:`~spectraforge.studio.contracts.CaptureResult` (same 2 nm union emission grid, ``NaN`` outside
each excitation's band span), so a measured and a simulated EEM can be subtracted, plotted on one
heat map or fed to :mod:`spectraforge.studio.calibration` without any glue code.

Units
-----
The processed lab data are *relative*: ``rad_*`` droplet cubes are ``(ADU - dark) / (exposure * lamp
power)`` followed by a single global scale; the collagen cubes are exposure(+power) normalised with a
per-dataset reference.  The unit string is recorded in :attr:`RealCube.units` and the calibration
therefore never assumes an absolute scale (it fits a free global or per-excitation gain, see
:mod:`~spectraforge.studio.calibration`).

Nothing here copies measurement files: arrays are read from the path given by the caller.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spectraforge.studio import analysis
from spectraforge.studio.analysis import EemResult, NormSpec

GRID_STEP_NM = analysis.GRID_STEP_NM


# --------------------------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------------------------
@dataclass
class RealCube:
    """A measured multi-excitation hyperspectral sample.

    ``cubes[ex]`` is ``(H, W, B_ex)`` (NaN allowed), ``emission_nm[ex]`` its ``(B_ex,)`` band centres.
    ``mask`` marks valid sample pixels (``True`` = keep).  ``rois`` maps a name to a boolean ``(H, W)``
    image.  ``exposure_ms`` / ``power`` are the acquisition metadata if the file carried them (they have
    usually already been divided out of the cubes: see ``units``)."""

    name: str
    excitations_nm: list[float]
    emission_nm: dict[float, np.ndarray]
    cubes: dict[float, np.ndarray]
    mask: np.ndarray | None = None
    rois: dict[str, np.ndarray] = field(default_factory=dict)
    exposure_ms: dict[float, float] = field(default_factory=dict)
    power: dict[float, float] = field(default_factory=dict)
    units: str = "relative (processed)"
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- shape -----------------------------------------------------------------------------------
    @property
    def height(self) -> int:
        return int(self.cubes[self.excitations_nm[0]].shape[0])

    @property
    def width(self) -> int:
        return int(self.cubes[self.excitations_nm[0]].shape[1])

    @property
    def n_bands(self) -> dict[float, int]:
        return {ex: int(self.cubes[ex].shape[2]) for ex in self.excitations_nm}

    def subset(self, excitations_nm) -> "RealCube":
        """A copy restricted to ``excitations_nm`` (cubes are shared, not copied)."""
        exs = [self._nearest(e) for e in excitations_nm]
        return RealCube(self.name, exs, {e: self.emission_nm[e] for e in exs}, {e: self.cubes[e] for e in exs},
                        self.mask, dict(self.rois), {e: v for e, v in self.exposure_ms.items() if e in exs},
                        {e: v for e, v in self.power.items() if e in exs}, self.units, self.source_path,
                        dict(self.metadata))

    def _nearest(self, ex: float) -> float:
        best = min(self.excitations_nm, key=lambda e: abs(e - ex))
        if abs(best - ex) > 0.5:
            raise KeyError(f"excitation {ex:g} nm is not in {self.excitations_nm}")
        return best

    # -- ROIs ------------------------------------------------------------------------------------
    def roi_names(self) -> list[str]:
        return list(self.rois)

    def roi_pixels(self, roi) -> np.ndarray:
        """Boolean ``(H, W)`` image of an ROI given as a registered name / class id, a boolean mask or an
        ``analysis`` ROI spec (``{"shape": "rect", ...}``).  The sample mask is *not* applied here."""
        if isinstance(roi, np.ndarray) and roi.dtype == bool:
            m = roi
        elif isinstance(roi, (str, int, np.integer)):
            key = str(roi)
            if key not in self.rois:
                raise KeyError(f"unknown ROI '{key}', known: {self.roi_names()}")
            m = self.rois[key]
        else:
            m = analysis.roi_mask(analysis.coerce_roi(roi), self.height, self.width)
        if m.shape != (self.height, self.width):
            raise ValueError(f"ROI mask {m.shape} does not match the image {(self.height, self.width)}")
        return m

    def roi_mean_spectra(self, roi, stat: str = "mean") -> dict[float, np.ndarray]:
        """Per-excitation ROI statistic ``{ex: (B_ex,)}`` (NaN-aware, sample mask applied)."""
        m = self._effective(roi)
        out = {}
        for ex in self.excitations_nm:
            sub = self.cubes[ex][m]
            with np.errstate(all="ignore"):
                out[ex] = np.nanmedian(sub, axis=0) if stat == "median" else np.nanmean(sub, axis=0)
        return out

    def _effective(self, roi) -> np.ndarray:
        m = self.roi_pixels(roi)
        if self.mask is not None:
            m = m & (self.mask > 0)
        if not m.any():
            raise ValueError("the ROI contains no valid pixel")
        return m

    def as_spectra_data(self):
        """Back-conversion to :class:`spectral_select.types.SpectraData` (for the selection pipeline)."""
        from spectral_select.types import ExcitationData, SpectraData
        ex = {e: ExcitationData(excitation_nm=e, cube=self.cubes[e], emission_wavelengths=list(self.emission_nm[e]),
                                exposure_time=self.exposure_ms.get(e), laser_power=self.power.get(e))
              for e in self.excitations_nm}
        return SpectraData(excitations=ex, mask=self.mask, sample_name=self.name, metadata=dict(self.metadata))


# --------------------------------------------------------------------------------------------
# adapters / loaders
# --------------------------------------------------------------------------------------------
def from_spectra_data(sd, rois: dict[str, np.ndarray] | None = None, *, name: str | None = None,
                      units: str = "relative (processed)") -> RealCube:
    """Adapt a :class:`spectral_select.types.SpectraData` (duck-typed: ``excitations`` dict of objects with
    ``cube`` / ``emission_wavelengths`` / ``exposure_time`` / ``laser_power``, optional ``mask``)."""
    exs = sorted(float(e) for e in sd.excitations)
    cubes, em, expo, pw = {}, {}, {}, {}
    for e in exs:
        ed = sd.excitations[e] if e in sd.excitations else next(v for k, v in sd.excitations.items() if float(k) == e)
        cubes[e] = np.asarray(ed.cube, dtype=np.float32)
        em[e] = np.asarray(ed.emission_wavelengths, dtype=float)
        if getattr(ed, "exposure_time", None) is not None:
            expo[e] = float(ed.exposure_time)
        if getattr(ed, "laser_power", None) is not None:
            pw[e] = float(ed.laser_power)
    mask = getattr(sd, "mask", None)
    mask = None if mask is None else np.asarray(mask) > 0
    return RealCube(name=name or getattr(sd, "sample_name", "sample"), excitations_nm=exs, emission_nm=em,
                    cubes=cubes, mask=mask, rois={str(k): np.asarray(v, bool) for k, v in (rois or {}).items()},
                    exposure_ms=expo, power=pw, units=units, metadata=dict(getattr(sd, "metadata", {}) or {}))


def rois_from_json(path: Path, height: int, width: int) -> dict[str, np.ndarray]:
    """``roi_regions.json`` -> ``{class_id: bool (H, W)}``; several rectangles of one class are merged."""
    spec = json.loads(Path(path).read_text())
    out: dict[str, np.ndarray] = {}
    for r in spec.get("regions", []):
        rc = r["rect"]
        m = out.setdefault(str(r["class_id"]), np.zeros((height, width), bool))
        m[int(rc["row_min"]):int(rc["row_max"]), int(rc["col_min"]):int(rc["col_max"])] = True
    return out


def rois_from_labels(labels: np.ndarray, background: int = 0) -> dict[str, np.ndarray]:
    """Integer label image -> ``{label: bool}`` (``background`` skipped)."""
    return {str(int(v)): labels == v for v in np.unique(labels) if int(v) != background}


def _find_pickle(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("spectra_masked.pkl", "spectra_data.pkl"):
        if (path / name).exists():
            return path / name
    pk = sorted(path.glob("*.pkl"))
    if len(pk) == 1:
        return pk[0]
    raise FileNotFoundError(f"no unambiguous spectra pickle in {path} (found {[p.name for p in pk]})")


def load_real_cube(path, *, labels: str | Path | np.ndarray | None = None, roi_json: str | Path | None = None,
                   name: str | None = None, units: str = "relative (processed)") -> RealCube:
    """Load a processed lab sample.

    ``path`` is a ``SpectraData`` pickle or a folder holding ``spectra_masked.pkl`` / ``spectra_data.pkl``.
    ROIs are taken, in this order, from ``labels`` (an ``.npy`` path or integer array, 0 = background),
    from ``roi_json`` or the ``roi_regions.json`` next to the pickle.  Warnings about the file extension are
    suppressed (the lab's droplet pickles have none)."""
    import warnings

    from spectral_select.types import SpectraData

    path = Path(path)
    pkl = _find_pickle(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = SpectraData.from_pickle(pkl)
    h, w = sd.spatial_shape
    rois: dict[str, np.ndarray] = {}
    if labels is not None:
        lab = np.load(labels) if isinstance(labels, (str, Path)) else np.asarray(labels)
        if lab.shape != (h, w):
            raise ValueError(f"label image {lab.shape} does not match the cube {(h, w)}")
        rois = rois_from_labels(lab)
    else:
        rj = Path(roi_json) if roi_json else pkl.parent / "roi_regions.json"
        if rj.exists():
            rois = rois_from_json(rj, h, w)
    cube = from_spectra_data(sd, rois, name=name or (pkl.parent.name if pkl.stem in ("spectra_masked", "spectra_data")
                                                      else pkl.stem), units=units)
    cube.source_path = str(pkl)
    return cube


# --------------------------------------------------------------------------------------------
# EEM
# --------------------------------------------------------------------------------------------
def union_grid(cube: RealCube, step: float = GRID_STEP_NM) -> np.ndarray:
    """Union-span emission grid, constructed exactly like :func:`analysis.emission_grid`."""
    lo = min(float(np.min(cube.emission_nm[ex])) for ex in cube.excitations_nm)
    hi = max(float(np.max(cube.emission_nm[ex])) for ex in cube.excitations_nm)
    return lo + step * np.arange(int(np.floor((hi - lo) / step + 1e-9)) + 1)


def real_roi_eem(cube: RealCube, roi, stat: str = "mean", *, grid: np.ndarray | None = None) -> EemResult:
    """Excitation-emission matrix of a measured ROI, structure-identical to :func:`analysis.eem`.

    ``roi``: a registered name / class id, a boolean mask or an analysis ROI spec dict.  ``grid`` overrides
    the common emission grid (pass the simulated EEM's ``grid_emission_nm`` to make the two directly
    subtractable; the default is the 2 nm union grid of the real bands).  Values are NaN outside each
    excitation's band span."""
    if stat not in ("mean", "median"):
        raise analysis.AnalysisError("stat must be 'mean' or 'median'")
    m = cube._effective(roi)
    grid = union_grid(cube) if grid is None else np.asarray(grid, float)
    value, std, emission = {}, {}, {}
    gval = np.full((len(cube.excitations_nm), grid.size), np.nan)
    for i, ex in enumerate(cube.excitations_nm):
        sub = cube.cubes[ex][m].astype(np.float64)                               # (N, B)
        value[ex], std[ex] = analysis._stat(sub, stat)
        em = np.asarray(cube.emission_nm[ex], dtype=float)
        emission[ex] = em
        ok = np.isfinite(value[ex])
        if ok.sum() >= 2:
            inside = (grid >= em[ok].min() - 1e-9) & (grid <= em[ok].max() + 1e-9)
            gval[i, inside] = np.interp(grid[inside], em[ok], value[ex][ok])
    spec = {"shape": "mask", "name": str(roi) if isinstance(roi, (str, int, np.integer)) else "custom"}
    return EemResult(roi=spec, n_pixels=int(m.sum()), norm=NormSpec(mode="adu"), unit=cube.units,
                     excitations_nm=[float(e) for e in cube.excitations_nm], emission_nm=emission, value=value,
                     std=std, grid_emission_nm=grid, grid_value=gval)


def eem_arrays(e: EemResult) -> tuple[list[float], dict[float, np.ndarray], dict[float, np.ndarray]]:
    """``(excitations, band centres, ROI statistic)`` of an :class:`EemResult` (real or simulated)."""
    return list(e.excitations_nm), {k: np.asarray(v, float) for k, v in e.emission_nm.items()}, \
        {k: np.asarray(v, float) for k, v in e.value.items()}
