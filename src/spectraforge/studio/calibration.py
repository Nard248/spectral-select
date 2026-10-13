"""Data-driven fluorophore calibration of the digital twin of the ME-HSI lab.

Goal
----
Fit the parameters of one to three :class:`~spectraforge.studio.schema.FluorophoreSpec` (excitation / emission
peak and FWHM, red-tail skew, vibronic shoulder, relative brightness) so that the *simulated* excitation-
emission matrix (EEM) of an optically thin, dilute fluorescent layer seen through the digital-twin instrument
matches a *measured* ROI EEM (:func:`spectraforge.studio.realdata.real_roi_eem`).

Fast forward model
------------------
For a thin, optically dilute layer the engine's fluorescence radiance is (``capture.fluorescence_spectrum``,
Section "Equations" of ``optics.py``)::

    L_k(lam | ex) = E_ex <mu_a,k>_line Phi_k e_k(lam) G,        <mu_a,k>_line = sum_lam line_ex(lam) mu_a,k(lam) dlam

with ``mu_a,k(lam) = ln10 eps_k(lam) c_k / 10`` and ``G`` a wavelength-independent geometry factor (thin clear
layer: no inner filter, no re-absorption, Fresnel factors constant over the 100 nm emission range; verified
against a full capture below).  The detector electrons in band ``b`` are the band projection used by
``sensor.project_bands`` (pass-band x long-pass x peak transmission x QE x photon energy)::

    S[ex, b] = sum_k  a_k  A_k(ex)  M_k(ex, b),
    A_k(ex)  = sum_lam line_ex(lam) exc_k(lam) dlam                         (excitation overlap)
    M_k(ex,b) = sum_lam R_b(lam) e_k(lam) dlam                                (emission band projection)

where ``exc_k`` is the peak-normalised Gaussian excitation of :class:`spectraforge.Fluorophore`, ``e_k`` the
unit-integral emission density (split-Gaussian + vibronic shoulder), ``R_b`` ``BandResponses.el[0, b]`` and ``line_ex``
the area-normalised excitation line (``source.line_shape``).  ``a_k = c_k eps_k Phi_k ln10/10`` collapses all
amplitude factors; it is **not separable** into concentration, extinction and quantum yield from relative data,
so the calibrated :class:`FluorophoreSpec` carries nominal ``quantum_yield = 0.5`` / ``extinction = 1`` and the
amplitude is reported separately as ``relative_brightness`` (see :class:`FitResult`).  ``G`` and the lamp
power are folded into the gains below.

Gain models (the processed lab data are in *relative* units)
-------------------------------------------------------------
* ``gain="global"``: one scale ``s`` for the whole EEM (valid only if the excitations share a radiometric
  calibration, e.g. the ``rad_merged`` droplet cubes); the excitation parameters are then identifiable.
* ``gain="per_excitation"``: a free scale ``g_ex`` per excitation line (power meter / exposure units that differ
  between lines).  The excitation *position* of a one-component fit is then not identifiable and is held at its
  prior (``fit_ex = False``); the fit constrains the emission shape and, for >= 2 components, their
  relative contribution.

Both gains are eliminated in closed form (variable projection, Golub & Pereyra 1973, SIAM J. Numer. Anal. 10, 413),
so the optimiser only sees the shape parameters.

Objective
---------
Per excitation the residual is ``(y - g m) / rms(y)`` (``loss="normalized"``) or ``log y - log(g m)`` (``"log"``),
minimised with the soft-L1 robust loss (scipy ``least_squares`` ``loss='soft_l1'``) from several deterministic
random starts.  Parameter covariance: ``s^2 (J^T J)^+`` with ``s^2 = sum r^2 / (n - p)`` from the Jacobian of the
projected residual (Kaufman 1975 approximation, the gains are profiled out); ``ci95 = 1.96 sigma`` in natural
units.  These are *statistical* intervals of the ROI-mean EEM; the dominant error is systematic (see
``docs/studio/SIM2REAL.md``), so :func:`bootstrap_ci` (pixel bootstrap) is provided as well.

Optional instrument ripple
--------------------------
``ripple=True`` adds a multiplicative per-band-wavelength gain ``T(lam)=exp(P_perp u)`` shared by all excitations,
restricted to the subspace orthogonal to cubic polynomials in ``lam`` (so it can absorb the LCTF transmission
ripple but not the smooth emission shape), with a small ridge penalty.

References
----------
Lakowicz, *Principles of Fluorescence Spectroscopy*, 3rd ed. (Springer 2006), ch. 1-2 (Kasha rule, excitation
profile x emission shape); Golub & Pereyra 1973 (separable least squares); Kaufman 1975, BIT 15, 49; Jacques 2013,
Phys. Med. Biol. 58, R37 (tissue optics, for ``calibrate_scattering``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from spectraforge.studio import optics as optics_mod
from spectraforge.studio import sensor as sensor_mod
from spectraforge.studio import source as srcmod
from spectraforge.studio.analysis import EemResult, NormSpec
from spectraforge.studio.contracts import SPECTRAL_GRID_NM
from spectraforge.studio.schema import CameraSpec, FluorophoreSpec, LightSourceSpec, PhysicsToggles

DATA_DIR = Path(__file__).parent / "data"
CALIBRATED_JSON = DATA_DIR / "calibrated_fluorophores.json"

#: parameter name -> (lower, upper, typical scale)
PARAM_BOUNDS = {
    "ex_peak_nm": (250.0, 450.0, 30.0),
    "ex_fwhm_nm": (15.0, 200.0, 30.0),
    "em_peak_nm": (380.0, 700.0, 30.0),
    "em_fwhm_nm": (15.0, 200.0, 30.0),
    "em_skew": (0.0, 1.5, 0.3),
    "vibronic": (0.0, 0.6, 0.2),
}
_ALL = tuple(PARAM_BOUNDS)


# --------------------------------------------------------------------------------------------
# instrument
# --------------------------------------------------------------------------------------------
@dataclass
class TwinInstrument:
    """The instrument pieces of the digital twin that matter for a thin dilute layer: camera (LCTF pass-band,
    long-pass, QE, peak transmission) and light source (line position, width)."""

    camera: CameraSpec
    source: LightSourceSpec
    toggles: PhysicsToggles = field(default_factory=PhysicsToggles)
    grid: np.ndarray = field(default_factory=lambda: SPECTRAL_GRID_NM)

    @classmethod
    def lab_replica(cls) -> "TwinInstrument":
        """Tunable Xe/monochromator lamp (10 nm line) + Nuance LCTF, as in ``presets.mehsi_lab_replica``."""
        from spectraforge.studio import presets
        exp = presets.mehsi_lab_replica()
        return cls(exp.camera(), exp.source(), exp.acquisition.physics)

    @classmethod
    def from_experiment(cls, experiment) -> "TwinInstrument":
        return cls(experiment.camera(), experiment.source(), experiment.acquisition.physics)


# --------------------------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------------------------
class ThinLayerModel:
    """Fast forward model ``S[ex, b] = sum_k a_k A_k(ex) M_k(ex, b)`` (see module docstring).

    ``bands`` maps each excitation to its band centres (nm).  Responses are built once per instance."""

    def __init__(self, instrument: TwinInstrument, bands: dict[float, np.ndarray]):
        self.inst = instrument
        self.grid = np.asarray(instrument.grid, float)
        self.step = float(np.mean(np.diff(self.grid)))
        self.bands = {float(k): np.asarray(v, float) for k, v in bands.items()}
        self.excitations = sorted(self.bands)
        self.R: dict[float, np.ndarray] = {}
        self.line: dict[float, np.ndarray] = {}
        for ex in self.excitations:
            resp = sensor_mod.band_responses(instrument.camera, self.bands[ex], instrument.toggles, 0.0,
                                             grid=self.grid)
            self.R[ex] = resp.el[0].astype(np.float64)                      # (B, W)
            self.line[ex] = srcmod.line_shape(instrument.source, ex, self.grid)

    # -- building blocks -------------------------------------------------------------------------
    def excitation_overlap(self, spec: FluorophoreSpec, ex: float) -> float:
        """``A(ex) = sum line(lam) exc(lam) dlam`` (peak-normalised Gaussian excitation)."""
        return float(np.sum(self.line[ex] * optics_mod.to_legacy(spec).excitation(self.grid)) * self.step)

    def emission_projection(self, spec: FluorophoreSpec, ex: float) -> np.ndarray:
        """``M(ex, b) = sum R_b(lam) e(lam) dlam`` with ``e`` the unit-integral emission density (1/nm)."""
        e = optics_mod.to_legacy(spec).emission(self.grid) / self.step
        with np.errstate(all="ignore"):
            return self.R[ex] @ e * self.step

    def predict(self, specs: list[FluorophoreSpec], amplitudes=None) -> dict[float, np.ndarray]:
        """Noise-free, gain-free EEM ``{ex: (B,)}`` (electrons per unit ``a_k``)."""
        amp = np.ones(len(specs)) if amplitudes is None else np.asarray(amplitudes, float)
        out = {}
        for ex in self.excitations:
            out[ex] = sum(a * self.excitation_overlap(s, ex) * self.emission_projection(s, ex)
                          for a, s in zip(amp, specs))
        return out

    def as_eem(self, specs, amplitudes=None, gains: dict[float, float] | None = None, grid=None,
               unit: str = "model (relative)") -> EemResult:
        """The prediction wrapped as an :class:`EemResult` (same structure as ``analysis.eem``)."""
        pred = self.predict(specs, amplitudes)
        g = gains or {}
        value = {ex: pred[ex] * g.get(ex, 1.0) for ex in self.excitations}
        return eem_from_arrays(self.excitations, self.bands, value, grid=grid, unit=unit)


def eem_from_arrays(excitations, bands: dict[float, np.ndarray], value: dict[float, np.ndarray], *, grid=None,
                    std=None, unit: str = "relative", n_pixels: int = 0) -> EemResult:
    """Build an :class:`EemResult` from per-excitation arrays (NaN outside each band span on the grid)."""
    exs = [float(e) for e in excitations]
    if grid is None:
        lo = min(float(np.min(bands[e])) for e in exs)
        hi = max(float(np.max(bands[e])) for e in exs)
        grid = lo + 2.0 * np.arange(int(np.floor((hi - lo) / 2.0 + 1e-9)) + 1)
    grid = np.asarray(grid, float)
    gval = np.full((len(exs), grid.size), np.nan)
    for i, ex in enumerate(exs):
        em = np.asarray(bands[ex], float)
        v = np.asarray(value[ex], float)
        ok = np.isfinite(v)
        if ok.sum() >= 2:
            inside = (grid >= em[ok].min() - 1e-9) & (grid <= em[ok].max() + 1e-9)
            gval[i, inside] = np.interp(grid[inside], em[ok], v[ok])
    return EemResult(roi={"shape": "model"}, n_pixels=n_pixels, norm=NormSpec(mode="adu"), unit=unit,
                     excitations_nm=exs, emission_nm={e: np.asarray(bands[e], float) for e in exs},
                     value={e: np.asarray(value[e], float) for e in exs},
                     std=std if std is not None else {e: np.zeros(len(bands[e])) for e in exs},
                     grid_emission_nm=grid, grid_value=gval)


# --------------------------------------------------------------------------------------------
# data selection
# --------------------------------------------------------------------------------------------
@dataclass
class FitData:
    excitations: list[float]
    bands: dict[float, np.ndarray]
    y: dict[float, np.ndarray]

    @classmethod
    def from_eem(cls, eem_: EemResult, *, excitations=None, min_offset_nm: float = 20.0,
                 emission_range_nm: tuple[float, float] | None = None, loss: str = "normalized") -> "FitData":
        """Select the usable bins: finite, emission >= excitation + ``min_offset_nm`` (the twin's band-start
        protocol, drops the elastic / Rayleigh region), inside ``emission_range_nm``, and positive for ``log``."""
        exs = [float(e) for e in (excitations if excitations is not None else eem_.excitations_nm)]
        bands, y = {}, {}
        for ex in exs:
            em = np.asarray(eem_.emission_nm[ex], float)
            v = np.asarray(eem_.value[ex], float)
            ok = np.isfinite(v) & (em >= ex + min_offset_nm - 1e-9)
            if emission_range_nm is not None:
                ok &= (em >= emission_range_nm[0]) & (em <= emission_range_nm[1])
            if loss == "log":
                ok &= v > 0
            if ok.sum() < 4:
                continue
            bands[ex], y[ex] = em[ok], v[ok]
        if not bands:
            raise ValueError("no usable bins in the EEM")
        return cls(sorted(bands), bands, y)


# --------------------------------------------------------------------------------------------
# parameterisation
# --------------------------------------------------------------------------------------------
@dataclass
class _Layout:
    n_comp: int
    free_by: list[tuple[str, ...]]
    start: list[dict[str, float]]
    ripple_basis: np.ndarray | None = None        # (n_lambda, q)
    ripple_wl: np.ndarray | None = None
    ridge: np.ndarray | None = None               # per-column ridge weight of the instrument terms

    @property
    def n_shape(self) -> int:
        return sum(len(f) for f in self.free_by)

    @property
    def free(self) -> tuple[str, ...]:
        """Union of the free parameter names (order of first appearance)."""
        return tuple(dict.fromkeys(n for f in self.free_by for n in f))

    @property
    def n_amp(self) -> int:
        return self.n_comp - 1

    @property
    def n_ripple(self) -> int:
        return 0 if self.ripple_basis is None else self.ripple_basis.shape[1]

    def unpack(self, x: np.ndarray):
        params = [dict(s) for s in self.start]
        i = 0
        for k in range(self.n_comp):
            for name in self.free_by[k]:
                params[k][name] = float(x[i])
                i += 1
        amp = np.ones(self.n_comp)
        for k in range(1, self.n_comp):
            amp[k] = np.exp(x[i])
            i += 1
        u = x[i:i + self.n_ripple]
        return params, amp, u

    def bounds(self):
        lo, hi = [], []
        for k in range(self.n_comp):
            for name in self.free_by[k]:
                lo.append(PARAM_BOUNDS[name][0])
                hi.append(PARAM_BOUNDS[name][1])
        lo += [-8.0] * self.n_amp
        hi += [8.0] * self.n_amp
        lo += [-3.0] * self.n_ripple
        hi += [3.0] * self.n_ripple
        return np.array(lo), np.array(hi)

    def xscale(self):
        s = []
        for k in range(self.n_comp):
            s += [PARAM_BOUNDS[n][2] for n in self.free_by[k]]
        return np.array(s + [1.0] * self.n_amp + [0.3] * self.n_ripple)

    def names(self) -> list[str]:
        out = [f"c{k}.{n}" for k in range(self.n_comp) for n in self.free_by[k]]
        out += [f"c{k}.log_amp" for k in range(1, self.n_comp)]
        out += [f"instrument[{j}]" for j in range(self.n_ripple)]
        return out


def _to_spec(name: str, p: dict[str, float], template: FluorophoreSpec | None = None) -> FluorophoreSpec:
    base = {"quantum_yield": 0.5, "extinction": 1.0}
    if template is not None:
        base = {"quantum_yield": template.quantum_yield, "extinction": template.extinction}
    return FluorophoreSpec(name=name, ex_peak_nm=float(p["ex_peak_nm"]), ex_fwhm_nm=float(p["ex_fwhm_nm"]),
                           em_peak_nm=float(p["em_peak_nm"]), em_fwhm_nm=float(p["em_fwhm_nm"]),
                           em_skew=float(p["em_skew"]), vibronic=float(p["vibronic"]), **base)


def _poly_basis(wavelengths: np.ndarray, degree: int) -> np.ndarray:
    """Columns ``t^1..t^degree`` with ``t = (lam - 550 nm) / 150 nm`` (smooth log-throughput correction)."""
    t = (wavelengths - 550.0) / 150.0
    return np.stack([t ** j for j in range(1, degree + 1)], axis=1)


def _ripple_basis(wavelengths: np.ndarray, degree: int = 3) -> np.ndarray:
    """Orthonormal basis of the space of band-wavelength functions orthogonal to polynomials of order <= degree."""
    t = (wavelengths - wavelengths.mean()) / max(np.ptp(wavelengths), 1.0)
    V = np.vander(t, degree + 1, increasing=True)
    q, _ = np.linalg.qr(V, mode="complete")
    return q[:, degree + 1:]


# --------------------------------------------------------------------------------------------
# core: residuals with the gains profiled out
# --------------------------------------------------------------------------------------------
def _gain_and_residual(y: dict[float, np.ndarray], m: dict[float, np.ndarray], gain: str, loss: str,
                       fixed_gain: dict[float, float] | float | None = None):
    """Profile the gains out.  Returns ``(residual vector, gains dict)``."""
    gains: dict[float, float] = {}
    res = []
    if loss == "log":
        ly = {e: np.log(y[e]) for e in y}
        lm = {e: np.log(np.maximum(m[e], 1e-30)) for e in y}
        if fixed_gain is not None:
            for e in y:
                gains[e] = fixed_gain[e] if isinstance(fixed_gain, dict) else fixed_gain
        elif gain == "global":
            s = np.mean(np.concatenate([ly[e] - lm[e] for e in y]))
            gains = {e: float(np.exp(s)) for e in y}
        else:
            gains = {e: float(np.exp(np.mean(ly[e] - lm[e]))) for e in y}
        for e in y:
            res.append(ly[e] - lm[e] - np.log(gains[e]))
        return np.concatenate(res), gains
    # normalised
    w = {e: 1.0 / max(np.sqrt(np.mean(y[e] ** 2)), 1e-30) for e in y}
    if fixed_gain is not None:
        for e in y:
            gains[e] = fixed_gain[e] if isinstance(fixed_gain, dict) else fixed_gain
    elif gain == "global":
        num = sum(w[e] ** 2 * np.dot(y[e], m[e]) for e in y)
        den = sum(w[e] ** 2 * np.dot(m[e], m[e]) for e in y)
        s = num / den if den > 0 else 0.0
        gains = {e: float(s) for e in y}
    else:
        for e in y:
            den = np.dot(m[e], m[e])
            gains[e] = float(np.dot(y[e], m[e]) / den) if den > 0 else 0.0
    for e in y:
        res.append((y[e] - gains[e] * m[e]) * w[e])
    return np.concatenate(res), gains


class _Fitter:
    def __init__(self, model: ThinLayerModel, data: FitData, layout: _Layout, names: list[str], gain: str,
                 loss: str, ridge: float):
        self.model, self.data, self.layout, self.names, self.gain, self.loss, self.ridge = \
            model, data, layout, names, gain, loss, ridge
        # restrict the model to the fitted bins (indices into each excitation's band list)
        self.idx = {}
        for ex in data.excitations:
            full = model.bands[ex]
            self.idx[ex] = np.array([int(np.argmin(np.abs(full - b))) for b in data.bands[ex]])

    def ripple_factor(self, u: np.ndarray, ex: float) -> np.ndarray | float:
        lay = self.layout
        if lay.n_ripple == 0:
            return 1.0
        h = lay.ripple_basis @ u
        wl = lay.ripple_wl
        pos = np.searchsorted(wl, self.data.bands[ex] - 1e-6)
        return np.exp(h[np.clip(pos, 0, len(wl) - 1)])

    def model_eem(self, x: np.ndarray) -> dict[float, np.ndarray]:
        params, amp, u = self.layout.unpack(x)
        specs = [_to_spec(f"c{k}", p) for k, p in enumerate(params)]
        out = {}
        for ex in self.data.excitations:
            tot = 0.0
            for a, s in zip(amp, specs):
                tot = tot + a * self.model.excitation_overlap(s, ex) * self.model.emission_projection(s, ex)[self.idx[ex]]
            out[ex] = tot * self.ripple_factor(u, ex)
        return out

    def residual(self, x: np.ndarray) -> np.ndarray:
        r, _ = _gain_and_residual(self.data.y, self.model_eem(x), self.gain, self.loss)
        if self.layout.n_ripple:
            u = x[-self.layout.n_ripple:]
            r = np.concatenate([r, np.sqrt(self.layout.ridge) * u])
        return r


# --------------------------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------------------------
@dataclass
class FitResult:
    """Outcome of :func:`fit_fluorophores`."""

    specs: list[FluorophoreSpec]
    relative_brightness: list[float]         # share of each component in the total fitted signal (sums to 1)
    amplitudes: list[float]                  # a_k with a_0 = 1 (model units)
    gains: dict[float, float]                # g_ex (per_excitation) or the global scale (same value per ex)
    param_names: list[str]
    param_values: list[float]
    ci95: list[float]                        # 1.96 sigma in natural units (nm, dimensionless); NaN if singular
    free_params: tuple[str, ...]
    fit_ex: Any
    gain_mode: str
    loss: str
    instrument_gain: dict[float, float]      # band wavelength -> shared multiplicative gain (poly x ripple); {} if off
    diagnostics: dict[str, Any]
    start_costs: list[float]
    excitations: list[float]
    n_obs: int
    n_params: int

    def spec_dict(self, k: int = 0) -> dict:
        return self.specs[k].model_dump()

    def summary(self) -> str:
        d = self.diagnostics
        lines = [f"R2={d['r2']:.4f}  mean SAM={d['sam_mean_deg']:.2f} deg  rms_rel={d['rms_rel']:.3f}  "
                 f"n_obs={self.n_obs} n_par={self.n_params}"]
        for k, s in enumerate(self.specs):
            lines.append(f"  c{k}: ex {s.ex_peak_nm:.1f}/{s.ex_fwhm_nm:.0f}  em {s.em_peak_nm:.1f}/{s.em_fwhm_nm:.1f} "
                         f"skew {s.em_skew:.2f} vib {s.vibronic:.2f}  share {self.relative_brightness[k]:.2f}")
        return "\n".join(lines)


def _diagnostics(data: FitData, m: dict[float, np.ndarray], gains: dict[float, float]) -> dict[str, Any]:
    """R2 / SAM / relative RMS of the gain-corrected comparison (definitions in docs/studio/SIM2REAL.md)."""
    sam, r2_ex, rms = {}, {}, {}
    ss_res = ss_tot = 0.0
    resid = {}
    for ex in data.excitations:
        y, mm = data.y[ex], gains[ex] * m[ex]
        c = np.dot(y, mm) / max(np.linalg.norm(y) * np.linalg.norm(mm), 1e-30)
        sam[ex] = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        scale = float(np.max(np.abs(y)))
        yn, mn = y / scale, mm / scale
        sr = float(np.sum((yn - mn) ** 2))
        st = float(np.sum((yn - yn.mean()) ** 2))
        r2_ex[ex] = 1.0 - sr / st if st > 0 else float("nan")
        rms[ex] = float(np.sqrt(np.mean((yn - mn) ** 2)))
        ss_res += sr
        ss_tot += st
        resid[ex] = (y - mm) / scale                # residual as a fraction of the excitation's peak
    return {"r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "r2_per_ex": {float(k): v for k, v in r2_ex.items()},
            "sam_deg": {float(k): v for k, v in sam.items()},
            "sam_mean_deg": float(np.mean(list(sam.values()))),
            "rms_rel": float(np.mean(list(rms.values()))), "rms_rel_per_ex": {float(k): v for k, v in rms.items()},
            "residual_frac_of_peak": {float(k): v for k, v in resid.items()}}


# --------------------------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------------------------
DEFAULT_PRIOR = FluorophoreSpec(name="prior", ex_peak_nm=340.0, ex_fwhm_nm=60.0, em_peak_nm=480.0,
                                em_fwhm_nm=90.0, em_skew=0.2, vibronic=0.0)


def fit_fluorophores(eem_: EemResult, *, n_components: int = 1, gain: str = "per_excitation",
                     loss: str = "normalized", fit_ex: bool | None = None, fit_vibronic: bool = False,
                     fit_skew: bool = True, prior: list[FluorophoreSpec] | FluorophoreSpec | None = None,
                     name: str = "calibrated", instrument: TwinInstrument | None = None, n_starts: int = 12,
                     seed: int = 0, excitations=None, min_offset_nm: float = 20.0,
                     emission_range_nm: tuple[float, float] | None = None, ripple: bool = False,
                     ripple_ridge: float = 1e-3, response_degree: int = 0, response_sigma: float = 0.5,
                     robust: bool = True, max_nfev: int = 200) -> FitResult:
    """Calibrate ``n_components`` fluorophores to a measured (or simulated) ROI EEM.

    Parameters
    ----------
    eem_: an :class:`EemResult` from :func:`~spectraforge.studio.realdata.real_roi_eem` or ``analysis.eem``
        (``value[ex]`` over ``emission_nm[ex]`` are used; the radiometric unit is irrelevant, see ``gain``).
    gain: ``"per_excitation"`` or ``"global"`` (module docstring).
    fit_ex: excitation peak / FWHM freedom: ``True`` (all components free; the default for ``gain="global"``),
        ``"relative"`` (components >= 1 free, component 0 pinned at its prior -- the default for
        ``per_excitation`` with >= 2 components, since a per-excitation gain absorbs the absolute excitation
        efficiency but not the *ratio* between components), ``False`` (all pinned; default for one component).
    prior: starting :class:`FluorophoreSpec` (s); excitation values stay at the prior when ``fit_ex`` is False.
    n_starts: deterministic random multi-start (the first start is the prior itself).
    ripple: add the shared instrument ripple (module docstring).
    response_degree: degree of a smooth multiplicative throughput correction ``exp(sum_j c_j t^j)`` (``t = (lam -
        550 nm)/150 nm``) shared by all excitations -- absorbs a wrong LCTF / QE curve in the digital twin; with
        a Gaussian prior ``c_j ~ N(0, response_sigma^2)`` so that it moves only if the data demand it.  It is
        partly degenerate with the emission shape, so compare fits with and without it (docs/studio/SIM2REAL.md).
    """
    if gain not in ("global", "per_excitation"):
        raise ValueError("gain must be 'global' or 'per_excitation'")
    if loss not in ("normalized", "log"):
        raise ValueError("loss must be 'normalized' or 'log'")
    if not 1 <= n_components <= 3:
        raise ValueError("n_components must be 1..3")
    inst = instrument or TwinInstrument.lab_replica()
    data = FitData.from_eem(eem_, excitations=excitations, min_offset_nm=min_offset_nm,
                            emission_range_nm=emission_range_nm, loss=loss)
    # the model needs the *full* band list of each fitted excitation (bands are matched by wavelength)
    model = ThinLayerModel(inst, {ex: np.asarray(eem_.emission_nm[ex], float) for ex in data.excitations})
    if fit_ex is None:
        fit_ex = True if gain == "global" else ("relative" if n_components >= 2 else False)
    base_free = ["em_peak_nm", "em_fwhm_nm"] + (["em_skew"] if fit_skew else []) + (["vibronic"] if fit_vibronic else [])
    free_by = [tuple(base_free + (["ex_peak_nm", "ex_fwhm_nm"] if (fit_ex is True or (fit_ex == "relative" and k > 0))
                                  else [])) for k in range(n_components)]
    free = tuple(dict.fromkeys(n for f in free_by for n in f))
    priors = [prior] if isinstance(prior, FluorophoreSpec) else list(prior or [])
    while len(priors) < n_components:
        base = DEFAULT_PRIOR if not priors else priors[-1]
        priors.append(base.model_copy(update={"em_peak_nm": float(np.clip(base.em_peak_nm + 40.0, 410, 690))}))
    start0 = [{n: getattr(p, n) for n in _ALL} for p in priors[:n_components]]

    wl_all = np.unique(np.concatenate([data.bands[e] for e in data.excitations]))
    cols, ridge = [], []
    if response_degree > 0:
        pb = _poly_basis(wl_all, response_degree)
        cols.append(pb)
        ridge += [1.0 / response_sigma ** 2] * pb.shape[1]
    if ripple and wl_all.size > 6:
        rb = _ripple_basis(wl_all)
        cols.append(rb)
        ridge += [ripple_ridge] * rb.shape[1]
    basis = np.hstack(cols) if cols else None
    layout = _Layout(n_components, free_by, start0, basis, wl_all if basis is not None else None,
                     np.array(ridge) if cols else None)
    fitter = _Fitter(model, data, layout, layout.names(), gain, loss, ripple_ridge)
    lo, hi = layout.bounds()
    xs = layout.xscale()

    # ---- multi-start ------------------------------------------------------------------------
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0xCA11B]))
    ymean_peak = {e: float(data.bands[e][int(np.argmax(data.y[e]))]) for e in data.excitations}
    pk0 = float(np.median(list(ymean_peak.values())))
    starts = []
    x0 = []
    for k in range(n_components):
        for n in free_by[k]:
            x0.append(start0[k][n])
    x0 += [0.0] * layout.n_amp + [0.0] * layout.n_ripple
    starts.append(np.array(x0, float))
    for _ in range(max(0, n_starts - 1)):
        x = []
        for k in range(n_components):
            for n in free_by[k]:
                if n == "em_peak_nm":
                    v = pk0 + rng.normal(0, 25) + 30.0 * k
                elif n == "em_fwhm_nm":
                    v = rng.uniform(40, 130)
                elif n == "em_skew":
                    v = rng.uniform(0.0, 0.8)
                elif n == "vibronic":
                    v = rng.uniform(0.0, 0.3)
                elif n == "ex_peak_nm":
                    v = start0[k][n] + rng.normal(0, 20)
                else:
                    v = rng.uniform(30, 100)
                x.append(v)
        x += list(rng.normal(0, 1.0, layout.n_amp)) + [0.0] * layout.n_ripple
        starts.append(np.array(x, float))
    best, costs = None, []
    for x in starts:
        x = np.clip(x, lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(fitter.residual, x, bounds=(lo, hi), x_scale=xs, loss="soft_l1" if robust else "linear",
                                f_scale=0.05 if robust else 1.0, max_nfev=max_nfev)
        except (ValueError, FloatingPointError):
            continue
        costs.append(float(sol.cost))
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("all starts failed")
    return _finish(fitter, best, costs, name, priors, gain, loss, ripple, fit_ex)


def _finish(fitter: _Fitter, sol, costs, name, priors, gain, loss, ripple, fit_ex) -> FitResult:
    lay, data = fitter.layout, fitter.data
    x = sol.x
    params, amp, u = lay.unpack(x)
    m = fitter.model_eem(x)
    r, gains = _gain_and_residual(data.y, m, gain, loss)
    n_obs, p = int(r.size), int(len(x))
    # covariance from the Jacobian of the (soft-L1-weighted) residual
    J = sol.jac
    with np.errstate(all="ignore"):
        s2 = float(np.sum(r ** 2) / max(n_obs - p, 1))
        try:
            cov = s2 * np.linalg.pinv(J.T @ J, rcond=1e-10)
            sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            sig = np.full(p, np.nan)
    at_bound = (x - lay.bounds()[0] < 1e-6) | (lay.bounds()[1] - x < 1e-6)
    ci = np.where(at_bound, np.nan, 1.96 * sig)
    specs = [_to_spec(f"{name}" if lay.n_comp == 1 else f"{name}_{k + 1}", p_, priors[k] if k < len(priors) else None)
             for k, p_ in enumerate(params)]
    # relative brightness: integral of each component's contribution over the fitted EEM (gain-free)
    contrib = []
    for a, s in zip(amp, specs):
        tot = 0.0
        for ex in data.excitations:
            tot += a * fitter.model.excitation_overlap(s, ex) * float(np.sum(fitter.model.emission_projection(s, ex)[fitter.idx[ex]]))
        contrib.append(tot)
    share = (np.array(contrib) / max(sum(contrib), 1e-30)).tolist()
    rip = {}
    if lay.n_ripple:
        h = lay.ripple_basis @ u
        rip = {float(w): float(np.exp(v)) for w, v in zip(lay.ripple_wl, h)}
    diag = _diagnostics(data, m, gains)
    nat = []
    for i, nme in enumerate(lay.names()):
        nat.append(float(x[i]))
    # BIC of the Gaussian-error model on the normalised residual
    rss = float(np.sum(r ** 2))
    diag["bic"] = float(n_obs * np.log(max(rss / n_obs, 1e-300)) + p * np.log(n_obs))
    diag["rss"] = rss
    diag["cost_spread_rel"] = float((np.max(costs) - np.min(costs)) / max(np.min(costs), 1e-300)) if costs else float("nan")
    diag["n_starts_within_1pct"] = int(np.sum(np.array(costs) <= 1.01 * min(costs) + 1e-12)) if costs else 0
    return FitResult(specs=specs, relative_brightness=share, amplitudes=amp.tolist(), gains=gains,
                     param_names=lay.names(), param_values=nat, ci95=ci.tolist(), free_params=lay.free,
                     fit_ex=fit_ex, gain_mode=gain, loss=loss, instrument_gain=rip, diagnostics=diag, start_costs=costs,
                     excitations=list(data.excitations), n_obs=n_obs, n_params=p)


# --------------------------------------------------------------------------------------------
# prediction / hold-out
# --------------------------------------------------------------------------------------------
def predict_eem(result: FitResult, bands: dict[float, np.ndarray], *, instrument: TwinInstrument | None = None,
                gains: dict[float, float] | None = None) -> dict[float, np.ndarray]:
    """Model EEM ``{ex: (B,)}`` of a calibrated fit on arbitrary excitations / bands.  ``gains`` default to the
    fitted ones where available (else 1).  Ripple gains are applied by nearest band wavelength where known."""
    inst = instrument or TwinInstrument.lab_replica()
    model = ThinLayerModel(inst, bands)
    pred = model.predict(result.specs, result.amplitudes)
    out = {}
    for ex, v in pred.items():
        t = np.ones_like(v)
        if result.instrument_gain:
            wl = np.array(sorted(result.instrument_gain))
            tv = np.array([result.instrument_gain[w] for w in wl])
            for j, b in enumerate(model.bands[ex]):
                i = int(np.argmin(np.abs(wl - b)))
                if abs(wl[i] - b) <= 1.0:
                    t[j] = tv[i]
        g = (gains or result.gains).get(ex, 1.0)
        out[ex] = v * t * g
    return out


def evaluate_prediction(result: FitResult, eem_: EemResult, *, excitations=None, refit_gains: bool = True,
                        min_offset_nm: float = 20.0, emission_range_nm=None,
                        instrument: TwinInstrument | None = None) -> dict[str, Any]:
    """Score a calibrated fit on (possibly unseen) data: spectral angle, R2 and relative RMS per excitation.

    ``refit_gains=True`` re-estimates only the nuisance gain(s) (per excitation or global, like the fit) so the
    score measures the *shape / spectral* prediction; ``False`` uses the training gain(s) as-is -- an absolute
    radiometric prediction, meaningful only for ``gain='global'`` fits on commonly calibrated data."""
    data = FitData.from_eem(eem_, excitations=excitations, min_offset_nm=min_offset_nm,
                            emission_range_nm=emission_range_nm, loss=result.loss)
    inst = instrument or TwinInstrument.lab_replica()
    full_bands = {ex: np.asarray(eem_.emission_nm[ex], float) for ex in data.excitations}
    model = ThinLayerModel(inst, full_bands)
    pred_full = predict_eem(replace_gains(result, {}), full_bands, instrument=inst)   # gain 1
    m = {}
    for ex in data.excitations:
        idx = np.array([int(np.argmin(np.abs(full_bands[ex] - b))) for b in data.bands[ex]])
        m[ex] = pred_full[ex][idx]
    if refit_gains:
        _, gains = _gain_and_residual(data.y, m, result.gain_mode, result.loss)
    elif result.gain_mode == "global":
        g0 = float(np.mean(list(result.gains.values())))
        gains = {ex: g0 for ex in data.excitations}
    else:
        missing = [ex for ex in data.excitations if ex not in result.gains]
        if missing:
            raise ValueError(f"no training gain for {missing} nm: use refit_gains=True")
        gains = {ex: result.gains[ex] for ex in data.excitations}
    d = _diagnostics(data, m, gains)
    d["gains"] = {float(k): float(v) for k, v in gains.items()}
    # absolute-amplitude metrics (meaningful when the gains are not refitted on the test data)
    yall = np.concatenate([data.y[e] for e in data.excitations])
    mall = np.concatenate([gains[e] * m[e] for e in data.excitations])
    d["r2_abs"] = float(1.0 - np.sum((yall - mall) ** 2) / max(np.sum((yall - yall.mean()) ** 2), 1e-30))
    d["peak_ratio_pred_over_meas"] = {float(e): float(np.max(gains[e] * m[e]) / np.max(data.y[e]))
                                      for e in data.excitations}
    return d


def replace_gains(result: FitResult, gains: dict[float, float]) -> FitResult:
    from dataclasses import replace
    return replace(result, gains=gains)


def baseline_scores(train: EemResult, test: EemResult, *, min_offset_nm: float = 20.0,
                    excitations=None) -> dict[str, float]:
    """Non-parametric reference predictors for a hold-out comparison (mean spectral angle, deg):

    * ``empirical``: the training EEM's pooled mean spectrum (normalised per excitation) interpolated onto
      the test bands -- the best a calibration-free "copy the other sample" predictor can do;
    * ``flat``: a constant spectrum."""
    tdat = FitData.from_eem(test, excitations=excitations, min_offset_nm=min_offset_nm)
    rdat = FitData.from_eem(train, min_offset_nm=min_offset_nm)
    pool_w = np.concatenate([rdat.bands[e] for e in rdat.excitations])
    pool_v = np.concatenate([rdat.y[e] / np.linalg.norm(rdat.y[e]) for e in rdat.excitations])
    order = np.argsort(pool_w)
    uw, inv = np.unique(pool_w[order], return_inverse=True)
    uv = np.bincount(inv, weights=pool_v[order]) / np.bincount(inv)
    sam_emp, sam_flat = [], []
    for ex in tdat.excitations:
        y = tdat.y[ex]
        for pred, store in ((np.interp(tdat.bands[ex], uw, uv), sam_emp), (np.ones_like(y), sam_flat)):
            c = np.dot(y, pred) / max(np.linalg.norm(y) * np.linalg.norm(pred), 1e-30)
            store.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return {"empirical_sam_deg": float(np.mean(sam_emp)), "flat_sam_deg": float(np.mean(sam_flat))}


def emission_summary(spec: FluorophoreSpec, window_nm: float = 420.0) -> dict[str, float]:
    """Identifiable descriptors of a calibrated emission band.

    The Nuance window starts at 420 nm, so for a blue emitter (collagen) the fitted ``em_peak_nm`` is only an
    *upper-limit-like* parameter (it can sit on the lower bound); these descriptors are well constrained by
    data in the window:

    * ``peak_nm``: argmax of the emission density;
    * ``centroid_window_nm``: mean wavelength of the emission above ``window_nm``;
    * ``red_half_max_nm``: wavelength on the red side where the density falls to half its maximum;
    * ``fraction_in_window``: share of the emitted photons above ``window_nm``."""
    g = np.arange(300.0, 900.0 + 1e-9, 1.0)
    e = optics_mod.to_legacy(spec).emission(g)
    pk = int(np.argmax(e))
    red = np.where((g > g[pk]) & (e < 0.5 * e[pk]))[0]
    w = g >= window_nm
    return {"peak_nm": float(g[pk]), "centroid_window_nm": float(np.sum(g[w] * e[w]) / np.sum(e[w])),
            "red_half_max_nm": float(g[red[0]]) if red.size else float("nan"),
            "fraction_in_window": float(np.sum(e[w]) / np.sum(e))}


def pixel_sam_map(cube, result: FitResult, region=None, *, instrument: TwinInstrument | None = None,
                  min_offset_nm: float = 20.0) -> np.ndarray:
    """Per-pixel spectral angle (degrees, mean over excitations) between each measured pixel spectrum and the
    calibrated model spectrum (the model's shape at that excitation; the amplitude is irrelevant to an angle).

    ``region``: boolean ``(H, W)`` image or ROI name (default: the cube's sample mask); pixels outside are NaN.
    This is the spatial residual map: it shows where the single calibrated spectral model fails (different
    material, thickness / scattering effects, edges)."""
    ex_list = [e for e in result.excitations if e in cube.cubes]
    bands = {ex: np.asarray(cube.emission_nm[ex], float) for ex in ex_list}
    pred = predict_eem(result, bands, instrument=instrument, gains={ex: 1.0 for ex in ex_list})
    if region is None:
        reg = np.ones((cube.height, cube.width), bool) if cube.mask is None else cube.mask > 0
    else:
        reg = cube.roi_pixels(region)
        if cube.mask is not None:
            reg = reg & (cube.mask > 0)
    out = np.full((cube.height, cube.width), np.nan)
    acc = np.zeros(int(reg.sum()))
    n = 0
    for ex in ex_list:
        em = bands[ex]
        use = (em >= ex + min_offset_nm - 1e-9)
        y = cube.cubes[ex][reg][:, use].astype(np.float64)
        m = pred[ex][use]
        with np.errstate(all="ignore"):
            c = (y @ m) / (np.linalg.norm(y, axis=1) * np.linalg.norm(m))
        acc += np.degrees(np.arccos(np.clip(c, -1, 1)))
        n += 1
    out[reg] = acc / max(n, 1)
    return out


# --------------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------------
def bootstrap_ci(cube, roi, result: FitResult, *, n_boot: int = 20, seed: int = 0, stat: str = "mean",
                 instrument: TwinInstrument | None = None, fit_kwargs: dict | None = None) -> dict[str, Any]:
    """Pixel bootstrap of the ROI-mean EEM: refit ``n_boot`` times (started at ``result``) -> percentile 95 %
    intervals of every spec parameter.  Captures sampling variability of the sample, *not* model error."""
    from spectraforge.studio import realdata as rd

    m = cube._effective(roi)
    rows, cols = np.nonzero(m)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0xB007]))
    kw = dict(fit_kwargs or {})
    kw.setdefault("gain", result.gain_mode)
    kw.setdefault("loss", result.loss)
    kw.setdefault("n_components", len(result.specs))
    kw.setdefault("excitations", result.excitations)
    kw["n_starts"] = 1
    kw["prior"] = result.specs
    kw["fit_ex"] = result.fit_ex
    kw["fit_skew"] = "em_skew" in result.free_params
    kw["fit_vibronic"] = "vibronic" in result.free_params
    vals = {f"c{k}.{n}": [] for k in range(len(result.specs)) for n in ("ex_peak_nm", "ex_fwhm_nm", "em_peak_nm",
                                                                         "em_fwhm_nm", "em_skew", "vibronic")}
    for _ in range(n_boot):
        pick = rng.integers(0, rows.size, rows.size)
        bm = np.zeros_like(m)
        # bootstrap by weighting: build a resampled-mean EEM
        value = {}
        for ex in cube.excitations_nm:
            sub = cube.cubes[ex][rows[pick], cols[pick]].astype(float)
            with np.errstate(all="ignore"):
                value[ex] = np.nanmedian(sub, axis=0) if stat == "median" else np.nanmean(sub, axis=0)
        ee = eem_from_arrays(cube.excitations_nm, cube.emission_nm, value, n_pixels=rows.size)
        fr = fit_fluorophores(ee, instrument=instrument, **kw)
        for k, s in enumerate(fr.specs):
            for n in ("ex_peak_nm", "ex_fwhm_nm", "em_peak_nm", "em_fwhm_nm", "em_skew", "vibronic"):
                vals[f"c{k}.{n}"].append(getattr(s, n))
            for n, v in emission_summary(s).items():
                vals.setdefault(f"c{k}.{n}", []).append(v)
    out = {}
    for key, v in vals.items():
        v = np.asarray(v)
        out[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                    "lo95": float(np.percentile(v, 2.5)), "hi95": float(np.percentile(v, 97.5))}
    return out


# --------------------------------------------------------------------------------------------
# fast-model verification against a full capture
# --------------------------------------------------------------------------------------------
def thin_layer_experiment(spec: FluorophoreSpec, *, excitations=(340.0, 365.0), concentration_uM: float = 0.05,
                          instrument: TwinInstrument | None = None, thickness_mm: float = 2.0,
                          power_mW: float = 5.0, exposure_ms: float = 100.0, noise: bool = False,
                          sensor_px: tuple[int, int] = (24, 18), seed: int = 0):
    """A flat, optically thin, clear and dilute layer of ``spec`` seen by the twin instrument.

    With ``noise=False`` only the layer's fluorescence remains (no Raman, elastic leak, glint, blur, bleaching,
    re-absorption or detector noise) -- the configuration the fast forward model describes.  With ``noise=True``
    the detector chain (shot / read / dark / fixed pattern / quantisation) is on, as in a real acquisition."""
    from spectraforge.studio import presets
    from spectraforge.studio.schema import (Background, BoxShape, DyeAmount, ExcitationStep, Experiment, MaterialSpec,
                                            PhysicsToggles, ScatteringSpec, SceneObject, SensorSpec, Transform)

    inst = instrument or TwinInstrument.lab_replica()
    mat = MaterialSpec(id="thin", name="thin dilute layer",
                       dyes=[DyeAmount(fluorophore=spec.name, concentration_uM=concentration_uM)],
                       refractive_index=1.33, scattering=ScatteringSpec(), surface_roughness=0.0)
    steps = [ExcitationStep(wavelength_nm=float(w), power_mW=power_mW, exposure_ms=exposure_ms) for w in excitations]
    src = inst.source.model_copy(update={"sequence": steps, "power_drift_pct": 0.0})
    w, h = sensor_px
    cam = inst.camera.model_copy(update={"sensor": inst.camera.sensor.model_copy(
        update={"width_px": w, "height_px": h, "pixel_pitch_um": inst.camera.sensor.pixel_pitch_um * 348 / w})})
    off = dict(raman=False, elastic_leak=False, specular_glints=False, shadows=False, secondary_emission=False,
               photobleaching=False, fret=False, quenching=False, vignetting=False, filter_angle_shift=False,
               power_drift=False, defocus=False, diffraction=False, chromatic_aberration=False)
    if not noise:
        off.update(shot_noise=False, read_noise=False, dark_current=False, fixed_pattern_noise=False,
                   quantization=False, saturation=False)
    tg = PhysicsToggles(**off)
    exp = Experiment(
        materials=presets.builtin_materials() + [mat], fluorophores=[spec],
        objects=[SceneObject(id="o", name="layer", shape=BoxShape(size_mm=(40.0, thickness_mm, 40.0)),
                             material_id="thin", transform=Transform(position_mm=(0.0, thickness_mm / 2, 0.0)))],
        background=Background(kind="plane", material_id="black_anodized"),
        sources=[src], cameras=[cam], active_source_id=src.id, active_camera_id=cam.id)
    acq = exp.acquisition.model_copy(update={"physics": tg, "seed": seed})
    return exp.model_copy(update={"acquisition": acq})


def verify_fast_model(spec: FluorophoreSpec, *, excitations=(340.0, 365.0), concentration_uM: float = 0.05,
                      instrument: TwinInstrument | None = None, thickness_mm: float = 2.0,
                      min_signal_frac: float = 0.05) -> dict[str, Any]:
    """Compare :class:`ThinLayerModel` with a full ``capture`` of a thin clear dilute layer of ``spec``.

    The full capture runs the whole engine (ray tracing, clear-medium transport with Fresnel + path, lens
    radiometry, band responses, detector in expected-value mode); noise, fixed pattern, blur, Raman, elastic
    leak and room light are disabled so that only the fluorescence of the layer remains.  The fast model
    must reproduce ``electrons_mean / (P_ex t_ex)`` at the image centre up to one global factor.  Returns the
    maximum / mean relative deviation over bins above ``min_signal_frac`` of the peak.
    """
    from spectraforge.studio import capture as cap

    inst = instrument or TwinInstrument.lab_replica()
    exp = thin_layer_experiment(spec, excitations=excitations, concentration_uM=concentration_uM, instrument=inst,
                                thickness_mm=thickness_mm)
    res = cap.capture(exp)
    cy, cx = res.height // 2, res.width // 2
    sl = (slice(cy - 1, cy + 2), slice(cx - 1, cx + 2))
    bands = {float(e): np.asarray(res.emission_nm[e], float) for e in res.excitations_nm}
    model = ThinLayerModel(inst, bands)
    pred = model.predict([spec], [1.0])
    num = den = 0.0
    full, fast = {}, {}
    for ex in res.excitations_nm:
        e_mean = res.electrons_mean[ex][sl].mean(axis=(0, 1)).astype(float)
        full[ex] = e_mean / (res.power_mW[ex] * res.exposure_ms[ex])
        fast[ex] = pred[float(ex)]
        num += float(np.dot(full[ex], fast[ex]))
        den += float(np.dot(fast[ex], fast[ex]))
    s = num / den
    devs, per_ex = [], {}
    for ex in res.excitations_nm:
        ok = full[ex] > min_signal_frac * np.max(full[ex])
        d = np.abs(s * fast[ex][ok] - full[ex][ok]) / full[ex][ok]
        per_ex[float(ex)] = {"max_rel": float(d.max()), "mean_rel": float(d.mean()), "n_bins": int(ok.sum()),
                             "scale_ratio": float(np.max(full[ex]) / (s * np.max(fast[ex])))}
        devs.append(d)
    dv = np.concatenate(devs)
    return {"max_rel_dev": float(dv.max()), "mean_rel_dev": float(dv.mean()), "per_excitation": per_ex,
            "global_scale": float(s), "full_capture_s": float(res.timings_s.get("radiance", float("nan")))}


# --------------------------------------------------------------------------------------------
# scattering
# --------------------------------------------------------------------------------------------
@dataclass
class ScatteringCalibration:
    identifiable: bool
    reason: str
    mus_prime_500_per_mm: float | None = None
    elastic_bands: dict[float, int] = field(default_factory=dict)


def calibrate_scattering(cube, roi, *, elastic_halfwidth_nm: float = 15.0) -> ScatteringCalibration:
    """Estimate the sample's effective ``mu_s'`` from the elastic (excitation-wavelength) channel -- if recorded.

    The elastic reflectance of a turbid sample is a monotone function of ``mu_s'/mu_a`` and would calibrate the
    scattering power.  The lab's processed cubes, however, carry the Nuance protocol's Rayleigh cut-off: the band
    list starts at ``max(420 nm, ex + >=20 nm)`` and the LCTF is operated behind a long-pass, so no band falls
    within ``elastic_halfwidth_nm`` of the excitation line.  In that case the elastic channel is simply not
    measured and the function reports ``identifiable=False`` (it does not guess).  Without a white reference
    (the droplet ``Whitelight`` is a scene, not a flat field) the absolute reflectance would be unknown anyway."""
    near = {}
    for ex in cube.excitations_nm:
        em = np.asarray(cube.emission_nm[ex], float)
        n = int(np.sum(np.abs(em - ex) <= elastic_halfwidth_nm))
        if n:
            near[ex] = n
    if not near:
        first = {ex: float(np.min(cube.emission_nm[ex]) - ex) for ex in cube.excitations_nm}
        return ScatteringCalibration(
            False, "no emission band lies within +-%.0f nm of any excitation line (smallest band-excitation offset "
                   "%.0f nm): the elastic/Rayleigh channel was cut off at acquisition, so mu_s' is not identifiable "
                   "from this dataset. Fitting it from the fluorescence spectral tilt alone is degenerate with the "
                   "emission shape." % (elastic_halfwidth_nm, min(first.values())), None, {})
    return ScatteringCalibration(
        False, f"elastic bands present at {sorted(near)} nm but no flat-field white reference to convert them "
               "to absolute reflectance; mu_s' left uncalibrated", None, near)


# --------------------------------------------------------------------------------------------
# calibrated library
# --------------------------------------------------------------------------------------------
def calibrated_fluorophores(with_provenance: bool = False, path: str | Path | None = None):
    """The data-driven fluorophores written by ``scripts/studio_sim2real.py``.

    Returns a ``list[FluorophoreSpec]`` (or ``[(spec, provenance_dict), ...]``); an empty list if the file
    has not been generated."""
    p = Path(path) if path else CALIBRATED_JSON
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    out = []
    for item in doc.get("fluorophores", []):
        spec = FluorophoreSpec(**item["spec"])
        out.append((spec, item.get("provenance", {})) if with_provenance else spec)
    return out


def write_calibrated_fluorophores(entries: list[tuple[FluorophoreSpec, dict]], path: str | Path | None = None,
                                  note: str = "") -> Path:
    """Write ``[(spec, provenance), ...]`` to the calibrated-fluorophore JSON."""
    p = Path(path) if path else CALIBRATED_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": 1, "generated": date.today().isoformat(), "note": note,
           "fluorophores": [{"spec": s.model_dump(), "provenance": prov} for s, prov in entries]}
    p.write_text(json.dumps(doc, indent=2, default=float))
    return p
