"""Excitation light source: beam irradiance, spectral line shapes, out-of-band leakage, drift.

Beam model
----------
The source sits at ``position_mm`` and emits along the axis ``a = (target - position)/|…|``. The
beam (1/e² Gaussian radius, or flat-top radius) grows linearly with the axial distance ``z``
from the exit aperture::

    w(z) = D/2 + z tan α                       D = aperture_diameter_mm, α = divergence half-angle

i.e. the rays appear to come from a virtual apex at ``z0 = −(D/2)/tan α`` behind the aperture, so
the on-axis irradiance falls as ``1/(z − z0)²`` (inverse square) at large ``z``.

Irradiance per 1 mW of source power on the plane perpendicular to the axis (mW/mm²):

* Gaussian:  ``I(r, z) = 2/(π w²) · exp(−2 r²/w²)``          (∫ I dA = 1, Siegman, *Lasers*, ch. 17)
* Flat-top:  ``I(r, z) = [½ erfc((r − w)/(√2 σ))] / (π (w² + σ²))``, ``σ = 0.03 w`` (soft edge,
  normalised so that ∫ I dA = 1).

The flux density along the local propagation direction ``ŝ`` (from the virtual apex through the
point) is ``S = I / cos φ`` with ``cos φ = ŝ·a``; the irradiance on a surface with outward normal
``n`` is ``E = S · max(0, −ŝ·n)``. Points at ``z <= 0`` (behind the exit aperture) get 0. The
shadow-ray direction to the source is ``u = −ŝ`` (toward the point where the ray's cone line
crosses the aperture plane) and ``dist = z / cos φ`` is its length.

Spectral model
--------------
* Line shapes are area-normalised (∫ = 1, units 1/nm) and integrated *analytically per bin*
  (differences of the Gaussian CDF), so a 1 nm laser on a 1 nm grid is exact. Tunable lamp and
  laser: Gaussian with FWHM ``line_fwhm_nm``; LED: split-normal with a 1.4× longer red half-width
  at the same total FWHM (LED emission has a long-wavelength tail).
* Out-of-band leakage per mW of in-band power (units 1/nm), suppressed by ``10^-OD`` of the
  clean-up filter: a tunable lamp leaks its xenon continuum
  ``10^-OD · X(λ) / (X(λ0) · 1.0645 FWHM)`` (relative to the selected line), an LED a broad
  Lorentzian tail, a laser essentially nothing (0.1 % of an LED-like ASE pedestal).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import erfc, ndtr

from spectraforge.studio.schema import LightSourceSpec

_FLAT_TOP_EDGE = 0.03          # edge width / radius
_LED_RED_TAIL = 1.4            # red half-width / blue half-width
_GAUSS_AREA = np.sqrt(np.pi / (4.0 * np.log(2.0)))   # ∫ exp(-4 ln2 (x/FWHM)²) dx = 1.0645 FWHM


# --------------------------------------------------------------------------------------------
# Beam geometry
# --------------------------------------------------------------------------------------------
def source_axis(source: LightSourceSpec) -> np.ndarray:
    a = np.asarray(source.target_mm, float) - np.asarray(source.position_mm, float)
    n = np.linalg.norm(a)
    return a / n if n > 0 else np.array([0.0, -1.0, 0.0])


@dataclass
class BeamSamples:
    """Beam quantities at N world points."""

    flux_per_mW: np.ndarray      # S: flux density along propagation, mW/mm² per mW (0 behind aperture)
    u_to_source: np.ndarray      # (N,3) unit vector from the point toward the source aperture
    dist_mm: np.ndarray          # distance from the point to the aperture plane along u_to_source
    z_mm: np.ndarray             # axial distance from the aperture plane
    w_mm: np.ndarray             # beam radius at the point's axial position
    r_mm: np.ndarray             # distance from the beam axis
    on_axis_cos: np.ndarray      # cos φ = ŝ·a


def _profile(source: LightSourceSpec, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    if source.beam.profile == "gaussian":
        return 2.0 / (np.pi * w * w) * np.exp(-2.0 * r * r / (w * w))
    sig = _FLAT_TOP_EDGE * w
    return 0.5 * erfc((r - w) / (np.sqrt(2.0) * sig)) / (np.pi * (w * w + sig * sig))


def beam_samples(source: LightSourceSpec, points: np.ndarray) -> BeamSamples:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(source.position_mm, float)
    a = source_axis(source)
    rel = pts - pos
    z = rel[:, 0] * a[0] + rel[:, 1] * a[1] + rel[:, 2] * a[2]
    r = np.linalg.norm(rel - z[:, None] * a, axis=1)
    half_d = 0.5 * source.beam.aperture_diameter_mm
    tan_a = np.tan(np.radians(source.beam.divergence_half_angle_deg))
    w = half_d + np.maximum(z, 0.0) * tan_a
    if tan_a > 1e-9:
        apex = pos - a * (half_d / tan_a)
        s = pts - apex
        s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    else:
        s = np.broadcast_to(a, pts.shape).copy()
    cphi = np.clip(s[:, 0] * a[0] + s[:, 1] * a[1] + s[:, 2] * a[2], 0.05, 1.0)
    flux = np.where(z > 0.0, _profile(source, r, w) / cphi, 0.0)
    return BeamSamples(flux, -s, z / cphi, z, w, r, cphi)


def irradiance_per_mW(source: LightSourceSpec, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Surface irradiance ``E`` (mW/mm² per mW) at ``points`` with outward unit ``normals``
    (includes the incidence cosine, before Fresnel; 0 if the source is behind the surface)."""
    bs = beam_samples(source, points)
    cos_inc = np.einsum("ij,ij->i", bs.u_to_source, np.asarray(normals, float).reshape(-1, 3))
    return bs.flux_per_mW * np.maximum(cos_inc, 0.0)


# --------------------------------------------------------------------------------------------
# Spectral line shapes
# --------------------------------------------------------------------------------------------
def _bin_edges(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    if g.size == 1:
        return np.array([g[0] - 0.5, g[0] + 0.5])
    mid = 0.5 * (g[1:] + g[:-1])
    return np.concatenate([[g[0] - (mid[0] - g[0])], mid, [g[-1] + (g[-1] - mid[-1])]])


def _gauss_cdf(x, mu, sigma):
    return ndtr((x - mu) / sigma)


def _split_cdf(x, mu, sb, sr):
    tot = sb + sr
    return np.where(x < mu, 2.0 * sb / tot * ndtr((x - mu) / sb),
                    sb / tot + 2.0 * sr / tot * (ndtr((x - mu) / sr) - 0.5))


def line_shape(source: LightSourceSpec, center_nm: float, grid_nm: np.ndarray) -> np.ndarray:
    """Area-normalised excitation line (1/nm) on ``grid_nm`` (bin-integrated, ``∑ f Δλ = 1`` up to
    mass falling outside the grid)."""
    edges = _bin_edges(grid_nm)
    widths = np.diff(edges)
    fwhm = source.line_fwhm_nm
    k = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))        # FWHM → σ
    if source.kind == "led_array":
        hb = fwhm / (1.0 + _LED_RED_TAIL)               # blue half-width at half maximum
        sb = hb / np.sqrt(2.0 * np.log(2.0))
        sr = _LED_RED_TAIL * hb / np.sqrt(2.0 * np.log(2.0))
        cdf = _split_cdf(edges, center_nm, sb, sr)
    else:
        cdf = _gauss_cdf(edges, center_nm, fwhm * k)
    return np.diff(cdf) / widths


# --------------------------------------------------------------------------------------------
# Xenon arc lamp
# --------------------------------------------------------------------------------------------
# Smooth continuum of a high-pressure Xe arc (relative, 1.0 near 500 nm): UV roll-off below
# ~300 nm, broad visible maximum, slow NIR decline; Xe I/II lines are added on top (ozone-free
# short-arc lamps, e.g. Newport/Ushio data sheets).
_XE_CONT_NM = np.array([200, 250, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520,
                        550, 580, 600, 650, 700, 750, 800, 900, 1000, 1100], dtype=float)
_XE_CONT_REL = np.array([0.002, 0.02, 0.06, 0.12, 0.25, 0.38, 0.46, 0.53, 0.60, 0.68, 0.78, 0.88,
                         0.95, 1.00, 1.00, 0.99, 0.96, 0.94, 0.88, 0.82, 0.76, 0.70, 0.60, 0.52,
                         0.45])
# (centre nm, height relative to the continuum scale, FWHM nm)
_XE_LINES = [(450.1, 0.06, 1.8), (462.4, 0.10, 1.8), (467.1, 0.12, 1.8), (473.4, 0.08, 1.8),
             (764.2, 0.9, 2.0), (823.2, 1.5, 2.0), (828.0, 1.8, 2.0), (834.7, 2.0, 2.0),
             (881.9, 3.0, 2.0), (895.2, 1.6, 2.0), (904.5, 2.0, 2.0), (916.3, 1.5, 2.0),
             (980.0, 1.8, 2.5), (992.0, 1.0, 2.5)]
_xe_spline = PchipInterpolator(_XE_CONT_NM, _XE_CONT_REL, extrapolate=True)


def xenon_relative_spectrum(nm) -> np.ndarray:
    """Relative spectral power density of a Xe arc lamp (dimensionless, ~1 at 500 nm)."""
    lam = np.asarray(nm, dtype=float)
    x = np.clip(_xe_spline(np.clip(lam, 150.0, 1200.0)), 0.0, None)
    x = np.where(lam < 200.0, x * np.exp(-(200.0 - lam) / 20.0), x)
    for c, h, f in _XE_LINES:
        x = x + h * np.exp(-4.0 * np.log(2.0) * ((lam - c) / f) ** 2)
    return x


def leakage_spectrum(source: LightSourceSpec, center_nm: float, grid_nm: np.ndarray) -> np.ndarray:
    """Out-of-band light per mW of in-band power (1/nm) on ``grid_nm`` (see module docstring)."""
    lam = np.asarray(grid_nm, dtype=float)
    supp = 10.0 ** (-source.excitation_filter_od)
    fwhm = source.line_fwhm_nm
    mask = 1.0 - np.exp(-np.log(2.0) * ((lam - center_nm) / (2.0 * fwhm)) ** 2)   # notch at the line
    if source.kind == "tunable_lamp":
        if source.lamp_curve == "xenon":
            shape = xenon_relative_spectrum(lam)
            ref = max(float(xenon_relative_spectrum(center_nm)), 1e-3)
        else:
            shape = np.ones_like(lam)
            ref = 1.0
        return supp * shape / (ref * _GAUSS_AREA * fwhm) * mask
    gamma = 2.0 * fwhm                                   # Lorentzian HWHM
    lor = gamma / np.pi / ((lam - center_nm) ** 2 + gamma ** 2)
    scale = 1.0 if source.kind == "led_array" else 1e-3
    return supp * scale * lor * mask


@dataclass
class ExcitationSpectrum:
    line: np.ndarray       # in-band, area-normalised (1/nm)
    leakage: np.ndarray    # out-of-band per mW in-band (1/nm)

    @property
    def total(self) -> np.ndarray:
        return self.line + self.leakage


def excitation_spectrum(source: LightSourceSpec, center_nm: float, grid_nm: np.ndarray) -> ExcitationSpectrum:
    return ExcitationSpectrum(line_shape(source, center_nm, grid_nm),
                              leakage_spectrum(source, center_nm, grid_nm))


# --------------------------------------------------------------------------------------------
# Power drift
# --------------------------------------------------------------------------------------------
def drifted_powers(source: LightSourceSpec, steps, rng: np.random.Generator, enabled: bool = True) -> list[float]:
    """Delivered power (mW) per excitation step: ``P (1 + σ ξ)``, ξ ~ N(0,1), σ = ``power_drift_pct``/100
    (clipped at 0). ``steps``: iterable of :class:`ExcitationStep`. No random numbers are drawn
    when ``enabled`` is False or σ = 0."""
    sigma = source.power_drift_pct / 100.0
    out = []
    for st in steps:
        p = float(st.power_mW)
        if enabled and sigma > 0.0:
            p = max(0.0, p * (1.0 + sigma * float(rng.standard_normal())))
        out.append(p)
    return out
