"""The hyperspectral camera: spectral bands, lens radiometry, optical blur and the detector.

Radiometric chain (spectral radiance -> band photo-electrons)
-------------------------------------------------------------
For a scene point radiating ``L(lam)`` [mW/(mm^2 sr nm)] toward the camera, the image-plane
irradiance of a thin lens of transmission ``T``, working f-number ``N`` and magnification ``m`` is
(Smith, *Modern Optical Engineering*; the cos^4 law is the off-axis fall-off)::

    E_img = pi * L * T / (4 N^2 (1 + m)^2) * cos^4(theta)                  [mW/mm^2]

with ``m = f / (S - f)`` for a lens focused at object distance ``S``.  A pixel of area ``A``
(``(pitch*binning)^2``) exposed for ``t`` collects, per band ``b`` with spectral response
``R_b(lam)`` (passband x long-pass), peak filter transmission ``Tp_b`` and quantum efficiency
``QE(lam)``::

    e_b = k0 cos^4(theta) A t  integral L(lam) R_b(lam) Tp_b QE(lam) (lam / hc) dlam,
    k0  = pi T / (4 N^2 (1+m)^2),     lam/hc = 5.0341e12 * lam[nm]  photons / (mW s)

The reported band *radiance* is ``integral L R_b dlam`` (mW/(mm^2 sr)): the radiance the band
passes, *before* peak transmission, lens and detector.

Spectral bands
--------------
*LCTF* (``kind='lctf'``, Lyot-Solc stack): Gaussian pass-band of the configured FWHM -- the main lobe of a
birefringent filter is close to Gaussian and its transmission falls to ~1 % one FWHM away (OD ~2).
*Interference filters* (``kind='filter_wheel'``): flat-top super-Gaussian of order 6,
``T = exp(-ln2 (2|lam-lam0|/FWHM)^6) + 10^-4`` -- steep skirts and OD 4 blocking, as for hard-coated
band-pass filters (Macleod, *Thin-Film Optical Filters*, ch. 7); a Gaussian would leak ~2 % of the
excitation line 1.2 FWHM away, which no interference filter does.  Both are multiplied by a smooth
long-pass edge (logistic, 2 nm scale) with out-of-band floor ``10^-OD``.  Interference/LCTF angle dependence: a ray at field angle
``theta`` sees the centre wavelength blue-shifted to ``lam0 sqrt(1 - sin^2(theta)/n_eff^2)``
(Macleod, *Thin-Film Optical Filters*); pixels are binned into field-angle rings and each ring
gets its own response matrix.  The object-space chief-ray angle is used as the filter incidence
angle (the image-space angle differs by the pupil magnification; neglected).

Optical blur (spatially varying, per band)
------------------------------------------
Thin-lens circle of confusion on the sensor for an object at axial distance ``S2`` with the lens
focused at ``S1`` (image distances ``v = f S/(S - f)``, aperture ``A = f/N``)::

    c = A |v1 - v2 - delta_b| / (v2 + delta_b)

where ``delta_b = axial_chromatic_mm_per_100nm * (lam_b - 550)/100`` is the longitudinal chromatic
focal shift.  A uniform disc of diameter ``c`` has per-axis standard deviation ``c/4``; diffraction
adds the Airy sigma ``0.42 lam N (1+m)``.  The blur is applied to the band images by quantising
sigma into geometric levels, filtering the cube once per level and blending adjacent levels linearly
per pixel (so the PSF varies smoothly with depth and band).

Detector
--------
``S`` signal electrons -> PRNU gain ``(1 + prnu)`` -> + dark (``i_d t`` + DSNU + hot pixels) ->
Poisson(shot) -> + Gaussian read noise -> full-well clip -> ``/gain + offset`` -> round (ADC) ->
clip to ``[0, 2^bits - 1]``.  Every stage is a toggle of :class:`PhysicsToggles`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from spectraforge.studio import spectra
from spectraforge.studio.contracts import SPECTRAL_GRID_NM
from spectraforge.studio.schema import CameraSpec, PhysicsToggles, SensorSpec, SpectralFilterSpec

#: hc in mW s mm:  photon energy = HC / lambda[mm].
_HC_MW_S_MM = 1.98644586e-19
#: photons per (mW s) per nanometre of wavelength: lambda[nm] * 1e-6 mm / HC
PHOTONS_PER_MW_S_PER_NM = 1e-6 / _HC_MW_S_MM          # 5.0341e12
_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
_LN2 = float(np.log(2.0))
LONGPASS_EDGE_NM = 2.0          # logistic scale of the long-pass edge


# --------------------------------------------------------------------------------------------
# band list
# --------------------------------------------------------------------------------------------
def band_centres(filter_spec: SpectralFilterSpec, ex_nm: float) -> np.ndarray:
    """Nominal band-centre wavelengths (nm) for excitation ``ex_nm``.

    ``band_start_policy='excitation_offset'`` (Nuance protocol): the first band is
    ``max(band_start_nm, ex_nm + start_offset_nm)``; e.g. 430 nm excitation -> first band 450 nm.
    The band count (``n_bands``) and step are unchanged."""
    start = filter_spec.band_start_nm
    if filter_spec.band_start_policy == "excitation_offset":
        start = max(start, ex_nm + filter_spec.start_offset_nm)
    return start + filter_spec.band_step_nm * np.arange(filter_spec.n_bands, dtype=float)


def shifted_centre(lam0, theta_rad, n_eff: float):
    """Angle-of-incidence blue shift ``lam0 sqrt(1 - sin^2(theta)/n_eff^2)``."""
    return np.asarray(lam0, dtype=float) * np.sqrt(1.0 - (np.sin(theta_rad) / n_eff) ** 2)


#: super-Gaussian order and blocking floor of the interference (filter-wheel) band-pass model
INTERFERENCE_ORDER = 6.0
INTERFERENCE_BLOCKING = 1e-4


def passband(grid: np.ndarray, centre_nm, fwhm_nm: float, order: float = 2.0, floor: float = 0.0) -> np.ndarray:
    """Peak-1 pass-band ``exp(-ln2 (2|x|/FWHM)^order) + floor``: ``order=2`` is the Gaussian, larger orders
    give the flat-top / steep-skirt shape of interference filters.  ``centre_nm`` may be an array (B,1)."""
    x = 2.0 * np.abs(np.asarray(grid) - centre_nm) / fwhm_nm
    return np.exp(-_LN2 * x ** order) + floor


def longpass(grid: np.ndarray, edge_nm: float | None, od: float) -> np.ndarray:
    """Smooth long-pass transmission with out-of-band floor ``10^-od`` (1 everywhere if no edge)."""
    if edge_nm is None:
        return np.ones_like(grid)
    floor = 10.0 ** (-od)
    z = np.clip((grid - edge_nm) / LONGPASS_EDGE_NM, -700, 700)
    return floor + (1.0 - floor) / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------------------------
# lens radiometry
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraOptics:
    """Constants of the lens + pixel radiometry for one camera."""

    magnification: float
    aperture_factor: float        # k0 = pi T / (4 N^2 (1+m)^2)  [E_img = k0 L cos^4]
    pixel_area_mm2: float
    pixel_pitch_mm: float         # effective (binned) pitch
    working_f_number: float       # N (1 + m)

    @staticmethod
    def from_camera(cam: CameraSpec, focus_distance_mm: float | None = None) -> "CameraOptics":
        lens, sens = cam.lens, cam.sensor
        s1 = float(focus_distance_mm or lens.focus_distance_mm)
        f = lens.focal_length_mm
        m = f / max(s1 - f, 1e-6)
        pitch = sens.pixel_pitch_um * sens.binning * 1e-3
        k0 = np.pi * lens.transmission / (4.0 * lens.f_number ** 2 * (1.0 + m) ** 2)
        return CameraOptics(m, float(k0), pitch * pitch, pitch, lens.f_number * (1.0 + m))

    def electron_scale(self, exposure_ms: float) -> float:
        """``k0 * A * t`` -- multiply by ``cos^4 * integral(L R Tp QE lam/hc)`` to get electrons."""
        return self.aperture_factor * self.pixel_area_mm2 * exposure_ms * 1e-3


def vignetting(field_angle_rad: np.ndarray, enabled: bool = True) -> np.ndarray:
    """cos^4 fall-off."""
    if not enabled:
        return np.ones_like(np.asarray(field_angle_rad, dtype=float))
    return np.cos(field_angle_rad) ** 4


# --------------------------------------------------------------------------------------------
# band response matrices
# --------------------------------------------------------------------------------------------
@dataclass
class BandResponses:
    """Spectral response of every band for every field-angle ring.

    ``rad[r, b, :]`` -- pass-band x long-pass (dimensionless): band radiance = ``L @ rad[r, b]``.
    ``el[r, b, :]``  -- ``rad * Tp_b * QE * lam * 5.03e12`` (photons per mW s per nm): electrons per
    unit ``k0 A t cos^4``.  ``window`` selects the wavelength samples where any response is non-zero.
    """

    centres_nm: np.ndarray            # (B,) nominal centres
    ring_angles_rad: np.ndarray       # (R,)
    rad: np.ndarray                   # (R, B, W) float32
    el: np.ndarray                    # (R, B, W) float32
    window: np.ndarray                # (W,) bool

    def ring_index(self, field_angle_rad: np.ndarray) -> np.ndarray:
        if self.ring_angles_rad.size == 1:
            return np.zeros(np.shape(field_angle_rad), dtype=np.int16)
        edges = 0.5 * (self.ring_angles_rad[1:] + self.ring_angles_rad[:-1])
        return np.searchsorted(edges, np.abs(field_angle_rad)).astype(np.int16)


def band_responses(cam: CameraSpec, centres_nm: np.ndarray, toggles: PhysicsToggles | None = None,
                   max_field_angle_rad: float = 0.0, n_rings: int = 6,
                   grid: np.ndarray = SPECTRAL_GRID_NM) -> BandResponses:
    """Build :class:`BandResponses` for the band list ``centres_nm``."""
    tg = toggles or PhysicsToggles()
    f = cam.filter
    if tg.filter_angle_shift and max_field_angle_rad > 1e-6:
        ring_angles = np.linspace(0.0, max_field_angle_rad, n_rings)
    else:
        ring_angles = np.zeros(1)
    lp = longpass(grid, f.emission_longpass_nm, f.longpass_od)
    qe_ = spectra.qe(cam.sensor.qe_curve, grid)
    tp = spectra.filter_peak_transmission(f.peak_transmission_curve, centres_nm)      # (B,)
    photon = grid * PHOTONS_PER_MW_S_PER_NM
    B, W, R = len(centres_nm), len(grid), len(ring_angles)
    rad = np.empty((R, B, W), dtype=np.float64)
    for r, th in enumerate(ring_angles):
        c = shifted_centre(centres_nm, th, f.interference_n_eff)                      # (B,)
        pb = (passband(grid[None, :], c[:, None], f.fwhm_nm, INTERFERENCE_ORDER, INTERFERENCE_BLOCKING)
              if f.kind == "filter_wheel" else passband(grid[None, :], c[:, None], f.fwhm_nm))
        rad[r] = pb * lp[None, :]
    el = rad * (tp[None, :, None] * (qe_ * photon)[None, None, :])
    window = (rad.max(axis=(0, 1)) > 1e-12)
    return BandResponses(np.asarray(centres_nm, dtype=float), ring_angles,
                         rad.astype(np.float32), el.astype(np.float32), window)


def project_bands(spectral: np.ndarray, responses: BandResponses, ring_idx: np.ndarray,
                  wsel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project spectral radiance ``(n, Wsel)`` onto bands.

    ``spectral`` must be sampled on ``grid[wsel]`` (default ``responses.window``).  Returns ``(radiance (n, B), electrons_per_unit (n, B))`` -- see :class:`BandResponses`.
    The grid step (1 nm) is folded in by the caller through ``step_nm``-weighted responses: here the
    integral is a plain sum with unit weight, correct for the 1 nm engine grid.  Radiance and electron
    responses are stacked so one GEMM per field-angle ring does both (the stack is cached per window)."""
    n = spectral.shape[0]
    B = responses.rad.shape[1]
    win = responses.window if wsel is None else wsel
    out = np.empty((n, 2 * B), dtype=np.float32)
    cache = responses.__dict__.setdefault("_stack_cache", {})
    wkey = hash(win.tobytes())
    spectral = np.asarray(spectral, dtype=np.float32)
    for r in np.unique(ring_idx):
        mat = cache.get((int(r), wkey))
        if mat is None:
            mat = np.ascontiguousarray(
                np.concatenate([responses.rad[r][:, win], responses.el[r][:, win]], axis=0).T)   # (Wsel, 2B)
            cache[(int(r), wkey)] = mat
        with np.errstate(all="ignore"):     # Accelerate BLAS raises spurious FP flags on matmul
            if r == ring_idx.flat[0] and np.all(ring_idx == r):
                out[:] = spectral @ mat
            else:
                sel = ring_idx == r
                out[sel] = spectral[sel] @ mat
    return out[:, :B], out[:, B:]


# --------------------------------------------------------------------------------------------
# optical blur
# --------------------------------------------------------------------------------------------
def blur_sigma_px(axial_distance_mm: np.ndarray, cam: CameraSpec, centres_nm: np.ndarray,
                  toggles: PhysicsToggles | None = None, focus_distance_mm: float | None = None) -> np.ndarray:
    """Per-pixel, per-band Gaussian blur sigma (in *native* sensor pixels) -> ``(P, B)``.

    ``axial_distance_mm``: object distance along the optical axis for every pixel (inf = at focus)."""
    tg = toggles or PhysicsToggles()
    lens, sens = cam.lens, cam.sensor
    f = lens.focal_length_mm
    s1 = float(focus_distance_mm or lens.focus_distance_mm)
    pitch = sens.pixel_pitch_um * sens.binning * 1e-3
    s2 = np.where(np.isfinite(axial_distance_mm), axial_distance_mm, s1)
    s2 = np.maximum(s2, f * 1.01)
    if not tg.defocus:
        s2 = np.full_like(s2, s1)
    v1 = f * s1 / (s1 - f)
    v2 = f * s2 / (s2 - f)                                                     # (P,)
    delta = (lens.axial_chromatic_mm_per_100nm * (centres_nm - 550.0) / 100.0) if tg.chromatic_aberration \
        else np.zeros_like(centres_nm)                                          # (B,)
    aperture = f / lens.f_number
    coc = aperture * np.abs(v1 - v2[:, None] - delta[None, :]) / (v2[:, None] + delta[None, :])
    var = (coc / 4.0 / pitch) ** 2
    if tg.diffraction:
        m = f / (s1 - f)
        sig_d = 0.42 * (centres_nm * 1e-6) * lens.f_number * (1.0 + m) / pitch    # (B,)
        var = var + (sig_d ** 2)[None, :]
    return np.sqrt(var)


def blur_cube(cube: np.ndarray, sigma_px: np.ndarray, min_sigma: float = 0.25, ratio: float = 1.6,
              max_levels: int = 14) -> np.ndarray:
    """Apply spatially-/band-varying Gaussian blur to ``cube (H, W, B)``.

    ``sigma_px`` has shape ``(H, W, B)`` (output-pixel units).  sigma is quantised to geometric
    levels; adjacent levels are blended linearly (hat functions) so the result is continuous in
    sigma.  Sigma below ``min_sigma`` blends toward the un-blurred image."""
    smax = float(sigma_px.max()) if sigma_px.size else 0.0
    if smax < 0.05:
        return cube
    levels = [0.0, min(min_sigma, smax)]
    while levels[-1] * ratio < smax and len(levels) < max_levels - 1:
        levels.append(levels[-1] * ratio)
    if smax > levels[-1] + 1e-9:
        levels.append(smax)
    levels = np.array(levels)
    s = np.minimum(sigma_px, levels[-1])
    out = np.zeros_like(cube, dtype=np.float32)
    for j, lv in enumerate(levels):
        if j == 0:
            w = np.clip((levels[1] - s) / (levels[1] - levels[0]), 0.0, 1.0)
        else:
            lo = levels[j - 1]
            w = np.clip((s - lo) / (lv - lo), 0.0, 1.0)
            if j + 1 < len(levels):
                hi = levels[j + 1]
                w = np.where(s > lv, np.clip((hi - s) / (hi - lv), 0.0, 1.0), w)
        if not np.any(w > 0):
            continue
        layer = cube if lv == 0.0 else gaussian_filter(cube, sigma=(lv, lv, 0.0), mode="nearest")
        out += (w * layer).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------------
# detector
# --------------------------------------------------------------------------------------------
@dataclass
class FixedPattern:
    """Deterministic per-sensor fixed-pattern terms (same for every frame of a capture)."""

    prnu: np.ndarray        # (H, W) multiplicative gain deviation (fraction)
    dsnu_e: np.ndarray      # (H, W) additive dark offset (e-)
    hot_e_per_s: np.ndarray  # (H, W) extra dark rate of hot pixels (e-/s), mostly 0


def make_fixed_pattern(sensor: SensorSpec, h: int, w: int, seed: int) -> FixedPattern:
    """PRNU / DSNU / hot pixels, deterministic in ``(seed, h, w)``."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x5E45, h, w]))
    prnu = rng.normal(0.0, sensor.prnu_pct / 100.0, (h, w))
    dsnu = rng.normal(0.0, sensor.dsnu_e, (h, w))
    hot = np.zeros((h, w))
    n_hot = int(round(sensor.hot_pixel_fraction * h * w))
    if n_hot > 0:
        idx = rng.choice(h * w, size=n_hot, replace=False)
        hot.flat[idx] = rng.uniform(100.0, 1000.0, size=n_hot)
    return FixedPattern(prnu, dsnu, hot)


def mean_dark_electrons(sensor: SensorSpec, exposure_ms: float, toggles: PhysicsToggles,
                        fp: FixedPattern | None) -> np.ndarray | float:
    """Expected dark electrons per pixel for one exposure (map if fixed pattern is on)."""
    t = exposure_ms * 1e-3
    dark = sensor.dark_current_e_per_s * t if toggles.dark_current else 0.0
    if fp is None or not toggles.fixed_pattern_noise:
        return dark
    extra = fp.hot_e_per_s * t if toggles.dark_current else 0.0
    return np.maximum(dark + fp.dsnu_e + extra, 0.0)


def detect(signal_e: np.ndarray, exposure_ms: float, sensor: SensorSpec, toggles: PhysicsToggles,
           fp: FixedPattern | None, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Detector model: expected signal electrons ``(H, W, B)`` -> raw ADU ``(H, W, B)`` float32.

    Returns ``(adu, saturated_fraction)`` where the fraction counts samples at full well / ADC max."""
    s = signal_e.astype(np.float64)
    use_fp = fp is not None and toggles.fixed_pattern_noise
    if use_fp:
        s = s * (1.0 + fp.prnu)[:, :, None]
    dark = mean_dark_electrons(sensor, exposure_ms, toggles, fp if use_fp else None)
    mean = s + (np.asarray(dark)[:, :, None] if np.ndim(dark) else dark)
    mean = np.maximum(mean, 0.0)
    e = rng.poisson(mean).astype(np.float64) if toggles.shot_noise else mean
    if toggles.read_noise and sensor.read_noise_e > 0:
        e = e + rng.normal(0.0, sensor.read_noise_e, e.shape)
    sat = 0.0
    if toggles.saturation:
        sat = float(np.mean(e >= sensor.full_well_e))
        e = np.minimum(e, sensor.full_well_e)
    adu = e / sensor.gain_e_per_adu + sensor.offset_adu
    if toggles.quantization:
        adu = np.round(adu)
    if toggles.saturation:
        top = float(2 ** sensor.bit_depth - 1)
        sat = max(sat, float(np.mean(adu >= top)))
        adu = np.clip(adu, 0.0, top)
    return adu.astype(np.float32), sat


def snr_estimate(signal_e: np.ndarray, exposure_ms: float, sensor: SensorSpec,
                 toggles: PhysicsToggles) -> np.ndarray:
    """Shot + dark + read noise SNR for expected signal electrons (no fixed pattern)."""
    dark = mean_dark_electrons(sensor, exposure_ms, toggles, None)
    var = (signal_e if toggles.shot_noise else 0.0) + (dark if toggles.shot_noise else 0.0)
    if toggles.read_noise:
        var = var + sensor.read_noise_e ** 2
    return signal_e / np.sqrt(np.maximum(var, 1e-12))
