"""Tabulated, cited, smooth spectra used by the SpectralForge Studio engine.

Every table is coarse (<= 10 nm spacing, some regions 20-25 nm where the curve is smooth) and is
interpolated *linearly in log or linear space as noted* onto any wavelength array, valid over
250-900 nm (values are held constant outside the tabulated range).  They are digitised from the
cited compilations to a few percent -- good enough for a simulator whose other uncertainties
(concentrations, scattering) are far larger -- and every curve is smooth so that band integrals do
not alias.

Contents
--------
* absorbers: :func:`water_mua_per_mm`, :func:`hemoglobin_shape`, :func:`melanin_shape`,
  :func:`carbon_black_shape`, :func:`absorber_mua_per_mm`
* detector / filter curves: :func:`qe`, :func:`filter_peak_transmission`
* ambient light: :func:`room_light_relative_spectrum`, :func:`room_light_irradiance_mW_mm2_nm`,
  :func:`cie_v_lambda`, :func:`planck_relative`

Units: absorption in 1/mm, irradiance in mW/mm^2/nm, wavelengths in nm.

References
----------
Pope & Fry 1997, Appl. Opt. 36, 8710 (water 380-727 nm); Segelstein 1981 (water, >700 nm);
Quickenden & Irvin 1980 (UV water); Prahl, "Tabulated molar extinction coefficient for
hemoglobin in water" (omlc.org/spectra/hemoglobin); Jacques 2013, Phys. Med. Biol. 58, R37
(melanin ~ lambda^-3.46, 10.1088/0031-9155/58/11/R37); Wyman, Sloan & Shirley 2013, JCGT 2, 1
(analytic CIE 1931 fits); Nuance/CRi LCTF and Sony ICX285 / back-thinned CCD data sheets for the
camera curves.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _interp(table: np.ndarray, nm) -> np.ndarray:
    """Linear interpolation of an ``(N, 2)`` table; constant extrapolation."""
    nm = np.asarray(nm, dtype=float)
    return np.interp(nm, table[:, 0], table[:, 1])


def _interp_log(table: np.ndarray, nm) -> np.ndarray:
    """Interpolation linear in log(value) (for quantities spanning decades, e.g. water absorption)."""
    nm = np.asarray(nm, dtype=float)
    return np.exp(np.interp(nm, table[:, 0], np.log(table[:, 1])))


# --------------------------------------------------------------------------------------------
# Pure water  (Pope & Fry 1997 / Segelstein 1981 / Quickenden & Irvin 1980)
# --------------------------------------------------------------------------------------------
# absorption coefficient a_w(lambda) in 1/m, digitised
_WATER_PER_M = np.array([
    (250, 0.30), (280, 0.15), (300, 0.085), (320, 0.045), (340, 0.025), (360, 0.016), (380, 0.0114),
    (400, 0.0066), (420, 0.0045), (440, 0.0063), (460, 0.0098), (480, 0.0150), (500, 0.0204),
    (520, 0.0409), (540, 0.0474), (560, 0.0709), (580, 0.1175), (600, 0.2224), (620, 0.2755),
    (640, 0.3200), (660, 0.4100), (680, 0.4700), (700, 0.6240), (720, 1.30), (740, 2.20),
    (760, 2.60), (780, 2.10), (800, 2.00), (820, 2.60), (840, 3.40), (860, 4.40), (880, 5.40),
    (900, 6.80),
])


def water_mua_per_mm(nm) -> np.ndarray:
    """Pure-water absorption coefficient (1/mm).  Tiny in the blue (~5e-6 /mm at 420 nm),
    ~6e-4 /mm at 700 nm, ~2e-3 /mm at 800 nm."""
    return _interp_log(np.column_stack([_WATER_PER_M[:, 0], _WATER_PER_M[:, 1]]), nm) * 1e-3


# --------------------------------------------------------------------------------------------
# Haemoglobin (Prahl compilation), molar extinction in 1e3 M^-1 cm^-1: (nm, HbO2, Hb)
# --------------------------------------------------------------------------------------------
_HB = np.array([
    (250, 110, 105), (280, 125, 120), (300, 80, 85), (320, 55, 55), (340, 50, 52), (360, 60, 65),
    (380, 110, 120), (400, 266, 223), (410, 466, 330), (415, 524, 390), (420, 480, 450),
    (430, 330, 528), (440, 190, 460), (450, 62.8, 103), (460, 44.5, 98), (470, 33, 60),
    (480, 26.6, 38), (500, 20.9, 20.0), (520, 40.6, 40.0), (540, 53.2, 46.6), (556, 38, 54.1),
    (560, 33, 53), (576, 55.5, 34), (590, 14, 36), (600, 3.2, 14.7), (620, 0.9, 6.7),
    (650, 0.37, 3.75), (700, 0.29, 1.8), (750, 0.52, 1.4), (800, 0.82, 0.77), (850, 1.06, 0.69),
    (900, 1.20, 0.76),
])


def hemoglobin_shape(nm, oxygenation: float = 0.9) -> np.ndarray:
    """Haemoglobin absorption spectrum normalised to **1 at 500 nm**.

    ``oxygenation`` is the HbO2 fraction (0 = Hb, 1 = HbO2); the blend is linear in extinction.
    The ``scale`` of an :class:`AbsorberSpec` is then mu_a(500 nm) in 1/mm."""
    nm = np.asarray(nm, dtype=float)
    ox = float(np.clip(oxygenation, 0.0, 1.0))
    e_hbo2 = np.interp(nm, _HB[:, 0], _HB[:, 1])
    e_hb = np.interp(nm, _HB[:, 0], _HB[:, 2])
    e = ox * e_hbo2 + (1 - ox) * e_hb
    ref = ox * np.interp(500.0, _HB[:, 0], _HB[:, 1]) + (1 - ox) * np.interp(500.0, _HB[:, 0], _HB[:, 2])
    return e / ref


def melanin_shape(nm) -> np.ndarray:
    """Melanin absorption ``(lambda / 500)^-3.46`` (Jacques 2013), normalised 1 at 500 nm."""
    return (np.asarray(nm, dtype=float) / 500.0) ** -3.46


def carbon_black_shape(nm) -> np.ndarray:
    """Carbon-black absorption ~ ``lambda^-1`` (soot / carbon pigment), normalised 1 at 500 nm."""
    return (np.asarray(nm, dtype=float) / 500.0) ** -1.0


def absorber_mua_per_mm(preset: str, scale: float, nm, custom=None) -> np.ndarray:
    """Background absorption coefficient (1/mm) of an :class:`AbsorberSpec`.

    ``water``: ``scale`` multiplies pure water (1 = pure water).  ``hemoglobin`` / ``melanin`` /
    ``carbon_black``: ``scale`` is mu_a at 500 nm (1/mm).  ``custom``: list of ``(nm, mua_per_mm)``
    interpolated linearly, times ``scale``.  ``none``: zeros."""
    nm = np.asarray(nm, dtype=float)
    if preset == "none":
        return np.zeros_like(nm)
    if preset == "water":
        return scale * water_mua_per_mm(nm)
    if preset == "hemoglobin":
        return scale * hemoglobin_shape(nm)
    if preset == "melanin":
        return scale * melanin_shape(nm)
    if preset == "carbon_black":
        return scale * carbon_black_shape(nm)
    if preset == "custom":
        if not custom:
            return np.zeros_like(nm)
        pts = np.array(sorted((float(a), float(b)) for a, b in custom))
        return scale * np.interp(nm, pts[:, 0], pts[:, 1])
    raise ValueError(f"unknown absorber preset '{preset}'")


# --------------------------------------------------------------------------------------------
# Detector QE curves (peak-normalised quantum efficiency, 0..1)
# --------------------------------------------------------------------------------------------
QE_FLAT = 0.8   # the ``flat`` curve: a constant 80 % (clean reference for hand calculations)

_QE = {
    # interline CCD, visible-optimised (Sony ICX285 class, Nuance FX): ~0.55-0.65 peak, weak UV/NIR
    "ccd_bv": np.array([(250, 0.0), (300, 0.02), (350, 0.08), (400, 0.38), (450, 0.58), (500, 0.66),
                        (550, 0.68), (600, 0.62), (650, 0.50), (700, 0.38), (750, 0.26),
                        (800, 0.17), (850, 0.10), (900, 0.05)]),
    # UV-enhanced back-thinned CCD
    "ccd_uv": np.array([(250, 0.28), (300, 0.42), (350, 0.60), (400, 0.76), (450, 0.84), (500, 0.88),
                        (550, 0.90), (600, 0.88), (650, 0.80), (700, 0.66), (750, 0.50),
                        (800, 0.36), (850, 0.22), (900, 0.12)]),
    # back-illuminated sCMOS (e.g. Prime BSI): ~0.95 peak
    "scmos": np.array([(250, 0.05), (300, 0.20), (350, 0.45), (400, 0.72), (450, 0.88), (500, 0.93),
                       (550, 0.95), (600, 0.95), (650, 0.90), (700, 0.80), (750, 0.66),
                       (800, 0.50), (850, 0.33), (900, 0.20)]),
}


def qe(curve: str, nm) -> np.ndarray:
    """Detector quantum efficiency (photo-electrons per photon)."""
    nm = np.asarray(nm, dtype=float)
    if curve == "flat":
        return np.full_like(nm, QE_FLAT)
    return np.clip(_interp(_QE[curve], nm), 0.0, 1.0)


# --------------------------------------------------------------------------------------------
# Spectral filter peak transmission
# --------------------------------------------------------------------------------------------
FILTER_FLAT = 0.9   # ``flat``: an interference filter's ~90 % in-band transmission

# Nuance / VariSpec VIS LCTF class: a few % at 420 nm, rising to ~45 % at 650-720 nm.
_LCTF_VIS = np.array([(250, 0.0), (380, 0.0), (400, 0.01), (420, 0.04), (440, 0.10), (460, 0.17),
                      (480, 0.24), (500, 0.30), (520, 0.35), (540, 0.38), (560, 0.40), (580, 0.42),
                      (600, 0.44), (620, 0.46), (650, 0.48), (680, 0.47), (700, 0.45),
                      (720, 0.40), (740, 0.12), (760, 0.03), (800, 0.01), (900, 0.0)])


def filter_peak_transmission(curve: str, nm) -> np.ndarray:
    """Peak (in-band) transmission of the tunable filter at band-centre ``nm``."""
    nm = np.asarray(nm, dtype=float)
    if curve == "flat":
        return np.full_like(nm, FILTER_FLAT)
    if curve == "lctf_vis":
        return _interp(_LCTF_VIS, nm)
    raise ValueError(f"unknown filter curve '{curve}'")


# --------------------------------------------------------------------------------------------
# Ambient (room) light
# --------------------------------------------------------------------------------------------
def cie_v_lambda(nm) -> np.ndarray:
    """CIE 1931 photopic luminosity V(lambda) (Wyman-Sloan-Shirley multi-lobe fit of y-bar)."""
    nm = np.asarray(nm, dtype=float)

    def lobe(mu, s1, s2):
        s = np.where(nm < mu, s1, s2)
        return np.exp(-0.5 * ((nm - mu) / s) ** 2)

    return 0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)


def planck_relative(nm, temp_k: float) -> np.ndarray:
    """Black-body spectral radiance shape (arbitrary scale) at ``temp_k``."""
    lam = np.asarray(nm, dtype=float) * 1e-9
    c2 = 1.438776877e-2  # m K
    x = np.clip(c2 / (lam * temp_k), 0, 700)
    return lam ** -5 / np.expm1(x)


def _gauss(nm, mu, fwhm):
    s = fwhm / 2.3548200450309493
    return np.exp(-0.5 * ((np.asarray(nm, dtype=float) - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


ROOM_LIGHT_KINDS = ("fluorescent_tube", "white_led", "daylight", "incandescent")


def room_light_relative_spectrum(kind: str, nm) -> np.ndarray:
    """Relative spectral power (arbitrary scale, smooth + narrow lines) of ambient lighting.

    * ``fluorescent_tube`` -- Hg lines (365, 405, 436, 546, 578 nm) + tri-phosphor (Eu2+ blue
      ~450 nm, Tb3+ green 543 nm, Eu3+ red 611 nm + 590/625 sidebands);
    * ``white_led`` -- blue pump 450 nm + broad YAG:Ce phosphor ~560 nm;
    * ``daylight`` -- 6504 K black-body approximation of D65;
    * ``incandescent`` -- Planck 2856 K (CIE illuminant A).
    """
    nm = np.asarray(nm, dtype=float)
    if kind == "fluorescent_tube":
        out = (0.020 * _gauss(nm, 365.0, 3.0) + 0.035 * _gauss(nm, 405.0, 3.0)
               + 0.120 * _gauss(nm, 436.0, 3.0) + 0.140 * _gauss(nm, 546.0, 3.0)
               + 0.060 * _gauss(nm, 578.0, 3.0)
               + 0.30 * _gauss(nm, 450.0, 50.0) + 0.55 * _gauss(nm, 543.0, 6.0)
               + 0.20 * _gauss(nm, 490.0, 60.0)
               + 0.95 * _gauss(nm, 611.0, 6.0) + 0.20 * _gauss(nm, 590.0, 8.0)
               + 0.18 * _gauss(nm, 626.0, 10.0) + 0.10 * _gauss(nm, 700.0, 12.0))
        return out
    if kind == "white_led":
        return 0.45 * _gauss(nm, 450.0, 20.0) + 1.0 * _gauss(nm, 565.0, 115.0) * 0.9
    if kind == "daylight":
        return planck_relative(nm, 6504.0) / planck_relative(560.0, 6504.0) * 0.01
    if kind == "incandescent":
        return planck_relative(nm, 2856.0) / planck_relative(560.0, 2856.0) * 0.01
    raise ValueError(f"unknown room light '{kind}'")


def room_light_irradiance_mW_mm2_nm(kind: str, lux: float, nm) -> np.ndarray:
    """Spectral irradiance on the table (mW/mm^2/nm) for an ambient illuminance ``lux``.

    The relative spectrum is scaled so that ``683 lm/W * integral V(l) E_e(l) dl = lux``
    (``E_e`` in W/m^2/nm; 1 W/m^2 = 1e-3 mW/mm^2).  Returned on the requested ``nm`` array, so the
    result integrates correctly only if ``nm`` is a dense (<= 1 nm) grid -- the normalisation is
    always done on a 1-nm 350-800 nm grid."""
    ref = np.arange(300.0, 850.0 + 1e-9, 0.5)
    rel_ref = room_light_relative_spectrum(kind, ref)
    lum = 683.0 * np.trapezoid(cie_v_lambda(ref) * rel_ref, ref)   # lux for unit scale (W/m^2/nm)
    k = float(lux) / lum if lum > 0 else 0.0                       # W/m^2/nm per unit of rel
    return room_light_relative_spectrum(kind, nm) * k * 1e-3
