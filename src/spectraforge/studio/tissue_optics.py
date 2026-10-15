"""Literature-grade tissue optics: chromophore absorption, Mie/Rayleigh scattering, phantoms.

All spectra are tabulated literature data stored as plain CSV with citation headers in
``spectraforge/studio/data/`` (regenerate with ``scripts/build_tissue_data.py``).  Units
are stated for every function; the engine-wide convention is **1/mm** for coefficients and **nm**
for wavelength.

Equations
---------
Jacques 2013, Phys. Med. Biol. 58, R37 (doi:10.1088/0031-9155/58/11/R37), eq. 1 -- tissue absorption::

    mua(lam) = B S mua_HbO2 + B (1-S) mua_Hb + W mua_water + F mua_lipid + M mua_mel + (dyes)
    mua_HbO2 [1/cm] = ln(10) eps_HbO2(lam) [1/cm/M] * C_Hb [g/L] / 64500 [g/mol]    (C_Hb = 150 g/L whole blood)
    mua_mel  [1/cm] = 519 (lam / 500 nm)^-3.5                                       (pure melanosome)

Jacques 2013, eq. 2 -- reduced scattering::

    mus'(lam) = a' [ f_ray (lam/500 nm)^-4 + (1 - f_ray)(lam/500 nm)^-b_mie ]

Intralipid-10 % (van Staveren et al. 1991, Appl. Opt. 30, 4507, doi:10.1364/AO.30.004507; as summarised
on omlc.org/spectra/intralipid)::

    mus(lam) [1/cm] = 2.54e9 * lam[nm]^-2.4         g(lam) = 1.1 - 0.58e-3 * lam[nm]        mus' = mus (1 - g)

Data sources
------------
* haemoglobin: S. Prahl compilation (Gratzer; Kollias), omlc.org/spectra/hemoglobin, 250-1000 nm;
* water: Pope & Fry 1997, Appl. Opt. 36, 8710 (380-727 nm) + Segelstein 1981 (the rest, ultimately
  Hale & Querry 1973, Appl. Opt. 12, 555);
* lipid: van Veen et al. 2005, J. Biomed. Opt. 10(5), 054004 (429-1098 nm);
* bilirubin (chloroform) and beta-carotene (hexane): PhotochemCAD (Du 1998; Dixon 2005);
* melanosome: Jacques 2013 analytic fit.

All tables are real downloads (none is reconstructed).  Pigments measured in organic solvent are
a fair proxy for the tissue-bound pigment but their peaks are shifted by up to ~10-20 nm in tissue.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

#: molar mass of the haemoglobin tetramer (g/mol), Prahl
HB_MOLAR_MASS_G_PER_MOL = 64_500.0
LN10 = float(np.log(10.0))

CHROMOPHORES = ("hbo2", "hb", "water", "lipid", "melanosome", "bilirubin", "beta_carotene")


@lru_cache(maxsize=None)
def _table(name: str) -> np.ndarray:
    """Load ``data/<name>.csv`` (``#`` header lines skipped, first row = column labels)."""
    lines = [ln for ln in (DATA_DIR / f"{name}.csv").read_text().splitlines() if ln and not ln.startswith("#")]
    return np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])


def _lin(tab: np.ndarray, col: int, nm) -> np.ndarray:
    return np.interp(np.asarray(nm, dtype=float), tab[:, 0], tab[:, col])


def _log(tab: np.ndarray, col: int, nm) -> np.ndarray:
    return np.exp(np.interp(np.asarray(nm, dtype=float), tab[:, 0], np.log(tab[:, col])))


# --------------------------------------------------------------------------------------------
# primitive spectra
# --------------------------------------------------------------------------------------------
def hemoglobin_extinction(nm, oxygenated: bool = True) -> np.ndarray:
    """Molar extinction (decadic) of HbO2 / Hb in water, 1/cm/M (Prahl)."""
    return _lin(_table("hemoglobin_prahl"), 1 if oxygenated else 2, nm)


def water_mua_per_cm(nm) -> np.ndarray:
    """Pure-water absorption coefficient (1/cm), interpolated in log space."""
    return _log(_table("water_pure"), 1, nm)


def lipid_mua_per_cm(nm) -> np.ndarray:
    """Pure-fat absorption coefficient (1/cm), log-interpolated; held constant outside 429-1098 nm."""
    return _log(_table("lipid_vanveen"), 1, nm)


def melanosome_mua_per_cm(nm) -> np.ndarray:
    """Pure-melanosome absorption ``519 (lam/500)^-3.5`` 1/cm (Jacques 2013)."""
    return 519.0 * (np.asarray(nm, dtype=float) / 500.0) ** -3.5


def bilirubin_extinction(nm) -> np.ndarray:
    """Bilirubin (chloroform) molar extinction, 1/cm/M (PhotochemCAD); 0 beyond the tabulated range."""
    t = _table("bilirubin_photochemcad")
    return np.interp(np.asarray(nm, dtype=float), t[:, 0], t[:, 1], left=t[0, 1], right=0.0)


def beta_carotene_extinction(nm) -> np.ndarray:
    """beta-Carotene (hexane) molar extinction, 1/cm/M (PhotochemCAD); 0 beyond the tabulated range."""
    t = _table("beta_carotene_photochemcad")
    return np.interp(np.asarray(nm, dtype=float), t[:, 0], t[:, 1], left=t[0, 1], right=0.0)


# --------------------------------------------------------------------------------------------
# absorption of tissue
# --------------------------------------------------------------------------------------------
def chromophore_mua_per_mm(name: str, nm, amount: float = 1.0) -> np.ndarray:
    """Absorption coefficient (1/mm) of ``amount`` of one chromophore at wavelength(s) ``nm``.

    ``amount`` units by chromophore:

    * ``hbo2`` / ``hb``: haemoglobin concentration in g/L of *the tissue* (mua = ln10 eps C / 64500, so
      150 g/L is whole blood);
    * ``water`` / ``lipid`` / ``melanosome``: volume fraction (0..1); 1 = the pure substance;
    * ``bilirubin`` / ``beta_carotene``: concentration in micromolar (uM).
    """
    nm = np.asarray(nm, dtype=float)
    if name in ("hbo2", "hb"):
        eps = hemoglobin_extinction(nm, oxygenated=(name == "hbo2"))
        return LN10 * eps * amount / HB_MOLAR_MASS_G_PER_MOL / 10.0
    if name == "water":
        return amount * water_mua_per_cm(nm) / 10.0
    if name == "lipid":
        return amount * lipid_mua_per_cm(nm) / 10.0
    if name == "melanosome":
        return amount * melanosome_mua_per_cm(nm) / 10.0
    if name == "bilirubin":
        return LN10 * bilirubin_extinction(nm) * amount * 1e-6 / 10.0
    if name == "beta_carotene":
        return LN10 * beta_carotene_extinction(nm) * amount * 1e-6 / 10.0
    raise ValueError(f"unknown chromophore '{name}' (known: {', '.join(CHROMOPHORES)})")


def whole_blood_mua_per_mm(nm, saturation: float = 1.0, hemoglobin_g_per_L: float = 150.0) -> np.ndarray:
    """Absorption of whole blood (1/mm) at oxygen saturation ``saturation``."""
    return (chromophore_mua_per_mm("hbo2", nm, hemoglobin_g_per_L * saturation)
            + chromophore_mua_per_mm("hb", nm, hemoglobin_g_per_L * (1.0 - saturation)))


def tissue_mua_per_mm(comp, nm) -> np.ndarray:
    """Jacques 2013 eq. 1 for a :class:`~spectraforge.studio.schema.TissueComposition` (1/mm)."""
    nm = np.asarray(nm, dtype=float)
    out = np.zeros_like(nm)
    if comp is None:
        return out
    b, s = comp.blood_volume_fraction, comp.oxygen_saturation
    if b > 0:
        out = out + b * whole_blood_mua_per_mm(nm, s, comp.hemoglobin_g_per_L)
    if comp.water_fraction > 0:
        out = out + chromophore_mua_per_mm("water", nm, comp.water_fraction)
    if comp.lipid_fraction > 0:
        out = out + chromophore_mua_per_mm("lipid", nm, comp.lipid_fraction)
    if comp.melanosome_fraction > 0:
        out = out + chromophore_mua_per_mm("melanosome", nm, comp.melanosome_fraction)
    if comp.bilirubin_uM > 0:
        out = out + chromophore_mua_per_mm("bilirubin", nm, comp.bilirubin_uM)
    if comp.beta_carotene_uM > 0:
        out = out + chromophore_mua_per_mm("beta_carotene", nm, comp.beta_carotene_uM)
    return out


# --------------------------------------------------------------------------------------------
# scattering
# --------------------------------------------------------------------------------------------
def mus_prime_mie_rayleigh(mus_prime_500: float, f_rayleigh: float, b_mie: float, nm) -> np.ndarray:
    """Jacques 2013 eq. 2 (any consistent unit): ``a' [f (lam/500)^-4 + (1-f)(lam/500)^-b]``."""
    x = np.asarray(nm, dtype=float) / 500.0
    return mus_prime_500 * (f_rayleigh * x ** -4.0 + (1.0 - f_rayleigh) * x ** -b_mie)


#: Jacques 2013, Table 2 (mean over the cited studies): ``name -> (a' [1/cm], f_ray, b_mie)``;
#: omlc.org/news/dec14/Jacques_PMB2013/table2_JacquesPMB2013.csv
JACQUES2013_TABLE2 = {
    "skin": (48.0, 0.409, 0.702),
    "brain": (27.4, 0.315, 1.087),
    "breast": (18.7, 0.288, 0.685),
    "bone": (15.3, 0.022, 0.326),
    "soft": (19.1, 0.153, 1.091),
    "fibrous": (29.2, 0.489, 0.644),
    "fatty": (19.3, 0.174, 0.447),
}


def jacques_generic_mus_prime_per_mm(kind: str, nm) -> np.ndarray:
    """mus' (1/mm) of a Jacques 2013 Table-2 average tissue ``kind``."""
    a, f, b = JACQUES2013_TABLE2[kind]
    return mus_prime_mie_rayleigh(a / 10.0, f, b, nm)


def intralipid_mus_per_mm(nm, percent: float = 10.0) -> np.ndarray:
    """Scattering coefficient mus (1/mm) of Intralipid at ``percent`` lipid (van Staveren 1991, valid
    400-1100 nm at 10 %; scaled linearly in concentration for dilutions, valid in the independent
    scattering regime)."""
    nm = np.asarray(nm, dtype=float)
    return 2.54e9 * nm ** -2.4 / 10.0 * (percent / 10.0)


def intralipid_g(nm) -> np.ndarray:
    """Intralipid anisotropy ``g = 1.1 - 0.58e-3 lam[nm]`` (van Staveren 1991, Mie fit)."""
    return 1.1 - 0.58e-3 * np.asarray(nm, dtype=float)


def intralipid_mus_prime_per_mm(nm, percent: float = 10.0) -> np.ndarray:
    return intralipid_mus_per_mm(nm, percent) * (1.0 - intralipid_g(nm))


def fit_power_law(nm, mus_prime, ref_nm: float = 500.0) -> tuple[float, float]:
    """Least-squares fit of ``mus'(lam) = a (lam/ref)^-b`` in log space -> ``(a, b)``."""
    x = np.log(np.asarray(nm, dtype=float) / ref_nm)
    y = np.log(np.asarray(mus_prime, dtype=float))
    slope, icpt = np.polyfit(x, y, 1)
    return float(np.exp(icpt)), float(-slope)


# --------------------------------------------------------------------------------------------
# refractive-index dispersion (utility; the schema keeps a scalar n)
# --------------------------------------------------------------------------------------------
# Daimon & Masumura 2007, Appl. Opt. 46, 3811 (doi:10.1364/AO.46.003811): Sellmeier fit of pure water at
# 20 C, 182 nm - 1129 nm.  n^2 - 1 = sum_i B_i lam^2 / (lam^2 - C_i), lam in um.
_WATER_SELLMEIER = ((5.684027565e-1, 5.101829712e-3), (1.726177391e-1, 1.821153936e-2),
                    (2.086189578e-2, 2.620722293e-2), (1.130748688e-1, 1.069792721e1))


def water_refractive_index(nm) -> np.ndarray:
    """Refractive index of pure water at 20 C (Daimon & Masumura 2007 Sellmeier fit)."""
    lam2 = (np.asarray(nm, dtype=float) * 1e-3) ** 2
    return np.sqrt(1.0 + sum(b * lam2 / (lam2 - c) for b, c in _WATER_SELLMEIER))


def cauchy_index(nm, n_ref: float, ref_nm: float = 589.0, b_nm2: float | None = None) -> np.ndarray:
    """Cauchy dispersion ``n(lam) = A + B / lam^2`` anchored so that ``n(ref_nm) = n_ref``.

    ``b_nm2`` (nm^2) defaults to the Cauchy slope of pure water between 450 and 800 nm (about 3.2e3 nm^2),
    the usual approximation for aqueous tissue: dispersion of tissue ground substance follows that of
    water, shifted to the tissue's index (documented approximation, no tissue-specific table)."""
    if b_nm2 is None:
        l1, l2 = 450.0, 800.0
        n1, n2 = water_refractive_index(l1), water_refractive_index(l2)
        b_nm2 = float((n1 - n2) / (l1 ** -2 - l2 ** -2))
    nm = np.asarray(nm, dtype=float)
    return n_ref + b_nm2 * (nm ** -2 - ref_nm ** -2)


def tissue_refractive_index(nm, n_ref: float = 1.38, ref_nm: float = 589.0) -> np.ndarray:
    """Dispersive tissue index (Cauchy, water-like slope) -- a utility; materials keep a scalar n."""
    return cauchy_index(nm, n_ref, ref_nm)
