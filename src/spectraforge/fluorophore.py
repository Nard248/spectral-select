"""Fluorophore: parametric Gaussian excitation/emission spectra (dilute-regime model)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493  # 1 / (2*sqrt(2*ln2))


@dataclass(frozen=True)
class Fluorophore:
    """A single fluorophore defined by Gaussian excitation/emission bands.

    The dilute-regime amplitude of this fluorophore's contribution is
    ``extinction * quantum_yield * excitation(λex) * emission(λem) * concentration``.
    """

    name: str
    ex_peak_nm: float
    ex_fwhm_nm: float
    em_peak_nm: float
    em_fwhm_nm: float
    quantum_yield: float = 0.5      # Phi
    extinction: float = 1.0         # relative epsilon (unitless in v1)

    def excitation(self, wl):
        """Relative absorption probability at ``wl``, normalized to peak 1."""
        wl = np.asarray(wl, dtype=float)
        sigma = self.ex_fwhm_nm * _FWHM_TO_SIGMA
        return np.exp(-0.5 * ((wl - self.ex_peak_nm) / sigma) ** 2)

    def emission(self, wl):
        """Emission shape over grid ``wl``, normalized to unit sum (area) on that grid."""
        wl = np.asarray(wl, dtype=float)
        sigma = self.em_fwhm_nm * _FWHM_TO_SIGMA
        g = np.exp(-0.5 * ((wl - self.em_peak_nm) / sigma) ** 2)
        total = g.sum()
        return g / total if total > 0 else g
