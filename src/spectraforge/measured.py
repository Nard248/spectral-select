"""Measured (real) fluorophore spectra, interpolated from data points.

A drop-in for :class:`spectraforge.fluorophore.Fluorophore`: ``render`` only needs
``.excitation(wl)``, ``.emission(wl)``, ``.extinction`` and ``.quantum_yield``, so any object
honouring those works. ``MeasuredFluorophore`` interpolates measured excitation/emission curves
instead of evaluating Gaussians — real spectra are asymmetric and multi-peaked, which lets us
test whether the band-selection method's behaviour on smooth synthetic Gaussians (see the
Increment B finding) carries over to realistic spectral shapes.

Convention matches Fluorophore exactly: ``excitation`` is peak-normalized (max = 1); ``emission``
is area-normalized (unit sum) on the query grid. Outside the measured support the value is 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeasuredFluorophore:
    name: str
    ex_wavelengths: np.ndarray
    ex_values: np.ndarray
    em_wavelengths: np.ndarray
    em_values: np.ndarray
    quantum_yield: float = 0.5
    extinction: float = 1.0

    def __post_init__(self):
        # store sorted ascending (np.interp requires increasing xp) and peak-normalize excitation
        self.ex_wavelengths, ex = _sorted_xy(self.ex_wavelengths, self.ex_values)
        peak = float(ex.max()) if ex.size else 0.0
        self.ex_values = ex / peak if peak > 0 else ex
        self.em_wavelengths, self.em_values = _sorted_xy(self.em_wavelengths, self.em_values)

    @property
    def ex_peak_nm(self) -> float:
        return float(self.ex_wavelengths[int(np.argmax(self.ex_values))]) if self.ex_values.size else 0.0

    @property
    def em_peak_nm(self) -> float:
        return float(self.em_wavelengths[int(np.argmax(self.em_values))]) if self.em_values.size else 0.0

    def excitation(self, wl):
        """Relative absorption at ``wl`` (peak-normalized), 0 outside the measured range."""
        wl = np.asarray(wl, dtype=float)
        return np.interp(wl, self.ex_wavelengths, self.ex_values, left=0.0, right=0.0)

    def emission(self, wl):
        """Emission shape over grid ``wl``, area-normalized to unit sum (0 outside the range)."""
        wl = np.asarray(wl, dtype=float)
        g = np.interp(wl, self.em_wavelengths, self.em_values, left=0.0, right=0.0)
        total = g.sum()
        return g / total if total > 0 else g


def _sorted_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def from_fpbase_payload(payload: dict, quantum_yield: "float | None" = None,
                        extinction: float = 1.0) -> MeasuredFluorophore:
    """Build a MeasuredFluorophore from an FPbase-style payload.

    Expects ``{"name": ..., "spectra": [{"subtype": "EX"|"AB"|"EM", "data": [[wl, val], ...]}, ...]}``.
    The first EX/AB spectrum is the excitation; the first EM spectrum is the emission. Fetch the
    payload yourself (FPbase API / a cached JSON) — this importer needs no network.
    """
    ex = em = None
    for spec in payload.get("spectra", []):
        sub = str(spec.get("subtype", "")).upper()
        data = np.asarray(spec["data"], dtype=float)
        if sub in ("EX", "AB") and ex is None:
            ex = data
        elif sub == "EM" and em is None:
            em = data
    if ex is None or em is None:
        raise ValueError("payload must contain both an excitation (EX/AB) and an emission (EM) spectrum")
    qy = quantum_yield if quantum_yield is not None else float(payload.get("qy", 0.5))
    return MeasuredFluorophore(
        name=payload.get("name", "measured"),
        ex_wavelengths=ex[:, 0], ex_values=ex[:, 1],
        em_wavelengths=em[:, 0], em_values=em[:, 1],
        quantum_yield=qy, extinction=extinction,
    )
