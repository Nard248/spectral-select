"""Material optics: :class:`MaterialSpec` -> :class:`MaterialOptics` on the fine spectral grid.

Equations
---------
Dye absorption (molar extinction -> 1/mm)::

    eps_k(lam) = extinction_k * EXTINCTION_REFERENCE_M_CM * exc_k(lam)        [M^-1 cm^-1]
    mu_a,k(lam) = ln(10) * eps_k(lam) * c_k[M] / 10                            [1/mm]

(``/10`` converts 1/cm to 1/mm; ``exc_k`` is the peak-normalised Gaussian excitation profile of
:class:`spectraforge.Fluorophore`.)  Total absorption adds the background absorber
(:mod:`spectraforge.studio.spectra`).  Scattering is the reduced-scattering power law
``mu_s'(lam) = a (lam/500 nm)^-b`` with Henyey-Greenstein anisotropy ``g`` so that
``mu_s = mu_s' / (1 - g)``.  Stern-Volmer self-quenching (collisional, Lakowicz 2006, ch. 8)::

    Phi_eff = Phi / (1 + K_sv * sum_j c_j[uM])

The emission density ``e_k(lam)`` is the legacy emission shape divided by the grid step so that
``integral e_k dlam = 1`` (1/nm).

Toggle semantics (``PhysicsToggles``)
-------------------------------------
* ``absorption=False`` -- *no inner filter*: ``mua_per_mm`` (which the transport model uses to
  attenuate excitation and emission) drops to a negligible floor, while ``mua_dyes_per_mm`` is kept
  so dyes still absorb enough to fluoresce.  Fluorescence then scales linearly with concentration
  (dilute limit) at any concentration.
* ``scattering=False`` -- ``mus = 0`` and the material is treated as a *clear* medium.
* ``quenching=False`` -- ``Phi_eff = Phi``.

``clear`` follows :func:`is_clear_material` (``scattering.model == 'none'`` or ``mu_s' == 0``).
"""
from __future__ import annotations

import numpy as np

from spectraforge.fluorophore import Fluorophore
from spectraforge.studio import spectra
from spectraforge.studio.contracts import SPECTRAL_GRID_NM, MaterialOptics
from spectraforge.studio.schema import (
    EXTINCTION_REFERENCE_M_CM,
    Experiment,
    FluorophoreSpec,
    MaterialSpec,
    PhysicsToggles,
)

#: absorption floor (1/mm) used when absorption is disabled, avoids division by zero in kernels
MUA_FLOOR_PER_MM = 1e-9


def merged_fluorophores(experiment: Experiment) -> dict[str, FluorophoreSpec]:
    """Built-in fluorophores overridden / extended by ``experiment.fluorophores`` (by name)."""
    from spectraforge.studio.presets import builtin_fluorophores

    out = {f.name: f for f in builtin_fluorophores()}
    for f in experiment.fluorophores:
        out[f.name] = f
    return out


def to_legacy(spec: FluorophoreSpec) -> Fluorophore:
    """The legacy :class:`spectraforge.Fluorophore` carrying the spectral shape of ``spec``."""
    return Fluorophore(name=spec.name, ex_peak_nm=spec.ex_peak_nm, ex_fwhm_nm=spec.ex_fwhm_nm,
                       em_peak_nm=spec.em_peak_nm, em_fwhm_nm=spec.em_fwhm_nm,
                       quantum_yield=spec.quantum_yield, extinction=spec.extinction,
                       em_skew=spec.em_skew, vibronic=spec.vibronic)


def is_clear_material(material: MaterialSpec) -> bool:
    """True if the material has no volume scattering (closed-form transport applies)."""
    return material.scattering.model == "none" or material.scattering.mus_prime_500_per_mm <= 0.0


def dye_extinction_M_cm(spec: FluorophoreSpec, nm) -> np.ndarray:
    """Molar extinction coefficient (M^-1 cm^-1) of a fluorophore at ``nm``."""
    return spec.extinction * EXTINCTION_REFERENCE_M_CM * to_legacy(spec).excitation(nm)


def material_optics(material: MaterialSpec, fluorophores: dict[str, FluorophoreSpec],
                    grid: np.ndarray = SPECTRAL_GRID_NM, toggles: PhysicsToggles | None = None,
                    concentration_scale: dict[str, float] | None = None) -> MaterialOptics:
    """Optical properties of one uniform material on ``grid`` (see module docstring).

    ``concentration_scale`` optionally multiplies individual dye concentrations (``{dye: factor}``),
    e.g. for concentration sweeps."""
    tg = toggles or PhysicsToggles()
    grid = np.asarray(grid, dtype=float)
    step = float(np.mean(np.diff(grid))) if grid.size > 1 else 1.0

    mua_dyes: dict[str, np.ndarray] = {}
    phi_eff: dict[str, float] = {}
    emission: dict[str, np.ndarray] = {}
    total_uM = sum(d.concentration_uM * (concentration_scale or {}).get(d.fluorophore, 1.0)
                   for d in material.dyes)
    quench = 1.0 + (material.quenching_ksv_per_uM * total_uM if tg.quenching else 0.0)
    for d in material.dyes:
        if d.fluorophore not in fluorophores:
            raise KeyError(f"material '{material.id}' uses unknown fluorophore '{d.fluorophore}'")
        spec = fluorophores[d.fluorophore]
        c_M = d.concentration_uM * (concentration_scale or {}).get(d.fluorophore, 1.0) * 1e-6
        eps = dye_extinction_M_cm(spec, grid)
        mua_k = np.log(10.0) * eps * c_M / 10.0
        # several entries of the same dye in one material add up
        mua_dyes[d.fluorophore] = mua_dyes.get(d.fluorophore, 0.0) + mua_k
        phi_eff[d.fluorophore] = spec.quantum_yield / quench
        emission[d.fluorophore] = to_legacy(spec).emission(grid) / step   # integral = 1 per nm

    bg = spectra.absorber_mua_per_mm(material.background_absorption.preset,
                                     material.background_absorption.scale, grid,
                                     material.background_absorption.custom)
    if tg.absorption:
        mua_total = bg + sum(mua_dyes.values(), np.zeros_like(grid))
        mua_total = np.maximum(mua_total, MUA_FLOOR_PER_MM)
    else:
        mua_total = np.full_like(grid, MUA_FLOOR_PER_MM)

    clear = is_clear_material(material)
    sc = material.scattering
    if clear or not tg.scattering:
        mus = np.zeros_like(grid)
        clear = True
        g = 0.0
    else:
        mus_prime = sc.mus_prime_500_per_mm * (grid / 500.0) ** (-sc.power_b)
        mus = mus_prime / (1.0 - sc.g)
        g = float(sc.g)

    return MaterialOptics(material_id=material.id, wavelengths_nm=grid, mua_per_mm=mua_total,
                          mua_dyes_per_mm=mua_dyes, mus_per_mm=mus, g=g, n=float(material.refractive_index),
                          clear=clear, quantum_yields=phi_eff, emission_density=emission)


def average_over_line(arr: np.ndarray, line_shape: np.ndarray) -> float | np.ndarray:
    """Line-shape-weighted mean of ``arr`` (last axis = wavelength): ``sum(arr*w)/sum(w)``.

    ``line_shape`` need not be normalised."""
    w = np.asarray(line_shape, dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("line_shape has no weight")
    return np.tensordot(np.asarray(arr, dtype=float), w, axes=([-1], [0])) / s
