"""Instrument artifacts: Rayleigh / 2nd-order scatter lines and detector noise."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass
class ArtifactConfig:
    rayleigh_strength: float = 0.0   # 0 disables scatter
    rayleigh_fwhm: float = 10.0
    second_order: bool = True
    photon_scale: float = 0.0        # 0 disables shot noise; else Poisson(signal*scale)/scale
    read_sigma: float = 0.0          # additive Gaussian read noise


def _line(em_grid, center, fwhm):
    sigma = fwhm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((em_grid - center) / sigma) ** 2)


def add_scatter_lines(cube, ex, em_grid, cfg, scale, reflectance=None) -> None:
    """Add Rayleigh (em=ex) and optional 2nd-order (em=2*ex) lines, in place."""
    if cfg.rayleigh_strength <= 0:
        return
    h, w, _ = cube.shape
    refl = np.ones((h, w)) if reflectance is None else reflectance
    bump = cfg.rayleigh_strength * scale * _line(em_grid, ex, cfg.rayleigh_fwhm)
    cube += refl[:, :, None] * bump[None, None, :]
    if cfg.second_order and (2 * ex) <= em_grid[-1]:
        bump2 = cfg.rayleigh_strength * scale * _line(em_grid, 2 * ex, cfg.rayleigh_fwhm)
        cube += refl[:, :, None] * bump2[None, None, :]


def add_noise(cube, cfg, rng):
    """Return cube with Poisson shot noise + Gaussian read noise (seeded by ``rng``)."""
    out = cube
    if cfg.photon_scale > 0:
        out = rng.poisson(np.clip(out, 0, None) * cfg.photon_scale) / cfg.photon_scale
    if cfg.read_sigma > 0:
        out = out + rng.normal(0.0, cfg.read_sigma, size=out.shape)
    return out
