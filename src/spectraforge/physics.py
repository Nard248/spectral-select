"""Optional richer sample/optics physics for the forward model.

All effects default OFF, so ``PhysicsConfig()`` (or ``physics=None``) leaves the exactly-linear
dilute model untouched — ``render(A+B) == render(A)+render(B)``. Turn effects on to add realism:

- ``psf_sigma_px``        — spatial Gaussian point-spread blur (optical resolution), per band.
- ``inner_filter``        — Beer-Lambert primary inner-filter: excitation light is absorbed as it
                            penetrates, so local signal is attenuated by ``exp(-strength * A_ex)``
                            where ``A_ex`` is the per-pixel absorbance at the excitation wavelength.
                            This is the model's first deliberate NONLINEARITY in concentration.
- ``autofluorescence``    — a spatially-uniform broadband fluorescent background.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass
class PhysicsConfig:
    psf_sigma_px: float = 0.0
    inner_filter: bool = False
    inner_filter_strength: float = 1.0
    autofluorescence: float = 0.0        # 0 disables
    autofluor_peak_nm: float = 480.0
    autofluor_fwhm_nm: float = 150.0


def _broadband(em_grid, peak, fwhm):
    sigma = fwhm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((em_grid - peak) / sigma) ** 2)


def apply_physics(cube, cfg: PhysicsConfig, em_grid, scale, absorbance):
    """Apply (in order) inner-filter attenuation, autofluorescence, then PSF blur.

    ``absorbance`` is the per-pixel (H, W) excitation absorbance accumulated by the caller
    (``Σ_k ε_k c_k(x,y) · excitation_k(λ_ex)``); only used when ``inner_filter`` is on.
    """
    out = cube
    if cfg.inner_filter:
        factor = np.exp(-cfg.inner_filter_strength * absorbance)     # (H, W)
        out = out * factor[:, :, None]
    if cfg.autofluorescence > 0:
        bg = cfg.autofluorescence * scale * _broadband(em_grid, cfg.autofluor_peak_nm, cfg.autofluor_fwhm_nm)
        out = out + bg[None, None, :]
    if cfg.psf_sigma_px > 0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, sigma=(cfg.psf_sigma_px, cfg.psf_sigma_px, 0.0), mode="reflect")
    return out
