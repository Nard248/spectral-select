"""Spectral / emission-band filtering (Rayleigh cutoff + manual crop)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from spectral_select.types import SpectraData


def apply_rayleigh_cutoff(
    spectra: "SpectraData",
    cutoff_offset: int = 30,
    apply_second_order: bool = True,
) -> "SpectraData":
    """Remove Rayleigh and (optionally) second-order bands.

    Ported from ``HyperspectralDataLoader.apply_spectral_cutoff()``.

    Returns a new ``SpectraData``.
    """
    from spectral_select.types import ExcitationData, SpectraData as SD

    new_excitations = {}
    for ex_nm, ex in spectra.excitations.items():
        wl = np.array(ex.emission_wavelengths)
        keep = np.ones(len(wl), dtype=bool)

        # Rayleigh: remove lambda < excitation + offset
        keep &= wl >= (ex_nm + cutoff_offset)

        # Second-order: remove lambda in [2*ex - offset, 2*ex + offset]
        if apply_second_order:
            so_min = 2 * ex_nm - cutoff_offset
            so_max = 2 * ex_nm + cutoff_offset
            keep &= (wl < so_min) | (wl > so_max)

        new_excitations[ex_nm] = ExcitationData(
            excitation_nm=ex_nm,
            cube=ex.cube[:, :, keep].copy(),
            emission_wavelengths=wl[keep].tolist(),
            exposure_time=ex.exposure_time,
            laser_power=ex.laser_power,
        )

    return SD(
        excitations=new_excitations,
        mask=spectra.mask.copy() if spectra.mask is not None else None,
        sample_name=spectra.sample_name,
        loading_options=spectra.loading_options,
        metadata=dict(spectra.metadata),
    )


def apply_manual_emission_crop(
    spectra: "SpectraData",
    per_excitation_ranges: Dict[float, Tuple[float, float]],
) -> "SpectraData":
    """Keep only emission bands within ``[em_min, em_max]`` per excitation.

    *per_excitation_ranges* maps ``excitation_nm`` to ``(em_min, em_max)``.
    Excitations not in the dict are left unchanged.
    """
    from spectral_select.types import ExcitationData, SpectraData as SD

    new_excitations = {}
    for ex_nm, ex in spectra.excitations.items():
        if ex_nm in per_excitation_ranges:
            em_min, em_max = per_excitation_ranges[ex_nm]
            wl = np.array(ex.emission_wavelengths)
            keep = (wl >= em_min) & (wl <= em_max)
            new_excitations[ex_nm] = ExcitationData(
                excitation_nm=ex_nm,
                cube=ex.cube[:, :, keep].copy(),
                emission_wavelengths=wl[keep].tolist(),
                exposure_time=ex.exposure_time,
                laser_power=ex.laser_power,
            )
        else:
            new_excitations[ex_nm] = ExcitationData(
                excitation_nm=ex_nm,
                cube=ex.cube.copy(),
                emission_wavelengths=list(ex.emission_wavelengths),
                exposure_time=ex.exposure_time,
                laser_power=ex.laser_power,
            )

    return SD(
        excitations=new_excitations,
        mask=spectra.mask.copy() if spectra.mask is not None else None,
        sample_name=spectra.sample_name,
        loading_options=spectra.loading_options,
        metadata=dict(spectra.metadata),
    )
