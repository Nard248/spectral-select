"""Exposure-time and laser-power normalization."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spectral_select.types import SpectraData


def normalize_spectra(
    spectra: "SpectraData",
    by_exposure: bool = True,
    by_laser_power: bool = True,
) -> "SpectraData":
    """Return a **new** SpectraData with cubes divided by exposure / power.

    Division is skipped for any excitation where the corresponding value
    is ``None`` or zero.
    """
    from spectral_select.types import ExcitationData, SpectraData as SD

    new_excitations = {}
    for ex_nm, ex in spectra.excitations.items():
        cube = ex.cube.astype(np.float64, copy=True)

        if by_exposure and ex.exposure_time and ex.exposure_time > 0:
            cube /= ex.exposure_time

        if by_laser_power and ex.laser_power and ex.laser_power > 0:
            cube /= ex.laser_power

        new_excitations[ex_nm] = ExcitationData(
            excitation_nm=ex.excitation_nm,
            cube=cube,
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
