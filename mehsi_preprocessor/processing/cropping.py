"""Spatial cropping of hyperspectral cubes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from spectral_select.types import SpectraData


def spatial_crop(
    spectra: "SpectraData",
    roi: Tuple[int, int, int, int],
) -> "SpectraData":
    """Crop all cubes to *roi* = ``(row_min, row_max, col_min, col_max)``.

    Returns a **new** ``SpectraData`` with cubes at the cropped size
    (origin reset to ``(0, 0)``).
    """
    from spectral_select.types import ExcitationData, SpectraData as SD

    r0, r1, c0, c1 = roi
    new_excitations = {}
    for ex_nm, ex in spectra.excitations.items():
        cropped = ex.cube[r0:r1, c0:c1, :].copy()
        new_excitations[ex_nm] = ExcitationData(
            excitation_nm=ex.excitation_nm,
            cube=cropped,
            emission_wavelengths=list(ex.emission_wavelengths),
            exposure_time=ex.exposure_time,
            laser_power=ex.laser_power,
        )

    mask = None
    if spectra.mask is not None:
        mask = spectra.mask[r0:r1, c0:c1].copy()

    return SD(
        excitations=new_excitations,
        mask=mask,
        sample_name=spectra.sample_name,
        loading_options=spectra.loading_options,
        metadata=dict(spectra.metadata),
    )
