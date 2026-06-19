"""The forward model: scene + fluorophores + acquisition -> SpectraData + GroundTruth."""
from __future__ import annotations

import numpy as np

from spectral_select.types import ExcitationData, SpectraData

from spectraforge.groundtruth import GroundTruth


def render(scene, library, acquisition, artifacts=None, seed=None, sample_name="synthetic"):
    """Render a synthetic ME-HSI dataset.

    Returns ``(SpectraData, GroundTruth)``. With ``artifacts=None`` the result is the
    clean, exactly-linear forward model: ``render(A+B) == render(A)+render(B)``.
    """
    conc = scene.resolve()                       # {fname: (H, W)}
    h, w = scene.height, scene.width
    em = acquisition.emission_grid()
    rng = np.random.default_rng(seed)

    excitations = {}
    clean_cubes = {}
    per_fluorophore = {}                          # fname -> {ex -> (n_em,) per-pixel-max spectrum}
    for ex in acquisition.excitations:
        scale = (
            acquisition.lamp_for(ex)
            * acquisition.exposure_for(ex)
            * acquisition.power_for(ex)
        )
        cube = np.zeros((h, w, len(em)), dtype=float)
        for fname, cmap in conc.items():
            f = library[fname]
            amp = f.extinction * f.quantum_yield * float(f.excitation(ex))  # scalar
            em_profile = f.emission(em)                                     # (n_em,)
            contrib = (cmap * amp)[:, :, None] * em_profile[None, None, :]
            cube += contrib
            band_max = contrib.reshape(-1, len(em)).max(axis=0) * scale     # (n_em,)
            per_fluorophore.setdefault(fname, {})[float(ex)] = band_max
        cube *= scale
        clean_cubes[float(ex)] = cube.copy()
        if artifacts is not None:
            from spectraforge.artifacts import add_noise, add_scatter_lines

            add_scatter_lines(cube, ex, em, artifacts, scale)
            cube = add_noise(cube, artifacts, rng)
        excitations[float(ex)] = ExcitationData(
            cube=cube,
            excitation_nm=float(ex),
            emission_wavelengths=[float(x) for x in em],
            exposure_time=acquisition.exposure_for(ex),
            laser_power=acquisition.power_for(ex),
        )

    spectra = SpectraData(excitations=excitations, sample_name=sample_name)
    gt = GroundTruth(
        concentration_maps=conc,
        clean_cubes=clean_cubes,
        emission_grid=em,
        excitations=[float(e) for e in acquisition.excitations],
        per_fluorophore_spectra=per_fluorophore,
        seed=seed,
    )
    return spectra, gt
