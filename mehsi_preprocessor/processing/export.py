"""Export functions for PKL, PNG mask, and ROI JSON.

Note: PKL (pickle) format is required by the existing codebase for data
interchange with the autoencoder pipeline via ``SpectraData.to_pickle()``
and ``SpectraData.from_pickle()``.  Only trusted, locally-generated data
is serialized.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from spectral_select.types import SpectraData

from mehsi_preprocessor.state import ClassDef, ROIRegion


def export_masked_pkl(
    spectra: "SpectraData",
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Export SpectraData with NaN outside the mask.

    Pixels where ``mask == 0`` are set to ``NaN``.
    The result is saved via ``SpectraData.to_pickle()`` – the format required
    by the downstream autoencoder pipeline.
    """
    from spectral_select.types import ExcitationData, SpectraData as SD

    binary = mask > 0  # any class != 0
    new_excitations = {}
    for ex_nm, ex in spectra.excitations.items():
        cube = ex.cube.astype(np.float64, copy=True)
        cube[~binary] = np.nan
        new_excitations[ex_nm] = ExcitationData(
            excitation_nm=ex_nm,
            cube=cube,
            emission_wavelengths=list(ex.emission_wavelengths),
            exposure_time=ex.exposure_time,
            laser_power=ex.laser_power,
        )

    masked = SD(
        excitations=new_excitations,
        mask=binary.astype(np.uint8),
        sample_name=spectra.sample_name,
        loading_options=spectra.loading_options,
        metadata=dict(spectra.metadata),
    )
    masked.to_pickle(output_path)


def export_unmasked_pkl(
    spectra: "SpectraData",
    output_path: Path,
) -> None:
    """Export SpectraData as-is (full cubes) via the existing pickle format."""
    spectra.to_pickle(output_path)


def export_mask_png(
    mask: np.ndarray,
    class_defs: List[ClassDef],
    output_path: Path,
) -> None:
    """Save a coloured class-mask PNG.

    Each class ID is rendered in its defined colour;
    background (0) is black.
    """
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    color_map = {cd.id: cd.color for cd in class_defs}

    for cls_id, rgb in color_map.items():
        where = mask == cls_id
        img[where] = rgb

    Image.fromarray(img).save(str(output_path))


def export_roi_json(
    roi_regions: List[ROIRegion],
    class_defs: List[ClassDef],
    output_path: Path,
) -> None:
    """Save ROI rectangles and class definitions as structured JSON."""
    data = {
        "classes": [
            {"id": cd.id, "name": cd.name, "color": list(cd.color)}
            for cd in class_defs
        ],
        "regions": [
            {
                "class_id": r.class_id,
                "class_name": r.class_name,
                "rect": {
                    "row_min": r.rect[0],
                    "row_max": r.rect[1],
                    "col_min": r.rect[2],
                    "col_max": r.rect[3],
                },
            }
            for r in roi_regions
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))
