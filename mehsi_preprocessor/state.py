"""Pipeline state management for the MEHSI preprocessor.

Holds all intermediate outputs and enforces downstream invalidation
when earlier steps are modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Small data-types used by the pipeline
# ---------------------------------------------------------------------------


@dataclass
class ClassDef:
    """Definition of a single annotation class."""

    id: int
    name: str
    color: Tuple[int, int, int]  # RGB 0-255


@dataclass
class ROIRegion:
    """A rectangular region of interest assigned to a class."""

    class_id: int
    class_name: str
    rect: Tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max)


# ---------------------------------------------------------------------------
# Default class colour palette
# ---------------------------------------------------------------------------

DEFAULT_CLASS_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),       # red
    (0, 0, 255),       # blue
    (0, 200, 0),       # green
    (255, 165, 0),     # orange
    (128, 0, 128),     # purple
    (0, 200, 200),     # cyan
    (255, 0, 255),     # magenta
    (200, 200, 0),     # yellow
    (128, 64, 0),      # brown
    (0, 128, 128),     # teal
]


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

# Step indices (used for invalidation)
STEP_LOAD = 1
STEP_METADATA = 2
STEP_NORMALIZE = 3
STEP_SPATIAL_CROP = 4
STEP_SPECTRAL_CROP = 5
STEP_DRAW_CLASSES = 6
STEP_ROI_REGIONS = 7
STEP_EXPORT = 8

# Maps each step to the attribute names it owns
_STEP_ATTRIBUTES: Dict[int, List[str]] = {
    STEP_LOAD: ["raw_spectra", "data_folder"],
    STEP_METADATA: ["exposure_times", "laser_powers"],
    STEP_NORMALIZE: ["normalized_spectra"],
    STEP_SPATIAL_CROP: ["crop_roi", "cropped_spectra"],
    STEP_SPECTRAL_CROP: ["filtered_spectra"],
    STEP_DRAW_CLASSES: ["class_mask", "class_definitions"],
    STEP_ROI_REGIONS: ["roi_regions"],
    STEP_EXPORT: [],  # export doesn't own mutable state
}


class PipelineState:
    """Single source of truth for all step outputs.

    Modifying step *N* via :meth:`invalidate_from` clears every attribute
    owned by steps *N+1* through 8.
    """

    def __init__(self) -> None:
        # Step 1 – Load
        self.raw_spectra: Optional[Any] = None  # SpectraData
        self.data_folder: Optional[Path] = None

        # Step 2 – Metadata
        self.exposure_times: Dict[float, float] = {}
        self.laser_powers: Dict[float, float] = {}

        # Step 3 – Normalize
        self.normalized_spectra: Optional[Any] = None  # SpectraData

        # Step 4 – Spatial crop
        self.crop_roi: Optional[Tuple[int, int, int, int]] = None
        self.cropped_spectra: Optional[Any] = None  # SpectraData

        # Step 5 – Spectral/emission crop
        self.filtered_spectra: Optional[Any] = None  # SpectraData

        # Step 6 – Class masks
        self.class_mask: Optional[np.ndarray] = None  # (H, W) int32
        self.class_definitions: List[ClassDef] = []

        # Step 7 – ROI regions
        self.roi_regions: List[ROIRegion] = []

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def current_spectra(self) -> Optional[Any]:
        """Return the most-processed SpectraData available."""
        for attr in ("filtered_spectra", "cropped_spectra",
                     "normalized_spectra", "raw_spectra"):
            val = getattr(self, attr)
            if val is not None:
                return val
        return None

    @property
    def spatial_shape(self) -> Optional[Tuple[int, int]]:
        """Return (H, W) of the current working data, or *None*."""
        spec = self.current_spectra
        if spec is not None:
            return spec.spatial_shape
        return None

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_from(self, step: int) -> None:
        """Clear all outputs from steps *after* ``step``."""
        for s in range(step + 1, STEP_EXPORT + 1):
            for attr in _STEP_ATTRIBUTES.get(s, []):
                default = self._default_for(attr)
                setattr(self, attr, default)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_for(attr: str) -> Any:
        """Return the reset value for an attribute name."""
        if attr in ("exposure_times", "laser_powers"):
            return {}
        if attr in ("class_definitions", "roi_regions"):
            return []
        return None
