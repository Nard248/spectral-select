"""Mask operations – ROI-to-mask conversion and mask merging."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mehsi_preprocessor.state import ROIRegion


def roi_regions_to_mask(
    regions: List[ROIRegion],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Convert a list of :class:`ROIRegion` to a class-ID mask.

    Parameters
    ----------
    regions : list of ROIRegion
        Each carries ``class_id`` and ``rect = (r0, r1, c0, c1)``.
    shape : (H, W)
        Output mask shape.

    Returns
    -------
    np.ndarray
        ``int32`` array where 0 = background and positive ints are class IDs.
        Later regions overwrite earlier ones when they overlap.
    """
    mask = np.zeros(shape, dtype=np.int32)
    for reg in regions:
        r0, r1, c0, c1 = reg.rect
        # Clamp to bounds
        r0 = max(r0, 0)
        r1 = min(r1, shape[0])
        c0 = max(c0, 0)
        c1 = min(c1, shape[1])
        mask[r0:r1, c0:c1] = reg.class_id
    return mask


def merge_masks(
    brush_mask: np.ndarray,
    roi_mask: np.ndarray,
    priority: str = "brush",
) -> np.ndarray:
    """Merge two class-ID masks.

    *priority* decides which mask wins when both have non-zero values
    at the same pixel.  ``"brush"`` means the brush mask wins.
    """
    out = roi_mask.copy()
    if priority == "brush":
        nonzero = brush_mask > 0
        out[nonzero] = brush_mask[nonzero]
    else:
        nonzero = roi_mask > 0
        out[~nonzero] = brush_mask[~nonzero]
    return out
