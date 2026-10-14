"""Analytic geometry of block inclusions (SpectralForge Studio v1.2).

An :class:`spectraforge.studio.schema.Inclusion` (sphere, ellipsoid, cylinder, box) lives in the block's local
frame.  This module provides the pure-geometry helpers shared by the voxelisation of the 3D Monte Carlo
(:func:`spectraforge.studio.transport3d.make_inclusion_map`) and by the ground-truth maps of
:mod:`spectraforge.studio.block_capture`:

* :func:`inside`           -- membership test of points;
* :func:`bbox`             -- axis-aligned bounding box;
* :func:`vertical_extent`  -- intersection of the vertical line through ``(x, z)`` with the shape;
* :func:`footprint_depth`  -- vertical-projection mask and depth of the inclusion top below the block top;
* :func:`analytic_volume`  -- exact volume of the (unclipped) shape.

The inclusion REPLACES the layer material wherever it overlaps the block; the part outside the block is
ignored (``footprint_depth`` clips the vertical extent to ``0 <= y <= top(x, z)``).
"""
from __future__ import annotations

import math

import numpy as np


def inside(inc, x, y, z) -> np.ndarray:
    """Boolean membership of the points ``(x, y, z)`` (broadcastable arrays, block-local mm)."""
    cx, cy, cz = inc.center_mm
    dx, dy, dz = x - cx, y - cy, z - cz
    k = inc.kind
    if k == "sphere":
        r = inc.radius_mm
        return dx * dx + dy * dy + dz * dz <= r * r
    if k == "ellipsoid":
        ax, ay, az = inc.semi_axes_mm
        return (dx / ax) ** 2 + (dy / ay) ** 2 + (dz / az) ** 2 <= 1.0
    if k == "cylinder":
        r, h = inc.radius_mm, 0.5 * inc.length_mm
        if inc.axis == "z":
            return (dx * dx + dy * dy <= r * r) & (np.abs(dz) <= h)
        return (dy * dy + dz * dz <= r * r) & (np.abs(dx) <= h)
    sx, sy, sz = inc.size_mm
    return (np.abs(dx) <= 0.5 * sx) & (np.abs(dy) <= 0.5 * sy) & (np.abs(dz) <= 0.5 * sz)


def bbox(inc) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` axis-aligned bounding box of the (unclipped) shape."""
    c = np.asarray(inc.center_mm, float)
    k = inc.kind
    if k == "sphere":
        h = np.full(3, inc.radius_mm)
    elif k == "ellipsoid":
        h = np.asarray(inc.semi_axes_mm, float)
    elif k == "cylinder":
        h = np.full(3, inc.radius_mm)
        h[0 if inc.axis == "x" else 2] = 0.5 * inc.length_mm
    else:
        h = 0.5 * np.asarray(inc.size_mm, float)
    return c - h, c + h


def vertical_extent(inc, x, z):
    """Intersection of the vertical line through ``(x, z)`` with the shape: ``(ylo, yhi, valid)`` (arrays)."""
    x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
    cx, cy, cz = inc.center_mm
    dx, dz = x - cx, z - cz
    k = inc.kind
    with np.errstate(invalid="ignore"):
        if k == "sphere":
            q = inc.radius_mm ** 2 - dx * dx - dz * dz
            valid = q >= 0.0
            half = np.sqrt(np.where(valid, q, 0.0))
        elif k == "ellipsoid":
            ax, ay, az = inc.semi_axes_mm
            q = 1.0 - (dx / ax) ** 2 - (dz / az) ** 2
            valid = q >= 0.0
            half = ay * np.sqrt(np.where(valid, q, 0.0))
        elif k == "cylinder":
            h = 0.5 * inc.length_mm
            if inc.axis == "z":
                q = inc.radius_mm ** 2 - dx * dx
                valid = (q >= 0.0) & (np.abs(dz) <= h)
            else:
                q = inc.radius_mm ** 2 - dz * dz
                valid = (q >= 0.0) & (np.abs(dx) <= h)
            half = np.sqrt(np.where(q >= 0.0, q, 0.0))
        else:
            sx, sy, sz = inc.size_mm
            valid = (np.abs(dx) <= 0.5 * sx) & (np.abs(dz) <= 0.5 * sz)
            half = np.full(x.shape, 0.5 * sy)
    return cy - half, cy + half, valid


def footprint_depth(inc, x, z, top):
    """Vertical projection of the inclusion inside the block.

    For the points ``(x, z)`` on the top surface (``top`` = surface height there) returns
    ``(mask, depth_mm)``: ``mask`` is True when the vertical line through the point meets the inclusion inside
    the block (``0 <= y <= top``); ``depth_mm`` is the depth of the inclusion's TOP below the local surface,
    ``top - min(y_high, top)`` (NaN where ``mask`` is False)."""
    ylo, yhi, valid = vertical_extent(inc, x, z)
    top = np.asarray(top, float)
    hi = np.minimum(yhi, top)
    lo = np.maximum(ylo, 0.0)
    mask = valid & (hi > lo)
    depth = np.where(mask, top - hi, np.nan)
    return mask, depth


def analytic_volume(inc) -> float:
    """Exact volume (mm^3) of the unclipped shape (a cylinder uses its full ``length_mm``)."""
    k = inc.kind
    if k == "sphere":
        return 4.0 / 3.0 * math.pi * inc.radius_mm ** 3
    if k == "ellipsoid":
        a, b, c = inc.semi_axes_mm
        return 4.0 / 3.0 * math.pi * a * b * c
    if k == "cylinder":
        return math.pi * inc.radius_mm ** 2 * inc.length_mm
    sx, sy, sz = inc.size_mm
    return float(sx * sy * sz)
