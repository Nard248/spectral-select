"""Procedural scene generation with realistic per-pixel spectral variance.

Increment B showed that flat solid-region scenes are *degenerate* — every pixel in a region
shares the identical spectrum, so the cube holds only a couple of unique spectra and the
perturbation-AE selection has nothing spatial to latch onto. These helpers paint each material
with a smooth random concentration field so spectra vary continuously across the scene (and
overlapping fields produce genuine mixtures), which is what real tissue looks like.
"""
from __future__ import annotations

import numpy as np

from spectraforge.scene import Scene


def random_field(height: int, width: int, seed: int, blur: int = 2) -> np.ndarray:
    """A smooth random concentration field in [0, 1] of shape (height, width), deterministic by seed."""
    rng = np.random.default_rng(seed)
    bh, bw = -(-height // 4), -(-width // 4)                 # ceil-div: coarse grid upsampled x4
    field = np.kron(rng.random((bh, bw)), np.ones((4, 4)))[:height, :width]
    for _ in range(blur):                                    # cheap box-ish blur -> smooth gradients
        field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                 + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5
    span = float(field.max() - field.min())
    return (field - field.min()) / span if span else field


def random_scene(materials, height: int, width: int, seed: int) -> Scene:
    """Paint each material with its own smooth random concentration field (overlap -> mixtures)."""
    scene = Scene(height, width)
    for i, material in enumerate(materials):
        scene.paint_map(material, random_field(height, width, seed * 101 + i * 7))
    return scene
