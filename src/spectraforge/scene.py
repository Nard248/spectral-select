"""Scene: paint materials onto an H x W canvas -> per-fluorophore concentration maps."""
from __future__ import annotations

import numpy as np

from spectraforge.material import Material


class Scene:
    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)
        self._maps: dict[str, np.ndarray] = {}

    def _ensure(self, name: str) -> np.ndarray:
        if name not in self._maps:
            self._maps[name] = np.zeros((self.height, self.width), dtype=float)
        return self._maps[name]

    def _add_region(self, mask: np.ndarray, material: Material, amount: float) -> None:
        for fname, conc in material.recipe.items():
            self._ensure(fname)[mask] += conc * amount

    def paint_rect(self, material: Material, r0, r1, c0, c1, amount: float = 1.0) -> None:
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[r0:r1, c0:c1] = True
        self._add_region(mask, material, amount)

    def paint_circle(self, material: Material, cy, cx, radius, amount: float = 1.0) -> None:
        yy, xx = np.ogrid[: self.height, : self.width]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        self._add_region(mask, material, amount)

    def paint_polygon(self, material: Material, vertices, amount: float = 1.0) -> None:
        from matplotlib.path import Path as MplPath

        yy, xx = np.mgrid[: self.height, : self.width]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = MplPath(vertices).contains_points(pts).reshape(self.height, self.width)
        self._add_region(inside, material, amount)

    def resolve(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._maps.items()}

    def __add__(self, other: "Scene") -> "Scene":
        if (self.height, self.width) != (other.height, other.width):
            raise ValueError("scenes must have the same shape to add")
        out = Scene(self.height, self.width)
        for src in (self, other):
            for name, m in src._maps.items():
                out._ensure(name)[:] += m
        return out
