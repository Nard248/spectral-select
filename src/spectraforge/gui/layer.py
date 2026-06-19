"""Layer model and scene assembly for the Forge GUI (pure, no Qt)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectraforge.material import Material
from spectraforge.scene import Scene


@dataclass
class Layer:
    name: str
    material: Material
    amount_map: np.ndarray   # (H, W) per-pixel painted amount
    visible: bool = True


def build_scene(state) -> Scene:
    """Assemble the engine Scene from a state's visible layers, in order."""
    scene = Scene(state.height, state.width)
    for layer in state.layers:
        if layer.visible:
            scene.paint_map(layer.material, layer.amount_map)
    return scene
