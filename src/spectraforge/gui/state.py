"""ForgeState: the working document for the Forge GUI."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.library import load_builtin_library
from spectraforge.material import Material
from spectraforge.gui.layer import Layer


def _default_acquisition() -> AcquisitionConfig:
    return AcquisitionConfig(excitations=[340.0, 450.0, 488.0], em_min=360, em_max=700, em_step=5)


def _default_artifacts() -> ArtifactConfig:
    return ArtifactConfig(rayleigh_strength=0.15, rayleigh_fwhm=12, photon_scale=400.0, read_sigma=0.005)


@dataclass
class ForgeState:
    height: int
    width: int
    library: dict = field(default_factory=load_builtin_library)   # name -> Fluorophore
    materials: dict = field(default_factory=dict)                 # name -> Material
    layers: list = field(default_factory=list)                    # list[Layer]
    acquisition: AcquisitionConfig = field(default_factory=_default_acquisition)
    artifacts: ArtifactConfig = field(default_factory=_default_artifacts)
    seed: int = 0
    active_layer: int = -1
    last_render: object = None                                    # (SpectraData, GroundTruth) | None

    def add_layer(self, name: str, material: Material) -> Layer:
        layer = Layer(
            name=name,
            material=material,
            amount_map=np.zeros((self.height, self.width), dtype=float),
        )
        self.layers.append(layer)
        self.active_layer = len(self.layers) - 1
        return layer
