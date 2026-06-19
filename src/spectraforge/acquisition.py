"""Instrument acquisition configuration for a synthetic ME-HSI scan."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AcquisitionConfig:
    excitations: list[float]
    em_min: float
    em_max: float
    em_step: float
    lamp: dict[float, float] = field(default_factory=dict)      # g(λex)
    exposure: dict[float, float] = field(default_factory=dict)  # per-excitation exposure
    power: dict[float, float] = field(default_factory=dict)     # per-excitation laser power

    def emission_grid(self) -> np.ndarray:
        return np.arange(self.em_min, self.em_max + 1e-9, self.em_step)

    def lamp_for(self, ex: float) -> float:
        return float(self.lamp.get(ex, 1.0))

    def exposure_for(self, ex: float) -> float:
        return float(self.exposure.get(ex, 1.0))

    def power_for(self, ex: float) -> float:
        return float(self.power.get(ex, 1.0))
