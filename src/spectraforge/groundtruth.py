"""Ground-truth maps that accompany a synthetic dataset (the validation oracle)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GroundTruth:
    concentration_maps: dict[str, np.ndarray]   # fluorophore_name -> (H, W)
    clean_cubes: dict[float, np.ndarray]         # excitation -> (H, W, n_em), noise/scatter-free
    emission_grid: np.ndarray
    excitations: list[float]
    materials: dict = field(default_factory=dict)
    seed: "int | None" = None

    def save(self, out_dir) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {f"conc__{k}": v for k, v in self.concentration_maps.items()}
        arrays.update({f"clean__{ex}": c for ex, c in self.clean_cubes.items()})
        arrays["emission_grid"] = self.emission_grid
        np.savez_compressed(out / "groundtruth.npz", **arrays)
        meta = {
            "fluorophores": list(self.concentration_maps.keys()),
            "excitations": [float(e) for e in self.excitations],
            "emission_grid": [float(x) for x in self.emission_grid],
            "materials": self.materials,
            "seed": self.seed,
        }
        (out / "groundtruth.json").write_text(json.dumps(meta, indent=2))

    def informative_bands(self, threshold: float = 0.01) -> dict[float, np.ndarray]:
        """Per excitation, boolean mask of emission bands whose max clean signal exceeds
        ``threshold * global_max`` — i.e., bands that actually carry fluorophore signal."""
        gmax = max((c.max() for c in self.clean_cubes.values()), default=0.0) or 1.0
        out = {}
        for ex, cube in self.clean_cubes.items():
            band_max = cube.reshape(-1, cube.shape[-1]).max(axis=0)
            out[ex] = band_max > (threshold * gmax)
        return out
