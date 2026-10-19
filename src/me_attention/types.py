"""Shared contracts for the ME-attention research package (lead-owned; change only via the lead).

Every loader returns an ``MEData``; every model is registered as a ``ModelSpec`` whose estimator
follows the ``Estimator`` protocol. The harness only talks to these three types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

import numpy as np

Track = Literal["spectral", "spatial"]


@dataclass
class MEData:
    """One multi-excitation fluorescence dataset, flattened to labelled foreground pixels.

    Features are the *valid* (excitation, emission) measurements only, ordered by
    (excitation, emission). Wavelengths are always in nm; loaders must raise if a cube has no
    wavelength vector (never fall back to index alignment -- see review bug B1).
    """

    name: str
    X: np.ndarray                 # [N, F] float32, raw measurement per pixel and feature
    ex_nm: np.ndarray             # [F] float, excitation wavelength of each feature
    em_nm: np.ndarray             # [F] float, emission wavelength of each feature
    y: np.ndarray                 # [N] int64, class index 0..C-1
    groups: np.ndarray            # [N] int64, specimen id 0..S-1 (connected component of the class mask)
    coords: np.ndarray            # [N, 2] int64, (row, col) of each pixel in the image
    specimen_classes: np.ndarray  # [S] int64, class of each specimen
    class_names: list[str]
    image_shape: tuple[int, int]  # (H, W)
    excitations: np.ndarray       # [E] sorted excitation wavelengths (nm)
    emissions: np.ndarray         # [M] sorted union of all emission wavelengths (nm), irregular allowed
    grid_index: np.ndarray        # [F, 2] int64, (e_idx, m_idx) of each feature in the E x M grid
    specimen_map: np.ndarray      # [H, W] int64, specimen id per pixel, -1 = not a kept specimen
    info: dict[str, Any] = field(default_factory=dict)  # provenance: source file, dropped px, notes

    @property
    def n_classes(self) -> int:
        return int(len(self.class_names))

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def meta(self) -> dict[str, Any]:
        """Static description passed to model builders (no pixel data)."""
        return dict(
            ex_nm=self.ex_nm, em_nm=self.em_nm, grid_index=self.grid_index,
            excitations=self.excitations, emissions=self.emissions,
            n_features=self.n_features, n_classes=self.n_classes,
        )


class Estimator(Protocol):
    """A trainable classifier. All preprocessing is fit inside ``fit`` on training data only.

    Spectral track: X is [n, F].  Spatial track: X is [n, P, P, F] (patch centred on the pixel,
    pixels outside the centre pixel's specimen are zero -- see protocol.make_patches).
    ``X_val``/``y_val`` (inner validation, from held-out *specimens*) may be used for early stopping
    only; never for fitting weights.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "Estimator": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...   # [n, C]


@dataclass
class ModelSpec:
    name: str
    track: Track
    family: Literal["baseline", "ours"]
    # build(cfg, meta, seed, device) -> Estimator
    build: Callable[[dict[str, Any], dict[str, Any], int, str], Estimator]
    # search space: param -> list of candidate values (random search samples uniformly)
    space: dict[str, list[Any]]
    default: dict[str, Any]
    description: str = ""
