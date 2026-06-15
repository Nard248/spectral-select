from __future__ import annotations
from dataclasses import dataclass, field
from typing import Hashable, Protocol, runtime_checkable

import torch

_VALID_DIM_METHODS = {"variance", "activation", "pca"}
_VALID_PERTURB_METHODS = {"percentile", "standard_deviation", "absolute_range"}
_VALID_NORM_METHODS = {"variance", "max_per_group", "none"}
_VALID_DIVERSITY = {"mmr", "min_distance", "none"}


@dataclass
class SelectionConfig:
    """Engine configuration. Field names mirror spectral_select.Config so the
    shared engine can be regression-tested against the HSI Analyzer."""

    dimension_selection_method: str = "variance"
    n_important_dimensions: int = 50
    perturbation_method: str = "percentile"
    perturbation_magnitudes: list[float] = field(default_factory=lambda: [10, 20, 30])
    perturbation_directions: list[str] = field(default_factory=lambda: ["bidirectional"])
    normalization_method: str = "variance"
    n_channels_to_select: int = 10
    diversity_method: str = "mmr"
    lambda_diversity: float = 0.5
    min_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.dimension_selection_method not in _VALID_DIM_METHODS:
            raise ValueError(f"dimension_selection_method must be one of {_VALID_DIM_METHODS}")
        if self.perturbation_method not in _VALID_PERTURB_METHODS:
            raise ValueError(f"perturbation_method must be one of {_VALID_PERTURB_METHODS}")
        if self.normalization_method not in _VALID_NORM_METHODS:
            raise ValueError(f"normalization_method must be one of {_VALID_NORM_METHODS}")
        if self.diversity_method not in _VALID_DIVERSITY:
            raise ValueError(f"diversity_method must be one of {_VALID_DIVERSITY}")


@runtime_checkable
class GroupStructuredModel(Protocol):
    groups: list[Hashable]
    channels_per_group: dict[Hashable, int]

    def encode(self, batch: dict[Hashable, torch.Tensor]) -> torch.Tensor: ...
    def decode(self, latent: torch.Tensor) -> dict[Hashable, torch.Tensor]: ...


@runtime_checkable
class GroupedChannelData(Protocol):
    groups: list[Hashable]
    channels_per_group: dict[Hashable, int]
    axis_type: str  # "spatial2d" | "temporal1d"

    def get_all_data(self) -> dict[Hashable, torch.Tensor]: ...
