"""Reward components and numerically stable action probabilities."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    individual: float = 0.1
    marginal: float = 0.3
    future: float = 0.5
    redundancy: float = -0.1
    global_marginal: float = 0.0


def normalize(values, method="zscore") -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    if method == "zscore":
        scale = values.std()
        return np.zeros_like(values) if scale < 1e-12 else (values - values.mean()) / scale
    if method == "minmax":
        span = values.max() - values.min()
        return np.zeros_like(values) if span < 1e-12 else (values - values.min()) / span
    if method == "none":
        return values
    raise ValueError(f"Unknown normalization: {method}")


def stable_softmax(values, temperature=1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(values, dtype=float) / temperature
    exp = np.exp(scaled - np.max(scaled))
    return exp / exp.sum()


def mean_redundancy(X: np.ndarray, candidate: int, selected: list[int]) -> float:
    if not selected:
        return 0.0
    correlations = []
    for other in selected:
        a, b = X[:, candidate], X[:, other]
        value = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else 0.0
        correlations.append(abs(float(np.nan_to_num(value))))
    return float(np.mean(correlations))
