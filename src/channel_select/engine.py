"""Domain-agnostic perturbation + selection engine for grouped/temporal channels.

The core perturbation->influence->normalize math now lives in ``selection_core`` (shared
with the hyperspectral ``spectral_select.Analyzer``). This module keeps the channel-domain
public API and the channel-index diversity selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import torch

# Shared primitives (single source of truth).
from selection_core import (
    select_important_dimensions,
    latent_statistics,
    perturbation_amount,
    measure_influence,
    accumulate_influence,
    normalize_influence,
)

__all__ = [
    "select_important_dimensions",
    "latent_statistics",
    "perturbation_amount",
    "measure_channel_influence",
    "normalize_influence",
    "select_channels",
    "SelectionResult",
    "run_selection",
]


def measure_channel_influence(
    model, perturbed_latent: torch.Tensor,
    baseline_recon: dict[Hashable, torch.Tensor], weight: float = 1.0,
) -> dict[Hashable, np.ndarray]:
    """Per-channel influence for a group-structured model (thin wrapper over
    ``selection_core.measure_influence`` using the model's decode + groups)."""
    return measure_influence(model.decode, model.groups, perturbed_latent, baseline_recon, weight)


def _channel_catalog(influence):
    combos = []
    for g, vec in influence.items():
        for c, val in enumerate(vec):
            combos.append({"group": g, "channel": int(c), "influence": float(val)})
    combos.sort(key=lambda x: x["influence"], reverse=True)
    return combos


def _profiles(data):
    from sklearn.preprocessing import normalize
    prof = {}
    for g, t in data.items():
        arr = t.cpu().numpy()
        n_ch = arr.shape[-1]
        for c in range(n_ch):
            v = arr[..., c].reshape(1, -1)
            prof[(g, c)] = normalize(v, axis=1).flatten()
    return prof


def select_channels(
    influence, data, K, method="mmr", lambda_diversity=0.5, min_distance=0.0,
):
    """Diversity-aware final channel selection (channel-index based)."""
    combos = _channel_catalog(influence)
    if method == "none" or len(combos) <= K:
        return [(c["group"], c["channel"]) for c in combos[:K]]

    if method == "mmr":
        prof = _profiles(data)
        max_inf = combos[0]["influence"] or 1.0
        selected = [combos[0]]
        keys = [(combos[0]["group"], combos[0]["channel"])]
        while len(selected) < K:
            best, best_key, best_score = None, None, -np.inf
            for combo in combos:
                key = (combo["group"], combo["channel"])
                if key in keys or key not in prof:
                    continue
                rel = combo["influence"] / max_inf
                max_sim = max(
                    (abs(float(np.dot(prof[key], prof[sk]))) for sk in keys if sk in prof),
                    default=0.0,
                )
                score = rel - lambda_diversity * max_sim
                if score > best_score:
                    best, best_key, best_score = combo, key, score
            if best is None:
                break
            selected.append(best)
            keys.append(best_key)
        return [(c["group"], c["channel"]) for c in selected]

    if method == "min_distance":
        selected = []
        for combo in combos:
            ok = all(
                not (combo["group"] == s["group"]
                     and abs(combo["channel"] - s["channel"]) < min_distance)
                for s in selected
            )
            if ok:
                selected.append(combo)
            if len(selected) >= K:
                break
        return [(c["group"], c["channel"]) for c in selected]

    raise ValueError(f"Unknown diversity method: {method}")


@dataclass
class SelectionResult:
    selected: list[tuple]                 # [(group, channel), ...]
    influence: dict                       # group -> normalized influence vector
    important_dims: list                  # [(score, coord), ...]


def run_selection(model, data: dict, config) -> "SelectionResult":
    """Full engine pipeline on an already-trained model + its training data."""
    all_data = data if isinstance(data, dict) else data.get_all_data()
    with torch.no_grad():
        latent = model.encode(all_data)
        baseline_recon = model.decode(latent)

    important = select_important_dimensions(
        latent, config.dimension_selection_method, config.n_important_dimensions
    )
    influence = accumulate_influence(
        model.decode, model.groups, model.channels_per_group,
        latent, baseline_recon, important,
        magnitudes=config.perturbation_magnitudes,
        directions=config.perturbation_directions,
        perturbation_method=config.perturbation_method,
    )

    influence = normalize_influence(influence, all_data, config.normalization_method)
    selected = select_channels(
        influence=influence, data=all_data, K=config.n_channels_to_select,
        method=config.diversity_method, lambda_diversity=config.lambda_diversity,
        min_distance=config.min_distance,
    )
    return SelectionResult(selected=selected, influence=influence, important_dims=important)
