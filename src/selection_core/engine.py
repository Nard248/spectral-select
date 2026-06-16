"""Shared, domain-agnostic perturbation + selection engine.

This is the single source of truth for the Perturbation-Based Autoencoder channel/
wavelength-selection algorithm. Both ``spectral_select.Analyzer`` (hyperspectral imaging)
and ``channel_select.engine`` (general grouped/temporal channels) delegate to these
primitives, so a fix to the math applies everywhere.

The only axis assumption is that the CHANNEL axis is last in each per-group reconstruction
and the BATCH axis is first in the latent. Domain-specific concerns — how the model is
built, how data is laid out, and how diversity-aware final selection is done (e.g. nm vs
channel-index distance) — live in the calling packages, not here.
"""
from __future__ import annotations

from typing import Callable, Hashable

import numpy as np
import torch


def select_important_dimensions(
    latent: torch.Tensor, method: str, n: int
) -> list[tuple[float, tuple[int, ...]]]:
    """Rank latent coordinates by importance. latent: (batch, *latent_dims)."""
    batch = latent.shape[0]
    latent_dims = tuple(latent.shape[1:])
    flat = latent.reshape(batch, -1)

    if method == "variance":
        scores = torch.var(flat, dim=0)
    elif method == "activation":
        scores = torch.mean(torch.abs(flat), dim=0)
    elif method == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(flat.cpu().numpy())
        n_comp = min(n * 2, scaled.shape[1], scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp).fit(scaled)
        scores = torch.tensor(np.sum(np.abs(pca.components_), axis=0))
    else:
        raise ValueError(f"Unknown dimension selection method: {method}")

    coords: list[tuple[float, tuple[int, ...]]] = []
    for i, s in enumerate(scores):
        coord = tuple(int(c) for c in np.unravel_index(i, latent_dims))
        coords.append((float(s.item()), coord))
    coords.sort(reverse=True, key=lambda x: x[0])
    return coords[:n]


def latent_statistics(latent_flat: torch.Tensor) -> dict:
    """Per-coordinate statistics used to scale perturbations."""
    pcts = [5, 10, 25, 50, 75, 90, 95]
    return {
        "std": torch.std(latent_flat, dim=0),
        "min": torch.min(latent_flat, dim=0)[0],
        "max": torch.max(latent_flat, dim=0)[0],
        "percentiles": {p: torch.quantile(latent_flat, p / 100.0, dim=0) for p in pcts},
    }


def _flat_index(coord: tuple[int, ...], latent_dims: tuple[int, ...]) -> int:
    return int(np.ravel_multi_index(coord, latent_dims))


def perturbation_amount(
    coord, latent_dims, magnitude, sign, stats, baseline_latent, method,
) -> float:
    """How much to add to ``baseline_latent`` at ``coord`` for one perturbation."""
    idx = _flat_index(coord, latent_dims)
    sel = (slice(None),) + coord
    if method == "percentile":
        target_pct = 50 + sign * magnitude / 2
        closest = min(stats["percentiles"].keys(), key=lambda x: abs(x - target_pct))
        target = stats["percentiles"][closest][idx]
        current_mean = torch.mean(baseline_latent[sel])
        return float((target - current_mean).item())
    if method == "standard_deviation":
        return float(sign * (magnitude / 100.0) * stats["std"][idx].item())
    if method == "absolute_range":
        rng = stats["max"][idx] - stats["min"][idx]
        return float(sign * (magnitude / 100.0) * rng.item())
    raise ValueError(f"Unknown perturbation method: {method}")


def measure_influence(
    decode_fn: Callable[[torch.Tensor], dict[Hashable, torch.Tensor]],
    groups: list[Hashable],
    perturbed_latent: torch.Tensor,
    baseline_recon: dict[Hashable, torch.Tensor],
    weight: float = 1.0,
) -> dict[Hashable, np.ndarray]:
    """Per-channel influence = mean |perturbed - baseline| over every axis except the
    last (channel) axis. For HSI this reduces ``dim=(0,1,2)`` leaving the band axis;
    the general form reduces all axes but the last regardless of rank.
    """
    influence: dict[Hashable, np.ndarray] = {}
    with torch.no_grad():
        recon = decode_fn(perturbed_latent)
        for g in groups:
            if g not in baseline_recon:
                continue
            base, pert = baseline_recon[g], recon[g]
            reduce_dims = tuple(range(pert.ndim - 1))  # all but channel (last)
            diff = torch.mean(torch.abs(pert - base), dim=reduce_dims)
            influence[g] = diff.cpu().numpy() * weight
    return influence


def accumulate_influence(
    decode_fn: Callable[[torch.Tensor], dict[Hashable, torch.Tensor]],
    groups: list[Hashable],
    channels_per_group: dict[Hashable, int],
    latent: torch.Tensor,
    baseline_recon: dict[Hashable, torch.Tensor],
    important_dims: list[tuple[float, tuple[int, ...]]],
    *,
    magnitudes: list[float],
    directions: list[str],
    perturbation_method: str,
) -> dict[Hashable, np.ndarray]:
    """Accumulate per-channel influence over every (dim, magnitude, direction, sign).

    Mirrors both ``Analyzer._compute_influence_scores`` and ``run_selection``'s loop:
    bidirectional perturbations are weighted 0.5 each, single-direction 1.0, and each
    contribution is weighted by the dimension's importance score.
    """
    latent_dims = tuple(latent.shape[1:])
    stats = latent_statistics(latent.reshape(latent.shape[0], -1))
    influence = {g: np.zeros(channels_per_group[g]) for g in groups}

    for score, coord in important_dims:
        for mag in magnitudes:
            for direction in directions:
                signs = {"bidirectional": [-1, 1], "positive": [1], "negative": [-1]}[direction]
                weight = 0.5 if len(signs) == 2 else 1.0
                for sign in signs:
                    amt = perturbation_amount(
                        coord, latent_dims, mag, sign, stats, latent, perturbation_method
                    )
                    pert = latent.clone()
                    pert[(slice(None),) + coord] += amt
                    contrib = measure_influence(decode_fn, groups, pert, baseline_recon, score)
                    for g in contrib:
                        influence[g] += contrib[g] * weight
    return influence


def normalize_influence(
    influence: dict[Hashable, np.ndarray],
    data: dict[Hashable, torch.Tensor],
    method: str,
    *,
    variance_float64: bool = True,
) -> dict[Hashable, np.ndarray]:
    """Normalize raw influence.

    ``variance_float64`` controls the dtype of the variance computation: the general
    engine uses float64 (so the 1e-10 clamp is exact); the HSI Analyzer historically
    stayed in the data's native float32, so it passes ``variance_float64=False`` to keep
    byte-identical output.
    """
    out = {g: v.copy() for g, v in influence.items()}
    if method == "none":
        return out
    if method == "max_per_group":
        for g in out:
            m = float(np.max(out[g]))
            if m > 1e-10:
                out[g] = out[g] / m
        return out
    if method == "variance":
        for g in out:
            if g not in data:
                continue
            arr = data[g].cpu().numpy()
            if variance_float64:
                arr = arr.astype(np.float64)
            var = np.var(arr, axis=tuple(range(arr.ndim - 1)))  # over all but channel
            var[var < 1e-10] = 1e-10
            out[g] = out[g] / var
        return out
    raise ValueError(f"Unknown normalization method: {method}")
