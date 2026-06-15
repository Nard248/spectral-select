"""Shared, domain-agnostic perturbation + selection engine.

Math mirrors spectral_select/analyzer.py, generalized so the only axis-specific
assumption is that the CHANNEL axis is last in each per-group reconstruction and
the BATCH axis is first in the latent.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable

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


def measure_channel_influence(
    model, perturbed_latent: torch.Tensor,
    baseline_recon: dict[Hashable, torch.Tensor], weight: float = 1.0,
) -> dict[Hashable, np.ndarray]:
    """Per-channel influence = mean |perturbed - baseline| over every axis
    except the last (channel) axis. Mirrors Analyzer._measure_band_influence
    where the HSI reduction dim=(0,1,2) leaves the band axis."""
    influence: dict[Hashable, np.ndarray] = {}
    with torch.no_grad():
        recon = model.decode(perturbed_latent)
        for g in model.groups:
            if g not in baseline_recon:
                continue
            base, pert = baseline_recon[g], recon[g]
            reduce_dims = tuple(range(pert.ndim - 1))  # all but channel (last)
            diff = torch.mean(torch.abs(pert - base), dim=reduce_dims)
            influence[g] = diff.cpu().numpy() * weight
    return influence


def latent_statistics(latent_flat: torch.Tensor) -> dict:
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


def normalize_influence(
    influence: dict[Hashable, np.ndarray], data: dict[Hashable, torch.Tensor], method: str,
) -> dict[Hashable, np.ndarray]:
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
            arr = data[g].cpu().numpy().astype(np.float64)  # float64 so the 1e-10 clamp is exact
            var = np.var(arr, axis=tuple(range(arr.ndim - 1)))  # over all but channel
            var[var < 1e-10] = 1e-10
            out[g] = out[g] / var
        return out
    raise ValueError(f"Unknown normalization method: {method}")


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
    latent_dims = tuple(latent.shape[1:])
    stats = latent_statistics(latent.reshape(latent.shape[0], -1))

    influence = {g: np.zeros(model.channels_per_group[g]) for g in model.groups}
    for score, coord in important:
        for mag in config.perturbation_magnitudes:
            for direction in config.perturbation_directions:
                signs = {"bidirectional": [-1, 1], "positive": [1], "negative": [-1]}[direction]
                weight = 0.5 if len(signs) == 2 else 1.0
                for sign in signs:
                    amt = perturbation_amount(
                        coord, latent_dims, mag, sign, stats, latent,
                        config.perturbation_method,
                    )
                    pert = latent.clone()
                    pert[(slice(None),) + coord] += amt
                    contrib = measure_channel_influence(model, pert, baseline_recon, score)
                    for g in contrib:
                        influence[g] += contrib[g] * weight

    influence = normalize_influence(influence, all_data, config.normalization_method)
    selected = select_channels(
        influence=influence, data=all_data, K=config.n_channels_to_select,
        method=config.diversity_method, lambda_diversity=config.lambda_diversity,
        min_distance=config.min_distance,
    )
    return SelectionResult(selected=selected, influence=influence, important_dims=important)
