"""Small policy-gradient selector for labelled hyperspectral images.

The selector treats each ``(excitation, emission)`` pair as an action.  An episode
samples ``budget`` unique actions and receives a terminal reward based on the
balanced accuracy of a nearest-centroid classifier on held-out connected objects.
This is intentionally a small, transparent RL baseline rather than a large DQN.
"""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from torch import nn

@dataclass(frozen=True)
class BandAction:
    """One selectable measurement plane."""

    excitation_nm: float
    emission_nm: float
    band_index: int


@dataclass
class LabeledBandData:
    """Pixel features, spatial split membership, and action metadata."""

    features: np.ndarray
    labels: np.ndarray
    coordinates: np.ndarray
    partitions: np.ndarray
    bands: list[BandAction]

    def indices(self, partition: str) -> np.ndarray:
        return np.flatnonzero(self.partitions == partition)


@dataclass
class RLSelectionResult:
    """Output of :func:`train_reinforce_selector`."""

    selected_indices: list[int]
    best_validation_reward: float
    best_validation_accuracy: float
    test_accuracy: float
    full_band_test_accuracy: float
    history: list[dict[str, float]]


def component_partition_map(
    class_mask: np.ndarray,
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> np.ndarray:
    """Split disconnected objects of every class into train/validation/test.

    Pixels from one connected component always stay in one partition.  At least
    one component is assigned to each partition, so every class needs at least
    three disconnected labelled objects.
    """

    if class_mask.ndim != 2:
        raise ValueError("class_mask must be a 2D array")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train_fraction and validation_fraction must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")

    result = np.full(class_mask.shape, "background", dtype="U10")
    rng = np.random.default_rng(seed)
    for class_id in sorted(int(v) for v in np.unique(class_mask) if v >= 0):
        components, n_components = connected_components(class_mask == class_id)
        if n_components < 3:
            raise ValueError(
                f"class {class_id} has {n_components} connected components; "
                "at least 3 are required for object-level train/validation/test splits"
            )

        ids = rng.permutation(np.arange(1, n_components + 1))
        n_train = max(1, int(np.floor(n_components * train_fraction)))
        n_validation = max(1, int(np.floor(n_components * validation_fraction)))
        if n_train + n_validation >= n_components:
            n_train = n_components - 2
            n_validation = 1

        assignments = (
            (ids[:n_train], "train"),
            (ids[n_train : n_train + n_validation], "validation"),
            (ids[n_train + n_validation :], "test"),
        )
        for component_ids, partition in assignments:
            result[np.isin(components, component_ids)] = partition

    return result


def build_labeled_band_data(
    excitations: Mapping[float, tuple[np.ndarray, Sequence[float]]],
    class_mask: np.ndarray,
    *,
    seed: int = 42,
) -> LabeledBandData:
    """Convert excitation cubes into labelled pixels by candidate band.

    Only labelled pixels (class id >= 0) are materialised, avoiding a dense copy
    of every background pixel in a large HSI cube.
    """

    if not excitations:
        raise ValueError("excitations cannot be empty")
    first_cube = next(iter(excitations.values()))[0]
    spatial_shape = tuple(first_cube.shape[:2])
    if class_mask.shape != spatial_shape:
        raise ValueError(
            f"class mask shape {class_mask.shape} does not match spectra {spatial_shape}"
        )

    rows, cols = np.where(class_mask >= 0)
    features: list[np.ndarray] = []
    bands: list[BandAction] = []
    for excitation_nm in sorted(excitations):
        cube, emission_wavelengths = excitations[excitation_nm]
        if cube.shape[:2] != spatial_shape:
            raise ValueError("all excitation cubes must share the same spatial shape")
        if cube.shape[2] != len(emission_wavelengths):
            raise ValueError("emission wavelength count does not match cube bands")
        values = np.asarray(cube[rows, cols, :], dtype=np.float32)
        features.append(values)
        bands.extend(
            BandAction(float(excitation_nm), float(emission_nm), band_index)
            for band_index, emission_nm in enumerate(emission_wavelengths)
        )

    partition_map = component_partition_map(class_mask, seed=seed)
    return LabeledBandData(
        features=np.ascontiguousarray(np.concatenate(features, axis=1)),
        labels=np.asarray(class_mask[rows, cols], dtype=np.int64),
        coordinates=np.column_stack((rows, cols)).astype(np.int64),
        partitions=np.asarray(partition_map[rows, cols]),
        bands=bands,
    )


def load_labeled_band_data(
    spectra_path: str | Path,
    class_mask_path: str | Path,
    *,
    seed: int = 42,
) -> LabeledBandData:
    """Load a processed spectral-select pickle and its integer class mask."""

    with Path(spectra_path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("processed pickle must contain a 'data' mapping")

    excitation_values = payload.get("excitation_wavelengths", payload["data"].keys())
    excitations: dict[float, tuple[np.ndarray, Sequence[float]]] = {}
    for excitation_value in excitation_values:
        excitation_nm = float(excitation_value)
        record = payload["data"].get(str(excitation_nm))
        if record is None:
            record = payload["data"].get(excitation_nm)
        if not isinstance(record, dict) or "cube" not in record:
            raise ValueError(f"missing cube record for excitation {excitation_nm}")
        cube = np.asarray(record["cube"])
        wavelengths = record.get("wavelengths", list(range(cube.shape[2])))
        excitations[excitation_nm] = (cube, wavelengths)

    class_mask = np.load(class_mask_path)
    return build_labeled_band_data(excitations, class_mask, seed=seed)


class NearestCentroidBandReward:
    """Fast deterministic subset reward with an untouched test partition."""

    def __init__(
        self,
        data: LabeledBandData,
        *,
        redundancy_weight: float = 0.02,
        max_pixels_per_class: int | None = 1000,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.redundancy_weight = float(redundancy_weight)
        self.classes = np.unique(data.labels)
        train_idx = data.indices("train")
        if train_idx.size == 0:
            raise ValueError("training partition is empty")

        mean = data.features[train_idx].mean(axis=0, dtype=np.float64)
        scale = data.features[train_idx].std(axis=0, dtype=np.float64)
        scale[scale < 1e-8] = 1.0
        self.features = np.asarray((data.features - mean) / scale, dtype=np.float32)
        self.centroids = np.stack(
            [self.features[train_idx[data.labels[train_idx] == c]].mean(axis=0) for c in self.classes]
        ).astype(np.float32)

        rng = np.random.default_rng(seed)
        self.evaluation_indices: dict[str, np.ndarray] = {}
        for partition in ("validation", "test"):
            selected: list[np.ndarray] = []
            partition_idx = data.indices(partition)
            for class_id in self.classes:
                choices = partition_idx[data.labels[partition_idx] == class_id]
                if choices.size == 0:
                    raise ValueError(f"class {class_id} is absent from {partition} partition")
                if max_pixels_per_class is not None and choices.size > max_pixels_per_class:
                    choices = rng.choice(choices, max_pixels_per_class, replace=False)
                selected.append(choices)
            self.evaluation_indices[partition] = np.concatenate(selected)

        corr_idx = train_idx
        if corr_idx.size > 5000:
            corr_idx = rng.choice(corr_idx, 5000, replace=False)
        corr = np.corrcoef(self.features[corr_idx], rowvar=False)
        self.absolute_correlation = np.nan_to_num(np.abs(corr), nan=0.0).astype(np.float32)
        self._cache: dict[tuple[int, ...], tuple[float, float]] = {}

    def balanced_accuracy(self, selected: Sequence[int], partition: str) -> float:
        selected_array = np.asarray(selected, dtype=np.int64)
        if selected_array.size == 0:
            raise ValueError("at least one band must be selected")
        idx = self.evaluation_indices[partition]
        x = self.features[np.ix_(idx, selected_array)]
        centroids = self.centroids[:, selected_array]
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predictions = self.classes[np.argmin(distances, axis=1)]
        labels = self.data.labels[idx]
        recalls = [np.mean(predictions[labels == class_id] == class_id) for class_id in self.classes]
        return float(np.mean(recalls))

    def redundancy(self, selected: Sequence[int]) -> float:
        if len(selected) < 2:
            return 0.0
        matrix = self.absolute_correlation[np.ix_(selected, selected)]
        upper = matrix[np.triu_indices(len(selected), k=1)]
        return float(upper.mean())

    def __call__(self, selected: Sequence[int]) -> tuple[float, float]:
        key = tuple(sorted(int(i) for i in selected))
        if key not in self._cache:
            accuracy = self.balanced_accuracy(key, "validation")
            reward = accuracy - self.redundancy_weight * self.redundancy(key)
            self._cache[key] = (float(reward), float(accuracy))
        return self._cache[key]


class MaskedBandPolicy(nn.Module):
    """A Plackett-Luce policy over bands, sampled without replacement."""

    def __init__(self, n_bands: int) -> None:
        super().__init__()
        if n_bands <= 0:
            raise ValueError("n_bands must be positive")
        self.logits = nn.Parameter(torch.zeros(n_bands, dtype=torch.float32))

    def sample(
        self, budget: int, *, generator: torch.Generator | None = None
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        if not 0 < budget <= self.logits.numel():
            raise ValueError("budget must be between 1 and the number of candidate bands")
        available = torch.ones_like(self.logits, dtype=torch.bool)
        selected: list[int] = []
        log_prob = torch.zeros((), dtype=self.logits.dtype, device=self.logits.device)
        entropy = torch.zeros((), dtype=self.logits.dtype, device=self.logits.device)
        for _ in range(budget):
            # Autograd saves boolean index tensors for backward, so use a per-step
            # clone rather than mutating a mask referenced by an earlier step.
            step_available = available.clone()
            masked_logits = self.logits.masked_fill(~step_available, -torch.inf)
            probabilities = torch.softmax(masked_logits, dim=0)
            action = torch.multinomial(probabilities, 1, generator=generator).squeeze(0)
            log_prob = log_prob + torch.log(probabilities[action].clamp_min(1e-12))
            valid_probabilities = probabilities[step_available]
            entropy = entropy - torch.sum(
                valid_probabilities * torch.log(valid_probabilities.clamp_min(1e-12))
            )
            action_index = int(action.item())
            selected.append(action_index)
            available[action_index] = False
        return selected, log_prob, entropy

    def greedy(self, budget: int) -> list[int]:
        if not 0 < budget <= self.logits.numel():
            raise ValueError("budget must be between 1 and the number of candidate bands")
        return torch.topk(self.logits.detach(), k=budget).indices.cpu().tolist()


def train_reinforce_selector(
    reward_fn: NearestCentroidBandReward,
    *,
    budget: int = 10,
    updates: int = 100,
    episodes_per_update: int = 16,
    learning_rate: float = 0.05,
    entropy_weight: float = 0.002,
    seed: int = 42,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[MaskedBandPolicy, RLSelectionResult]:
    """Train a small terminal-reward REINFORCE policy."""

    if updates <= 0 or episodes_per_update < 2:
        raise ValueError("updates must be positive and episodes_per_update must be at least 2")
    n_candidates = reward_fn.data.features.shape[1]
    policy = MaskedBandPolicy(n_candidates)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    generator = torch.Generator(device=policy.logits.device).manual_seed(seed)

    best_subset: list[int] | None = None
    best_reward = -np.inf
    best_accuracy = 0.0
    history: list[dict[str, float]] = []

    for update in range(1, updates + 1):
        traces: list[tuple[torch.Tensor, torch.Tensor]] = []
        rewards: list[float] = []
        accuracies: list[float] = []
        for _ in range(episodes_per_update):
            subset, log_probability, entropy = policy.sample(budget, generator=generator)
            reward, accuracy = reward_fn(subset)
            traces.append((log_probability, entropy))
            rewards.append(reward)
            accuracies.append(accuracy)
            if reward > best_reward:
                best_reward, best_accuracy, best_subset = reward, accuracy, list(subset)

        reward_tensor = torch.tensor(rewards, dtype=policy.logits.dtype)
        advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-6)
        log_probabilities = torch.stack([trace[0] for trace in traces])
        entropies = torch.stack([trace[1] for trace in traces])
        loss = -(advantages.detach() * log_probabilities).mean() - entropy_weight * entropies.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        greedy_subset = policy.greedy(budget)
        greedy_reward, greedy_accuracy = reward_fn(greedy_subset)
        if greedy_reward > best_reward:
            best_reward = greedy_reward
            best_accuracy = greedy_accuracy
            best_subset = greedy_subset

        record = {
            "update": float(update),
            "mean_reward": float(np.mean(rewards)),
            "mean_accuracy": float(np.mean(accuracies)),
            "greedy_reward": float(greedy_reward),
            "greedy_accuracy": float(greedy_accuracy),
            "best_reward": float(best_reward),
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(update, record)

    assert best_subset is not None
    # The subset reward is order-invariant.  Give exported records a stable,
    # policy-derived order rather than the arbitrary order of the sampled episode.
    learned_logits = policy.logits.detach().cpu().numpy()
    best_subset = sorted(best_subset, key=lambda index: learned_logits[index], reverse=True)
    test_accuracy = reward_fn.balanced_accuracy(best_subset, "test")
    full_band_test_accuracy = reward_fn.balanced_accuracy(
        list(range(n_candidates)), "test"
    )
    result = RLSelectionResult(
        selected_indices=best_subset,
        best_validation_reward=float(best_reward),
        best_validation_accuracy=float(best_accuracy),
        test_accuracy=float(test_accuracy),
        full_band_test_accuracy=float(full_band_test_accuracy),
        history=history,
    )
    return policy, result


def selected_band_records(
    data: LabeledBandData, selected_indices: Iterable[int]
) -> list[dict[str, float | int]]:
    """Return JSON-ready ranked band metadata."""

    records = []
    for rank, candidate_index in enumerate(selected_indices, start=1):
        band = data.bands[int(candidate_index)]
        records.append(
            {"rank": rank, "candidate_index": int(candidate_index), **asdict(band)}
        )
    return records
