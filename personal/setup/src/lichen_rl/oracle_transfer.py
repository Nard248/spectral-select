"""Exact three-band oracle and oracle-initialized state-dependent RL policy."""
from __future__ import annotations

import heapq
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .selection import (
    MaskedBandPolicy,
    NearestCentroidBandReward,
    RLSelectionResult,
)


@dataclass
class TripleOracleResult:
    """Exact ranking summary for evaluated three-band combinations."""

    top_combinations: np.ndarray
    top_rewards: np.ndarray
    top_accuracies: np.ndarray
    band_prior: np.ndarray
    pair_prior: np.ndarray
    total_combinations: int
    evaluated_combinations: int
    complete: bool

    @property
    def best_combination(self) -> list[int]:
        return self.top_combinations[0].astype(int).tolist()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            top_combinations=self.top_combinations,
            top_rewards=self.top_rewards,
            top_accuracies=self.top_accuracies,
            band_prior=self.band_prior,
            pair_prior=self.pair_prior,
            total_combinations=np.int64(self.total_combinations),
            evaluated_combinations=np.int64(self.evaluated_combinations),
            complete=np.bool_(self.complete),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TripleOracleResult":
        payload = np.load(path)
        return cls(
            top_combinations=payload["top_combinations"],
            top_rewards=payload["top_rewards"],
            top_accuracies=payload["top_accuracies"],
            band_prior=payload["band_prior"],
            pair_prior=payload["pair_prior"],
            total_combinations=int(payload["total_combinations"]),
            evaluated_combinations=int(payload["evaluated_combinations"]),
            complete=bool(payload["complete"]),
        )


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _derive_priors(
    combinations: np.ndarray,
    rewards: np.ndarray,
    n_candidates: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft participation priors from the retained top triples."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    weights = np.exp((rewards - rewards.max()) / temperature)
    band = np.zeros(n_candidates, dtype=np.float64)
    pair = np.zeros((n_candidates, n_candidates), dtype=np.float64)
    for combo, weight in zip(combinations, weights):
        i, j, k = (int(v) for v in combo)
        band[[i, j, k]] += weight
        pair[i, j] += weight
        pair[j, i] += weight
        pair[i, k] += weight
        pair[k, i] += weight
        pair[j, k] += weight
        pair[k, j] += weight
    if band.max() > 0:
        band /= band.max()
    if pair.max() > 0:
        pair /= pair.max()
    return band.astype(np.float32), pair.astype(np.float32)


def _checkpoint_payload(
    processed: int,
    heap: list[tuple[float, float, tuple[int, int, int]]],
    *,
    n_candidates: int,
    top_k: int,
    total: int,
) -> dict:
    ordered = sorted(heap, reverse=True)
    return {
        "processed": processed,
        "n_candidates": n_candidates,
        "top_k": top_k,
        "total": total,
        "top": [
            {"reward": reward, "accuracy": accuracy, "combination": list(combo)}
            for reward, accuracy, combo in ordered
        ],
    }


def exhaustive_triple_oracle(
    reward_fn: NearestCentroidBandReward,
    *,
    top_k: int = 5000,
    batch_size: int = 256,
    device: str = "auto",
    prior_temperature: float = 0.02,
    max_combinations: int | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int = 50_000,
    resume: bool = True,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TripleOracleResult:
    """Enumerate triples and retain the exact top-K under ``reward_fn``.

    Accuracy is computed in vectorized batches from cached per-band squared
    distances. If ``max_combinations`` is omitted, every ``n choose 3`` triple is
    evaluated. A checkpoint stores the enumeration offset and current top-K heap.
    """

    if top_k <= 0 or batch_size <= 0:
        raise ValueError("top_k and batch_size must be positive")
    n_candidates = reward_fn.data.features.shape[1]
    if n_candidates < 3:
        raise ValueError("at least three candidate bands are required")
    total = math.comb(n_candidates, 3)
    target = total if max_combinations is None else min(total, max_combinations)
    compute_device = _device(device)

    eval_idx = reward_fn.evaluation_indices["validation"]
    x = torch.as_tensor(reward_fn.features[eval_idx], dtype=torch.float32, device=compute_device)
    centroids = torch.as_tensor(reward_fn.centroids, dtype=torch.float32, device=compute_device)
    squared = (x[:, None, :] - centroids[None, :, :]).square()
    labels = reward_fn.data.labels[eval_idx]
    class_masks = [
        torch.as_tensor(labels == class_id, device=compute_device)
        for class_id in reward_fn.classes
    ]
    correlation = torch.as_tensor(
        reward_fn.absolute_correlation, dtype=torch.float32, device=compute_device
    )

    processed = 0
    heap: list[tuple[float, float, tuple[int, int, int]]] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["n_candidates"] != n_candidates or saved["top_k"] != top_k:
            raise ValueError("checkpoint configuration does not match this oracle run")
        processed = int(saved["processed"])
        heap = [
            (float(row["reward"]), float(row["accuracy"]), tuple(row["combination"]))
            for row in saved["top"]
        ]
        heapq.heapify(heap)

    iterator = itertools.islice(
        itertools.combinations(range(n_candidates), 3), processed, target
    )
    next_checkpoint = processed + checkpoint_interval
    while processed < target:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        combo = torch.as_tensor(batch, dtype=torch.long, device=compute_device)
        distances = (
            squared[:, :, combo[:, 0]]
            + squared[:, :, combo[:, 1]]
            + squared[:, :, combo[:, 2]]
        )
        predictions = distances.argmin(dim=1)
        recall_rows = []
        for class_index, mask in enumerate(class_masks):
            recall_rows.append((predictions[mask] == class_index).float().mean(dim=0))
        accuracy = torch.stack(recall_rows).mean(dim=0)
        redundancy = (
            correlation[combo[:, 0], combo[:, 1]]
            + correlation[combo[:, 0], combo[:, 2]]
            + correlation[combo[:, 1], combo[:, 2]]
        ) / 3.0
        rewards = accuracy - reward_fn.redundancy_weight * redundancy

        rewards_np = rewards.detach().cpu().numpy()
        accuracy_np = accuracy.detach().cpu().numpy()
        if len(heap) < top_k:
            candidate_positions = np.arange(len(batch))
        else:
            candidate_positions = np.flatnonzero(rewards_np > heap[0][0])
        for position in candidate_positions:
            row = (
                float(rewards_np[position]),
                float(accuracy_np[position]),
                tuple(int(v) for v in batch[int(position)]),
            )
            if len(heap) < top_k:
                heapq.heappush(heap, row)
            elif row[0] > heap[0][0]:
                heapq.heapreplace(heap, row)

        processed += len(batch)
        if progress_callback is not None:
            progress_callback(processed, target, float(max(rewards_np)))
        if checkpoint is not None and processed >= next_checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps(
                    _checkpoint_payload(
                        processed,
                        heap,
                        n_candidates=n_candidates,
                        top_k=top_k,
                        total=total,
                    )
                )
            )
            next_checkpoint = processed + checkpoint_interval

    ordered = sorted(heap, reverse=True)
    combinations_array = np.asarray([row[2] for row in ordered], dtype=np.int32)
    rewards_array = np.asarray([row[0] for row in ordered], dtype=np.float32)
    accuracies_array = np.asarray([row[1] for row in ordered], dtype=np.float32)
    band_prior, pair_prior = _derive_priors(
        combinations_array, rewards_array, n_candidates, prior_temperature
    )
    complete = processed == total
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps(
                _checkpoint_payload(
                    processed,
                    heap,
                    n_candidates=n_candidates,
                    top_k=top_k,
                    total=total,
                )
                | {"complete": complete}
            )
        )
    return TripleOracleResult(
        top_combinations=combinations_array,
        top_rewards=rewards_array,
        top_accuracies=accuracies_array,
        band_prior=band_prior,
        pair_prior=pair_prior,
        total_combinations=total,
        evaluated_combinations=processed,
        complete=complete,
    )


class OracleTransferPolicy(MaskedBandPolicy):
    """State-dependent policy initialized by triple band and pair priors."""

    def __init__(
        self,
        band_prior: np.ndarray,
        pair_prior: np.ndarray,
        absolute_correlation: np.ndarray,
        *,
        initialization_scale: float = 2.0,
    ) -> None:
        super().__init__(len(band_prior))
        prior = torch.as_tensor(band_prior, dtype=torch.float32)
        prior = (prior - prior.mean()) / prior.std().clamp_min(1e-6)
        with torch.no_grad():
            self.logits.copy_(initialization_scale * prior)
        self.register_buffer("pair_prior", torch.as_tensor(pair_prior, dtype=torch.float32))
        self.register_buffer(
            "absolute_correlation", torch.as_tensor(absolute_correlation, dtype=torch.float32)
        )
        self.pair_scale = nn.Parameter(torch.tensor(1.0))
        self.redundancy_scale = nn.Parameter(torch.tensor(0.25))

    def state_logits(self, selected: Sequence[int]) -> torch.Tensor:
        logits = self.logits
        if selected:
            chosen = torch.as_tensor(selected, dtype=torch.long, device=logits.device)
            compatibility = self.pair_prior[:, chosen].mean(dim=1)
            redundancy = self.absolute_correlation[:, chosen].mean(dim=1)
            logits = logits + self.pair_scale * compatibility - self.redundancy_scale * redundancy
        return logits

    def sample(
        self, budget: int, *, generator: torch.Generator | None = None
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        if not 0 < budget <= self.logits.numel():
            raise ValueError("budget must be between 1 and the number of candidate bands")
        available = torch.ones_like(self.logits, dtype=torch.bool)
        selected: list[int] = []
        log_probability = torch.zeros((), dtype=self.logits.dtype, device=self.logits.device)
        entropy = torch.zeros_like(log_probability)
        for _ in range(budget):
            step_available = available.clone()
            state_logits = self.state_logits(selected).masked_fill(~step_available, -torch.inf)
            probabilities = torch.softmax(state_logits, dim=0)
            action = torch.multinomial(probabilities, 1, generator=generator).squeeze(0)
            log_probability = log_probability + torch.log(
                probabilities[action].clamp_min(1e-12)
            )
            valid = probabilities[step_available]
            entropy = entropy - torch.sum(valid * torch.log(valid.clamp_min(1e-12)))
            action_index = int(action.item())
            selected.append(action_index)
            available[action_index] = False
        return selected, log_probability, entropy

    def greedy(self, budget: int) -> list[int]:
        available = torch.ones_like(self.logits, dtype=torch.bool)
        selected: list[int] = []
        for _ in range(budget):
            scores = self.state_logits(selected).masked_fill(~available, -torch.inf)
            action = int(torch.argmax(scores).item())
            selected.append(action)
            available[action] = False
        return selected


def train_oracle_transfer_selector(
    reward_fn: NearestCentroidBandReward,
    oracle: TripleOracleResult,
    *,
    budget: int = 13,
    updates: int = 150,
    episodes_per_update: int = 16,
    learning_rate: float = 0.02,
    entropy_weight: float = 0.002,
    seed: int = 42,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[OracleTransferPolicy, RLSelectionResult]:
    """Warm-start a larger-budget policy with exact triple-derived priors."""

    policy = OracleTransferPolicy(
        oracle.band_prior, oracle.pair_prior, reward_fn.absolute_correlation
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_subset: list[int] | None = None
    best_reward = -np.inf
    best_accuracy = 0.0
    history: list[dict[str, float]] = []

    for update in range(1, updates + 1):
        traces = []
        rewards = []
        accuracies = []
        for _ in range(episodes_per_update):
            subset, log_probability, entropy = policy.sample(budget, generator=generator)
            reward, accuracy = reward_fn(subset)
            traces.append((log_probability, entropy))
            rewards.append(reward)
            accuracies.append(accuracy)
            if reward > best_reward:
                best_reward, best_accuracy, best_subset = reward, accuracy, list(subset)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-6)
        loss = -(
            advantages.detach() * torch.stack([row[0] for row in traces])
        ).mean() - entropy_weight * torch.stack([row[1] for row in traces]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        greedy = policy.greedy(budget)
        greedy_reward, greedy_accuracy = reward_fn(greedy)
        if greedy_reward > best_reward:
            best_reward, best_accuracy, best_subset = greedy_reward, greedy_accuracy, greedy
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
    learned_logits = policy.logits.detach().cpu().numpy()
    best_subset = sorted(best_subset, key=lambda index: learned_logits[index], reverse=True)
    result = RLSelectionResult(
        selected_indices=best_subset,
        best_validation_reward=float(best_reward),
        best_validation_accuracy=float(best_accuracy),
        test_accuracy=reward_fn.balanced_accuracy(best_subset, "test"),
        full_band_test_accuracy=reward_fn.balanced_accuracy(
            list(range(reward_fn.data.features.shape[1])), "test"
        ),
        history=history,
    )
    return policy, result
