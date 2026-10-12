import numpy as np
import pytest
import torch

from lichen_rl.selection import (
    BandAction,
    LabeledBandData,
    MaskedBandPolicy,
    NearestCentroidBandReward,
    component_partition_map,
    train_reinforce_selector,
)


def test_component_partition_keeps_objects_together():
    mask = np.full((9, 12), -1, dtype=np.int16)
    for class_id, row in enumerate((0, 5)):
        for col in (0, 4, 8):
            mask[row : row + 3, col : col + 3] = class_id

    split = component_partition_map(mask, seed=3)
    for class_id in (0, 1):
        assert set(split[mask == class_id]) == {"train", "validation", "test"}
    assert np.all(split[mask < 0] == "background")


def test_component_partition_rejects_too_few_objects():
    mask = np.full((4, 4), -1, dtype=np.int16)
    mask[:2, :2] = 0
    mask[2:, 2:] = 0
    with pytest.raises(ValueError, match="at least 3"):
        component_partition_map(mask)


def test_policy_samples_unique_actions_and_backpropagates():
    policy = MaskedBandPolicy(8)
    generator = torch.Generator().manual_seed(4)
    selected, log_probability, entropy = policy.sample(5, generator=generator)
    assert len(selected) == len(set(selected)) == 5
    loss = -log_probability - 0.01 * entropy
    loss.backward()
    assert policy.logits.grad is not None
    assert torch.isfinite(policy.logits.grad).all()


def _tiny_problem(seed=0):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(3), 90)
    partitions = np.tile(np.repeat(["train", "validation", "test"], 30), 3)
    features = rng.normal(0, 1, (labels.size, 8)).astype(np.float32)
    features[:, 0] += 3.0 * labels
    features[:, 1] += 1.5 * labels
    data = LabeledBandData(
        features=features,
        labels=labels,
        coordinates=np.zeros((labels.size, 2), dtype=np.int64),
        partitions=partitions,
        bands=[BandAction(300.0, 400.0 + i * 10, i) for i in range(features.shape[1])],
    )
    return NearestCentroidBandReward(data, redundancy_weight=0.0, max_pixels_per_class=None)


def test_reinforce_returns_requested_budget_and_metrics():
    reward = _tiny_problem()
    _, result = train_reinforce_selector(
        reward, budget=2, updates=8, episodes_per_update=6, seed=5
    )
    assert len(result.selected_indices) == len(set(result.selected_indices)) == 2
    assert 0.0 <= result.best_validation_accuracy <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0
    assert len(result.history) == 8
