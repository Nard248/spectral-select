import itertools
import math

import numpy as np

from lichen_rl.oracle_transfer import (
    TripleOracleResult,
    exhaustive_triple_oracle,
    train_oracle_transfer_selector,
)
from lichen_rl.selection import BandAction, LabeledBandData, NearestCentroidBandReward


def _problem(seed=2):
    rng = np.random.default_rng(seed)
    labels = []
    partitions = []
    rows = []
    for partition in ("train", "validation", "test"):
        for class_id in range(3):
            for _ in range(18):
                row = rng.normal(0, 0.8, 6)
                row[0] += 2.5 * class_id
                row[2] += 1.5 * (class_id == 1)
                row[4] += 1.5 * (class_id == 2)
                rows.append(row)
                labels.append(class_id)
                partitions.append(partition)
    data = LabeledBandData(
        features=np.asarray(rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        coordinates=np.zeros((len(rows), 2), dtype=np.int64),
        partitions=np.asarray(partitions),
        bands=[BandAction(350.0, 450.0 + 10 * i, i) for i in range(6)],
    )
    return NearestCentroidBandReward(
        data, redundancy_weight=0.01, max_pixels_per_class=None, seed=seed
    )


def test_exhaustive_oracle_matches_brute_force(tmp_path):
    reward = _problem()
    brute = sorted(
        ((reward(combo)[0], combo) for combo in itertools.combinations(range(6), 3)),
        reverse=True,
    )
    oracle = exhaustive_triple_oracle(
        reward,
        top_k=10,
        batch_size=4,
        device="cpu",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert oracle.complete
    assert oracle.evaluated_combinations == math.comb(6, 3)
    assert oracle.best_combination == list(brute[0][1])
    assert np.isclose(oracle.top_rewards[0], brute[0][0], atol=1e-6)
    assert oracle.band_prior.shape == (6,)
    assert oracle.pair_prior.shape == (6, 6)


def test_oracle_round_trip(tmp_path):
    oracle = exhaustive_triple_oracle(
        _problem(), top_k=5, batch_size=5, device="cpu"
    )
    path = oracle.save(tmp_path / "oracle.npz")
    restored = TripleOracleResult.load(path)
    assert restored.best_combination == oracle.best_combination
    assert np.array_equal(restored.top_combinations, oracle.top_combinations)
    assert restored.complete


def test_oracle_transfer_trains_for_larger_budget():
    reward = _problem()
    oracle = exhaustive_triple_oracle(
        reward, top_k=10, batch_size=5, device="cpu"
    )
    policy, result = train_oracle_transfer_selector(
        reward,
        oracle,
        budget=4,
        updates=4,
        episodes_per_update=4,
        seed=7,
    )
    assert len(result.selected_indices) == len(set(result.selected_indices)) == 4
    assert len(result.history) == 4
    assert policy.pair_prior.shape == (6, 6)
