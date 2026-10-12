"""Train the small REINFORCE wavelength selector on the personal Lichens_2 data.

Run after installing ``personal/setup``, for example:

    lichen-rl-select --budget 10 --updates 100
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .selection import (
    NearestCentroidBandReward,
    load_labeled_band_data,
    selected_band_records,
    train_reinforce_selector,
)


SETUP_ROOT = Path(__file__).resolve().parents[2]
PERSONAL_ROOT = SETUP_ROOT.parent
DEFAULT_DATA = (
    PERSONAL_ROOT
    / "Processed/Lichens_2/data_cutoff_40nm_exposure_max_power_min.pkl"
)
DEFAULT_MASK = PERSONAL_ROOT / "Processed/Lichens_2/lichens_2_class_mask.npy"
DEFAULT_OUTPUT = SETUP_ROOT / "results/rl_lichens2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--class-mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--redundancy-weight", type=float, default=0.02)
    parser.add_argument("--random-baselines", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.data}")
    data = load_labeled_band_data(args.data, args.class_mask, seed=args.seed)
    counts = {p: int(np.sum(data.partitions == p)) for p in ("train", "validation", "test")}
    print(
        f"Loaded {data.features.shape[0]:,} labelled pixels, "
        f"{data.features.shape[1]} candidate bands; split pixels={counts}"
    )
    reward = NearestCentroidBandReward(
        data, redundancy_weight=args.redundancy_weight, seed=args.seed
    )

    report_every = max(1, args.updates // 10)

    def progress(update: int, record: dict[str, float]) -> None:
        if update == 1 or update % report_every == 0 or update == args.updates:
            print(
                f"update {update:4d}/{args.updates}: "
                f"mean reward={record['mean_reward']:.4f}, "
                f"greedy val BA={record['greedy_accuracy']:.4f}, "
                f"best reward={record['best_reward']:.4f}"
            )

    policy, result = train_reinforce_selector(
        reward,
        budget=args.budget,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        progress_callback=progress,
    )

    train_idx = data.indices("train")
    raw_variance = data.features[train_idx].var(axis=0)
    variance_subset = np.argsort(raw_variance)[-args.budget :].tolist()

    # Supervised filter baseline calculated strictly from training objects.
    grand_mean = data.features[train_idx].mean(axis=0, dtype=np.float64)
    between = np.zeros(data.features.shape[1], dtype=np.float64)
    within = np.zeros_like(between)
    for class_id in np.unique(data.labels):
        class_values = data.features[
            train_idx[data.labels[train_idx] == class_id]
        ].astype(np.float64)
        class_mean = class_values.mean(axis=0)
        between += len(class_values) * (class_mean - grand_mean) ** 2
        within += ((class_values - class_mean) ** 2).sum(axis=0)
    fisher_subset = np.argsort(between / (within + 1e-12))[-args.budget :].tolist()

    def subset_metrics(subset: list[int]) -> dict[str, float]:
        return {
            "validation_balanced_accuracy": reward.balanced_accuracy(subset, "validation"),
            "test_balanced_accuracy": reward.balanced_accuracy(subset, "test"),
            "mean_absolute_correlation": reward.redundancy(subset),
        }

    baselines = {
        "variance": subset_metrics(variance_subset),
        "fisher_train_only": subset_metrics(fisher_subset),
        "all_bands": subset_metrics(list(range(data.features.shape[1]))),
    }
    random_validation: list[float] = []
    random_test: list[float] = []
    baseline_rng = np.random.default_rng(args.seed)
    for _ in range(args.random_baselines):
        subset = baseline_rng.choice(
            data.features.shape[1], args.budget, replace=False
        ).tolist()
        random_validation.append(reward.balanced_accuracy(subset, "validation"))
        random_test.append(reward.balanced_accuracy(subset, "test"))
    baselines["random"] = {
        "trials": args.random_baselines,
        "validation_mean": float(np.mean(random_validation)),
        "validation_std": float(np.std(random_validation)),
        "validation_95th_percentile": float(np.percentile(random_validation, 95)),
        "test_mean": float(np.mean(random_test)),
        "test_std": float(np.std(random_test)),
        "test_95th_percentile": float(np.percentile(random_test, 95)),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    bands = selected_band_records(data, result.selected_indices)
    payload = {
        "method": "REINFORCE masked categorical band selector",
        "data_path": str(args.data),
        "class_mask_path": str(args.class_mask),
        "config": vars(args) | {"data": str(args.data), "class_mask": str(args.class_mask), "output": str(args.output)},
        "split_pixel_counts": counts,
        "metrics": {k: v for k, v in asdict(result).items() if k not in {"history", "selected_indices"}},
        "selected_bands": bands,
        "baselines": baselines,
        "history": result.history,
    }
    (args.output / "selection.json").write_text(json.dumps(payload, indent=2))
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "n_candidates": data.features.shape[1],
            "budget": args.budget,
            "selected_indices": result.selected_indices,
            "bands": bands,
        },
        args.output / "policy.pth",
    )

    print("\nSelected bands:")
    for band in bands:
        print(
            f"  {band['rank']:2d}. Ex {band['excitation_nm']:6.1f} nm / "
            f"Em {band['emission_nm']:6.1f} nm"
        )
    print(f"Validation balanced accuracy: {result.best_validation_accuracy:.4f}")
    print(f"Test balanced accuracy:       {result.test_accuracy:.4f}")
    print(f"Full-band test accuracy:      {result.full_band_test_accuracy:.4f}")
    print(
        f"Random-{args.budget} test mean±std:   "
        f"{baselines['random']['test_mean']:.4f} ± {baselines['random']['test_std']:.4f}"
    )
    print(
        f"Variance-{args.budget} test accuracy: "
        f"{baselines['variance']['test_balanced_accuracy']:.4f}"
    )
    print(
        f"Fisher-{args.budget} test accuracy:   "
        f"{baselines['fisher_train_only']['test_balanced_accuracy']:.4f}"
    )
    print(f"Saved model and report to {args.output}")


if __name__ == "__main__":
    main()
