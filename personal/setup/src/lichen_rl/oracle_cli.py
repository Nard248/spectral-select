"""Exhaustive 3-band oracle followed by oracle-initialized larger-budget RL."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .oracle_transfer import (
    TripleOracleResult,
    exhaustive_triple_oracle,
    train_oracle_transfer_selector,
)
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
DEFAULT_OUTPUT = SETUP_ROOT / "results/oracle_transfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--class-mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=13)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--redundancy-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--oracle-pixels-per-class", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--skip-scratch", action="store_true")
    parser.add_argument("--rebuild-oracle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_labeled_band_data(args.data, args.class_mask, seed=args.seed)
    max_pixels = (
        None if args.oracle_pixels_per_class <= 0 else args.oracle_pixels_per_class
    )
    reward = NearestCentroidBandReward(
        data,
        redundancy_weight=args.redundancy_weight,
        max_pixels_per_class=max_pixels,
        seed=args.seed,
    )
    n_candidates = data.features.shape[1]
    total = math.comb(n_candidates, 3)
    print(
        f"Loaded {len(data.labels):,} labelled pixels and {n_candidates} candidates; "
        f"the 3-band search contains {total:,} combinations."
    )

    oracle_path = args.output / "triple_oracle.npz"
    checkpoint_path = args.output / "triple_oracle_checkpoint.json"
    if oracle_path.exists() and not args.rebuild_oracle:
        oracle = TripleOracleResult.load(oracle_path)
        print(
            f"Loaded oracle artifact with {oracle.evaluated_combinations:,}/"
            f"{oracle.total_combinations:,} evaluated combinations."
        )
    else:
        last_percent = -1

        def oracle_progress(processed: int, target: int, batch_best: float) -> None:
            nonlocal last_percent
            percent = int(100 * processed / target)
            if percent != last_percent and (percent % 1 == 0):
                print(
                    f"oracle {processed:,}/{target:,} ({percent:3d}%), "
                    f"batch best={batch_best:.4f}"
                )
                last_percent = percent

        oracle = exhaustive_triple_oracle(
            reward,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            max_combinations=args.max_combinations,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            resume=not args.rebuild_oracle,
            progress_callback=oracle_progress,
        )
        oracle.save(oracle_path)

    oracle_best = oracle.best_combination
    oracle_metrics = {
        "complete": oracle.complete,
        "evaluated_combinations": oracle.evaluated_combinations,
        "total_combinations": oracle.total_combinations,
        "validation_reward": float(oracle.top_rewards[0]),
        "validation_balanced_accuracy": float(oracle.top_accuracies[0]),
        "test_balanced_accuracy": reward.balanced_accuracy(oracle_best, "test"),
        "selected_bands": selected_band_records(data, oracle_best),
    }
    print(
        f"Oracle best triple: {oracle_best}, validation reward "
        f"{oracle_metrics['validation_reward']:.4f}, test BA "
        f"{oracle_metrics['test_balanced_accuracy']:.4f}"
    )

    summary: dict = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "oracle": oracle_metrics,
    }
    if not args.oracle_only:
        report_every = max(1, args.updates // 10)

        def transfer_progress(update: int, record: dict[str, float]) -> None:
            if update == 1 or update % report_every == 0 or update == args.updates:
                print(
                    f"transfer {update:4d}/{args.updates}: mean reward="
                    f"{record['mean_reward']:.4f}, best={record['best_reward']:.4f}"
                )

        transfer_policy, transfer_result = train_oracle_transfer_selector(
            reward,
            oracle,
            budget=args.budget,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            learning_rate=args.learning_rate,
            entropy_weight=args.entropy_weight,
            seed=args.seed,
            progress_callback=transfer_progress,
        )
        summary["oracle_transfer"] = {
            "metrics": {
                key: value
                for key, value in asdict(transfer_result).items()
                if key not in {"history", "selected_indices"}
            },
            "selected_bands": selected_band_records(
                data, transfer_result.selected_indices
            ),
            "history": transfer_result.history,
        }
        torch.save(
            {
                "state_dict": transfer_policy.state_dict(),
                "budget": args.budget,
                "selected_indices": transfer_result.selected_indices,
                "oracle_path": str(oracle_path),
            },
            args.output / "oracle_transfer_policy.pth",
        )

        if not args.skip_scratch:
            scratch_policy, scratch_result = train_reinforce_selector(
                reward,
                budget=args.budget,
                updates=args.updates,
                episodes_per_update=args.episodes_per_update,
                learning_rate=args.learning_rate,
                entropy_weight=args.entropy_weight,
                seed=args.seed,
            )
            summary["scratch"] = {
                "metrics": {
                    key: value
                    for key, value in asdict(scratch_result).items()
                    if key not in {"history", "selected_indices"}
                },
                "selected_bands": selected_band_records(
                    data, scratch_result.selected_indices
                ),
                "history": scratch_result.history,
            }
            torch.save(
                {
                    "state_dict": scratch_policy.state_dict(),
                    "budget": args.budget,
                    "selected_indices": scratch_result.selected_indices,
                },
                args.output / "scratch_policy.pth",
            )
            print(
                f"Budget {args.budget}: transfer test BA="
                f"{transfer_result.test_accuracy:.4f}; scratch test BA="
                f"{scratch_result.test_accuracy:.4f}"
            )

    (args.output / "oracle_transfer_report.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Saved oracle and transfer report to {args.output}")


if __name__ == "__main__":
    main()
