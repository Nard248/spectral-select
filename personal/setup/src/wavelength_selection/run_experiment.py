"""CLI for the Monte-Carlo-guided selection experiment."""

import argparse, csv, json, os, tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"spectral-select-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())/"spectral-select-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .baselines import run_baselines
from .data_loader import load_lichen_dataset
from .evaluator import SubsetEvaluator
from .monte_carlo_selector import MonteCarloSelector, SelectorConfig
from .reward import RewardWeights


def _pair(pair):
    return {"pair_id": pair.pair_id, "excitation_nm": pair.excitation_nm, "emission_nm": pair.emission_nm}


def save_results(output, dataset, result, baselines, evaluator):
    output.mkdir(parents=True, exist_ok=True)
    pairs, best = dataset.pairs, result["best"]
    summary = {"dataset": {"processed_path": str(dataset.source_path), "samples": len(dataset.y), "raw_pairs": dataset.raw_pair_count, "physically_accepted": dataset.physically_accepted_count, "usable_pairs": len(pairs), "rejected_pairs": len(dataset.rejected)}, "best": {**best, "pairs": [_pair(pairs[i]) for i in best["selected"]]}, "baselines": baselines, "cache": {"unique_evaluations": len(evaluator.cache), "hits": evaluator.cache_hits}}
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output/"episodes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output/"rejected_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "excitation_nm", "emission_nm", "reasons"]); writer.writeheader()
        for item in dataset.rejected: writer.writerow({**_pair(item.pair), "reasons": ";".join(item.reasons)})
    with (output/"pair_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "pair_id", "excitation_nm", "emission_nm", "individual_gain", "selection_count"]); writer.writeheader()
        for i, pair in enumerate(pairs): writer.writerow({"index": i, **_pair(pair), "individual_gain": result["individual_scores"][i], "selection_count": result["selection_frequencies"][i]})
    plt.figure(figsize=(7,4)); plt.plot([e["episode"] for e in result["episodes"]], [e["score"] for e in result["episodes"]], marker="o"); plt.xlabel("Episode"); plt.ylabel(evaluator.metric_name); plt.tight_layout(); plt.savefig(output/"episode_scores.png", dpi=150); plt.close()
    order = np.argsort(result["selection_frequencies"])[-20:]
    plt.figure(figsize=(9,5)); plt.barh([pairs[i].pair_id for i in order], np.asarray(result["selection_frequencies"])[order]); plt.xlabel("Selection count"); plt.tight_layout(); plt.savefig(output/"selection_frequency.png", dpi=150); plt.close()
    selected = best["selected"]
    plt.figure(figsize=(6,5)); plt.scatter([p.emission_nm for p in pairs], [p.excitation_nm for p in pairs], s=12, alpha=.25); plt.scatter([pairs[i].emission_nm for i in selected], [pairs[i].excitation_nm for i in selected], c="red", s=55); plt.xlabel("Emission (nm)"); plt.ylabel("Excitation (nm)"); plt.tight_layout(); plt.savefig(output/"selected_eem.png", dpi=150); plt.close()


def build_parser():
    personal = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description="Monte-Carlo-guided lichen wavelength selection")
    p.add_argument("--processed", type=Path, default=personal/"Processed/Lichens_2/data_cutoff_40nm_exposure_max_power_min.pkl")
    p.add_argument("--mask", type=Path, default=personal/"Processed/Lichens_2/lichens_2_class_mask.npy")
    p.add_argument("--output", type=Path, default=personal/"setup/results/monte_carlo_selection")
    p.add_argument("--n-pairs", type=int, default=3); p.add_argument("--episodes", type=int, default=10); p.add_argument("--rollouts", type=int, default=5)
    p.add_argument("--classifier", choices=["knn","logreg","svm"], default="knn"); p.add_argument("--metric", choices=["balanced_accuracy","macro_f1","accuracy"], default="balanced_accuracy")
    p.add_argument("--folds", type=int, default=3); p.add_argument("--offset-nm", type=float, default=40); p.add_argument("--max-pixels-per-group", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1); p.add_argument("--temperature-decay", type=float, default=.95); p.add_argument("--normalization", choices=["zscore","minmax","none"], default="zscore")
    p.add_argument("--global-contexts", type=int, default=0, help="Contexts per candidate for optional global marginal reward")
    p.add_argument("--weight-individual", type=float, default=.1); p.add_argument("--weight-marginal", type=float, default=.3)
    p.add_argument("--weight-future", type=float, default=.5); p.add_argument("--weight-redundancy", type=float, default=-.1); p.add_argument("--weight-global", type=float, default=0)
    p.add_argument("--deterministic", action="store_true"); p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    data = load_lichen_dataset(a.processed, a.mask, a.offset_nm, a.max_pixels_per_group, a.seed)
    print(f"Raw pairs: {data.raw_pair_count}; accepted: {data.physically_accepted_count}; rejected: {len(data.rejected)}; usable: {len(data.pairs)}")
    reasons = {}
    for item in data.rejected:
        for reason in item.reasons: reasons[reason] = reasons.get(reason, 0) + 1
    print("Rejection reasons:", reasons)
    evaluator = SubsetEvaluator(data.X, data.y, data.groups, a.folds, a.classifier, a.metric, a.seed)
    weights = RewardWeights(a.weight_individual, a.weight_marginal, a.weight_future, a.weight_redundancy, a.weight_global)
    config = SelectorConfig(a.n_pairs, a.episodes, a.rollouts, a.temperature, a.temperature_decay, normalization=a.normalization, deterministic=a.deterministic, seed=a.seed, global_contexts=a.global_contexts, weights=weights)
    result = MonteCarloSelector(evaluator, data.X, config).run()
    baselines = run_baselines(evaluator, len(data.pairs), a.n_pairs, result["individual_scores"], a.episodes, a.seed)
    save_results(a.output, data, result, baselines, evaluator)
    print("Best score:", result["best"]["score"]); print("Best pairs:", ", ".join(data.pairs[i].pair_id for i in result["best"]["selected"])); print("Results:", a.output.resolve())
    return 0


if __name__ == "__main__": raise SystemExit(main())
