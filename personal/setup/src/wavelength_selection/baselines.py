"""Comparable wavelength-selection baselines."""

import numpy as np


def run_baselines(evaluator, n_candidates, n_pairs, individual_scores, repeats=20, seed=42):
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(repeats):
        subset = rng.choice(n_candidates, n_pairs, replace=False).tolist()
        trials.append({"selected": subset, "score": evaluator.score(subset)})
    top = np.argsort(individual_scores)[-n_pairs:][::-1].tolist()
    greedy = []
    while len(greedy) < n_pairs:
        remaining = [i for i in range(n_candidates) if i not in greedy]
        greedy.append(max(remaining, key=lambda i: evaluator.score(greedy + [i])))
    return {"random": {"mean": float(np.mean([r["score"] for r in trials])), "best": max(trials, key=lambda r: r["score"]), "trials": trials}, "top_individual": {"selected": top, "score": evaluator.score(top)}, "greedy": {"selected": greedy, "score": evaluator.score(greedy)}}
