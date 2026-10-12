"""Sequential, non-greedy Monte-Carlo wavelength-pair selector."""

from dataclasses import asdict, dataclass
import numpy as np

from .evaluator import SubsetEvaluator
from .reward import RewardWeights, mean_redundancy, normalize, stable_softmax


@dataclass(frozen=True)
class SelectorConfig:
    n_pairs: int
    episodes: int = 20
    rollouts: int = 10
    temperature: float = 1.0
    temperature_decay: float = 0.95
    min_temperature: float = 0.05
    normalization: str = "zscore"
    deterministic: bool = False
    seed: int = 42
    global_contexts: int = 0
    weights: RewardWeights = RewardWeights()


class MonteCarloSelector:
    def __init__(self, evaluator: SubsetEvaluator, X: np.ndarray, config: SelectorConfig):
        self.evaluator, self.X, self.config = evaluator, np.asarray(X), config
        if not 1 <= config.n_pairs <= self.X.shape[1]:
            raise ValueError("n_pairs must be between 1 and the number of candidates")
        if config.rollouts < 1 or config.episodes < 1:
            raise ValueError("episodes and rollouts must be positive")
        self.rng = np.random.default_rng(config.seed)
        baseline = evaluator.score([])
        self.individual = np.array([evaluator.score([i]) - baseline for i in range(self.X.shape[1])])
        self.global_marginal = self._global_marginals()

    def _global_marginals(self) -> np.ndarray:
        """Estimate context-averaged marginal value once; zero disables it."""
        values = np.zeros(self.X.shape[1])
        if self.config.global_contexts < 1 or self.config.n_pairs == 1:
            return values
        for candidate in range(self.X.shape[1]):
            available = np.array([i for i in range(self.X.shape[1]) if i != candidate])
            gains = []
            for _ in range(self.config.global_contexts):
                context = self.rng.choice(available, self.config.n_pairs - 1, replace=False).tolist()
                gains.append(self.evaluator.score(context + [candidate]) - self.evaluator.score(context))
            values[candidate] = np.mean(gains)
        return values

    def _future(self, selected: list[int], candidate: int) -> tuple[float, float, float, float]:
        prefix = selected + [candidate]
        slots = self.config.n_pairs - len(prefix)
        available = np.array([i for i in range(self.X.shape[1]) if i not in prefix], dtype=int)
        scores = []
        for _ in range(self.config.rollouts):
            completion = self.rng.choice(available, slots, replace=False).tolist() if slots else []
            scores.append(self.evaluator.score(prefix + completion))
        return float(np.mean(scores)), float(np.std(scores)), float(np.min(scores)), float(np.max(scores))

    def run(self) -> dict:
        episodes, frequencies = [], np.zeros(self.X.shape[1], dtype=int)
        best = None
        for episode in range(self.config.episodes):
            selected: list[int] = []
            steps = []
            temperature = max(self.config.min_temperature, self.config.temperature * self.config.temperature_decay**episode)
            while len(selected) < self.config.n_pairs:
                current = self.evaluator.score(selected)
                candidates = [i for i in range(self.X.shape[1]) if i not in selected]
                rows = []
                for candidate in candidates:
                    future_mean, future_std, future_min, future_max = self._future(selected, candidate)
                    rows.append({"candidate": candidate, "individual": float(self.individual[candidate]), "marginal": self.evaluator.score(selected + [candidate]) - current, "future": future_mean, "future_std": future_std, "future_min": future_min, "future_max": future_max, "global_marginal": float(self.global_marginal[candidate]), "redundancy": mean_redundancy(self.X, candidate, selected)})
                for component in ("individual", "marginal", "future", "global_marginal", "redundancy"):
                    values = normalize([r[component] for r in rows], self.config.normalization)
                    for row, value in zip(rows, values):
                        row[f"normalized_{component}"] = float(value)
                w = self.config.weights
                q = np.array([w.individual*r["normalized_individual"] + w.marginal*r["normalized_marginal"] + w.future*r["normalized_future"] + w.global_marginal*r["normalized_global_marginal"] + w.redundancy*r["normalized_redundancy"] for r in rows])
                probabilities = stable_softmax(q, temperature)
                pos = int(np.argmax(q)) if self.config.deterministic else int(self.rng.choice(len(rows), p=probabilities))
                for row, q_value, probability in zip(rows, q, probabilities):
                    row["q"], row["probability"] = float(q_value), float(probability)
                chosen = rows[pos]["candidate"]
                steps.append({"selected_before": selected.copy(), "chosen": chosen, "temperature": temperature, "candidates": rows})
                selected.append(chosen)
            score = self.evaluator.score(selected)
            for index in selected: frequencies[index] += 1
            record = {"episode": episode, "selected": selected, "score": score, "steps": steps}
            episodes.append(record)
            if best is None or score > best["score"]:
                best = {"selected": selected.copy(), "score": score, "episode": episode}
        return {"config": {**asdict(self.config), "weights": asdict(self.config.weights)}, "best": best, "episodes": episodes, "selection_frequencies": frequencies.tolist(), "individual_scores": self.individual.tolist()}
