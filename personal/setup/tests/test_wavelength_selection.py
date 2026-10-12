import numpy as np
from sklearn.datasets import make_classification

from wavelength_selection.evaluator import SubsetEvaluator
from wavelength_selection.monte_carlo_selector import MonteCarloSelector, SelectorConfig
from wavelength_selection.rayleigh_filter import WavelengthPair, apply_rayleigh_filter
from wavelength_selection.reward import normalize, stable_softmax


def _evaluator(seed=4):
    X, y = make_classification(n_samples=120, n_features=8, n_informative=4, n_classes=3, random_state=seed)
    groups = np.repeat(np.arange(30), 4)
    y = np.repeat(np.arange(3), 40)
    return X, SubsetEvaluator(X, y, groups, folds=2, seed=seed)


def test_rayleigh_filter_reports_both_rules():
    pairs = [WavelengthPair(300, 320), WavelengthPair(300, 600), WavelengthPair(300, 700)]
    accepted, rejected = apply_rayleigh_filter(pairs, 40)
    assert accepted == [pairs[2]]
    assert {reason for item in rejected for reason in item.reasons} == {"first_order_rayleigh", "second_order_rayleigh"}


def test_normalization_and_softmax_are_stable():
    assert np.all(normalize([2, 2, 2]) == 0)
    probability = stable_softmax([10000, 10001])
    assert np.isclose(probability.sum(), 1) and np.isfinite(probability).all()


def test_evaluator_cache_is_order_independent():
    _, evaluator = _evaluator()
    assert evaluator.score([2, 1]) == evaluator.score([1, 2])
    assert evaluator.cache_hits == 1


def test_selector_returns_exact_unique_count_and_reproduces():
    X, evaluator1 = _evaluator()
    config = SelectorConfig(n_pairs=3, episodes=2, rollouts=2, seed=8)
    first = MonteCarloSelector(evaluator1, X, config).run()
    _, evaluator2 = _evaluator()
    second = MonteCarloSelector(evaluator2, X, config).run()
    assert first["best"]["selected"] == second["best"]["selected"]
    assert len(first["best"]["selected"]) == len(set(first["best"]["selected"])) == 3
