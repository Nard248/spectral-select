"""Cached, grouped cross-validation objective for wavelength subsets."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


@dataclass(frozen=True)
class Evaluation:
    balanced_accuracy: float
    macro_f1: float
    accuracy: float

    def metric(self, name: str) -> float:
        return float(getattr(self, name))


def _classifier(name: str, seed: int):
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=seed))
    raise ValueError(f"Unknown classifier: {name}")


class SubsetEvaluator:
    def __init__(self, X, y, groups, folds=3, classifier="knn", metric="balanced_accuracy", seed=42):
        self.X, self.y, self.groups = np.asarray(X), np.asarray(y), np.asarray(groups)
        self.metric_name = metric
        per_class_groups = [len(np.unique(self.groups[self.y == c])) for c in np.unique(self.y)]
        if folds > min(per_class_groups):
            raise ValueError(f"folds={folds} exceeds the smallest per-class object count ({min(per_class_groups)})")
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        self.splits = list(splitter.split(self.X, self.y, self.groups))
        self.model = _classifier(classifier, seed)
        self.cache: dict[tuple[int, ...], Evaluation] = {}
        self.cache_hits = 0

    def evaluate(self, subset) -> Evaluation:
        key = tuple(sorted(set(int(i) for i in subset)))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        scores = []
        for train, test in self.splits:
            if key:
                model = clone(self.model)
                train_X, test_X = self.X[train][:, key], self.X[test][:, key]
            else:
                model = DummyClassifier(strategy="most_frequent")
                train_X, test_X = np.zeros((len(train), 1)), np.zeros((len(test), 1))
            model.fit(train_X, self.y[train])
            pred = model.predict(test_X)
            scores.append((balanced_accuracy_score(self.y[test], pred), f1_score(self.y[test], pred, average="macro", zero_division=0), accuracy_score(self.y[test], pred)))
        result = Evaluation(*np.mean(scores, axis=0))
        self.cache[key] = result
        return result

    def score(self, subset) -> float:
        return self.evaluate(subset).metric(self.metric_name)
