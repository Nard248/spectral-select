"""Shared contracts for the reversible-RL band-selection study (lead-owned; change only via the lead).

A ``Task`` is one selection problem: the training specimens of one outer fold of one dataset, plus the
held-out outer-test specimens that only ``evaluate.py`` may touch. Every searcher (baseline or RL) sees
``Task`` + an ``Objective`` and returns a ``SelectionResult``. See docs/rl_select/SPEC.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass
class Task:
    name: str                    # e.g. "lichens1/r0/f2"
    dataset: str                 # cache key: lichens1 | lichens2 | collagen_acetic | collagen_pepsin
    repeat: int
    fold: int
    # --- selection side (training specimens of the outer fold) ---
    Xtr: np.ndarray              # [Ntr, F] float32, preprocessed (log1p of clipped counts, z-scored w/ train stats)
    ytr: np.ndarray              # [Ntr] int64
    gtr: np.ndarray              # [Ntr] int64 specimen ids
    sub_idx: np.ndarray          # [Nsub] indices into Xtr: per-specimen capped subsample used by J evaluators
    # --- outer test side (NEVER used by selection; only evaluate.py) ---
    Xte: np.ndarray              # [Nte, F]
    yte: np.ndarray
    gte: np.ndarray
    # --- static description ---
    ex_nm: np.ndarray            # [F] excitation of each candidate pair (nm)
    em_nm: np.ndarray            # [F] emission of each candidate pair (nm)
    raw_logmean: np.ndarray      # [F] mean log1p raw intensity on training pixels (label-free brightness)
    raw_logstd: np.ndarray       # [F] std of log1p raw intensity on training pixels
    n_classes: int
    valid: np.ndarray            # [F] bool: physically admissible candidates (cached sets already apply
                                 #     first- and second-order Rayleigh rejection; extra filters AND into this)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.Xtr.shape[1])


class Objective(Protocol):
    """J(S) -> float, higher is better. Implementations cache by frozenset(S) and count evaluations.

    ``n_evals`` counts UNCACHED evaluations (the budget currency for every search method).
    ``uses_labels`` tells deployment code whether the objective may be used on a label-free target.
    """

    name: str
    uses_labels: bool
    n_evals: int

    def __call__(self, S: Sequence[int]) -> float: ...

    def reset_counter(self) -> None: ...


@dataclass
class SelectionResult:
    method: str
    task: str
    n: int                       # budget (number of pairs)
    selected: list[int]          # final set, len == n, indices into the candidate list
    objective_name: str          # objective used to guide search ("none" for filter methods)
    inner_J: float | None        # J of the returned set under the guiding objective (None if no objective)
    n_evals: int                 # uncached objective evaluations consumed
    runtime_s: float
    seed: int = 0
    trace: list[tuple[int, float]] = field(default_factory=list)  # (n_evals, best-so-far J) anytime curve
    extra: dict[str, Any] = field(default_factory=dict)
