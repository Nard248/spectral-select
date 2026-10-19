"""Add / add-remove subset-selection MDP (single episode, vectorised batch wrapper).

Actions are ints in [0, Fmax]; index Fmax is STOP (only legal in the remove phase). Masks are always
exact: invalid actions are never available. See docs/rl_select/SPEC.md section 7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np

from .features import TaskFeatures, global_features, N_GLOBAL
from .types import Objective

ADD, REMOVE = 0, 1


class BudgetExhausted(Exception):
    """Raised by a Tracker when an objective call would exceed the evaluation budget."""


class Tracker:
    """Wraps an objective: enforces an uncached-eval budget and tracks best-so-far n-set + trace."""

    def __init__(self, obj: Objective, n: int, budget: int | None = None, offset: int = 0):
        self.obj, self.n, self.budget, self.offset = obj, n, budget, offset
        self.n0 = obj.n_evals
        self.best_J = -np.inf
        self.best_set: list[int] | None = None
        self.trace: list[tuple[int, float]] = []
        self.trace_sets: list[list[int]] = []     # best set at each trace entry (for budget-prefix reads)

    @property
    def used(self) -> int:
        return self.obj.n_evals - self.n0 + self.offset

    def eval(self, S: Sequence[int]) -> float:
        j = float(self.obj(list(S)))
        if self.budget is not None and self.used > self.budget:
            raise BudgetExhausted
        if len(S) == self.n and j > self.best_J:
            self.best_J, self.best_set = j, sorted(int(s) for s in S)
            self.trace.append((self.used, j))
            self.trace_sets.append(list(self.best_set))
        return j


@dataclass
class EnvConfig:
    mode: str = "add"             # add | addremove
    init: str = "empty"           # empty | css_greedy
    reward: str = "shaped"        # shaped | terminal
    T: int | None = None          # refinement cycles (None -> n)
    tabu_L: int = 3
    dropout: float = 0.2
    repeat_penalty: float = 0.01
    ns: tuple = (3, 5, 8, 13)

    def __post_init__(self):
        if self.mode not in ("add", "addremove"):
            raise ValueError(f"bad mode {self.mode}")
        if self.init not in ("empty", "css_greedy"):
            raise ValueError(f"bad init {self.init}")
        if self.reward not in ("shaped", "terminal"):
            raise ValueError(f"bad reward {self.reward}")
        if self.init == "css_greedy" and self.mode != "addremove":
            raise ValueError("init=css_greedy requires mode=addremove")


def empty_value(obj: Objective | None, n_classes: int) -> float:
    """J(empty set): chance balanced accuracy for label-based objectives, 0 for reconstruction ones."""
    if obj is None:
        return 0.0
    if hasattr(obj, "empty_value"):
        return float(obj.empty_value)
    return 1.0 / n_classes if getattr(obj, "uses_labels", False) else 0.0


class Obs(NamedTuple):
    env_idx: np.ndarray    # [A] indices of active episodes
    cand: np.ndarray       # [A, Fmax, d]
    glob: np.ndarray       # [A, g]
    sel: np.ndarray        # [A, Fmax] bool (current set)
    add_mask: np.ndarray   # [A, Fmax] bool
    rem_mask: np.ndarray   # [A, Fmax] bool
    stop_ok: np.ndarray    # [A] bool
    phase: np.ndarray      # [A] int (0 add, 1 remove)


class Episode:
    """One selection episode on one task."""

    def __init__(self, tf: TaskFeatures, obj: Objective | None, n: int, cfg: EnvConfig,
                 rng: np.random.Generator, tracker: Tracker | None = None, dropout: float | None = None,
                 eval_all: bool | None = None):
        F = tf.F
        if n < 2:
            raise ValueError("n must be >= 2")
        self.tf, self.obj, self.n, self.cfg, self.tracker = tf, obj, n, cfg, tracker
        self.T = (n if cfg.T is None else cfg.T) if cfg.mode == "addremove" else 0
        p = cfg.dropout if dropout is None else dropout
        self.avail = tf.valid & (rng.random(F) >= p)
        if self.avail.sum() < n + cfg.tabu_L + 2:
            raise ValueError("too few available candidates after dropout")
        self.eval_all = (cfg.reward == "shaped") if eval_all is None else eval_all
        if self.eval_all and obj is None:
            self.eval_all = False
        self._eval = tracker.eval if tracker is not None else (obj if obj is not None else None)
        self.S = np.zeros(F, bool)
        self.order: list[int] = []
        self.phase = ADD
        self.constructing = True
        self.cycle = 0
        self.last_added: int | None = None
        self.tabu_until: dict[int, int] = {}
        self.visited: set = set()
        self.done = False
        self.n_steps = 0
        self.return_ = 0.0
        self.n_penalties = 0
        self.best_J, self.best_set = -np.inf, None
        self.final_J = float("nan")
        self.J0 = empty_value(obj, tf.task.n_classes)
        self.J_cur = self.J0
        if cfg.init == "css_greedy":
            for _ in range(n):
                g = tf.state(self.order).feats[:, 6].copy()      # gain_rec column
                g[self.S | ~self.avail] = -np.inf
                self._add_raw(int(np.argmax(g)))
            self.constructing = False
            self.visited.add(frozenset(self.order))
            if obj is not None:
                self.J0 = self.J_cur = self._J(self.order)
            self.phase = REMOVE if self.T > 0 else ADD
            if self.T == 0:
                self.done = True

    # -- helpers
    def _J(self, S) -> float:
        j = float(self._eval(list(S)))
        if len(S) == self.n and j > self.best_J:
            self.best_J, self.best_set = j, sorted(int(s) for s in S)
        return j

    def _add_raw(self, c: int) -> None:
        self.S[c] = True
        self.order.append(c)

    def add_mask(self) -> np.ndarray:
        m = self.avail & ~self.S
        for c, u in self.tabu_until.items():
            if self.cycle <= u:
                m[c] = False
        return m

    def remove_mask(self) -> np.ndarray:
        m = self.S.copy()
        if self.last_added is not None:
            m[self.last_added] = False
        return m

    def stop_allowed(self) -> bool:
        return (not self.done) and self.phase == REMOVE

    def current_set(self) -> list[int]:
        return sorted(self.order)

    def observe(self):
        tf = self.tf
        feats = tf.state(self.order).feats
        glob = global_features(tf, self.S, self.phase, self.n, self.cycle, self.T)
        if self.phase == ADD:
            am, rm = self.add_mask(), np.zeros(tf.F, bool)
        else:
            am, rm = np.zeros(tf.F, bool), self.remove_mask()
        return feats, glob, self.S.copy(), am, rm, self.stop_allowed(), self.phase

    def clone(self) -> "Episode":
        import copy
        new = object.__new__(Episode)
        for k, v in self.__dict__.items():
            new.__dict__[k] = (v.copy() if isinstance(v, (np.ndarray, list, dict, set)) else v)
        return new

    # -- dynamics
    def step(self, action: int) -> tuple[float, bool]:
        """Apply action (int in [0,F]; F == STOP). Returns (reward, done)."""
        if self.done:
            raise RuntimeError("step on finished episode")
        F = self.tf.F
        r = 0.0
        self.n_steps += 1
        if self.phase == ADD:
            if not (0 <= action < F) or not self.add_mask()[action]:
                raise ValueError(f"invalid ADD action {action}")
            self._add_raw(action)
            if self.constructing:
                full = len(self.order) == self.n
                if self.eval_all:
                    j = self._J(self.order)
                    r += j - self.J_cur
                    self.J_cur = j
                if full:
                    self.constructing = False
                    self.visited.add(frozenset(self.order))
                    if self.T > 0:
                        self.phase = REMOVE
                    else:
                        r += self._finish()
                        return self._ret(r)
            else:
                key = frozenset(self.order)
                if self.eval_all:
                    j = self._J(self.order)
                    r += j - self.J_cur
                    self.J_cur = j
                if key in self.visited:
                    r -= self.cfg.repeat_penalty
                    self.n_penalties += 1
                self.visited.add(key)
                self.last_added = action
                self.cycle += 1
                if self.cycle >= self.T:
                    r += self._finish()
                    return self._ret(r)
                self.phase = REMOVE
        else:
            if action == F:
                r += self._finish()
                return self._ret(r)
            if not (0 <= action < F) or not self.remove_mask()[action]:
                raise ValueError(f"invalid REMOVE action {action}")
            self.S[action] = False
            self.order.remove(action)
            self.tabu_until[action] = self.cycle + self.cfg.tabu_L
            self.phase = ADD
        return self._ret(r)

    def _finish(self) -> float:
        self.done = True
        r = 0.0
        if self._eval is not None:
            j = self._J(self.order) if len(self.order) == self.n else None
            self.final_J = j
            if self.cfg.reward == "terminal":
                r += j - self.J0
        return r

    def _ret(self, r: float):
        self.return_ += r
        return r, self.done


class EpSpec(NamedTuple):
    task_idx: int
    n: int
    dropout: float | None = None


class VecSelectEnv:
    """B episodes, each with its own task / n / dropout. ``pool`` is a list of (TaskFeatures, objective)."""

    def __init__(self, pool: Sequence[tuple], cfg: EnvConfig, seed: int = 0):
        self.pool, self.cfg = list(pool), cfg
        if not self.pool:
            raise ValueError("empty task pool")
        ds = {(tf.d) for tf, _ in self.pool}
        if len(ds) != 1:
            raise ValueError("all tasks in a pool must use the same feature set")
        self.d = ds.pop()
        self.g = N_GLOBAL
        self.Fmax = max(tf.F for tf, _ in self.pool)
        self.rng = np.random.default_rng(seed)
        self.episodes: list[Episode] = []

    @property
    def stop_action(self) -> int:
        return self.Fmax

    def sample_specs(self, B: int) -> list[EpSpec]:
        return [EpSpec(int(self.rng.integers(len(self.pool))), int(self.rng.choice(self.cfg.ns)))
                for _ in range(B)]

    def reset(self, specs: Sequence[EpSpec] | None = None, B: int | None = None, trackers=None) -> None:
        if specs is None:
            specs = self.sample_specs(B)
        self.episodes = []
        for i, sp in enumerate(specs):
            tf, obj = self.pool[sp.task_idx]
            tr = trackers[i] if trackers is not None else None
            self.episodes.append(Episode(tf, obj, sp.n, self.cfg, self.rng, tracker=tr, dropout=sp.dropout))

    def active(self) -> np.ndarray:
        return np.array([i for i, e in enumerate(self.episodes) if not e.done], dtype=np.int64)

    def observe(self) -> Obs:
        idx = self.active()
        A, Fm = len(idx), self.Fmax
        cand = np.zeros((A, Fm, self.d), np.float32)
        glob = np.zeros((A, self.g), np.float32)
        sel = np.zeros((A, Fm), bool)
        am = np.zeros((A, Fm), bool)
        rm = np.zeros((A, Fm), bool)
        stop = np.zeros(A, bool)
        ph = np.zeros(A, np.int64)
        for k, i in enumerate(idx):
            f, g, s, a, r, st, p = self.episodes[i].observe()
            F = f.shape[0]
            cand[k, :F], glob[k], sel[k, :F], am[k, :F], rm[k, :F], stop[k], ph[k] = f, g, s, a, r, st, p
        return Obs(idx, cand, glob, sel, am, rm, stop, ph)

    def step(self, env_idx: np.ndarray, actions: np.ndarray):
        rews = np.zeros(len(env_idx), np.float32)
        dones = np.zeros(len(env_idx), bool)
        for k, (i, a) in enumerate(zip(env_idx, actions)):
            ep = self.episodes[i]
            a = int(a)
            if a == self.Fmax:
                a = ep.tf.F          # STOP
            elif a >= ep.tf.F:
                raise ValueError(f"action {a} is padding for episode {i}")
            rews[k], dones[k] = ep.step(a)
        return rews, dones


class SelectEnv(VecSelectEnv):
    """Single-episode convenience wrapper."""

    def __init__(self, tf: TaskFeatures, obj, cfg: EnvConfig, seed: int = 0):
        super().__init__([(tf, obj)], cfg, seed)

    def reset_one(self, n: int, dropout: float | None = None) -> Obs:
        self.reset([EpSpec(0, n, dropout)])
        return self.observe()
