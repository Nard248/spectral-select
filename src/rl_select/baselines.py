"""Baselines (docs/rl_select/SPEC.md section 6). Each returns a SelectionResult.

Wrapper searches count UNCACHED evaluations in a tracker that is local to the call (so a shared,
already-warm objective cache never gives free budget) and stop at budget B. If the budget runs out before
any complete n-set is scored, the best partial set is filled with the highest-variance valid pairs
(``extra['filled']`` records how many) and its J is computed outside the budget.
"""
from __future__ import annotations

import time
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed

from .objective import JRecon, JReconPC, correlation_matrix
from .types import SelectionResult, Task


# ----------------------------------------------------------------------------- helpers
def _pool(task: Task) -> np.ndarray:
    pool = np.flatnonzero(task.valid)
    return pool


def _check(task: Task, n: int) -> np.ndarray:
    pool = _pool(task)
    if not 1 <= n <= len(pool):
        raise ValueError(f"n={n} must be in [1, {len(pool)}]")
    return pool


def _result(method: str, task: Task, n: int, S: Sequence[int], t0: float, seed: int, obj_name: str = "none",
            inner_J: float | None = None, n_evals: int = 0, trace=None, extra=None) -> SelectionResult:
    S = [int(i) for i in S]
    if len(S) != n or len(set(S)) != n or any(not task.valid[i] for i in S):
        raise AssertionError(f"{method} produced an invalid set {S} for n={n}")
    return SelectionResult(method=method, task=task.name, n=n, selected=sorted(S), objective_name=obj_name,
                           inner_J=inner_J, n_evals=n_evals, runtime_s=time.perf_counter() - t0, seed=seed,
                           trace=list(trace or []), extra=dict(extra or {}))


class _Stop(Exception):
    pass


class _Tracker:
    """Budgeted, locally-cached view of an objective that records the best complete n-set."""

    def __init__(self, obj, n: int, budget: int):
        if obj is None or budget is None:
            raise ValueError("wrapper searches need an objective and a budget")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        self.obj, self.n, self.B = obj, n, int(budget)
        self.cache: dict[frozenset, float] = {}
        self.count = 0
        self.best_v, self.best_S = -np.inf, None
        self.partial: dict[int, tuple[float, tuple]] = {}
        self.trace: list[tuple[int, float]] = []

    def __call__(self, S) -> float:
        key = frozenset(int(i) for i in S)
        if key in self.cache:
            return self.cache[key]
        if self.count >= self.B:
            raise _Stop
        v = float(self.obj(sorted(key)))
        self.count += 1
        self.cache[key] = v
        k = len(key)
        if k == self.n:
            if v > self.best_v:
                self.best_v, self.best_S = v, tuple(sorted(key))
                self.trace.append((self.count, v))
        elif k not in self.partial or v > self.partial[k][0]:
            self.partial[k] = (v, tuple(sorted(key)))
        return v

    def finish(self, method: str, task: Task, seed: int, t0: float, extra=None) -> SelectionResult:
        extra = dict(extra or {})
        if self.best_S is not None:
            S, J = list(self.best_S), self.best_v
        else:  # budget exhausted before a complete set: complete the best partial set
            base = list(self.partial[max(self.partial)][1]) if self.partial else []
            order = [int(i) for i in np.argsort(-task.raw_logstd) if task.valid[i] and int(i) not in base]
            S = base + order[: self.n - len(base)]
            extra["filled"] = self.n - len(base)
            J = float(self.obj(S))  # outside the budget
        tr = self.trace + ([(self.count, self.best_v)] if self.best_S is not None and
                           (not self.trace or self.trace[-1][0] != self.count) else [])
        return _result(method, task, self.n, S, t0, seed, self.obj.name, J, self.count, tr, extra)


def _random_set(rng, pool, n):
    return tuple(sorted(int(i) for i in rng.choice(pool, size=n, replace=False)))


def _swap_neighbours(cur: Sequence[int], pool: np.ndarray, rng):
    inset = set(cur)
    outs = [int(c) for c in pool if int(c) not in inset]
    moves = [(i, c) for i in cur for c in outs]
    for j in rng.permutation(len(moves)):
        i, c = moves[j]
        yield tuple(sorted((set(cur) - {i}) | {c}))


def _swap_search(T: _Tracker, pool, start, rng):
    """Best-improvement 1-swap local search until no gain or the budget stops it."""
    cur, curv = tuple(start), T(start)
    while True:
        bv, bs = curv, None
        for nb in _swap_neighbours(cur, pool, rng):
            v = T(nb)
            if v > bv + 1e-12:
                bv, bs = v, nb
        if bs is None:
            return
        cur, curv = bs, bv


def _sfs_core(T: _Tracker, pool, rng, n) -> tuple:
    cur: list[int] = []
    while len(cur) < n:
        cand = [int(c) for c in rng.permutation(pool) if int(c) not in cur]
        bv, bc = -np.inf, None
        for c in cand:
            v = T(cur + [c])
            if v > bv:
                bv, bc = v, c
        cur.append(bc)
    return tuple(sorted(cur))


# ----------------------------------------------------------------------------- wrapper searches
def random_search(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    T, rng = _Tracker(obj, n, budget), np.random.default_rng(seed)
    try:
        for _ in range(200 * T.B):
            T(_random_set(rng, pool, n))
    except _Stop:
        pass
    return T.finish("random_search", task, seed, t0)


def sfs(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    T, rng = _Tracker(obj, n, budget), np.random.default_rng(seed)
    try:
        _sfs_core(T, pool, rng, n)
    except _Stop:
        pass
    return T.finish("sfs", task, seed, t0)


def sfs_swap(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    T, rng = _Tracker(obj, n, budget), np.random.default_rng(seed)
    try:
        _swap_search(T, pool, _sfs_core(T, pool, rng, n), rng)
    except _Stop:
        pass
    return T.finish("sfs_swap", task, seed, t0)


def beam_swap(task: Task, n: int, obj=None, budget=None, seed: int = 0, width: int = 5) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    T, rng = _Tracker(obj, n, budget), np.random.default_rng(seed)
    try:
        beam: list[tuple] = [()]
        for _ in range(n):
            cands = {tuple(sorted(b + (int(c),))) for b in beam for c in pool if int(c) not in b}
            cands = [cands_i for cands_i in (cands)]
            order = rng.permutation(len(cands))
            scored = [(T(cands[j]), cands[j]) for j in order]
            scored.sort(key=lambda t: -t[0])
            beam = [s for _, s in scored[:width]]
        _swap_search(T, pool, beam[0], rng)
    except _Stop:
        pass
    return T.finish("beam_swap", task, seed, t0)


def sa(task: Task, n: int, obj=None, budget=None, seed: int = 0, t_start: float = 0.02,
       t_end: float = 5e-4) -> SelectionResult:
    """Simulated annealing over 1-swaps with geometric cooling indexed by the fraction of budget used."""
    t0 = time.perf_counter(); pool = _check(task, n)
    T, rng = _Tracker(obj, n, budget), np.random.default_rng(seed)
    try:
        cur = _random_set(rng, pool, n)
        curv = T(cur)
        for _ in range(200 * T.B):
            temp = t_start * (t_end / t_start) ** min(1.0, T.count / T.B)
            i = cur[rng.integers(n)]
            while True:
                c = int(pool[rng.integers(len(pool))])
                if c not in cur:
                    break
            new = tuple(sorted((set(cur) - {i}) | {c}))
            v = T(new)
            if v >= curv or rng.random() < np.exp((v - curv) / temp):
                cur, curv = new, v
    except _Stop:
        pass
    return T.finish("sa", task, seed, t0)


def ga(task: Task, n: int, obj=None, budget=None, seed: int = 0, pop: int = 30, p_mut: float = 0.3) -> SelectionResult:
    """Steady-state GA on fixed-size sets: tournament selection, uniform crossover + repair, swap mutation."""
    t0 = time.perf_counter(); pool = _check(task, n)
    T, rng = _Tracker(obj, n, budget), np.random.default_rng(seed)
    try:
        P: list[tuple] = []
        while len(P) < min(pop, T.B):
            s = _random_set(rng, pool, n)
            if s not in P:
                P.append(s)
        fit = [T(s) for s in P]

        def tourney():
            idx = rng.choice(len(P), size=min(3, len(P)), replace=False)
            return P[max(idx, key=lambda j: fit[j])]

        for _ in range(200 * T.B):
            a, b = set(tourney()), set(tourney())
            child = a & b
            for g in a ^ b:
                if rng.random() < 0.5:
                    child.add(g)
            union = sorted(a | b)
            while len(child) > n:
                child.discard(int(rng.choice(sorted(child))))
            while len(child) < n:
                src = [g for g in union if g not in child] or [int(g) for g in pool if int(g) not in child]
                child.add(int(rng.choice(src)))
            if rng.random() < p_mut:
                child.discard(int(rng.choice(sorted(child))))
                while len(child) < n:
                    c = int(pool[rng.integers(len(pool))])
                    child.add(c)
            child = tuple(sorted(child))
            if child in P:
                continue
            v = T(child)
            w = int(np.argmin(fit))
            if v > fit[w]:
                P[w], fit[w] = child, v
    except _Stop:
        pass
    return T.finish("ga", task, seed, t0)


def _exact_chunk(obj, pool: np.ndarray, firsts: list[int], n: int):
    obj.use_cache = False
    obj.n_evals = 0
    best_v, best_S = -np.inf, None
    for p in firsts:
        for rest in combinations(pool[p + 1:].tolist(), n - 1):
            S = (int(pool[p]),) + rest
            v = obj(S)
            if v > best_v:
                best_v, best_S = v, S
    return best_v, best_S, obj.n_evals


def exact(task: Task, n: int, obj=None, budget=None, seed: int = 0, n_jobs: int = 12) -> SelectionResult:
    """Exhaustive search over all n-subsets (n <= 3), parallel over the first index. Ignores budget."""
    t0 = time.perf_counter(); pool = _check(task, n)
    if n > 3 or obj is None:
        raise ValueError("exact needs an objective and n <= 3")
    starts = list(range(len(pool) - n + 1))
    m = max(1, min(len(starts), n_jobs * 4))
    chunks = [starts[i::m] for i in range(m)]
    res = Parallel(n_jobs=n_jobs)(delayed(_exact_chunk)(obj, pool, c, n) for c in chunks)
    v, S, _ = max(res, key=lambda r: r[0])
    total = int(sum(r[2] for r in res))
    obj.n_evals += total
    return _result("exact", task, n, S, t0, seed, obj.name, float(v), total, [(total, float(v))],
                   {"n_subsets": total})


# ----------------------------------------------------------------------------- label-free
def random_(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    return _result("random", task, n, _random_set(np.random.default_rng(seed), pool, n), t0, seed)


def variance(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    S = pool[np.argsort(-task.raw_logstd[pool], kind="stable")[:n]]
    return _result("variance", task, n, S, t0, seed)


def uniform_grid(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    """Spread n pairs evenly over excitations, then emission quantiles within each excitation."""
    t0 = time.perf_counter(); pool = _check(task, n)
    excs = np.unique(task.ex_nm[pool])
    m = len(excs)
    n_ex = min(n, m)
    chosen = excs[np.unique(np.round(np.linspace(0, m - 1, n_ex)).astype(int))]
    cnt = np.full(n_ex, n // n_ex)
    rem = n % n_ex
    if rem:
        cnt[np.unique(np.round(np.linspace(0, n_ex - 1, rem)).astype(int))] += 1
    members = {e: pool[task.ex_nm[pool] == e] for e in excs}
    cap = {e: len(members[e]) for e in excs}
    want = {e: int(c) for e, c in zip(chosen, cnt)}
    spill = 0
    for e in chosen:
        if want[e] > cap[e]:
            spill += want[e] - cap[e]; want[e] = cap[e]
    while spill:
        room = [e for e in excs if want.get(e, 0) < cap[e]]
        if not room:
            raise ValueError("not enough valid pairs")
        e = room[0] if len(room) == 1 else min(room, key=lambda x: (want.get(x, 0), x))
        want[e] = want.get(e, 0) + 1; spill -= 1
    S: list[int] = []
    for e, c in want.items():
        if c == 0:
            continue
        mem = members[e][np.argsort(task.em_nm[members[e]], kind="stable")]
        pos = np.unique(np.round((np.arange(c) + 0.5) / c * len(mem) - 0.5).astype(int))
        S.extend(mem[pos].tolist())
    return _result("uniform_grid", task, n, S, t0, seed)


def spa(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    """Successive projections algorithm on z-scored Xtr[sub_idx], started from the highest-variance pair."""
    t0 = time.perf_counter(); pool = _check(task, n)
    P = task.Xtr[task.sub_idx][:, pool].astype(np.float64)
    j = int(np.argmax(task.raw_logstd[pool]))
    sel = [j]
    for _ in range(n - 1):
        v = P[:, j].copy()
        P -= np.outer(v, v @ P) / max(v @ v, 1e-12)
        norms = (P ** 2).sum(0)
        norms[sel] = -1.0
        j = int(np.argmax(norms))
        sel.append(j)
    return _result("spa", task, n, pool[sel], t0, seed)


def _css(task: Task, n: int, obj_cls, name: str, seed: int) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    J = obj_cls(task)
    allowed = np.zeros(task.n_features, dtype=bool); allowed[pool] = True
    S: list[int] = []
    for _ in range(n):
        g = J.stats(S)["gain_add"]
        g = np.where(allowed, g, -np.inf)
        g[S] = -np.inf
        S.append(int(np.argmax(g)))
    return _result(name, task, n, S, t0, seed, J.name, float(J(S)))


def css_greedy(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    return _css(task, n, JRecon, "css_greedy", seed)


def css_greedy_pc(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    return _css(task, n, JReconPC, "css_greedy_pc", seed)


def mmr_var(task: Task, n: int, obj=None, budget=None, seed: int = 0, lam: float = 0.5) -> SelectionResult:
    """MMR: normalised raw_logstd relevance minus lam * max |corr| with the selected set."""
    t0 = time.perf_counter(); pool = _check(task, n)
    sd = task.raw_logstd
    rel = (sd - sd[pool].min()) / max(sd[pool].max() - sd[pool].min(), 1e-12)
    corr = np.abs(correlation_matrix(task))
    S = [int(pool[np.argmax(rel[pool])])]
    while len(S) < n:
        score = rel - lam * corr[:, S].max(1)
        score[~task.valid] = -np.inf; score[S] = -np.inf
        S.append(int(np.argmax(score)))
    return _result("mmr_var", task, n, S, t0, seed)


# ----------------------------------------------------------------------------- label-based filters
def _class_stats(task: Task):
    X, y = task.Xtr[task.sub_idx].astype(np.float64), task.ytr[task.sub_idx]
    mu = X.mean(0)
    ssb = np.zeros(X.shape[1]); ssw = np.zeros(X.shape[1]); nc = 0
    for c in range(task.n_classes):
        Xc = X[y == c]
        ssb += len(Xc) * (Xc.mean(0) - mu) ** 2
        ssw += ((Xc - Xc.mean(0)) ** 2).sum(0)
        nc += 1
    return ssb, ssw, len(X), nc


def fisher(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    t0 = time.perf_counter(); pool = _check(task, n)
    ssb, ssw, _, _ = _class_stats(task)
    score = ssb / np.maximum(ssw, 1e-12)
    return _result("fisher", task, n, pool[np.argsort(-score[pool], kind="stable")[:n]], t0, seed)


def mrmr(task: Task, n: int, obj=None, budget=None, seed: int = 0) -> SelectionResult:
    """mRMR: ANOVA F relevance (scaled to max 1) minus mean |corr| with the selected set."""
    t0 = time.perf_counter(); pool = _check(task, n)
    ssb, ssw, N, C = _class_stats(task)
    Fst = (ssb / (C - 1)) / np.maximum(ssw / (N - C), 1e-12)
    rel = Fst / Fst[pool].max()
    corr = np.abs(correlation_matrix(task))
    S = [int(pool[np.argmax(rel[pool])])]
    while len(S) < n:
        score = rel - corr[:, S].mean(1)
        score[~task.valid] = -np.inf; score[S] = -np.inf
        S.append(int(np.argmax(score)))
    return _result("mrmr", task, n, S, t0, seed)


BASELINES: dict[str, Callable[..., SelectionResult]] = {
    "random": random_, "variance": variance, "uniform_grid": uniform_grid, "spa": spa,
    "css_greedy": css_greedy, "css_greedy_pc": css_greedy_pc, "mmr_var": mmr_var,
    "fisher": fisher, "mrmr": mrmr,
    "random_search": random_search, "sfs": sfs, "sfs_swap": sfs_swap, "beam_swap": beam_swap,
    "sa": sa, "ga": ga, "exact": exact,
}
LABEL_FREE = ("random", "variance", "uniform_grid", "spa", "css_greedy", "css_greedy_pc", "mmr_var")
FILTERS = ("fisher", "mrmr")
WRAPPERS = ("random_search", "sfs", "sfs_swap", "beam_swap", "sa", "ga")
