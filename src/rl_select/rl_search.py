"""High-level RL selection API returning ``SelectionResult``.

* ``train_policy``      : train on a pool of (Task, objective) -> SelectPolicy
* ``rl_per_instance``   : Q1, train from scratch on one task under an objective-eval budget
* ``deploy_label_free`` : Q2, no objective calls
* ``deploy_few_shot``   : Q3, objective observable, small budget
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from .env import (ADD, REMOVE, BudgetExhausted, EnvConfig, Episode, EpSpec, Tracker, VecSelectEnv)
from .features import N_GLOBAL, TaskFeatures
from .policy import SelectPolicy, action_probs, obs_to_tensors
from .ppo import PPOConfig, collect_rollout, train
from .types import Objective, SelectionResult, Task


def _sup_level(sup_features) -> str:
    if sup_features in (False, None, "none"):
        return "none"
    if sup_features is True or sup_features == "fisher":
        return "fisher"
    if sup_features == "oracle":
        return "oracle"
    raise ValueError(f"bad sup_features {sup_features!r}")


def build_tf(task: Task, obj: Objective | None, sup_features=False, oracle_dir=None) -> TaskFeatures:
    return TaskFeatures(task, obj, sup=_sup_level(sup_features), oracle_dir=oracle_dir)


def train_policy(task_pool: Sequence[tuple], mode: str = "addremove", init: str = "empty",
                 reward: str = "shaped", sup_features=False, algo: str = "ppo", seed: int = 0,
                 n_updates: int | None = 100, out_dir=None, max_seconds: float | None = None,
                 n_envs: int = 64, ns=(3, 5, 8, 13), dropout: float = 0.2, T: int | None = None,
                 oracle_dir=None, verbose: bool = False, ppo_kwargs: dict | None = None) -> SelectPolicy:
    """Train a policy on [(Task, objective), ...]. Objectives give the training reward."""
    tfs = [build_tf(t, o, sup_features, oracle_dir) for t, o in task_pool]
    pool = [(tf, o) for tf, (_, o) in zip(tfs, task_pool)]
    ecfg = EnvConfig(mode=mode, init=init, reward=reward, dropout=dropout, ns=tuple(ns), T=T)
    venv = VecSelectEnv(pool, ecfg, seed)
    meta = dict(sup=_sup_level(sup_features), mode=mode, init=init, reward=reward, algo=algo, T=T)
    torch.manual_seed(seed)
    policy = SelectPolicy(tfs[0].d, N_GLOBAL, meta=meta)
    pcfg = PPOConfig(algo=algo, n_envs=n_envs, **(ppo_kwargs or {}))
    info = train(policy, venv, pcfg, seed, n_updates=n_updates, max_seconds=max_seconds, out_dir=out_dir,
                 verbose=verbose)
    policy.train_info = info
    return policy


# ---------------------------------------------------------------------------------------------------
def rl_per_instance(task: Task, n: int, obj: Objective, mode: str = "add", budget: int = 2000, seed: int = 0,
                    sup_features=True, n_envs: int = 16, max_updates: int = 100000, oracle_dir=None,
                    stagnation: int = 50, reward: str = "shaped") -> SelectionResult:
    """Q1: train a fresh policy on this single task until ``budget`` uncached objective evals are spent.

    sup_features: False | True (Fisher rank) | "oracle" (Fisher + exact n=2 priors; its F(F-1)/2 evals are
    charged to the budget). Returns the best-so-far n-set.
    """
    t0 = time.time()
    tf = build_tf(task, obj, sup_features, oracle_dir)
    if tf.oracle_cost > budget:
        raise ValueError(f"oracle priors cost {tf.oracle_cost} evals > budget {budget}")
    tracker = Tracker(obj, n, budget, offset=tf.oracle_cost)
    ecfg = EnvConfig(mode=mode, init="empty", reward=reward, dropout=0.0, ns=(n,))
    venv = VecSelectEnv([(tf, obj)], ecfg, seed)
    torch.manual_seed(seed)
    policy = SelectPolicy(tf.d, N_GLOBAL, meta=dict(sup=tf.sup, mode=mode))
    cfg = PPOConfig(n_envs=n_envs, minibatch=128)
    # route every episode through the tracker (budget + best-so-far)
    orig_reset = venv.reset

    def reset(specs=None, B=None, trackers=None):
        specs = specs if specs is not None else venv.sample_specs(B)
        orig_reset(specs, trackers=[tracker] * len(specs))
    venv.reset = reset
    last = {"used": tracker.used, "t": 0}

    def stop_fn():
        if tracker.used == last["used"]:
            last["t"] += 1
        else:
            last["used"], last["t"] = tracker.used, 0
        return last["t"] >= stagnation

    info = train(policy, venv, cfg, seed, n_updates=max_updates, stop_fn=stop_fn,
                 progress_fn=lambda: tracker.used / budget)
    if tracker.best_set is None:
        raise RuntimeError("budget too small to complete a single n-set")
    return SelectionResult(
        method=f"rl_{mode}", task=task.name, n=n, selected=tracker.best_set, objective_name=obj.name,
        inner_J=tracker.best_J, n_evals=min(tracker.used, budget), runtime_s=time.time() - t0, seed=seed,
        trace=list(tracker.trace),
        extra=dict(updates=info["updates"], stopped=info["stopped"], sup=tf.sup, oracle_cost=tf.oracle_cost,
                   reward=reward))


# ---------------------------------------------------------------------------------------------------
def _check_policy(policy: SelectPolicy, tf: TaskFeatures) -> None:
    if policy.d != tf.d:
        raise ValueError(f"policy expects d={policy.d} features (sup={policy.meta.get('sup')}), "
                         f"task features have d={tf.d}")


def _rollouts(policy, tf, obj, n, ecfg, k, greedy, seed, trackers=None):
    venv = VecSelectEnv([(tf, obj)], ecfg, seed)
    torch.manual_seed(seed)
    venv.reset([EpSpec(0, n, 0.0)] * k, trackers=trackers)
    collect_rollout(policy, venv, greedy=greedy)
    return venv.episodes


def deploy_label_free(policy: SelectPolicy, task: Task, n: int, mode: str = "addremove", T: int | None = None,
                      init: str = "empty", n_samples: int = 1, seed: int = 0, oracle_dir=None
                      ) -> SelectionResult:
    """Q2: roll out using only label-free features; no objective is ever called. Returns the FINAL set.
    n_samples == 1 -> greedy (argmax) rollout. n_samples > 1 -> sample k stochastic rollouts and keep
    the one with highest JRecon (label-free rule)."""
    t0 = time.time()
    if policy.meta.get("sup", "none") != "none":
        raise ValueError("label-free deployment needs a policy trained without supervised features")
    tf = TaskFeatures(task, None, sup="none")
    _check_policy(policy, tf)
    ecfg = EnvConfig(mode=mode, init=init, reward="terminal", dropout=0.0, T=T, ns=(n,))
    greedy = n_samples == 1
    eps = _rollouts(policy, tf, None, n, ecfg, n_samples, greedy, seed)
    sets = [e.current_set() for e in eps]
    scores = [tf.jrecon(s) for s in sets]
    best = int(np.argmax(scores))
    name = f"rl_lf_{mode}" + ("" if init == "empty" else f"_{init}") + (f"_best{n_samples}" if n_samples > 1 else "")
    return SelectionResult(method=name, task=task.name, n=n, selected=sets[best], objective_name="JRecon",
                           inner_J=float(scores[best]), n_evals=0, runtime_s=time.time() - t0, seed=seed,
                           extra=dict(n_samples=n_samples, mode=mode, init=init,
                                      steps=int(eps[best].n_steps)))


def deploy_few_shot(policy: SelectPolicy, task: Task, n: int, obj: Objective, budget: int,
                    mode: str = "addremove", mc_lookahead: bool = False, m: int = 4, K: int = 4,
                    T: int | None = None, init: str = "empty", seed: int = 0, oracle_dir=None
                    ) -> SelectionResult:
    """Q3: objective observable with an eval budget. add: best-of-k sampling (first rollout greedy).
    addremove: with ``mc_lookahead`` the SPEC look-ahead (top-m removals x K sampled adds, accept the best
    if it improves J else stop); without, best-so-far over stochastic add/remove rollouts."""
    t0 = time.time()
    sup = policy.meta.get("sup", "none")
    tf = build_tf(task, obj, sup, oracle_dir)
    _check_policy(policy, tf)
    if tf.oracle_cost > budget:
        raise ValueError(f"oracle priors cost {tf.oracle_cost} evals > budget {budget}")
    tracker = Tracker(obj, n, budget, offset=tf.oracle_cost)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    stopped = "budget"
    extra = {}
    try:
        if mode == "addremove" and mc_lookahead:
            extra = _mc_lookahead(policy, tf, obj, tracker, n, T, init, m, K, rng)
            stopped = "converged"
        else:
            rewards = "terminal" if mode == "add" else "shaped"
            ecfg = EnvConfig(mode=mode, init=init, reward=rewards, dropout=0.0, T=T, ns=(n,))
            first = True
            while True:
                k = 1 if first else 8
                before = tracker.used
                _rollouts(policy, tf, obj, n, ecfg, k, first, int(rng.integers(1 << 30)),
                          trackers=[tracker] * k)
                if tracker.used == before and not first:
                    extra["stalled"] = True
                    stopped = "stalled"
                    break
                first = False
    except BudgetExhausted:
        pass
    if tracker.best_set is None:
        raise RuntimeError("budget too small to complete one n-set")
    return SelectionResult(
        method=f"rl_fs_{mode}" + ("_mc" if mc_lookahead else ""), task=task.name, n=n,
        selected=tracker.best_set, objective_name=obj.name, inner_J=tracker.best_J,
        n_evals=min(tracker.used, budget), runtime_s=time.time() - t0, seed=seed, trace=list(tracker.trace),
        extra=dict(stopped=stopped, m=m, K=K, sup=tf.sup, oracle_cost=tf.oracle_cost, **extra))


def _probs(policy, ep: Episode, Fmax: int) -> np.ndarray:
    """Action probabilities [F+1] for a single episode (STOP last)."""
    f, g, s, a, r, st, p = ep.observe()
    from .env import Obs
    obs = Obs(np.zeros(1, np.int64), f[None], g[None], s[None], a[None], r[None], np.array([st]), np.array([p]))
    return action_probs(policy, obs)[0]


def _mc_lookahead(policy, tf, obj, tracker, n, T, init, m, K, rng) -> dict:
    ecfg = EnvConfig(mode="addremove", init=init, reward="shaped", dropout=0.0, T=T, ns=(n,))
    ep = Episode(tf, obj, n, ecfg, rng, tracker=tracker, dropout=0.0, eval_all=True)
    while ep.phase == ADD and not ep.done:      # greedy construction
        ep.step(int(np.argmax(_probs(policy, ep, tf.F))))
    if ep.done:
        return dict(cycles=0, accepted=0)
    J = ep.J_cur
    cycles = accepted = 0
    while not ep.done:
        cycles += 1
        p = _probs(policy, ep, tf.F)[:-1]
        k = min(m, int((p > 0).sum()))
        tops = np.argsort(-p)[:k]
        best = (-np.inf, None, None)
        for a in tops:
            e2 = ep.clone()
            e2.eval_all = False
            e2.step(int(a))                       # remove (no eval)
            pa = _probs(policy, e2, tf.F)[:-1]
            pa = pa / pa.sum()
            for b in np.unique(rng.choice(len(pa), size=K, p=pa)):
                e3 = e2.clone()
                e3.eval_all = True
                e3.J_cur = J
                r, _ = e3.step(int(b))             # evaluates J(S - a + b) through the tracker
                jb = e3.J_cur
                if jb > best[0]:
                    best = (jb, int(a), int(b))
        if best[0] > J + 1e-12:
            ep.step(best[1])
            ep.step(best[2])
            J = best[0]
            accepted += 1
        else:
            break
    return dict(cycles=cycles, accepted=accepted)
