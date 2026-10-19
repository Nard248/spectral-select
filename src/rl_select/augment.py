"""Task augmentation for meta-training the transfer policy (RL iteration v3).

Builds extra *training* Task views from source datasets only (never from held-out targets):
  (a) class-subset views: pixels of a subset (>=2, < all) of classes, labels remapped, sub_idx / inner folds
      rebuilt, standardisation re-fit on the view's pixels (so the view's JRecon / features are a genuinely
      different landscape);
  (b) excitation-dropout views: 1-2 whole excitations marked invalid (valid=False); arrays and objective shared
      with the base task.
All views drop Xte (empty arrays) and cap the stored training pixels, to bound memory.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Sequence

import numpy as np

from .data import _subsample, all_tasks, load_dataset
from .objective import JRecon, JSup, _Base
from .types import Task

MAX_PX = 8000   # stored training pixels per class-subset view


def light(task: Task) -> Task:
    """Same Xtr/labels (shared arrays), but with the outer-test side dropped."""
    F = task.n_features
    return dataclasses.replace(task, Xte=np.zeros((0, F), np.float32), yte=task.yte[:0], gte=task.gte[:0])


def class_subset_view(task: Task, classes: Sequence[int], coords_all: np.ndarray, seed: int) -> Task:
    """New Task restricted to ``classes`` (sorted, remapped to 0..k-1). ``coords_all`` = coords of ALL
    training pixels of ``task`` (aligned with task.Xtr rows)."""
    classes = sorted(int(c) for c in classes)
    rng = np.random.default_rng(seed)
    in_c = np.isin(task.ytr, classes)
    specs = np.unique(task.gtr[in_c])
    cap = max(300, MAX_PX // len(specs))
    rows = []
    for s in specs:
        idx = np.flatnonzero(in_c & (task.gtr == s))
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        rows.append(idx)
    rows = np.sort(np.concatenate(rows))
    remap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([remap[int(v)] for v in task.ytr[rows]], dtype=np.int64)
    gtr = task.gtr[rows]
    L = task.Xtr[rows].astype(np.float64) * np.maximum(task.raw_logstd, 1e-6) + task.raw_logmean  # back to log1p
    mean, std = L.mean(0), L.std(0)
    Xtr = ((L - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    F = task.n_features
    sub = _subsample(gtr, seed)
    tag = "c" + "".join(str(c) for c in classes)
    return Task(
        name=f"{task.name}#{tag}", dataset=task.dataset, repeat=task.repeat, fold=task.fold,
        Xtr=Xtr, ytr=ytr, gtr=gtr, sub_idx=sub,
        Xte=np.zeros((0, F), np.float32), yte=np.zeros(0, np.int64), gte=np.zeros(0, np.int64),
        ex_nm=task.ex_nm, em_nm=task.em_nm, raw_logmean=mean, raw_logstd=std,
        n_classes=len(classes), valid=task.valid.copy(),
        info={"coords_sub": coords_all[rows][sub], "view": "classes", "classes": classes})


def excitation_dropout_view(task: Task, rng: np.random.Generator, k: int | None = None) -> Task:
    """Share arrays with ``task``; mark 1-2 whole excitations invalid. Task.info is shared (corr cache etc.)."""
    exs = np.unique(task.ex_nm)
    k = int(rng.integers(1, 3)) if k is None else k
    k = min(k, len(exs) - 3)
    drop = rng.choice(exs, size=k, replace=False)
    valid = task.valid & ~np.isin(task.ex_nm, drop)
    return dataclasses.replace(task, valid=valid, name=f"{task.name}#xd{'_'.join(str(int(x)) for x in sorted(drop))}")


class SafeJSup(JSup):
    """JSup that returns chance for degenerate sets (e.g. only zero-variance features in a class-subset view)."""

    def _base(self, S):
        try:
            return super()._base(S)
        except (np.linalg.LinAlgError, FloatingPointError):
            return 1.0 / self.C


class MixObjective:
    """J = JSup + w * JRecon on the same task (training-reward regulariser; uses labels)."""
    uses_labels = True

    def __init__(self, sup: JSup, recon: JRecon, w: float = 0.5):
        self.sup, self.recon, self.w = sup, recon, float(w)
        self.name = f"jsup+{w}jrecon"
        self.n_evals = 0
        self._cache: dict = {}
        self.empty_value = 1.0 / sup.C

    def __call__(self, S) -> float:
        key = frozenset(int(i) for i in S)
        v = self._cache.get(key)
        if v is None:
            v = float(self.sup(S) + self.w * self.recon(S))
            self._cache[key] = v
            self.n_evals += 1
        return v

    def reset_counter(self) -> None:
        self.n_evals = 0


def build_pool(sources: Sequence[str], mix: bool = False, repeat: int = 0, max_subsets: int = 10,
               n_dropout: int = 3, seed: int = 0, verbose: bool = False) -> list[tuple[Task, object]]:
    """[(Task view, objective)] for all outer-fold training tasks of ``sources`` (repeat 0) plus augmentations."""
    rng = np.random.default_rng(seed)
    pool: list[tuple[Task, object]] = []

    def mk_obj(t):
        j = SafeJSup(t)
        return MixObjective(j, JRecon(t)) if mix else j

    for ds in sources:
        d = load_dataset(ds)
        for base in all_tasks(ds, repeat):
            tr = np.flatnonzero(~np.isin(d["groups"], base.info["test_specimens"]))
            coords_all = d["coords"][tr]
            b = light(base)
            ob = mk_obj(b)
            pool.append((b, ob))
            for j in range(n_dropout):          # (b) excitation dropout on the original
                pool.append((excitation_dropout_view(b, rng), ob))
            C = base.n_classes
            subs = [s for r in range(2, C) for s in itertools.combinations(range(C), r)]
            if len(subs) > max_subsets:
                subs = [subs[i] for i in rng.choice(len(subs), size=max_subsets, replace=False)]
            for s in subs:                      # (a) class subsets
                try:
                    v = class_subset_view(b, s, coords_all, seed=int(rng.integers(1 << 30)))
                    pool.append((v, mk_obj(v)))
                except Exception as e:           # degenerate split etc.: skip this view
                    if verbose:
                        print("skip view", base.name, s, repr(e))
        if verbose:
            print(ds, "pool size", len(pool), flush=True)
    return pool
