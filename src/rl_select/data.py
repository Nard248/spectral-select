"""Dataset loading, outer folds, Task construction and inner folds (docs/rl_select/SPEC.md sections 2-3)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import paths
from .types import Task

DATASETS = paths.datasets()          # the study's dataset list (env RL_SELECT_DATASETS)
KNOWN_DATASETS = paths.KNOWN_DATASETS  # also loadable (e.g. full ``lichens2`` for comparison)
CACHE_DIR = paths.cache_dir()           # env RL_SELECT_CACHE, default results/rl_select/cache_recwl


@lru_cache(maxsize=4)
def load_dataset(key: str) -> dict:
    """Load a dataset cache as a plain dict of arrays (read-only by convention)."""
    if key not in KNOWN_DATASETS:
        raise ValueError(f"unknown dataset {key!r}; expected one of {KNOWN_DATASETS}")
    path = CACHE_DIR / f"{key}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["key"] = key
    d["n_classes"] = int(d["y"].max()) + 1
    return d


def class_covering_deal(spec_ids: np.ndarray, spec_classes: np.ndarray, k: int,
                        rng: np.random.Generator) -> list[list[int]]:
    """Seeded per-class shuffle of specimens dealt round-robin into k folds (each fold gets every class)."""
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(spec_classes):
        sids = spec_ids[spec_classes == c].copy()
        if len(sids) < k:
            raise ValueError(f"class {c} has {len(sids)} specimens < k={k}")
        rng.shuffle(sids)
        for i, s in enumerate(sids):
            folds[i % k].append(int(s))
    return folds


def specimen_folds(key_or_data: str | dict, repeat: int) -> list[list[int]]:
    """Outer folds: k = min specimens per class; per-class seeded shuffle (seed=repeat), round-robin deal."""
    d = load_dataset(key_or_data) if isinstance(key_or_data, str) else key_or_data
    sc = d["specimen_classes"]
    k = int(np.bincount(sc).min())
    folds = class_covering_deal(np.arange(len(sc)), sc, k, np.random.default_rng(int(repeat)))
    for f in folds:
        assert set(sc[f].tolist()) == set(range(d["n_classes"])), "fold misses a class"
    return folds


def n_outer_folds(key: str) -> int:
    return int(np.bincount(load_dataset(key)["specimen_classes"]).min())


def _subsample(gtr: np.ndarray, seed: int) -> np.ndarray:
    """Per training specimen random cap of max(150, 3000 // n_specimens) pixels; sorted indices."""
    sids = np.unique(gtr)
    cap = max(150, 3000 // len(sids))
    rng = np.random.default_rng(seed)
    out = []
    for s in sids:
        idx = np.flatnonzero(gtr == s)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        out.append(idx)
    return np.sort(np.concatenate(out))


def _build_task(key: str, repeat: int, fold: int) -> Task:
    d = load_dataset(key)
    folds = specimen_folds(d, repeat)
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold {fold} out of range for {key} (k={len(folds)})")
    te_mask = np.isin(d["groups"], folds[fold])
    tr, te = np.flatnonzero(~te_mask), np.flatnonzero(te_mask)
    Ltr = np.log1p(np.clip(d["X"][tr], 0, None))
    mean = Ltr.mean(0, dtype=np.float64)
    std = Ltr.std(0, dtype=np.float64)
    sd = np.maximum(std, 1e-6)
    Xtr = ((Ltr - mean.astype(np.float32)) / sd.astype(np.float32)).astype(np.float32)
    del Ltr
    Lte = np.log1p(np.clip(d["X"][te], 0, None))
    Xte = ((Lte - mean.astype(np.float32)) / sd.astype(np.float32)).astype(np.float32)
    gtr, ytr = d["groups"][tr], d["y"][tr]
    sub = _subsample(gtr, repeat * 100 + fold)
    F = Xtr.shape[1]
    return Task(
        name=f"{key}/r{repeat}/f{fold}", dataset=key, repeat=repeat, fold=fold,
        Xtr=Xtr, ytr=ytr, gtr=gtr, sub_idx=sub,
        Xte=Xte, yte=d["y"][te], gte=d["groups"][te],
        ex_nm=d["ex_nm"].astype(np.float64), em_nm=d["em_nm"].astype(np.float64),
        raw_logmean=mean, raw_logstd=std, n_classes=d["n_classes"], valid=np.ones(F, dtype=bool),
        info={"coords_sub": d["coords"][tr][sub], "specimen_classes": d["specimen_classes"],
              "n_outer_folds": len(folds), "test_specimens": folds[fold]},
    )


@lru_cache(maxsize=6)
def make_task(key: str, repeat: int, fold: int) -> Task:
    """Construct (and cache in-process) the Task for one outer fold of one dataset."""
    return _build_task(key, int(repeat), int(fold))


def all_tasks(key: str, repeat: int) -> list[Task]:
    """All outer-fold Tasks of a dataset for one repeat."""
    return [make_task(key, repeat, f) for f in range(n_outer_folds(key))]


def inner_folds(task: Task) -> list[tuple[np.ndarray, np.ndarray]]:
    """Class-covering whole-specimen inner folds over the rows of ``Xtr[sub_idx]``.

    Returns [(train_rows, val_rows)] with row indices into the sub-sampled array. k_in = min training
    specimens per class (capped at 4). A class with a single training specimen is split into two
    pseudo-specimens at the median coordinate along the longer image axis (SPEC section 3).
    """
    if "inner_folds" in task.info:
        return task.info["inner_folds"]
    g = task.gtr[task.sub_idx].copy()
    y = task.ytr[task.sub_idx]
    coords = task.info["coords_sub"]
    spec = np.unique(g)
    spec_cls = np.array([y[g == s][0] for s in spec])
    nxt = int(g.max()) + 1
    extra_ids, extra_cls = [], []
    for c in range(task.n_classes):
        ids = spec[spec_cls == c]
        if len(ids) == 0:
            raise ValueError(f"class {c} absent from training specimens of {task.name}")
        if len(ids) == 1:
            rows = np.flatnonzero(g == ids[0])
            xy = coords[rows].astype(float)
            ax = int(np.argmax(xy.max(0) - xy.min(0)))
            hi = xy[:, ax] > np.median(xy[:, ax])
            if hi.all() or (~hi).all():
                raise ValueError("cannot split single-specimen class (degenerate coordinates)")
            g[rows[hi]] = nxt
            extra_ids.append(nxt); extra_cls.append(c)
            nxt += 1
    spec = np.concatenate([spec, np.array(extra_ids, dtype=spec.dtype)]) if extra_ids else spec
    spec_cls = np.concatenate([spec_cls, np.array(extra_cls, dtype=spec_cls.dtype)]) if extra_ids else spec_cls
    k_in = min(int(np.bincount(spec_cls, minlength=task.n_classes).min()), 4)
    if k_in < 2:
        raise ValueError("fewer than 2 (pseudo-)specimens in some class")
    rng = np.random.default_rng(task.repeat * 100 + task.fold + 7919)
    folds = class_covering_deal(spec, spec_cls, k_in, rng)
    out = []
    for f in folds:
        val = np.isin(g, f)
        out.append((np.flatnonzero(~val), np.flatnonzero(val)))
    task.info["inner_folds"] = out
    return out
