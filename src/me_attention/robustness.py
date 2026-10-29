"""Phase B robustness sweep (docs/me_attention/PREREG_ROBUSTNESS_2026-10-28.md, PHASEB_IMPL_SPEC.md).

One unit = (dataset, model, repeat, fold). Reuses the tuned ``best_config`` of the matching clean record (never
re-tunes): one clean fit, then every test-time level of T1/T2/T3 evaluated on that same fitted estimator, then one
refit per level of L and C/a/C/b. Every perturbation is a pure function of raw X (or raw y) plus a seed derived
from ``(dataset, repeat, fold, family, level)`` only -- never the model, config or estimator -- so every model sees
byte-identical perturbed inputs.

    python -m me_attention.robustness --datasets lichens2_sub3 --models lr,rbfsvm,ex_tokens_v2 \\
        --repeats lichens2_sub3=5 --device cuda --workers 8 --gpu-workers 2 \\
        [--families clean,T1,T2,T3,L,C] [--only-repeat 0 --only-fold 0] [--dry-run] [--results-root ...]

Results: ``results/me_attention/robust/<dataset>/<model>.jsonl``, one line per (family, level, subset), written by
the parent process only (``harness.append_record``). Resumable: a unit re-runs only when at least one of its
expected keys is missing; the writer drops records whose key is already present.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import math
import multiprocessing as mp
import os
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .harness import RESULTS_ROOT as CLEAN_ROOT
from .harness import _free, _Log, _set_thread_env, _worker_init, append_record, is_neural, job_seed, read_records
from .harness import result_path as _clean_result_path

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "me_attention" / "robust"

PERTURB_SEED_SALT = "robust-2026-10-28"

# Clean sweep each dataset's tuned configs come from (frozen: PREREG_ROBUSTNESS_2026-10-28.md, COMPARATORS doc).
CLEAN_SWEEP: dict[str, str] = {
    "lichens2_sub3": "sub3",
    "lichens2": "final_recwl",
    "collagen_pepsin": "final_recwl",
    "lichens1": "final_recwl",
}
# datasets that need the recorded-wavelength loader path (PHASEB_IMPL_SPEC.md section 0).
RECWL_DATASETS = {"lichens2", "collagen_pepsin", "lichens1"}

TEST_FAMILIES = ("T1", "T2", "T3")
TRAIN_FAMILIES = ("L", "C")
ALL_FAMILIES = ("clean", *TEST_FAMILIES, *TRAIN_FAMILIES)

T2_SIGMAS = (0.1, 0.25, 0.5)
T3_SCALES = (0.1, 0.25, 0.5, 1.0)
L_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
C_A_LEVELS = (25, 100, 400)
C_B_LEVELS = (1, 2)


# ----------------------------------------------------------------------------- registries / hooks (monkeypatchable)
def get_models():
    from .models.registry import MODELS
    return MODELS


def load_dataset(key: str):
    if key in RECWL_DATASETS:
        os.environ["ME_ATTENTION_RECORDED_WL"] = "1"
    from . import data as D
    return D.load(key)


_DATA_CACHE: "OrderedDict[str, Any]" = OrderedDict()


def _data(key: str):
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_dataset(key)
        while len(_DATA_CACHE) > 2:
            _DATA_CACHE.popitem(last=False)
    return _DATA_CACHE[key]


def clean_sweep_for(dataset: str) -> str:
    try:
        return CLEAN_SWEEP[dataset]
    except KeyError:
        raise KeyError(f"no clean sweep mapping for dataset {dataset!r}; known: {sorted(CLEAN_SWEEP)}")


# ----------------------------------------------------------------------------- fairness: perturbation seed
def perturb_seed(dataset: str, repeat: int, fold: int, family: str, level) -> int:
    """Pure function of (dataset, repeat, fold, family, level) only -- never the model, config or estimator."""
    import hashlib
    key = f"{PERTURB_SEED_SALT}|{dataset}|{repeat}|{fold}|{family}|{level}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- T1 -- missing excitations
def impute_values(X_train: np.ndarray) -> np.ndarray:
    """Pre-registered per-feature T1 fill value: ``expm1(mean(log1p(max(X_train, 0))))``."""
    X_train = np.asarray(X_train, dtype=np.float64)
    return np.expm1(np.log1p(np.maximum(X_train, 0.0)).mean(axis=0)).astype(np.float32)


def ex_blocks(data) -> list[np.ndarray]:
    """Feature indices of each excitation, ordered by ``data.excitations``."""
    ex_nm = np.asarray(data.ex_nm, dtype=float)
    return [np.flatnonzero(ex_nm == e) for e in np.asarray(data.excitations, dtype=float)]


def t1_subsets(n_ex: int, m: int, seed: int, n_draw: int = 10) -> list[tuple[int, ...]]:
    """Excitation-index subsets of size m to remove. All C(n_ex, m) subsets if that is <= n_draw (sorted),
    else n_draw distinct subsets drawn with ``seed`` (Deviation 1, PREREG_ROBUSTNESS_2026-10-28.md)."""
    total = math.comb(n_ex, m)
    if total <= n_draw:
        return [c for c in itertools.combinations(range(n_ex), m)]
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < n_draw:
        draw = tuple(sorted(int(i) for i in rng.choice(n_ex, size=m, replace=False)))
        seen.add(draw)
    return sorted(seen)


def apply_t1(X_test: np.ndarray, blocks: list[np.ndarray], drop_ex, impute: np.ndarray) -> np.ndarray:
    """Replace every feature of every excitation in ``drop_ex`` by its training-set imputed value."""
    out = np.array(X_test, dtype=np.float32, copy=True)
    for e in drop_ex:
        idx = blocks[e]
        out[:, idx] = impute[idx]
    return out


# ----------------------------------------------------------------------------- T2 -- per-excitation gain drift
def apply_t2(X_test: np.ndarray, blocks: list[np.ndarray], groups_test: np.ndarray, sigma, seed) -> np.ndarray:
    """Multiply each (specimen, excitation) block by one ``exp(N(0, sigma^2))`` factor, identical for every
    pixel of that specimen and excitation."""
    out = np.array(X_test, dtype=np.float32, copy=True)
    if not sigma:
        return out
    groups_test = np.asarray(groups_test)
    rng = np.random.default_rng(seed)
    for s in np.unique(groups_test):
        rows = groups_test == s
        for idx in blocks:
            g = float(np.exp(rng.normal(0.0, float(sigma))))
            out[np.ix_(rows, idx)] *= g
    return out


# ----------------------------------------------------------------------------- T3 -- measurement noise
def apply_t3(X_test: np.ndarray, std_train: np.ndarray, s, seed) -> np.ndarray:
    """Add ``N(0, (s * std_train[f])^2)`` per feature in raw space, then clip at 0."""
    out = np.array(X_test, dtype=np.float32, copy=True)
    if not s:
        return out
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(out.shape).astype(np.float32) * (float(s) * np.asarray(std_train, np.float32))
    return np.maximum(out + noise, 0.0).astype(np.float32)


# ----------------------------------------------------------------------------- L -- label noise
def flip_labels(y_train: np.ndarray, n_classes: int, p, seed) -> np.ndarray:
    """Reassign ``round(p * n)`` uniformly chosen training pixels to a uniformly chosen *other* class."""
    y = np.array(y_train, dtype=np.int64, copy=True)
    n = len(y)
    k = int(round(float(p) * n))
    if k <= 0:
        return y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        others = [c for c in range(n_classes) if c != int(y[i])]
        y[i] = others[int(rng.integers(len(others)))]
    return y


# ----------------------------------------------------------------------------- C -- learning curve
def subsample_pixels(idx: np.ndarray, groups: np.ndarray, n, seed) -> np.ndarray:
    """C(a): keep ``min(n, count_s)`` pixels of each specimen ``s`` in ``idx``, drawn uniformly."""
    idx = np.asarray(idx)
    if n == "all":
        return np.sort(idx)
    g = np.asarray(groups)[idx]
    rng = np.random.default_rng(seed)
    keep = []
    for s in np.unique(g):
        pix = idx[g == s]
        take = min(int(n), len(pix))
        keep.append(pix if take >= len(pix) else rng.choice(pix, size=take, replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=idx.dtype)


def subsample_specimens(idx: np.ndarray, groups: np.ndarray, specimen_classes: np.ndarray, s, seed) -> np.ndarray:
    """C(b): keep ``min(s, avail_c)`` specimens (every one of their pixels) of each class present in ``idx``."""
    idx = np.asarray(idx)
    if s == "all":
        return np.sort(idx)
    g = np.asarray(groups)[idx]
    specimen_classes = np.asarray(specimen_classes)
    present = np.unique(g)
    rng = np.random.default_rng(seed)
    keep_sp = []
    for c in np.unique(specimen_classes[present]):
        sids = present[specimen_classes[present] == c]
        take = min(int(s), len(sids))
        keep_sp.append(sids if take >= len(sids) else rng.choice(sids, size=take, replace=False))
    if not keep_sp:
        return np.array([], dtype=idx.dtype)
    keep_sp = np.concatenate(keep_sp)
    return np.sort(idx[np.isin(g, keep_sp)])


def levels_for(family: str, data, train_idx) -> list:
    """Valid non-clean levels of a family for one fold's training indices. ``family == "C"`` returns
    ``(subset, level)`` pairs; C(b) levels that remove nothing from every class are skipped (Deviation 3)."""
    if family == "T1":
        n_ex = len(data.excitations)
        return [m for m in (1, 2, 3) if m <= n_ex - 1]
    if family == "T2":
        return list(T2_SIGMAS)
    if family == "T3":
        return list(T3_SCALES)
    if family == "L":
        return list(L_FRACTIONS)
    if family == "C":
        train_idx = np.asarray(train_idx)
        g = np.asarray(data.groups)[train_idx]
        present = np.unique(g)
        sc = np.asarray(data.specimen_classes)
        avail = {int(c): int((sc[present] == c).sum()) for c in np.unique(sc[present])}
        out: list[tuple[str, int]] = [("a", n) for n in C_A_LEVELS]
        for s in C_B_LEVELS:
            if avail and any(s < a for a in avail.values()):
                out.append(("b", s))
        return out
    raise ValueError(f"unknown family {family!r}")


# ----------------------------------------------------------------------------- config clipping
def clip_config(cfg: dict, model: str, n_train: int) -> tuple[dict, list[str]]:
    """Fix hyperparameters invalid on a shrunken training set. ``knn``'s neighbour-count hyperparameter is
    named ``k`` in this codebase (models/baselines_spectral.py): treated as the spec's ``n_neighbors``."""
    cfg = dict(cfg)
    notes: list[str] = []
    n_train = int(n_train)
    if model == "knn" and "k" in cfg:
        k = int(cfg["k"])
        new_k = max(1, min(k, n_train))
        if new_k != k:
            cfg["k"] = new_k
            notes.append(f"knn.k {k} -> {new_k} (n_train={n_train})")
    if cfg.get("max_train") is not None:
        mt = int(cfg["max_train"])
        new_mt = min(mt, n_train)
        if new_mt != mt:
            cfg["max_train"] = new_mt
            notes.append(f"max_train {mt} -> {new_mt} (n_train={n_train})")
    spec = get_models().get(model)
    if spec is not None and is_neural(spec) and "batch_size" in cfg:
        bs = int(cfg["batch_size"])
        new_bs = min(bs, max(8, n_train))
        if new_bs != bs:
            cfg["batch_size"] = new_bs
            notes.append(f"batch_size {bs} -> {new_bs} (n_train={n_train})")
    return cfg, notes


def _strip_njobs(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "n_jobs"}


# ----------------------------------------------------------------------------- metric aggregation (T1 only)
def _avg_metric_blocks(blocks: list[dict]) -> dict:
    """Average a list of ``protocol.metrics`` dicts (same fold geometry) into one block of the same shape."""
    b0 = blocks[0]
    out: dict[str, Any] = dict(n=b0["n"], n_specimens=b0["n_specimens"], labels=b0["labels"])
    for key in ("balanced_acc", "macro_f1", "pixel_acc", "specimen_avg_acc"):
        out[key] = float(np.mean([b[key] for b in blocks]))
    if all(b.get("specimen_vote_acc") is not None for b in blocks):
        out["specimen_vote_acc"] = float(np.mean([b["specimen_vote_acc"] for b in blocks]))
    else:
        out["specimen_vote_acc"] = None
    C = len(b0["labels"])
    rec = np.full((len(blocks), C), np.nan)
    for i, b in enumerate(blocks):
        for c, v in enumerate(b["per_class_recall"]):
            rec[i, c] = np.nan if v is None else v
    with np.errstate(invalid="ignore"):
        mean_rec = np.nanmean(rec, axis=0)
    out["per_class_recall"] = [None if np.isnan(v) else float(v) for v in mean_rec]
    # mean, not sum: keeps sum(confusion) == n (each subset's own confusion already sums to n)
    out["confusion"] = np.mean([np.array(b["confusion"]) for b in blocks], axis=0).tolist()
    return out


# ----------------------------------------------------------------------------- one unit
def run_unit(job: dict) -> list[dict]:
    """Run one (dataset, model, repeat, fold) unit; returns the list of its records. Never raises: a failure
    (missing/errored clean record, or any exception) becomes a single ``{"error": ...}`` record."""
    base = {k: job[k] for k in ("dataset", "model", "repeat", "fold")}
    try:
        return _run_unit(job)
    except Exception:
        return [{**base, "error": traceback.format_exc()}]


def _run_unit(job: dict) -> list[dict]:
    from .protocol import metrics, outer_folds
    ds, model, repeat, fold = job["dataset"], job["model"], job["repeat"], job["fold"]
    device, threads = job.get("device", "cpu"), job.get("threads", 1)
    families = list(job.get("families") or ALL_FAMILIES)
    clean_root = Path(job.get("clean_root", CLEAN_ROOT))
    sweep = job.get("clean_sweep") or clean_sweep_for(ds)
    base = dict(dataset=ds, model=model, repeat=repeat, fold=fold)

    clean_recs = read_records(_clean_result_path(clean_root, sweep, ds, model))
    match = [r for r in clean_recs
             if r.get("repeat") == repeat and r.get("fold") == fold and "error" not in r and not r.get("tiny")]
    if not match:
        return [{**base, "error": f"no clean record for {ds}/{model} r{repeat} f{fold} in sweep {sweep!r}"}]
    clean = match[-1]

    spec = get_models()[model]
    data = _data(ds)
    meta = data.meta()
    tr, te = outer_folds(data, repeat)[fold]
    if int(clean.get("n_train", -1)) != len(tr) or int(clean.get("n_test", -1)) != len(te):
        return [{**base, "error": "fold size mismatch between the clean record and the current outer_folds()"}]
    seed = int(clean.get("seed", job_seed(ds, model, repeat, fold)))
    cfg0 = dict(clean["best_config"])
    neural = is_neural(spec)
    fit_cfg0 = {**cfg0, "n_jobs": threads} if not neural else dict(cfg0)
    cfg0_out = _strip_njobs(cfg0)

    Xtr, ytr, gtr = data.X[tr], data.y[tr], data.groups[tr]
    Xte, yte, gte = data.X[te], data.y[te], data.groups[te]
    n_classes = data.n_classes
    n_train, n_test = int(len(tr)), int(len(te))

    def rec_base(family, level, subset, n_subsets, best_config, clip_notes, test, subset_baccs,
                 fit_s, predict_s) -> dict:
        return dict(dataset=ds, model=model, track=spec.track, family=family, level=level, subset=subset,
                    repeat=repeat, fold=fold, seed=seed, n_train=n_train, n_test=n_test, n_subsets=n_subsets,
                    best_config=best_config, clip_notes=clip_notes, test=test, subset_baccs=subset_baccs,
                    device=device, fit_s=fit_s, predict_s=predict_s, wall_s=fit_s + predict_s)

    out: list[dict] = []

    # ---- one clean fit, reused for every test-time family
    t0 = time.time()
    est = spec.build(dict(fit_cfg0), meta, seed, device)
    est.fit(Xtr, ytr, gtr)
    fit_s_clean = time.time() - t0

    t0 = time.time()
    proba = est.predict_proba(Xte)
    predict_s_clean = time.time() - t0
    if "clean" in families:
        out.append(rec_base("clean", 0, None, None, cfg0_out, [],
                            metrics(yte, proba, gte, n_classes, whole_specimen=True), None,
                            fit_s_clean, predict_s_clean))

    blocks = ex_blocks(data)
    n_ex = len(blocks)
    impute = impute_values(Xtr)
    std_train = np.std(Xtr.astype(np.float64), axis=0).astype(np.float32)

    if "T1" in families:
        for m in levels_for("T1", data, tr):
            subsets = t1_subsets(n_ex, m, perturb_seed(ds, repeat, fold, "T1", m))
            t0 = time.time()
            mblocks, baccs = [], []
            for drop in subsets:
                Xp = apply_t1(Xte, blocks, drop, impute)
                p = est.predict_proba(Xp)
                mm = metrics(yte, p, gte, n_classes, whole_specimen=True)
                mblocks.append(mm)
                baccs.append(mm["balanced_acc"])
            predict_s = time.time() - t0
            out.append(rec_base("T1", m, None, len(subsets), cfg0_out, [], _avg_metric_blocks(mblocks),
                                [float(b) for b in baccs], 0.0, predict_s))

    if "T2" in families:
        for sigma in levels_for("T2", data, tr):
            Xp = apply_t2(Xte, blocks, gte, sigma, perturb_seed(ds, repeat, fold, "T2", sigma))
            t0 = time.time()
            p = est.predict_proba(Xp)
            predict_s = time.time() - t0
            out.append(rec_base("T2", sigma, None, None, cfg0_out, [],
                                metrics(yte, p, gte, n_classes, whole_specimen=True), None, 0.0, predict_s))

    if "T3" in families:
        for s in levels_for("T3", data, tr):
            Xp = apply_t3(Xte, std_train, s, perturb_seed(ds, repeat, fold, "T3", s))
            t0 = time.time()
            p = est.predict_proba(Xp)
            predict_s = time.time() - t0
            out.append(rec_base("T3", s, None, None, cfg0_out, [],
                                metrics(yte, p, gte, n_classes, whole_specimen=True), None, 0.0, predict_s))

    del est
    _free()

    # ---- refits: L, C/a, C/b (evaluated on the clean test fold)
    if "L" in families:
        for p_frac in levels_for("L", data, tr):
            y_pert = flip_labels(ytr, n_classes, p_frac, perturb_seed(ds, repeat, fold, "L", p_frac))
            cfg, notes = clip_config(cfg0, model, n_train)
            fit_cfg = {**cfg, "n_jobs": threads} if not neural else dict(cfg)
            t0 = time.time()
            est_r = spec.build(dict(fit_cfg), meta, seed, device)
            est_r.fit(Xtr, y_pert, gtr)
            fit_s = time.time() - t0
            t0 = time.time()
            p = est_r.predict_proba(Xte)
            predict_s = time.time() - t0
            out.append(rec_base("L", p_frac, None, None, _strip_njobs(cfg), notes,
                                metrics(yte, p, gte, n_classes, whole_specimen=True), None, fit_s, predict_s))
            del est_r
            _free()

    if "C" in families:
        for subset, level in levels_for("C", data, tr):
            seed_c = perturb_seed(ds, repeat, fold, "C", f"{subset}{level}")
            if subset == "a":
                idx = subsample_pixels(tr, data.groups, level, seed_c)
            else:
                idx = subsample_specimens(tr, data.groups, data.specimen_classes, level, seed_c)
            Xp, yp, gp = data.X[idx], data.y[idx], data.groups[idx]
            cfg, notes = clip_config(cfg0, model, len(idx))
            fit_cfg = {**cfg, "n_jobs": threads} if not neural else dict(cfg)
            t0 = time.time()
            est_r = spec.build(dict(fit_cfg), meta, seed, device)
            est_r.fit(Xp, yp, gp)
            fit_s = time.time() - t0
            t0 = time.time()
            p = est_r.predict_proba(Xte)
            predict_s = time.time() - t0
            out.append(rec_base("C", level, subset, None, _strip_njobs(cfg), notes,
                                metrics(yte, p, gte, n_classes, whole_specimen=True), None, fit_s, predict_s))
            del est_r
            _free()

    return out


# ----------------------------------------------------------------------------- orchestration
def result_path(root, dataset: str, model: str) -> Path:
    return Path(root) / dataset / f"{model}.jsonl"


def _record_key(r: dict) -> tuple:
    return (r["family"], str(r["level"]), r.get("subset"))


def _expected_keys(data, train_idx, families: list[str]) -> set[tuple]:
    out: set[tuple] = set()
    if "clean" in families:
        out.add(("clean", "0", None))
    for fam in ("T1", "T2", "T3", "L"):
        if fam in families:
            out |= {(fam, str(lv), None) for lv in levels_for(fam, data, train_idx)}
    if "C" in families:
        out |= {("C", str(lv), subset) for subset, lv in levels_for("C", data, train_idx)}
    return out


def build_jobs(root, datasets, models, repeats: dict[str, int], families=None, only_repeat=None,
              only_fold=None, **common) -> tuple[list[dict], int]:
    from .protocol import n_outer_folds, outer_folds
    fam = list(families) if families else list(ALL_FAMILIES)
    jobs, skipped = [], 0
    for ds in datasets:
        data = _data(ds)
        k = n_outer_folds(data)
        for m in models:
            path = result_path(root, ds, m)
            existing = read_records(path)
            done_by_unit: dict[tuple, set] = {}
            for r in existing:
                if "error" in r:
                    continue
                done_by_unit.setdefault((r["repeat"], r["fold"]), set()).add(_record_key(r))
            for r in range(repeats.get(ds, 5)):
                if only_repeat is not None and r != only_repeat:
                    continue
                for f in range(k):
                    if only_fold is not None and f != only_fold:
                        continue
                    tr, _ = outer_folds(data, r)[f]
                    expected = _expected_keys(data, tr, fam)
                    if expected <= done_by_unit.get((r, f), set()):
                        skipped += 1
                        continue
                    jobs.append(dict(dataset=ds, model=m, repeat=r, fold=f, families=fam, **common))
    return jobs, skipped


def run_robust_sweep(datasets, models, repeats, families=None, device="cuda", workers=0, gpu_workers=1,
                     threads=2, root: Path = RESULTS_ROOT, only_repeat=None, only_fold=None) -> dict:
    root = Path(root)
    log = _Log(root / "harness.log")
    M = get_models()
    common = dict(device=device, threads=threads)
    jobs, skipped = build_jobs(root, datasets, models, repeats, families=families, only_repeat=only_repeat,
                               only_fold=only_fold, **common)
    log(f"robust units={len(jobs)} skipped(done)={skipped} workers={workers} gpu_workers={gpu_workers}")
    n_ok = n_err = 0
    t0 = time.time()

    def handle(job, recs):
        nonlocal n_ok, n_err
        path = result_path(root, job["dataset"], job["model"])
        have = {_record_key(r) for r in read_records(path)
                if r.get("repeat") == job["repeat"] and r.get("fold") == job["fold"] and "error" not in r}
        wrote = 0
        for rec in recs:
            if "error" in rec:
                n_err += 1
                append_record(path, rec)
                log(f"ERROR {job['dataset']}/{job['model']} r{job['repeat']} f{job['fold']}: "
                    f"{rec['error'].strip().splitlines()[-1]}")
                continue
            key = _record_key(rec)
            if key in have:
                continue
            append_record(path, rec)
            have.add(key)
            wrote += 1
            n_ok += 1
        el = time.time() - t0
        n = n_ok + n_err
        eta = el / n * max(0, len(jobs) - n) if n else 0.0
        log(f"[{n_ok + n_err}] {job['dataset']}/{job['model']} r{job['repeat']} f{job['fold']} wrote={wrote} "
            f"| elapsed {el/60:.1f}m ETA {eta/60:.1f}m")

    if workers <= 0:
        for j in jobs:
            handle(j, run_unit(j))
    else:
        _set_thread_env(threads)
        ctx = mp.get_context("spawn")
        n_gpu = max(1, min(gpu_workers, workers))
        n_cpu = max(1, workers - n_gpu)
        neural_jobs = [j for j in jobs if is_neural(M[j["model"]])]
        cpu_jobs = [j for j in jobs if not is_neural(M[j["model"]])]
        pools: list = []
        futs: dict = {}
        if neural_jobs:
            p = cf.ProcessPoolExecutor(n_gpu, mp_context=ctx, initializer=_worker_init, initargs=(threads,))
            pools.append(p)
            futs.update({p.submit(run_unit, j): j for j in neural_jobs})
        if cpu_jobs:
            p = cf.ProcessPoolExecutor(n_cpu, mp_context=ctx, initializer=_worker_init, initargs=(threads,))
            pools.append(p)
            futs.update({p.submit(run_unit, j): j for j in cpu_jobs})
        for fu in cf.as_completed(futs):
            j = futs[fu]
            try:
                recs = fu.result()
            except Exception:
                recs = [{**{k: j[k] for k in ("dataset", "model", "repeat", "fold")},
                        "error": traceback.format_exc()}]
            handle(j, recs)
        for p in pools:
            p.shutdown()
    log(f"done: ok={n_ok} errors={n_err} wall={(time.time() - t0)/60:.1f}m")
    return dict(ok=n_ok, errors=n_err, skipped=skipped)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    from .harness import expand_models, parse_repeats
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--repeats", default="5", help="int, or 'ds1=5,ds2=3[,default=5]'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=0, help="total worker processes (0 = run inline)")
    ap.add_argument("--gpu-workers", type=int, default=1, help="max concurrent neural (cuda) workers")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--families", default=",".join(ALL_FAMILIES))
    ap.add_argument("--only-repeat", type=int, default=None)
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    datasets = [d for d in a.datasets.split(",") if d]
    models = expand_models(a.models)
    repeats = parse_repeats(a.repeats, datasets)
    families = [f for f in a.families.split(",") if f]
    root = Path(a.results_root)
    if a.dry_run:
        jobs, skipped = build_jobs(root, datasets, models, repeats, families=families, only_repeat=a.only_repeat,
                                   only_fold=a.only_fold, device=a.device, threads=a.threads)
        print(f"{len(jobs)} units ({skipped} already done)")
        return 0
    s = run_robust_sweep(datasets, models, repeats, families, a.device, a.workers, a.gpu_workers, a.threads,
                         root, a.only_repeat, a.only_fold)
    return 0 if s["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
