"""Nested-CV sweep runner (dataset x model x repeat x fold), resumable, JSONL results.

    python -m me_attention.harness --sweep NAME --datasets d1,d2 --models m1,m2|all-spectral|all-spatial \
        --repeats R --budget T --device mps --workers N --gpu-workers G [--max-epochs E] [--dry-run]

Per job: outer fold -> inner split -> tuning by successive halving over T random configs (default config is
candidate #0; score = inner-val balanced accuracy) -> refit best config on the full outer-train (neural: for the
best inner epoch, >= 3, no validation) -> one evaluation on the outer test fold.
Results: ``results/me_attention/<sweep>/<dataset>/<model>.jsonl`` (one line per job, written by the parent process
only, whole line + flush). Neural jobs run in at most G (mps) workers, classical ones in the remaining CPU workers.

Spatial models: patches ([n,P,P,F]) are built for a fixed, seeded per-specimen pool of at most ``--spatial-cap``
pixels (default 2000 = the largest ``samples_per_specimen``) used for tuning and the final fit, cached per
(dataset, P) in each worker; the outer *test* fold always uses all its pixels (patches built in chunks).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "me_attention"
SPATIAL_CAP = 2000
TEST_CHUNK = 2048


# ----------------------------------------------------------------------------- registries / hooks (monkeypatchable)
def get_models():
    from .models.registry import MODELS
    return MODELS


def load_dataset(key: str):
    from . import data as D
    return D.load(key)


# ----------------------------------------------------------------------------- helpers
def job_seed(dataset: str, model: str, repeat: int, fold: int) -> int:
    h = hashlib.md5(f"{dataset}|{model}|{repeat}|{fold}".encode()).hexdigest()
    return int(h[:8], 16)


def is_neural(spec) -> bool:
    return "epochs" in spec.default or "samples_per_specimen" in spec.space


def sample_configs(spec, T: int, seed: int) -> list[dict[str, Any]]:
    """Candidate #0 = default; then T-1 distinct random draws from the space (draws overlay the default)."""
    rng = np.random.default_rng(seed)
    out, seen = [dict(spec.default)], {json.dumps(spec.default, sort_keys=True, default=str)}
    tries = 0
    while len(out) < T and tries < 50 * T:
        tries += 1
        cfg = {**spec.default, **{k: v[int(rng.integers(len(v)))] for k, v in spec.space.items()}}
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def rung_epochs(E: int) -> list[int]:
    return sorted({max(1, math.ceil(E / 5)), max(1, math.ceil(E / 2)), int(E)})


def _compact(cfg: dict, spec) -> dict:
    return {k: cfg[k] for k in spec.space if k in cfg}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _cap_per_specimen(idx: np.ndarray, groups: np.ndarray, cap: int, rng) -> np.ndarray:
    keep = []
    for s in np.unique(groups[idx]):
        m = idx[groups[idx] == s]
        keep.append(m if len(m) <= cap else rng.choice(m, cap, replace=False))
    return np.sort(np.concatenate(keep))


class SpatialPool:
    """Fixed seeded per-specimen pixel pool with its patches [n,P,P,F]."""

    def __init__(self, data, P: int, cap: int):
        from .protocol import make_patches
        self.P = P
        self.idx = _cap_per_specimen(np.arange(len(data.y)), data.groups, cap, np.random.default_rng(12345))
        self.patches = make_patches(data, self.idx, P)
        self.pos = np.full(len(data.y), -1, np.int64)
        self.pos[self.idx] = np.arange(len(self.idx))

    def rows(self, pixel_idx: np.ndarray) -> np.ndarray:
        r = self.pos[np.asarray(pixel_idx)]
        return r[r >= 0]


_POOLS: "OrderedDict[tuple, SpatialPool]" = OrderedDict()
_DATA_CACHE: "OrderedDict[str, Any]" = OrderedDict()


def _data(key: str):
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_dataset(key)
        while len(_DATA_CACHE) > 2:
            _DATA_CACHE.popitem(last=False)
    return _DATA_CACHE[key]


def _pool(data, P: int, cap: int) -> SpatialPool:
    key = (data.name, P, cap)
    if key not in _POOLS:
        _POOLS[key] = SpatialPool(data, P, cap)
        while len(_POOLS) > 2:
            _POOLS.popitem(last=False)
            gc.collect()
    _POOLS.move_to_end(key)
    return _POOLS[key]


# ----------------------------------------------------------------------------- one job
class _Ctx:
    """Data access for one job: spectral rows or spatial patches for any (cfg, pixel-index set)."""

    def __init__(self, data, spatial: bool, cap: int):
        self.data, self.spatial, self.cap = data, spatial, cap

    def get(self, cfg, idx):
        d = self.data
        if not self.spatial:
            return d.X[idx], d.y[idx], d.groups[idx]
        pool = _pool(d, int(cfg.get("P", 5)), self.cap)
        r = pool.rows(idx)
        px = pool.idx[r]
        return pool.patches[r], d.y[px], d.groups[px]

    def predict(self, est, cfg, idx):
        d = self.data
        if not self.spatial:
            return est.predict_proba(d.X[idx])
        from .protocol import make_patches
        P = int(cfg.get("P", 5))
        out = [est.predict_proba(make_patches(d, idx[s:s + TEST_CHUNK], P)) for s in range(0, len(idx), TEST_CHUNK)]
        return np.concatenate(out)


def _fit_eval(spec, cfg, meta, seed, device, ctx, itr, iva):
    """Fit on itr (validation iva for neural early stopping) and return (est, val_score, fit_seconds, epochs_run)."""
    from .train import balanced_accuracy
    Xt, yt, gt = ctx.get(cfg, itr)
    Xv, yv, _ = ctx.get(cfg, iva)
    t0 = time.time()
    est = spec.build(dict(cfg), meta, seed, device)
    est.fit(Xt, yt, gt, Xv, yv)
    dt = time.time() - t0
    hist = getattr(est, "history_", None)
    if hist:
        score = max(h.get("val_bacc", -1.0) for h in hist)
    else:
        score = balanced_accuracy(yv, est.predict_proba(Xv).argmax(1), meta["n_classes"])
    return est, float(score), dt, (len(hist) if hist else 0)


def _free():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def run_job(job: dict) -> dict:
    """Run one (dataset, model, repeat, fold) job; returns the result record (never raises: errors become records)."""
    base = {k: job[k] for k in ("dataset", "model", "repeat", "fold")}
    t_start = time.time()
    try:
        return _run_job(job)
    except Exception:
        return {**base, "error": traceback.format_exc(), "wall_s": time.time() - t_start}


def _run_job(job: dict) -> dict:
    from .protocol import inner_split, metrics, outer_folds
    spec = get_models()[job["model"]]
    ds, repeat, fold, T, E = job["dataset"], job["repeat"], job["fold"], job["budget"], job["max_epochs"]
    device, tiny = job["device"], job.get("tiny", False)
    cap = 100 if tiny else job.get("spatial_cap", SPATIAL_CAP)
    data = _data(ds)
    meta = data.meta()
    seed = job_seed(ds, spec.name, repeat, fold)
    rng = np.random.default_rng(seed)
    tr, te = outer_folds(data, repeat)[fold]
    itr, iva = inner_split(data, tr, seed)
    if tiny:
        itr, iva, te = (_cap_per_specimen(a, data.groups, 100, rng) for a in (itr, iva, te))
    spatial = spec.track == "spatial"
    ctx = _Ctx(data, spatial, cap)
    neural = is_neural(spec)
    cands = sample_configs(spec, T, seed)
    if not neural:
        for c in cands:
            c["n_jobs"] = job.get("threads", 2)

    trials: list[dict] = []
    epochs_run, fit_time = 0, 0.0
    t0 = time.time()
    # ---- tuning
    def trial(cfg, tag, **extra):
        nonlocal epochs_run, fit_time
        try:
            est, sc, dt, ne = _fit_eval(spec, cfg, meta, seed, device, ctx, itr, iva)
            rec = dict(i=tag, cfg=_compact(cfg, spec), score=sc, **extra)
            if neural:
                rec["best_epoch"] = int(est.best_epoch_)
            epochs_run += ne
            fit_time += dt
            del est
        except Exception as e:  # a bad config must not kill the job
            rec = dict(i=tag, cfg=_compact(cfg, spec), score=None, error=f"{type(e).__name__}: {e}", **extra)
            rec["_tb"] = traceback.format_exc()
        _free()
        return rec

    if neural:
        rungs = rung_epochs(E)
        alive = list(range(len(cands)))
        for ri, ep in enumerate(rungs):
            last = ri == len(rungs) - 1
            if len(alive) == 1 and not last:
                continue  # a single survivor goes straight to the final rung
            res = []
            for i in alive:
                cfg = {**cands[i], "epochs": ep, "warmup_epochs": min(int(cands[i].get("warmup_epochs", 3)), ep // 4)}
                r = trial(cfg, i, rung=ep)
                trials.append({k: v for k, v in r.items() if k != "_tb"})
                res.append((r["score"] if r["score"] is not None else -1.0, i, r))
            res.sort(key=lambda t: -t[0])
            if last:
                top = res[0]
            else:
                alive = [i for _, i, _ in res[:max(1, math.ceil(len(res) / 2))]]
        best_score, best_i, best_r = top
        if best_r["score"] is None:
            raise RuntimeError("all configs failed: " + str(best_r.get("_tb")))
        best_epoch = max(3, int(best_r["best_epoch"]))
    else:
        res = []
        for i, c in enumerate(cands):
            r = trial(c, i)
            trials.append({k: v for k, v in r.items() if k != "_tb"})
            res.append((r["score"] if r["score"] is not None else -1.0, i, r))
        res.sort(key=lambda t: -t[0])
        best_score, best_i, best_r = res[0]
        if best_r["score"] is None:
            raise RuntimeError("all configs failed: " + str(best_r.get("_tb")))
        best_epoch = None
    t_tune = time.time() - t0

    # ---- final fit on full outer-train, single test evaluation
    best_cfg = dict(cands[best_i])
    if neural:
        best_cfg.update(epochs=best_epoch, warmup_epochs=min(int(best_cfg.get("warmup_epochs", 3)), best_epoch // 4))
    Xa, ya, ga = ctx.get(best_cfg, tr if not tiny else itr)
    t0 = time.time()
    est = spec.build(dict(best_cfg), meta, seed, device)
    est.fit(Xa, ya, ga)
    t_final = time.time() - t0
    if neural:
        epochs_run += len(est.history_)
        fit_time += t_final
    del Xa
    t0 = time.time()
    proba = ctx.predict(est, best_cfg, te)
    t_pred = time.time() - t0
    m = metrics(data.y[te], proba, data.groups[te], data.n_classes, whole_specimen=True)
    pred = proba.argmax(1)
    per_spec = {int(s): float((pred[data.groups[te] == s] == data.y[te][data.groups[te] == s]).mean())
                for s in np.unique(data.groups[te])}
    n_params = None
    if getattr(est, "module", None) is not None:
        n_params = int(sum(p.numel() for p in est.module.parameters()))
    cfg_out = {k: v for k, v in best_cfg.items() if k != "n_jobs"}
    rec = dict(
        dataset=ds, model=spec.name, track=spec.track, family=spec.family, repeat=repeat, fold=fold,
        test_specimens=[int(s) for s in np.unique(data.groups[te])], n_train=int(len(tr)), n_test=int(len(te)),
        budget=T, max_epochs=E if neural else None, best_config=cfg_out, best_epoch=best_epoch,
        inner_score=float(best_score), trials=trials, test=m, per_specimen_acc=per_spec, n_params=n_params,
        epochs_run=epochs_run, epoch_time_s=(fit_time / epochs_run) if neural and epochs_run else None,
        t_tune_s=t_tune, t_final_s=t_final, t_predict_s=t_pred, tiny=bool(tiny), device=device, seed=seed)
    del est
    _free()
    return _jsonable(rec)


# ----------------------------------------------------------------------------- sweep orchestration
def result_path(root: Path, sweep: str, dataset: str, model: str) -> Path:
    return Path(root) / sweep / dataset / f"{model}.jsonl"


def read_records(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn trailing line from a killed run: that job is simply redone
    return out


def done_keys(root: Path, sweep: str, dataset: str, model: str) -> set[tuple[int, int]]:
    return {(r["repeat"], r["fold"]) for r in read_records(result_path(root, sweep, dataset, model))
            if "error" not in r and not r.get("tiny")}


def append_record(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_jsonable(rec)) + "\n")
        f.flush()
        os.fsync(f.fileno())


def expand_models(spec_str: str) -> list[str]:
    M = get_models()
    names: list[str] = []
    for tok in [t.strip() for t in spec_str.split(",") if t.strip()]:
        if tok == "all-spectral":
            names += [n for n, s in M.items() if s.track == "spectral"]
        elif tok == "all-spatial":
            names += [n for n, s in M.items() if s.track == "spatial"]
        elif tok == "all":
            names += list(M)
        elif tok in M:
            names.append(tok)
        else:
            raise KeyError(f"unknown model {tok!r}; known: {sorted(M)}")
    return list(dict.fromkeys(names))


def parse_repeats(s: str, datasets: list[str]) -> dict[str, int]:
    """'5' or 'collagen_pepsin=5,lichens1=3' (unlisted datasets get the 'default=' value or 5)."""
    if "=" not in s:
        return {d: int(s) for d in datasets}
    kv = dict(p.split("=") for p in s.split(","))
    dflt = int(kv.get("default", 5))
    return {d: int(kv.get(d, dflt)) for d in datasets}


def model_cost(spec) -> tuple:
    """Rough cost rank for ordering: classical < neural; within neural by parameter-ish keys. Cheap first."""
    neural = is_neural(spec)
    return (int(neural), 0)


def order_jobs(jobs: list[dict], datasets: list[str], models: list[str], mode: str = "none") -> list[dict]:
    """'spectral-first': track (spectral < spatial), dataset order as given, model cost (classical < neural, then the
    order of --models); stable within (repeat, fold)."""
    if mode == "none":
        return jobs
    if mode != "spectral-first":
        raise ValueError(mode)
    M = get_models()
    di = {d: i for i, d in enumerate(datasets)}
    mi = {m: i for i, m in enumerate(models)}
    key = lambda j: (M[j["model"]].track != "spectral", di[j["dataset"]], model_cost(M[j["model"]]),
                     mi[j["model"]], j["repeat"], j["fold"])
    return sorted(jobs, key=key)


def build_jobs(root, sweep, datasets, models, repeats: dict[str, int], **common) -> tuple[list[dict], int]:
    from .protocol import n_outer_folds
    jobs, skipped = [], 0
    for ds in datasets:
        k = n_outer_folds(_data(ds))
        for m in models:
            done = done_keys(root, sweep, ds, m)
            for r in range(repeats[ds]):
                for f in range(k):
                    if (r, f) in done:
                        skipped += 1
                    else:
                        jobs.append(dict(dataset=ds, model=m, repeat=r, fold=f, **common))
    return jobs, skipped


def _worker_init(threads: int):
    import warnings
    # macOS Accelerate BLAS emits spurious 'matmul' RuntimeWarnings from sklearn linear models; keep logs readable
    warnings.filterwarnings("ignore", message=".*encountered in matmul")
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def _set_thread_env(threads: int):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(threads)


class _Log:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def run_sweep(sweep, datasets, models, repeats, budget, max_epochs, device="mps", workers=0, gpu_workers=1,
              threads=2, spatial_cap=SPATIAL_CAP, root: Path = RESULTS_ROOT, order: str = "none") -> dict:
    root = Path(root)
    log = _Log(root / sweep / "harness.log")
    M = get_models()
    common = dict(budget=budget, max_epochs=max_epochs, device=device, threads=threads, spatial_cap=spatial_cap)
    jobs, skipped = build_jobs(root, sweep, datasets, models, repeats, **common)
    jobs = order_jobs(jobs, datasets, models, order)
    log(f"sweep={sweep} jobs={len(jobs)} skipped(done)={skipped} workers={workers} gpu_workers={gpu_workers}")
    n_ok = n_err = 0
    t0 = time.time()

    def handle(job, rec):
        nonlocal n_ok, n_err
        if "error" in rec:
            n_err += 1
            rec = {**{k: job[k] for k in ("dataset", "model", "repeat", "fold")}, **rec}
            log(f"ERROR {job['dataset']}/{job['model']} r{job['repeat']} f{job['fold']}: "
                f"{rec['error'].strip().splitlines()[-1]}")
        else:
            n_ok += 1
        append_record(result_path(root, sweep, job["dataset"], job["model"]), rec)
        n = n_ok + n_err
        el = time.time() - t0
        eta = el / n * (len(jobs) - n)
        t = rec.get("test", {})
        log(f"[{n}/{len(jobs)}] {job['dataset']}/{job['model']} r{job['repeat']} f{job['fold']} "
            f"bacc={t.get('balanced_acc', float('nan')):.3f} "
            f"tune={rec.get('t_tune_s', 0):.0f}s final={rec.get('t_final_s', 0):.0f}s "
            f"| elapsed {el/60:.1f}m ETA {eta/60:.1f}m")

    if workers <= 0:
        for j in jobs:
            handle(j, run_job(j))
    else:
        _set_thread_env(threads)
        ctx = mp.get_context("spawn")
        n_gpu = max(1, min(gpu_workers, workers))
        n_cpu = max(1, workers - n_gpu)
        neural_jobs = [j for j in jobs if is_neural(M[j["model"]])]
        cpu_jobs = [j for j in jobs if not is_neural(M[j["model"]])]
        pools = []
        futs: dict = {}
        if neural_jobs:
            p = cf.ProcessPoolExecutor(n_gpu, mp_context=ctx, initializer=_worker_init, initargs=(threads,))
            pools.append(p)
            futs.update({p.submit(run_job, j): j for j in neural_jobs})
        if cpu_jobs:
            p = cf.ProcessPoolExecutor(n_cpu, mp_context=ctx, initializer=_worker_init, initargs=(threads,))
            pools.append(p)
            futs.update({p.submit(run_job, j): j for j in cpu_jobs})
        for fu in cf.as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
            except Exception:
                rec = {"error": traceback.format_exc()}
            handle(j, rec)
        for p in pools:
            p.shutdown()
    log(f"done: ok={n_ok} errors={n_err} wall={(time.time() - t0)/60:.1f}m")
    return dict(ok=n_ok, errors=n_err, skipped=skipped)


# ----------------------------------------------------------------------------- dry run
def dry_run(datasets, models, repeats, budget, max_epochs, device, threads, spatial_cap, root, sweep) -> bool:
    M = get_models()
    from .protocol import n_outer_folds
    jobs, skipped = build_jobs(root, sweep, datasets, models, repeats)
    print(f"{len(jobs)} jobs ({skipped} already done)")
    per_model: dict[str, int] = {}
    for j in jobs:
        per_model[j["model"]] = per_model.get(j["model"], 0) + 1
    rungs = rung_epochs(max_epochs)
    for m, n in per_model.items():
        if is_neural(M[m]):
            alive, ep_tot = budget, 0
            for ri, ep in enumerate(rungs):
                last = ri == len(rungs) - 1
                if alive == 1 and not last:
                    continue
                ep_tot += alive * ep
                alive = max(1, math.ceil(alive / 2))
            print(f"  {m:22s} {n:4d} jobs  neural  ~{ep_tot} tuning epochs + <= {max_epochs} final per job")
        else:
            print(f"  {m:22s} {n:4d} jobs  classical  {budget} trials per job")
    print(f"  datasets/folds: " + ", ".join(f"{d}: R={repeats[d]} x k={n_outer_folds(_data(d))}" for d in datasets))
    print("wiring check (tiny subsample, 1 trial, 1 epoch; first dataset, repeat 0, fold 0):")
    ok = True
    for m in models:
        job = dict(dataset=datasets[0], model=m, repeat=0, fold=0, budget=1, max_epochs=1, device=device,
                   threads=threads, spatial_cap=spatial_cap, tiny=True)
        t0 = time.time()
        rec = run_job(job)
        if "error" in rec:
            ok = False
            print(f"  FAIL {m}: {rec['error'].strip().splitlines()[-1]}")
        else:
            print(f"  ok   {m:22s} {time.time() - t0:5.1f}s  bacc={rec['test']['balanced_acc']:.3f}")
    return ok


# ----------------------------------------------------------------------------- summary
def summarize(sweep: str, root: Path = RESULTS_ROOT, print_table: bool = True) -> list[dict]:
    rows = []
    base = Path(root) / sweep
    for ds_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(ds_dir.glob("*.jsonl")):
            recs = [r for r in read_records(f) if "error" not in r and not r.get("tiny")]
            n_err = sum(1 for r in read_records(f) if "error" in r)
            if not recs:
                rows.append(dict(dataset=ds_dir.name, model=f.stem, n=0, n_err=n_err))
                continue
            def ms(key):
                v = np.array([r["test"][key] for r in recs], float)
                return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rows.append(dict(dataset=ds_dir.name, model=f.stem, n=len(recs), n_err=n_err,
                             bacc=ms("balanced_acc"), f1=ms("macro_f1"), vote=ms("specimen_vote_acc")))
    if print_table:
        print(f"{'dataset':18s} {'model':22s} {'bal.acc':>13s} {'macro-F1':>13s} {'spec.vote':>13s} {'n':>4s} {'err':>4s}")
        for r in rows:
            if r["n"] == 0:
                print(f"{r['dataset']:18s} {r['model']:22s} {'-':>13s} {'-':>13s} {'-':>13s} {0:4d} {r['n_err']:4d}")
                continue
            c = lambda t: f"{t[0]:.3f}+-{t[1]:.3f}"
            print(f"{r['dataset']:18s} {r['model']:22s} {c(r['bacc']):>13s} {c(r['f1']):>13s} {c(r['vote']):>13s} "
                  f"{r['n']:4d} {r['n_err']:4d}")
    return rows


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--repeats", default="5", help="int, or 'ds1=5,ds2=3[,default=5]'")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=0, help="total worker processes (0 = run inline)")
    ap.add_argument("--gpu-workers", type=int, default=1, help="max concurrent neural (mps) workers")
    ap.add_argument("--threads", type=int, default=2, help="torch/BLAS threads per worker")
    ap.add_argument("--spatial-cap", type=int, default=SPATIAL_CAP)
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    ap.add_argument("--order", default="none", choices=["none", "spectral-first"],
                    help="job order: spectral-first = all spectral jobs (dataset order as given, cheap models first), then spatial")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.results_root)
    if a.summarize:
        summarize(a.sweep, root)
        return 0
    datasets = [d for d in a.datasets.split(",") if d]
    models = expand_models(a.models)
    repeats = parse_repeats(a.repeats, datasets)
    if a.dry_run:
        return 0 if dry_run(datasets, models, repeats, a.budget, a.max_epochs, a.device, a.threads,
                            a.spatial_cap, root, a.sweep) else 1
    s = run_sweep(a.sweep, datasets, models, repeats, a.budget, a.max_epochs, a.device, a.workers, a.gpu_workers,
                  a.threads, a.spatial_cap, root, a.order)
    summarize(a.sweep, root)
    return 0 if s["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
