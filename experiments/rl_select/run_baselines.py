"""Baseline runs for the rl_select study (docs/rl_select/SPEC.md sections 6, 8, 9). Resumable.

Settings: S1 (search quality, B=5000, repeat 0), S2 (label-free reference, repeats 0-2), S3 (small budgets,
repeat 0). One JSON line per (setting, dataset, repeat, fold, n, method, seed, budget) written to
results/rl_select/<setting>/<dataset>.jsonl.  Work unit = (dataset, repeat, fold, n): one process builds the
task once, shares an outer-evaluation cache by frozenset(S) across settings, and returns rows to the parent,
which is the only writer.  ``--phase exact`` runs exhaustive n=2,3 search (joblib inside) after the pool.

  PYTHONPATH=src python experiments/rl_select/run_baselines.py --phase main --jobs 6
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rl_select import paths  # noqa: E402

ROOT = paths.ROOT
RES = paths.results_root()   # env RL_SELECT_RESULTS
NS = (3, 5, 8, 13)
B1 = 5000
B3 = (25, 50, 100, 200, 400)
LABEL_FREE = ("variance", "uniform_grid", "spa", "css_greedy", "css_greedy_pc", "mmr_var")
CFG = {"s3_seeds": 3, "ga_seeds": 3, "s2_rf": True}


def key_of(r: dict) -> tuple:
    return (r["setting"], r["dataset"], r["repeat"], r["fold"], r["n"], r["method"], r["seed"], r["budget"])


def load_done() -> set:
    done = set()
    for s in ("S1", "S2", "S3"):
        for f in (RES / s).glob("*.jsonl"):
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(key_of(json.loads(line)))
                    except json.JSONDecodeError:
                        pass  # truncated last line of a killed run: will be recomputed
    return done


def specs_for(repeat: int, n: int) -> list[dict]:
    """All (setting, method, seed, budget) jobs of one unit."""
    out = []
    add = lambda s, m, sd=0, b=0: out.append(dict(setting=s, method=m, seed=sd, budget=b))
    if repeat == 0:
        for m in ("random_search", "sa", "ga"):
            for sd in range(CFG["ga_seeds"] if m == "ga" else 3):
                add("S1", m, sd, B1)
        for m in ("sfs", "sfs_swap", "beam_swap"):
            add("S1", m, 0, B1)
        for m in ("fisher", "mrmr", "random") + LABEL_FREE:
            add("S1", m, 0, 0)
        for b in B3:
            for m in ("random_search", "sa", "ga", "sfs_swap"):
                for sd in range(CFG["s3_seeds"]):
                    add("S3", m, sd, b)
    for sd in range(10):
        add("S2", "random", sd, 0)
    for m in LABEL_FREE + ("fisher", "mrmr"):
        add("S2", m, 0, 0)
    add("S2", "sfs_swap", 0, B1)
    return out


def unit(args: tuple) -> list[dict]:
    ds, repeat, fold, n, done, cfg = args
    warnings.filterwarnings("ignore")
    CFG.update(cfg)
    from rl_select import baselines as B
    from rl_select.data import make_task
    from rl_select.evaluate import all_pairs_reference, evaluate, set_metrics
    from rl_select.objective import JSup

    todo = [s for s in specs_for(repeat, n)
            if (s["setting"], ds, repeat, fold, n, s["method"], s["seed"], s["budget"]) not in done]
    want_ap = n == NS[0] and ("S2", ds, repeat, fold, 0, "all_pairs", 0, 0) not in done
    if not todo and not want_ap:
        return []
    task = make_task(ds, repeat, fold)
    obj = JSup(task)
    sel_cache: dict = {}
    ev_cache: dict = {}
    rows = []

    def clfs(setting: str) -> tuple:
        rf = (setting == "S2" and CFG["s2_rf"]) or (setting == "S1" and n in (5, 13))
        return ("knn", "lda", "svm") + (("rf",) if rf else ())

    def run_eval(S, cl):
        k = frozenset(S)
        have = ev_cache.setdefault(k, {})
        miss = [c for c in cl if c not in have]
        if miss:
            have.update(evaluate(task, sorted(k), miss, seed=0))
        return {c: have[c] for c in cl}

    for s in todo:
        m, sd, b = s["method"], s["seed"], s["budget"]
        sk = (m, sd, b)
        if sk not in sel_cache:
            fn = B.BASELINES[m]
            sel_cache[sk] = fn(task, n, obj=obj if m in B.WRAPPERS or m == "exact" else None,
                               budget=b or None, seed=sd)
        res = sel_cache[sk]
        row = asdict(res)
        row["trace"] = [list(t) for t in res.trace]
        row.update(setting=s["setting"], dataset=ds, repeat=repeat, fold=fold, budget=b,
                   eval=run_eval(res.selected, clfs(s["setting"])), set_metrics=set_metrics(task, res.selected))
        rows.append(row)
    if want_ap:
        cl = ("knn", "lda", "svm") + (("rf",) if CFG["s2_rf"] else ())
        t0 = time.perf_counter()
        rows.append(dict(method="all_pairs", task=task.name, n=int(task.valid.sum()), selected=[], objective_name="none",
                         inner_J=None, n_evals=0, runtime_s=time.perf_counter() - t0, seed=0, trace=[], extra={},
                         setting="S2", dataset=ds, repeat=repeat, fold=fold, budget=0,
                         eval=all_pairs_reference(task, cl, seed=0), set_metrics={}))
        rows[-1]["n"] = 0  # key n=0 marks the per-task reference (n_pairs recorded in extra)
        rows[-1]["extra"] = {"n_pairs": int(task.valid.sum())}
    return rows


def write_rows(rows: list[dict]) -> None:
    for r in rows:
        d = RES / r["setting"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{r['dataset']}.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")


def run_main(jobs: int, datasets, repeats) -> None:
    from rl_select.data import n_outer_folds
    done = load_done()
    units = []
    for ds in datasets:
        for r in repeats:
            for f in range(n_outer_folds(ds)):
                for n in NS:
                    units.append((ds, r, f, n, done, dict(CFG)))
    units.sort(key=lambda u: (u[0] != "lichens1", u[0], u[1], u[2], u[3]))  # heavy dataset first
    print(f"{len(units)} units, {len(done)} rows already done", flush=True)
    t0 = time.time(); nrows = 0
    with ProcessPoolExecutor(jobs) as ex:
        futs = {ex.submit(unit, u): u[:4] for u in units}
        for i, fu in enumerate(as_completed(futs), 1):
            rows = fu.result()
            write_rows(rows); nrows += len(rows)
            print(f"[{i}/{len(units)}] {futs[fu]} +{len(rows)} rows  total {nrows}  {time.time() - t0:.0f}s", flush=True)


def run_exact(jobs: int, datasets) -> None:
    warnings.filterwarnings("ignore")
    from rl_select import baselines as B
    from rl_select.data import make_task, n_outer_folds
    from rl_select.evaluate import evaluate, set_metrics
    from rl_select.objective import JSup
    done = load_done()
    for ds in datasets:
        for f in range(n_outer_folds(ds)):
            for n in (2, 3):
                if ("S1", ds, 0, f, n, "exact", 0, 0) in done:
                    continue
                task = make_task(ds, 0, f)
                obj = JSup(task)
                res = B.exact(task, n, obj=obj, n_jobs=jobs)
                row = asdict(res)
                row["trace"] = [list(t) for t in res.trace]
                row.update(setting="S1", dataset=ds, repeat=0, fold=f, budget=0,
                           eval=evaluate(task, res.selected, ("knn", "lda", "svm"), seed=0),
                           set_metrics=set_metrics(task, res.selected))
                write_rows([row])
                print(f"exact {ds} f{f} n={n} J={res.inner_J:.4f} {res.runtime_s:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("main", "exact"), default="main")
    ap.add_argument("--jobs", type=int, default=paths.workers())
    ap.add_argument("--datasets", nargs="*", default=list(paths.datasets()))
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--s3-seeds", type=int, default=3)
    ap.add_argument("--ga-seeds", type=int, default=3)
    ap.add_argument("--no-s2-rf", action="store_true")
    a = ap.parse_args()
    CFG.update(s3_seeds=a.s3_seeds, ga_seeds=a.ga_seeds, s2_rf=not a.no_s2_rf)
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    run_main(a.jobs, a.datasets, a.repeats) if a.phase == "main" else run_exact(a.jobs, a.datasets)
