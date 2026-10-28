"""RL iteration v3: task-augmented meta-training + the same label-free S2 deployment as run_rl.py.

  PYTHONPATH=src python experiments/rl_select/run_rl_v3.py run [--workers 4] [--updates 400]
Policies -> <results root>/policies_v3/<src>_<variant>_s<seed> ; rows -> <results root>/S2/<dataset>_rl_v3.jsonl
(results root: env RL_SELECT_RESULTS; seeds: --seeds / env RL_SELECT_SEEDS, default 0 1 2; workers: --workers)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import heapq
import itertools
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rl as R  # noqa: E402  (Agent D's driver: mkrow, Evals, POOLS, NS, tag_of, _setup, _json_default)

RES, POL3 = R.RES, R.RES / "policies_v3"
N_UPDATES = 400
VARIANTS = {  # name -> (mode, init, mix)
    "rl_add_aug": ("add", "empty", False),
    "rl_addremove_aug": ("addremove", "empty", False),
    "rl_addremove_css_aug": ("addremove", "css_greedy", False),
    "rl_addremove_mix_aug": ("addremove", "empty", True),
}


def pol_dir(src, variant, seed=0):
    return POL3 / f"{src}_{variant}_s{seed}"


def job_train(a):
    src, variant, seed, n_upd = a
    R._setup()
    out = pol_dir(src, variant, seed)
    c = out / "config.json"
    if c.exists() and (out / "policy.pt").exists() and json.loads(c.read_text()).get("updates") == n_upd:
        return []
    from rl_select.augment import build_pool
    from rl_select.rl_search import train_policy
    mode, init, mix = VARIANTS[variant]
    pool = build_pool(R.POOLS[src][0], mix=mix, seed=seed)
    print(f"  pool {src}/{variant}: {len(pool)} task views", flush=True)
    train_policy(pool, mode=mode, init=init, sup_features=False, seed=seed, n_updates=n_upd, out_dir=out, feat="nowl")
    return []


def job_s2(a):
    src, variant, seed, target, repeat, done = a
    R._setup()
    from rl_select.data import make_task, n_outer_folds
    from rl_select.policy import SelectPolicy
    from rl_select.rl_search import deploy_label_free
    pol = SelectPolicy.load(pol_dir(src, variant, seed))
    mode, init, _ = VARIANTS[variant]
    rows = []
    for fold in range(n_outer_folds(target)):
        task = make_task(target, repeat, fold)
        ev = R.Evals(task, ("knn", "lda", "svm", "rf"))
        for n in R.NS:
            for ns in (1, 16):
                meth = f"{variant}{'_k16' if ns > 1 else ''}@{R.tag_of(src)}"
                if ("S2", target, repeat, fold, n, meth, seed, 0) in done:
                    continue
                res = deploy_label_free(pol, task, n, mode=mode, init=init, n_samples=ns, seed=17 if ns > 1 else 0)
                rows.append(R.mkrow(res, meth, "S2", task, 0, seed, ev(res.selected), source=src,
                                    source_tag=R.tag_of(src), variant=variant, n_samples=ns, v3=True))
    return rows


def load_done():
    done = set()
    for f in (RES / "S2").glob("*_rl_v3.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    done.add(R.key_of(json.loads(line)))
                except json.JSONDecodeError:
                    pass
    return done


def write_rows(rows):
    for r in rows:
        with open(RES / "S2" / f"{r['dataset']}_rl_v3.jsonl", "a") as f:
            f.write(json.dumps(r, default=R._json_default) + "\n")


JOBS = dict(train=job_train, s2=job_s2)


def run_job(kind, args):
    import gc
    t0 = time.time()
    rows = JOBS[kind](args)
    try:
        from rl_select import data
        data.make_task.cache_clear()
        data.load_dataset.cache_clear()
    except Exception:
        pass
    gc.collect()
    return rows, time.time() - t0


def main(workers, n_upd, seeds=(0,)):
    done = load_done()
    srcs = ["C2L", "L2C"] + [s for s in R.POOLS if s.startswith("lodo")]
    prio = {s: (0 if s in ("C2L", "L2C") else 1) for s in srcs}
    vorder = ["rl_addremove_aug", "rl_add_aug", "rl_addremove_css_aug", "rl_addremove_mix_aug"]
    heap, cnt = [], itertools.count()
    for seed in seeds:
        for s in srcs:
            for k, v in enumerate(vorder):
                heapq.heappush(heap, (seed * 100 + prio[s] * 10 + k, next(cnt), "train", (s, v, seed, n_upd)))
    running, nrows, t0 = {}, 0, time.time()
    with ProcessPoolExecutor(workers) as ex:
        while heap or running:
            while heap and len(running) < workers:
                p, _, k, a = heapq.heappop(heap)
                running[ex.submit(run_job, k, a)] = (p, k, a)
            fin, _ = wait(list(running), timeout=30, return_when=FIRST_COMPLETED)
            for fu in fin:
                p, k, a = running.pop(fu)
                try:
                    rows, dt = fu.result()
                except Exception as e:
                    print(f"FAILED {k} {a[:4]}: {e!r}", flush=True)
                    continue
                write_rows(rows)
                nrows += len(rows)
                if k == "train":
                    src, v, sd, _n = a
                    for tgt in R.POOLS[src][1]:
                        for rep in (0, 1, 2):
                            heapq.heappush(heap, (p - 0.5, next(cnt), "s2", (src, v, sd, tgt, rep, done)))
                print(f"{k} {str(a[:5])[:90]} +{len(rows)} rows {dt:.0f}s | rows {nrows} | {time.time() - t0:.0f}s "
                      f"heap={len(heap)} run={len(running)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("run",))
    ap.add_argument("--workers", type=int, default=R.WORKERS)
    ap.add_argument("--updates", type=int, default=N_UPDATES)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(R.SEEDS))
    a = ap.parse_args()
    POL3.mkdir(parents=True, exist_ok=True)
    (RES / "S2").mkdir(parents=True, exist_ok=True)
    main(a.workers, a.updates, tuple(a.seeds))
