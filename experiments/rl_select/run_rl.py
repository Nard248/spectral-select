"""RL experiment driver (docs/rl_select/SPEC.md sections 7-9). Resumable; one process pool with a
priority scheduler whose width follows the CPU rule (<=5 workers while run_baselines is alive, else <=8,
minus the integer in results/rl_select/logs/reserve if present).

  PYTHONPATH=src python experiments/rl_select/run_rl.py s2   [--seeds 0 1]       # train + deploy S2 (+S3 for LODO)
  PYTHONPATH=src python experiments/rl_select/run_rl.py s1                         # per-instance search S1 (D4a)
  PYTHONPATH=src python experiments/rl_select/run_rl.py abl                        # S1 ablation on collagen_acetic
  PYTHONPATH=src python experiments/rl_select/run_rl.py s4                         # hardware-cost Pareto
  PYTHONPATH=src python experiments/rl_select/run_rl.py pilot                      # D1 pilot report
Rows go to results/rl_select/<setting>/<dataset>_rl.jsonl (same schema as run_baselines.py).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import heapq
import itertools
import json
import subprocess
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rl_select import paths  # noqa: E402

ROOT = paths.ROOT
RES = paths.results_root()          # env RL_SELECT_RESULTS (v2 rerun: results/rl_select_v2)
POL = RES / "policies"
NS = (3, 5, 8, 13)
DATASETS = paths.datasets()         # env RL_SELECT_DATASETS; default lichens1, lichens2_sub3, collagen_*
SEEDS = paths.seeds()               # env RL_SELECT_SEEDS; default 0,1,2
WORKERS = paths.workers()           # env RL_SELECT_WORKERS
B3 = (25, 50, 100, 200, 400)
N_UPDATES = 250
VARIANTS = {  # name -> (mode, init, feat); *_nowl = wavelength-agnostic features (ex/em/stokes zeroed)
    "rl_add": ("add", "empty", "full"),
    "rl_addremove": ("addremove", "empty", "full"),
    "rl_addremove_css": ("addremove", "css_greedy", "full"),
    "rl_add_nowl": ("add", "empty", "nowl"),
    "rl_addremove_nowl": ("addremove", "empty", "nowl"),
    "rl_addremove_css_nowl": ("addremove", "css_greedy", "nowl"),
}
POOLS = dict(paths.cross_pools(DATASETS))  # C2L / L2C: source name -> (training datasets, targets)
for _t in DATASETS:
    POOLS[f"lodo-{_t}"] = (tuple(d for d in DATASETS if d != _t), (_t,))


def tag_of(src: str) -> str:
    return "lodo" if src.startswith("lodo") else src


def pol_name(src: str, variant: str, seed: int) -> str:
    return f"{src}_{variant}_s{seed}"


# ------------------------------------------------------------------------------------------- io
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def key_of(r: dict) -> tuple:
    return (r["setting"], r["dataset"], r["repeat"], r["fold"], r["n"], r["method"], r["seed"], r["budget"])


def load_done() -> set:
    done = set()
    for s in ("S1", "S2", "S3", "S4"):
        for f in (RES / s).glob("*_rl.jsonl"):
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(key_of(json.loads(line)))
                    except json.JSONDecodeError:
                        pass
    return done


def write_rows(rows: list[dict]) -> None:
    for r in rows:
        d = RES / r["setting"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{r['dataset']}_rl.jsonl", "a") as f:
            f.write(json.dumps(r, default=_json_default) + "\n")


def mkrow(res, method, setting, task, budget, seed, ev, **extra) -> dict:
    from rl_select.evaluate import set_metrics
    row = asdict(res)
    row["trace"] = [list(t) for t in res.trace]
    row["extra"] = {k: v for k, v in row["extra"].items() if k != "trace_sets"}
    row["extra"].update(extra)
    row.update(method=method, setting=setting, dataset=task.dataset, repeat=task.repeat, fold=task.fold,
               budget=budget, seed=seed, eval=ev, set_metrics=set_metrics(task, res.selected))
    return row


class Evals:
    """Per-task outer-evaluation cache by frozenset(S)."""

    def __init__(self, task, clfs):
        self.task, self.clfs, self.c = task, tuple(clfs), {}

    def __call__(self, S):
        from rl_select.evaluate import evaluate
        k = frozenset(int(i) for i in S)
        have = self.c.setdefault(k, {})
        miss = [c for c in self.clfs if c not in have]
        if miss:
            have.update(evaluate(self.task, sorted(k), miss, seed=0))
        return {c: have[c] for c in self.clfs}


def _setup():
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)


# ------------------------------------------------------------------------------------------- jobs
def policy_done(out: Path) -> bool:
    c = out / "config.json"
    if not (c.exists() and (out / "policy.pt").exists()):
        return False
    return json.loads(c.read_text()).get("updates") == N_UPDATES


def job_train(a) -> list:
    src, variant, seed = a
    _setup()
    out = POL / pol_name(src, variant, seed)
    if policy_done(out):
        return []
    from rl_select.data import all_tasks
    from rl_select.objective import JSup
    from rl_select.rl_search import train_policy
    mode, init, feat = VARIANTS[variant]
    pool = [(t, JSup(t)) for ds in POOLS[src][0] for t in all_tasks(ds, 0)]
    train_policy(pool, mode=mode, init=init, sup_features=False, seed=seed, n_updates=N_UPDATES, out_dir=out,
                 snapshot_at=(150,), feat=feat)
    return []


def job_s2(a) -> list:
    src, variant, seed, target, repeat, done = a
    _setup()
    from rl_select.data import make_task, n_outer_folds
    from rl_select.policy import SelectPolicy
    from rl_select.rl_search import deploy_label_free
    pol = SelectPolicy.load(POL / pol_name(src, variant, seed))
    mode, init, _ = VARIANTS[variant]
    rows = []
    for fold in range(n_outer_folds(target)):
        task = make_task(target, repeat, fold)
        ev = Evals(task, ("knn", "lda", "svm", "rf"))
        for n in NS:
            for ns in (1, 16):
                meth = f"{variant}{'_k16' if ns > 1 else ''}@{tag_of(src)}"
                if ("S2", target, repeat, fold, n, meth, seed, 0) in done:
                    continue
                res = deploy_label_free(pol, task, n, mode=mode, init=init, n_samples=ns, seed=17 if ns > 1 else 0)
                rows.append(mkrow(res, meth, "S2", task, 0, seed, ev(res.selected), source=src,
                                  source_tag=tag_of(src), variant=variant, n_samples=ns))
    return rows


def _fresh(obj):
    obj._cache.clear()
    obj.n_evals = 0
    return obj


def job_s3(a) -> list:
    src, variant, seed, target, done = a
    _setup()
    from rl_select.data import make_task, n_outer_folds
    from rl_select.objective import JSup
    from rl_select.policy import SelectPolicy
    from rl_select.rl_search import deploy_few_shot
    pol = SelectPolicy.load(POL / pol_name(src, variant, seed))
    mode = VARIANTS[variant][0]
    meth = ("rl_add_bok" if mode == "add" else "rl_addremove_mc") + ("_nowl" if variant.endswith("nowl") else "") + "@lodo"
    rows = []
    for fold in range(n_outer_folds(target)):
        task = make_task(target, 0, fold)
        ev = Evals(task, ("knn", "lda", "svm"))
        obj = JSup(task)
        for n in NS:
            if all(("S3", target, 0, fold, n, meth, seed, b) in done for b in B3):
                continue
            _fresh(obj)
            res = deploy_few_shot(pol, task, n, obj, budget=max(B3), mode=mode, mc_lookahead=(mode == "addremove"),
                                  seed=seed)
            sets = res.extra["trace_sets"]
            for b in B3:
                idx = [i for i, (u, _) in enumerate(res.trace) if u <= b]
                if not idx:
                    continue
                r2 = copy.copy(res)
                last = idx[-1]
                r2.selected, r2.inner_J = sets[last], res.trace[last][1]
                r2.n_evals = min(res.n_evals, b)
                r2.trace = res.trace[:last + 1]
                rows.append(mkrow(r2, meth, "S3", task, b, seed, ev(r2.selected), source=src, variant=variant))
    return rows


def job_oracle(a) -> list:
    (fold,) = a
    _setup()
    from rl_select.data import make_task
    from rl_select.features import TaskFeatures
    from rl_select.objective import JSup
    t = make_task("collagen_acetic", 0, fold)
    TaskFeatures(t, JSup(t), sup="oracle")
    return []


def _per_instance(task, n, obj, mode, budget, seed, **kw):
    from rl_select.rl_search import rl_per_instance
    return rl_per_instance(task, n, obj, mode=mode, budget=budget, seed=seed, **kw)


def job_s1(a) -> list:
    ds, fold, n, mode, done = a
    _setup()
    from rl_select.data import make_task
    from rl_select.objective import JSup
    meth = f"rl_{mode}"
    if ("S1", ds, 0, fold, n, meth, 0, 5000) in done:
        return []
    task = make_task(ds, 0, fold)
    obj = JSup(task)
    res = _per_instance(task, n, obj, mode, 5000, 0, sup_features=True)
    ev = Evals(task, ("knn", "lda", "svm") + (("rf",) if n in (5, 13) else ()))
    return [mkrow(res, meth, "S1", task, 5000, 0, ev(res.selected))]


ABL = ("rl_add", "rl_add_oracle", "rl_addremove", "rl_addremove_oracle", "rl_addremove_terminal",
       "rl_addremove_reinforce", "rl_addremove_mc", "sa", "sfs_swap")
ABL_B = 20000


def job_abl(a) -> list:
    fold, meth, done = a
    _setup()
    from rl_select import baselines as B
    from rl_select.data import make_task
    from rl_select.objective import JSup
    if ("S1", "collagen_acetic", 0, fold, 8, meth, 0, ABL_B) in done:
        return []
    task = make_task("collagen_acetic", 0, fold)
    obj = JSup(task)
    n = 8
    if meth in ("sa", "sfs_swap"):
        res = B.BASELINES[meth](task, n, obj=obj, budget=ABL_B, seed=0)
    elif meth == "rl_addremove_mc":
        from rl_select.rl_search import deploy_few_shot, rl_per_instance
        t0 = time.time()
        half = ABL_B // 2
        res = rl_per_instance(task, n, obj, mode="addremove", budget=half, seed=0, sup_features=True,
                              keep_policy=True)
        used0 = res.n_evals
        la = deploy_few_shot(res.policy, task, n, obj, budget=ABL_B - used0, mode="addremove", mc_lookahead=True,
                             seed=0)
        extra = dict(train_evals=used0, lookahead_evals=la.n_evals, lookahead_accepted=la.extra.get("accepted"))
        if la.inner_J is not None and la.inner_J > res.inner_J:
            res.selected, res.inner_J = la.selected, la.inner_J
        res.trace = res.trace + [(used0 + u, j) for u, j in la.trace if j > res.trace[-1][1]]
        res.n_evals = used0 + la.n_evals
        res.runtime_s = time.time() - t0
        res.extra.update(extra)
    else:
        kw = dict(sup_features="oracle" if meth.endswith("_oracle") else True)
        mode = "add" if meth.startswith("rl_add") and "remove" not in meth else "addremove"
        if meth.endswith("terminal"):
            kw["reward"] = "terminal"
        if meth.endswith("reinforce"):
            kw["algo"] = "reinforce"
        res = _per_instance(task, n, obj, mode, ABL_B, 0, **kw)
    ev = Evals(task, ("knn", "lda", "svm", "rf"))
    return [mkrow(res, meth, "S1", task, ABL_B, 0, ev(res.selected), ablation=True)]


MUS = (0.0, 0.005, 0.01, 0.02, 0.05)


def job_s4(a) -> list:
    fold, mu, meth, done = a
    _setup()
    from rl_select import baselines as B
    from rl_select.data import make_task
    from rl_select.objective import JSup
    name = f"{meth}@mu{mu}"
    if ("S4", "lichens1", 0, fold, 8, name, 0, 5000) in done:
        return []
    task = make_task("lichens1", 0, fold)
    obj = JSup(task, mu_hw=mu)
    if meth == "sa":
        res = B.BASELINES["sa"](task, 8, obj=obj, budget=5000, seed=0)
    else:
        res = _per_instance(task, 8, obj, "addremove", 5000, 0, sup_features=True)
    ev = Evals(task, ("knn", "lda", "svm"))
    return [mkrow(res, name, "S4", task, 5000, 0, ev(res.selected), mu_hw=mu)]


JOBS = dict(train=job_train, s2=job_s2, s3=job_s3, oracle=job_oracle, s1=job_s1, abl=job_abl, s4=job_s4)


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


# ------------------------------------------------------------------------------------------- scheduler
def width() -> int:
    """Concurrent jobs. Fixed by RL_SELECT_WORKERS / --workers (the old pgrep/reserve laptop rule is gone)."""
    return WORKERS


def schedule(initial: list, followups=None, label="") -> None:
    """initial: [(prio, kind, args)]. followups(kind, args, rows) -> [(prio, kind, args)]."""
    cnt = itertools.count()
    heap = [(p, next(cnt), k, a) for p, k, a in initial]
    heapq.heapify(heap)
    running: dict = {}
    t0 = time.time()
    nrows = ndone = 0
    ntot = len(heap)
    with ProcessPoolExecutor(WORKERS) as ex:
        while heap or running:
            lim = width()
            while heap and len(running) < lim:
                p, _, k, a = heapq.heappop(heap)
                running[ex.submit(run_job, k, a)] = (p, k, a)
            done, _ = wait(list(running), timeout=15, return_when=FIRST_COMPLETED)
            for fu in done:
                p, k, a = running.pop(fu)
                try:
                    rows, dt = fu.result()
                except Exception as e:  # keep going; report
                    print(f"FAILED {k} {str(a[:5])[:100]}: {e!r}", flush=True)
                    continue
                write_rows(rows)
                nrows += len(rows)
                ndone += 1
                new = followups(k, a, rows) if followups else []
                for np_, nk, na in new:
                    heapq.heappush(heap, (np_, next(cnt), nk, na))
                    ntot += 1
                print(f"[{label} {ndone}/{ntot}] {k} {str(a[:5])[:80]} +{len(rows)} rows {dt:.0f}s | total rows {nrows} | "
                      f"{time.time() - t0:.0f}s run={len(running)} lim={lim}", flush=True)


# ------------------------------------------------------------------------------------------- phases
def s2_order(seeds):
    """(priority, source, variant, seed). Every variant x source x seed is trained (default 3 seeds).
    nowl is the primary grid (cross-family first); full-feature variants are the ablation (last)."""
    order = []
    for sd in seeds:
        base = sd * 10                      # seed 0 completes everything before seed 1, etc.
        for src in POOLS:
            for v in ("rl_add_nowl", "rl_addremove_nowl"):
                order.append((base + (0 if src in ("C2L", "L2C") else 1), src, v, sd))
        for src in POOLS:
            order.append((base + 2, src, "rl_addremove_css_nowl", sd))
        for src in POOLS:
            for v in ("rl_add", "rl_addremove", "rl_addremove_css"):
                order.append((base + 3.5, src, v, sd))
    return order


def run_s2(seeds, include_s3=True):
    done = load_done()
    order = s2_order(seeds)
    init = [(st, "train", (src, v, sd)) for st, src, v, sd in order]
    stage = {(src, v, sd): st for st, src, v, sd in order}

    def follow(kind, a, rows):
        out = []
        if kind != "train":
            return out
        src, v, sd = a
        st = stage[(src, v, sd)]
        for target in POOLS[src][1]:
            for rep in (0, 1, 2):
                out.append((st + 0.5, "s2", (src, v, sd, target, rep, done)))
            if include_s3 and src.startswith("lodo") and v in ("rl_add_nowl", "rl_addremove_nowl"):
                out.append((st + 0.6, "s3", (src, v, sd, target, done)))
        return out

    return init, follow


def run_s1():
    from rl_select.data import n_outer_folds
    done = load_done()
    init = []
    for ds in DATASETS:
        for f in range(n_outer_folds(ds)):
            for n in (5, 13):
                for mode in ("add", "addremove"):
                    init.append((2 if ds == "lichens1" else 2.1, "s1", (ds, f, n, mode, done)))
    return init, None


def run_abl():
    from rl_select.data import n_outer_folds
    done = load_done()
    nf = n_outer_folds("collagen_acetic")
    init = [(2, "oracle", (f,)) for f in range(nf)]
    for f in range(nf):
        for m in ABL:
            if not m.endswith("_oracle"):
                init.append((2.2, "abl", (f, m, done)))

    def follow(kind, a, rows):
        if kind == "oracle":
            return [(2.15, "abl", (a[0], m, done)) for m in ABL if m.endswith("_oracle")]
        return []
    return init, follow


def run_s4():
    from rl_select.data import n_outer_folds
    done = load_done()
    init = [(5, "s4", (f, mu, m, done)) for f in range(n_outer_folds("lichens1")) for mu in MUS for m in ("rl_addremove", "sa")]
    return init, None


def run_phases(names, seeds, include_s3):
    inits, follows = [], []
    for nm in names:
        i, f = {"s2": lambda: run_s2(seeds, include_s3), "s1": run_s1, "abl": run_abl, "s4": run_s4}[nm]()
        inits += i
        follows.append(f)

    def follow(kind, a, rows):
        out = []
        for f in follows:
            if f:
                out += f(kind, a, rows)
        return out
    schedule(inits, follow, "+".join(names))


# ------------------------------------------------------------------------------------------- pilot (D1)
PV = ('rl_add', 'rl_addremove', 'rl_add_nowl', 'rl_addremove_nowl')


def run_pilot():
    from rl_select import baselines as B
    from rl_select.data import make_task, n_outer_folds
    from rl_select.policy import SelectPolicy
    from rl_select.rl_search import deploy_label_free
    _setup()
    paths = {v: POL / pol_name("C2L", v, 0) / "ckpt_u150" for v in PV}
    while not all((p / "policy.pt").exists() and (p / "config.json").exists() for p in paths.values()):
        time.sleep(20)
    time.sleep(3)
    pols = {v: SelectPolicy.load(p) for v, p in paths.items()}
    tabs = {n: {} for n in NS}
    target = "lichens1"
    for fold in range(n_outer_folds(target)):
        task = make_task(target, 0, fold)
        ev = Evals(task, ("knn", "lda"))
        for n in NS:
            cell = tabs[n]
            rs = [ev(B.BASELINES["random"](task, n, seed=s).selected) for s in range(10)]
            cell.setdefault("random", []).append(np.mean([r["knn"]["balanced_acc"] for r in rs]))
            cell.setdefault("random_lda", []).append(np.mean([r["lda"]["balanced_acc"] for r in rs]))
            r = ev(B.BASELINES["css_greedy"](task, n, seed=0).selected)
            cell.setdefault("css_greedy", []).append(r["knn"]["balanced_acc"])
            cell.setdefault("css_greedy_lda", []).append(r["lda"]["balanced_acc"])
            for v, pol in pols.items():
                mode = VARIANTS[v][0]
                for ns in (1, 16):
                    res = deploy_label_free(pol, task, n, mode=mode, n_samples=ns, seed=17 if ns > 1 else 0)
                    r = ev(res.selected)
                    key = f"{v}{'_k16' if ns > 1 else ''}"
                    cell.setdefault(key, []).append(r["knn"]["balanced_acc"])
                    cell.setdefault(key + "_lda", []).append(r["lda"]["balanced_acc"])
        print("pilot fold", fold, flush=True)
    methods = ["random", "css_greedy"] + [v + k for v in PV for k in ("", "_k16")]
    lines = ["# D1 pilot: collagen -> lichens1 transfer (150 updates, label-free deployment)", "",
             "Policies trained on collagen_acetic + collagen_pepsin repeat-0 outer-fold training tasks (6 tasks), "
             "JSup reward, label-free features, n in {3,5,8,13}, 20% candidate dropout; snapshot after 150 PPO "
             "updates of the 250-update D2 run (C2L seed 0, so the entropy schedule is that of 250 updates). "
             "Deployed on lichens1 repeat 0, all 4 outer folds. Metric: outer test balanced accuracy (mean over "
             "folds); random = mean of 10 seeds.", "",
             "| clf | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for clf, suf in (("kNN", ""), ("LDA", "_lda")):
        for n in NS:
            vals = [np.mean(tabs[n][m + suf]) for m in methods]
            lines.append(f"| {clf} | {n} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    verdict = []
    for n in NS:
        c = tabs[n]
        best = max(np.mean(c[m]) for m in methods[2:])
        verdict.append(f"n={n}: best RL {best:.3f} vs random {np.mean(c['random']):.3f} vs css_greedy "
                       f"{np.mean(c['css_greedy']):.3f}")
    lines += ["", "kNN summary: " + "; ".join(verdict) + "."]
    (RES / "pilot_transfer.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="+", choices=("s2", "s1", "abl", "s4", "pilot", "all"))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS), help="policy training seeds (default 0 1 2)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-s3", action="store_true")
    a = ap.parse_args()
    if a.workers:
        WORKERS = a.workers
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    if a.phase == ["pilot"]:
        run_pilot()
    else:
        run_phases(["s2", "s1", "abl", "s4"] if a.phase == ["all"] else a.phase, a.seeds, not a.no_s3)
