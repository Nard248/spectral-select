"""Speed + sanity report for the rl_select core. Writes results/rl_select/bench_core.txt."""
from __future__ import annotations

import io
import time
import warnings
from pathlib import Path

import numpy as np

from rl_select import baselines as B
from rl_select.data import DATASETS, inner_folds, make_task
from rl_select.evaluate import evaluate
from rl_select.objective import JRecon, JReconPC, JSup

warnings.filterwarnings("ignore")
OUT = io.StringIO()


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + "\n")


def ms_per_eval(J, F, size, n=300, seed=0):
    rng = np.random.default_rng(seed)
    Ss = [rng.choice(F, size, replace=False) for _ in range(n)]
    t = time.perf_counter()
    for S in Ss:
        J(S)
    return (time.perf_counter() - t) / n * 1e3


p("== task construction / objective init ==")
for k in DATASETS:
    make_task.cache_clear()
    t = time.perf_counter(); T = make_task(k, 0, 0); dt = time.perf_counter() - t
    t = time.perf_counter(); J = JSup(T); dj = time.perf_counter() - t
    t = time.perf_counter(); R = JRecon(T); JReconPC(T); dr = time.perf_counter() - t
    p(f"{k}: Xtr{T.Xtr.shape} Xte{T.Xte.shape} sub={len(T.sub_idx)} inner_folds={len(inner_folds(T))} "
      f"task={dt:.2f}s JSup_init={dj:.2f}s recon_init(corr+pca)={dr:.2f}s")
    p("   JSup ms/eval  " + "  ".join(f"|S|={s}: {ms_per_eval(J, T.n_features, s):.3f}" for s in (3, 8, 13, 20)))
    p(f"   JRecon ms/eval |S|=13: {ms_per_eval(R, T.n_features, 13):.3f}   "
      f"JReconPC ms/eval |S|=13: {ms_per_eval(JReconPC(T), T.n_features, 13):.3f}")

p("\n== evaluate() time per classifier (lichens1 r0 f0, |S|=13) ==")
T = make_task("lichens1", 0, 0)
S13 = list(np.random.default_rng(0).choice(T.n_features, 13, replace=False))
for c in ("knn", "lda", "rf", "svm"):
    t = time.perf_counter(); evaluate(T, S13, [c]); p(f"{c}: {time.perf_counter() - t:.2f}s")

p("\n== sanity table: lichens1 r0 f0, n=13 (inner JSup, outer kNN balanced acc) ==")
J = JSup(T)
rows = []
def add(name, sets):
    inner = np.mean([J(s) for s in sets])
    knn = np.mean([evaluate(T, s, ["knn"])["knn"]["balanced_acc"] for s in sets])
    rows.append((name, inner, knn))
add("random (mean of 10)", [B.random_(T, 13, seed=s).selected for s in range(10)])
for name in ("variance", "css_greedy", "fisher"):
    add(name, [B.BASELINES[name](T, 13).selected])
for name in ("sfs", "sa"):
    t = time.perf_counter(); r = B.BASELINES[name](T, 13, obj=JSup(T), budget=2000, seed=0)
    rows.append((f"{name}(B=2000) [{r.runtime_s:.1f}s, evals={r.n_evals}]", J(r.selected),
                 evaluate(T, r.selected, ["knn"])["knn"]["balanced_acc"]))
p(f"{'method':44s} {'innerJSup':>10s} {'outer kNN BA':>13s}")
for n_, a, b in rows:
    p(f"{n_:44s} {a:10.4f} {b:13.4f}")

Path("results/rl_select").mkdir(parents=True, exist_ok=True)
Path("results/rl_select/bench_core.txt").write_text(OUT.getvalue())
