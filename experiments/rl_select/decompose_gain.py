"""Decompose the add/remove - add-only gain of the S2 label-free transfer into a construction and a refinement term.

  BA(S_f) - BA(S_a) = [BA(S_c) - BA(S_a)] (construction: the reversibly-trained policy builds a different set)
                    + [BA(S_f) - BA(S_c)] (refinement: the remove/add swaps)
S_c = add/remove set after the n construction adds, S_f = its final set, S_a = final set of the matched add-only policy.
Rollouts replicate run_rl.job_s2 / deploy_label_free exactly (greedy, and best-of-16 by J_rec with seed 17), every final
set is asserted equal to the stored S2 record, and kNN BA is computed with evaluate(seed=0) as in run_rl.Evals.

  PYTHONPATH=src .venv/bin/python experiments/rl_select/decompose_gain.py [--reuse]
Writes results/rl_select/summary/{decompose_gain_units.csv, decompose_gain.csv, decompose_gain.md}.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rl_select import paths

ROOT = paths.ROOT
RES = paths.results_root()
SUM = RES / "summary"
DATASETS = paths.datasets()
CROSS = paths.cross_of(DATASETS)
NS = (3, 5, 8, 13)
SEEDS = paths.seeds()


def episodes(policy, tf, n, mode, k):
    """deploy_label_free rollouts, additionally returning the post-construction set of each episode."""
    import torch
    from torch.distributions import Categorical
    from rl_select.env import EnvConfig, EpSpec, VecSelectEnv
    from rl_select.policy import obs_to_tensors
    seed = 17 if k > 1 else 0
    ecfg = EnvConfig(mode=mode, init="empty", reward="terminal", dropout=0.0, T=None, ns=(n,))
    venv = VecSelectEnv([(tf, None)], ecfg, seed)
    torch.manual_seed(seed)
    venv.reset([EpSpec(0, n, 0.0)] * k)
    Sc = [None] * k
    while len(venv.active()):                      # identical to ppo.collect_rollout
        obs = venv.observe()
        with torch.no_grad():
            logits, _ = policy(*obs_to_tensors(obs))
            a = logits.argmax(-1) if k == 1 else Categorical(logits=logits).sample()
        venv.step(obs.env_idx, a.numpy())
        for i in obs.env_idx:
            e = venv.episodes[i]
            if e.n_steps == n:
                Sc[i] = sorted(int(x) for x in e.order)
    sets = [e.current_set() for e in venv.episodes]
    b = int(np.argmax([tf.jrecon(s) for s in sets]))
    return Sc[b], sets[b]


def stored_index():
    idx = {}
    for ds in DATASETS:
        for line in open(RES / "S2" / f"{ds}_rl.jsonl"):
            r = json.loads(line)
            if "nowl" in r["method"] and "css" not in r["method"]:
                idx[(r["method"], r["dataset"], r["repeat"], r["fold"], r["n"], r["seed"])] = r
    return idx


def work(ds):
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from rl_select.data import make_task, n_outer_folds
    from rl_select.evaluate import evaluate
    from rl_select.features import TaskFeatures
    from rl_select.policy import SelectPolicy
    idx = stored_index()
    pols = {}
    for src in (CROSS[ds], f"lodo-{ds}"):
        for sd in SEEDS:
            for v in ("rl_addremove_nowl", "rl_add_nowl"):
                pols[(src, v, sd)] = SelectPolicy.load(RES / "policies" / f"{src}_{v}_s{sd}")
    rows = []
    for rep in (0, 1, 2):
        for fold in range(n_outer_folds(ds)):
            task = make_task(ds, rep, fold)
            tf = TaskFeatures(task, None, sup="none", feat="nowl")
            cache = {}

            def ba(S):
                k = frozenset(S)
                if k not in cache:
                    cache[k] = evaluate(task, sorted(k), ("knn",), seed=0)["knn"]["balanced_acc"]
                return cache[k]

            def nex(S):
                return len(np.unique(task.ex_nm[list(S)]))

            for src in (CROSS[ds], f"lodo-{ds}"):
                tag = "lodo" if src.startswith("lodo") else src
                for sd in SEEDS:
                    for n in NS:
                        for k in (1, 16):
                            sfx = "_k16" if k > 1 else ""
                            Sc, Sf = episodes(pols[(src, "rl_addremove_nowl", sd)], tf, n, "addremove", k)
                            _, Sa = episodes(pols[(src, "rl_add_nowl", sd)], tf, n, "add", k)
                            ra = idx[(f"rl_addremove_nowl{sfx}@{tag}", ds, rep, fold, n, sd)]
                            rd = idx[(f"rl_add_nowl{sfx}@{tag}", ds, rep, fold, n, sd)]
                            assert sorted(ra["selected"]) == Sf and sorted(rd["selected"]) == Sa, (ds, rep, fold, n)
                            bf, bc, bA = ba(Sf), ba(Sc), ba(Sa)
                            sa_ = set(Sa)
                            rows.append(dict(
                                target=ds, source=("cross-family" if tag != "lodo" else "lodo"), src=src, seed=sd,
                                deploy="k16" if k > 1 else "greedy", repeat=rep, fold=fold, n=n,
                                ba_Sa=bA, ba_Sc=bc, ba_Sf=bf,
                                stored_ba_ar=ra["eval"]["knn"]["balanced_acc"], stored_ba_add=rd["eval"]["knn"]["balanced_acc"],
                                jac_Sc_Sa=len(set(Sc) & sa_) / len(set(Sc) | sa_),
                                jac_Sf_Sa=len(set(Sf) & sa_) / len(set(Sf) | sa_),
                                jac_Sf_Sc=len(set(Sf) & set(Sc)) / len(set(Sf) | set(Sc)),
                                exc_Sa=nex(Sa), exc_Sc=nex(Sc), exc_Sf=nex(Sf), changed=int(Sc != Sf)))
    print("done", ds, len(rows), flush=True)
    return rows


def boot(x, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.integers(0, len(x), (nb, len(x)))
    mu = x[m].mean(1)
    return float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))


def summarise(units):
    A = lambda c: np.array([u[c] for u in units], float)
    terms = {"total": A("ba_Sf") - A("ba_Sa"), "construction": A("ba_Sc") - A("ba_Sa"),
             "refinement": A("ba_Sf") - A("ba_Sc")}
    out = []
    for t, x in terms.items():
        lo, hi = boot(x)
        out.append((t, x.mean(), lo, hi, int((x > 1e-12).sum()), int((x < -1e-12).sum()), int((abs(x) <= 1e-12).sum())))
    for t in ("jac_Sc_Sa", "jac_Sf_Sa", "jac_Sf_Sc", "exc_Sa", "exc_Sc", "exc_Sf", "ba_Sa", "ba_Sc", "ba_Sf", "changed"):
        out.append((t, A(t).mean(), None, None, None, None, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    SUM.mkdir(parents=True, exist_ok=True)
    up = SUM / "decompose_gain_units.csv"
    if a.reuse and up.exists():
        units = list(csv.DictReader(open(up)))
        for u in units:
            for k, v in u.items():
                if k not in ("target", "source", "src", "deploy"):
                    u[k] = float(v)
    else:
        with ProcessPoolExecutor(min(paths.workers(), len(DATASETS))) as ex:
            units = [r for rs in ex.map(work, DATASETS) for r in rs]
        with open(up, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(units[0]))
            w.writeheader()
            w.writerows(units)
    # verification against stored S2 gain
    mine = np.array([u["ba_Sf"] - u["ba_Sa"] for u in units])
    stored = np.array([u["stored_ba_ar"] - u["stored_ba_add"] for u in units])
    ver = dict(n=len(units), mine=mine.mean(), stored=stored.mean(), maxdiff=float(np.abs(mine - stored).max()),
               wins=int((stored > 1e-12).sum()), losses=int((stored < -1e-12).sum()))
    groups = [("pooled", "all", units)]
    for n in NS:
        groups.append(("n", str(n), [u for u in units if u["n"] == n]))
    for d in DATASETS:
        groups.append(("target", d, [u for u in units if u["target"] == d]))
    for s in ("cross-family", "lodo"):
        groups.append(("source", s, [u for u in units if u["source"] == s]))
        for n in NS:
            groups.append((f"{s} x n", str(n), [u for u in units if u["source"] == s and u["n"] == n]))
        for d in DATASETS:
            groups.append((f"{s} x target", d, [u for u in units if u["source"] == s and u["target"] == d]))
    for dp in ("greedy", "k16"):
        groups.append(("deploy", dp, [u for u in units if u["deploy"] == dp]))
    for sd in SEEDS:
        groups.append(("seed", str(sd), [u for u in units if int(u["seed"]) == sd]))
    rows = []
    for g, lv, us in groups:
        for t, m, lo, hi, w, l, ti in summarise(us):
            rows.append(dict(group=g, level=lv, n_units=len(us), term=t, mean=m, ci_lo=lo, ci_hi=hi, wins=w,
                             losses=l, ties=ti))
    with open(SUM / "decompose_gain.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def cell(g, lv, t):
        r = next(r for r in rows if (r["group"], r["level"], r["term"]) == (g, lv, t))
        return r

    md = ["# Decomposition of the add/remove - add-only gain (S2, label-free, wavelength-free policies)", "",
          "BA(S_f) - BA(S_a) = [BA(S_c) - BA(S_a)] (construction) + [BA(S_f) - BA(S_c)] (refinement). Unit = target "
          "fold x repeat x n x source x policy seed x deployment (greedy / best-of-16). Outer kNN BA; mean, 95% "
          "unit-level bootstrap CI (2000 resamples), wins/losses/ties. S_c: add/remove set after the n construction "
          "adds; S_f: its final set; S_a: matched add-only final set.", "",
          f"**Verification.** {ver['n']} paired units; pooled stored S2 gain (add/remove - add-only, same records) "
          f"{ver['stored']:+.4f} vs recomputed BA(S_f)-BA(S_a) {ver['mine']:+.4f}; max per-unit |diff| "
          f"{ver['maxdiff']:.2e}; stored wins/losses {ver['wins']}/{ver['losses']}. Every final set of both policies "
          "was asserted identical to the stored record.", ""]
    hdr = "| {} | units | total | construction | refinement |\n|---|---|---|---|---|"

    def fmt(r, w=True):
        s = f"{r['mean']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        return s + (f" {r['wins']}/{r['losses']}" if w else "")

    for title, g, lvls in (("Pooled", "pooled", ["all"]), ("Per n", "n", [str(n) for n in NS]),
                           ("Per target", "target", list(DATASETS)), ("Source", "source", ["cross-family", "lodo"]),
                           ("Cross-family x n", "cross-family x n", [str(n) for n in NS]),
                           ("LODO x n", "lodo x n", [str(n) for n in NS]),
                           ("Cross-family x target", "cross-family x target", list(DATASETS)),
                           ("LODO x target", "lodo x target", list(DATASETS)),
                           ("Deployment", "deploy", ["greedy", "k16"]), ("Policy seed", "seed", [str(s) for s in SEEDS])):
        md += [f"## {title}", "", hdr.format(g)]
        for lv in lvls:
            c = [cell(g, lv, t) for t in ("total", "construction", "refinement")]
            md.append(f"| {lv} | {c[0]['n_units']} | " + " | ".join(fmt(x) for x in c) + " |")
        md.append("")
    md += ["## Set structure (mean over units)", "",
           "| group | Jaccard(S_c,S_a) | Jaccard(S_f,S_a) | Jaccard(S_f,S_c) | #exc S_a | #exc S_c | #exc S_f | frac. S_f != S_c |"
           "\n|---|---|---|---|---|---|---|---|"]
    for g, lv in [("pooled", "all")] + [("n", str(n)) for n in NS] + [("source", s) for s in ("cross-family", "lodo")] + \
                 [("target", d) for d in DATASETS]:
        v = [cell(g, lv, t)["mean"] for t in ("jac_Sc_Sa", "jac_Sf_Sa", "jac_Sf_Sc", "exc_Sa", "exc_Sc", "exc_Sf", "changed")]
        md.append(f"| {g}={lv} | " + " | ".join(f"{x:.3f}" if i < 3 or i == 6 else f"{x:.2f}" for i, x in enumerate(v)) + " |")
    md += ["", "Cell format: mean [95% CI] wins/losses."]
    (SUM / "decompose_gain.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
