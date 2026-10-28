"""Digest of the RL experiments -> results/rl_select/summary/digest_rl.md (+ training_curves.csv).
Reads results/rl_select/S{1,2,3,4}/*_rl.jsonl (mine) and the baseline files <dataset>.jsonl (Agent C) when present.

  PYTHONPATH=src python experiments/rl_select/aggregate_rl.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_select import paths

ROOT = paths.ROOT
RES = paths.results_root()
OUT = RES / "summary"
NS = (3, 5, 8, 13)
DS = paths.datasets()
LF = ("variance", "uniform_grid", "spa", "css_greedy", "css_greedy_pc", "mmr_var")
RLV = tuple(f"{b}{s}{k}" for s in ("", "_nowl") for b in ("rl_add", "rl_addremove", "rl_addremove_css") for k in ("", "_k16"))


def load(setting: str, mine: bool | None = None) -> list[dict]:
    rows = []
    for f in sorted((RES / setting).glob("*.jsonl")):
        is_mine = f.stem.endswith("_rl")
        if mine is not None and is_mine != mine:
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def ba(r: dict, clf: str = "knn") -> float | None:
    e = r.get("eval", {}).get(clf)
    return None if e is None else e["balanced_acc"]


def ms(v) -> str:
    v = [x for x in v if x is not None]
    return "n/a" if not v else f"{np.mean(v):.3f}±{np.std(v):.3f}"


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def unit_means(rows, clf="knn") -> dict:
    """{(dataset, repeat, fold, n): mean BA over seeds} for a row list."""
    acc = defaultdict(list)
    for r in rows:
        b = ba(r, clf)
        if b is not None:
            acc[(r["dataset"], r["repeat"], r["fold"], r["n"])].append(b)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def wl(a: dict, b: dict) -> str:
    ks = sorted(set(a) & set(b))
    if not ks:
        return "n/a"
    w = sum(a[k] > b[k] + 1e-9 for k in ks)
    l = sum(a[k] < b[k] - 1e-9 for k in ks)
    return f"{w}/{l}/{len(ks) - w - l}"


# ----------------------------------------------------------------------------------------------- S2
def s2_section(clf: str = "knn") -> str:
    mine, base = load("S2", True), load("S2", False)
    if not mine:
        return "## S2\n(no rows yet)\n"
    by = defaultdict(list)  # (method_base, dataset, n) rows
    for r in mine:
        by[(r["method"], r["dataset"], r["n"])].append(r)
    bl = defaultdict(list)
    for r in base:
        bl[(r["method"], r["dataset"], r["n"])].append(r)
    srcs = sorted({r["method"].split("@")[1] for r in mine})
    out = [f"## S2 label-free transfer: outer {clf} balanced accuracy (mean±std over repeat x fold units, seeds averaged)", ""]
    wlrows = []
    for src in srcs:
        out.append(f"### source = {src}")
        rows = []
        tg = sorted({r["dataset"] for r in mine if r["method"].endswith("@" + src)}, key=DS.index)
        for ds in tg:
            for n in NS:
                cells = []
                um = {}
                for m in ("random",) + LF:
                    um[m] = unit_means(bl.get((m, ds, n), []), clf)
                best_lf = max(LF, key=lambda m: np.mean(list(um[m].values())) if um[m] else -1)
                for m in ("random", "css_greedy"):
                    cells.append(ms(um[m].values()))
                cells.append(f"{ms(um[best_lf].values())} ({best_lf})" if um[best_lf] else "n/a")
                rl = {}
                for v in RLV:
                    rl[v] = unit_means(by.get((f"{v}@{src}", ds, n), []), clf)
                    cells.append(ms(rl[v].values()))
                rows.append([ds, n] + cells)
                for sf in ("_nowl", ""):
                    g = lambda b, k="": rl[b + sf + k]
                    cand = [v for v in RLV if ("nowl" in v) == bool(sf) and rl[v]]
                    if not cand:
                        continue
                    bestv = max(cand, key=lambda v: np.mean(list(rl[v].values())))
                    wlrows.append([src, ds, n, bestv,
                                   wl(rl[bestv], um["random"]), wl(rl[bestv], um["css_greedy"]),
                                   wl(g("rl_addremove"), g("rl_add")), wl(g("rl_addremove", "_k16"), g("rl_add", "_k16")),
                                   wl(g("rl_addremove_css"), um["css_greedy"])])
        out.append(table(["target", "n", "random", "css_greedy", "best label-free"] + list(RLV), rows))
        out.append("")
    out.append("### Paired win/loss/tie counts over (repeat, fold) units (seed-averaged RL)")
    out.append("best RL = variant with highest mean on that cell; add/remove-vs-add compares greedy and k16 separately; "
               "css = rl_addremove_css (refinement from css_greedy) vs css_greedy itself.")
    out.append("")
    out.append(table(["source", "target", "n", "best RL", "best RL vs random", "best RL vs css_greedy",
                      "addremove vs add", "addremove_k16 vs add_k16", "addremove_css vs css_greedy"], wlrows))
    out.append("")
    # other classifiers, brief: mean over n of best RL vs random
    out.append("### Other classifiers (mean over units, rl_addremove_k16 / rl_add_k16 / random), lodo source")
    rows = []
    for c in ("lda", "svm", "rf"):
        for ds in DS:
            vals = []
            for m, pool in (("random", bl), ("rl_add_nowl_k16@lodo", by), ("rl_addremove_nowl_k16@lodo", by)):
                u = {}
                for n in NS:
                    u.update(unit_means(pool.get((m, ds, n), []), c))
                vals.append(ms(u.values()))
            rows.append([c, ds] + vals)
    out.append(table(["clf", "target", "random", "rl_add_nowl_k16", "rl_addremove_nowl_k16"], rows))
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------- S3
def s3_section() -> str:
    mine, base = load("S3", True), load("S3", False)
    if not mine:
        return "## S3\n(no rows yet)\n"
    B = (25, 50, 100, 200, 400)
    out = ["## S3 few-shot (LODO policies, JSup observable): outer kNN BA (mean over folds x seeds) / inner JSup", ""]
    meths = ["rl_add_bok_nowl@lodo", "rl_addremove_mc_nowl@lodo", "random_search", "sa", "ga", "sfs_swap"]
    pool = defaultdict(list)
    for r in mine + base:
        pool[(r["method"], r["dataset"], r["n"], r["budget"])].append(r)
    for ds in DS:
        rows = []
        for n in NS:
            for b in B:
                cells = []
                for m in meths:
                    rs = pool.get((m, ds, n, b), [])
                    if not rs:
                        cells.append("n/a")
                        continue
                    cells.append(f"{np.mean([ba(r) for r in rs]):.3f} / {np.mean([r['inner_J'] for r in rs if r['inner_J'] is not None]):.3f}")
                rows.append([n, b] + cells)
        out += [f"### {ds}", table(["n", "B"] + meths, rows), ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------- S1 / ablation / S4
def s1_section() -> str:
    rows = [r for r in load("S1") if r["budget"] == 5000]
    if not rows:
        return "## S1\n(no rows yet)\n"
    meths = ["rl_add", "rl_addremove", "sfs_swap", "sa", "ga", "beam_swap", "sfs", "random_search"]
    pool = defaultdict(list)
    for r in rows:
        pool[(r["method"], r["dataset"], r["n"])].append(r)
    out = ["## S1 per-instance search, budget 5000 JSup evals (repeat 0): inner JSup / outer kNN BA, mean over folds (and seeds)", ""]
    tr = []
    for ds in DS:
        for n in (5, 13):
            cells = []
            for m in meths:
                rs = pool.get((m, ds, n), [])
                cells.append("n/a" if not rs else f"{np.mean([r['inner_J'] for r in rs]):.4f} / {np.mean([ba(r) for r in rs]):.3f}")
            tr.append([ds, n] + cells)
    out.append(table(["dataset", "n"] + meths, tr))
    # evals to reach within 0.005 of sfs_swap final
    out.append("")
    out.append("RL evals to within 0.005 of own final J (mean over tasks): "
               + ", ".join(f"{m}={np.mean([_evals_to(r) for k, v in pool.items() if k[0] == m for r in v]):.0f}"
                           for m in ("rl_add", "rl_addremove") if any(k[0] == m for k in pool)))
    return "\n".join(out) + "\n"


def _evals_to(r, tol=0.005):
    t = r.get("trace") or []
    if not t:
        return r["n_evals"]
    fin = t[-1][1]
    for u, j in t:
        if j >= fin - tol:
            return u
    return t[-1][0]


def abl_section() -> str:
    rows = [r for r in load("S1") if r["budget"] == 20000 and r["dataset"] == "collagen_acetic"]
    if not rows:
        return "## Ablation\n(no rows yet)\n"
    order = ["rl_add", "rl_add_oracle", "rl_addremove", "rl_addremove_oracle", "rl_addremove_terminal",
             "rl_addremove_reinforce", "rl_addremove_mc", "sa", "sfs_swap"]
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    tb = []
    for m in order:
        rs = by.get(m, [])
        if rs:
            tb.append([m, len(rs), ms([r["inner_J"] for r in rs]), ms([ba(r) for r in rs]), ms([ba(r, "svm") for r in rs]),
                       f"{np.mean([r['n_evals'] for r in rs]):.0f}", f"{np.mean([_evals_to(r) for r in rs]):.0f}"])
    return ("## Ablation: collagen_acetic repeat 0, n=8, budget 20000 (oracle priors cost F(F-1)/2=11026 evals, charged)\n\n"
            + table(["method", "folds", "inner JSup", "outer kNN BA", "outer SVM BA", "evals used", "evals to final-0.005"], tb) + "\n")


def s4_section() -> str:
    rows = load("S4")
    if not rows:
        return "## S4\n(no rows yet)\n"
    by = defaultdict(list)
    for r in rows:
        by[(r["method"].split("@")[0], r["extra"]["mu_hw"])].append(r)
    tb = []
    for (m, mu), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        tb.append([m, mu, len(rs), f"{np.mean([r['set_metrics']['n_distinct_excitations'] for r in rs]):.2f}",
                   ms([ba(r) for r in rs]), ms([r["inner_J"] for r in rs])])
    return ("## S4 hardware cost (lichens1 repeat 0, n=8, budget 5000)\n\n"
            + table(["method", "mu_hw", "folds", "mean #distinct excitations", "outer kNN BA", "inner J (with penalty)"], tb) + "\n")


# ----------------------------------------------------------------------------------------------- training curves
def curves_section() -> str:
    pol = RES / "policies"
    rows, summ = [], defaultdict(list)
    for d in sorted(pol.glob("*_s*")):
        f = d / "train_log.csv"
        if not f.exists():
            continue
        mm = re.match(r"^(.*?)_(rl_.*)_s(\d+)$", d.name)
        src, variant, sd = mm.group(1), mm.group(2), mm.group(3)
        with open(f) as fh:
            log = list(csv.DictReader(fh))
        if len(log) < 20:
            continue
        J = np.array([float(r["mean_final_J"]) for r in log])
        ent = np.array([float(r["entropy"]) for r in log])
        for r in log:
            rows.append([src, variant, sd, r["update"], r["mean_final_J"], r["entropy"]])
        k = len(J)
        summ[variant].append(dict(src=src, J10=J[:10].mean(), J50=J[40:60].mean() if k >= 60 else np.nan,
                                  J100=J[90:110].mean() if k >= 110 else np.nan, Jend=J[-20:].mean(),
                                  u90=int(np.argmax(np.convolve(J, np.ones(10) / 10, "same") >= J[-20:].mean() - 0.02 * 0 - 0.01)),
                                  ent0=ent[:10].mean(), ent_end=ent[-20:].mean(), n=k))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "training_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "variant", "seed", "update", "mean_final_J", "entropy"])
        w.writerows(rows)
    if not summ:
        return "## Training curves\n(none yet)\n"
    tb = []
    for v, L in summ.items():
        tb.append([v, len(L), f"{np.mean([x['J10'] for x in L]):.3f}", f"{np.nanmean([x['J50'] for x in L]):.3f}",
                   f"{np.nanmean([x['J100'] for x in L]):.3f}", f"{np.mean([x['Jend'] for x in L]):.3f}",
                   f"{np.mean([x['u90'] for x in L]):.0f}", f"{np.mean([x['ent0'] for x in L]):.2f}->{np.mean([x['ent_end'] for x in L]):.2f}"])
    per = []
    srcs = sorted({x["src"] for L in summ.values() for x in L})
    for s in srcs:
        cells = []
        for v in ("rl_add", "rl_addremove", "rl_addremove_css", "rl_add_nowl", "rl_addremove_nowl", "rl_addremove_css_nowl"):
            xs = [x["Jend"] for x in summ.get(v, []) if x["src"] == s]
            cells.append(f"{np.mean(xs):.3f}" if xs else "n/a")
        per.append([s] + cells)
    return ("## Training curves (mean final JSup of training episodes, 20% candidate dropout, n in {3,5,8,13} mixed; "
            "full per-update curves in training_curves.csv)\n\n"
            + table(["variant", "#policies", "J@first10", "J@~50", "J@~100", "J@last20", "update reaching last-20 level -0.01", "entropy first->last"], tb)
            + "\n\nFinal training J by source pool:\n\n" + table(["source", "add", "addremove", "addremove_css", "add_nowl", "addremove_nowl", "addremove_css_nowl"], per) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = "# RL study digest\n\n" + "\n".join(
        f() for f in (s2_section, s3_section, s1_section, abl_section, s4_section, curves_section))
    (OUT / "digest_rl.md").write_text(text)
    print(f"wrote {OUT / 'digest_rl.md'} ({len(text)} chars)")


if __name__ == "__main__":
    main()
