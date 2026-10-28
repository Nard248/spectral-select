"""Downstream score vs. number of pairs n, for every algorithm and selection method (PROTOTYPE).

Grid: rows = datasets, columns = kNN / LDA / SVM / RF (balanced accuracy), k-means / GMM (ARI), NMF (matched BA).
x = n in {3,5,8,13}; one line per selection method; "All pairs" as a horizontal band.  Mean +- 95 % bootstrap CI.

  * S2 methods (RL add/remove, RL add-only, CSS, Random): ALL stored repeats x folds (units = (repeat, fold); Random is
    first averaged over its stored draws).  RL = the matched `*_nowl_k16@<src>` policy, policy seed(s) from --rl-seeds.
  * SA (S1): repeat 0 only; units = folds, averaged over the stored search seeds.
  * kNN / LDA / SVM / RF: read from the records' `eval` (same protocol as rl_select.evaluate); recomputed with
    outcome_maps.fit_predict only if a record lacks that classifier.
  * k-means ARI, GMM ARI, NMF matched BA: computed fresh per (repeat, fold) set on a FIXED random subsample of at most
    --max-pixels annotated pixels (train + held-out specimens of that fold; seed fixed per task), features log1p +
    standardised with the fold's training statistics (k-means, GMM) or raw p99-scaled non-negative (NMF), as outcome_maps.

  PYTHONPATH=src .venv/bin/python experiments/rl_select/score_vs_n.py --results-root results/rl_select \
      --datasets lichens1,lichens2,collagen_acetic,collagen_pepsin

Writes to --out: score_vs_n_raw.csv (every unit value), score_vs_n_summary.csv (mean, CI; read by
outcome_maps.py --label-mean-over-repeats), fig_score_vs_n.{pdf,png}.  --render-only redraws from the CSVs.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["ME_ATTENTION_RECORDED_WL"] = "1"

import argparse
import csv
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outcome_maps as om  # noqa: E402

ROOT = om.ROOT
SUP, UNSUP = om.SUP, om.UNSUP
ALGOS = om.ALGOS
METHOD_KEYS = ("all", "rl_addremove", "rl_add", "css", "random", "sa")
DATASETS = ("lichens1", "lichens2", "collagen_acetic", "collagen_pepsin")


# ------------------------------------------------------------------------------------------------ selections
def collect(key: str, ns, rl_seeds):
    """{(method, n): {(repeat, fold): [records or None for all-pairs]}} straight from the JSONL files."""
    src = om.cross_tag(key)
    s2 = om._read_jsonl(om.RES / "S2" / f"{key}.jsonl")
    s2rl = om._read_jsonl(om.RES / "S2" / f"{key}_rl.jsonl")
    s1 = om._read_jsonl(om.RES / "S1" / f"{key}.jsonl")
    spec = {"rl_addremove": (s2rl, f"rl_addremove_nowl_k16@{src}"), "rl_add": (s2rl, f"rl_add_nowl_k16@{src}"),
            "css": (s2, "css_greedy"), "random": (s2, "random"), "sa": (s1, "sa")}
    out: dict = defaultdict(lambda: defaultdict(list))
    for m, (rows, name) in spec.items():
        for r in rows:
            if r["method"] != name or r["n"] not in ns:
                continue
            if m == "sa" and r["repeat"] != 0:
                continue
            if m.startswith("rl") and rl_seeds is not None and r["seed"] not in rl_seeds:
                continue
            out[(m, r["n"])][(r["repeat"], r["fold"])].append(r)
    allp = defaultdict(list)
    for r in s2:
        if r["method"] == "all_pairs":
            allp[(r["repeat"], r["fold"])].append(r)
    for n in ns:
        out[("all", n)] = allp
    return out


# ------------------------------------------------------------------------------------------------ per-task worker
def worker(key, results_root, repeat, fold, items, max_pixels, need_unsup, n_init):
    """items: list of (method, n, draw, S(list or None=all), seed, missing_sup list). Returns raw rows."""
    warnings.filterwarnings("ignore")
    om.set_results_root(results_root)
    from rl_select.data import load_dataset, make_task, specimen_folds
    from rl_select.stats import metrics
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    d = load_dataset(key)
    K = d["n_classes"]
    task = make_task(key, repeat, fold)
    rows = []
    sub = Xs = Xr = ys = gs = None
    if need_unsup:
        folds = specimen_folds(d, repeat)
        te = np.isin(d["groups"], folds[fold])
        order = np.concatenate([np.flatnonzero(~te), np.flatnonzero(te)])
        Xall = np.vstack([task.Xtr, task.Xte])
        yall = np.concatenate([task.ytr, task.yte])
        gall = np.concatenate([task.gtr, task.gte])
        assert np.array_equal(d["y"][order], yall)
        rng = np.random.default_rng(1000 + 100 * repeat + fold)
        sub = np.sort(rng.choice(len(yall), size=min(max_pixels, len(yall)), replace=False))
        Xs, ys, gs = Xall[sub], yall[sub], gall[sub]
        Xr = np.clip(d["X"][order][sub].astype(np.float64), 0, None)
        del Xall
    for (m, n, draw, S, seed, missing) in items:
        S = np.arange(task.Xtr.shape[1]) if S is None else np.asarray(S, dtype=np.int64)
        for clf in missing:
            pred = om.fit_predict(task, S, clf, seed)
            rows.append((m, n, repeat, fold, draw, clf, metrics(task.yte, pred, task.gte, K)["balanced_acc"]))
        if not need_unsup:
            continue
        Z = Xs[:, S]
        lab = KMeans(n_clusters=K, n_init=n_init, random_state=0).fit_predict(Z)
        rows.append((m, n, repeat, fold, draw, "kmeans", float(adjusted_rand_score(ys, lab))))
        reg = 1e-6
        for _ in range(4):
            try:
                gm = GaussianMixture(n_components=K, covariance_type="full", random_state=0, reg_covar=reg).fit(Z)
                break
            except ValueError:
                reg *= 100
        else:
            raise RuntimeError("GMM failed")
        rows.append((m, n, repeat, fold, draw, "gmm", float(adjusted_rand_score(ys, gm.predict(Z)))))
        A = Xr[:, S].copy()
        p99 = np.percentile(A, 99, axis=0)
        p99[p99 <= 0] = 1.0
        A /= p99
        nm = NMF(n_components=K, init="nndsvda" if K <= A.shape[1] else "random", max_iter=1000, random_state=0)
        W = nm.fit_transform(A)
        hn = np.linalg.norm(nm.components_, axis=1)
        hn[hn == 0] = 1.0
        Wm = W * hn[None, :]
        tot = Wm.sum(1, keepdims=True)
        Wn = np.where(tot <= 1e-12, 1.0 / K, Wm / np.maximum(tot, 1e-12))
        Mx = np.stack([Wn[ys == c].mean(0) for c in range(K)])
        r, c = om.linear_sum_assignment(-Mx)
        c2c = np.zeros(K, dtype=np.int64)
        c2c[c] = r
        pred = c2c[Wn.argmax(1)]
        rows.append((m, n, repeat, fold, draw, "nmf", float(metrics(ys, pred, gs, K)["balanced_acc"])))
    return rows


# ------------------------------------------------------------------------------------------------ driver
def run_dataset(key, args, log):
    from joblib import Parallel, delayed
    ns = [int(v) for v in args.ns.split(",")]
    rl_seeds = None if args.rl_seeds == "all" else {int(v) for v in args.rl_seeds.split(",")}
    col = collect(key, set(ns), rl_seeds)
    raw = []
    per_task: dict = defaultdict(list)
    for (m, n), tasks in col.items():
        if m == "all" and n != ns[0]:
            continue  # all pairs: n-independent, compute once
        for (rep, fold), recs in tasks.items():
            if m == "random" and args.unsup_random_draws:
                unsup_ok = lambda j: j < args.unsup_random_draws  # noqa: E731
            else:
                unsup_ok = lambda j: True  # noqa: E731
            for j, r in enumerate(recs):
                missing = []
                for clf in SUP:
                    ev = r.get("eval", {}).get(clf)
                    if ev is not None:
                        raw.append((m, n if m != "all" else 0, rep, fold, j, clf, ev["balanced_acc"]))
                    else:
                        missing.append(clf)
                S = None if m == "all" else r["selected"]
                per_task[(rep, fold)].append((m, n if m != "all" else 0, j, S, int(r["seed"]), missing, unsup_ok(j)))
    # unsup subset only where allowed: run two passes inside one worker call by flagging via need_unsup per item
    def job_items(its):
        sup_only = [(m, n, j, S, sd, mi) for (m, n, j, S, sd, mi, u) in its if (not u) and mi]
        both = [(m, n, j, S, sd, mi) for (m, n, j, S, sd, mi, u) in its if u]
        return sup_only, both

    calls = []
    for (rep, fold), its in sorted(per_task.items()):
        so, both = job_items(its)
        if so:
            calls.append((rep, fold, so, False))
        if not args.no_unsup and both:
            calls.append((rep, fold, both, True))
        elif both and any(mi for *_, mi in both):
            calls.append((rep, fold, [b for b in both if b[5]], False))
    log(f"{key}: {len(raw)} stored supervised values, {len(calls)} worker calls "
        f"({sum(len(c[2]) for c in calls if c[3])} unsupervised sets)")
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(worker)(key, str(om.RES), rep, fold, items, args.max_pixels, nu, args.kmeans_ninit)
        for (rep, fold, items, nu) in calls)
    for rr in res:
        raw.extend(rr)
    log(f"{key}: workers done in {time.time() - t0:.0f}s")
    return raw


def boot_ci(v, rng, B):
    v = np.asarray(v, dtype=float)
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bm = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def summarise(raw_rows, ns, B, seed=0):
    """raw rows: (dataset, method, n, repeat, fold, draw, algo, value) -> summary dict."""
    by = defaultdict(lambda: defaultdict(list))   # (ds, m, n, algo) -> {(rep,fold): [draw values]}
    for ds, m, n, rep, fold, draw, algo, v in raw_rows:
        by[(ds, m, n, algo)][(rep, fold)].append(v)
    rng = np.random.default_rng(seed)
    out = []
    for (ds, m, n, algo), tasks in sorted(by.items()):
        units = [float(np.mean(v)) for v in tasks.values()]
        mean, lo, hi = boot_ci(units, rng, B)
        if m == "all":
            for nn in ns:
                out.append((ds, nn, m, algo, mean, lo, hi, len(units)))
        else:
            out.append((ds, n, m, algo, mean, lo, hi, len(units)))
    return out


METRIC = {"knn": "BA", "lda": "BA", "svm": "BA", "rf": "BA", "kmeans": "ARI", "gmm": "ARI", "nmf": "matched BA"}


def write_csvs(out, raw_rows, summ, args):
    hdr = (f"# score_vs_n: mean over units (repeat,fold), 95% bootstrap CI ({args.boot} resamples). Unsupervised/NMF values: "
           f"fixed random subsample of <= {args.max_pixels} annotated pixels per (dataset,repeat,fold), seed 1000+100*repeat+fold; "
           f"kmeans n_init={args.kmeans_ninit}; random draws used for unsup: "
           f"{args.unsup_random_draws or 'all'}; RL policy seeds: {args.rl_seeds}; SA = repeat 0, seeds averaged.")
    with open(out / "score_vs_n_raw.csv", "w", newline="") as fh:
        fh.write(hdr.replace("mean over units", "RAW values; mean over units") + "\n")
        w = csv.writer(fh)
        w.writerow(["dataset", "method", "n", "repeat", "fold", "draw", "algo", "value"])
        for r in raw_rows:
            w.writerow(list(r[:7]) + [f"{r[7]:.8f}"])
    with open(out / "score_vs_n_summary.csv", "w", newline="") as fh:
        fh.write(hdr + "\n")
        w = csv.writer(fh)
        w.writerow(["dataset", "n", "method", "algo", "mean", "ci_lo", "ci_hi", "n_units"])
        for r in summ:
            w.writerow([r[0], r[1], r[2], r[3]] + [f"{v:.6f}" for v in r[4:7]] + [r[7]])


# ------------------------------------------------------------------------------------------------ figure
COL = {"all": "#555555", "rl_addremove": "#d62728", "rl_add": "#ff9a3c", "css": "#1f77b4", "random": "#8c8c8c", "sa": "#2ca02c"}
LS = {"rl_addremove": "-", "rl_add": "--", "css": "-", "random": ":", "sa": "-"}
MK = {"rl_addremove": "o", "rl_add": "s", "css": "^", "random": "x", "sa": "D"}
NAME = {"all": "All pairs", "rl_addremove": "RL add/remove", "rl_add": "RL add-only", "css": "CSS", "random": "Random", "sa": "SA (supervised search)"}
TITLE = {"lichens1": "Lichens 1", "lichens2": "Lichens 2", "lichens2_sub3": "Lichens 2 (3 cl.)", "collagen_acetic": "Collagen Acetic",
         "collagen_pepsin": "Collagen Pepsin"}
HDR = {"knn": "kNN (BA)", "lda": "LDA (BA)", "svm": "SVM (BA)", "rf": "RF (BA)", "kmeans": "k-means (ARI)", "gmm": "GMM (ARI)",
       "nmf": "NMF (BA)"}


def render(out, summ, datasets, ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8, "ytick.major.size": 1.8,
                         "xtick.major.pad": 1.2, "ytick.major.pad": 1.2, "pdf.fonttype": 42, "font.family": "sans-serif"})
    S = {(r[0], r[1], r[2], r[3]): r[4:7] for r in summ}
    nr, nc = len(datasets), len(ALGOS)
    W, L, R, TOPH, BOT = 7.16, 0.50, 0.04, 0.50, 0.34
    gx, gy = 0.40, 0.10
    pw = (W - L - R - (nc - 1) * gx) / nc
    ph = 0.95
    H = TOPH + nr * ph + (nr - 1) * gy + BOT
    fig = plt.figure(figsize=(W, H))
    xs = np.arange(len(ns))
    for i, ds in enumerate(datasets):
        for j, a in enumerate(ALGOS):
            ax = fig.add_axes([(L + j * (pw + gx)) / W, (H - TOPH - (i + 1) * ph - i * gy) / H, pw / W, ph / H])
            lo_all, hi_all = [], []
            if (ds, ns[0], "all", a) in S:
                m_, l_, h_ = S[(ds, ns[0], "all", a)]
                ax.axhspan(l_, h_, color=COL["all"], alpha=0.25, lw=0)
                ax.axhline(m_, color=COL["all"], lw=0.7)
                lo_all += [l_]; hi_all += [h_]
            dod = {m: (k - 2) * 0.07 for k, m in enumerate(METHOD_KEYS[1:])}
            for m in METHOD_KEYS[1:]:
                pts = [(x, S[(ds, n, m, a)]) for x, n in zip(xs, ns) if (ds, n, m, a) in S]
                if not pts:
                    continue
                px = np.array([p[0] for p in pts]) + dod[m]
                mu = np.array([p[1][0] for p in pts]); lo = np.array([p[1][1] for p in pts]); hi = np.array([p[1][2] for p in pts])
                ax.errorbar(px, mu, yerr=[mu - lo, hi - mu], color=COL[m], ls=LS[m], marker=MK[m], ms=2.4, lw=0.8, elinewidth=0.5,
                            capsize=1.0, capthick=0.4, mew=0.6)
                lo_all += list(lo); hi_all += list(hi)
            if lo_all:
                pad = 0.06 * (max(hi_all) - min(lo_all) + 1e-6)
                ax.set_ylim(min(lo_all) - pad, max(hi_all) + pad)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(n) for n in ns] if i == nr - 1 else [])
            ax.set_xlim(-0.4, len(ns) - 0.6)
            ax.locator_params(axis="y", nbins=4)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if i == 0:
                ax.set_title(HDR[a], fontsize=7, pad=2)
            if j == 0:
                fig.text(0.012, (H - TOPH - i * (ph + gy) - ph / 2) / H, TITLE.get(ds, ds), rotation=90, fontsize=7.5,
                         fontweight="bold", ha="center", va="center")
    fig.text((L + (W - L - R) / 2) / W, 0.07 / H, "number of selected pairs n", fontsize=7, ha="center", va="center")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(fc=COL["all"], alpha=0.5, label="All pairs (mean, 95% CI)")] + [
        Line2D([], [], color=COL[m], ls=LS[m], marker=MK[m], ms=3, lw=0.9, label=NAME[m]) for m in METHOD_KEYS[1:]]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=7, handlelength=1.6,
               columnspacing=0.9, handletextpad=0.4, borderpad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_score_vs_n.{ext}", dpi=300)
    plt.close(fig)
    return W, H


def read_raw(out):
    rows = []
    with open(out / "score_vs_n_raw.csv") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("dataset,"):
                continue
            r = line.rstrip("\n").split(",")
            rows.append((r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5]), r[6], float(r[7])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/rl_select")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--ns", default="3,5,8,13")
    ap.add_argument("--out", default="publications/rl_band_selection/figures/proto/")
    ap.add_argument("--max-pixels", type=int, default=30000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--rl-seeds", default="0", help="policy seeds to include (comma list) or 'all'")
    ap.add_argument("--unsup-random-draws", type=int, default=0, help="limit random draws used for clustering/NMF (0 = all)")
    ap.add_argument("--kmeans-ninit", type=int, default=3)
    ap.add_argument("--no-unsup", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    om.set_results_root(args.results_root)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    datasets = [k for k in args.datasets.split(",") if k]
    ns = [int(v) for v in args.ns.split(",")]
    T0 = time.time()
    log = lambda s: print(f"[{time.time() - T0:7.1f}s] {s}", flush=True)  # noqa: E731
    if args.render_only:
        raw_rows = read_raw(out)
    else:
        raw_rows = []
        for key in datasets:
            raw_rows += [(key,) + r for r in run_dataset(key, args, log)]
            # checkpoint after each dataset
            write_csvs(out, raw_rows, summarise(raw_rows, ns, args.boot), args)
    summ = summarise(raw_rows, ns, args.boot)
    write_csvs(out, raw_rows, summ, args)
    w, h = render(out, summ, datasets, ns)
    log(f"saved fig_score_vs_n ({w:.2f} x {h:.2f} in)")


if __name__ == "__main__":
    main()
