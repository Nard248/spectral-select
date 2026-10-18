"""Aggregate rl_select result jsonl (S1/S2/S3) into summary CSVs + a markdown digest.

Generic over methods (RL rows with the same schema, e.g. rl_add / rl_addremove*, are picked up automatically).
Task-level protocol: rows are first averaged over seeds within a task (repeat, fold), then mean / std / 95%
bootstrap CI are taken across tasks.  Output: results/rl_select/summary/.

  PYTHONPATH=src python experiments/rl_select/aggregate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "rl_select"
OUT = RES / "summary"
NFEAT = {"lichens1": 192, "lichens2": 210, "collagen_acetic": 149, "collagen_pepsin": 158}
LABEL_FREE = ("random", "variance", "uniform_grid", "spa", "css_greedy", "css_greedy_pc", "mmr_var")
METRICS = ["knn_ba", "knn_f1", "lda_ba", "svm_ba", "rf_ba", "inner_J", "n_distinct_exc", "R_corr", "jrecon",
           "n_evals", "runtime_s"]
rng = np.random.default_rng(0)


def load() -> pd.DataFrame:
    recs = []
    for s in ("S1", "S2", "S3"):
        for f in sorted((RES / s).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                e, sm = r.get("eval", {}), r.get("set_metrics", {})
                g = lambda c, k: e.get(c, {}).get(k, np.nan)
                recs.append(dict(
                    setting=r["setting"], dataset=r["dataset"], repeat=r["repeat"], fold=r["fold"], n=r["n"],
                    method=r["method"], seed=r["seed"], budget=r["budget"], objective=r["objective_name"],
                    selected=tuple(r["selected"]), trace=r.get("trace", []),
                    knn_ba=g("knn", "balanced_acc"), knn_f1=g("knn", "macro_f1"), lda_ba=g("lda", "balanced_acc"),
                    svm_ba=g("svm", "balanced_acc"), rf_ba=g("rf", "balanced_acc"),
                    inner_J=r["inner_J"] if r["inner_J"] is not None else np.nan,
                    n_distinct_exc=sm.get("n_distinct_excitations", np.nan), R_corr=sm.get("mean_abs_corr", np.nan),
                    jrecon=sm.get("jrecon", np.nan), n_evals=r["n_evals"], runtime_s=r["runtime_s"]))
    return pd.DataFrame(recs)


def boot_ci(x: np.ndarray, B: int = 2000) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def task_level(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average over seeds inside each task."""
    return df.groupby(keys + ["repeat", "fold"], as_index=False)[METRICS].mean()


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    tl = task_level(df, keys)
    rows = []
    for k, g in tl.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k)); row["n_tasks"] = len(g)
        for m in METRICS:
            v = g[m].to_numpy(float)
            row[f"{m}_mean"], row[f"{m}_std"] = np.nanmean(v) if (~np.isnan(v)).any() else np.nan, \
                np.nanstd(v, ddof=1) if (~np.isnan(v)).sum() > 1 else np.nan
            if m in ("knn_ba", "knn_f1", "lda_ba", "svm_ba", "rf_ba", "inner_J"):
                row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = boot_ci(v)
        rows.append(row)
    return pd.DataFrame(rows)


def stability_table(df: pd.DataFrame) -> pd.DataFrame:
    from rl_select.evaluate import stability
    rows = []
    d = df[df.n > 0]
    for (s, ds, n, m, b), g in d.groupby(["setting", "dataset", "n", "method", "budget"]):
        vals = []
        for (_, sd), gg in g.groupby(["repeat", "seed"]):
            if len(gg) >= 2 and gg.fold.nunique() == len(gg):
                vals.append(stability([list(x) for x in gg.selected], NFEAT[ds]))
        if vals:
            rows.append(dict(setting=s, dataset=ds, n=n, method=m, budget=b,
                             jaccard=np.mean([v["jaccard"] for v in vals]), kuncheva=np.mean([v["kuncheva"] for v in vals]),
                             n_groups=len(vals)))
    return pd.DataFrame(rows)


def anytime(df: pd.DataFrame) -> pd.DataFrame:
    """Best-so-far inner J vs uncached evals from S1 traces (mean over tasks and seeds)."""
    grid = np.unique(np.concatenate([np.round(np.logspace(0, np.log10(5000), 40)).astype(int), [25, 50, 100, 200, 400, 1000, 5000]]))
    d = df[(df.setting == "S1") & (df.budget > 0) & (df.objective == "jsup")]
    recs = []
    for r in d.itertuples():
        tr = r.trace
        if not tr:
            continue
        ev = np.array([t[0] for t in tr]); v = np.array([t[1] for t in tr])
        for g in grid:
            i = np.searchsorted(ev, g, side="right") - 1
            recs.append((r.dataset, r.n, r.method, r.seed, r.repeat, r.fold, g, v[i] if i >= 0 else np.nan))
    a = pd.DataFrame(recs, columns=["dataset", "n", "method", "seed", "repeat", "fold", "evals", "best_J"])
    return a.groupby(["dataset", "n", "method", "evals"], as_index=False).best_J.mean() if len(a) else a


def exact_gap(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.setting == "S1") & (df.objective == "jsup")]
    ex = d[d.method == "exact"].groupby(["dataset", "n", "repeat", "fold"], as_index=False).inner_J.first()
    ex = ex.rename(columns={"inner_J": "exact_J"})
    o = d[(d.method != "exact") & ((d.budget > 0) | d.method.isin(["fisher", "mrmr"]))]
    o = o.groupby(["dataset", "n", "method", "repeat", "fold"], as_index=False).inner_J.mean().merge(
        ex, on=["dataset", "n", "repeat", "fold"])
    o["gap"] = o.exact_J - o.inner_J
    o["ratio"] = o.inner_J / o.exact_J
    return o.groupby(["dataset", "n", "method"], as_index=False).agg(
        n_tasks=("gap", "size"), inner_J=("inner_J", "mean"), exact_J=("exact_J", "mean"), gap=("gap", "mean"),
        ratio=("ratio", "mean"))


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.objective == "jsup") & df.inner_J.notna() & (df.n > 0)]
    tl = task_level(d, ["setting", "dataset", "n", "method", "budget"])
    tl["gap_knn"] = tl.inner_J - tl.knn_ba
    tl["gap_lda"] = tl.inner_J - tl.lda_ba
    return tl.groupby(["setting", "dataset", "n", "method", "budget"], as_index=False).agg(
        inner_J=("inner_J", "mean"), knn_ba=("knn_ba", "mean"), lda_ba=("lda_ba", "mean"),
        gap_knn=("gap_knn", "mean"), gap_lda=("gap_lda", "mean"))


def md(df: pd.DataFrame, floatfmt=".3f") -> str:
    """Markdown table (no tabulate dependency)."""
    df = df.reset_index(drop=True)
    cell = lambda x: format(x, floatfmt) if isinstance(x, (float, np.floating)) else str(x)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def pm(s: pd.DataFrame, m: str = "knn_ba") -> pd.Series:
    return s[f"{m}_mean"].map("{:.3f}".format) + " ± " + s[f"{m}_std"].map("{:.3f}".format)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    if df.empty:
        raise SystemExit("no results found")
    ap = df[df.method == "all_pairs"]
    df = df[df.method != "all_pairs"]
    out = {}
    for s in ("S1", "S2"):
        d = df[df.setting == s]
        if len(d):
            out[s] = summarize(d, ["dataset", "n", "method"]); out[s].to_csv(OUT / f"{s}_summary.csv", index=False)
    d3 = df[df.setting == "S3"]
    if len(d3):
        out["S3"] = summarize(d3, ["dataset", "n", "method", "budget"]); out["S3"].to_csv(OUT / "S3_summary.csv", index=False)
    if len(ap):
        apS = summarize(ap.assign(n=0, method="all_pairs"), ["dataset", "n", "method"])
        apS.to_csv(OUT / "all_pairs_reference.csv", index=False)
    st = stability_table(df); st.to_csv(OUT / "stability.csv", index=False)
    gp = gap_table(df); gp.to_csv(OUT / "inner_outer_gap.csv", index=False)
    eg = exact_gap(df); eg.to_csv(OUT / "exact_gap.csv", index=False)
    an = anytime(df); an.to_csv(OUT / "anytime_curves.csv", index=False)

    # ---------------- digest
    L = ["# rl_select baselines digest", "", "Outer test, mean ± std across outer-fold tasks (seeds averaged within task). "
         "kNN k=5 balanced accuracy unless stated.", ""]
    if len(ap):
        L += ["## All-pairs reference (outer kNN BA / LDA / SVM)", "",
              md(apS[["dataset"]].assign(kNN=pm(apS), LDA=pm(apS, "lda_ba"), SVM=pm(apS, "svm_ba"))), ""]
    if "S2" in out:
        S2 = out["S2"]
        L += ["## S2 label-free + reference methods, outer kNN BA (repeats 0-2)", ""]
        for ds, g in S2.groupby("dataset"):
            t = g.assign(v=pm(g)).pivot(index="method", columns="n", values="v")
            L += [f"### {ds}", "", md(t.reset_index()), ""]
    if "S1" in out:
        S1 = out["S1"]
        L += ["## S1 search quality (B=5000, repeat 0), outer kNN BA", ""]
        for ds, g in S1.groupby("dataset"):
            t = g.assign(v=pm(g)).pivot(index="method", columns="n", values="v")
            L += [f"### {ds}", "", md(t.reset_index()), ""]
        L += ["## S1 outer LDA BA at n=8", "", md(S1[S1.n == 8].assign(v=pm(S1[S1.n == 8], "lda_ba")).pivot(
            index="method", columns="dataset", values="v").reset_index()), ""]
    if "S3" in out:
        S3 = out["S3"]
        L += ["## S3 small budgets, outer kNN BA (n=8)", ""]
        g = S3[S3.n == 8]
        for ds, gg in g.groupby("dataset"):
            t = gg.assign(v=pm(gg)).pivot(index="method", columns="budget", values="v")
            L += [f"### {ds} (n=8)", "", md(t.reset_index()), ""]
        t = S3.groupby(["method", "budget"], as_index=False).knn_ba_mean.mean().pivot(index="method", columns="budget",
                                                                                         values="knn_ba_mean")
        L += ["### pooled over datasets and n (mean outer kNN BA)", "", md(t.reset_index()), ""]
    if len(gp):
        g = gp[(gp.setting == "S1") & (gp.budget > 0)].groupby(["method", "n"], as_index=False)[["inner_J", "knn_ba", "lda_ba", "gap_knn", "gap_lda"]].mean()
        L += ["## Inner-J vs outer BA gap (S1, mean over datasets; gap = inner J - outer BA)", "", md(g), ""]
    if len(eg):
        g = eg.groupby(["n", "method"], as_index=False)[["inner_J", "exact_J", "gap", "ratio"]].mean()
        L += ["## Gap to exact optimum (inner J; exact enumerated for n=2,3; only n=3 overlaps other methods)", "", md(g, ".4f"), ""]
        ex = out["S1"][out["S1"].method == "exact"] if "S1" in out else None
        if ex is not None and len(ex):
            L += ["### exact: runtime and outer BA", "", md(ex[["dataset", "n", "n_tasks", "inner_J_mean", "knn_ba_mean", "runtime_s_mean"]]), ""]
    if len(st):
        g = st[st.setting == "S2"].groupby(["method", "n"], as_index=False)[["jaccard", "kuncheva"]].mean()
        t = g.pivot(index="method", columns="n", values="kuncheva")
        L += ["## Stability across outer folds (S2, Kuncheva, mean over datasets/repeats/seeds)", "", md(t.reset_index()), ""]
    if len(an):
        pts = an[an.evals.isin([25, 50, 100, 200, 400, 1000, 5000]) | (an.evals == an.evals.max())]
        t = pts.groupby(["method", "evals"], as_index=False).best_J.mean().pivot(index="method", columns="evals", values="best_J")
        L += ["## Anytime: mean best-so-far inner J vs uncached evals (S1 traces, pooled over datasets and n)", "",
              md(t.reset_index(), ".4f"), ""]
    (OUT / "digest_baselines.md").write_text("\n".join(L))
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
