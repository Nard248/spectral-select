"""Paper figures, LaTeX tables and paired statistics for the reversible-RL band-selection study.

Re-runnable: reads whatever exists under results/rl_select/{S1,S2,S3,S4}/*.jsonl (incl. *_rl_v3.jsonl).

  PYTHONPATH=src python experiments/rl_select/paper_figures.py

Writes PDFs to publications/rl_band_selection/figures/ and .tex/.csv/stats_summary.md to .../tables/.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_select import paths
from rl_select.stats import holm, stats_paired

ROOT = paths.ROOT
RES = paths.results_root()      # env RL_SELECT_RESULTS (v2: results/rl_select_v2)
PUB = paths.paper_out()         # legacy root -> publications/rl_band_selection, else <root>/paper
FIG = PUB / "figures"
TAB = PUB / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DS = paths.datasets()
DSN = {"lichens1": "Lichens 1", "lichens2": "Lichens 2 (full)", "lichens2_sub3": "Lichens 2", "collagen_acetic": "Collagen-AA",
       "collagen_pepsin": "Collagen-P"}
CROSS = paths.cross_of(DS)
NS = (3, 5, 8, 13)
W1, W2 = 3.5, 7.16

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "font.family": "sans-serif", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.4, "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Okabe-Ito, consistent across figures.
STY = {
    "random": dict(c="#7f7f7f", m="x", ls="-", lab="random"),
    "css_greedy": dict(c="#E69F00", m="s", ls="-", lab="CSS-greedy"),
    "uniform_grid": dict(c="#56B4E9", m="^", ls="-", lab="uniform grid"),
    "rl_add_nowl_k16": dict(c="#009E73", m="o", ls="-", lab="RL add-only"),
    "rl_addremove_nowl_k16": dict(c="#D55E00", m="D", ls="-", lab="RL add/remove"),
    "v3": dict(c="#CC79A7", m="*", ls="-", lab="RL add/remove + aug"),
    "sfs_swap": dict(c="k", m=None, ls="--", lab="SFS+swap (supervised)"),
    "rl_add_bok_nowl": dict(c="#009E73", m="o", ls="-", lab="RL add-only (best-of-$k$)"),
    "rl_addremove_mc_nowl": dict(c="#D55E00", m="D", ls="-", lab="RL add/remove (+MC)"),
    "random_search": dict(c="#7f7f7f", m="x", ls="-", lab="random search"),
    "sa": dict(c="#0072B2", m="v", ls="-", lab="SA"),
    "ga": dict(c="#332288", m="<", ls="-", lab="GA"),
    "rl_add": dict(c="#009E73", m="o", ls="-", lab="RL add-only"),
    "rl_addremove": dict(c="#D55E00", m="D", ls="-", lab="RL add/remove"),
}


# ----------------------------------------------------------------------------- data
def load_all() -> dict[str, list[dict]]:
    out = {}
    for s in ("S1", "S2", "S3", "S4"):
        rows, seen = [], {}
        for f in sorted((RES / s).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m, src = (r["method"].split("@", 1) + [""])[:2] if "@" in r["method"] else (r["method"], "")
                r["base"], r["src"] = m, src
                k = (r["method"], r["task"], r["n"], r["seed"], r["budget"])
                seen[k] = r  # last write wins (dedupe re-runs)
        out[s] = list(seen.values())
    return out


D = load_all()


def ba(r, clf="knn"):
    e = r.get("eval", {}).get(clf)
    return None if not e else e.get("balanced_acc")


def unit_vals(rows, key=lambda r: (r["dataset"], r["repeat"], r["fold"], r["n"]), clf="knn"):
    """{unit: seed-averaged BA}."""
    acc = defaultdict(list)
    for r in rows:
        v = ba(r, clf)
        if v is not None:
            acc[key(r)].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def sel(setting, base=None, src=None, dataset=None, n=None, budget=None):
    return [r for r in D[setting]
            if (base is None or r["base"] == base) and (src is None or r["src"] == src)
            and (dataset is None or r["dataset"] == dataset) and (n is None or r["n"] == n)
            and (budget is None or r["budget"] == budget)]


def boot_ci(x, nb=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(nb, len(x)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def paired(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    return ks, np.array([a[k] for k in ks]), np.array([b[k] for k in ks])


# ----------------------------------------------------------------------------- best v3 variant
V3_RE = re.compile(r"^rl_addremove_.*aug")


def pick_v3():
    """Best v3 add/remove(+aug) variant (cross-family source), by mean kNN BA over all targets x n with full coverage."""
    cand = defaultdict(lambda: defaultdict(list))
    for r in D["S2"]:
        if V3_RE.match(r["base"]) and r["src"] == CROSS.get(r["dataset"]) and ba(r) is not None:
            cand[r["base"]][(r["dataset"], r["n"])].append(ba(r))
    if not cand:
        return None
    best, bs = None, -1
    full = max(len(c) for c in cand.values())
    for m, c in cand.items():
        if len(c) < full:
            continue
        s = np.mean([np.mean(v) for v in c.values()])
        if s > bs:
            best, bs = m, s
    return best


V3 = pick_v3()
if V3:
    STY["v3"]["lab"] = "RL add/remove + aug" + (" (CSS init)" if "css" in V3 else "")
print("best v3 variant:", V3)


def s2_units(base, ds, n, src=None):
    src = CROSS[ds] if src is None else src
    return unit_vals(sel("S2", base, src, ds, n))


def s2_free(base, ds, n):  # label-free baselines have no source tag
    return unit_vals(sel("S2", base, "", ds, n))


def s2_cell(name, ds, n):
    """Units for a method name, handling baselines / RL (cross-family source) / v3 alias."""
    if name == "v3":
        return s2_units(V3, ds, n) if V3 else {}
    if name.startswith("rl_"):
        return s2_units(name, ds, n)
    return s2_free(name, ds, n)


# ----------------------------------------------------------------------------- Fig A
def fig_transfer():
    fig, axs = plt.subplots(1, len(DS), figsize=(W2, 2.2), sharey=False)
    methods = ["random", "css_greedy", "uniform_grid", "rl_add_nowl_k16", "rl_addremove_nowl_k16"]
    if V3:
        methods.append("v3")
    methods.append("sfs_swap")
    for ax, ds in zip(axs, DS):
        for k, mth in enumerate(methods):
            st = STY[mth]
            xs, ms, lo, hi = [], [], [], []
            for n in NS:
                u = s2_cell(mth, ds, n)
                if not u:
                    continue
                m, l, h = boot_ci(list(u.values()))
                xs.append(n); ms.append(m); lo.append(l); hi.append(h)
            if not xs:
                continue
            off = (k - (len(methods) - 1) / 2) * 0.09
            xx = np.array(xs) + off
            ax.errorbar(xx, ms, yerr=[np.array(ms) - lo, np.array(hi) - ms], color=st["c"], ls=st["ls"],
                        marker=st["m"], capsize=1.2, elinewidth=0.6, mew=0.9 if st["m"] == "x" else 0.3,
                        zorder=3 if "rl_" in mth or mth == "v3" else 2, ms=5.5 if st["m"] == "*" else 3.6,
                        label=st["lab"])
        ax.set_xticks(NS)
        ax.set_xlabel("selected pairs $n$")
        ax.set_title(f"({'abcd'[DS.index(ds)]}) {DSN[ds]}", loc="left", fontsize=7.5, fontweight="bold", pad=3)
    axs[0].set_ylabel("outer kNN BA")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.07),
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig_transfer.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig B
FEW = ["rl_add_bok_nowl", "rl_addremove_mc_nowl", "random_search", "sa", "ga", "sfs_swap"]
BUD = (25, 50, 100, 200, 400)


def s3_units(base, ds, n, B):
    """Row-level units (fold x seed) -> kNN BA."""
    src = "lodo" if base.startswith("rl_") else ""
    out = {}
    for r in sel("S3", base, src, ds, n, B):
        v = ba(r)
        if v is not None:
            out[(r["dataset"], r["repeat"], r["fold"], r["n"], r["seed"])] = v
    return out


def fig_fewshot():
    fig, axs = plt.subplots(1, len(DS), figsize=(W2, 2.2))
    for ax, ds in zip(axs, DS):
        for k, mth in enumerate(FEW):
            st = STY[mth]
            xs, ms, lo, hi = [], [], [], []
            for B in BUD:
                u = s3_units(mth, ds, 8, B)
                if not u:
                    continue
                m, l, h = boot_ci(list(u.values()))
                xs.append(B); ms.append(m); lo.append(l); hi.append(h)
            if not xs:
                continue
            xx = np.array(xs) * 1.045 ** (k - (len(FEW) - 1) / 2)
            ax.errorbar(xx, ms, yerr=[np.array(ms) - lo, np.array(hi) - ms], color=st["c"], ls=st["ls"],
                        marker=st["m"], capsize=1.2, elinewidth=0.6, mew=0.9 if st["m"] == "x" else 0.3,
                        zorder=3 if mth.startswith("rl_") else 2, label=st["lab"])
        ax.set_xscale("log")
        ax.set_xticks(BUD)
        ax.set_xticklabels([str(b) for b in BUD])
        ax.minorticks_off()
        ax.set_xlabel("evaluation budget $B$")
        ax.set_title(f"({'abcd'[DS.index(ds)]}) {DSN[ds]}", loc="left", fontsize=7.5, fontweight="bold", pad=3)
    axs[0].set_ylabel("outer kNN BA")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(FEW), frameon=False, bbox_to_anchor=(0.5, -0.07),
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig_fewshot.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig C
def training_curves():
    runs = defaultdict(list)  # variant -> list of (pool, seed, J array)
    for d in sorted((RES / "policies").glob("*")):
        f = d / "train_log.csv"
        if not f.exists() or "_rl_" not in d.name:
            continue
        pool, rest = d.name.split("_rl_", 1)
        m = re.match(r"^(.*)_s(\d+)$", rest)
        if not m:
            continue
        var = "rl_" + m.group(1)
        if var not in ("rl_add", "rl_addremove", "rl_add_nowl", "rl_addremove_nowl"):
            continue
        with f.open() as fh:
            J = [float(r["mean_final_J"]) for r in csv.DictReader(fh)]
        runs[var].append((pool, int(m.group(2)), np.array(J)))
    return runs


def smooth(x, w=9):
    k = np.ones(w) / w
    xp = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.convolve(xp, k, mode="valid")


def anytime():
    """Normalised best-so-far inner J vs evaluations, pooled over datasets at one n."""
    rows = [r for r in D["S1"] if r["trace"] and r["objective_name"] == "jsup" and r["budget"] == 5000
            and r["base"] in ("random_search", "sfs_swap", "sa", "ga", "rl_add", "rl_addremove")]
    cnt = defaultdict(set)
    for r in rows:
        cnt[r["n"]].add((r["dataset"], r["fold"], r["base"]))
    n = max((nn for nn in cnt if nn in (5, 13)), key=lambda nn: (len(cnt[nn]), nn))
    rows = [r for r in rows if r["n"] == n]
    tasks = defaultdict(list)
    for r in rows:
        tasks[(r["dataset"], r["fold"])].append(r)
    grid = np.unique(np.round(np.logspace(0, np.log10(5000), 120)).astype(int))
    curves = defaultdict(list)

    def step(trace):
        t = np.array(trace, float)
        out = np.full(len(grid), np.nan)
        for i, g in enumerate(grid):
            idx = np.searchsorted(t[:, 0], g, side="right") - 1
            if idx >= 0:
                out[i] = t[idx, 1]
        return out

    for key, rs in tasks.items():
        best = max(r["inner_J"] for r in rs)
        per = defaultdict(list)
        for r in rs:
            per[r["base"]].append(step(r["trace"]) / best)
        for m, c in per.items():
            curves[m].append(np.nanmean(c, 0) if len(c) > 1 else c[0])
    return n, grid, curves


def fig_training():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6))
    ax = axs[0]
    runs = training_curves()
    spec = [("rl_add_nowl", "add", "#009E73", "-"), ("rl_addremove_nowl", "add/rem", "#D55E00", "-"),
            ("rl_add", "add (+$\\lambda$)", "#009E73", "--"), ("rl_addremove", "add/rem (+$\\lambda$)", "#D55E00", "--")]
    for var, lab, c, ls in spec:
        rs = runs.get(var, [])
        if not rs:
            continue
        L = min(len(j) for _, _, j in rs)
        A = np.stack([smooth(j[:L]) for _, _, j in rs])
        # normalise each run by its own initial level so pools with different n-mix are comparable? keep raw.
        m, s = A.mean(0), A.std(0)
        x = np.arange(L)
        ax.plot(x, m, color=c, ls=ls, label=lab)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("mean final $J_{\\mathrm{sup}}$")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.5, handlelength=1.1, columnspacing=0.5, handletextpad=0.3)
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
    ax = axs[1]
    n, grid, curves = anytime()
    for m in ("random_search", "sfs_swap", "sa", "ga", "rl_add", "rl_addremove"):
        if m not in curves:
            continue
        A = np.stack(curves[m])
        mu = np.nanmean(A, 0)
        st = STY[m]
        ok = ~np.isnan(mu)
        ax.plot(grid[ok], mu[ok], color=st["c"], ls=st["ls"], label=st["lab"].replace(" (supervised)", "").replace("SFS+swap","SFS+sw."))
    ax.set_xscale("log")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best-so-far / best $J_{\\mathrm{sup}}$")
    ax.set_ylim(top=1.005)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.5, handlelength=1.1, columnspacing=0.5, handletextpad=0.3)
    ax.text(0.03, 0.97, f"(b) $n={n}$", transform=ax.transAxes, va="top", fontweight="bold")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig_training.pdf")
    plt.close(fig)
    return n


# ----------------------------------------------------------------------------- Fig D
def fig_eem():
    z = np.load(paths.cache_dir() / "lichens1.npz", allow_pickle=True)
    ex, em = z["ex_nm"], z["em_nm"].astype(float).copy()
    recorded = bool(json.loads(str(z["meta"])).get("info", {}).get("recorded_wavelengths", False))
    # plot on the emission wavelengths recorded in the .im3 headers (dataset audit 2026-10-22); the cached labels of
    # the 385-430 nm cubes are shifted. Feature identity (and thus every selection) is unchanged.
    if not recorded:   # legacy cache: shift to the recorded axis; recwl caches already carry it
        from me_attention.data import RECORDED_WL_OFFSET
        for e_nm, off in RECORDED_WL_OFFSET["lichens1"].items():
            em[ex == e_nm] += off
    uex, uem = np.unique(ex), np.unique(em)
    ie = np.searchsorted(uex, ex)
    im = np.searchsorted(uem, em)
    F = len(ex)
    panels = [("css_greedy", "S2", "", "CSS-greedy (label-free)"),
              ("rl_addremove_nowl_k16", "S2", "C2L", "RL add/remove (label-free)"),
              ("sfs_swap", "S2", "", "SFS+swap (supervised)"),
              ("sa", "S1", "", "SA (supervised)")]
    panels = [p for p in panels if sel(p[1], p[0], p[2], "lichens1", 8)]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 2.55), sharex=True, sharey=True)
    base = np.full((len(uex), len(uem)), np.nan)
    base[ie, im] = 0.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    for k, (ax, (m, s, src, lab)) in enumerate(zip(axs.ravel(), panels)):
        rs = sel(s, m, src, "lichens1", 8)
        cnt = np.zeros(F)
        for r in rs:
            for j in r["selected"]:
                cnt[j] += 1
        fr = cnt / len(rs)
        grid = np.full((len(uex), len(uem)), np.nan)
        grid[ie, im] = fr
        rgba = np.ones(base.shape + (4,))
        rgba[~np.isnan(base)] = (0.86, 0.86, 0.86, 1.0)
        pos = ~np.isnan(grid) & (grid > 0)
        rgba[pos] = cmap(np.clip(grid[pos], 0, 1))
        ax.imshow(rgba, origin="lower", aspect="auto", interpolation="nearest",
                  extent=(-0.5, len(uem) - 0.5, -0.5, len(uex) - 0.5))
        ax.set_yticks(range(len(uex)))
        ax.set_yticklabels([f"{int(v)}" for v in uex])
        step = 6
        ax.set_xticks(range(0, len(uem), step))
        ax.set_xticklabels([f"{int(uem[i])}" for i in range(0, len(uem), step)])
        ax.grid(False)
        ax.text(0.01, 1.02, f"({'abcd'[k]}) {lab}, {len(rs)} sets", transform=ax.transAxes, va="bottom",
                ha="left", fontsize=7.5, fontweight="bold")
    for ax in axs[1]:
        ax.set_xlabel("emission (nm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("excitation (nm)")
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.4, w_pad=0.5)
    pm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap)
    cb = fig.colorbar(pm, ax=axs, shrink=0.8, pad=0.015, aspect=30)
    cb.set_label("selection frequency", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG / "fig_eem_pairs.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- stats
TESTS = []  # dicts


def add_test(group, name, a: dict, b: dict):
    ks, x, y = paired(a, b)
    if len(ks) < 2:
        return
    s = stats_paired(x, y, 0.25)
    TESTS.append(dict(group=group, name=name, **s))
    return s


def run_stats():
    # ---- central: add/remove vs add-only, S2, all targets / sources / n
    variants = [("greedy, no-label feats", "_nowl", ""), ("k16, no-label feats", "_nowl", "_k16"),
                ("greedy, full feats", "", ""), ("k16, full feats", "", "_k16")]
    pooled = {"all": ({}, {})}
    for lab, wl, k in variants:
        A, B = {}, {}
        cells = defaultdict(lambda: ([], []))
        for ds in DS:
            for srcname in (CROSS[ds], "lodo"):
                for n in NS:
                    a = s2_units(f"rl_addremove{wl}{k}", ds, n, srcname)
                    b = s2_units(f"rl_add{wl}{k}", ds, n, srcname)
                    for key in set(a) & set(b):
                        kk = (ds, srcname, n) + key[1:3]
                        A[kk], B[kk] = a[key], b[key]
                        cells[(ds, srcname, n)][0].append(a[key]); cells[(ds, srcname, n)][1].append(b[key])
        add_test("central", f"add/remove vs add-only, pooled S2 [{lab}]", A, B)
        # cell-level (conservative): average within (target, source, n)
        Ac = {c: np.mean(v[0]) for c, v in cells.items()}
        Bc = {c: np.mean(v[1]) for c, v in cells.items()}
        add_test("central", f"add/remove vs add-only, cell-level means (target x source x n) [{lab}]", Ac, Bc)
        for kk in A:
            pooled["all"][0][(lab,) + kk] = A[kk]
            pooled["all"][1][(lab,) + kk] = B[kk]
        # by source type
        for st in ("cross-family", "lodo"):
            Ax = {kk: v for kk, v in A.items() if (kk[1] == "lodo") == (st == "lodo")}
            Bx = {kk: B[kk] for kk in Ax}
            add_test("central_by_source", f"add/remove vs add-only, {st} [{lab}]", Ax, Bx)
    add_test("central", "add/remove vs add-only, ALL four variants pooled", *pooled["all"])
    # ---- nowl vs full
    for base in ("rl_add", "rl_addremove"):
        for k in ("", "_k16"):
            A, B = {}, {}
            for ds in DS:
                for srcname in (CROSS[ds], "lodo"):
                    for n in NS:
                        a = s2_units(f"{base}_nowl{k}", ds, n, srcname)
                        b = s2_units(f"{base}{k}", ds, n, srcname)
                        for key in set(a) & set(b):
                            kk = (ds, srcname, n) + key[1:3]
                            A[kk], B[kk] = a[key], b[key]
            add_test("nowl_vs_full", f"{base}{k}: no-label feats vs full feats (pooled S2)", A, B)
    # ---- S3: mc vs bok per budget, RL vs sa
    for Bv in BUD:
        A, Bd = {}, {}
        for ds in DS:
            for n in NS:
                a = s3_units("rl_addremove_mc_nowl", ds, n, Bv)
                b = s3_units("rl_add_bok_nowl", ds, n, Bv)
                for k in set(a) & set(b):
                    A[k], Bd[k] = a[k], b[k]
        add_test("S3_mc_vs_bok", f"rl_addremove_mc vs rl_add_bok, B={Bv} (pooled datasets, n; fold x seed)", A, Bd)
    for Bv in (25, 400):
        for rl in ("rl_addremove_mc_nowl", "rl_add_bok_nowl"):
            A, Bd = {}, {}
            for ds in DS:
                for n in NS:
                    # seed-average RL and sa within (dataset, n, fold)
                    def avg(base):
                        acc = defaultdict(list)
                        for k, v in s3_units(base, ds, n, Bv).items():
                            acc[k[:4]].append(v)
                        return {k: np.mean(v) for k, v in acc.items()}
                    a, b = avg(rl), avg("sa")
                    for k in set(a) & set(b):
                        A[k], Bd[k] = a[k], b[k]
            add_test("S3_rl_vs_sa", f"{rl} vs sa, B={Bv} (pooled datasets, n; seed-averaged fold units)", A, Bd)
    # ---- T1 tests (Holm over table)
    t1 = []
    for ds in DS:
        for n in (3, 8, 13):
            ar = s2_cell("rl_addremove_nowl_k16", ds, n)
            ad = s2_cell("rl_add_nowl_k16", ds, n)
            cs = s2_cell("css_greedy", ds, n)
            t1.append(((ds, n, "vs add"), add_test("T1", f"{ds} n={n}: add/remove vs add-only (k16 nowl, {CROSS[ds]})", ar, ad)))
            t1.append(((ds, n, "vs css"), add_test("T1", f"{ds} n={n}: add/remove (k16 nowl, {CROSS[ds]}) vs css_greedy", ar, cs)))
    # Holm within group
    groups = defaultdict(list)
    for i, t in enumerate(TESTS):
        groups[t["group"]].append(i)
    for g, idx in groups.items():
        h = holm([TESTS[i]["wilcoxon_p"] for i in idx])
        for i, hp in zip(idx, h):
            TESTS[i]["holm_p"] = hp
    return t1


def fmt_p(p):
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0") if p < 1 else "1.00"


def write_stats_md():
    L = ["# Paired statistics (auto-generated by experiments/rl_select/paper_figures.py)", "",
         "Wilcoxon signed-rank (two-sided) on paired units; mean diff = first minus second (kNN BA); CI = bootstrap 95% of the "
         "mean paired difference; W/L/T = units where first > / < / = second. Holm correction applied within each group. "
         "Units are seed-averaged repeat x fold (S2) or fold x seed (S3) and are NOT independent across targets/n/variants: "
         "pooled p-values are optimistic, so cell-level rows (one mean per target x source x n cell) are given as a conservative check.", ""]
    cur = None
    order = list(dict.fromkeys(t["group"] for t in TESTS))
    TESTS.sort(key=lambda t: order.index(t["group"]))
    for t in TESTS:
        if t["group"] != cur:
            cur = t["group"]
            L += ["", f"## {cur}", "", "| test | n units | mean diff | 95% CI | Wilcoxon p | Holm p | W/L/T |", "|---|---|---|---|---|---|---|"]
        L.append(f"| {t['name']} | {t['n']} | {t['mean_diff']:+.4f} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | "
                 f"{t['wilcoxon_p']:.2e} | {t['holm_p']:.2e} | {t['wins']}/{t['losses']}/{t['ties']} |")
    (TAB / "stats_summary.md").write_text("\n".join(L) + "\n")
    with (TAB / "stats_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "test", "n", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "holm_p", "wins", "losses", "ties"])
        for t in TESTS:
            w.writerow([t["group"], t["name"], t["n"], t["mean_diff"], t["ci95"][0], t["ci95"][1], t["wilcoxon_p"],
                        t["holm_p"], t["wins"], t["losses"], t["ties"]])


# ----------------------------------------------------------------------------- tables
LF_OTHER = ("variance", "uniform_grid", "spa", "css_greedy_pc", "mmr_var", "fisher_free")  # label-free alternatives
LF_OTHER = ("variance", "uniform_grid", "spa", "css_greedy_pc", "mmr_var")
NAME_SHORT = {"variance": "var", "uniform_grid": "grid", "spa": "SPA", "css_greedy_pc": "CSS-PC", "mmr_var": "MMR"}


def tab_transfer(t1):
    pmap = {k: s for k, s in t1 if s}
    hp = {}
    tl = [i for i, t in enumerate(TESTS) if t["group"] == "T1"]
    for i in tl:
        hp[TESTS[i]["name"]] = TESTS[i]["holm_p"]
    hdr = [r"\begin{tabular}{llccccccrr}", r"\toprule",
           r"Target & $n$ & Random & CSS & Best other & RL add & RL add/rem & " + (r"RL +aug" if V3 else r"RL +aug") +
           r" & $p_{\mathrm{add}}$ & $p_{\mathrm{CSS}}$ \\", r"\midrule"]
    out, csvrows = list(hdr), [["target", "n", "random", "css_greedy", "best_other", "best_other_name", "rl_add_nowl_k16",
                                "rl_addremove_nowl_k16", "v3", "p_vs_add_holm", "p_vs_css_holm"]]
    for ds in DS:
        for j, n in enumerate((3, 8, 13)):
            vals = {}
            for nm in ("random", "css_greedy", "rl_add_nowl_k16", "rl_addremove_nowl_k16", "v3"):
                u = s2_cell(nm, ds, n)
                vals[nm] = np.mean(list(u.values())) if u else np.nan
            oth = {o: np.mean(list(s2_cell(o, ds, n).values())) for o in LF_OTHER if s2_cell(o, ds, n)}
            bo = max(oth, key=oth.get) if oth else None
            vals["other"] = oth[bo] if bo else np.nan
            cols = ["random", "css_greedy", "other", "rl_add_nowl_k16", "rl_addremove_nowl_k16", "v3"]
            top = np.nanmax([vals[c] for c in cols])
            cells = []
            for c in cols:
                v = vals[c]
                if np.isnan(v):
                    cells.append("--")
                    continue
                s = f"{v:.3f}"
                if round(v, 3) == round(top, 3):
                    s = r"\textbf{" + s + "}"
                if c == "other":
                    s += r"\,{\scriptsize(" + NAME_SHORT[bo] + ")}"
                cells.append(s)
            nm1 = f"{ds} n={n}: add/remove vs add-only (k16 nowl, {CROSS[ds]})"
            nm2 = f"{ds} n={n}: add/remove (k16 nowl, {CROSS[ds]}) vs css_greedy"

            def pf(nm, key):
                s = pmap.get((ds, n, key))
                if s is None:
                    return "--"
                sign = "$+$" if s["mean_diff"] > 0 else "$-$"
                return f"{sign}{fmt_p(hp[nm])}" + (r"$^{*}$" if hp[nm] < 0.05 else "")
            out.append((DSN[ds] if j == 0 else "") + f" & {n} & " + " & ".join(cells) + " & " + pf(nm1, "vs add") +
                       " & " + pf(nm2, "vs css") + r" \\")
            csvrows.append([ds, n] + [vals[c] for c in ("random", "css_greedy", "other", "rl_add_nowl_k16",
                                                       "rl_addremove_nowl_k16", "v3")][0:2] + [vals["other"], bo,
                            vals["rl_add_nowl_k16"], vals["rl_addremove_nowl_k16"], vals["v3"], pf(nm1, "vs add"), pf(nm2, "vs css")])
        if ds != DS[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    tex = [r"{\small", r"\setlength{\tabcolsep}{3pt}"] + out + ["}"]
    (TAB / "tab_transfer.tex").write_text("\n".join(tex) + "\n")
    with (TAB / "tab_transfer.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(csvrows)


def tab_fewshot():
    Bs = (25, 100, 400)
    cols = FEW
    short = {"rl_add_bok_nowl": "RL add (BoK)", "rl_addremove_mc_nowl": "RL add/rem (MC)", "random_search": "Random",
             "sa": "SA", "ga": "GA", "sfs_swap": "SFS+swap"}
    out = [r"{\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llcccccc}", r"\toprule",
           r"Dataset & $B$ & " + " & ".join(short[c] for c in cols) + r" \\", r"\midrule"]
    rows_csv = [["dataset", "B"] + list(cols)]
    for ds in DS:
        for j, B in enumerate(Bs):
            v = []
            for c in cols:
                u = s3_units(c, ds, 8, B)
                v.append(np.mean(list(u.values())) if u else np.nan)
            top = np.nanmax(v) if not all(np.isnan(v)) else np.nan
            cells = ["--" if np.isnan(x) else (r"\textbf{%.3f}" % x if round(x, 3) == round(top, 3) else "%.3f" % x) for x in v]
            out.append((DSN[ds] if j == 0 else "") + f" & {B} & " + " & ".join(cells) + r" \\")
            rows_csv.append([ds, B] + v)
        if ds != DS[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", "}"]
    (TAB / "tab_fewshot.tex").write_text("\n".join(out) + "\n")
    with (TAB / "tab_fewshot.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows_csv)


def tab_search():
    s1 = [r for r in D["S1"] if r["objective_name"] == "jsup"]
    meths = [("rl_add", "RL add-only"), ("rl_addremove", "RL add/remove"), ("sfs_swap", "SFS+swap"),
             ("beam_swap", "Beam+swap"), ("sa", "SA"), ("ga", "GA"), ("random_search", "Random search")]

    def task_mean(base, n, field, budget=5000):
        per = defaultdict(lambda: defaultdict(list))  # ds -> task -> vals
        for r in s1:
            if r["base"] == base and r["n"] == n and r["budget"] == budget:
                v = r["inner_J"] if field == "J" else ba(r)
                if v is not None:
                    per[r["dataset"]][(r["repeat"], r["fold"])].append(v)
        dsm = [np.mean([np.mean(v) for v in t.values()]) for t in per.values()]
        return float(np.mean(dsm)) if dsm else np.nan, len(per)

    def exact_gap(base):
        ex = {}
        for r in s1:
            if r["base"] == "exact" and r["n"] == 3:
                ex[(r["dataset"], r["repeat"], r["fold"])] = r["inner_J"]
        g = defaultdict(list)
        for r in s1:
            if r["base"] == base and r["n"] == 3 and (r["dataset"], r["repeat"], r["fold"]) in ex:
                g[(r["dataset"], r["repeat"], r["fold"])].append(r["inner_J"])
        d = [ex[k] - np.mean(v) for k, v in g.items()]
        return (float(np.mean(d)), len(d)) if d else (np.nan, 0)

    ex3 = np.mean([r["inner_J"] for r in s1 if r["base"] == "exact" and r["n"] == 3]) if any(
        r["base"] == "exact" and r["n"] == 3 for r in s1) else np.nan
    out = [r"{\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}", r"\toprule",
           r" & \multicolumn{2}{c}{$n=5$} & \multicolumn{2}{c}{$n=13$} & $n=3$ \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}",
           r"Method & $J_{\mathrm{sup}}$ & kNN BA & $J_{\mathrm{sup}}$ & kNN BA & gap to exact \\", r"\midrule"]
    csvrows = [["method", "J5", "BA5", "J13", "BA13", "gap_exact_n3"]]
    for b, lab in meths:
        v = [task_mean(b, 5, "J")[0], task_mean(b, 5, "BA")[0], task_mean(b, 13, "J")[0], task_mean(b, 13, "BA")[0]]
        g, ng = exact_gap(b)
        cells = ["--" if np.isnan(x) else f"{x:.3f}" for x in v] + ["--" if np.isnan(g) else f"{g:.3f}"]
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
        csvrows.append([b] + v + [g])
    out.append(rf"\multicolumn{{6}}{{l}}{{\scriptsize Exact optimum at $n=3$: mean $J_{{\mathrm{{sup}}}}={ex3:.3f}$ (budget 5000 per method; mean over datasets).}} \\")
    # collagen_acetic ablation block (n=8, B=20000)
    abl = [("rl_add", "RL add-only"), ("rl_add_oracle", "\\quad + oracle prior"), ("rl_addremove", "RL add/remove"),
           ("rl_addremove_oracle", "\\quad + oracle prior"), ("rl_addremove_terminal", "\\quad terminal reward"),
           ("rl_addremove_reinforce", "\\quad REINFORCE"), ("rl_addremove_mc", "\\quad + MC look-ahead"),
           ("sa", "SA"), ("sfs_swap", "SFS+swap")]
    out += [r"\midrule", r"\multicolumn{6}{l}{\emph{Ablation: collagen (acetic), $n=8$, $B=20000$, 3 outer folds}} \\",
            r" & \multicolumn{2}{c}{$J_{\mathrm{sup}}$} & \multicolumn{2}{c}{kNN BA} & \\"]
    abl_rows = [r for r in D["S1"] if r["dataset"] == "collagen_acetic" and r["n"] == 8 and r["budget"] == 20000]
    for b, lab in abl:
        rs = [r for r in abl_rows if r["base"] == b]
        if not rs:
            continue
        J = [r["inner_J"] for r in rs]
        BA = [ba(r) for r in rs if ba(r) is not None]
        out.append(f"{lab} & \\multicolumn{{2}}{{c}}{{{np.mean(J):.3f}}} & \\multicolumn{{2}}{{c}}{{{np.mean(BA):.3f}}} & \\\\")
        csvrows.append(["ablation:" + b, np.mean(J), np.mean(BA), len(rs)])
    out += [r"\bottomrule", r"\end{tabular}", "}"]
    (TAB / "tab_search_ablation.tex").write_text("\n".join(out) + "\n")
    with (TAB / "tab_search_ablation.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(csvrows)


def tab_cost():
    rows = D["S4"]
    if not rows:
        return False
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mu = r.get("extra", {}).get("mu_hw")
        if mu is None:
            m = re.search(r"mu([0-9.]+)", r["method"])
            mu = float(m.group(1)) if m else None
        if mu is None:
            continue
        agg[(r["base"].split("@")[0], float(mu))]["ex"].append(r["set_metrics"]["n_distinct_excitations"])
        agg[(r["base"].split("@")[0], float(mu))]["ba"].append(ba(r))
        agg[(r["base"].split("@")[0], float(mu))]["J"].append(r["inner_J"])
    mus = sorted({k[1] for k in agg})
    out = [r"{\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{RL add/remove} & \multicolumn{3}{c}{SA} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"$\mu_{\mathrm{hw}}$ & \#exc. & kNN BA & $J$ & \#exc. & kNN BA & $J$ \\", r"\midrule"]
    csvrows = [["mu", "rl_exc", "rl_ba", "rl_J", "sa_exc", "sa_ba", "sa_J", "n_folds"]]
    for mu in mus:
        cells, c = [], [mu]
        for m in ("rl_addremove", "sa"):
            a = agg.get((m, mu))
            if not a:
                cells += ["--"] * 3
                c += [np.nan] * 3
                continue
            cells += [f"{np.mean(a['ex']):.2f}", f"{np.mean(a['ba']):.3f}", f"{np.mean(a['J']):.3f}"]
            c += [np.mean(a["ex"]), np.mean(a["ba"]), np.mean(a["J"])]
        nf = len(agg[("sa", mu)]["ex"]) if ("sa", mu) in agg else 0
        out.append(f"{mu:g} & " + " & ".join(cells) + r" \\")
        csvrows.append(c + [nf])
    out += [r"\bottomrule", r"\end{tabular}", "}"]
    (TAB / "tab_cost.tex").write_text("\n".join(out) + "\n")
    with (TAB / "tab_cost.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(csvrows)
    return True


if __name__ == "__main__":
    t1 = run_stats()
    write_stats_md()
    tab_transfer(t1)
    tab_fewshot()
    tab_search()
    print("cost table:", tab_cost())
    fig_transfer()
    fig_fewshot()
    print("anytime n =", fig_training())
    fig_eem()
    print("done; v3 =", V3)
