"""Analysis for the ME-attention study: tables, paired statistics, figures, LaTeX number macros.

    PYTHONPATH=src python -m me_attention.analysis --sweep final [--out publications/me_attention/generated]

Reads ``results/me_attention/<sweep>/<dataset>/<model>.jsonl`` (written by ``harness``), tolerates partial sweeps
(missing models / datasets / units; paired statistics use only the units present for both models) and writes

    <out>/tab_spectral.tex|md, tab_spatial.tex|md          main tables (balanced accuracy, mean +- std over units)
    <out>/tab_spectral_extra.tex|md, tab_spatial_extra...  macro-F1 and specimen-vote accuracy
    <out>/tab_stats.tex|md                                 paired statistics (headline + per-baseline + every "ours")
    <out>/numbers.tex                                      \\newcommand macros for headline numbers
    <out>/summary.json                                     machine-readable stats
    <out>/figures/*.pdf|png                                box, paired difference, CD/ranks, cost scatter, recall heatmap

Everything is derived from the records; nothing is hard-coded about which datasets / models exist.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import holm, stats_paired

RESULTS_ROOT = Path("results/me_attention")
DATASET_ORDER = ["collagen_pepsin", "lichens1", "lichens2", "lichens2_sub3", "collagen_acetic"]
DATASET_LABEL = {"collagen_pepsin": "Collagen Pepsin", "lichens1": "Lichens 1", "lichens2": "Lichens 2",
                 "lichens2_sub3": "Lichens 2 (3-class subset)", "collagen_acetic": "Collagen Acetic (dev)"}
DATASET_TAG = {"collagen_pepsin": "Pepsin", "lichens1": "LichensOne", "lichens2": "LichensTwo",
               "lichens2_sub3": "LichensTwoSub", "collagen_acetic": "Acetic"}
TRACKS = ["spectral", "spatial"]
MODEL_ORDER = ["lr", "linsvm", "rbfsvm", "rf", "et", "hgb", "knn", "mlp", "cnn1d", "spectralformer", "bandvit",
               "smartattention_v0", "meas_tokens", "axial", "ex_tokens", "ex_tokens_v2", "ex_tokens_lin", "hybrid_conv_attn",
               "patchmean_lr", "patchmean_rbfsvm", "patchmean_rf", "cnn2d", "cnn3d", "hybridsn",
               "spectralformer_patch", "ssftt", "spatial_attn"]
MODEL_LABEL = {"lr": "LR", "linsvm": "Linear SVM", "rbfsvm": "RBF-SVM", "rf": "RF", "et": "ET", "hgb": "HGB",
               "knn": "kNN", "mlp": "MLP", "cnn1d": "1D-CNN", "spectralformer": "SpectralFormer",
               "bandvit": "Band-ViT", "smartattention_v0": "SmartAttn v0", "meas_tokens": "ME-Attn (meas.)",
               "axial": "ME-Attn (axial)", "ex_tokens": "ME-Attn (ex-tok.)", "ex_tokens_lin": "ex tokens + linear", "spatial_attn": "ME-Attn (spatial)",
               "patchmean_lr": "Patch-mean LR", "patchmean_rbfsvm": "Patch-mean SVM", "patchmean_rf": "Patch-mean RF",
               "cnn2d": "2D-CNN", "cnn3d": "3D-CNN", "hybridsn": "HybridSN", "spectralformer_patch": "SpectralFormer-P",
               "ssftt": "SSFTT"}
BASE_COLORS = ["#4C78A8", "#7F7F7F", "#9ECAE9", "#2B5C8A", "#B0B0B0", "#6B8EAD", "#555555", "#C5DCEF",
               "#3A6EA5", "#999999", "#7FA8CF", "#8DA0B3"]
OURS_COLORS = ["#D55E00", "#E69F00", "#B2182B", "#F4A582", "#8C2D04", "#CC6600"]
IEEE_COL, IEEE_2COL = 3.5, 7.16


def label(m: str) -> str:
    return MODEL_LABEL.get(m, m.replace("_", " "))


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


# =============================================================================================== loading
def load_sweep(sweep: str, root: Path = RESULTS_ROOT) -> tuple[pd.DataFrame, dict]:
    """Tidy DataFrame of successful, non-tiny records (deduplicated on dataset/model/repeat/fold, last wins)."""
    rows, info = OrderedDict(), dict(n_files=0, n_errors=0, n_tiny=0, n_torn=0, n_dups=0, errors=[])
    base = Path(root) / sweep
    for f in sorted(base.glob("*/*.jsonl")):
        info["n_files"] += 1
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                info["n_torn"] += 1
                continue
            if "error" in r:
                info["n_errors"] += 1
                info["errors"].append(f"{r.get('dataset')}/{r.get('model')} r{r.get('repeat')} f{r.get('fold')}")
                continue
            if r.get("tiny"):
                info["n_tiny"] += 1
                continue
            t = r.get("test") or {}
            key = (r["dataset"], r["model"], r["repeat"], r["fold"])
            if key in rows:
                info["n_dups"] += 1
            tt = (r.get("t_tune_s") or 0.0) + (r.get("t_final_s") or 0.0)
            rows[key] = dict(
                dataset=r["dataset"], track=r.get("track"), family=r.get("family"), model=r["model"],
                repeat=r["repeat"], fold=r["fold"], balanced_acc=t.get("balanced_acc"), macro_f1=t.get("macro_f1"),
                pixel_acc=t.get("pixel_acc"), specimen_avg_acc=t.get("specimen_avg_acc"),
                specimen_vote_acc=t.get("specimen_vote_acc"), per_class_recall=t.get("per_class_recall"),
                n_train=r.get("n_train"), n_test=r.get("n_test"), n_params=r.get("n_params"),
                time_s=tt if tt > 0 else None, epoch_time_s=r.get("epoch_time_s"), budget=r.get("budget"),
                max_epochs=r.get("max_epochs"), inner_score=r.get("inner_score"), device=r.get("device"))
    df = pd.DataFrame(list(rows.values()))
    if len(df):
        df["unit"] = list(zip(df["repeat"], df["fold"]))
    return df, info


def ordered(models, family=None, df=None) -> list[str]:
    models = list(dict.fromkeys(models))
    return sorted(models, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 999, m))


def datasets_in(df: pd.DataFrame) -> list[str]:
    ds = set(df["dataset"]) if len(df) else set()
    return [d for d in DATASET_ORDER if d in ds] + sorted(ds - set(DATASET_ORDER))


def family_of(df: pd.DataFrame, model: str) -> str:
    return str(df.loc[df.model == model, "family"].iloc[0])


def _series(df, ds, model, col="balanced_acc") -> pd.Series:
    s = df[(df.dataset == ds) & (df.model == model)].set_index("unit")[col]
    return s.dropna()


# =============================================================================================== statistics
def paired(df: pd.DataFrame, ds: str, a: str, b: str, col="balanced_acc") -> dict | None:
    sa, sb = _series(df, ds, a, col), _series(df, ds, b, col)
    common = sorted(set(sa.index) & set(sb.index))
    if not common:
        return None
    sub = df[(df.dataset == ds) & (df.model == a) & df.unit.isin(common)].set_index("unit").loc[common]
    ratio = float((sub.n_test / sub.n_train).mean()) if sub.n_train.notna().all() and sub.n_train.min() > 0 \
        else None
    if ratio is None:  # k-fold fallback: 1/(k-1), k = folds per repeat seen in the records
        k = max(1, len({f for _, f in common}))
        ratio = 1.0 / max(1, k - 1)
    st = stats_paired(sa.loc[common].values, sb.loc[common].values, ratio)
    st["ratio"] = ratio
    st["units"] = common
    return st


def model_summary(df: pd.DataFrame, ds: str, track: str) -> pd.DataFrame:
    sub = df[(df.dataset == ds) & (df.track == track)]
    g = sub.groupby("model")
    out = pd.DataFrame(dict(mean=g.balanced_acc.mean(), std=g.balanced_acc.std(ddof=1), n=g.balanced_acc.count()))
    out["family"] = [family_of(sub, m) for m in out.index]
    return out


def best_of(summ: pd.DataFrame, family: str) -> str | None:
    s = summ[summ.family == family]
    return None if s.empty else str(s["mean"].idxmax())


def compute_stats(df: pd.DataFrame) -> dict:
    """res[track][ds] = dict(best_ours, best_base, ours: {m: {vs_base: {...}, vs_best: {...}}}, headline, summ)."""
    res: dict = {}
    for track in TRACKS:
        for ds in datasets_in(df):
            summ = model_summary(df, ds, track)
            if summ.empty:
                continue
            bo, bb = best_of(summ, "ours"), best_of(summ, "baseline")
            ent = dict(best_ours=bo, best_base=bb, summ=summ, ours={}, headline=None)
            bases = ordered(summ[summ.family == "baseline"].index)
            for om in ordered(summ[summ.family == "ours"].index):
                vs_base = {b: paired(df, ds, om, b) for b in bases}
                vs_base = {b: v for b, v in vs_base.items() if v}
                if vs_base:
                    adj = holm([v["wilcoxon_p"] for v in vs_base.values()])
                    adj_nb = holm([v["nb_p"] for v in vs_base.values()])
                    for (b, v), pa, pn in zip(vs_base.items(), adj, adj_nb):
                        v["wilcoxon_holm"], v["nb_holm"] = pa, pn
                vs_best = paired(df, ds, om, bb) if bb else None
                ent["ours"][om] = dict(vs_base=vs_base, vs_best=vs_best)
            # Holm across the "ours" variants against the strongest baseline (guards against picking the best ours)
            vb = [(m, e["vs_best"]) for m, e in ent["ours"].items() if e["vs_best"]]
            if vb:
                ad = holm([v["wilcoxon_p"] for _, v in vb])
                an = holm([v["nb_p"] for _, v in vb])
                for (_, v), pa, pn in zip(vb, ad, an):
                    v["wilcoxon_holm_ours"], v["nb_holm_ours"] = pa, pn
            if bo and bo in ent["ours"]:
                ent["headline"] = ent["ours"][bo]["vs_best"]
            res.setdefault(track, {})[ds] = ent
    return res


# =============================================================================================== ranks / CD
def nemenyi_q(k: int, alpha=0.05) -> float:
    try:
        from scipy.stats import studentized_range
        return float(studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2))
    except Exception:  # pragma: no cover
        tab = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        return tab.get(k, 3.2)


def ranks_per_dataset(df: pd.DataFrame, track: str, min_cov: float = 0.8) -> dict:
    """Average rank (1 = best) over units, per dataset. Only models covering >= min_cov of the max unit count are
    ranked, over the units they share, so partial sweeps stay comparable."""
    out = {}
    for ds in datasets_in(df):
        sub = df[(df.dataset == ds) & (df.track == track)]
        if sub.empty:
            continue
        counts = sub.groupby("model").unit.nunique()
        elig = [m for m, c in counts.items() if c >= min_cov * counts.max()]
        if len(elig) < 2:
            continue
        wide = sub[sub.model.isin(elig)].pivot(index="unit", columns="model", values="balanced_acc").dropna()
        if wide.empty:
            continue
        rk = wide.rank(axis=1, ascending=False, method="average")
        out[ds] = dict(avg=rk.mean(0).to_dict(), n_units=int(len(wide)), models=list(wide.columns))
    return out


def cross_dataset_ranks(df: pd.DataFrame, track: str) -> dict:
    per = ranks_per_dataset(df, track)
    if not per:
        return {}
    models = sorted(set().union(*[set(v["avg"]) for v in per.values()]))
    table = {m: {ds: v["avg"][m] for ds, v in per.items() if m in v["avg"]} for m in models}
    full = [m for m in models if len(table[m]) == len(per)]
    avg = {m: float(np.mean(list(table[m].values()))) for m in models}
    avg_full = {m: avg[m] for m in full}
    k = len(full)
    n_ds = len(per)
    n_blocks = sum(v["n_units"] for v in per.values())
    cd_ds = nemenyi_q(k) * math.sqrt(k * (k + 1) / (6 * n_ds)) if k >= 2 else float("nan")
    cd_blocks = nemenyi_q(k) * math.sqrt(k * (k + 1) / (6 * n_blocks)) if k >= 2 else float("nan")
    return dict(per_dataset=per, table=table, avg=avg, avg_full=avg_full, n_datasets=n_ds, k=k,
                n_blocks=n_blocks, cd_datasets=cd_ds, cd_blocks=cd_blocks)


# =============================================================================================== tables
class Tbl:
    """Tiny table model rendered to LaTeX (booktabs) and Markdown. Cells: (text, style) with style in
    {None, 'best', 'second'}; rows may be the string '---' for a mid rule."""

    def __init__(self, header: list[str], align: str):
        self.header, self.align, self.rows = header, align, []

    def add(self, *cells):
        self.rows.append(list(cells))

    def rule(self):
        self.rows.append("---")

    @staticmethod
    def _cell(c):
        return (c, None) if not isinstance(c, tuple) else c

    def latex(self) -> str:
        L = [*(f"% {c}" for c in getattr(self, "comments", [])), r"\small", rf"\begin{{tabular}}{{{self.align}}}", r"\toprule", " & ".join(self.header) + r" \\", r"\midrule"]
        for r in self.rows:
            if r == "---":
                L.append(r"\midrule")
                continue
            cs = []
            for c in r:
                t, s = self._cell(c)
                cs.append(rf"\textbf{{{t}}}" if s == "best" else rf"\underline{{{t}}}" if s == "second" else t)
            L.append(" & ".join(cs) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(L) + "\n"

    def markdown(self) -> str:
        strip = lambda t: re.sub(r"\\(textbf|underline|mathrm)\{([^}]*)\}", r"\2", t).replace(r"\pm", "±") \
            .replace("$", "").replace(r"\,", " ").replace(r"\%", "%").replace(r"\_", "_").replace("--", "–")
        L = ["| " + " | ".join(strip(h) for h in self.header) + " |", "|" + "---|" * len(self.header)]
        for r in self.rows:
            if r == "---":
                continue
            cs = []
            for c in r:
                t, s = self._cell(c)
                t = strip(t)
                cs.append(f"**{t}**" if s == "best" else f"_{t}_" if s == "second" else t)
            L.append("| " + " | ".join(cs) + " |")
        return "\n".join(L) + "\n"


def fmt_pm(m, s, scale=100.0):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    sd = "" if s is None or (isinstance(s, float) and np.isnan(s)) else rf"{s * scale:.1f}"
    return rf"{m * scale:.1f} $\pm$ {sd}" if sd else rf"{m * scale:.1f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _styles(means: dict) -> dict:
    vals = sorted({round(v, 6) for v in means.values() if v == v}, reverse=True)
    out = {}
    for m, v in means.items():
        out[m] = "best" if vals and round(v, 6) == vals[0] else "second" if len(vals) > 1 and round(v, 6) == vals[1] else None
    return out


def main_table(df, stats, track, metric="balanced_acc") -> Tbl:
    dss = [d for d in datasets_in(df) if d in stats.get(track, {})]
    sub = df[df.track == track]
    models = ordered(sub.model.unique())
    base = [m for m in models if family_of(sub, m) == "baseline"]
    ours = [m for m in models if family_of(sub, m) == "ours"]
    T = Tbl(["Model"] + [DATASET_LABEL.get(d, d) for d in dss], "l" + "c" * len(dss))
    agg = {d: sub[sub.dataset == d].groupby("model")[metric] for d in dss}
    cell = {d: {m: (g.mean().get(m, np.nan), g.std(ddof=1).get(m, np.nan), g.count().get(m, 0)) for m in models}
            for d, g in agg.items()}
    sty = {d: _styles({m: v[0] for m, v in cell[d].items() if v[2] > 0}) for d in dss}

    def row(m):
        return [tex_escape(label(m))] + [(fmt_pm(*cell[d][m][:2]), sty[d].get(m)) if cell[d][m][2] > 0 else "--"
                                         for d in dss]
    for m in base:
        T.add(*row(m))
    if base and ours:
        T.rule()
    for m in ours:
        T.add(*row(m))
    if metric == "balanced_acc" and base and ours:
        T.rule()
        cells = []
        for d in dss:
            h = stats[track][d]["headline"]
            cells.append("--" if not h else f"{sgn(h['mean_diff'] * 100)} [{sgn(h['ci95'][0] * 100)}, {sgn(h['ci95'][1] * 100)}]")
        T.add("Highest-mean ours $-$ highest-mean baseline [95\\% CI]", *cells)
        T.add("Highest-mean baseline", *[tex_escape(label(stats[track][d]["best_base"] or "--")) for d in dss])
    ns = []
    for d in dss:
        c = [v[2] for v in cell[d].values() if v[2] > 0]
        ns.append("--" if not c else (f"{min(c)}" if min(c) == max(c) else f"{min(c)}--{max(c)}"))
    T.rule()
    T.add("Units $n$", *ns)
    return T


def extra_table(df, stats, track) -> Tbl:
    dss = [d for d in datasets_in(df) if d in stats.get(track, {})]
    sub = df[df.track == track]
    models = ordered(sub.model.unique())
    hdr = ["Model"]
    for d in dss:
        hdr += [f"{DATASET_LABEL.get(d, d)} F1", f"{DATASET_LABEL.get(d, d)} vote"]
    T = Tbl(hdr, "l" + "cc" * len(dss))
    base = [m for m in models if family_of(sub, m) == "baseline"]
    ours = [m for m in models if family_of(sub, m) == "ours"]
    stats_ = {}
    for d in dss:
        for col in ("macro_f1", "specimen_vote_acc"):
            g = sub[sub.dataset == d].groupby("model")[col]
            stats_[d, col] = (g.mean(), g.std(ddof=1), g.count())
    sty = {k: _styles({m: v for m, v in s[0].items() if s[2].get(m, 0) > 0}) for k, s in stats_.items()}

    def row(m):
        r = [tex_escape(label(m))]
        for d in dss:
            for col in ("macro_f1", "specimen_vote_acc"):
                mu, sd, n = stats_[d, col]
                r.append((fmt_pm(mu.get(m), sd.get(m)), sty[d, col].get(m)) if n.get(m, 0) > 0 else "--")
        return r
    for m in base:
        T.add(*row(m))
    if base and ours:
        T.rule()
    for m in ours:
        T.add(*row(m))
    return T


def stats_table(stats) -> Tbl:
    T = Tbl(["Track", "Dataset", "Ours", "vs.", "n", r"$\bar d$ [95\% CI]", "W/T/L", r"$p_{\mathrm{W}}$",
             r"$p_{\mathrm{NB}}$", r"$p_{\mathrm{W}}^{\mathrm{Holm}}$"], "llllcclccc")

    def line(track, ds, om, bname, v, holm_p, tag=""):
        T.add(track, DATASET_LABEL.get(ds, ds), tex_escape(label(om)) + tag, tex_escape(label(bname)), str(v["n"]),
              f"{sgn(v['mean_diff'] * 100)} [{sgn(v['ci95'][0] * 100)}, {sgn(v['ci95'][1] * 100)}]",
              f"{v['wins']}/{v['ties']}/{v['losses']}", fmt_p(v["wilcoxon_p"]), fmt_p(v["nb_p"]), fmt_p(holm_p))
    T.comments = ["* = best 'ours' variant by mean balanced accuracy (selected on test units: optimistic).",
                  "Rows vs the strongest baseline list every 'ours' variant; Holm across the 'ours' variants.",
                  "Remaining rows: best ours vs every other baseline; Holm across baselines.",
                  "p_NB: Nadeau-Bengio corrected resampled t-test with mean n_test/n_train."]
    first = True
    for track in TRACKS:
        for ds, e in stats.get(track, {}).items():
            if not first:
                T.rule()
            first = False
            # headline: best ours (+ every ours) vs strongest baseline, Holm across ours variants
            for om, oe in e["ours"].items():
                if oe["vs_best"]:
                    line(track, ds, om, e["best_base"], oe["vs_best"], oe["vs_best"].get("wilcoxon_holm_ours"),
                         "*" if om == e["best_ours"] else "")
            bo = e["best_ours"]
            if bo and bo in e["ours"]:
                for b, v in e["ours"][bo]["vs_base"].items():
                    if b != e["best_base"]:
                        line(track, ds, bo, b, v, v.get("wilcoxon_holm"), "*")
    return T


# =============================================================================================== numbers.tex
_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def macro_name(*parts) -> str:
    """LaTeX-safe macro name: letters only (digits spelled out, separators camel-cased)."""
    out = []
    for p in parts:
        p = re.sub(r"\d", lambda m: "_" + _DIGITS[int(m.group())] + "_", p)
        out.append("".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", p)))
    return "".join(out)


def sgn(x: float, nd=1) -> str:
    """Signed number with a typographic minus, math mode."""
    return f"${'-' if x < 0 else '+'}{abs(x):.{nd}f}$"


def numbers_tex(df, info, stats, ranks) -> str:
    L = ["% auto-generated by me_attention.analysis -- do not edit", ""]

    def mac(name, val):
        L.append(rf"\newcommand{{\{name}}}{{{val}}}")
    T = df["budget"].dropna().mode()
    E = df["max_epochs"].dropna().mode()
    mac("BudgetT", int(T.iloc[0]) if len(T) else "?")
    mac("MaxEpochsE", int(E.iloc[0]) if len(E) else "?")
    nb = {t: df[(df.track == t) & (df.family == "baseline")].model.nunique() for t in TRACKS}
    mac("NBaselinesSpectral", nb["spectral"])
    mac("NBaselinesSpatial", nb["spatial"])
    mac("NBaselines", sum(nb.values()))
    mac("NErrorRecords", info["n_errors"])
    mac("NRecords", len(df))
    for track in TRACKS:
        T_ = track.capitalize()
        gaps, k_first, k_sig, k_ds = [], 0, 0, 0
        for ds, e in stats.get(track, {}).items():
            tag = DATASET_TAG.get(ds, macro_name(ds)) + T_
            h, s = e["headline"], e["summ"]
            if e["best_ours"]:
                mac(macro_name("BestOursName", tag), tex_escape(label(e["best_ours"])))
                mac(macro_name("BestOursAcc", tag), f"{s.loc[e['best_ours'], 'mean'] * 100:.1f}")
                mac(macro_name("BestOursStd", tag), f"{s.loc[e['best_ours'], 'std'] * 100:.1f}" if s.loc[e['best_ours'], 'n'] > 1 else "--")
            if e["best_base"]:
                mac(macro_name("BestBaseName", tag), tex_escape(label(e["best_base"])))
                mac(macro_name("BestBaseAcc", tag), f"{s.loc[e['best_base'], 'mean'] * 100:.1f}")
            k_ds += 1
            if h:
                mac(macro_name("Gap", tag), sgn(h['mean_diff'] * 100))
                mac(macro_name("GapLo", tag), sgn(h['ci95'][0] * 100))
                mac(macro_name("GapHi", tag), sgn(h['ci95'][1] * 100))
                mac(macro_name("PWilcox", tag), fmt_p(h["wilcoxon_p"]).replace("<", r"$<$"))
                mac(macro_name("PNB", tag), fmt_p(h["nb_p"]).replace("<", r"$<$"))
                ph = h.get("wilcoxon_holm_ours", h["wilcoxon_p"])
                mac(macro_name("PHolm", tag), fmt_p(ph).replace("<", r"$<$"))
                mac(macro_name("NUnits", tag), h["n"])
                mac(macro_name("Wins", tag), h["wins"])
                mac(macro_name("Ties", tag), h["ties"])
                mac(macro_name("Losses", tag), h["losses"])
                gaps.append(h["mean_diff"] * 100)
                k_first += int(h["mean_diff"] > 0)
                k_sig += int(h["mean_diff"] > 0 and ph < 0.05 and h["nb_p"] < 0.05)
        sfx = T_
        mac(macro_name("NDatasets", sfx), k_ds)
        mac(macro_name("KFirst", sfx), k_first)
        mac(macro_name("KSignificant", sfx), k_sig)
        if gaps:
            mac(macro_name("GapMin", sfx), sgn(min(gaps)))
            mac(macro_name("GapMax", sfx), sgn(max(gaps)))
            mac(macro_name("GapRange", sfx), f"{sgn(min(gaps))} to {sgn(max(gaps))}")
        r = ranks.get(track)
        if r:
            for m, v in r["avg"].items():
                mac(macro_name("AvgRank", m, sfx), f"{v:.2f}")
            mac(macro_name("CDNemenyi", sfx), f"{r['cd_datasets']:.2f}")
    return "\n".join(L) + "\n"


# =============================================================================================== figures
def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    return plt


def colors_for(df: pd.DataFrame) -> dict:
    cols, bi, oi = {}, 0, 0
    for m in ordered(df.model.unique()):
        if family_of(df, m) == "ours":
            cols[m] = OURS_COLORS[oi % len(OURS_COLORS)]
            oi += 1
        else:
            cols[m] = BASE_COLORS[bi % len(BASE_COLORS)]
            bi += 1
    return cols


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_box(df, stats, out):
    plt = _rc()
    cols = colors_for(df)
    for track in TRACKS:
        dss = [d for d in datasets_in(df) if d in stats.get(track, {})]
        if not dss:
            continue
        sub = df[df.track == track]
        models = ordered(sub.model.unique())[::-1]
        fig, axes = plt.subplots(1, len(dss), figsize=(IEEE_2COL, 0.17 * len(models) + 0.9), sharey=True, squeeze=False)
        rng = np.random.default_rng(0)
        for ax, ds in zip(axes[0], dss):
            for i, m in enumerate(models):
                v = sub[(sub.dataset == ds) & (sub.model == m)].balanced_acc.dropna().values * 100
                if not len(v):
                    continue
                ax.boxplot([v], positions=[i], orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=cols[m], alpha=0.35, linewidth=0.5, edgecolor=cols[m]),
                           medianprops=dict(color="black", linewidth=0.8), whiskerprops=dict(linewidth=0.5),
                           capprops=dict(linewidth=0.5))
                ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=4, color=cols[m], zorder=3, linewidths=0)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([label(m) for m in models])
            ax.set_title(DATASET_LABEL.get(ds, ds))
            ax.set_xlabel("Balanced accuracy (%)")
            ax.grid(axis="x", linewidth=0.3, alpha=0.5)
            for t in ax.get_yticklabels():
                nm = [mm for mm in models if label(mm) == t.get_text()]
                if nm and family_of(sub, nm[0]) == "ours":
                    t.set_fontweight("bold")
        fig.suptitle(f"{track.capitalize()} track", y=1.0, fontsize=7.5)
        fig.tight_layout()
        _save(fig, out, f"fig_box_{track}")


def fig_paired(df, stats, out):
    plt = _rc()
    cols = colors_for(df)
    for track in TRACKS:
        dss = [d for d in datasets_in(df) if d in stats.get(track, {}) and stats[track][d]["best_ours"]]
        if not dss:
            continue
        fig, axes = plt.subplots(1, len(dss), figsize=(IEEE_2COL, 2.6), sharey=False, squeeze=False)
        for ax, ds in zip(axes[0], dss):
            e = stats[track][ds]
            vs = e["ours"][e["best_ours"]]["vs_base"]
            names = list(vs)[::-1]
            for i, b in enumerate(names):
                v = vs[b]
                ax.errorbar(v["mean_diff"] * 100, i, xerr=[[(v["mean_diff"] - v["ci95"][0]) * 100],
                                                          [(v["ci95"][1] - v["mean_diff"]) * 100]],
                            fmt="o", ms=3, color=cols[b], ecolor=cols[b], elinewidth=0.9, capsize=1.5)
            ax.axvline(0, color="k", linewidth=0.6)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([label(b) for b in names])
            ax.set_title(f"{DATASET_LABEL.get(ds, ds)}\n{label(e['best_ours'])} $-$ baseline", fontsize=6.5)
            ax.set_xlabel("$\\Delta$ balanced acc. (points, 95% CI)")
            ax.grid(axis="x", linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        _save(fig, out, f"fig_paired_{track}")


def _cd_draw(ax, avg: dict, cd: float, cols: dict, title: str):
    names = sorted(avg, key=avg.get)
    k = len(names)
    lo, hi = 1, max(k, math.ceil(max(avg.values())))
    ax.set_xlim(lo - 1.9, hi + 1.9)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.75, lo, hi, color="k", linewidth=0.8)
    for t in range(lo, hi + 1):
        ax.vlines(0.75 and t, 0.75, 0.78, color="k", linewidth=0.6)
        ax.text(t, 0.8, str(t), ha="center", va="bottom", fontsize=6)
    half = (k + 1) // 2
    for i, m in enumerate(names):
        left = i < half
        y = 0.62 - 0.07 * (i if left else i - half)
        x_lab = lo - 0.1 if left else hi + 0.1
        ax.plot([avg[m], avg[m]], [0.75, y], color=cols.get(m, "k"), linewidth=0.7)
        ax.plot([avg[m], x_lab], [y, y], color=cols.get(m, "k"), linewidth=0.7)
        ax.text(x_lab + (-0.05 if left else 0.05), y, f"{label(m)} ({avg[m]:.2f})", ha="right" if left else "left",
                va="center", fontsize=6, color=cols.get(m, "k"))
    if cd == cd:
        ax.hlines(0.95, lo, lo + cd, color="k", linewidth=1.2)
        ax.text(lo + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6)
    # bars joining maximal groups of models not separable by CD
    groups, last_j = [], -1
    for i, m in enumerate(names):
        j = i
        while j + 1 < k and avg[names[j + 1]] - avg[m] < cd:
            j += 1
        if j > i and j > last_j:
            groups.append((i, j))
            last_j = j
    for g, (i, j) in enumerate(groups):
        ax.hlines(0.71 - 0.022 * g, avg[names[i]], avg[names[j]], color="k", linewidth=2.0)
    ax.set_title(title, fontsize=7, y=1.0)


def fig_ranks(df, ranks, out):
    plt = _rc()
    cols = colors_for(df)
    for track, r in ranks.items():
        if not r:
            continue
        avg = r["avg_full"] or r["avg"]
        if not avg:
            continue
        cd = r["cd_datasets"] if r["n_datasets"] >= 3 and r["k"] >= 2 else float("nan")
        use_cd = r["n_datasets"] >= 3
        fig = plt.figure(figsize=(IEEE_2COL, 2.4 if use_cd else 1.8))
        ax = fig.add_subplot(111)
        if use_cd:
            _cd_draw(ax, avg, cd, cols, f"{track.capitalize()} track: average rank over {r['n_datasets']} datasets "
                                        f"(Nemenyi, N = datasets)")
        else:
            names = sorted(avg, key=avg.get)
            ax.barh(range(len(names))[::-1], [avg[m] for m in names], color=[cols.get(m, "grey") for m in names],
                    height=0.7)
            ax.set_yticks(range(len(names))[::-1])
            ax.set_yticklabels([label(m) for m in names])
            ax.set_xlabel(f"Average rank (1 = best) over {r['n_datasets']} dataset(s); CD needs >= 3 datasets")
            ax.set_title(f"{track.capitalize()} track")
        _save(fig, out, f"fig_ranks_{track}")


def fig_cost(df, out):
    plt = _rc()
    cols = colors_for(df)
    mk = dict(zip(DATASET_ORDER, ["o", "s", "^", "D"]))
    for track in TRACKS:
        sub = df[df.track == track]
        if sub.empty or sub.n_params.notna().sum() == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL, 2.4))
        for ax, (col, xl) in zip(axes, [("n_params", "Parameters"), ("time_s", "Train + tune time per unit (s)")]):
            g = sub.groupby(["dataset", "model"]).agg(acc=("balanced_acc", "mean"), x=(col, "mean")).dropna().reset_index()
            if col == "n_params":
                g = g[g.x > 0]
            for _, r in g.iterrows():
                ax.scatter(r.x, r.acc * 100, color=cols[r.model], marker=mk.get(r.dataset, "o"), s=14,
                           edgecolors="k", linewidths=0.3)
            ax.set_xscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("Balanced acc. (%)")
            ax.grid(linewidth=0.3, alpha=0.5)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker=mk.get(d, "o"), ls="", color="grey", label=DATASET_LABEL.get(d, d), ms=4)
             for d in datasets_in(sub)]
        h += [Line2D([], [], marker="o", ls="", color=cols[m], label=label(m), ms=4)
              for m in ordered(sub.model.unique())]
        fig.legend(handles=h, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)
        fig.suptitle(f"{track.capitalize()} track (neural models have parameter counts)", fontsize=7.5)
        fig.tight_layout()
        _save(fig, out, f"fig_cost_{track}")


def pooled_recall(df, ds, model) -> np.ndarray | None:
    rec = df[(df.dataset == ds) & (df.model == model)].per_class_recall.dropna()
    if rec.empty:
        return None
    arr = np.array([[np.nan if x is None else x for x in r] for r in rec], float)
    return np.nanmean(arr, 0)


def fig_recall(df, stats, out, class_names: dict | None = None):
    plt = _rc()
    for track in TRACKS:
        dss = [d for d in datasets_in(df) if d in stats.get(track, {}) and stats[track][d]["best_ours"]
               and stats[track][d]["best_base"]]
        if not dss:
            continue
        fig, axes = plt.subplots(1, len(dss), figsize=(IEEE_2COL, 1.4), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            e = stats[track][ds]
            rows = [e["best_ours"], e["best_base"]]
            M = np.array([pooled_recall(df, ds, m) for m in rows]) * 100
            im = ax.imshow(M, vmin=0, vmax=100, cmap="viridis", aspect="auto")
            names = (class_names or {}).get(ds) or [f"c{i}" for i in range(M.shape[1])]
            ax.set_xticks(range(M.shape[1]))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks([0, 1])
            ax.set_yticklabels([label(m) for m in rows])
            ax.set_title(DATASET_LABEL.get(ds, ds))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] < 60 else "black")
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01, label="Recall (%)")
        _save(fig, out, f"fig_recall_{track}")


def class_names_for(datasets) -> dict:
    out = {}
    try:
        from .data import load as load_dataset
        for d in datasets:
            try:
                out[d] = list(load_dataset(d).class_names)
            except Exception:
                pass
    except Exception:
        pass
    return out


# =============================================================================================== driver
def write_tbl(T: Tbl, out: Path, name: str):
    (out / f"{name}.tex").write_text(T.latex(), encoding="utf-8")
    (out / f"{name}.md").write_text(T.markdown(), encoding="utf-8")


def run(sweep: str, out: Path, root: Path = RESULTS_ROOT, figures: bool = True, names: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, info = load_sweep(sweep, root)
    summary = dict(sweep=sweep, n_records=int(len(df)), **{k: v for k, v in info.items() if k != "errors"},
                   errors=info["errors"][:20])
    if df.empty:
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"[analysis] no records under {Path(root) / sweep}")
        return summary
    stats = compute_stats(df)
    ranks = {t: cross_dataset_ranks(df, t) for t in TRACKS}
    for track in TRACKS:
        if track in stats:
            write_tbl(main_table(df, stats, track), out, f"tab_{track}")
            write_tbl(extra_table(df, stats, track), out, f"tab_{track}_extra")
    write_tbl(stats_table(stats), out, "tab_stats")
    (out / "numbers.tex").write_text(numbers_tex(df, info, stats, ranks), encoding="utf-8")
    if figures:
        figdir = out / "figures"
        cn = class_names_for(datasets_in(df)) if names else {}
        for fn, args in [(fig_box, (df, stats, figdir)), (fig_paired, (df, stats, figdir)),
                         (fig_ranks, (df, ranks, figdir)), (fig_cost, (df, figdir)),
                         (fig_recall, (df, stats, figdir, cn))]:
            try:
                fn(*args)
            except Exception as ex:  # a broken figure must not kill tables
                print(f"[analysis] figure {fn.__name__} failed: {type(ex).__name__}: {ex}")
    coverage = {f"{t}/{d}": {m: int(n) for m, n in e["summ"]["n"].items()} for t in stats for d, e in stats[t].items()}
    summary["coverage"] = coverage
    summary["headline"] = {f"{t}/{d}": (None if not e["headline"] else {
        "best_ours": e["best_ours"], "best_base": e["best_base"],
        **{k: e["headline"][k] for k in ("n", "mean_diff", "ci95", "wilcoxon_p", "nb_p", "wins", "ties", "losses")}})
        for t in stats for d, e in stats[t].items()}
    summary["ranks"] = {t: {"avg": r.get("avg"), "cd_datasets": r.get("cd_datasets"), "cd_blocks": r.get("cd_blocks"),
                            "n_datasets": r.get("n_datasets")} for t, r in ranks.items() if r}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"[analysis] {len(df)} records, {info['n_errors']} error records skipped; wrote {out}")
    for k, v in coverage.items():
        print(f"  {k}: " + ", ".join(f"{m}={n}" for m, n in v.items()))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", default="publications/me_attention/generated")
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-class-names", action="store_true", help="skip loading datasets for class names")
    a = ap.parse_args(argv)
    run(a.sweep, Path(a.out), Path(a.results_root), figures=not a.no_figures, names=not a.no_class_names)


if __name__ == "__main__":
    main()
