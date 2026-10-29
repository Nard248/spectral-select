"""Analysis for the ME-attention robustness study (PREREG_ROBUSTNESS_2026-10-28.md, PHASEB_IMPL_SPEC.md section 1).

    PYTHONPATH=src python -m me_attention.robust_analysis [--root results/me_attention/robust] \\
        [--out results/me_attention/analysis_robust]

Reads ``results/me_attention/robust/<dataset>/<model>.jsonl`` (written by ``robustness.py``), reuses
``protocol.stats_paired`` / ``protocol.holm`` and ``analysis.py``'s label / colour / ``Tbl`` / ``_save`` helpers.
Tolerates partial sweeps; never drops a model or condition because it looks bad -- every (dataset, model, family,
level) that was run is shown, missing cells are marked missing (``"--"`` / ``null``), never omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import BASE_COLORS, OURS_COLORS, Tbl, _rc, _save, label, ordered, tex_escape, write_tbl
from .protocol import holm, stats_paired
from .robustness import RESULTS_ROOT as ROBUST_ROOT

OUT_ROOT = Path("results/me_attention/analysis_robust")
NON_CLEAN_FAMILIES = ["T1", "T2", "T3", "L", "C"]

# Frozen confirmatory design: PREREG_ROBUSTNESS_2026-10-28.md + COMPARATORS_2026-10-28.md (fixed before any run).
CONFIRMATORY_DATASETS = ["lichens2_sub3", "lichens2"]
CONFIRMATORY_CANDIDATES = ["ex_tokens_v2", "ex_tokens_lin"]
CONFIRMATORY_COMPARATOR = {"lichens2_sub3": "rbfsvm", "lichens2": "rbfsvm",
                          "collagen_pepsin": "lr", "lichens1": "spectralformer"}


def _model_family(model: str) -> str:
    """'baseline' or 'ours' for a model name. Robust records' own ``"family"`` field is the *perturbation*
    family (clean/T1/T2/T3/L/C), so -- unlike analysis.py's clean-sweep records -- it cannot be read off the
    robust DataFrame; ask the model registry directly."""
    try:
        from .models.registry import MODELS
        return MODELS[model].family
    except Exception:
        return "baseline"


def _colors_for(df: pd.DataFrame) -> dict:
    """Like analysis.colors_for, but keyed off the model registry's family (see ``_model_family``) instead of a
    df ``"family"`` column, which on robust records means the perturbation family, not baseline/ours."""
    cols, bi, oi = {}, 0, 0
    for m in ordered(df.model.unique()):
        if _model_family(m) == "ours":
            cols[m] = OURS_COLORS[oi % len(OURS_COLORS)]
            oi += 1
        else:
            cols[m] = BASE_COLORS[bi % len(BASE_COLORS)]
            bi += 1
    return cols


# =============================================================================================== loading
def load_robust(root: Path = ROBUST_ROOT) -> tuple[pd.DataFrame, dict]:
    """Tidy DataFrame of successful records, one row per (dataset, model, repeat, fold, family, level, subset),
    deduplicated (last wins -- a resumed run's newest line for a key)."""
    rows: dict[tuple, dict] = {}
    info = dict(n_files=0, n_errors=0, n_torn=0, n_dups=0, errors=[])
    base = Path(root)
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
                info["errors"].append(f"{r.get('dataset')}/{r.get('model')} r{r.get('repeat')} f{r.get('fold')}: "
                                      f"{str(r['error']).strip().splitlines()[-1]}")
                continue
            key = (r["dataset"], r["model"], r["repeat"], r["fold"], r["family"], str(r["level"]), r.get("subset"))
            if key in rows:
                info["n_dups"] += 1
            t = r.get("test") or {}
            rows[key] = dict(
                dataset=r["dataset"], model=r["model"], track=r.get("track"), family=r["family"], level=r["level"],
                subset=r.get("subset"), repeat=r["repeat"], fold=r["fold"], n_train=r.get("n_train"),
                n_test=r.get("n_test"), n_subsets=r.get("n_subsets"), balanced_acc=t.get("balanced_acc"),
                macro_f1=t.get("macro_f1"), pixel_acc=t.get("pixel_acc"), specimen_avg_acc=t.get("specimen_avg_acc"),
                specimen_vote_acc=t.get("specimen_vote_acc"), clip_notes=r.get("clip_notes") or [])
    df = pd.DataFrame(list(rows.values()))
    if len(df):
        df["unit"] = list(zip(df["repeat"], df["fold"]))
    return df, info


def datasets_in(df: pd.DataFrame) -> list[str]:
    ds = set(df["dataset"]) if len(df) else set()
    return [d for d in CONFIRMATORY_DATASETS if d in ds] + sorted(ds - set(CONFIRMATORY_DATASETS))


def _ratio(df: pd.DataFrame, ds: str) -> float:
    """test_train_ratio of a dataset: mean n_test/n_train over its available units (protocol.stats_paired)."""
    sub = df[(df.dataset == ds) & df.n_train.notna() & df.n_test.notna() & (df.n_train > 0)]
    return float((sub.n_test / sub.n_train).mean()) if len(sub) else 1.0


def _family_unit_series(df: pd.DataFrame, ds: str, model: str, family: str) -> pd.Series:
    """One value per (repeat, fold) unit: that model's mean balanced accuracy over every non-clean level (and,
    for C, both sub-families) of ``family`` run for that unit -- the per-unit robustness value used for pairing."""
    sub = df[(df.dataset == ds) & (df.model == model) & (df.family == family)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("unit").balanced_acc.mean().dropna()


# =============================================================================================== robustness scores
def robustness_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, model, family): mean balanced accuracy over that family's non-clean levels, averaged with
    equal weight per level (T1 records are already per-level averages over their subsets; C averages the levels
    of both sub-families together), plus the clean value for reference. Missing combinations get score=None."""
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    for (ds, model), g in df.groupby(["dataset", "model"]):
        clean = g[g.family == "clean"].balanced_acc
        clean_mean = float(clean.mean()) if len(clean.dropna()) else None
        for fam in NON_CLEAN_FAMILIES:
            fg = g[g.family == fam]
            if fg.empty:
                rows.append(dict(dataset=ds, model=model, family=fam, clean=clean_mean, score=None,
                                 n_levels=0, n_units=0))
                continue
            per_level = fg.groupby(["level", "subset"], dropna=False).balanced_acc.mean()
            rows.append(dict(dataset=ds, model=model, family=fam, clean=clean_mean,
                             score=float(per_level.mean()) if len(per_level.dropna()) else None,
                             n_levels=int(per_level.notna().sum()), n_units=int(fg.unit.nunique())))
    return pd.DataFrame(rows)


def tab_robust_scores(df: pd.DataFrame, scores: pd.DataFrame) -> Tbl:
    T = Tbl(["Dataset", "Model", "Clean"] + NON_CLEAN_FAMILIES, "ll" + "c" * (1 + len(NON_CLEAN_FAMILIES)))
    T.comments = ["Robustness score = mean balanced accuracy over a family's non-clean levels "
                  "(equal weight per level; T1 already averaged over its subsets; C pools sub-families a and b).",
                  "'--' = that (dataset, model, family) was never run -- not omitted."]
    for ds in datasets_in(df):
        models = ordered(df[df.dataset == ds].model.unique())
        for m in models:
            sub = scores[(scores.dataset == ds) & (scores.model == m)]
            clean_v = sub["clean"].dropna().iloc[0] if len(sub["clean"].dropna()) else None
            row = [ds if m == models[0] else "", tex_escape(label(m)),
                  "--" if clean_v is None else f"{clean_v * 100:.1f}"]
            for fam in NON_CLEAN_FAMILIES:
                s = sub[sub.family == fam]
                v = s["score"].iloc[0] if len(s) else None
                row.append("--" if v is None else f"{v * 100:.1f}")
            T.add(*row)
        T.rule()
    if T.rows and T.rows[-1] == "---":
        T.rows.pop()
    return T


# =============================================================================================== degradation curve
def _level_cols(df: pd.DataFrame, ds: str, family: str) -> list[tuple[str, object, object]]:
    """(column label, subset, level) for one (dataset, family). T1/C levels are integer counts; the shared
    ``level`` column is float64 once the df also holds T2/T3/L's float levels, so cast back for a clean label
    (equality against the float column below still holds: ``1 == 1.0``)."""
    sub = df[(df.dataset == ds) & (df.family == family)]
    cast = int if family in ("T1", "C") else (lambda v: v)
    if family == "C":
        pairs = sorted({(r.subset, cast(r.level)) for r in sub.itertuples()}, key=lambda t: (str(t[0]), t[1]))
        return [(f"{s}:{v}", s, v) for s, v in pairs]
    vals = sorted({cast(v) for v in sub.level.tolist()})
    return [(str(v), None, v) for v in vals]


def degradation_table(df: pd.DataFrame, ds: str, family: str) -> Tbl:
    """Full per-level curve for one (dataset, family): every model that was run on this dataset at all is a row
    (missing cells '--'), never omitted for a bad score."""
    sub_ds = df[df.dataset == ds]
    models = ordered(sub_ds.model.unique())
    cols = _level_cols(df, ds, family)
    clean = sub_ds[sub_ds.family == "clean"].groupby("model").balanced_acc.mean()
    T = Tbl(["Model", "Clean"] + [c[0] for c in cols], "l" + "c" * (1 + len(cols)))
    for m in models:
        row = [tex_escape(label(m)), f"{clean[m] * 100:.1f}" if m in clean.index and clean[m] == clean[m] else "--"]
        for _, subset, val in cols:
            mask = (sub_ds.model == m) & (sub_ds.family == family) & (sub_ds.level == val)
            if family == "C":
                mask &= sub_ds.subset == subset
            cell = sub_ds[mask].balanced_acc.dropna()
            row.append(f"{cell.mean() * 100:.1f}" if len(cell) else "--")
        T.add(*row)
    return T


def write_degradation(df: pd.DataFrame, out: Path):
    md_parts, tex_parts = [], []
    for ds in datasets_in(df):
        for family in NON_CLEAN_FAMILIES:
            if df[(df.dataset == ds) & (df.family == family)].empty:
                continue
            t = degradation_table(df, ds, family)
            md_parts.append(f"### {ds} / {family}\n\n" + t.markdown())
            tex_parts.append(f"\\subsection*{{{tex_escape(ds)} / {family}}}\n" + t.latex())
    (out / "tab_degradation.md").write_text(
        "\n".join(md_parts) if md_parts else "(no robustness records yet)\n", encoding="utf-8")
    (out / "tab_degradation.tex").write_text(
        "\n".join(tex_parts) if tex_parts else "% no robustness records yet\n", encoding="utf-8")


# =============================================================================================== confirmatory (20 tests)
def confirmatory_tests(df: pd.DataFrame) -> list[dict]:
    rows = []
    for ds in CONFIRMATORY_DATASETS:
        comparator = CONFIRMATORY_COMPARATOR[ds]
        ratio = _ratio(df, ds)
        for family in NON_CLEAN_FAMILIES:
            for cand in CONFIRMATORY_CANDIDATES:
                sa = _family_unit_series(df, ds, cand, family)
                sb = _family_unit_series(df, ds, comparator, family)
                common = sorted(set(sa.index) & set(sb.index))
                entry = dict(dataset=ds, family=family, candidate=cand, comparator=comparator, n=len(common))
                if not common:
                    entry["missing"] = True
                else:
                    entry["missing"] = False
                    entry.update(stats_paired(sa.loc[common].values, sb.loc[common].values, ratio))
                rows.append(entry)
    pvals = [r["nb_p"] for r in rows if not r["missing"]]
    adj = iter(holm(pvals)) if pvals else iter([])
    for r in rows:
        if r["missing"]:
            r["nb_p_holm"], r["decision"] = None, "missing"
        else:
            r["nb_p_holm"] = next(adj)
            r["decision"] = "advantage" if (r["mean_diff"] > 0 and r["nb_p_holm"] < 0.05) else "no advantage"
    assert len(rows) == len(CONFIRMATORY_DATASETS) * len(NON_CLEAN_FAMILIES) * len(CONFIRMATORY_CANDIDATES)
    return rows


def tab_confirmatory(rows: list[dict]) -> Tbl:
    T = Tbl(["Dataset", "Family", "Candidate", "vs.", "n", r"$\bar d$ [95\% CI]", r"$p_{\mathrm{NB}}$",
             r"$p_{\mathrm{NB}}^{\mathrm{Holm}}$", "Decision"], "lllllccl")
    T.comments = ["The 20 pre-registered confirmatory tests (frozen before any robustness result): "
                  "{lichens2_sub3, lichens2} x {T1,T2,T3,L,C} x {ex_tokens_v2, ex_tokens_lin}.",
                  "Delta = candidate robustness score - comparator robustness score, paired by (repeat, fold).",
                  "One Holm family across all 20 Nadeau-Bengio p-values. Decision = advantage iff Delta>0 and "
                  "Holm-adjusted p<0.05.", "'missing' = the pair has no common (repeat, fold) unit yet."]
    for r in rows:
        if r["missing"]:
            T.add(r["dataset"], r["family"], tex_escape(label(r["candidate"])), tex_escape(label(r["comparator"])),
                 "0", "--", "--", "--", "missing")
            continue
        ci = f"{r['mean_diff'] * 100:+.1f} [{r['ci95'][0] * 100:+.1f}, {r['ci95'][1] * 100:+.1f}]"
        style = "best" if r["decision"] == "advantage" else None
        T.add(r["dataset"], r["family"], tex_escape(label(r["candidate"])), tex_escape(label(r["comparator"])),
             str(r["n"]), ci, f"{r['nb_p']:.3f}", f"{r['nb_p_holm']:.3f}", (r["decision"], style))
    return T


# =============================================================================================== exploratory (everything else)
def exploratory_tests(df: pd.DataFrame) -> list[dict]:
    """Every (dataset, family, ours-model, baseline) comparison, Holm-corrected across baselines within each
    (dataset, family, ours-model) group; flags the per-condition best baseline. Clearly exploratory: not part of
    the 20 pre-registered tests, no claim is made from it alone."""
    if df.empty:
        return []
    ours = sorted(m for m in df.model.unique() if _model_family(m) == "ours")
    bases = sorted(m for m in df.model.unique() if _model_family(m) == "baseline")
    rows: list[dict] = []
    for ds in datasets_in(df):
        ratio = _ratio(df, ds)
        for family in NON_CLEAN_FAMILIES:
            fam_scores = {m: _family_unit_series(df, ds, m, family) for m in [*ours, *bases]}
            base_means = {m: fam_scores[m].mean() for m in bases if len(fam_scores[m])}
            best_base = max(base_means, key=base_means.get) if base_means else None
            for cand in ours:
                sa = fam_scores[cand]
                group: list[dict] = []
                for b in bases:
                    sb = fam_scores[b]
                    common = sorted(set(sa.index) & set(sb.index))
                    entry = dict(dataset=ds, family=family, candidate=cand, baseline=b,
                                is_best_baseline=(b == best_base), confirmatory=(
                                    ds in CONFIRMATORY_DATASETS and cand in CONFIRMATORY_CANDIDATES and
                                    b == CONFIRMATORY_COMPARATOR.get(ds)))
                    if not common:
                        entry["missing"] = True
                    else:
                        entry["missing"] = False
                        entry.update(stats_paired(sa.loc[common].values, sb.loc[common].values, ratio))
                    group.append(entry)
                pv = [e["nb_p"] for e in group if not e["missing"]]
                if pv:
                    adj = iter(holm(pv))
                    for e in group:
                        e["nb_p_holm"] = None if e["missing"] else next(adj)
                else:
                    for e in group:
                        e["nb_p_holm"] = None
                rows.extend(group)
    return rows


def tab_exploratory(rows: list[dict]) -> Tbl:
    T = Tbl(["Dataset", "Family", "Candidate", "Baseline", "Best?", "n", r"$\bar d$ [95\% CI]",
             r"$p_{\mathrm{NB}}$", r"$p_{\mathrm{NB}}^{\mathrm{Holm}}$"], "lllllcccc")
    T.comments = ["EXPLORATORY -- not pre-registered; every secondary dataset, every attention model, every "
                  "baseline, Holm-corrected across baselines within (dataset, family, candidate) only.",
                  "'Best?' marks the baseline with the highest robustness score for that (dataset, family).",
                  "Rows marked * duplicate a confirmatory test (same comparator) and are informational only here."]
    for r in rows:
        tag = "*" if r.get("confirmatory") else ""
        best = "yes" if r["is_best_baseline"] else ""
        if r["missing"]:
            T.add(r["dataset"], r["family"], tex_escape(label(r["candidate"])) + tag, tex_escape(label(r["baseline"])),
                 best, "0", "--", "--", "--")
            continue
        ci = f"{r['mean_diff'] * 100:+.1f} [{r['ci95'][0] * 100:+.1f}, {r['ci95'][1] * 100:+.1f}]"
        ph = "--" if r["nb_p_holm"] is None else f"{r['nb_p_holm']:.3f}"
        T.add(r["dataset"], r["family"], tex_escape(label(r["candidate"])) + tag, tex_escape(label(r["baseline"])),
             best, str(r["n"]), ci, f"{r['nb_p']:.3f}", ph)
    return T


# =============================================================================================== figures
def fig_degradation(df: pd.DataFrame, out: Path):
    if df.empty:
        return
    plt = _rc()
    cols = _colors_for(df)
    for ds in datasets_in(df):
        sub_ds = df[df.dataset == ds]
        families = [f for f in NON_CLEAN_FAMILIES if not sub_ds[sub_ds.family == f].empty]
        if not families:
            continue
        fig, axes = plt.subplots(1, len(families), figsize=(2.4 * len(families), 2.6), squeeze=False)
        for ax, family in zip(axes[0], families):
            cols_list = _level_cols(df, ds, family)
            x = list(range(len(cols_list)))
            for m in ordered(sub_ds.model.unique()):
                y = []
                for _, subset, val in cols_list:
                    mask = (sub_ds.model == m) & (sub_ds.family == family) & (sub_ds.level == val)
                    if family == "C":
                        mask &= sub_ds.subset == subset
                    v = sub_ds[mask].balanced_acc.dropna()
                    y.append(float(v.mean()) * 100 if len(v) else np.nan)
                style = dict(color=cols[m], linewidth=1.4 if _model_family(m) == "ours" else 0.9,
                            marker="o", markersize=3)
                ax.plot(x, y, **style)
            ax.set_xticks(x)
            ax.set_xticklabels([c[0] for c in cols_list], rotation=45, ha="right", fontsize=6)
            ax.set_title(f"{family}", fontsize=7.5)
            ax.set_ylabel("Balanced acc. (%)")
            ax.grid(linewidth=0.3, alpha=0.5)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=cols[m], label=label(m),
                          linewidth=1.4 if _model_family(m) == "ours" else 0.9)
                  for m in ordered(sub_ds.model.unique())]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=6)
        fig.suptitle(f"{ds}: degradation per family", fontsize=8)
        fig.tight_layout()
        _save(fig, out, f"fig_degradation_{ds}")


# =============================================================================================== driver
def run(root: Path = ROBUST_ROOT, out: Path = OUT_ROOT, figures: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, info = load_robust(root)
    summary = dict(root=str(root), n_records=int(len(df)), **{k: v for k, v in info.items() if k != "errors"},
                  errors=info["errors"][:20])
    if df.empty:
        (out / "summary_robust.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"[robust_analysis] no records under {root}")
        return summary

    scores = robustness_scores(df)
    write_tbl(tab_robust_scores(df, scores), out, "tab_robust_scores")
    write_degradation(df, out)
    conf_rows = confirmatory_tests(df)
    write_tbl(tab_confirmatory(conf_rows), out, "tab_confirmatory")
    expl_rows = exploratory_tests(df)
    write_tbl(tab_exploratory(expl_rows), out, "tab_exploratory")

    if figures:
        figdir = out / "figures"
        try:
            fig_degradation(df, figdir)
        except Exception as ex:  # a broken figure must not kill the tables
            print(f"[robust_analysis] fig_degradation failed: {type(ex).__name__}: {ex}")

    summary["coverage"] = {ds: {m: int((df.dataset == ds).sum() and (df[(df.dataset == ds) & (df.model == m)]).
                                      unit.nunique()) for m in sorted(df[df.dataset == ds].model.unique())}
                          for ds in datasets_in(df)}
    summary["robustness_scores"] = scores.to_dict(orient="records")
    summary["confirmatory"] = [{k: v for k, v in r.items()} for r in conf_rows]
    summary["exploratory"] = [{k: v for k, v in r.items()} for r in expl_rows]
    (out / "summary_robust.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"[robust_analysis] {len(df)} records ({info['n_errors']} error records skipped); wrote {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROBUST_ROOT))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    run(Path(a.root), Path(a.out), figures=not a.no_figures)


if __name__ == "__main__":
    main()
