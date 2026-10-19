"""Attention / gate importance on the (excitation, emission) grid, and overlap with spectral-select selections.

    PYTHONPATH=src python experiments/me_attention/importance.py --dataset collagen_pepsin --model meas_tokens \
        --sweep final [--repeat 0 --fold 0] [--epochs N] [--out publications/me_attention/generated/importance]

* config: taken from the best record (highest inner_score) of <results-root>/<sweep>/<dataset>/<model>.jsonl; if there
  is none (e.g. smoke / dev set) the model's default config is used (``--epochs`` overrides the epoch count).
* retrains once on the outer-train set of (repeat, fold) with the same inner split as the harness (validation for early
  stopping), then reads ``module.attention_importance(x, mode)`` (rollout and last-layer) and ``feature_importance()``
  (gates; None if the model is not gated) on test pixels (capped).
* outputs: importance heatmaps (PDF+PNG), ``top10_<mode>.md`` and ``importance.json`` (full per-feature scores, top-10,
  overlap with the spectral-select selection when a selection file exists for the dataset).

Selections (main repo ``results/``): Lichens 1 -> ``Lichens_Dataset_1_MasterRun/wavelength_analysis/consensus_wavelengths.csv``
(same content as sheet ``Consensus_All_Configs`` of top10_wavelengths_analysis.xlsx; ranked by appearance_pct, avg_rank);
Collagen Pepsin -> ``Pepsin_Wavelength_Exports/frequency_analysis.csv`` (sheet ``Frequency_Analysis`` of
pepsin_wavelengths.xlsx; ranked by importance_score). No selection file is known for Lichens 2 or Collagen Acetic
(overlap is skipped there). Spatial models return a [P, P] map and are not supported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MAIN_REPO = Path("/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering")
SELECTIONS = {
    "lichens1": (MAIN_REPO / "results/Lichens_Dataset_1_MasterRun/wavelength_analysis/consensus_wavelengths.csv",
                 ["appearance_pct", "avg_rank"], [False, True], 13),
    "collagen_pepsin": (MAIN_REPO / "results/Pepsin_Wavelength_Exports/frequency_analysis.csv",
                        ["importance_score"], [False], 30),
}


def load_selection(dataset: str):
    """Returns (DataFrame[excitation_nm, emission_nm, score], K) sorted best-first, or None."""
    if dataset not in SELECTIONS:
        return None
    path, cols, asc, k = SELECTIONS[dataset]
    if not path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(path).sort_values(cols, ascending=asc).reset_index(drop=True)
    sc = df[cols[0]].astype(float)
    return df.assign(score=sc)[["excitation_nm", "emission_nm", "score"]], k


def best_config(results_root: Path, sweep: str, dataset: str, model: str):
    p = Path(results_root) / sweep / dataset / f"{model}.jsonl"
    if not p.exists():
        return None
    recs = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in r and not r.get("tiny"):
            recs.append(r)
    return max(recs, key=lambda r: r["inner_score"]) if recs else None


def snap(sel, data, tol_em=5.0):
    """Map each selected (ex, em) to the feature index with the same excitation and nearest emission (<= tol_em nm)."""
    ex, em = np.asarray(data.ex_nm, float), np.asarray(data.em_nm, float)
    out = []
    for e, m in zip(sel.excitation_nm, sel.emission_nm):
        cand = np.flatnonzero(np.isclose(ex, e))
        if len(cand) == 0:
            out.append(-1)
            continue
        j = cand[np.argmin(np.abs(em[cand] - m))]
        out.append(int(j) if abs(em[j] - m) <= tol_em else -1)
    return np.array(out)


def to_grid(data, v: np.ndarray) -> np.ndarray:
    g = np.full((len(data.excitations), len(data.emissions)), np.nan)
    g[data.grid_index[:, 0], data.grid_index[:, 1]] = v
    return g


def top_features(data, v, k=10):
    order = np.argsort(-np.asarray(v))[:k]
    return [dict(rank=i + 1, ex_nm=float(data.ex_nm[j]), em_nm=float(data.em_nm[j]), score=float(v[j]), feature=int(j))
            for i, j in enumerate(order)]


def overlap(data, v, sel_df, k):
    """Overlap between top-k features of ``v`` and the top-k selected features (same ex, em within tolerance)."""
    from scipy.stats import hypergeom, spearmanr
    idx = snap(sel_df.iloc[:k], data)
    idx = idx[idx >= 0]
    topk = set(np.argsort(-v)[:k].tolist())
    n_inter = len(topk & set(idx.tolist()))
    F = len(v)
    p = float(hypergeom.sf(n_inter - 1, F, len(idx), k)) if len(idx) else float("nan")
    sel_score = np.zeros(F)
    for j, s in zip(snap(sel_df, data), sel_df.score):
        if j >= 0:
            sel_score[j] = max(sel_score[j], s)
    rho = float(spearmanr(v, sel_score).statistic) if sel_score.std() > 0 else float("nan")
    return dict(k=int(k), n_selected_matched=int(len(idx)), n_overlap=int(n_inter), n_features=int(F),
                expected_overlap=float(k * len(idx) / F), hypergeom_p=p, spearman_importance_vs_selection=rho)


def plot_heat(data, vecs: dict, sel_idx, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(vecs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 2.6), squeeze=False)
    for ax, (name, v) in zip(axes[0], vecs.items()):
        g = to_grid(data, v)
        im = ax.imshow(g, aspect="auto", origin="lower", cmap="magma")
        ax.set_xticks(np.arange(0, len(data.emissions), max(1, len(data.emissions) // 8)))
        ax.set_xticklabels([f"{data.emissions[i]:.0f}" for i in ax.get_xticks().astype(int)], fontsize=6)
        ax.set_yticks(range(len(data.excitations)))
        ax.set_yticklabels([f"{e:.0f}" for e in data.excitations], fontsize=6)
        ax.set_xlabel("Emission (nm)", fontsize=7)
        ax.set_ylabel("Excitation (nm)", fontsize=7)
        ax.set_title(name, fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        if sel_idx is not None and len(sel_idx):
            ei, mi = data.grid_index[sel_idx, 0], data.grid_index[sel_idx, 1]
            ax.scatter(mi, ei, s=26, facecolors="none", edgecolors="cyan", linewidths=0.9, label="spectral-select")
            ax.legend(fontsize=5, loc="upper right")
    fig.suptitle(title, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="meas_tokens")
    ap.add_argument("--sweep", default="final")
    ap.add_argument("--results-root", default="results/me_attention")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="override epochs (0 = config's own)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-pixels", type=int, default=4000, help="test pixels used to average attention")
    ap.add_argument("--out", default="publications/me_attention/generated/importance")
    a = ap.parse_args(argv)

    import torch
    from me_attention import data as D
    from me_attention.models.registry import MODELS
    from me_attention.protocol import inner_split, outer_folds

    spec = MODELS[a.model]
    if spec.track != "spectral":
        raise SystemExit("importance.py supports spectral-track models only")
    data = D.load(a.dataset)
    rec = best_config(Path(a.results_root), a.sweep, a.dataset, a.model)
    cfg = dict(rec["best_config"]) if rec else dict(spec.default)
    src = f"best record (inner {rec['inner_score']:.3f}, r{rec['repeat']} f{rec['fold']})" if rec else "model default"
    if a.epochs:
        cfg["epochs"] = a.epochs
        cfg["warmup_epochs"] = min(int(cfg.get("warmup_epochs", 3)), max(0, a.epochs // 4))
    seed = int(hashlib.md5(f"imp|{a.dataset}|{a.model}|{a.repeat}|{a.fold}".encode()).hexdigest()[:8], 16)
    tr, te = outer_folds(data, a.repeat)[a.fold]
    itr, iva = inner_split(data, tr, seed)
    print(f"[importance] {a.dataset}/{a.model}: config from {src}; train={len(itr)} val={len(iva)} test={len(te)}")
    est = spec.build(dict(cfg), data.meta(), seed, a.device)
    # train on inner-train only; the inner-val specimens are used for early stopping and must not be in the fit set
    est.fit(data.X[itr], data.y[itr], data.groups[itr], data.X[iva], data.y[iva])
    mod = est.module
    rng = np.random.default_rng(0)
    sub = te if len(te) <= a.max_pixels else np.sort(rng.choice(te, a.max_pixels, replace=False))
    x = torch.as_tensor(est.pre.transform(data.X[sub]), dtype=torch.float32, device=a.device)
    vecs = {}
    for mode in ("rollout", "last"):
        try:
            # chunked: attention maps are O(B * heads * F^2), a single 20k-pixel batch exceeds mps memory
            parts = [(np.asarray(mod.attention_importance(x[i:i + 256], mode), float), len(x[i:i + 256]))
                     for i in range(0, len(x), 256)]
            vecs[f"attention ({mode})"] = sum(v * n for v, n in parts) / sum(n for _, n in parts)
        except Exception as ex:
            print(f"  attention_importance[{mode}] unavailable: {type(ex).__name__}: {ex}")
    gi = mod.feature_importance() if hasattr(mod, "feature_importance") else None
    if gi is not None:
        vecs["gate"] = np.asarray(gi, float)
    vecs = {k: v for k, v in vecs.items() if v.ndim == 1 and len(v) == data.n_features}
    if not vecs:
        raise SystemExit("model exposes no per-feature importance")

    out = Path(a.out) / f"{a.dataset}_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    selection = load_selection(a.dataset)
    sel_idx = None
    result = dict(dataset=a.dataset, model=a.model, config=cfg, config_source=src, repeat=a.repeat, fold=a.fold,
                  n_features=data.n_features, importances={}, selection_source=None)
    if selection:
        sel_df, k = selection
        sel_idx = snap(sel_df.iloc[:k], data)
        sel_idx = sel_idx[sel_idx >= 0]
        result["selection_source"] = str(SELECTIONS[a.dataset][0])
    for name, v in vecs.items():
        ent = dict(top10=top_features(data, v), scores=v.tolist())
        if selection:
            ent["overlap"] = overlap(data, v, *selection)
        result["importances"][name] = ent
        tag = name.replace(" ", "_").replace("(", "").replace(")", "")
        md = ["| rank | ex (nm) | em (nm) | score |", "|---|---|---|---|"] + [
            f"| {t['rank']} | {t['ex_nm']:.0f} | {t['em_nm']:.0f} | {t['score']:.4g} |" for t in ent["top10"]]
        (out / f"top10_{tag}.md").write_text("\n".join(md) + "\n")
        print(f"  top-10 [{name}]: " + ", ".join(f"({t['ex_nm']:.0f},{t['em_nm']:.0f})" for t in ent["top10"]))
        if "overlap" in ent:
            print(f"    overlap with spectral-select: {ent['overlap']}")
    plot_heat(data, vecs, sel_idx, out / "importance_heatmap", f"{a.dataset} / {a.model} ({src})")
    (out / "importance.json").write_text(json.dumps(result, indent=1))
    print(f"[importance] wrote {out}")


if __name__ == "__main__":
    main()
