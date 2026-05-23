"""Run SOTA-baseline comparison on a chosen dataset.

For each (method, K, seed):
    1. Select K bands.
    2. Evaluate:
        - For labeled datasets: KNN accuracy on the selected bands.
        - For Drop Data: ARI of Ward(K-band drop-means, k=3) vs drop_types.
    3. Record selection time, indices, metrics.

Outputs:
    {OUT_DIR}/comparison_results.csv     — long-form: dataset, method, K, seed, accuracy, ARI, ...
    {OUT_DIR}/selections.json            — dict of (method, K, seed) -> list of (ex, em)
    {OUT_DIR}/method_summary.csv         — mean ± std per (method, K)

Usage:
    python3 revision/baselines/run_comparison.py --dataset drop_data_full_cr \
        --methods variance pca_loading sam_greedy spa mcuve issc bsnet_fc random \
        --K 3 5 7 10 13 18 25 --seeds 5

By default the Drop Data sweep runs with the sensible defaults baked in.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revision.baselines import ALL_METHODS, METHOD_TYPE, SUPERVISED
from revision.baselines.data_adapter import DATASET_LOADERS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "revision" / "baselines" / "results"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_selection_drop_data(selected: np.ndarray, data: dict,
                                  seed: int = 0) -> dict:
    """Evaluate a Drop Data selection.

    Three complementary metrics, because no single metric is sufficient:

    1. **knn_acc** (primary, R1.6 / R2.4 friendly) - per-pixel KNN-5 accuracy
       on the K selected bands, predicting per-pixel type label. Requires
       the selection to be *informative* across multiple drops; can't be
       gamed by including noise bands.
    2. **ari_drop_means** (Ward k=3 on per-drop-mean K-D vectors vs Ward(full)
       types) - the original, but flawed: rewards including one strong band
       and noise. Kept for transparency.
    3. **fratio_kband** - sum of per-band F-ratio across the K bands; reflects
       total discriminative budget. Sums dim noise bands as ~0, sums info-
       rich bands as high values. Maps directly to the AE-perturb design.
    """
    from itertools import permutations
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    out = {}
    drop_types_true = data["drop_types"]

    # 1. ARI (kept for transparency)
    sub = data["drop_mean_spectra_full"][:, selected]
    if sub.shape[1] >= 2 and sub.shape[0] >= 4:
        ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
        pred = ward.fit_predict(sub)
        ari = adjusted_rand_score(drop_types_true, pred)
        nmi = normalized_mutual_info_score(drop_types_true, pred)
        best = 0
        for perm in permutations(range(3)):
            remapped = np.array([perm[p] for p in pred])
            best = max(best, int(np.sum(remapped == drop_types_true)))
        out.update({"ari_drop_means": float(ari),
                    "nmi_drop_means": float(nmi),
                    "n_drops_correct": int(best)})
    else:
        out.update({"ari_drop_means": np.nan, "nmi_drop_means": np.nan,
                    "n_drops_correct": np.nan})

    # 2. Per-pixel KNN accuracy (PRIMARY)
    F = data["features"][:, selected]
    y = data["labels"]
    keep = y >= 0
    F, y = F[keep], y[keep]
    if F.shape[1] >= 1 and len(np.unique(y)) >= 2 and len(y) > 50:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(F, y):
            sc = StandardScaler().fit(F[tr])
            Xtr = sc.transform(F[tr])
            Xte = sc.transform(F[te])
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
            knn.fit(Xtr, y[tr])
            accs.append(float(knn.score(Xte, y[te])))
        out["knn_acc"] = float(np.mean(accs))
        out["knn_acc_std"] = float(np.std(accs))
    else:
        out["knn_acc"] = np.nan
        out["knn_acc_std"] = np.nan

    # 3. Sum F-ratio across selected bands
    F_full = data["features"]
    drop_id = data["drop_id"]
    # Compute per-band F-ratio (between-type variance / within-type variance)
    types = data["drop_types"]
    # Restrict to in-drop pixels only and assign per-pixel type
    in_drop = drop_id > 0
    types_per_pixel = np.array([types[d-1] if d > 0 else -1 for d in drop_id])
    vals_in = F_full[in_drop]
    y_in = types_per_pixel[in_drop]
    fratios = np.zeros(F_full.shape[1])
    overall_mean = vals_in.mean(axis=0)
    n_total = len(y_in)
    n_classes = len(np.unique(y_in))
    if n_classes >= 2 and n_total > n_classes:
        ssb = np.zeros(F_full.shape[1])
        ssw = np.zeros(F_full.shape[1])
        for c in np.unique(y_in):
            mask_c = y_in == c
            nc = mask_c.sum()
            if nc == 0:
                continue
            mean_c = vals_in[mask_c].mean(axis=0)
            ssb += nc * (mean_c - overall_mean) ** 2
            ssw += ((vals_in[mask_c] - mean_c) ** 2).sum(axis=0)
        df_b = n_classes - 1
        df_w = n_total - n_classes
        fratios = (ssb / max(df_b, 1)) / (ssw / max(df_w, 1) + 1e-12)
    out["sum_fratio"] = float(fratios[selected].sum())
    out["mean_fratio"] = float(fratios[selected].mean())
    out["max_fratio"] = float(fratios[selected].max())

    return out


def evaluate_selection_labeled(selected: np.ndarray, data: dict,
                                seed: int = 0) -> dict:
    """KNN accuracy on the selected bands (5-fold CV)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    F = data["features"][:, selected]
    y = data["labels"]
    # Use only pixels with non-negative labels
    keep = y >= 0
    F, y = F[keep], y[keep]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(F, y):
        sc = StandardScaler().fit(F[tr])
        Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        knn.fit(Xtr, y[tr])
        accs.append(float(knn.score(Xte, y[te])))
    return {
        "knn_acc_mean": float(np.mean(accs)),
        "knn_acc_std": float(np.std(accs)),
    }


# ---------------------------------------------------------------------------
def is_drop_dataset(name: str) -> bool:
    return name.startswith("drop_data")


def run(dataset_name: str, method_names: list[str],
        K_list: list[int], n_seeds: int, out_dir: Path,
        verbose: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] Loading dataset: {dataset_name}")
    data = DATASET_LOADERS[dataset_name]()
    n_total_bands = data["features"].shape[1]
    band_catalog = data["band_catalog"]
    print(f"[run] Features: {data['features'].shape}, "
          f"labels unique: {sorted(set(data['labels'].tolist()))}, "
          f"n_bands={n_total_bands}")

    drop_mode = is_drop_dataset(dataset_name)
    if drop_mode:
        evaluate = lambda sel, d, seed=0: evaluate_selection_drop_data(sel, d, seed=seed)
    else:
        evaluate = lambda sel, d, seed=0: evaluate_selection_labeled(sel, d, seed=seed)

    rows = []
    selections = {}

    for method_name in method_names:
        if method_name not in ALL_METHODS:
            print(f"[run] WARN: unknown method '{method_name}', skip")
            continue
        is_sup = method_name in SUPERVISED
        fn = ALL_METHODS[method_name]

        for K in K_list:
            for seed in range(n_seeds):
                t0 = time.time()
                kwargs = {"seed": seed}
                if is_sup:
                    kwargs["labels"] = data["labels"]
                # ae_perturb adapter needs the band catalog + dataset name
                if method_name == "ae_perturb":
                    kwargs["dataset"] = dataset_name
                    kwargs["band_catalog"] = band_catalog
                # Deterministic methods don't change with seed - run once
                deterministic = method_name in ("variance", "pca_loading", "spa", "ae_perturb")
                if deterministic and seed > 0:
                    continue
                try:
                    sel = fn(data["features"], K=K, **kwargs)
                except Exception as e:
                    print(f"[run]   {method_name} K={K} seed={seed} FAILED: {type(e).__name__}: {e}")
                    continue
                dt = time.time() - t0
                metrics = evaluate(sel, data, seed=seed)

                rec = {
                    "dataset": dataset_name,
                    "method": method_name,
                    "method_type": METHOD_TYPE[method_name],
                    "K": K,
                    "seed": seed,
                    "select_time_s": round(dt, 3),
                }
                rec.update(metrics)
                rows.append(rec)
                selections[f"{method_name}__K{K}__s{seed}"] = [
                    list(band_catalog[i]) for i in sel.tolist()
                ]
                if verbose:
                    primary = metrics.get("knn_acc", metrics.get("knn_acc_mean", np.nan))
                    secondary = metrics.get("ari_drop_means", np.nan)
                    if not np.isnan(secondary):
                        msg = f"knn={primary:.3f} ari={secondary:.3f}"
                    else:
                        msg = f"knn={primary:.3f}"
                    print(f"[run]   {method_name:14s} K={K:>2}  seed={seed}  "
                          f"{msg}  ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    try:
        rel_csv = csv_path.resolve().relative_to(ROOT)
    except ValueError:
        rel_csv = csv_path
    print(f"[run] Wrote {rel_csv}  ({len(df)} rows)")

    sel_path = out_dir / "selections.json"
    sel_path.write_text(json.dumps(selections, indent=2))
    try:
        rel_sel = sel_path.resolve().relative_to(ROOT)
    except ValueError:
        rel_sel = sel_path
    print(f"[run] Wrote {rel_sel}")

    # Method summary
    metric_col = "knn_acc" if drop_mode else "knn_acc_mean"
    summary = df.groupby(["method", "K"]).agg(
        mean=(metric_col, "mean"),
        std=(metric_col, "std"),
        n=(metric_col, "size"),
        mean_time=("select_time_s", "mean"),
    ).reset_index()
    summary_path = out_dir / "method_summary.csv"
    summary.to_csv(summary_path, index=False)
    try:
        rel_sum = summary_path.resolve().relative_to(ROOT)
    except ValueError:
        rel_sum = summary_path
    print(f"[run] Wrote {rel_sum}")

    print("\nMethod summary (mean ± std):")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="drop_data_full_cr",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--methods", nargs="+",
                        default=["variance", "pca_loading", "sam_greedy",
                                 "spa", "mcuve", "issc", "bsnet_fc",
                                 "sparse_lasso", "random"])
    parser.add_argument("--K", nargs="+", type=int,
                        default=[3, 5, 7, 10, 13, 18, 25])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.dataset
    run(args.dataset, args.methods, args.K, args.seeds, out_dir,
        verbose=not args.quiet)


if __name__ == "__main__":
    main()
