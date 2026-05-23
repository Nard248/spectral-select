#!/usr/bin/env python3
"""
Drop Data — Selection Sweep
============================
For each preprocessing variant, runs multiple wavelength-selection methods at
several band counts (n=3..10) and computes light blind metrics so the user
can compare the AE perturbation method against classical baselines.

Methods:
  ae_perturb     - spectral_select Analyzer (the protagonist)
  variance       - rank bands by their variance over in-mask pixels
  pca_loading    - rank bands by L2 norm of loadings in top PCA components
  sam_greedy     - greedy max-min spectral angle (forward selection)
  spa            - Successive Projections Algorithm (classical HSI baseline)
  mcuve          - Monte Carlo Uninformative Variable Elimination
  random         - uniform random selection (sanity check, 5 seeds)

Blind metrics (per drop-mean-spectrum in the selected bands):
  silhouette_max        - best silhouette over k in [2 .. n_drops//2]
  silhouette_k          - k that achieved silhouette_max
  median_pairwise_sam   - median spectral-angle between drop-pairs (degrees)
  intra_drop_mean_var   - mean per-band variance inside drops, normalized

Outputs:
  results/Drop_Data_Selection_Sweep/
      sweep_results.csv               - one row per (variant, method, n)
      <variant>/<method>/              - per-method ranking
      figures/                        - comparison plots across methods/variants
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_select import Analyzer, Config, SpectraData  # noqa: E402

PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Selection_Sweep"
DROP_LABELS_PATH = PROC_ROOT / "drop_labels.npy"

VARIANTS = ["raw", "dark", "dark_norm", "dark_norm_mask", "full"]
METHODS = ["ae_perturb", "variance", "pca_loading", "sam_greedy", "spa", "mcuve", "random"]
N_BAND_GRID = list(range(3, 11))
RANDOM_SEEDS = [0, 1, 2, 3, 4]
AE_TRAIN_EPOCHS = 25
AE_PATCH_SIZE = 16
AE_PATCH_STRIDE = 8


@dataclass(frozen=True)
class Band:
    excitation_nm: float
    emission_idx: int
    emission_nm: float

    def label(self) -> str:
        return f"{int(self.excitation_nm)}/{self.emission_nm:.0f}"


def build_band_catalog(data: SpectraData) -> list[Band]:
    bands: list[Band] = []
    for ex in data.excitation_wavelengths:
        ed = data.get_excitation(ex)
        for i, em in enumerate(ed.emission_wavelengths):
            bands.append(Band(float(ex), i, float(em)))
    return bands


def stack_pixel_features(
    data: SpectraData, bands: list[Band], pixel_mask: np.ndarray | None
) -> np.ndarray:
    H, W = data.spatial_shape
    if pixel_mask is None:
        pixel_mask = np.ones((H, W), dtype=bool)
    else:
        pixel_mask = pixel_mask.astype(bool)
    flat_idx = np.flatnonzero(pixel_mask.ravel())
    out = np.empty((flat_idx.size, len(bands)), dtype=np.float32)
    by_ex: dict[float, np.ndarray] = {}
    for ex in data.excitation_wavelengths:
        cube = data.get_excitation(ex).cube
        by_ex[float(ex)] = cube.reshape(-1, cube.shape[2])
    for c, b in enumerate(bands):
        out[:, c] = by_ex[b.excitation_nm][flat_idx, b.emission_idx]
    return out


def per_drop_mean_spectra(
    data: SpectraData, bands: list[Band], drop_labels: np.ndarray
) -> np.ndarray:
    n_drops = int(drop_labels.max())
    out = np.zeros((n_drops, len(bands)), dtype=np.float32)
    by_ex: dict[float, np.ndarray] = {}
    for ex in data.excitation_wavelengths:
        cube = data.get_excitation(ex).cube
        by_ex[float(ex)] = cube
    for d in range(1, n_drops + 1):
        sel = drop_labels == d
        if not sel.any():
            continue
        for c, b in enumerate(bands):
            out[d - 1, c] = float(by_ex[b.excitation_nm][sel, b.emission_idx].mean())
    return out


def rank_variance(features: np.ndarray) -> np.ndarray:
    return np.argsort(-features.var(axis=0))


def rank_pca_loading(features: np.ndarray, n_components: int = 5) -> np.ndarray:
    n_components = min(n_components, features.shape[1] - 1)
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(features)
    weight = (pca.explained_variance_ratio_[:, None] *
              np.abs(pca.components_)).sum(axis=0)
    return np.argsort(-weight)


def rank_sam_greedy(features: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    F = features
    norms = np.linalg.norm(F, axis=0) + 1e-12
    Fn = F / norms
    n_bands = F.shape[1]
    chosen = [int(np.argmax(F.var(axis=0)))]
    while len(chosen) < n_bands:
        cos = Fn.T @ Fn[:, chosen]
        cos = np.clip(cos, -1.0, 1.0)
        angles = np.arccos(np.abs(cos))
        min_angle = angles.min(axis=1)
        min_angle[chosen] = -np.inf
        noise = rng.uniform(-1e-9, 1e-9, size=min_angle.shape)
        chosen.append(int(np.argmax(min_angle + noise)))
    return np.array(chosen)


def rank_spa(features: np.ndarray) -> np.ndarray:
    F = features.astype(np.float64, copy=True)
    n_bands = F.shape[1]
    norms = np.linalg.norm(F, axis=0)
    chosen = [int(np.argmax(norms))]
    R = F.copy()
    while len(chosen) < n_bands:
        v = R[:, chosen[-1]:chosen[-1] + 1]
        denom = float(v.T @ v) + 1e-12
        proj_coef = (v.T @ R) / denom
        R = R - v @ proj_coef
        norms = np.linalg.norm(R, axis=0)
        norms[chosen] = -np.inf
        chosen.append(int(np.argmax(norms)))
    return np.array(chosen)


def rank_mcuve(features: np.ndarray, n_iter: int = 50,
               subset_frac: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N, B = features.shape
    target = PCA(n_components=1, random_state=0).fit_transform(features)[:, 0]
    coefs = np.zeros((n_iter, B), dtype=np.float64)
    for i in range(n_iter):
        idx = rng.choice(N, size=max(int(subset_frac * N), 100), replace=False)
        X = features[idx]
        y = target[idx]
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        denom = (Xc * Xc).sum(axis=0) + 1e-12
        coefs[i] = (Xc * yc[:, None]).sum(axis=0) / denom
    mean = coefs.mean(axis=0)
    std = coefs.std(axis=0) + 1e-12
    stability = np.abs(mean) / std
    return np.argsort(-stability)


def rank_random(n_bands: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(n_bands)
    rng.shuffle(perm)
    return perm


def fit_ae_get_influence(
    data: SpectraData, variant_name: str
) -> dict[float, np.ndarray]:
    cfg = Config(
        sample_name=f"DropData_sweep_{variant_name}",
        n_bands_to_select=10,
        device="mps",
        training_epochs=AE_TRAIN_EPOCHS,
        patch_size=AE_PATCH_SIZE,
        patch_stride=AE_PATCH_STRIDE,
        n_baseline_patches=30,
        n_important_dimensions=12,
        save_visualizations=False,
        save_tiff_layers=False,
        save_detailed_results=False,
        output_dir=OUT_ROOT / variant_name / "ae_perturb",
    )
    analyzer = Analyzer(cfg)
    analyzer.fit(data)
    inf = analyzer.influence_matrix or {}
    out: dict[float, np.ndarray] = {}
    for ex in data.excitation_wavelengths:
        if ex in inf:
            out[float(ex)] = np.asarray(inf[ex], dtype=float)
        elif str(ex) in inf:
            out[float(ex)] = np.asarray(inf[str(ex)], dtype=float)
    return out


def ae_ranking_to_band_order(
    influence: dict[float, np.ndarray], bands: list[Band]
) -> np.ndarray:
    scores = []
    for b in bands:
        ex = b.excitation_nm
        if ex in influence and b.emission_idx < len(influence[ex]):
            scores.append(float(influence[ex][b.emission_idx]))
        else:
            scores.append(0.0)
    return np.argsort(-np.array(scores))


def metrics_for_selection(
    drop_spectra_full: np.ndarray, selected_indices: np.ndarray
) -> dict:
    sub = drop_spectra_full[:, selected_indices]
    n_drops = sub.shape[0]
    if n_drops < 4 or sub.shape[1] < 2:
        return dict(silhouette_max=np.nan, silhouette_k=0,
                    median_pairwise_sam=np.nan, intra_drop_mean_var=np.nan)
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12
    unit = sub / norms
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    upper = np.triu_indices(n_drops, k=1)
    angles = np.degrees(np.arccos(np.abs(cos[upper])))
    median_sam = float(np.median(angles))

    best_sil, best_k = -1.0, 0
    max_k = min(n_drops // 2, 8)
    for k in range(2, max(3, max_k + 1)):
        if k >= n_drops:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sub)
            if len(set(km.labels_)) < 2:
                continue
            s = silhouette_score(sub, km.labels_)
        except Exception:
            continue
        if s > best_sil:
            best_sil, best_k = float(s), int(k)

    intra_var_pct = float(sub.var(axis=0).mean() / (sub.var() + 1e-12))
    return dict(
        silhouette_max=best_sil if best_sil > -1 else np.nan,
        silhouette_k=best_k,
        median_pairwise_sam=median_sam,
        intra_drop_mean_var=intra_var_pct,
    )


def run_variant(variant: str, drop_labels: np.ndarray) -> list[dict]:
    pkl = PROC_ROOT / variant / "spectra_data.pkl"
    if not pkl.exists():
        print(f"[sweep] skip {variant}: no pickle")
        return []
    print(f"\n[sweep] === variant: {variant} ===")
    data = SpectraData.from_pickle(pkl)
    bands = build_band_catalog(data)
    print(f"[sweep] {len(bands)} candidate bands ({data.n_excitations} excitations)")

    pixel_mask = data.mask.astype(bool) if data.mask is not None else None
    if pixel_mask is None or pixel_mask.sum() < 200:
        H, W = data.spatial_shape
        feature_mask = np.ones((H, W), dtype=bool)
        feature_mask[187:, :] = False
    else:
        feature_mask = pixel_mask
    features = stack_pixel_features(data, bands, feature_mask)
    print(f"[sweep] features shape: {features.shape}")

    drop_spectra = per_drop_mean_spectra(data, bands, drop_labels)
    print(f"[sweep] drop spectra: {drop_spectra.shape}")

    rankings: dict[str, np.ndarray] = {}
    print("[sweep] variance ranking...")
    rankings["variance"] = rank_variance(features)
    print("[sweep] pca_loading ranking...")
    rankings["pca_loading"] = rank_pca_loading(features)
    print("[sweep] sam_greedy ranking...")
    rankings["sam_greedy"] = rank_sam_greedy(features)
    print("[sweep] spa ranking...")
    rankings["spa"] = rank_spa(features)
    print("[sweep] mcuve ranking...")
    rankings["mcuve"] = rank_mcuve(features)

    print(f"[sweep] AE training ({AE_TRAIN_EPOCHS} epochs)...")
    t0 = time.time()
    influence = fit_ae_get_influence(data, variant)
    print(f"[sweep] AE training took {time.time() - t0:.1f}s")
    rankings["ae_perturb"] = ae_ranking_to_band_order(influence, bands)

    rows: list[dict] = []
    for method, ranking in rankings.items():
        for n in N_BAND_GRID:
            sel = ranking[:n]
            m = metrics_for_selection(drop_spectra, sel)
            rows.append({
                "variant": variant,
                "method": method,
                "n_bands": n,
                "selected_bands": [bands[i].label() for i in sel],
                **m,
            })
    for seed in RANDOM_SEEDS:
        ranking = rank_random(len(bands), seed)
        for n in N_BAND_GRID:
            sel = ranking[:n]
            m = metrics_for_selection(drop_spectra, sel)
            rows.append({
                "variant": variant,
                "method": f"random_s{seed}",
                "n_bands": n,
                "selected_bands": [bands[i].label() for i in sel],
                **m,
            })

    var_dir = OUT_ROOT / variant
    var_dir.mkdir(parents=True, exist_ok=True)
    for method, ranking in rankings.items():
        m_dir = var_dir / method
        m_dir.mkdir(exist_ok=True)
        full_label = [bands[i].label() for i in ranking]
        (m_dir / "ranking.json").write_text(json.dumps({
            "method": method,
            "ranked_bands": full_label,
            "ranking_indices": ranking.tolist(),
        }, indent=2))
    np.save(var_dir / "drop_mean_spectra.npy", drop_spectra)
    with open(var_dir / "bands.json", "w") as f:
        json.dump([asdict(b) for b in bands], f, indent=2)
    return rows


def write_summary_visualization(df: pd.DataFrame) -> None:
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df_plot = df.copy()
    df_plot["method_group"] = df_plot["method"].apply(
        lambda m: "random" if m.startswith("random_") else m
    )
    for variant, sub in df_plot.groupby("variant"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for metric, ax, ylabel in [
            ("silhouette_max", axes[0], "Silhouette (drops)"),
            ("median_pairwise_sam", axes[1], "Median pairwise SAM (deg)"),
            ("intra_drop_mean_var", axes[2], "Intra-drop var fraction"),
        ]:
            for method, msub in sub.groupby("method_group"):
                if method == "random":
                    g = msub.groupby("n_bands")[metric].agg(["mean", "std"])
                    ax.plot(g.index, g["mean"], label="random", color="gray",
                            linewidth=1.5, linestyle=":")
                    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                                    color="gray", alpha=0.15)
                else:
                    g = msub.sort_values("n_bands")
                    ax.plot(g["n_bands"], g[metric], marker="o", linewidth=1.6,
                            label=method, alpha=0.85)
            ax.set_xlabel("n bands")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Drop Data selection sweep - variant: {variant}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"sweep_{variant}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    drop_labels = np.load(DROP_LABELS_PATH)
    print(f"[sweep] drop labels: {drop_labels.shape}, n_drops={int(drop_labels.max())}")
    all_rows: list[dict] = []
    for v in VARIANTS:
        rows = run_variant(v, drop_labels)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / "sweep_results_partial.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "sweep_results.csv", index=False)
    print(f"\n[sweep] wrote {OUT_ROOT / 'sweep_results.csv'} ({len(df)} rows)")
    write_summary_visualization(df)
    print(f"[sweep] wrote per-variant summary plots to {OUT_ROOT / 'figures'}")


if __name__ == "__main__":
    main()
