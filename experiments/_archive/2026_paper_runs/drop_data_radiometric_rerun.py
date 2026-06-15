#!/usr/bin/env python3
"""
Drop Data — Radiometric Re-run (exposure + lamp-power normalized)
=================================================================
The original Drop Data run had no per-excitation radiometric metadata, so it
fell back to ``per_cube_p99`` normalization — dividing each excitation cube by
its OWN 99th percentile. That collapses every excitation onto its own [0,1]
scale and destroys the cross-excitation intensity structure an EEM encodes
(the very signal the method is meant to find). The 7 excitations were shot at
exposures spanning ~250x (310:1500 .. 385:6 SPF) and lamp powers spanning ~77x
(400:60.5 .. 325:0.79), so the un-normalized counts are dominated by
acquisition artifacts rather than fluorescence.

Now that exposure (from filenames) and lamp power (260414 lamp scan) are
available, we replace p99 with the SAME radiometric correction the working
Collagen/Lichens datasets used (mehsi_preprocessor.HyperspectralProcessor):

    corrected = (raw - dark) / (exposure x lamp_power)        # per-excitation

followed by a SINGLE GLOBAL scale (one p99 over all in-drop voxels across all
excitations) so values sit in ~[0,1] for stable AE training WITHOUT re-erasing
the cross-excitation structure.

Two HDR strategies (user asked for both):
  best    - single best (longest non-saturating) bracket per excitation
  merged  - exposure-normalized HDR fusion of all 2-3 brackets per excitation
            (recovers the saturated 385/400/415 pixels the single-bracket path
            clips)

Variants per mode: rad_<mode>, rad_<mode>_detrend (+ poly background detrend).

For each variant we run the FULL method span (same 7 implementations as
drop_data_selection_sweep.py) plus the AE perturbation method, Ward k=3 ground
truth, F-ratio scoring and the blind silhouette/SAM/intra-var metrics, then
compare against the old p99 numbers.

Inputs : Data/processed/Drop Data/raw/*.npy   (cached, == 'top illumination 2')
         /Users/.../Downloads/260414 lamp scan.xlsx  (lamp power per excitation)
Outputs: Data/processed/Drop Data Radiometric/<variant>/spectra_data.pkl
         results/Drop_Data_Radiometric/
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for importing sibling script

from spectral_select import Analyzer, Config, SpectraData  # noqa: E402
from spectral_select.types import ExcitationData, LoadingOptions  # noqa: E402

# Reuse the *validated* ranking + metric implementations verbatim.
import drop_data_selection_sweep as sweep  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
CACHE_DIR = PROJECT_ROOT / "Data" / "processed" / "Drop Data" / "raw"
PROC_CROP_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data Cropped"
PROC_OUT_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data Radiometric"
INSPECT_DIR = PROJECT_ROOT / "results" / "Drop_Data_Inspection"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Radiometric"
RAW_DATASET_DIR = PROJECT_ROOT / "Data" / "Raw" / "Drop Data"

LAMP_SCAN_XLSX = Path("/Users/narekmeloyan/Downloads/260414 lamp scan.xlsx")

RULER_ROW_START = 175
EM_START_NM = 420.0
EM_STEP_NM = 10.0
N_BANDS_RAW = 31
RAYLEIGH_OFFSET_NM = 20.0
SAT_CEILING = 3886.0
SAT_RELATIVE = 0.999
DETREND_POLY_DEGREE = 2
N_BAND_GRID = list(range(3, 11))
RANDOM_SEEDS = [0, 1, 2, 3, 4]
AE_EPOCHS = 25
AE_PATCH_SIZE = 16
AE_PATCH_STRIDE = 8

SAMPLE_RE = re.compile(r"^(?P<ex>\d+)\s+(?P<expo>\d+)\s+SPF$")


@dataclass(frozen=True)
class VariantConfig:
    name: str
    hdr_mode: str        # "best" | "merged"
    poly_detrend: bool


VARIANTS = [
    VariantConfig("rad_best",            "best",   False),
    VariantConfig("rad_best_detrend",    "best",   True),
    VariantConfig("rad_merged",          "merged", False),
    VariantConfig("rad_merged_detrend",  "merged", True),
]


# --------------------------------------------------------------------------- #
# Metadata: exposure (filenames) + lamp power (scan)
# --------------------------------------------------------------------------- #
def load_lamp_power() -> dict[float, float]:
    df = pd.read_excel(LAMP_SCAN_XLSX)
    df.columns = [str(c).strip().lower() for c in df.columns]
    ex_col = next(c for c in df.columns if "exc" in c or "wave" in c)
    pw_col = next(c for c in df.columns if "pow" in c)
    power = {float(r[ex_col]): float(r[pw_col]) for _, r in df.iterrows()}
    return power


def brackets_for_excitation() -> dict[float, list[tuple[int, str]]]:
    """Map excitation -> [(exposure, stem), ...] from the cached cube names."""
    out: dict[float, list[tuple[int, str]]] = {}
    for p in sorted(CACHE_DIR.glob("*.npy")):
        m = SAMPLE_RE.match(p.stem)
        if not m:
            continue
        ex = float(m["ex"])
        out.setdefault(ex, []).append((int(m["expo"]), p.stem))
    for ex in out:
        out[ex].sort()
    return out


def load_recommended() -> dict[float, dict]:
    rec = json.loads((INSPECT_DIR / "recommended_cubes.json").read_text())
    return {float(k): v for k, v in rec["recommended_per_excitation"].items()}


def persist_metadata_files(power: dict[float, float],
                           recommended: dict[float, dict]) -> None:
    """Write metadata.xlsx + TLS Scans/average_power.xlsx so Drop Data matches
    the Collagen/Lichens layout that HyperspectralProcessor expects."""
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DATASET_DIR / "TLS Scans").mkdir(parents=True, exist_ok=True)
    exp_rows = [{"Excitation": int(ex), "Exposure": int(recommended[ex]["exposure_ms"])}
                for ex in sorted(recommended)]
    pd.DataFrame(exp_rows).to_excel(RAW_DATASET_DIR / "metadata.xlsx", index=False)
    pw_rows = [{"Excitation Wavelength (nm)": int(ex), "Average Power (W)": power[ex]}
               for ex in sorted(power)]
    pd.DataFrame(pw_rows).to_excel(
        RAW_DATASET_DIR / "TLS Scans" / "average_power.xlsx", index=False)
    print(f"[meta] wrote metadata.xlsx + TLS Scans/average_power.xlsx to {RAW_DATASET_DIR}")


# --------------------------------------------------------------------------- #
# Preprocessing helpers (cropped, radiometric)
# --------------------------------------------------------------------------- #
def crop(cube: np.ndarray) -> np.ndarray:
    return cube[:RULER_ROW_START, :, :]


def emission_wavelengths() -> list[float]:
    return [EM_START_NM + i * EM_STEP_NM for i in range(N_BANDS_RAW)]


def apply_rayleigh_cutoff(cube: np.ndarray, ex_nm: float, em_wls: list[float]):
    cutoff = ex_nm + RAYLEIGH_OFFSET_NM
    keep = np.array([wl > cutoff for wl in em_wls])
    return cube[:, :, keep], [wl for wl, k in zip(em_wls, keep) if k]


def fit_poly_per_band(cube: np.ndarray, fit_mask: np.ndarray,
                      degree: int = DETREND_POLY_DEGREE) -> np.ndarray:
    H, W, B = cube.shape
    yy, xx = np.mgrid[0:H, 0:W]
    yn = (yy / H).ravel()
    xn = (xx / W).ravel()
    terms = []
    for dy in range(degree + 1):
        for dx in range(degree + 1 - dy):
            terms.append((yn ** dy) * (xn ** dx))
    A = np.stack(terms, axis=1)
    flat_mask = fit_mask.ravel()
    A_fit = A[flat_mask]
    bg = np.zeros_like(cube)
    for b in range(B):
        y = cube[:, :, b].ravel()
        y_fit = y[flat_mask]
        if y_fit.size == 0 or not np.any(y_fit > 0):
            continue
        coef, *_ = np.linalg.lstsq(A_fit, y_fit, rcond=None)
        if not np.all(np.isfinite(coef)):
            continue
        bg[:, :, b] = (A @ coef).reshape(H, W)
    return bg


def build_corrected_cube(ex: float, mode: str, dark_full: np.ndarray,
                         brackets: dict, recommended: dict,
                         power: dict[float, float]):
    """Return (corrected_full_HxWxB, saturated_2d_full) with exposure & power
    normalization applied. Channel/excitation intensities are now comparable."""
    pw = power[ex]
    if mode == "best":
        stem = recommended[ex]["stem"]
        expo = float(recommended[ex]["exposure_ms"])
        raw = np.load(CACHE_DIR / f"{stem}.npy").astype(np.float32)
        sat2d = np.any(raw >= SAT_CEILING * SAT_RELATIVE, axis=2)
        c = np.maximum(raw - dark_full, 0.0) / (expo * pw)
        return c, sat2d

    # merged HDR: exposure-normalize each bracket, average the unsaturated ones
    rows = brackets[ex]
    shortest_expo, shortest_stem = min(rows, key=lambda t: t[0])
    H, W, B = dark_full.shape
    acc = np.zeros((H, W, B), dtype=np.float32)
    cnt = np.zeros((H, W, B), dtype=np.float32)
    for expo, stem in rows:
        raw = np.load(CACHE_DIR / f"{stem}.npy").astype(np.float32)
        unsat = raw < SAT_CEILING * SAT_RELATIVE
        cn = np.maximum(raw - dark_full, 0.0) / float(expo)
        acc += np.where(unsat, cn, 0.0)
        cnt += unsat.astype(np.float32)
    merged = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
    fully_sat = cnt == 0
    if fully_sat.any():
        raw_s = np.load(CACHE_DIR / f"{shortest_stem}.npy").astype(np.float32)
        cn_s = np.maximum(raw_s - dark_full, 0.0) / float(shortest_expo)
        merged = np.where(fully_sat, cn_s, merged)
    sat2d = fully_sat.any(axis=2)        # voxels never captured unsaturated
    corrected = merged / pw
    return corrected, sat2d


def assemble_variant(cfg: VariantConfig, dark_full: np.ndarray,
                     brackets: dict, recommended: dict, power: dict,
                     drop_mask: np.ndarray, drop_labels: np.ndarray):
    excitations_nm = sorted(recommended)
    cubes: dict[float, np.ndarray] = {}
    em_by_ex: dict[float, list[float]] = {}
    sat_union = np.zeros(drop_mask.shape, dtype=bool)

    for ex in excitations_nm:
        corrected_full, sat2d_full = build_corrected_cube(
            ex, cfg.hdr_mode, dark_full, brackets, recommended, power)
        cube = crop(corrected_full)
        sat_union |= crop(sat2d_full[:, :, None])[:, :, 0]
        cube, em_wls = apply_rayleigh_cutoff(cube, ex, emission_wavelengths())
        if cfg.poly_detrend:
            bg = fit_poly_per_band(cube, ~drop_mask)
            cube = np.maximum(cube - bg, 0.0)
        cubes[ex] = cube
        em_by_ex[ex] = em_wls

    # SINGLE global scale across ALL excitations (preserve cross-ex structure)
    pooled = np.concatenate([cubes[ex][drop_mask].ravel() for ex in excitations_nm])
    pooled = pooled[pooled > 0]
    scale = float(np.percentile(pooled, 99)) if pooled.size else 1.0
    if scale <= 0:
        scale = 1.0
    for ex in excitations_nm:
        cubes[ex] = (cubes[ex] / scale).astype(np.float32)

    spatial_mask = (drop_mask.astype(np.float32) * (~sat_union).astype(np.float32))

    excit = {
        ex: ExcitationData(
            excitation_nm=float(ex),
            cube=cubes[ex],
            emission_wavelengths=em_by_ex[ex],
            exposure_time=float(recommended[ex]["exposure_ms"]) / 1000.0,
        )
        for ex in excitations_nm
    }
    sd = SpectraData(
        excitations=excit,
        mask=spatial_mask,
        sample_name=f"DropDataRadiometric_{cfg.name}",
        loading_options=LoadingOptions(
            cutoff_offset=int(RAYLEIGH_OFFSET_NM),
            apply_rayleigh_cutoff=True,
            apply_second_order_cutoff=False,
            normalize_exposure=True,
            normalize_laser_power=True,
        ),
        metadata={
            "variant": cfg.name,
            "hdr_mode": cfg.hdr_mode,
            "poly_detrend": cfg.poly_detrend,
            "global_scale_p99": scale,
            "lamp_power": {str(k): v for k, v in power.items()},
            "exposures_ms": {str(k): recommended[k]["exposure_ms"] for k in recommended},
            "cropped_to_rows": RULER_ROW_START,
            "normalization": "radiometric (raw-dark)/(exposure*power), single global p99 scale",
        },
    )
    out_dir = PROC_OUT_ROOT / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sd.to_pickle(out_dir / "spectra_data.pkl")
    return sd


# --------------------------------------------------------------------------- #
# Scoring (Ward GT + F-ratio), reuses sweep's pure functions for the span
# --------------------------------------------------------------------------- #
def cluster_to_3types(spec: np.ndarray) -> np.ndarray:
    norm = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-12)
    Z = linkage(norm, method="ward")
    return fcluster(Z, t=3, criterion="maxclust") - 1


def f_ratio_per_band(spec: np.ndarray, types: np.ndarray, k: int = 3) -> np.ndarray:
    n_total, B = spec.shape
    overall = spec.mean(axis=0)
    between = np.zeros(B)
    within = np.zeros(B)
    for t in range(k):
        members = spec[types == t]
        if len(members) == 0:
            continue
        tm = members.mean(axis=0)
        between += len(members) * (tm - overall) ** 2
        within += ((members - tm) ** 2).sum(axis=0)
    return (between / max(k - 1, 1)) / (within / max(n_total - k, 1) + 1e-12)


def fit_ae_influence(data: SpectraData, variant: str) -> dict[float, np.ndarray]:
    cfg = Config(
        sample_name=f"DropDataRad_{variant}",
        n_bands_to_select=10,
        device="mps",
        training_epochs=AE_EPOCHS,
        patch_size=AE_PATCH_SIZE,
        patch_stride=AE_PATCH_STRIDE,
        n_baseline_patches=30,
        n_important_dimensions=12,
        normalization_method="max_per_excitation",
        save_visualizations=False,
        save_tiff_layers=False,
        save_detailed_results=False,
        output_dir=OUT_ROOT / variant / "ae_perturb",
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


def run_variant(cfg: VariantConfig, data: SpectraData,
                drop_mask: np.ndarray, drop_labels: np.ndarray) -> tuple[list[dict], dict]:
    bands = sweep.build_band_catalog(data)
    pixel_mask = data.mask.astype(bool) if data.mask is not None else drop_mask
    features = sweep.stack_pixel_features(data, bands, pixel_mask)
    drop_spectra = sweep.per_drop_mean_spectra(data, bands, drop_labels)
    print(f"[rad]   {cfg.name}: {len(bands)} bands, features {features.shape}, "
          f"drop spectra {drop_spectra.shape}")

    rankings: dict[str, np.ndarray] = {
        "variance": sweep.rank_variance(features),
        "pca_loading": sweep.rank_pca_loading(features),
        "sam_greedy": sweep.rank_sam_greedy(features),
        "spa": sweep.rank_spa(features),
        "mcuve": sweep.rank_mcuve(features),
    }
    t0 = time.time()
    influence = fit_ae_influence(data, cfg.name)
    rankings["ae_perturb"] = sweep.ae_ranking_to_band_order(influence, bands)
    print(f"[rad]   {cfg.name}: AE trained in {time.time() - t0:.1f}s")

    # Ward k=3 ground truth + F-ratio on the corrected drop spectra
    types = cluster_to_3types(drop_spectra)
    f_ratio = f_ratio_per_band(drop_spectra, types)
    f_order = np.argsort(np.argsort(-f_ratio))  # rank of each band (0 = best)

    rows: list[dict] = []

    def add_rows(method: str, ranking: np.ndarray):
        for n in N_BAND_GRID:
            sel = ranking[:n]
            m = sweep.metrics_for_selection(drop_spectra, sel)
            mean_f = float(f_ratio[sel].mean())
            rank_pct = 1.0 - (f_order[sel].mean() / max(len(bands) - 1, 1))
            rows.append({
                "variant": cfg.name, "hdr_mode": cfg.hdr_mode, "method": method,
                "n_bands": n, "mean_f": mean_f, "rank_percentile": float(rank_pct),
                "selected_bands": [bands[i].label() for i in sel], **m,
            })

    for method, ranking in rankings.items():
        add_rows(method, ranking)
    for seed in RANDOM_SEEDS:
        add_rows(f"random_s{seed}", sweep.rank_random(len(bands), seed))

    # persist per-variant artifacts
    var_dir = OUT_ROOT / cfg.name
    var_dir.mkdir(parents=True, exist_ok=True)
    np.save(var_dir / "drop_mean_spectra.npy", drop_spectra)
    np.save(var_dir / "drop_types.npy", types)
    pd.DataFrame({
        "excitation_nm": [b.excitation_nm for b in bands],
        "emission_nm": [b.emission_nm for b in bands],
        "f_ratio": f_ratio,
    }).sort_values("f_ratio", ascending=False).to_csv(
        var_dir / "f_ratio_table.csv", index=False)
    for method, ranking in rankings.items():
        (var_dir / f"ranking_{method}.json").write_text(json.dumps(
            [bands[i].label() for i in ranking], indent=1))

    info = {
        "variant": cfg.name,
        "ward_type_sizes": [int((types == t).sum()) for t in range(3)],
        "ae_top5": [bands[i].label() for i in rankings["ae_perturb"][:5]],
        "f_top5": [bands[i].label() for i in np.argsort(-f_ratio)[:5]],
        "drop_spectra": drop_spectra, "types": types, "f_ratio": f_ratio,
        "bands": bands,
    }
    return rows, info


# --------------------------------------------------------------------------- #
# Diagnostic figure (the cheap checkpoint)
# --------------------------------------------------------------------------- #
def diagnostic_figure(cfg: VariantConfig, info: dict, out_path: Path) -> None:
    spec, types, f_ratio, bands = (info["drop_spectra"], info["types"],
                                   info["f_ratio"], info["bands"])
    exes = sorted({b.excitation_nm for b in bands})
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    # (1) per-drop spectra colored by Ward type
    ax = axes[0]
    for d in range(spec.shape[0]):
        ax.plot(spec[d], color=colors[types[d] % 3], alpha=0.6, linewidth=0.8)
    ax.set_title(f"{cfg.name}: per-drop spectra (Ward color)")
    ax.set_xlabel("band index (ex x em)")
    ax.set_ylabel("corrected intensity")
    ax.grid(alpha=0.3)
    # (2) type-mean EEM-ish: mean spectrum per type
    ax = axes[1]
    for t in range(3):
        members = spec[types == t]
        if len(members):
            ax.plot(members.mean(axis=0), color=colors[t], linewidth=2,
                    label=f"type {t} (n={len(members)})")
    ax.set_title("type-mean spectra")
    ax.set_xlabel("band index (ex x em)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (3) F-ratio over bands, AE top-5 marked
    ax = axes[2]
    ax.plot(f_ratio, color="black", linewidth=0.9)
    ae_idx = [i for i, b in enumerate(bands)
              if b.label() in set(info["ae_top5"])]
    ax.scatter(ae_idx, f_ratio[ae_idx], color="red", zorder=5,
               label="AE top-5")
    ax.set_title(f"F-ratio per band (mean={f_ratio.mean():.1f})")
    ax.set_xlabel("band index (ex x em)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Comparison vs old p99 run
# --------------------------------------------------------------------------- #
def compare_to_old(df_new: pd.DataFrame) -> pd.DataFrame:
    """Compare AE-perturb mean-F of the radiometric variants against the old
    cropped p99 run (results/Drop_Data_Cropped_Sweep/all_results.csv)."""
    rows = []
    ae_new = df_new[df_new["method"] == "ae_perturb"]
    for variant, sub in ae_new.groupby("variant"):
        rows.append({
            "run": "radiometric", "variant": variant,
            "ae_meanF_all_k": round(float(sub["mean_f"].mean()), 3),
            "ae_meanF_k5": round(float(sub[sub["n_bands"] == 5]["mean_f"].iloc[0]), 3),
        })
    old_csv = PROJECT_ROOT / "results" / "Drop_Data_Cropped_Sweep" / "all_results.csv"
    if old_csv.exists():
        old = pd.read_csv(old_csv)
        for variant, sub in old.groupby("variant"):
            k5 = sub[sub["n_bands"] == 5]["mean_f"]
            rows.append({
                "run": "old_p99", "variant": variant,
                "ae_meanF_all_k": round(float(sub["mean_f"].mean()), 3),
                "ae_meanF_k5": round(float(k5.iloc[0]), 3) if len(k5) else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[rad] === Drop Data radiometric re-run ===")

    power = load_lamp_power()
    recommended = load_recommended()
    brackets = brackets_for_excitation()
    print(f"[rad] lamp power: { {int(k): v for k, v in sorted(power.items())} }")
    print(f"[rad] exposures(best): "
          f"{ {int(k): recommended[k]['exposure_ms'] for k in sorted(recommended)} }")
    persist_metadata_files(power, recommended)

    dark_full = np.load(CACHE_DIR / "Background.npy").astype(np.float32)
    drop_mask = np.load(PROC_CROP_ROOT / "drop_mask.npy").astype(bool)
    drop_labels = np.load(PROC_CROP_ROOT / "drop_labels.npy").astype(np.int32)
    assert drop_mask.shape[0] == RULER_ROW_START, drop_mask.shape
    print(f"[rad] drop mask {drop_mask.shape}, n_drops={int(drop_labels.max())}, "
          f"in-drop px={int(drop_mask.sum())}")

    all_rows: list[dict] = []
    diag = {}
    for cfg in VARIANTS:
        print(f"\n[rad] === variant {cfg.name} (hdr={cfg.hdr_mode}, "
              f"detrend={cfg.poly_detrend}) ===")
        sd = assemble_variant(cfg, dark_full, brackets, recommended, power,
                              drop_mask, drop_labels)
        rng = [float(sd.get_excitation(ex).cube.min()) for ex in sd.excitation_wavelengths]
        mx = [float(sd.get_excitation(ex).cube.max()) for ex in sd.excitation_wavelengths]
        print(f"[rad]   corrected cube ranges per-ex min={min(rng):.3g} max={max(mx):.3g}")
        rows, info = run_variant(cfg, sd, drop_mask, drop_labels)
        all_rows.extend(rows)
        diag[cfg.name] = info
        diagnostic_figure(cfg, info, OUT_ROOT / f"diag_{cfg.name}.png")
        print(f"[rad]   Ward type sizes={info['ward_type_sizes']}  "
              f"AE top5={info['ae_top5']}")
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / "sweep_results_partial.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "sweep_results.csv", index=False)
    print(f"\n[rad] wrote {OUT_ROOT / 'sweep_results.csv'} ({len(df)} rows)")

    cmp = compare_to_old(df)
    cmp.to_csv(OUT_ROOT / "radiometric_vs_old.csv", index=False)
    print("\n[rad] === AE-perturb mean F-ratio: radiometric vs old p99 ===")
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
