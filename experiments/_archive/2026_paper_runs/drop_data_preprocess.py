#!/usr/bin/env python3
"""
Drop Data — Preprocessing Sweep
================================
Produces 5 preprocessing variants of the Drop Data, each saved as a
SpectraData-compatible pickle. Downstream selection sweep runs once per
variant to compare what each preprocessing step contributes.

Variants (cumulative ablation):
  raw                       — no preprocessing (control)
  dark                      — Background.im3 subtracted, Rayleigh cutoff
  dark_norm                 — + per-cube p99 normalization
  dark_norm_mask            — + spatial mask (ruler + drop focus + saturation)
  full                      — + polynomial background detrend per cube

Inputs : Data/processed/Drop Data/raw/*.npy   (from drop_data_inspect.py)
         results/Drop_Data_Inspection/recommended_cubes.json
Outputs:
  Data/processed/Drop Data/<variant>/spectra_data.pkl
  Data/processed/Drop Data/drop_mask.npy            (shared across variants)
  results/Drop_Data_Preprocessing/<variant>/
      drop_spectra.png      (28 mean-spectrum overlays)
      mask_overlay.png
      summary.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_select.types import (  # noqa: E402
    ExcitationData,
    LoadingOptions,
    SpectraData,
)

CACHE_DIR = PROJECT_ROOT / "Data" / "processed" / "Drop Data" / "raw"
PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
INSPECTION_DIR = PROJECT_ROOT / "results" / "Drop_Data_Inspection"
PREP_OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Preprocessing"

EM_START_NM = 420.0
EM_STEP_NM = 10.0
N_BANDS_RAW = 31
RAYLEIGH_OFFSET_NM = 20.0           # drop emission bands at lambda <= ex + offset
SAT_CEILING = 3886.0
SAT_RELATIVE = 0.999
RULER_ROW_START = 187               # rows >= this contain the ruler
DETREND_POLY_DEGREE = 2


@dataclass(frozen=True)
class VariantConfig:
    name: str
    dark_subtract: bool
    rayleigh_cutoff: bool
    p99_normalize: bool
    spatial_mask: str            # "none" | "ruler_only" | "ruler_drops"
    poly_detrend: bool
    saturation_to_mask: bool


VARIANTS: list[VariantConfig] = [
    VariantConfig("raw",              False, False, False, "none",          False, False),
    VariantConfig("dark",              True,  True,  False, "none",          False, False),
    VariantConfig("dark_norm",         True,  True,  True,  "none",          False, False),
    VariantConfig("dark_norm_mask",    True,  True,  True,  "ruler_drops",   False, True),
    VariantConfig("full",              True,  True,  True,  "ruler_drops",   True,  True),
]


def emission_wavelengths(n_bands: int = N_BANDS_RAW) -> list[float]:
    return [EM_START_NM + i * EM_STEP_NM for i in range(n_bands)]


def apply_rayleigh_cutoff(
    cube: np.ndarray, ex_nm: float, em_wls: list[float]
) -> tuple[np.ndarray, list[float]]:
    cutoff = ex_nm + RAYLEIGH_OFFSET_NM
    keep = np.array([wl > cutoff for wl in em_wls])
    return cube[:, :, keep], [wl for wl, k in zip(em_wls, keep) if k]


def saturated_mask(cube: np.ndarray) -> np.ndarray:
    return np.any(cube >= SAT_CEILING * SAT_RELATIVE, axis=2)


def build_drop_mask_and_labels(
    cube_310_dark_subtracted: np.ndarray,
    ruler_row_start: int = RULER_ROW_START,
    top_edge_exclude: int = 18,
    min_drop_area: int = 30,
    max_drop_area: int = 600,
    peak_min_distance: int = 18,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Detect drops as bright local maxima, then grow via watershed.

    Strategy:
      1. Smooth max-projection of (dark-subtracted) cube to suppress speckle.
      2. Find local maxima (one per drop) inside the analysis window.
      3. Use a moderate intensity threshold to define a generous binary
         "candidate region" for watershed to fill.
      4. Watershed from peak markers, constrained to the candidate region.
      5. Filter the resulting components by area to remove halo/noise blobs.
    """
    H = cube_310_dark_subtracted.shape[0]
    band_max = cube_310_dark_subtracted.max(axis=2)
    band_max = np.where(band_max > 0, band_max, 0.0)

    # Restrict analysis window: above ruler, below the noisy top edge
    analysis_window = np.zeros_like(band_max, dtype=bool)
    analysis_window[top_edge_exclude:ruler_row_start, :] = True
    band_max[~analysis_window] = 0

    smoothed = ndi.gaussian_filter(band_max, sigma=2.0)

    # Drop centers = bright local maxima in the analysis window
    peaks = peak_local_max(
        smoothed,
        min_distance=peak_min_distance,
        threshold_abs=float(np.percentile(smoothed[analysis_window], 60)),
        exclude_border=False,
    )
    if len(peaks) == 0:
        empty = np.zeros_like(band_max, dtype=bool)
        return empty, np.zeros_like(band_max, dtype=np.int32), 0

    valid = band_max[band_max > 0]
    candidate_threshold = float(np.percentile(valid, 65)) if valid.size else 0.0
    candidate_mask = (band_max > candidate_threshold) & analysis_window
    candidate_mask = ndi.binary_opening(candidate_mask, iterations=1)
    candidate_mask = ndi.binary_closing(candidate_mask, iterations=1)

    distance = ndi.distance_transform_edt(candidate_mask)
    markers = np.zeros_like(candidate_mask, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        if candidate_mask[r, c]:
            markers[r, c] = i

    labels = watershed(-distance, markers, mask=candidate_mask).astype(np.int32)

    # Area filter: drop a label if its component is too small or too large
    keep_label = np.zeros(labels.max() + 1, dtype=bool)
    for lid in range(1, labels.max() + 1):
        area = int((labels == lid).sum())
        if min_drop_area <= area <= max_drop_area:
            keep_label[lid] = True
    relabel = np.cumsum(keep_label) * keep_label
    labels = relabel[labels].astype(np.int32)
    drop_mask = labels > 0
    return drop_mask, labels, int(labels.max())


def fit_poly_background_per_band(
    cube: np.ndarray, fit_pixels_mask: np.ndarray, degree: int = DETREND_POLY_DEGREE
) -> np.ndarray:
    H, W, B = cube.shape
    yy, xx = np.mgrid[0:H, 0:W]
    yn = (yy / H).ravel()
    xn = (xx / W).ravel()
    terms = []
    for dy in range(degree + 1):
        for dx in range(degree + 1 - dy):
            terms.append((yn ** dy) * (xn ** dx))
    A = np.stack(terms, axis=1)
    flat_mask = fit_pixels_mask.ravel()
    A_fit = A[flat_mask]
    backgrounds = np.zeros_like(cube)
    for b in range(B):
        y = cube[:, :, b].ravel()
        y_fit = y[flat_mask]
        if y_fit.size == 0 or not np.any(y_fit > 0):
            continue
        coef, *_ = np.linalg.lstsq(A_fit, y_fit, rcond=None)
        if not np.all(np.isfinite(coef)):
            continue
        backgrounds[:, :, b] = (A @ coef).reshape(H, W)
    return backgrounds


def per_cube_p99(cube: np.ndarray, valid_mask_2d: np.ndarray | None) -> float:
    if valid_mask_2d is None:
        return float(np.percentile(cube, 99))
    pixels = cube[valid_mask_2d]
    if pixels.size == 0:
        return float(np.percentile(cube, 99))
    return float(np.percentile(pixels, 99))


def preprocess_cube(
    cube: np.ndarray,
    ex_nm: float,
    background: np.ndarray,
    drop_mask_2d: np.ndarray,
    cfg: VariantConfig,
) -> tuple[np.ndarray, list[float], dict]:
    cube = cube.astype(np.float32, copy=True)
    em_wls = emission_wavelengths()
    info: dict = {"ex_nm": ex_nm}

    if cfg.dark_subtract:
        cube = cube - background.astype(np.float32)
        np.maximum(cube, 0, out=cube)
        info["dark_subtracted"] = True

    if cfg.rayleigh_cutoff:
        cube, em_wls = apply_rayleigh_cutoff(cube, ex_nm, em_wls)
        info["rayleigh_cutoff_nm"] = ex_nm + RAYLEIGH_OFFSET_NM
        info["bands_kept"] = len(em_wls)

    if cfg.poly_detrend:
        H = cube.shape[0]
        non_drop = ~drop_mask_2d
        non_drop[RULER_ROW_START:H, :] = False
        bg = fit_poly_background_per_band(cube, non_drop)
        cube = cube - bg
        np.maximum(cube, 0, out=cube)
        info["poly_detrended"] = True

    if cfg.p99_normalize:
        if cfg.spatial_mask == "ruler_drops":
            valid_for_p99 = drop_mask_2d
        else:
            valid_for_p99 = np.ones(cube.shape[:2], dtype=bool)
            valid_for_p99[RULER_ROW_START:cube.shape[0], :] = False
        p99 = per_cube_p99(cube, valid_for_p99)
        if p99 > 0:
            cube = cube / p99
        info["p99_value"] = p99

    return cube, em_wls, info


def build_spatial_mask(
    cfg: VariantConfig,
    drop_mask_2d: np.ndarray,
    saturation_union_2d: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray | None:
    if cfg.spatial_mask == "none":
        return None
    mask = np.ones((H, W), dtype=np.float32)
    mask[RULER_ROW_START:H, :] = 0.0
    if cfg.spatial_mask == "ruler_drops":
        mask = mask * drop_mask_2d.astype(np.float32)
    if cfg.saturation_to_mask:
        mask = mask * (~saturation_union_2d).astype(np.float32)
    return mask


def save_drop_spectra_overview(
    excitation_data: dict[float, ExcitationData],
    drop_label_map: np.ndarray,
    n_drops: int,
    out_path: Path,
    title_suffix: str,
) -> None:
    excitations = sorted(excitation_data.keys())
    fig, axes = plt.subplots(
        len(excitations), 1, figsize=(11, 1.7 * len(excitations)),
        sharex=False, squeeze=False,
    )
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_drops, 1)))
    for r, ex in enumerate(excitations):
        ed = excitation_data[ex]
        ax = axes[r, 0]
        for d in range(1, n_drops + 1):
            sel = drop_label_map == d
            if not sel.any():
                continue
            spec = ed.cube[sel].mean(axis=0)
            ax.plot(ed.emission_wavelengths, spec, color=cmap[d - 1],
                    linewidth=0.8, alpha=0.75)
        ax.set_title(f"Ex={ex:.0f} nm  ·  per-drop mean spectra ({n_drops} drops)",
                     fontsize=9)
        ax.set_xlabel("Emission lambda (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Drop Data preprocessing variant: {title_suffix}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_mask_overlay(
    cube_310: np.ndarray,
    drop_mask: np.ndarray,
    drop_labels: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    proj = cube_310.max(axis=2)
    axes[0].imshow(proj, cmap="magma", vmin=0, vmax=np.percentile(proj, 99.5))
    axes[0].set_title("310 1500 SPF max projection")
    axes[0].axis("off")
    axes[1].imshow(drop_mask, cmap="gray")
    axes[1].set_title(f"Drop mask ({int(drop_mask.sum())} px valid)")
    axes[1].axis("off")
    rng = np.random.default_rng(0)
    label_colors = rng.random((drop_labels.max() + 1, 3))
    label_colors[0] = 0
    axes[2].imshow(label_colors[drop_labels])
    axes[2].set_title(f"Drop labels (n={int(drop_labels.max())})")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def load_recommendations() -> tuple[dict[float, str], dict]:
    rec_path = INSPECTION_DIR / "recommended_cubes.json"
    if not rec_path.exists():
        raise SystemExit(f"Missing {rec_path}. Run drop_data_montage.py first.")
    payload = json.loads(rec_path.read_text())
    rec = payload["recommended_per_excitation"]
    chosen: dict[float, str] = {float(k): v["stem"] for k, v in rec.items()}
    return chosen, payload


def load_cube(stem: str) -> np.ndarray:
    return np.load(CACHE_DIR / f"{stem}.npy")


def main() -> None:
    PREP_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    chosen, rec_payload = load_recommendations()
    print("[prep] using cubes:")
    for ex, stem in sorted(chosen.items()):
        print(f"   ex={ex:6.1f} -> {stem}")

    print("[prep] loading raw cubes & references...")
    background = load_cube("Background")
    raw_cubes = {ex: load_cube(stem) for ex, stem in chosen.items()}
    cube_310 = raw_cubes[310.0]
    H, W, _ = cube_310.shape

    print("[prep] building drop mask from 310 1500 SPF (dark-subtracted)...")
    drop_mask, drop_labels, n_drops = build_drop_mask_and_labels(cube_310 - background)
    print(f"[prep] detected n_drops={n_drops}, valid pixels={int(drop_mask.sum())}")
    np.save(PROC_ROOT / "drop_mask.npy", drop_mask)
    np.save(PROC_ROOT / "drop_labels.npy", drop_labels)

    sat_union = np.zeros((H, W), dtype=bool)
    for cube in raw_cubes.values():
        sat_union |= saturated_mask(cube)
    print(f"[prep] saturated pixels (union over excitations): {int(sat_union.sum())}")

    save_mask_overlay(cube_310, drop_mask, drop_labels,
                      PREP_OUT_ROOT / "_mask_overlay.png")

    overall: dict = {
        "n_drops": n_drops,
        "drop_mask_pixels": int(drop_mask.sum()),
        "saturation_pixels": int(sat_union.sum()),
        "ruler_row_start": RULER_ROW_START,
        "recommended_cubes": {str(k): v for k, v in chosen.items()},
        "variants": {},
    }

    for cfg in VARIANTS:
        out_dir = PREP_OUT_ROOT / cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        proc_dir = PROC_ROOT / cfg.name
        proc_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[prep] === variant: {cfg.name} ===")

        excitations: dict[float, ExcitationData] = {}
        per_ex_info: dict = {}
        for ex, raw in raw_cubes.items():
            cube, em_wls, info = preprocess_cube(
                raw, ex, background, drop_mask, cfg
            )
            ed = ExcitationData(
                excitation_nm=float(ex),
                cube=cube.astype(np.float32),
                emission_wavelengths=em_wls,
                exposure_time=float(rec_payload["recommended_per_excitation"][str(int(ex))]["exposure_ms"]) / 1000.0,
            )
            excitations[float(ex)] = ed
            per_ex_info[str(ex)] = info
            print(f"    ex={ex:6.1f}  shape={cube.shape}  range=[{cube.min():.4g}, {cube.max():.4g}]  bands_kept={ed.n_bands}")

        spatial_mask = build_spatial_mask(cfg, drop_mask, sat_union, H, W)

        sd = SpectraData(
            excitations=excitations,
            mask=spatial_mask,
            sample_name=f"DropData_{cfg.name}",
            loading_options=LoadingOptions(
                cutoff_offset=int(RAYLEIGH_OFFSET_NM),
                apply_rayleigh_cutoff=cfg.rayleigh_cutoff,
                apply_second_order_cutoff=False,
                normalize_exposure=False,
                normalize_laser_power=False,
            ),
            metadata={
                "variant": cfg.name,
                "variant_config": cfg.__dict__,
                "n_drops": n_drops,
                "drop_mask_pixels": int(drop_mask.sum()),
                "per_excitation": per_ex_info,
            },
        )
        pkl_path = proc_dir / "spectra_data.pkl"
        sd.to_pickle(pkl_path)
        print(f"    wrote {pkl_path}")

        save_drop_spectra_overview(
            excitations, drop_labels, n_drops,
            out_dir / "drop_spectra.png", cfg.name,
        )
        (out_dir / "summary.json").write_text(json.dumps({
            "variant": cfg.name,
            "config": cfg.__dict__,
            "spectra_data_pickle": str(pkl_path),
            "spatial_mask_valid_pixels": (
                int(spatial_mask.sum()) if spatial_mask is not None else None
            ),
            "per_excitation": per_ex_info,
        }, indent=2, default=float))

        overall["variants"][cfg.name] = {
            "config": cfg.__dict__,
            "spectra_pickle": str(pkl_path),
            "spatial_mask_valid_pixels": (
                int(spatial_mask.sum()) if spatial_mask is not None else None
            ),
        }

    (PREP_OUT_ROOT / "overall_summary.json").write_text(
        json.dumps(overall, indent=2, default=float)
    )
    print(f"\n[prep] wrote {PREP_OUT_ROOT / 'overall_summary.json'}")


if __name__ == "__main__":
    main()
