#!/usr/bin/env python3
"""
Drop Data — Cropped pipeline (ruler physically removed)
=========================================================
End-to-end re-run of preprocessing + AE training + selection sweep with
the ruler band physically removed (rows >= 187 dropped from every cube).
This eliminates any chance the ruler influences either AE training or the
perturbation analysis.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_select import Analyzer, Config, SpectraData  # noqa: E402
from spectral_select.types import ExcitationData, LoadingOptions  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "Data" / "processed" / "Drop Data" / "raw"
PROC_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data"
PROC_CROP_ROOT = PROJECT_ROOT / "Data" / "processed" / "Drop Data Cropped"
INSPECT_DIR = PROJECT_ROOT / "results" / "Drop_Data_Inspection"
OUT_ROOT = PROJECT_ROOT / "results" / "Drop_Data_Cropped_Sweep"

RULER_ROW_START = 175             # tightened from 187: drops 16/18 were ruler artifacts
EM_START_NM = 420.0
EM_STEP_NM = 10.0
N_BANDS_RAW = 31
RAYLEIGH_OFFSET_NM = 20.0
SAT_CEILING = 3886.0
SAT_RELATIVE = 0.999
DETREND_POLY_DEGREE = 2
N_GRID = list(range(3, 11))
AE_EPOCHS = 25


@dataclass(frozen=True)
class VariantConfig:
    name: str
    dark_subtract: bool
    rayleigh_cutoff: bool
    p99_normalize: bool
    drop_focus_mask: bool
    poly_detrend: bool
    saturation_to_mask: bool


VARIANTS = [
    VariantConfig("raw_cr",            False, False, False, False, False, False),
    VariantConfig("dark_cr",           True,  True,  False, False, False, False),
    VariantConfig("dark_norm_cr",      True,  True,  True,  False, False, False),
    VariantConfig("dark_norm_mask_cr", True,  True,  True,  True,  False, True),
    VariantConfig("full_cr",           True,  True,  True,  True,  True,  True),
]


def emission_wavelengths():
    return [EM_START_NM + i * EM_STEP_NM for i in range(N_BANDS_RAW)]


def apply_rayleigh_cutoff(cube, ex_nm, em_wls):
    cutoff = ex_nm + RAYLEIGH_OFFSET_NM
    keep = np.array([wl > cutoff for wl in em_wls])
    return cube[:, :, keep], [wl for wl, k in zip(em_wls, keep) if k]


def fit_poly_per_band(cube, fit_mask, degree=DETREND_POLY_DEGREE):
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


def saturated_mask(cube):
    return np.any(cube >= SAT_CEILING * SAT_RELATIVE, axis=2)


def per_cube_p99(cube, valid_2d):
    pix = cube[valid_2d]
    if pix.size == 0:
        return float(np.percentile(cube, 99))
    return float(np.percentile(pix, 99))


def crop_cube(cube):
    return cube[:RULER_ROW_START, :, :]


def build_cropped_drop_mask(orig_drop_labels):
    """Crop labels to rows < RULER_ROW_START and renumber 1..n_drops contiguously."""
    cropped = orig_drop_labels[:RULER_ROW_START, :].astype(np.int32)
    surviving_ids = sorted(int(v) for v in np.unique(cropped) if v > 0)
    remap = {old: new for new, old in enumerate(surviving_ids, start=1)}
    remap[0] = 0
    new_labels = np.zeros_like(cropped)
    for old, new in remap.items():
        if old == 0:
            continue
        new_labels[cropped == old] = new
    return new_labels > 0, new_labels


def preprocess_cube_cropped(cube_full, ex_nm, background_full, drop_mask_2d, cfg):
    cube = crop_cube(cube_full).astype(np.float32, copy=True)
    background = crop_cube(background_full).astype(np.float32)
    em_wls = emission_wavelengths()
    if cfg.dark_subtract:
        cube = cube - background
        np.maximum(cube, 0, out=cube)
    if cfg.rayleigh_cutoff:
        cube, em_wls = apply_rayleigh_cutoff(cube, ex_nm, em_wls)
    if cfg.poly_detrend:
        non_drop = ~drop_mask_2d
        bg_fit = fit_poly_per_band(cube, non_drop)
        cube = cube - bg_fit
        np.maximum(cube, 0, out=cube)
    if cfg.p99_normalize:
        if cfg.drop_focus_mask:
            valid_2d = drop_mask_2d
        else:
            valid_2d = np.ones(cube.shape[:2], dtype=bool)
        p99 = per_cube_p99(cube, valid_2d)
        if p99 > 0:
            cube = cube / p99
    return cube, em_wls


def build_spatial_mask_cropped(cfg, drop_mask_2d, sat_union_2d):
    if not cfg.drop_focus_mask and not cfg.saturation_to_mask:
        return None
    H, W = drop_mask_2d.shape
    mask = np.ones((H, W), dtype=np.float32)
    if cfg.drop_focus_mask:
        mask = mask * drop_mask_2d.astype(np.float32)
    if cfg.saturation_to_mask:
        mask = mask * (~sat_union_2d).astype(np.float32)
    return mask


def stage1_preprocess(rec_payload, recommended_cubes, cropped_drop_mask,
                      cropped_drop_labels, full_background, sat_union_2d):
    print(f"\n[crop] Stage 1: cropped variants  shape="
          f"{cropped_drop_mask.shape}")
    PROC_CROP_ROOT.mkdir(parents=True, exist_ok=True)
    for cfg in VARIANTS:
        proc_dir = PROC_CROP_ROOT / cfg.name
        proc_dir.mkdir(parents=True, exist_ok=True)
        excitations = {}
        for ex, raw in recommended_cubes.items():
            cube, em_wls = preprocess_cube_cropped(
                raw, ex, full_background, cropped_drop_mask, cfg
            )
            ed = ExcitationData(
                excitation_nm=float(ex),
                cube=cube.astype(np.float32),
                emission_wavelengths=em_wls,
                exposure_time=float(rec_payload["recommended_per_excitation"][str(int(ex))]["exposure_ms"]) / 1000.0,
            )
            excitations[float(ex)] = ed
        spatial_mask = build_spatial_mask_cropped(
            cfg, cropped_drop_mask, sat_union_2d
        )
        sd = SpectraData(
            excitations=excitations,
            mask=spatial_mask,
            sample_name=f"DropDataCropped_{cfg.name}",
            loading_options=LoadingOptions(
                cutoff_offset=int(RAYLEIGH_OFFSET_NM),
                apply_rayleigh_cutoff=cfg.rayleigh_cutoff,
                apply_second_order_cutoff=False,
                normalize_exposure=False,
                normalize_laser_power=False,
            ),
            metadata={
                "variant": cfg.name,
                "cropped_to_rows": RULER_ROW_START,
                "n_drops": int(cropped_drop_labels.max()),
            },
        )
        sd.to_pickle(proc_dir / "spectra_data.pkl")
        first_cube = next(iter(excitations.values())).cube
        print(f"[crop]   wrote {cfg.name}  shape={first_cube.shape}")
    np.save(PROC_CROP_ROOT / "drop_mask.npy", cropped_drop_mask)
    np.save(PROC_CROP_ROOT / "drop_labels.npy", cropped_drop_labels)


def stage2_train(variant_name):
    pkl_path = PROC_CROP_ROOT / variant_name / "spectra_data.pkl"
    data = SpectraData.from_pickle(pkl_path)
    cfg = Config(
        sample_name=f"DropDataCropped_{variant_name}",
        n_bands_to_select=10,
        device="mps",
        training_epochs=AE_EPOCHS,
        patch_size=16,
        patch_stride=8,
        n_baseline_patches=30,
        n_important_dimensions=12,
        normalization_method="max_per_excitation",
        save_visualizations=False,
        save_tiff_layers=False,
        save_detailed_results=False,
        output_dir=OUT_ROOT / variant_name / "ae_perturb",
    )
    print(f"[crop] Stage 2: training AE for {variant_name} "
          f"({AE_EPOCHS} epochs, max_per_excitation)")
    t0 = time.time()
    analyzer = Analyzer(cfg)
    analyzer.fit(data)
    inf = analyzer.influence_matrix or {}
    out = {}
    for ex in data.excitation_wavelengths:
        if ex in inf:
            out[float(ex)] = np.asarray(inf[ex], dtype=float)
        elif str(ex) in inf:
            out[float(ex)] = np.asarray(inf[str(ex)], dtype=float)
    print(f"[crop]   {variant_name}: trained in {time.time() - t0:.1f}s")
    return data, out


def per_drop_mean_spectra(data, drop_labels):
    bands_meta = []
    by_ex_cube = {}
    for ex in data.excitation_wavelengths:
        ed = data.get_excitation(ex)
        by_ex_cube[float(ex)] = ed.cube
        for i, em in enumerate(ed.emission_wavelengths):
            bands_meta.append((float(ex), int(i), float(em)))
    n_drops = int(drop_labels.max())
    spec = np.zeros((n_drops, len(bands_meta)), dtype=np.float32)
    for d in range(1, n_drops + 1):
        sel = drop_labels == d
        if not sel.any():
            continue
        for c, (ex, em_idx, _) in enumerate(bands_meta):
            spec[d - 1, c] = float(by_ex_cube[ex][sel, em_idx].mean())
    return spec, bands_meta


def f_ratio_per_band(spec, types, k=3):
    n_total, B = spec.shape
    overall = spec.mean(axis=0)
    between = np.zeros(B); within = np.zeros(B)
    for t in range(k):
        members = spec[types == t]
        if len(members) == 0:
            continue
        type_mean = members.mean(axis=0)
        between += len(members) * (type_mean - overall) ** 2
        within += ((members - type_mean) ** 2).sum(axis=0)
    df_between = max(k - 1, 1)
    df_within = max(n_total - k, 1)
    return (between / df_between) / (within / df_within + 1e-12)


def cluster_to_3types(spec):
    norm = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-12)
    Z = linkage(norm, method="ward")
    return fcluster(Z, t=3, criterion="maxclust") - 1


def flatten_influence(infl, bands_meta):
    out = np.zeros(len(bands_meta), dtype=float)
    for i, (ex, em_idx, _) in enumerate(bands_meta):
        if ex in infl and em_idx < len(infl[ex]):
            out[i] = float(infl[ex][em_idx])
    return out


def evaluate(scores, f_ratio, n):
    sel = np.argsort(-scores)[:n]
    sub = f_ratio[sel]
    n_total = len(f_ratio)
    rank_pct = 1.0 - (np.argsort(np.argsort(-f_ratio))[sel].mean()
                      / max(n_total - 1, 1))
    return dict(
        mean_f=float(sub.mean()),
        max_f=float(sub.max()),
        rank_percentile=float(rank_pct),
        sel_idx=sel.tolist(),
    )


def stage3_score(variant_name, data, influence, drop_labels):
    spec, bands_meta = per_drop_mean_spectra(data, drop_labels)
    types = cluster_to_3types(spec)
    f_ratio = f_ratio_per_band(spec, types)

    out_dir = OUT_ROOT / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "drop_mean_spectra.npy", spec)
    np.save(out_dir / "drop_types.npy", types)

    infl = flatten_influence(influence, bands_meta)
    rows = []
    for n in N_GRID:
        m = evaluate(infl, f_ratio, n)
        rows.append({
            "variant": variant_name,
            "method": "ae_perturb_max_per_ex",
            "n_bands": n,
            **{k: v for k, v in m.items() if k != "sel_idx"},
            "selected_bands": [
                f"{int(bands_meta[i][0])}/{int(bands_meta[i][2])}"
                for i in m["sel_idx"]
            ],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ae_perturb_results.csv", index=False)

    f_table = pd.DataFrame({
        "excitation_nm": [b[0] for b in bands_meta],
        "emission_nm": [b[2] for b in bands_meta],
        "f_ratio": f_ratio,
    }).sort_values("f_ratio", ascending=False)
    f_table.to_csv(out_dir / "f_ratio_table.csv", index=False)
    return df


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[crop] === START ===")

    rec = json.loads((INSPECT_DIR / "recommended_cubes.json").read_text())
    recommended_cubes = {
        float(ex): np.load(CACHE_DIR / f"{info['stem']}.npy")
        for ex, info in rec["recommended_per_excitation"].items()
    }
    full_background = np.load(CACHE_DIR / "Background.npy")
    print(f"[crop] loaded {len(recommended_cubes)} cubes  full shape="
          f"{next(iter(recommended_cubes.values())).shape}")

    orig_drop_labels = np.load(PROC_ROOT / "drop_labels.npy")
    cropped_drop_mask, cropped_drop_labels = build_cropped_drop_mask(orig_drop_labels)
    print(f"[crop] cropped to rows 0:{RULER_ROW_START}  "
          f"n_drops_kept={int(cropped_drop_labels.max())}")

    sat_union = np.zeros(cropped_drop_mask.shape, dtype=bool)
    for cube in recommended_cubes.values():
        sat_union |= saturated_mask(crop_cube(cube))

    stage1_preprocess(rec, recommended_cubes, cropped_drop_mask,
                      cropped_drop_labels, full_background, sat_union)

    all_results = []
    for cfg in VARIANTS:
        try:
            data, influence = stage2_train(cfg.name)
            df = stage3_score(cfg.name, data, influence, cropped_drop_labels)
            all_results.append(df)
            mean_at_5 = df[df["n_bands"] == 5]["mean_f"].iloc[0]
            print(f"[crop]   {cfg.name}: ae_perturb mean_F(n=5)={mean_at_5:.2f}")
        except Exception as e:
            print(f"[crop]   {cfg.name} failed: {e}")

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(OUT_ROOT / "all_results.csv", index=False)

    summary = df_all.groupby("variant")["mean_f"].mean().round(2).to_frame("cropped_mean_F")
    try:
        prev_csv = (PROJECT_ROOT / "results" / "Drop_Data_Norm_Fix" /
                    "all_variants.csv")
        prev = pd.read_csv(prev_csv)
        prev_mpx = (prev[prev["normalization"] == "max_per_excitation"]
                    .groupby("variant")["mean_f"].mean().round(2))
        prev_mpx.index = prev_mpx.index.map(lambda v: f"{v}_cr")
        summary["non_cropped_mean_F"] = prev_mpx
        summary["delta"] = (summary["cropped_mean_F"] - summary["non_cropped_mean_F"]).round(2)
    except Exception as e:
        print(f"[crop] could not load prior results: {e}")
    summary.to_csv(OUT_ROOT / "cropped_vs_noncropped.csv")
    print(f"\n[crop] === FINAL: cropped vs non-cropped mean F ===")
    print(summary)


if __name__ == "__main__":
    main()
