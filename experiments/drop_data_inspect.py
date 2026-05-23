#!/usr/bin/env python3
"""
Drop Data — Initial Inspection
================================
Loads every .im3 file in `Data/Raw/Drop Data/` (sample cubes + Background +
Whitelight) via PyImageJ, caches each cube as a `.npy` for fast reload, and
generates a summary report + quick-look visualizations.

Inputs : Data/Raw/Drop Data/*.im3
Outputs: Data/processed/Drop Data/raw/*.npy
         results/Drop_Data_Inspection/
             summary.csv
             summary.json
             max_proj/<stem>.png       (per-file max projection)
             mean_spectra.png          (overlay of all mean spectra)
             reference_check.png       (Background vs Whitelight stats)

Run order: this script first, then `drop_data_normalize.py` (TBD), then
selection sweep.

Conventions assumed (to be verified by what we see):
  - Sample filename: "<excitation> <integration> SPF.im3"
  - Reference filenames: "Background.im3", "Whitelight.im3"
  - Cube layout returned by ImageJ is (bands, H, W) for Maestro/Nuance .im3,
    but we will detect and store (H, W, bands) consistently.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "Data" / "Raw" / "Drop Data"
CACHE_DIR = PROJECT_ROOT / "Data" / "processed" / "Drop Data" / "raw"
OUT_DIR = PROJECT_ROOT / "results" / "Drop_Data_Inspection"
MAX_PROJ_DIR = OUT_DIR / "max_proj"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_PROJ_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
SAMPLE_RE = re.compile(r"^\s*(?P<ex>\d+)\s+(?P<expo>\d+)\s+SPF\s*$", re.IGNORECASE)


@dataclass
class Im3File:
    path: Path
    stem: str
    role: str                  # "sample" | "background" | "whitelight"
    excitation: float | None
    exposure_ms: float | None

    @classmethod
    def parse(cls, path: Path) -> "Im3File":
        stem = path.stem
        low = stem.lower()
        if low == "background":
            return cls(path, stem, "background", None, None)
        if low == "whitelight":
            return cls(path, stem, "whitelight", None, None)
        m = SAMPLE_RE.match(stem)
        if not m:
            raise ValueError(f"Cannot parse filename: {path.name!r}")
        return cls(
            path=path,
            stem=stem,
            role="sample",
            excitation=float(m["ex"]),
            exposure_ms=float(m["expo"]),
        )


# ---------------------------------------------------------------------------
# Cube loading (PyImageJ + cache)
# ---------------------------------------------------------------------------
def normalize_cube_layout(arr: np.ndarray) -> np.ndarray:
    """Return cube as (H, W, bands), float32. Assumes the smallest axis is bands."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {arr.shape}")
    smallest = int(np.argmin(arr.shape))
    if smallest == 0:
        arr = np.transpose(arr, (1, 2, 0))
    elif smallest == 1:
        arr = np.transpose(arr, (0, 2, 1))
    return arr.astype(np.float32, copy=False)


def load_cube_via_ij(ij, path: Path) -> np.ndarray:
    img = ij.io().open(str(path))
    return np.asarray(ij.py.from_java(img))


def get_or_load_cube(ij, im3: Im3File) -> np.ndarray:
    cache_path = CACHE_DIR / f"{im3.stem}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    raw = load_cube_via_ij(ij, im3.path)
    cube = normalize_cube_layout(raw)
    np.save(cache_path, cube)
    return cube


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@dataclass
class CubeStats:
    stem: str
    role: str
    excitation: float | None
    exposure_ms: float | None
    height: int
    width: int
    bands: int
    dtype: str
    min: float
    max: float
    mean: float
    median: float
    std: float
    p99: float
    saturated_frac: float
    file_size_mb: float


SATURATION_THRESHOLDS = (16383.0, 65535.0)  # 14-bit / 16-bit Maestro outputs


def compute_stats(im3: Im3File, cube: np.ndarray) -> CubeStats:
    cmax = float(cube.max())
    sat_threshold = next((t for t in SATURATION_THRESHOLDS if cmax <= t * 1.001), None)
    if sat_threshold is None:
        # Fall back to the cube's own max as the saturation reference
        sat_threshold = cmax
    sat_frac = float((cube >= sat_threshold * 0.999).mean())
    return CubeStats(
        stem=im3.stem,
        role=im3.role,
        excitation=im3.excitation,
        exposure_ms=im3.exposure_ms,
        height=cube.shape[0],
        width=cube.shape[1],
        bands=cube.shape[2],
        dtype=str(cube.dtype),
        min=float(cube.min()),
        max=cmax,
        mean=float(cube.mean()),
        median=float(np.median(cube)),
        std=float(cube.std()),
        p99=float(np.percentile(cube, 99)),
        saturated_frac=sat_frac,
        file_size_mb=im3.path.stat().st_size / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_max_projection(cube: np.ndarray, title: str, out_path: Path) -> None:
    proj = cube.max(axis=2)
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.percentile(proj, 99.5)
    im = ax.imshow(proj, cmap="magma", vmin=proj.min(), vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_mean_spectra_overlay(
    spectra: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(spectra)))
    for (label, spec), c in zip(spectra.items(), colors):
        ax.plot(spec, label=label, color=c, linewidth=1.4, alpha=0.85)
    ax.set_xlabel("Band index")
    ax.set_ylabel("Mean intensity")
    ax.set_title("Mean spectrum per cube (all 22 files)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_reference_check(
    bg_cube: np.ndarray, wl_cube: np.ndarray, out_path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for row, (cube, name) in enumerate([(bg_cube, "Background"), (wl_cube, "Whitelight")]):
        proj = cube.max(axis=2)
        mean_spec = cube.mean(axis=(0, 1))
        std_spec = cube.std(axis=(0, 1))
        spatial_mean = cube.mean(axis=2)

        axes[row, 0].imshow(proj, cmap="magma")
        axes[row, 0].set_title(f"{name}: max projection")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(spatial_mean, cmap="viridis")
        axes[row, 1].set_title(f"{name}: spatial mean (all bands)")
        axes[row, 1].axis("off")

        axes[row, 2].plot(mean_spec, label="mean", color="C0")
        axes[row, 2].fill_between(
            np.arange(len(mean_spec)),
            mean_spec - std_spec,
            mean_spec + std_spec,
            alpha=0.25,
            color="C0",
            label="±1 std",
        )
        axes[row, 2].set_title(f"{name}: spectrum across image")
        axes[row, 2].set_xlabel("Band index")
        axes[row, 2].set_ylabel("Intensity")
        axes[row, 2].legend()
        axes[row, 2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    files = sorted(RAW_DIR.glob("*.im3"))
    if not files:
        raise SystemExit(f"No .im3 files in {RAW_DIR}")
    parsed = [Im3File.parse(p) for p in files]
    print(f"[inspect] {len(parsed)} files: "
          f"{sum(1 for f in parsed if f.role=='sample')} samples, "
          f"{sum(1 for f in parsed if f.role=='background')} background, "
          f"{sum(1 for f in parsed if f.role=='whitelight')} whitelight")

    # Lazy ImageJ init only if any cache miss
    need_ij = any(not (CACHE_DIR / f"{f.stem}.npy").exists() for f in parsed)
    ij = None
    if need_ij:
        import imagej
        print("[inspect] initializing ImageJ (Fiji)... (slow on first run)")
        ij = imagej.init("sc.fiji:fiji", mode="headless")
        print("[inspect] ImageJ ready")

    stats: list[CubeStats] = []
    spectra: dict[str, np.ndarray] = {}
    bg_cube = wl_cube = None

    for im3 in parsed:
        print(f"[inspect] {im3.stem:30s} role={im3.role}")
        cube = get_or_load_cube(ij, im3)
        s = compute_stats(im3, cube)
        stats.append(s)
        print(f"    shape={cube.shape} dtype={cube.dtype} "
              f"min={s.min:.1f} max={s.max:.1f} mean={s.mean:.1f} "
              f"sat_frac={s.saturated_frac:.4f}")
        save_max_projection(cube, im3.stem, MAX_PROJ_DIR / f"{im3.stem}.png")
        spectra[im3.stem] = cube.mean(axis=(0, 1))
        if im3.role == "background":
            bg_cube = cube
        elif im3.role == "whitelight":
            wl_cube = cube

    df = pd.DataFrame([asdict(s) for s in stats])
    df = df.sort_values(["role", "excitation", "exposure_ms"], na_position="first")
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"[inspect] wrote {OUT_DIR / 'summary.csv'}")

    summary_json = {
        "n_files": len(stats),
        "samples_by_excitation": (
            df[df.role == "sample"]
            .groupby("excitation")["exposure_ms"]
            .apply(list)
            .to_dict()
        ),
        "common_shape": (
            list(df[["height", "width", "bands"]].mode().iloc[0])
            if len(df) else None
        ),
        "shape_consistent": bool(df[["height", "width", "bands"]].nunique().eq(1).all()),
        "saturation_warning": df[df.saturated_frac > 0.001][["stem", "saturated_frac"]].to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2, default=float))
    print(f"[inspect] wrote {OUT_DIR / 'summary.json'}")

    save_mean_spectra_overlay(spectra, OUT_DIR / "mean_spectra.png")
    if bg_cube is not None and wl_cube is not None:
        save_reference_check(bg_cube, wl_cube, OUT_DIR / "reference_check.png")
        print(f"[inspect] wrote {OUT_DIR / 'reference_check.png'}")
    else:
        print("[inspect] WARNING: missing Background or Whitelight cube")


if __name__ == "__main__":
    main()
