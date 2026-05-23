#!/usr/bin/env python3
"""
Drop Data — Excitation Montage & Saturation/Mask Diagnostics
==============================================================
Builds two diagnostic figures from the cached .npy cubes (no PyImageJ needed):

  1. excitation_montage.png — for each of the 7 excitations, show the
     max-projection of every available integration time side-by-side, with
     the saturation fraction in the title. Lets us pick the best integration
     per excitation by eye and by number.

  2. ruler_mask_check.png — overlay a candidate ruler mask on the 310 nm
     longest-integration cube so we can confirm where the ruler lives and
     decide on a spatial crop before any clustering.

Inputs : Data/processed/Drop Data/raw/*.npy   (produced by drop_data_inspect.py)
Outputs: results/Drop_Data_Inspection/
            excitation_montage.png
            ruler_mask_check.png
            recommended_cubes.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "Data" / "processed" / "Drop Data" / "raw"
OUT_DIR = PROJECT_ROOT / "results" / "Drop_Data_Inspection"

# 12-bit camera with ~52 ADU offset → effective ceiling ~3886
SAT_CEILING = 3886.0
SAT_TAIL_FRAC_CUTOFF = 0.001  # >0.1% pixels at ceiling = "saturating"

SAMPLE_RE = re.compile(r"^(?P<ex>\d+)\s+(?P<expo>\d+)\s+SPF$")


def load_cache() -> dict[str, np.ndarray]:
    return {p.stem: np.load(p) for p in sorted(CACHE_DIR.glob("*.npy"))}


def saturation_fraction(cube: np.ndarray) -> float:
    return float((cube >= SAT_CEILING * 0.999).mean())


# ---------------------------------------------------------------------------
# Excitation montage
# ---------------------------------------------------------------------------
def build_excitation_montage(cubes: dict[str, np.ndarray], out_path: Path) -> dict:
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for stem in cubes:
        m = SAMPLE_RE.match(stem)
        if not m:
            continue
        grouped[int(m["ex"])].append((int(m["expo"]), stem))
    excitations = sorted(grouped)
    max_cols = max(len(grouped[e]) for e in excitations)

    fig, axes = plt.subplots(
        len(excitations), max_cols,
        figsize=(3.4 * max_cols, 3.0 * len(excitations)),
        squeeze=False,
    )

    recommendations: dict[str, dict] = {}

    for r, ex in enumerate(excitations):
        rows = sorted(grouped[ex])
        # Pick recommended cube: longest integration that is NOT saturating;
        # if all saturate, pick the shortest (least clipping).
        sat = [(expo, stem, saturation_fraction(cubes[stem])) for expo, stem in rows]
        clean = [(expo, stem, s) for expo, stem, s in sat if s < SAT_TAIL_FRAC_CUTOFF]
        if clean:
            chosen_expo, chosen_stem, chosen_sat = max(clean, key=lambda t: t[0])
            chosen_reason = "longest non-saturating"
        else:
            chosen_expo, chosen_stem, chosen_sat = min(sat, key=lambda t: t[0])
            chosen_reason = "shortest (all saturating)"
        recommendations[str(ex)] = {
            "stem": chosen_stem,
            "exposure_ms": chosen_expo,
            "saturated_frac": chosen_sat,
            "reason": chosen_reason,
        }

        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            expo, stem = rows[c]
            cube = cubes[stem]
            proj = cube.max(axis=2)
            vmax = np.percentile(proj, 99.5)
            ax.imshow(proj, cmap="magma", vmin=0, vmax=vmax)
            sat_frac = saturation_fraction(cube)
            star = " ★" if stem == chosen_stem else ""
            ax.set_title(
                f"Ex={ex} nm, {expo} ms{star}\n"
                f"max={cube.max():.0f}  sat={sat_frac*100:.2f}%",
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle(
        "Drop Data: max-projection per (excitation × integration). "
        "★ = recommended cube per excitation",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return recommendations


# ---------------------------------------------------------------------------
# Ruler mask check
# ---------------------------------------------------------------------------
def detect_ruler_band(cube: np.ndarray) -> tuple[int, int]:
    """Estimate ruler band by scanning rows for high mean intensity.

    The ruler is consistently bright across all bands while the drop area is
    locally bright only where drops are, so a row-wise mean over the *spatial
    mean across bands* (which suppresses drop-specific peaks) tends to spike
    where the ruler lives.
    """
    spatial_mean = cube.mean(axis=2)            # (H, W)
    row_mean = spatial_mean.mean(axis=1)        # (H,)
    threshold = row_mean.mean() + 0.5 * row_mean.std()
    bright_rows = np.where(row_mean > threshold)[0]
    if len(bright_rows) == 0:
        return 0, 0
    # Find longest contiguous bright stretch at the bottom of the image
    h = cube.shape[0]
    candidate = bright_rows[bright_rows > 0.5 * h]
    if len(candidate) == 0:
        return 0, 0
    return int(candidate.min()), int(candidate.max())


def build_ruler_mask_check(cubes: dict[str, np.ndarray], out_path: Path) -> dict:
    # Use the 310 nm 1500 ms cube — most drops visible, cleanest.
    key = "310 1500 SPF"
    cube = cubes[key]
    proj = cube.max(axis=2)
    spatial_mean = cube.mean(axis=2)
    row_mean = spatial_mean.mean(axis=1)

    r0, r1 = detect_ruler_band(cube)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(proj, cmap="magma", vmin=0, vmax=np.percentile(proj, 99.5))
    axes[0].set_title(f"{key}: max projection")
    axes[0].axis("off")

    axes[1].imshow(spatial_mean, cmap="viridis")
    if r1 > 0:
        rect = patches.Rectangle(
            (0, r0), cube.shape[1], r1 - r0,
            linewidth=2, edgecolor="red", facecolor="none",
            label=f"detected ruler band rows {r0}-{r1}",
        )
        axes[1].add_patch(rect)
        axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_title("Spatial mean (band-averaged) + auto-detected ruler band")
    axes[1].axis("off")

    axes[2].plot(row_mean, np.arange(len(row_mean)), color="C0")
    axes[2].set_xlabel("Row-mean intensity")
    axes[2].set_ylabel("Row")
    axes[2].invert_yaxis()
    if r1 > 0:
        axes[2].axhspan(r0, r1, color="red", alpha=0.2, label="ruler band")
        axes[2].legend()
    axes[2].set_title("Row-wise mean intensity")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"ruler_rows": [r0, r1], "image_height": cube.shape[0]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    cubes = load_cache()
    if not cubes:
        raise SystemExit(f"No cached cubes in {CACHE_DIR}. Run drop_data_inspect.py first.")
    print(f"[montage] loaded {len(cubes)} cached cubes")

    recs = build_excitation_montage(cubes, OUT_DIR / "excitation_montage.png")
    print(f"[montage] wrote {OUT_DIR / 'excitation_montage.png'}")
    print("[montage] recommended cubes per excitation:")
    for ex, info in recs.items():
        print(f"   ex={ex} nm -> {info['stem']:20s} ({info['reason']}, sat={info['saturated_frac']*100:.2f}%)")

    ruler_info = build_ruler_mask_check(cubes, OUT_DIR / "ruler_mask_check.png")
    print(f"[montage] wrote {OUT_DIR / 'ruler_mask_check.png'} ruler={ruler_info}")

    out = {
        "saturation_ceiling": SAT_CEILING,
        "saturation_tail_cutoff": SAT_TAIL_FRAC_CUTOFF,
        "recommended_per_excitation": recs,
        "ruler_band": ruler_info,
    }
    (OUT_DIR / "recommended_cubes.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[montage] wrote {OUT_DIR / 'recommended_cubes.json'}")


if __name__ == "__main__":
    main()
