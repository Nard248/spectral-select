"""Re-render the Collagen Sponges ROI overlay image with display labels
remapped to Class 1 / Class 2 / Class 3 (instead of the underlying data IDs
Class 2 / Class 3 / Class 4).

Only the *legend text* changes — the underlying data class IDs are untouched.
This is a thesis-only override; the poster source image and the original
roi_regions JSON remain unmodified.

Output:
    MasterThesis_Narek_Meloyan/figures/collagen_sponges/CollagenLabels_and_ROI.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MASK_PATH = ROOT / "Data" / "processed" / "Collagen Pepsin" / "class_mask.png"
ROI_PATH = ROOT / "Data" / "processed" / "Collagen Pepsin" / "roi_regions_expanded.json"
OUT_PATH = (ROOT / "MasterThesis_Narek_Meloyan" / "figures"
            / "collagen_sponges" / "CollagenLabels_and_ROI.png")

# Display rename: data class ID -> the label we want shown in the figure.
# Keeps the underlying data alone; only re-orders the visible class names
# so the thesis legend reads "Class 1, Class 2, Class 3" instead of
# "Class 2, Class 3, Class 4".
DISPLAY_LABEL = {2: "Class 1", 3: "Class 2", 4: "Class 3"}


def main() -> None:
    mask = np.array(Image.open(MASK_PATH))
    rois = json.loads(ROI_PATH.read_text())

    # Black background -> white for poster-friendly rendering
    bg = (mask == 0).all(axis=2)
    display = mask.copy()
    display[bg] = 255

    fig, ax = plt.subplots(figsize=(9, 7.0), dpi=300)
    ax.imshow(display, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White-square ROI markers on the first instance of each class.
    seen: set[int] = set()
    classes_meta = {c["id"]: c for c in rois["classes"]}
    for region in rois["regions"]:
        cid = region["class_id"]
        if cid in seen:
            continue
        seen.add(cid)
        rect = region["rect"]
        h = rect["row_max"] - rect["row_min"]
        w = rect["col_max"] - rect["col_min"]
        cy = (rect["row_min"] + rect["row_max"]) / 2
        cx = (rect["col_min"] + rect["col_max"]) / 2
        mw, mh = 0.45 * w, 0.32 * h
        ax.add_patch(mpatches.Rectangle(
            (cx - mw / 2, cy - mh / 2), mw, mh,
            linewidth=2.4, edgecolor="white", facecolor="none", zorder=10))

    # Legend with the renamed labels.
    legend_handles = []
    for cid in sorted(classes_meta):
        meta = classes_meta[cid]
        rgb = tuple(c / 255.0 for c in meta["color"])
        legend_handles.append(mpatches.Patch(
            facecolor=rgb, edgecolor="black", linewidth=0.6,
            label=DISPLAY_LABEL.get(cid, meta["name"])))
    ax.legend(handles=legend_handles,
              loc="lower center",
              bbox_to_anchor=(0.5, -0.10),
              ncol=len(legend_handles),
              frameon=True, fontsize=12,
              handlelength=1.5, handleheight=1.2,
              borderpad=0.7, columnspacing=2.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
