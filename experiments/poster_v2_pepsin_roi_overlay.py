#!/usr/bin/env python3
"""Render the Pepsin ROI overlay in the same style as the Lichens overlay
(white square ROI markers on the first instance of each class, colour-
coded class regions, class legend below).

Saves: Showcase_Poster/03_collagen_pepsin_IASIM/11_roi_overlay_pepsin.png
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

ROOT = Path(__file__).resolve().parent.parent
MASK_PATH = ROOT / "Data" / "processed" / "Collagen Pepsin" / "class_mask.png"
ROI_PATH = ROOT / "Data" / "processed" / "Collagen Pepsin" / "roi_regions_expanded.json"
OUT_PATH = (ROOT / "Showcase_Poster" / "03_collagen_pepsin_IASIM"
            / "11_roi_overlay_pepsin.png")
OUT_PDF = OUT_PATH.with_suffix(".pdf")


def main():
    mask = np.array(Image.open(MASK_PATH))            # (H, W, 3)
    rois = json.loads(ROI_PATH.read_text())

    # Convert black background to white so the figure reads on a poster
    # against a white background (matches the Lichens overlay style).
    bg = (mask == 0).all(axis=2)
    display = mask.copy()
    display[bg] = 255

    fig, ax = plt.subplots(figsize=(9, 7.0), dpi=300)
    ax.imshow(display, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw a white square ROI marker on the first instance of each class.
    # Marker is a centred rectangle ~30 % of the sample box.
    seen_classes = set()
    classes_meta = {c["id"]: c for c in rois["classes"]}
    for region in rois["regions"]:
        cid = region["class_id"]
        if cid in seen_classes:
            continue
        seen_classes.add(cid)
        rect = region["rect"]
        h = rect["row_max"] - rect["row_min"]
        w = rect["col_max"] - rect["col_min"]
        cy = (rect["row_min"] + rect["row_max"]) / 2
        cx = (rect["col_min"] + rect["col_max"]) / 2
        mw = 0.45 * w
        mh = 0.32 * h
        ax.add_patch(mpatches.Rectangle(
            (cx - mw / 2, cy - mh / 2), mw, mh,
            linewidth=2.4, edgecolor="white", facecolor="none", zorder=10))

    # Legend along the bottom.
    legend_handles = []
    for cid in sorted(classes_meta):
        meta = classes_meta[cid]
        rgb = tuple(c / 255.0 for c in meta["color"])
        legend_handles.append(mpatches.Patch(
            facecolor=rgb, edgecolor="black", linewidth=0.6,
            label=meta["name"]))
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
    fig.savefig(OUT_PDF, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
