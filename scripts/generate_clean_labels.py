#!/usr/bin/env python3
"""
Generate clean classification label maps (no titles, no percentage tags).

Crops the title from existing classification map PNGs, producing images
with only the colored class regions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from pathlib import Path

PAPER_FIGURES = (
    Path(__file__).parent.parent
    / "results" / "Lichens_Dataset_1_MasterRun" / "paper_figures"
)
OUTPUT_DIR = Path(__file__).parent.parent / "Paper Source" / "paper" / "figures-updated"

CLASS_COLORS = {
    1: '#FF0000',   # Class 1 = Red
    3: '#0000FF',   # Class 2 = Blue
    6: '#00C800',   # Class 3 = Green
    7: '#FFA500',   # Class 4 = Orange
}
CLASS_PAPER_NAMES = {1: 'Class 1', 3: 'Class 2', 6: 'Class 3', 7: 'Class 4'}

IMAGES = {
    'labels_clean_baseline.png': 'classification_baseline.png',
    'labels_clean_best_80.png': 'classification_best_80.png',
    'labels_clean_efficient_9.png': 'classification_efficient_9.png',
}


def crop_title(img_path: Path, out_path: Path):
    """Crop the title area from a classification map image.

    Detects where the title ends by scanning for the first row
    that is entirely white (or nearly white) below the title text.
    """
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)

    # Find rows that are all white (>= 250 in all channels)
    row_is_white = np.all(arr >= 250, axis=(1, 2))

    # The title may span multiple lines. Find the large white gap between
    # the title and the map content. We look for runs of consecutive white
    # rows and pick the first gap that is > 30 rows (title line gaps are small).
    # Scan for non-white → white transitions and measure gap lengths.
    last_nonwhite = -1
    crop_y = 0
    for y in range(arr.shape[0]):
        if not row_is_white[y]:
            last_nonwhite = y
        elif last_nonwhite >= 0 and (y - last_nonwhite) > 30:
            # Large gap found after title text — crop from just past the title
            crop_y = last_nonwhite + 1
            break

    if crop_y == 0:
        print(f"  Warning: Could not detect title boundary for {img_path.name}")

    # Also trim bottom whitespace
    crop_y_bottom = arr.shape[0]
    for y in range(arr.shape[0] - 1, crop_y, -1):
        if not row_is_white[y]:
            crop_y_bottom = y + 1
            break

    # Crop
    cropped = img.crop((0, crop_y, arr.shape[1], crop_y_bottom))
    cropped.save(out_path, dpi=(300, 300))
    print(f"  Saved: {out_path.name} (cropped rows {crop_y}–{crop_y_bottom} of {arr.shape[0]})")


def add_legend_and_roi(cropped_img_path: Path, out_path: Path, roi_regions: list,
                       crop_y: int, img_height: int, data_height: int):
    """Take a cropped clean image and render it with class legend + ROI boxes.

    The cropped image corresponds to a portion of the matplotlib-rendered figure.
    We need to map ROI coordinates (in data-pixel space) to the cropped image
    pixel space. The mapping accounts for:
      - matplotlib figure padding/margins (data area within the rendered image)
      - the title crop offset
    """
    img = Image.open(cropped_img_path)
    img_arr = np.array(img)

    # To correctly place ROI boxes, we re-render using matplotlib with the
    # same approach as the original: load the ground truth mask to get the
    # classification map dimensions, then use imshow coordinates.
    # Since we only have the rendered PNG, we'll compute the data-to-pixel
    # mapping from the original full image.

    # The original image was rendered at figsize=(5, 4.5) @ 300 dpi = 1500x1350 px.
    # After bbox_inches='tight', it gets cropped to content, yielding the actual
    # image size. The imshow data area has some padding.
    # Instead of reverse-engineering the mapping, render fresh with matplotlib.

    # Load ground truth mask to get the classification map
    data_dir = Path(__file__).parent.parent / "Data" / "processed" / "Lichens Dataset 1"
    mask_img = Image.open(data_dir / "class_mask.png").convert('RGB')
    mask_arr = np.array(mask_img)

    # Build the same display_map as the original code
    unique_classes = sorted(CLASS_COLORS.keys())
    ground_truth = np.full(mask_arr.shape[:2], -1, dtype=int)
    class_map = {1: [255, 0, 0], 3: [0, 0, 255], 6: [0, 200, 0], 7: [255, 165, 0]}
    for cls_id, color in class_map.items():
        mask = np.all(mask_arr == color, axis=2)
        ground_truth[mask] = cls_id

    # For the legend+ROI image, we use the ground truth as a stand-in for
    # the classification map (we just need the spatial layout, not predictions).
    # Actually, let's just display the cropped PNG and overlay ROI boxes
    # by computing the coordinate transform.

    # Simpler approach: re-render from scratch using the ground truth directly.
    # This gives us exact matplotlib coordinate control.
    color_list = [mcolors.hex2color(CLASS_COLORS[c]) for c in unique_classes]
    cmap = mcolors.ListedColormap(color_list)

    display_map = np.full(ground_truth.shape, np.nan, dtype=float)
    for i, cls_id in enumerate(unique_classes):
        display_map[ground_truth == cls_id] = i

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(display_map, cmap=cmap, vmin=0, vmax=len(unique_classes) - 1,
              interpolation='nearest')

    # Draw ROI rectangles
    for roi in roi_regions:
        r = roi['rect']
        rect = Rectangle(
            (r['col_min'], r['row_min']),
            r['col_max'] - r['col_min'],
            r['row_max'] - r['row_min'],
            fill=False, edgecolor='white', linewidth=2, linestyle='-'
        )
        ax.add_patch(rect)

    ax.axis('off')

    # Legend
    legend_elements = [
        Patch(facecolor=CLASS_COLORS[c], edgecolor='black', linewidth=0.5,
              label=CLASS_PAPER_NAMES[c])
        for c in unique_classes
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=12,
              framealpha=0.9, edgecolor='gray', ncol=4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, src_name in IMAGES.items():
        src_path = PAPER_FIGURES / src_name
        out_path = OUTPUT_DIR / out_name
        if not src_path.exists():
            print(f"  Skipping {src_name} (not found)")
            continue
        print(f"Processing {src_name}...")
        crop_title(src_path, out_path)

    # Also generate baseline variant with legend + ROI boxes
    baseline_clean = OUTPUT_DIR / 'labels_clean_baseline.png'
    roi_json = Path(__file__).parent.parent / "Data" / "processed" / "Lichens Dataset 1" / "roi_regions.json"
    if baseline_clean.exists() and roi_json.exists():
        with open(roi_json) as f:
            roi_data = json.load(f)
        legend_path = OUTPUT_DIR / 'labels_clean_baseline_legend.png'
        print(f"\nAdding legend + ROI boxes to baseline...")
        add_legend_and_roi(baseline_clean, legend_path, roi_data['regions'],
                           crop_y=0, img_height=0, data_height=0)

    print("\nDone!")


if __name__ == '__main__':
    main()
