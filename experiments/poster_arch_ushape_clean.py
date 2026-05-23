#!/usr/bin/env python3
"""Two-row U-shape architecture diagram for the poster.

Row 1 (top, left → right):
    start image · 4D ME-HSI · Preprocessing · Encoder · Latent · Decoder
Row 2 (bottom, right → left):
    Latent Perturbation · Wavelength Influence · MMR · Selected Wavelengths · finish image

Single vertical drop at the latent column joins the two rows.

Output: Showcase_Poster/architecture/arch_method_ushape_clean.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
ARCH_DIR = ROOT / "Showcase_Poster" / "architecture"
OUT_PNG = ARCH_DIR / "arch_method_ushape_clean.png"
OUT_PDF = ARCH_DIR / "arch_method_ushape_clean.pdf"
START_IMG = ARCH_DIR / "lichens_start.jpg"
FINISH_IMG = ARCH_DIR / "lichens_finish.jpg"

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------
GREEN_FILL  = "#B7DDB9"
BLUE_FILL   = "#A8C5E0"
LATENT_FILL = "#7BA8D0"
PEACH_FILL  = "#F4C998"
PURPLE_FILL = "#C4B0DC"
BORDER      = "#1F2A37"
TEXT_BLACK  = "#000000"

# ----------------------------------------------------------------------
# Geometry — strict 6-column grid, 2 rows
# ----------------------------------------------------------------------
BOX_W, BOX_H = 2.30, 0.95
GRID_X = {"img": 1.5, "input": 4.05, "preproc": 6.55,
          "enc": 9.05, "latent": 11.55, "dec": 14.05}
TOP_Y = 3.55
BOT_Y = 1.10
CANVAS_W, CANVAS_H = 15.5, 4.85

ARROW_KW = dict(arrowstyle="-|>,head_length=10,head_width=6",
                lw=2.0, color=BORDER, mutation_scale=1.0,
                shrinkA=2, shrinkB=2)


def add_box(ax, cx, cy, label, sublabel=None, fill=BLUE_FILL,
            w=BOX_W, h=BOX_H, label_size=10, sub_size=8):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor=BORDER, facecolor=fill, zorder=2)
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.13, label, ha="center", va="center",
                fontsize=label_size, fontweight="bold",
                color=TEXT_BLACK, zorder=3)
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=sub_size, color=TEXT_BLACK, zorder=3)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=label_size, fontweight="bold",
                color=TEXT_BLACK, zorder=3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), **ARROW_KW))


def add_image(ax, path, cx, cy, zoom=0.24):
    img = mpimg.imread(str(path))
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, (cx, cy), frameon=True,
                        bboxprops=dict(edgecolor=BORDER, linewidth=1.2))
    ax.add_artist(ab)


def add_brace(ax, x0, x1, y, depth=0.14, label=None, label_dy=0.24):
    mid = (x0 + x1) / 2
    ax.plot([x0, x0, mid - 0.05, mid - 0.05],
            [y, y + depth, y + depth, y + depth + 0.07],
            color=BORDER, lw=1.4, solid_capstyle="round", zorder=1)
    ax.plot([x1, x1, mid + 0.05, mid + 0.05],
            [y, y + depth, y + depth, y + depth + 0.07],
            color=BORDER, lw=1.4, solid_capstyle="round", zorder=1)
    ax.plot([mid - 0.05, mid + 0.05],
            [y + depth + 0.07, y + depth + 0.07],
            color=BORDER, lw=1.4, solid_capstyle="round", zorder=1)
    if label:
        ax.text(mid, y + depth + label_dy, label,
                ha="center", va="bottom", fontsize=10,
                fontstyle="italic", color=TEXT_BLACK, zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H), dpi=300)
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Lichens images on the left -----------------------------------
    add_image(ax, START_IMG,  GRID_X["img"], TOP_Y, zoom=0.24)
    add_image(ax, FINISH_IMG, GRID_X["img"], BOT_Y, zoom=0.24)

    # ---- Top row -------------------------------------------------------
    add_box(ax, GRID_X["input"],   TOP_Y, "4D ME-HSI",
            sublabel=r"$(x,\,y,\,\lambda_{em},\,\lambda_{ex})$",
            fill=GREEN_FILL)
    add_box(ax, GRID_X["preproc"], TOP_Y, "Preprocessing",
            sublabel="Normalization · Rayleigh cutoff", fill=BLUE_FILL)
    add_box(ax, GRID_X["enc"],     TOP_Y, "Parallel Encoder",
            sublabel="per-excitation branches", fill=BLUE_FILL)
    add_box(ax, GRID_X["latent"],  TOP_Y, "Shared Latent",
            sublabel=r"$z\in\mathbb{R}^{20}$", fill=LATENT_FILL)
    add_box(ax, GRID_X["dec"],     TOP_Y, "Parallel Decoder",
            sublabel="per-excitation branches", fill=BLUE_FILL)

    # ---- Bottom row (right → left) -------------------------------------
    add_box(ax, GRID_X["latent"], BOT_Y, "Latent Perturbation",
            sublabel=r"$z'_d = z \pm \varepsilon\,\sigma_d\,e_d$",
            fill=PEACH_FILL)
    add_box(ax, GRID_X["enc"],    BOT_Y, "Wavelength Influence",
            sublabel=r"$|\Delta\hat X(\lambda_{ex},\lambda_{em})|$",
            fill=PEACH_FILL)
    add_box(ax, GRID_X["preproc"], BOT_Y, "MMR Selection",
            sublabel=r"diverse re-rank, $\lambda{=}0.5$",
            fill=PEACH_FILL)
    add_box(ax, GRID_X["input"],   BOT_Y, "Selected λ",
            sublabel="diverse top-K bands",
            fill=PURPLE_FILL)

    # ---- Arrows --------------------------------------------------------
    half = BOX_W / 2
    img_edge = 0.62

    arrow(ax, GRID_X["img"]    + img_edge, TOP_Y, GRID_X["input"]   - half, TOP_Y)
    arrow(ax, GRID_X["input"]  + half,     TOP_Y, GRID_X["preproc"] - half, TOP_Y)
    arrow(ax, GRID_X["preproc"]+ half,     TOP_Y, GRID_X["enc"]     - half, TOP_Y)
    arrow(ax, GRID_X["enc"]    + half,     TOP_Y, GRID_X["latent"]  - half, TOP_Y)
    arrow(ax, GRID_X["latent"] + half,     TOP_Y, GRID_X["dec"]     - half, TOP_Y)

    # latent → perturbation (single vertical drop)
    arrow(ax, GRID_X["latent"], TOP_Y - BOX_H/2,
                 GRID_X["latent"], BOT_Y + BOX_H/2)

    # bottom row leftward chain
    arrow(ax, GRID_X["latent"]  - half, BOT_Y, GRID_X["enc"]     + half, BOT_Y)
    arrow(ax, GRID_X["enc"]     - half, BOT_Y, GRID_X["preproc"] + half, BOT_Y)
    arrow(ax, GRID_X["preproc"] - half, BOT_Y, GRID_X["input"]   + half, BOT_Y)

    # Selected → finish image
    arrow(ax, GRID_X["input"] - half, BOT_Y, GRID_X["img"] + img_edge, BOT_Y)

    # ---- Stage labels --------------------------------------------------
    add_brace(ax,
              GRID_X["enc"]    - half - 0.05,
              GRID_X["dec"]    + half + 0.05,
              TOP_Y + BOX_H/2 + 0.12,
              depth=0.14,
              label="Stage 1 · Representation Learning  ·  3D CAE training",
              label_dy=0.24)

    ax.text((GRID_X["enc"] + GRID_X["latent"]) / 2,
            BOT_Y - BOX_H/2 - 0.18,
            "Stage 2 · Attribution",
            ha="center", va="top", fontsize=10, fontstyle="italic",
            color=TEXT_BLACK)

    ax.text((GRID_X["input"] + GRID_X["preproc"]) / 2,
            BOT_Y - BOX_H/2 - 0.18,
            "Stage 3 · Selection",
            ha="center", va="top", fontsize=10, fontstyle="italic",
            color=TEXT_BLACK)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.10)
    fig.savefig(OUT_PDF, bbox_inches="tight",
                facecolor="white", pad_inches=0.10)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
