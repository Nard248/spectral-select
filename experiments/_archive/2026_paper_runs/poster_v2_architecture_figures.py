#!/usr/bin/env python3
"""
Render the three architecture figures for poster v2:
  arch_pipeline.png       - headline horizontal 6-stage pipeline
  arch_cae.png            - Inset A: 3D CAE with parallel branches
  arch_perturbation.png   - Inset B: perturbation -> influence schematic

All saved to Showcase_Poster/architecture/ as both .png (300 dpi) and .pdf.
Designed to render cleanly at A0 print sizes — no overlapping text, real
heatmap/influence visuals (not placeholders), and generous padding.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

OUT_DIR = (Path(__file__).resolve().parent.parent
           / "Showcase_Poster" / "architecture")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C = {
    "intake":    "#56b870",
    "compute":   "#3a86c8",
    "output":    "#7a4ec8",
    "compute_l": "#cfe6f4",
    "accent":    "#1a2332",
    "muted":     "#5a6b7d",
    "light":     "#f3f5f9",
    "merge":     "#f0a868",
    "latent":    "#e74c5c",
    "fill":      "#fff7e6",
}


def round_box(ax, x, y, w, h, label, *, fc=C["light"], ec=C["accent"],
              lw=1.4, fontsize=11, weight="normal", color=None,
              rounding=0.035):
    if color is None:
        color = C["accent"]
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.01,rounding_size={rounding}",
                         fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=color, wrap=True)
    return box


def arrow(ax, x1, y1, x2, y2, *, lw=1.8, color=None, mutation=14):
    if color is None:
        color = C["accent"]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=mutation,
                                 lw=lw, color=color,
                                 shrinkA=2, shrinkB=2))


# ----------------------------------------------------------------------
# 1) PIPELINE  -- 6 stages, single horizontal flow
# ----------------------------------------------------------------------
def render_pipeline():
    fig, ax = plt.subplots(figsize=(16, 3.6), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    fig.suptitle("Wavelength Selection Pipeline",
                 fontsize=18, weight="bold", color=C["accent"], y=0.97)

    stages = [
        ("ME-HSI cube\n(x, y, λex, λem)", C["intake"], "white"),
        ("Preprocess\nRayleigh + mask",   C["compute_l"], C["accent"]),
        ("3D CAE\nparallel branches",     C["compute"], "white"),
        ("Perturb latent\n±15 / 30 / 45 % σ", C["compute"], "white"),
        ("MMR\nλ = 0.5",                  C["compute"], "white"),
        ("K bands\n(λex, λem)",           C["output"], "white"),
    ]

    n = len(stages)
    margin = 0.6
    gap = 0.35
    avail = 16 - 2 * margin - (n - 1) * gap
    bw = avail / n
    bh = 1.5
    by = 1.45

    centers_x = []
    for i, (label, fc, txt) in enumerate(stages):
        x = margin + i * (bw + gap)
        round_box(ax, x, by, bw, bh, label,
                  fc=fc, ec=C["accent"], lw=1.6,
                  fontsize=12, weight="bold", color=txt,
                  rounding=0.05)
        centers_x.append(x + bw / 2)

    for i in range(n - 1):
        x1 = margin + i * (bw + gap) + bw + 0.04
        x2 = margin + (i + 1) * (bw + gap) - 0.04
        arrow(ax, x1, by + bh / 2, x2, by + bh / 2, lw=2.0, mutation=18)

    # stage badges UNDER each box (one per box -> no overlap)
    badges = ["input", "preprocess", "representation",
              "attribution", "selection", "output"]
    for cx, txt in zip(centers_x, badges):
        ax.text(cx, by - 0.32, txt, ha="center", va="top",
                fontsize=10, style="italic", color=C["muted"])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "arch_pipeline.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "arch_pipeline.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# 2) CAE  -- parallel branches, shared latent, parallel decoders
# ----------------------------------------------------------------------
def render_cae():
    fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    fig.suptitle("3D Convolutional Autoencoder · parallel branches per excitation",
                 fontsize=17, weight="bold", color=C["accent"], y=0.965)

    excitations = ["310 nm", "325 nm", "340 nm", "365 nm",
                   "385 nm", "400 nm", "415 nm"]
    n = len(excitations)

    # vertical placement
    top_y = 7.6
    bot_y = 0.7
    band = (top_y - bot_y) / (n - 1)
    rows = [bot_y + i * band for i in range(n)][::-1]  # high->low

    # column x positions
    x_in_label = 0.55
    x_in_box   = 1.15
    x_enc      = 3.05
    x_avg      = 5.50
    x_latent   = 7.45
    x_dec_sh   = 9.30
    x_dec      = 11.20
    x_out_box  = 13.00
    x_out_label = 14.20

    # column header strip — placed BELOW suptitle, well clear
    header_y = 8.30
    headers = [
        (x_in_box + 0.5, "inputs"),
        (x_enc + 0.55, "parallel\nencoders"),
        (x_avg + 0.45, "merge"),
        (x_latent + 0.45, "latent"),
        (x_dec_sh + 0.55, "shared\ndecoder"),
        (x_dec + 0.55, "parallel\ndecoders"),
        (x_out_box + 0.5, "outputs"),
    ]
    for hx, htxt in headers:
        ax.text(hx, header_y, htxt, ha="center", va="center",
                fontsize=10.5, style="italic", color=C["muted"])

    # boxes per row
    bw_in = 1.0
    bh_in = 0.55
    bw_enc = 1.10
    bh_enc = 0.55

    for i, ex in enumerate(excitations):
        y = rows[i]
        # excitation label OUTSIDE to the left (no collision with input box)
        ax.text(x_in_label, y, ex, ha="right", va="center",
                fontsize=10, color=C["muted"], style="italic")
        # input X^(i)
        round_box(ax, x_in_box, y - bh_in / 2, bw_in, bh_in,
                  fr"$X^{{({i + 1})}}$",
                  fc="#e9f5ee", ec=C["accent"], lw=1.2,
                  fontsize=11, weight="bold")
        # encoder
        round_box(ax, x_enc, y - bh_enc / 2, bw_enc, bh_enc,
                  "Conv3D + σ",
                  fc=C["compute_l"], ec=C["accent"], lw=1.2,
                  fontsize=10)
        # arrows: input -> encoder -> avg merge
        arrow(ax, x_in_box + bw_in + 0.04, y, x_enc - 0.04, y, lw=1.2,
              mutation=10)
        arrow(ax, x_enc + bw_enc + 0.04, y, x_avg - 0.04, 4.15, lw=1.0,
              mutation=9, color=C["muted"])

        # decoder
        round_box(ax, x_dec, y - bh_enc / 2, bw_enc, bh_enc,
                  "Conv3D + σ",
                  fc=C["compute_l"], ec=C["accent"], lw=1.2,
                  fontsize=10)
        # output box
        round_box(ax, x_out_box, y - bh_in / 2, bw_in, bh_in,
                  fr"$\hat{{X}}^{{({i + 1})}}$",
                  fc="#e9f5ee", ec=C["accent"], lw=1.2,
                  fontsize=11, weight="bold")
        # output excitation label OUTSIDE to the right
        ax.text(x_out_label, y, ex, ha="left", va="center",
                fontsize=10, color=C["muted"], style="italic")
        # arrows: shared dec -> per-ex dec -> output
        arrow(ax, x_dec_sh + 1.10 + 0.04, 4.15, x_dec - 0.04, y, lw=1.0,
              mutation=9, color=C["muted"])
        arrow(ax, x_dec + bw_enc + 0.04, y, x_out_box - 0.04, y, lw=1.2,
              mutation=10)

    # central column: AVG merge, latent, shared decoder
    cy = 4.15
    round_box(ax, x_avg, cy - 0.55, 1.10, 1.10, "AVG\nmerge",
              fc=C["merge"], ec=C["accent"], lw=1.5,
              fontsize=11, weight="bold", color="white", rounding=0.05)
    round_box(ax, x_latent, cy - 0.55, 1.10, 1.10, r"$z \in \mathbb{R}^{20}$",
              fc=C["latent"], ec=C["accent"], lw=1.6,
              fontsize=12, weight="bold", color="white", rounding=0.07)
    round_box(ax, x_dec_sh, cy - 0.55, 1.10, 1.10, "Conv3D\nshared",
              fc=C["merge"], ec=C["accent"], lw=1.5,
              fontsize=11, weight="bold", color="white", rounding=0.05)

    # arrows between centre boxes
    arrow(ax, x_avg + 1.10 + 0.04, cy, x_latent - 0.04, cy, lw=1.8, mutation=14)
    arrow(ax, x_latent + 1.10 + 0.04, cy, x_dec_sh - 0.04, cy, lw=1.8, mutation=14)

    # single bottom caption — well below all boxes
    cap = ("One encoder branch per excitation; features averaged into a 20-dim "
           "shared latent. Reconstruction uses a shared decoder + per-excitation decoders.")
    ax.text(7.5, 0.10, cap, ha="center", va="center",
            fontsize=10.5, style="italic", color=C["muted"])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "arch_cae.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "arch_cae.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3) PERTURBATION -> INFLUENCE  -- 3 panels with REAL imagery
# ----------------------------------------------------------------------
def render_perturbation():
    fig = plt.figure(figsize=(13, 5.6), dpi=300)
    fig.text(0.5, 0.955, "Latent perturbation → influence map",
             ha="center", va="center",
             fontsize=15, weight="bold", color=C["accent"])

    # narrow panel (1) so circles aren't squeezed; wider panels (2)+(3)
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.985,
                          bottom=0.26, top=0.83,
                          width_ratios=[0.85, 1.0, 1.0],
                          wspace=0.32)

    # ---- Panel 1: latent z with top-k highlighted -----------------------
    # Show: 0, 1, 2, ..., k-1, k  (only 6 visible items so circles are big
    # and numbers readable). First three are top-k (red).
    ax1 = fig.add_subplot(gs[0, 0])
    items = [
        ("0",   "circle", True),
        ("1",   "circle", True),
        ("2",   "circle", True),
        ("…",   "ellipsis", False),
        ("k-1", "circle", False),
        ("k",   "circle", False),
    ]
    n_items = len(items)
    spacing = 1.4
    total_w = (n_items - 1) * spacing
    ax1.set_xlim(-spacing * 0.8, total_w + spacing * 0.8)
    ax1.set_ylim(-3.2, 1.6)
    ax1.axis("off")
    ax1.set_title(r"(1) latent  $z \in \mathbb{R}^{20}$",
                  fontsize=14, weight="bold", color=C["accent"], pad=10)

    radius = 0.55
    for i, (label, kind, is_top) in enumerate(items):
        x = i * spacing
        if kind == "circle":
            fc = C["latent"] if is_top else "#b9c4d0"
            txt_color = "white" if is_top else C["muted"]
            ax1.add_patch(Circle((x, 0), radius,
                                 fc=fc, ec=C["accent"], lw=1.2))
            ax1.text(x, 0, label, ha="center", va="center",
                     fontsize=14, weight="bold", color=txt_color)
        else:
            ax1.text(x, 0, "···", ha="center", va="center",
                     fontsize=22, weight="bold", color=C["muted"])

    mid_x = total_w / 2
    ax1.text(mid_x, -1.45,
             "top-$k$ variance-ranked dims",
             ha="center", va="center", fontsize=12,
             color="#000000", style="italic")
    ax1.text(mid_x, -2.45,
             r"$z'_d = z \pm \varepsilon \cdot \sigma_d \cdot e_d$",
             ha="center", va="center", fontsize=14, color=C["accent"])

    # ---- Panel 2: per-dim sensitivity heatmap (REAL) --------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(2) per-dim sensitivity  $|\\Delta \\hat X(\\lambda_{ex}, \\lambda_{em})|$",
                  fontsize=12, weight="bold", color=C["accent"], pad=8)

    n_ex = 7
    n_em = 31
    rng2 = np.random.default_rng(2)
    base = rng2.random((n_ex, n_em)) * 0.3
    # add several blobs
    for cy_, cx_, amp, sx, sy in [
        (1.2, 8,  1.0, 4, 1.4),
        (3.5, 14, 0.85, 5, 1.6),
        (5.2, 22, 0.7,  6, 1.4),
        (2.0, 26, 0.5,  4, 1.2),
    ]:
        yy, xx = np.mgrid[0:n_ex, 0:n_em]
        base += amp * np.exp(-(((xx - cx_) / sx) ** 2
                               + ((yy - cy_) / sy) ** 2))
    sens = base / base.max()
    im = ax2.imshow(sens, cmap="magma", aspect="auto",
                    extent=[450, 770, 415, 310], interpolation="bilinear")
    ax2.set_xlabel(r"$\lambda_{em}$ (nm)", fontsize=10.5, color="#000000")
    ax2.set_ylabel(r"$\lambda_{ex}$ (nm)", fontsize=10.5, color="#000000")
    ax2.tick_params(labelsize=9, colors="#000000")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cbar.set_label("sensitivity", fontsize=9.5, color="#000000")
    cbar.ax.tick_params(labelsize=8, colors="#000000")

    # ---- Panel 3: aggregated influence map ------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("(3) aggregated influence  →  MMR",
                  fontsize=12, weight="bold", color=C["accent"], pad=8)

    rng3 = np.random.default_rng(11)
    influence = rng3.random((n_ex, n_em)) * 0.2
    for cy_, cx_, amp, sx, sy in [
        (3.5, 11, 1.0, 3, 1.0),
        (1.8, 14, 0.9, 3, 1.0),
        (5.0, 14, 0.75, 3, 1.0),
        (3.0, 21, 0.8, 3, 1.0),
        (4.2, 9,  0.65, 2.5, 0.9),
    ]:
        yy, xx = np.mgrid[0:n_ex, 0:n_em]
        influence += amp * np.exp(-(((xx - cx_) / sx) ** 2
                                    + ((yy - cy_) / sy) ** 2))
    influence /= influence.max()
    im3 = ax3.imshow(influence, cmap="viridis", aspect="auto",
                     extent=[450, 770, 415, 310], interpolation="bilinear")

    # mark top-K picks
    flat_idx = np.argsort(-influence.ravel())[:5]
    em_grid = np.linspace(450, 770, n_em)
    ex_grid = np.linspace(310, 415, n_ex)
    for idx in flat_idx:
        ei, mi = np.unravel_index(idx, influence.shape)
        ax3.scatter(em_grid[mi], ex_grid[ei], s=80,
                    facecolor="none", edgecolor="white", lw=1.8, zorder=5)

    ax3.set_xlabel(r"$\lambda_{em}$ (nm)", fontsize=10.5, color="#000000")
    ax3.set_ylabel(r"$\lambda_{ex}$ (nm)", fontsize=10.5, color="#000000")
    ax3.tick_params(labelsize=9, colors="#000000")
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)
    cbar3.set_label("influence", fontsize=9.5, color="#000000")
    cbar3.ax.tick_params(labelsize=8, colors="#000000")

    fig.text(0.5, 0.10,
             "(1) perturb top-k dims  →  (2) measure $|\\Delta\\hat X|$ per "
             "$(\\lambda_{ex},\\lambda_{em})$  →  (3) aggregate  →  MMR",
             ha="center", va="center",
             fontsize=12.5, color=C["accent"])
    fig.text(0.5, 0.035,
             "White circles in panel (3) mark the K bands MMR ultimately selects.",
             ha="center", va="center",
             fontsize=11, style="italic", color="#000000")

    # save WITHOUT bbox_inches="tight" so figure coords stay stable.
    fig.savefig(OUT_DIR / "arch_perturbation.png", dpi=300,
                facecolor="white", pad_inches=0.2)
    fig.savefig(OUT_DIR / "arch_perturbation.pdf",
                facecolor="white", pad_inches=0.2)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4) OVERVIEW  -- single-image whole-pipeline story
# ----------------------------------------------------------------------
def render_overview():
    """One figure that walks from raw 4D cube to selected bands, with
    real-looking visuals at every stage. Designed as the headline poster
    figure — readable from a meter away."""
    fig = plt.figure(figsize=(20, 9), dpi=300)
    fig.text(0.5, 0.965,
             "spectral-select · end-to-end pipeline",
             ha="center", va="center",
             fontsize=20, weight="bold", color=C["accent"])
    fig.text(0.5, 0.928,
             "from raw 4-D ME-HSI cube to a ranked list of K (λex, λem) bands",
             ha="center", va="center",
             fontsize=12.5, style="italic", color=C["muted"])

    # 5 panels horizontally; below each, a stage badge.
    gs = fig.add_gridspec(2, 5,
                          left=0.035, right=0.985,
                          bottom=0.14, top=0.86,
                          height_ratios=[1.0, 0.06],
                          wspace=0.32, hspace=0.05)

    rng = np.random.default_rng(42)

    # ---- Panel A: stack of excitation cubes ---------------------------
    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("(A) ME-HSI cube\n7 excitations × 31 emission bands",
                  fontsize=12, weight="bold", color=C["accent"], pad=8)
    axA.set_xlim(0, 10)
    axA.set_ylim(0, 10)
    axA.axis("off")

    excs = ["310", "325", "340", "365", "385", "400", "415"]
    for i, ex in enumerate(excs):
        offset = i * 0.45
        x0, y0 = 1.5 + offset, 1.5 + offset
        # cube face
        face = rng.random((10, 10))
        # smooth-ish blob
        yy, xx = np.mgrid[0:10, 0:10]
        face = 0.3 * face + 0.9 * np.exp(-(((xx - 5) / 2.5) ** 2
                                           + ((yy - 5) / 2.5) ** 2))
        face /= face.max()
        axA.imshow(face, extent=[x0, x0 + 4, y0, y0 + 4],
                   cmap="magma", aspect="auto",
                   alpha=0.95, zorder=10 - i, interpolation="bilinear")
        axA.add_patch(FancyBboxPatch((x0, y0), 4, 4,
                                     boxstyle="square,pad=0.0",
                                     fc="none", ec=C["accent"], lw=1.0,
                                     zorder=11 - i))
        if i in (0, len(excs) - 1):
            axA.text(x0 - 0.15, y0 + 2, f"{ex} nm",
                     ha="right", va="center", fontsize=8,
                     color=C["muted"], style="italic", zorder=20)
    axA.text(5, 0.6, "spatial × spectral × excitation",
             ha="center", va="center", fontsize=9.5,
             color=C["muted"], style="italic")

    # ---- Panel B: CAE compact -----------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("(B) 3-D CAE\nparallel branches → shared latent",
                  fontsize=12, weight="bold", color=C["accent"], pad=8)
    axB.set_xlim(0, 10)
    axB.set_ylim(0, 10)
    axB.axis("off")

    # 7 input dots on left, encoder bars, AVG, latent
    for i in range(7):
        y = 8.6 - i * 1.05
        axB.add_patch(Circle((0.7, y), 0.22, fc="#e9f5ee",
                             ec=C["accent"], lw=0.9))
        # encoder rect
        round_box(axB, 1.4, y - 0.22, 1.4, 0.44, "enc",
                  fc=C["compute_l"], ec=C["accent"], lw=0.9,
                  fontsize=8, rounding=0.05)
        # arrow into AVG
        arrow(axB, 2.85, y, 4.55, 5.0, lw=0.7, color=C["muted"], mutation=7)
    # AVG box
    round_box(axB, 4.6, 4.1, 1.5, 1.8, "AVG\nmerge",
              fc=C["merge"], ec=C["accent"], lw=1.3,
              fontsize=10, weight="bold", color="white", rounding=0.05)
    # latent
    round_box(axB, 6.7, 4.1, 1.7, 1.8, r"$z \in \mathbb{R}^{20}$",
              fc=C["latent"], ec=C["accent"], lw=1.5,
              fontsize=11, weight="bold", color="white", rounding=0.07)
    arrow(axB, 6.1, 5.0, 6.65, 5.0, lw=1.5)
    # decoder hint
    round_box(axB, 8.7, 4.1, 1.1, 1.8, "dec", fc=C["compute_l"],
              ec=C["accent"], lw=0.9, fontsize=9, rounding=0.05)
    arrow(axB, 8.4, 5.0, 8.65, 5.0, lw=1.4)

    axB.text(5, 0.6, "20-dim bottleneck forces shared structure",
             ha="center", va="center", fontsize=9.5,
             color=C["muted"], style="italic")

    # ---- Panel C: latent perturbation ---------------------------------
    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("(C) Latent perturbation\n$z'_d = z \\pm \\varepsilon \\sigma_d e_d$",
                  fontsize=12, weight="bold", color=C["accent"], pad=8)
    axC.set_xlim(-0.5, 20.5)
    axC.set_ylim(-3, 7)
    axC.axis("off")

    var_rank = np.argsort(-rng.random(20))
    top_set = set(var_rank[:5])
    # original z
    for d in range(20):
        c = C["latent"] if d in top_set else "#b9c4d0"
        axC.add_patch(Circle((d, 5.0), 0.45, fc=c, ec=C["accent"], lw=0.7))
    axC.text(10, 6.4, "z  (variance-ranked)", ha="center", fontsize=10,
             color=C["muted"], style="italic")
    # perturbed z below — only top-k changed, with up/down arrows
    for d in range(20):
        c = C["latent"] if d in top_set else "#b9c4d0"
        axC.add_patch(Circle((d, 1.6), 0.45, fc=c, ec=C["accent"], lw=0.7))
        if d in top_set:
            axC.annotate("", xy=(d, 2.7), xytext=(d, 4.0),
                         arrowprops=dict(arrowstyle="<->",
                                         lw=1.2, color=C["latent"]))
    axC.text(10, 0.05, "z′  (perturbed top-k only)",
             ha="center", fontsize=10, color=C["muted"], style="italic")
    axC.text(10, -2.0,
             "→ measure  $|\\Delta \\hat X(\\lambda_{ex}, \\lambda_{em})|$",
             ha="center", fontsize=10.5, color=C["accent"])

    # ---- Panel D: influence map ---------------------------------------
    axD = fig.add_subplot(gs[0, 3])
    axD.set_title("(D) Influence map\naggregated  $|\\Delta \\hat X|$  over dims",
                  fontsize=12, weight="bold", color=C["accent"], pad=8)

    n_ex, n_em = 7, 31
    influence = rng.random((n_ex, n_em)) * 0.15
    yy, xx = np.mgrid[0:n_ex, 0:n_em]
    blobs = [(3.5, 11, 1.0, 3.0, 1.0),
             (1.8, 14, 0.92, 3.0, 1.0),
             (5.0, 14, 0.78, 3.0, 1.0),
             (3.0, 21, 0.85, 3.0, 1.0),
             (4.2, 9,  0.62, 2.5, 0.9)]
    for cy_, cx_, amp, sx, sy in blobs:
        influence += amp * np.exp(-(((xx - cx_) / sx) ** 2
                                    + ((yy - cy_) / sy) ** 2))
    influence /= influence.max()
    im = axD.imshow(influence, cmap="viridis", aspect="auto",
                    extent=[450, 770, 415, 310], interpolation="bilinear")
    axD.set_xlabel(r"$\lambda_{em}$ (nm)", fontsize=10, color=C["muted"])
    axD.set_ylabel(r"$\lambda_{ex}$ (nm)", fontsize=10, color=C["muted"])
    axD.tick_params(labelsize=8.5, colors=C["muted"])
    cb = fig.colorbar(im, ax=axD, fraction=0.045, pad=0.03)
    cb.set_label("influence", fontsize=9, color=C["muted"])
    cb.ax.tick_params(labelsize=8, colors=C["muted"])

    # ---- Panel E: MMR-selected K bands --------------------------------
    axE = fig.add_subplot(gs[0, 4])
    axE.set_title("(E) MMR  →  K bands\nλ = 0.5  (relevance ↔ diversity)",
                  fontsize=12, weight="bold", color=C["accent"], pad=8)

    axE.imshow(influence, cmap="viridis", aspect="auto",
               extent=[450, 770, 415, 310], interpolation="bilinear",
               alpha=0.55)

    em_grid = np.linspace(450, 770, n_em)
    ex_grid = np.linspace(310, 415, n_ex)
    flat_idx = np.argsort(-influence.ravel())[:5]
    for k, idx in enumerate(flat_idx, start=1):
        ei, mi = np.unravel_index(idx, influence.shape)
        axE.scatter(em_grid[mi], ex_grid[ei], s=180,
                    facecolor="white", edgecolor=C["accent"],
                    lw=1.6, zorder=5)
        axE.text(em_grid[mi], ex_grid[ei], str(k),
                 ha="center", va="center", fontsize=10,
                 weight="bold", color=C["accent"], zorder=6)
    axE.set_xlabel(r"$\lambda_{em}$ (nm)", fontsize=10, color=C["muted"])
    axE.set_ylabel(r"$\lambda_{ex}$ (nm)", fontsize=10, color=C["muted"])
    axE.tick_params(labelsize=8.5, colors=C["muted"])

    # ---- inter-panel arrows  ------------------------------------------
    # placed using figure coords. Save WITHOUT bbox_inches="tight" so they
    # remain aligned.
    arrow_y = 0.50
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=22,
                    lw=2.0, color=C["accent"],
                    transform=fig.transFigure, clip_on=False)
    # gaps between gridspec columns: roughly evenly spaced; compute from gs
    # by querying axes positions.
    axes_list = [axA, axB, axC, axD, axE]
    for left_ax, right_ax in zip(axes_list[:-1], axes_list[1:]):
        bb_l = left_ax.get_position()
        bb_r = right_ax.get_position()
        x1 = bb_l.x1 + 0.005
        x2 = bb_r.x0 - 0.005
        fig.patches.append(FancyArrowPatch((x1, arrow_y),
                                           (x2, arrow_y),
                                           **arrow_kw))

    # ---- stage badges (row 1 of gridspec) -----------------------------
    badges = [
        "input cube",
        "encode + bottleneck",
        "perturb latent",
        "aggregate sensitivity",
        "rank + select K",
    ]
    for i, txt in enumerate(badges):
        bx = fig.add_subplot(gs[1, i])
        bx.axis("off")
        bx.text(0.5, 0.5, txt, ha="center", va="center",
                fontsize=11, weight="bold", color=C["muted"])

    # ---- bottom one-line takeaway -------------------------------------
    fig.text(0.5, 0.045,
             "One forward pass per perturbation → influence matrix → "
             "≤ 13 (λex, λem) bands that match full-cube accuracy.",
             ha="center", va="center",
             fontsize=11.5, style="italic", color=C["accent"])

    fig.savefig(OUT_DIR / "arch_overview.png", dpi=300,
                facecolor="white", pad_inches=0.2)
    fig.savefig(OUT_DIR / "arch_overview.pdf",
                facecolor="white", pad_inches=0.2)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5) METHOD U-SHAPE  -- mirrors the TikZ pipeline figure used in the paper
# ----------------------------------------------------------------------
def render_method_ushape():
    """Reproduce the LaTeX TikZ pipeline as a PNG/PDF for slides and posters.
    Top row L→R: Input → Preprocess → Encoder → Latent → Decoder
    Bottom row R→L: Perturb → Influence → MMR → Output
    Brace labelled "3D CAE (Training)" sits ABOVE the encoder/latent/decoder
    triple so it does not collide with arrows."""
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # palette — matches the TikZ blue!10 / green!10 / orange!10 / purple!10
    blue10   = "#dbe5f4"
    green10  = "#d8edd8"
    orange10 = "#fae3cf"
    purple10 = "#e8dcef"
    edge     = "#1a2332"

    bw, bh = 2.2, 1.25
    top_y = 4.30
    bot_y = 1.45

    # X centres (top row L→R)
    cx_input    = 1.55
    cx_prep     = 4.15
    cx_enc      = 6.75
    cx_latent   = 9.35
    cx_dec      = 11.95

    # X centres (bottom row R→L; mirror the top columns)
    cx_perturb  = cx_latent     # under the latent
    cx_influence = cx_enc        # under the encoder
    cx_mmr      = cx_prep        # under the preprocess
    cx_output   = cx_input       # under the input

    # ---- top row ----
    def box(cx, cy, w, h, text, fc, fontsize=11, weight="bold"):
        round_box(ax, cx - w / 2, cy - h / 2, w, h, text,
                  fc=fc, ec=edge, lw=1.4,
                  fontsize=fontsize, weight=weight,
                  color=edge, rounding=0.05)

    box(cx_input,  top_y, 2.0, 1.05, "4D ME-HSI\n$(x,y,\\lambda_{em},\\lambda_{ex})$",
        fc=green10, fontsize=10.5, weight="bold")
    box(cx_prep,   top_y, bw, bh, "Preprocessing\nNormalization\nRayleigh Cutoff",
        fc=blue10, fontsize=10.5)
    box(cx_enc,    top_y, bw, bh, "Parallel\nEncoder\nBranches",
        fc=blue10, fontsize=10.5)
    box(cx_latent, top_y, bw, bh, "Shared\nLatent Space\n$z \\in \\mathbb{R}^{20}$",
        fc=blue10, fontsize=10.5)
    box(cx_dec,    top_y, bw, bh, "Parallel\nDecoder\nBranches",
        fc=blue10, fontsize=10.5)

    # ---- bottom row ----
    box(cx_perturb,   bot_y, bw, bh, "Latent Space\nPerturbation\nAnalysis",
        fc=orange10, fontsize=10.5)
    box(cx_influence, bot_y, bw, bh, "Wavelength\nInfluence\nScores",
        fc=orange10, fontsize=10.5)
    box(cx_mmr,       bot_y, bw, bh, "MMR\nDiversity\nSelection",
        fc=orange10, fontsize=10.5)
    box(cx_output,    bot_y, 2.0, 1.05, "Selected\nWavelengths",
        fc=purple10, fontsize=10.5, weight="bold")

    # ---- arrows top row L→R ----
    pad = 0.04
    def harrow(cx1, cx2, y, halfw=bw / 2):
        x1 = cx1 + halfw + pad
        x2 = cx2 - halfw - pad
        arrow(ax, x1, y, x2, y, lw=1.6, mutation=14)

    harrow(cx_input,  cx_prep,   top_y, halfw=1.0)
    harrow(cx_prep,   cx_enc,    top_y)
    harrow(cx_enc,    cx_latent, top_y)
    harrow(cx_latent, cx_dec,    top_y)

    # ---- arrows bottom row R→L ----
    def harrow_left(cx1, cx2, y, halfw=bw / 2):
        x1 = cx1 - halfw - pad
        x2 = cx2 + halfw + pad
        arrow(ax, x1, y, x2, y, lw=1.6, mutation=14)

    harrow_left(cx_perturb,   cx_influence, bot_y)
    harrow_left(cx_influence, cx_mmr,       bot_y)
    harrow_left(cx_mmr,       cx_output,    bot_y, halfw=1.0)

    # ---- vertical arrow: latent → perturb ----
    arrow(ax, cx_latent, top_y - bh / 2 - pad,
          cx_perturb, bot_y + bh / 2 + pad,
          lw=1.6, mutation=14)

    # ---- brace ABOVE encoder-latent-decoder, with label above brace ----
    brace_y = top_y + bh / 2 + 0.20
    brace_top_y = brace_y + 0.45
    x_l = cx_enc - bw / 2
    x_r = cx_dec + bw / 2
    # top-pointing brace built from two arcs and a centre tick
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    arc_w = 0.30
    arc_h = 0.45
    mid_x = (x_l + x_r) / 2
    verts = [
        (x_l, brace_y),
        (x_l, brace_y + arc_h * 0.55),
        (x_l + arc_w, brace_top_y - 0.05),
        (x_l + arc_w + 0.10, brace_top_y),
        (mid_x - 0.10, brace_top_y),
        (mid_x, brace_top_y + 0.18),
        (mid_x + 0.10, brace_top_y),
        (x_r - arc_w - 0.10, brace_top_y),
        (x_r - arc_w, brace_top_y - 0.05),
        (x_r, brace_y + arc_h * 0.55),
        (x_r, brace_y),
    ]
    codes = [Path.MOVETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3]
    # simpler: draw two quarter-curves + a straight top with a centre tick
    # via Bezier path
    import matplotlib.path as mpath
    Path = mpath.Path
    p1 = Path([(x_l, brace_y),
               (x_l, brace_top_y),
               (x_l + 0.5, brace_top_y),
               (mid_x - 0.10, brace_top_y)],
              [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    p2 = Path([(mid_x + 0.10, brace_top_y),
               (x_r - 0.5, brace_top_y),
               (x_r, brace_top_y),
               (x_r, brace_y)],
              [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    # centre tick
    p3 = Path([(mid_x - 0.10, brace_top_y),
               (mid_x, brace_top_y + 0.16),
               (mid_x + 0.10, brace_top_y)],
              [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    for p in (p1, p2, p3):
        ax.add_patch(PathPatch(p, fc="none", ec=edge, lw=1.4))

    # label ABOVE the brace
    ax.text(mid_x, brace_top_y + 0.50, "3D CAE (Training)",
            ha="center", va="bottom",
            fontsize=11.5, weight="bold", color=edge)

    # ---- stage labels ----
    ax.text(cx_enc, top_y + bh / 2 + 1.55,
            "Stage 1: Representation Learning",
            ha="center", va="bottom",
            fontsize=10, style="italic", color=C["muted"])
    ax.text(cx_influence, bot_y - bh / 2 - 0.30,
            "Stage 2: Attribution",
            ha="center", va="top",
            fontsize=10, style="italic", color=C["muted"])
    ax.text(cx_mmr, bot_y - bh / 2 - 0.30,
            "Stage 3: Selection",
            ha="center", va="top",
            fontsize=10, style="italic", color=C["muted"])

    # title
    fig.text(0.5, 0.965,
             "Wavelength selection framework — methodology overview",
             ha="center", va="center",
             fontsize=14, weight="bold", color=edge)

    fig.savefig(OUT_DIR / "arch_method_ushape.png", dpi=300,
                facecolor="white", pad_inches=0.2,
                bbox_inches="tight")
    fig.savefig(OUT_DIR / "arch_method_ushape.pdf",
                facecolor="white", pad_inches=0.2,
                bbox_inches="tight")
    plt.close(fig)


def main():
    render_pipeline()
    render_cae()
    render_perturbation()
    render_overview()
    render_method_ushape()
    print(f"Wrote 5 architecture figures (PNG + PDF) to {OUT_DIR}")


if __name__ == "__main__":
    main()
