"""Render a generic (de-branded) architecture figure for the CODASSCA paper.

Shows the dependency-aware encoder: per-channel-group convolutional branches,
latent fusion by averaging, a shared latent code, and parallel decoders. Labels
are deliberately generic ("channel group", "fusion") rather than ME-HSI-specific
("excitation", "Conv3D") so the figure matches the generalized framing.

Output: fig_architecture.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "fig_architecture.png"

GREEN = "#CDE7CB"
BLUE = "#CBD6F2"
ORANGE = "#FBE2BE"
PINK = "#F6C9CE"
EDGE = "#4A4A4A"


def box(ax, x, y, w, h, text, fill, fontsize=10, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                       linewidth=1.2, edgecolor=EDGE, facecolor=fill, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal", zorder=3)
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                        linewidth=1.1, color=EDGE, zorder=1,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    bw, bh = 10, 13           # input/output box size
    cw = 11                   # parallel conv box width
    rows_y = [74, 45, 8]      # group 1, group 2, group N
    in_x = 1.5
    enc_x = 13.5
    avg_x, avg_w = 27.5, 7.5      # average (fusion)
    es_x, es_w = 37.5, 7.5       # encoder shared conv (enc_conv3)
    z_x, z_w = 47.5, 6           # latent z
    ds_x, ds_w = 56, 7.5         # decoder shared conv (dec_conv1)
    dec_x = 67
    out_x = 81

    cy = 45
    ch = 13                       # center-row box height
    cyb = cy - ch / 2             # center-row box bottom

    # Column header labels
    ax.text(in_x + bw / 2, 95, "Channel\ngroups", ha="center", fontsize=8.5, style="italic")
    ax.text(enc_x + cw / 2, 95, "Parallel\nencoders", ha="center", fontsize=8.5, style="italic")
    ax.text(avg_x + avg_w / 2, 95, "Fusion", ha="center", fontsize=8.5, style="italic")
    ax.text((es_x + ds_x + ds_w) / 2 + 1, 95, "Shared convs / latent",
            ha="center", fontsize=8.5, style="italic")
    ax.text(dec_x + cw / 2, 95, "Parallel\ndecoders", ha="center", fontsize=8.5, style="italic")
    ax.text(out_x + bw / 2, 95, "Reconstr.", ha="center", fontsize=8.5, style="italic")

    labels = [r"$\mathbf{X}^{(1)}$", r"$\mathbf{X}^{(2)}$", r"$\mathbf{X}^{(N)}$"]
    outs = [r"$\hat{\mathbf{X}}^{(1)}$", r"$\hat{\mathbf{X}}^{(2)}$", r"$\hat{\mathbf{X}}^{(N)}$"]

    # Center spine: average -> enc shared conv -> z -> dec shared conv
    box(ax, avg_x, cy - 13, avg_w, 26, "Average\n(fusion)", ORANGE, fontsize=8, bold=True)
    box(ax, es_x, cyb, es_w, ch, "Conv\n(shared)", BLUE, fontsize=8)
    box(ax, z_x, cyb, z_w, ch, r"$\mathbf{z}$", PINK, fontsize=13, bold=True)
    box(ax, ds_x, cyb, ds_w, ch, "Conv\n(shared)", BLUE, fontsize=8)

    for i, ry in enumerate(rows_y):
        box(ax, in_x, ry, bw, bh, labels[i], GREEN, fontsize=11, bold=True)
        box(ax, enc_x, ry, cw, bh, "Conv\nencoder", BLUE, fontsize=8)
        arrow(ax, in_x + bw, ry + bh / 2, enc_x, ry + bh / 2)
        arrow(ax, enc_x + cw, ry + bh / 2, avg_x, cy)            # encoder -> average
        box(ax, dec_x, ry, cw, bh, "Conv\ndecoder", BLUE, fontsize=8)
        box(ax, out_x, ry, bw, bh, outs[i], GREEN, fontsize=11, bold=True)
        arrow(ax, ds_x + ds_w, cy, dec_x, ry + bh / 2)           # dec shared conv -> decoder
        arrow(ax, dec_x + cw, ry + bh / 2, out_x, ry + bh / 2)

    # center-spine arrows
    arrow(ax, avg_x + avg_w, cy, es_x, cy)      # average -> enc shared conv
    arrow(ax, es_x + es_w, cy, z_x, cy)         # enc shared conv -> z
    arrow(ax, z_x + z_w, cy, ds_x, cy)          # z -> dec shared conv

    # vertical dots between group 2 and group N
    for col_x, w in [(in_x, bw), (enc_x, cw), (dec_x, cw), (out_x, bw)]:
        ax.text(col_x + w / 2, 31, r"$\vdots$", ha="center", va="center", fontsize=15)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
