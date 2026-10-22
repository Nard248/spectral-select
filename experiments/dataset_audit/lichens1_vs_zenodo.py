"""Show where 'Lichens 1' sits inside the Zenodo 10.5281/zenodo.18640119 palette (250707) and which species each class is.

Registration: our class mask (1040x925) matches the Zenodo Mask_Manual.png (1024x1392) at column offset 466 and
row offset -10 (our row r == Zenodo mask row r-10); every pixel of each of our four classes falls on a single
Zenodo mask value (50, 100, 150, 200).
Usage: python experiments/dataset_audit/lichens1_vs_zenodo.py ZENODO_DIR OUT_PNG
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
from PIL import Image

zdir, out = Path(sys.argv[1]), Path(sys.argv[2])
rgb = np.array(Image.open(zdir / "RGBRendering_WhiteLight.png").convert("RGB"))
zm = np.array(Image.open(zdir / "Mask_Manual.png"))
# the mask has 1024 rows; it aligns with HSI rows 10..1033 (registration against our class mask)
zm = np.pad(zm, ((10, rgb.shape[0] - zm.shape[0] - 10), (0, 0)))
names = {50: "R. sinensis\nlower surface", 100: "R. sinensis\nupper surface", 150: "F. soredica", 200: "R. fraxinea"}
cols = {50: "#e6194b", 100: "#3cb44b", 150: "#0082c8", 200: "#ffe119"}
DX = 466
fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)
ax[0].imshow(rgb)
ax[0].add_patch(mp.Rectangle((DX, 0), 925, 1039, fill=False, ec="cyan", lw=2.5, ls="--"))
ax[0].text(DX + 10, 30, "region used as 'Lichens 1' (x >= 466 px)", color="cyan", fontsize=10, va="top")
ax[0].set_title("Zenodo 18640119 (250707 palette), white-light RGB, 1392x1040 px", fontsize=11)
ov = np.zeros(zm.shape + (4,))
for v, c in cols.items():
    ov[zm == v] = matplotlib.colors.to_rgba(c, 0.85)
    rr, cc = np.nonzero(zm == v)
ax[1].imshow(rgb, alpha=0.35)
ax[1].imshow(ov)
ax[1].add_patch(mp.Rectangle((DX, 0), 925, 1039, fill=False, ec="black", lw=2.0, ls="--"))
ax[1].set_title("Zenodo Mask_Manual.png: 6 columns x 4 rows; Lichens 1 = columns 3-6", fontsize=11)
handles = [mp.Patch(color=cols[v], label=f"mask {v}: {names[v].replace(chr(10), ' ')}") for v in cols]
ax[1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9, frameon=False)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.savefig(out, dpi=120)
