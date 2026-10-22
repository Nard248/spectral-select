"""True-colour (white-light reflectance) rendering of the Lichens 2 palette from Data/Raw/Lichens_2/Reflectance/White.im3.

im3 layout (verified against the raw cube stored in the processed Lichens 2 pickle): uint16, band-interleaved by pixel,
256 rows x 348 cols x 31 bands (420-720 nm, 10 nm; wavelengths read from the file header), ending 16 bytes before EOF.
Usage: python experiments/dataset_audit/lichens2_whitelight_rgb.py OUT_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
H, W, B = 256, 348, 31
WL = 420 + 10 * np.arange(B)


def read_im3(p: Path) -> np.ndarray:
    b = p.read_bytes()
    n = H * W * B * 2
    return np.frombuffer(b[len(b) - 16 - n:len(b) - 16], dtype="<u2").reshape(H, W, B).astype(np.float32)


def to_rgb(cube: np.ndarray) -> np.ndarray:
    ch = [cube[..., (WL >= lo) & (WL <= hi)].mean(2) for lo, hi in ((590, 660), (500, 570), (430, 490))]
    rgb = np.dstack(ch)
    lo, hi = np.percentile(rgb, [1, 99.5])
    return np.clip((rgb - lo) / (hi - lo), 0, 1) ** (1 / 1.8)


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rgb = to_rgb(read_im3(REPO / "Data/Raw/Lichens_2/Reflectance/White.im3"))
plt.imsave(out / "lichens2_whitelight_rgb.png", rgb)
cm = np.load(REPO / "Data/processed/Lichens_2/lichens_2_class_mask.npy")
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
ax[0].imshow(rgb); ax[0].set_title("Lichens 2 (acquired 2026-01-09): white-light image from raw White.im3", fontsize=10)
cols = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [255, 225, 25], [240, 50, 230]]) / 255
ov = rgb.copy()
for c in range(5):
    ov[cm == c] = 0.45 * ov[cm == c] + 0.55 * cols[c]
ax[1].imshow(ov); ax[1].set_title("Annotated classes used as Lichens 2 labels (class names not recorded)", fontsize=10)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.savefig(out / "lichens2_whitelight_with_classes.png", dpi=130)
