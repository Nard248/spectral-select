"""Render the four processed ME-HSI datasets used in the ME-Attention and RL papers.

For each dataset: (a) a false-colour fluorescence image from the 365 nm excitation cube
(R = 600-680 nm, G = 500-590 nm, B = 420-490 nm emission; the 365 nm emission axis is correct
in every processed file), (b) the class mask exactly as used in the experiments, and
(c) the 8-connected "specimen" components (>= 100 px) that the evaluation treats as specimens,
numbered, with their pixel counts.

Usage: python experiments/dataset_audit/render_datasets.py OUT_DIR
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from me_attention.data import (  # noqa: E402
    DATA_ROOT, DATASETS, _class_map_npy, _class_map_png, specimens_from_class_map,
)

TITLES = {
    "lichens1": "Lichens 1  (processed 'Lichens Dataset 1', 1040x925 px)",
    "lichens2": "Lichens 2  (processed 'Lichens_2', 256x348 px, acquired 2026-01-09)",
    "collagen_pepsin": "Collagen Pepsin  (250929 acquisition, 256x348 px)",
    "collagen_acetic": "Collagen Acetic Acid  (250929 acquisition, 256x348 px; dev set)",
}
CLASS_COLOURS = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [255, 225, 25], [240, 50, 230],
                          [245, 130, 48], [70, 240, 240]]) / 255.0


def band_mean(entry: dict, lo: float, hi: float) -> np.ndarray:
    wl = np.asarray(entry["wavelengths"], float)
    sel = (wl >= lo) & (wl <= hi)
    return np.nan_to_num(entry["cube"][..., sel].astype(np.float32), nan=0.0).mean(axis=2)


def stretch(img: np.ndarray, fg: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img[fg], [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)


def render(key: str, out: Path) -> dict:
    cfg = DATASETS[key]
    d = DATA_ROOT / cfg["dir"]
    class_map, names = (_class_map_png if cfg["kind"] == "png" else _class_map_npy)(d)
    spec, spec_cls, dropped_px, dropped_n = specimens_from_class_map(class_map, len(names))
    # the masked files zero/NaN the background; use the unmasked cube for the picture when it exists
    pic = d / "spectra_unmasked.pkl" if (d / "spectra_unmasked.pkl").exists() else d / cfg["pkl"]
    with open(pic, "rb") as f:
        raw = pickle.load(f)
    data = raw["data"]
    e365 = data[365.0] if 365.0 in data else data["365.0"]
    fg = class_map >= 0
    rgb = np.dstack([stretch(band_mean(e365, lo, hi), np.ones_like(fg))
                     for lo, hi in ((600, 680), (500, 590), (420, 490))])
    del raw, data

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("False-colour fluorescence, 365 nm excitation\nR 600-680, G 500-590, B 420-490 nm")
    over = rgb * 0.45
    for c in range(len(names)):
        over[class_map == c] = 0.35 * over[class_map == c] + 0.65 * CLASS_COLOURS[c % len(CLASS_COLOURS)]
    axes[1].imshow(over)
    axes[1].set_title("Class mask used in the experiments")
    handles = [plt.Line2D([], [], ls="", marker="s", ms=10, color=CLASS_COLOURS[c % len(CLASS_COLOURS)],
                          label=f"{c}: {names[c]} ({int((class_map == c).sum()):,} px)") for c in range(len(names))]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=8, ncol=2, frameon=False)
    axes[2].imshow(over)
    comps = []
    for s in range(len(spec_cls)):
        rr, cc = np.nonzero(spec == s)
        comps.append(dict(id=s, cls=int(spec_cls[s]), px=int(rr.size), row=float(rr.mean()), col=float(cc.mean())))
        axes[2].text(cc.mean(), rr.mean(), f"{s}\n{rr.size}", color="white", fontsize=7, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    per_cls = np.bincount(spec_cls, minlength=len(names)).tolist()
    axes[2].set_title(f"'Specimens' = 8-connected components >= 100 px\nper class {per_cls}; "
                      f"dropped {dropped_n} component(s), {dropped_px} px")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(TITLES[key], fontsize=14)
    fig.savefig(out / f"{key}_overview.png", dpi=130)
    plt.imsave(out / f"{key}_fluorescence_rgb.png", rgb)
    plt.close(fig)
    return dict(dataset=key, source=str(Path(cfg["dir"]) / cfg["pkl"]), image_shape=list(class_map.shape),
                class_names=names, pixels_per_class=[int((class_map == c).sum()) for c in range(len(names))],
                specimens_per_class=per_cls, dropped_components=int(dropped_n), dropped_px=int(dropped_px),
                components=comps)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    summary = [render(k, out) for k in ("collagen_pepsin", "collagen_acetic", "lichens2", "lichens1")]
    (out / "render_summary.json").write_text(json.dumps(summary, indent=1))
    for s in summary:
        print(s["dataset"], s["image_shape"], s["specimens_per_class"], s["pixels_per_class"])


if __name__ == "__main__":
    main()
