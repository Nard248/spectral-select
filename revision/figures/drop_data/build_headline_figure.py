"""Drop Data headline figure — Panels A and B.

Panel A — per-type EEM heatmaps with selected-band markers
    3 small heatmaps (one per Ward(full-217) drop type, computed on the
    full_cr variant). Each is 7 x 31, mean intensity over drops of that
    type, Rayleigh-invalid cells grayed. Overlaid: 5 markers at the
    K=5 selected (ex, em) positions.

Panel B - per-excitation emission slices
    7 subplots (one per ex), x = em, y = ROI-mean intensity. One
    curve per drop (16 curves), colored by Ward(full) type. At
    excitations where the AE selected a band, the chosen em is marked
    with a vertical line.

Configuration (canonical for Drop Data per the conversation history):
    - preprocessing variant: full_cr (ruler crop @ row 175, full pipeline)
    - normalization: max_per_excitation
    - K = 5 selections = ['325/530','365/490','400/490','415/490','385/470']

Reads only the safe `.npz` cache (built once by the inline pkl->npz
conversion). Writes:
    panel_A_eem_per_type.{png,pdf}
    panel_B_emission_slices.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
CUBE_NPZ = Path(__file__).resolve().parent / "full_cr_cube.npz"
DROP_LABELS_PATH = ROOT / "Data" / "processed" / "Drop Data Cropped" / "drop_labels.npy"
DROP_TYPES_PATH = ROOT / "results" / "Drop_Data_Cropped_Sweep" / "full_cr" / "drop_types.npy"
OUT_DIR = Path(__file__).resolve().parent

SELECTED_PAIRS = [(325, 530), (365, 490), (400, 490), (415, 490), (385, 470)]

EX_GRID = [310, 325, 340, 365, 385, 400, 415]
EM_GRID = list(range(420, 721, 10))

TYPE_LABELS = {0: "Bright (n=3)", 1: "Moderate (n=5)", 2: "Baseline (n=8)"}
TYPE_COLORS = {0: "#C8102E", 1: "#E89B23", 2: "#3E7CB1"}


def rayleigh_valid_mask(ex_grid=EX_GRID, em_grid=EM_GRID):
    """1st order em < ex+40, 2nd order |em - 2*ex| < 40 -> invalid."""
    valid = np.ones((len(ex_grid), len(em_grid)), dtype=bool)
    for i, ex in enumerate(ex_grid):
        for j, em in enumerate(em_grid):
            if em < ex + 40 or abs(em - 2 * ex) < 40:
                valid[i, j] = False
    return valid


def load_cubes_from_npz():
    """Load cubes from npz cache. Returns dict {ex_nm: cube_HxWx31_padded}."""
    cubes = {}
    with np.load(CUBE_NPZ) as z:
        ex_grid = z["ex_grid"].tolist()
        for ex in ex_grid:
            cube_raw = z[f"cube_{ex}"]
            wls = z[f"wl_{ex}"].tolist()
            H, W, _ = cube_raw.shape
            full = np.full((H, W, len(EM_GRID)), np.nan, dtype=cube_raw.dtype)
            for k, em in enumerate(wls):
                j = EM_GRID.index(int(em))
                full[:, :, j] = cube_raw[:, :, k]
            cubes[ex] = full
    return cubes


def per_drop_eem(cubes, drop_labels, n_drops=16):
    """(n_drops, n_ex, n_em) per-drop ROI-mean intensity."""
    eem = np.full((n_drops, len(EX_GRID), len(EM_GRID)), np.nan, dtype=np.float32)
    for d in range(1, n_drops + 1):
        m = drop_labels == d
        if not m.any():
            continue
        for i, ex in enumerate(EX_GRID):
            cube = cubes[ex]
            # Vectorized mean over masked pixels per band
            slab = cube[m]  # (n_pix, n_em)
            with np.errstate(all="ignore"):
                eem[d - 1, i, :] = np.nanmean(slab, axis=0)
    return eem


def plot_panel_A(eem_per_drop, drop_types, out_stem):
    valid = rayleigh_valid_mask()
    types = sorted(set(drop_types.tolist()))
    n_types = len(types)

    type_eems = np.stack([
        np.nanmean(eem_per_drop[drop_types == t], axis=0) for t in types
    ])
    vmin, vmax = np.nanmin(type_eems), np.nanmax(type_eems)

    fig, axes = plt.subplots(1, n_types, figsize=(3.4 * n_types + 0.4, 3.6))
    if n_types == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#D3D3D3")

    for ax_i, (ax, t, eem_t) in enumerate(zip(axes, types, type_eems)):
        masked = np.ma.masked_array(eem_t, mask=~valid)
        im = ax.pcolormesh(
            np.arange(len(EM_GRID) + 1) - 0.5,
            np.arange(len(EX_GRID) + 1) - 0.5,
            masked, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
        )
        for (sex, sem) in SELECTED_PAIRS:
            xi = EM_GRID.index(sem)
            yi = EX_GRID.index(sex)
            ax.scatter(xi, yi, s=160, facecolor="none", edgecolor="white",
                       linewidth=2.0, zorder=4)
            ax.scatter(xi, yi, s=42, color="white", marker="x",
                       linewidth=2.0, zorder=5)

        ax.set_xticks(range(0, len(EM_GRID), 3))
        ax.set_xticklabels([str(EM_GRID[k]) for k in range(0, len(EM_GRID), 3)], fontsize=8)
        ax.set_yticks(range(len(EX_GRID)))
        ax.set_yticklabels([str(e) for e in EX_GRID], fontsize=9)
        ax.set_xlabel("Emission lambda (nm)", fontsize=10)
        if ax_i == 0:
            ax.set_ylabel("Excitation lambda (nm)", fontsize=10)
        ax.set_title(TYPE_LABELS[t], fontsize=11, color=TYPE_COLORS[t], pad=6)
        ax.invert_yaxis()

    cax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean intensity (a.u.)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Excitation-Emission Matrix per drop type · K=5 selected (ex, em) shown as circles",
        fontsize=11, y=0.97,
    )
    fig.subplots_adjust(left=0.06, right=0.90, top=0.86, bottom=0.13, wspace=0.18)

    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"  wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def plot_panel_B(eem_per_drop, drop_types, out_stem):
    valid = rayleigh_valid_mask()
    types = sorted(set(drop_types.tolist()))
    selected_by_ex = {ex: None for ex in EX_GRID}
    for (sex, sem) in SELECTED_PAIRS:
        selected_by_ex[sex] = sem

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    axes = axes.flatten()

    em_arr = np.array(EM_GRID)

    for i, ex in enumerate(EX_GRID):
        ax = axes[i]
        valid_row = valid[i]

        # Individual drops as faint lines
        for d in range(eem_per_drop.shape[0]):
            t = int(drop_types[d])
            spec = eem_per_drop[d, i, :]
            spec_v = np.where(valid_row, spec, np.nan)
            ax.plot(em_arr, spec_v, color=TYPE_COLORS[t], alpha=0.40, linewidth=1.0)

        # Per-type means as bold
        for t in types:
            members = np.where(drop_types == t)[0]
            mean_spec = np.nanmean(eem_per_drop[members, i, :], axis=0)
            ax.plot(em_arr, np.where(valid_row, mean_spec, np.nan),
                    color=TYPE_COLORS[t], linewidth=2.4)

        # Mark selected emission band (label rendered inside the plot to
        # avoid overlap with the subplot title)
        sem = selected_by_ex[ex]
        if sem is not None and valid_row[EM_GRID.index(sem)]:
            ax.axvline(sem, color="#2C2C2C", linewidth=1.0, linestyle="--", alpha=0.7)
            # Label inside the upper-right corner of the panel
            ax.text(sem + 6, 0.78, f"selected: {sem} nm",
                    ha="left", va="center",
                    fontsize=8.5, color="#2C2C2C", weight="bold",
                    transform=ax.get_xaxis_transform())

        title = f"lambda_ex = {ex} nm" + ("  (selected)" if sem is not None else "  (silent)")
        title_color = "#2C2C2C" if sem is not None else "#888888"
        ax.set_title(title, fontsize=10, color=title_color)
        ax.set_xlabel("lambda_em (nm)", fontsize=9)
        if i % 4 == 0:
            ax.set_ylabel("ROI-mean intensity", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(415, 725)
        ax.grid(True, alpha=0.25, linestyle=":")

    axes[7].axis("off")
    handles = [plt.Line2D([0], [0], color=TYPE_COLORS[t], linewidth=2.4, label=TYPE_LABELS[t])
               for t in types]
    handles.append(plt.Line2D([0], [0], color="#2C2C2C", linestyle="--", linewidth=1.5,
                              label="AE-selected lambda_em"))
    axes[7].legend(handles=handles, loc="center", fontsize=11, frameon=False,
                   title="Drop type", title_fontsize=12)

    fig.suptitle(
        "Per-excitation emission spectra · curves separate at the bands the AE selected",
        fontsize=12, y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.09,
                        wspace=0.10, hspace=0.40)

    for ext in ("png", "pdf"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"  wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def main():
    print("Loading cubes from npz cache...")
    cubes = load_cubes_from_npz()
    print(f"  {len(cubes)} excitations loaded")

    print("Loading drop labels & types...")
    drop_labels = np.load(DROP_LABELS_PATH)
    drop_types = np.load(DROP_TYPES_PATH)
    n_drops = int(drop_labels.max())
    print(f"  n_drops={n_drops}; types={np.bincount(drop_types).tolist()}")

    print("Computing per-drop EEMs...")
    eem_per_drop = per_drop_eem(cubes, drop_labels, n_drops=n_drops)
    print(f"  EEM tensor shape: {eem_per_drop.shape}")

    print("Plotting Panel A...")
    plot_panel_A(eem_per_drop, drop_types, OUT_DIR / "panel_A_eem_per_type")

    print("Plotting Panel B...")
    plot_panel_B(eem_per_drop, drop_types, OUT_DIR / "panel_B_emission_slices")

    print("Done.")


if __name__ == "__main__":
    main()
