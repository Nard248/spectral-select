"""Hi-res paper figures for the WHISPERS full paper:
  (1) fig_scene.png  -- regenerated at higher spatial resolution + DPI (crisp sample bands)
  (2) fig_cube4d.png -- a visualization that makes the 4D (excitation x emission x H x W)
                        structure obvious: spatial slices across ex/em, the mean EEM, and
                        two per-pixel excitation-emission matrices.

Run: PYTHONPATH=src:reports python reports/paper_figures_4d.py
Writes into publications/whispers2026/full_paper/figures/.
"""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classification_experiment import feature_matrix
from realistic_benchmark import EXCITATIONS, build_dataset, band_is_discriminative, LIB, DISC, NUIS
from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.forward import render
from spectraforge.scene import Scene


def build_geometric_scene(size=256):
    """A painted example scene: each fluorophore laid down as a distinct shape, so a slice
    taken at that dye's emission peak shows a clean, recognizable structure (not noise)."""
    s = Scene(size, size)
    u = size / 256.0
    S = lambda v: int(round(v * u))
    s.paint_rect(Material("m_NADH", {"NADH": 1.0}), S(22), S(118), S(22), S(118))          # top-left square
    s.paint_circle(Material("m_D1", {"D1": 1.0}), cy=S(64), cx=S(192), radius=S(46))        # top-right disk
    s.paint_rect(Material("m_D2", {"D2": 1.0}), S(120), S(150), S(30), S(228))              # mid horizontal bar
    s.paint_circle(Material("m_D3", {"D3": 1.0}), cy=S(190), cx=S(72), radius=S(46))        # bottom-left disk
    s.paint_rect(Material("m_lipo", {"lipofuscin": 1.0}), S(152), S(234), S(140), S(234))   # bottom-right block
    acq = AcquisitionConfig(excitations=[405.0, 470.0, 488.0, 506.0], em_min=420, em_max=700, em_step=5)
    artifacts = ArtifactConfig(rayleigh_strength=0.3, raman_strength=0.2, second_order=True,
                               photon_scale=14000, read_sigma=0.0012)
    physics = PhysicsConfig(psf_sigma_px=1.2)
    return render(s, LIB, acq, artifacts=artifacts, physics=physics, seed=1)

OUT = pathlib.Path(__file__).resolve().parents[1] / "publications/whispers2026/full_paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300
plt.rcParams.update({"font.size": 9, "axes.grid": False})


def band_index(colmap, ex, em):
    cands = [(j, m) for j, (e, m) in enumerate(colmap) if e == ex]
    return min(cands, key=lambda t: abs(t[1] - em))[0]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("wrote", name, "at", DPI, "dpi")


# ------------------------------------------ fig_scene: single-column variance-vs-info evidence
def fig_scene(size=160):
    p = dict(seed=1, size=size, em_step=5, nuisance_amp=1.2, photon_scale=2000, read_sigma=0.005)
    sp, gt, y, acq = build_dataset(**p)
    X, colmap = feature_matrix(sp)
    disc = np.asarray(band_is_discriminative(colmap), dtype=bool)
    var = X.var(0)
    from sklearn.feature_selection import f_classif
    F = np.nan_to_num(f_classif(X, y)[0])
    corr = np.corrcoef(var, F)[0, 1]
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.scatter(var[~disc], F[~disc], s=12, c="#6f9bc3", alpha=0.55, label="nuisance / scatter bands")
    ax.scatter(var[disc], F[disc], s=16, c="#c0392b", alpha=0.9, label="discriminative bands")
    ax.set_xlabel("per-band variance"); ax.set_ylabel("per-band discriminability")
    ax.set_title(f"variance vs.\\ informativeness  (corr $= {corr:+.2f}$)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_scene.png")


# ---------------------------------------------------------------- fig_cube4d (NEW)
# Dense excitation x emission grids for a smooth EEM landscape (a spectrofluorometer samples
# excitation finely; the acquired cube in panel (a) samples it at only a few excitations).
_EXG = np.linspace(395.0, 560.0, 150)
_EMG = np.linspace(420.0, 700.0, 190)
_EX, _EM = np.meshgrid(_EXG, _EMG)   # shape (emission, excitation)


def _eem_landscape(weights):
    """True EEM of the simulated scene: sum_k conc_k * eps_k * Phi_k * a_k(ex) * e_k(em)."""
    S = np.zeros_like(_EX)
    for name, f in LIB.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        S += (f.extinction * f.quantum_yield * w) * f.excitation(_EX) * f.emission(_EM)
    return S


def _draw_eem(ax, S, title, normalize=True):
    Z = S / S.max() if (normalize and S.max() > 0) else S
    ax.contourf(_EXG, _EMG, Z, levels=16, cmap="turbo")
    ax.contour(_EXG, _EMG, Z, levels=6, colors="k", linewidths=0.3, alpha=0.4)
    ax.set_title(title, fontsize=9)


def fig_cube4d(size=256):
    sp, gt = build_geometric_scene(size)
    X, colmap = feature_matrix(sp)
    R = lambda j: X[:, j].reshape(size, size)
    conc = gt.concentration_maps
    mean_c = {n: float(conc[n].mean()) for n in conc}
    disc_w = {n: mean_c.get(n, 0.0) for n in DISC}
    nuis_w = {n: mean_c.get(n, 0.0) for n in NUIS}

    # (a) spatial slices sampled at each dye's emission peak, where its painted field is lit up
    slices = [(405, 445, "NADH"), (470, 515, "dye D1"),
              (506, 555, "dye D3"), (488, 665, "lipofuscin")]

    fig = plt.figure(figsize=(11.0, 5.0))
    outer = fig.add_gridspec(2, 3, width_ratios=[0.92, 1.85, 1.18], height_ratios=[1, 1],
                             wspace=0.22, hspace=0.34, left=0.05, right=0.99, top=0.86, bottom=0.11)

    mgs = outer[:, 0].subgridspec(2, 2, wspace=0.06, hspace=0.06)
    ax_tl = None
    for idx, (e, emv, dye) in enumerate(slices):
        a = fig.add_subplot(mgs[idx // 2, idx % 2])
        if idx == 0:
            ax_tl = a
        a.imshow(R(band_index(colmap, e, emv)), cmap="viridis",
                 interpolation="bilinear", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        a.text(0.045, 0.955, f"{dye}\nex{e}/em{emv}", transform=a.transAxes, va="top", ha="left",
               color="white", fontsize=7, linespacing=0.95,
               bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.4))
    pos = ax_tl.get_position()
    fig.text(pos.x0, pos.y1 + 0.03, "(a) spatial slices at dye peaks", ha="left", fontsize=9)

    # (b) mean EEM of the whole scene (the hero landscape)
    axb = fig.add_subplot(outer[:, 1])
    _draw_eem(axb, _eem_landscape(mean_c), "(b) mean EEM of the scene")
    axb.set_xlabel("excitation (nm)"); axb.set_ylabel("emission (nm)")

    # (c) EEM decomposed by role: dim discriminative dyes vs bright class-irrelevant nuisances
    axc1 = fig.add_subplot(outer[0, 2])
    _draw_eem(axc1, _eem_landscape(disc_w), "(c) discriminative dyes (dim)")
    axc1.set_xticklabels([]); axc1.set_ylabel("emission (nm)", fontsize=8)
    axc2 = fig.add_subplot(outer[1, 2])
    _draw_eem(axc2, _eem_landscape(nuis_w), "nuisance dyes (bright)")
    axc2.set_xlabel("excitation (nm)"); axc2.set_ylabel("emission (nm)", fontsize=8)

    save(fig, "fig_cube4d.png")


if __name__ == "__main__":
    fig_scene()
    fig_cube4d()
