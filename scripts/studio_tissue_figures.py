"""Publication figures for the v1.2 tissue optics: fig20_chromophores.png, fig21_tissue_optics.png."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectraforge.studio import optics, presets_tissue
from spectraforge.studio import tissue_optics as to
from spectraforge.studio.presets import builtin_fluorophores

OUT = Path(__file__).resolve().parents[1] / "docs/studio/figures"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 100, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans"})


def fig20() -> None:
    nm = np.arange(250.0, 1000.0, 1.0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    # (a) extinction coefficients, molar
    a.semilogy(nm, to.hemoglobin_extinction(nm, True), c="#c1121f", lw=1.5, label="HbO$_2$ (Prahl)")
    a.semilogy(nm, to.hemoglobin_extinction(nm, False), c="#1d4e89", lw=1.5, label="Hb (Prahl)")
    nb = np.arange(250.0, 540.0)
    a.semilogy(nb, np.maximum(to.bilirubin_extinction(nb), 1), c="#e09f3e", lw=1.2, label="bilirubin (PhotochemCAD)")
    nc = np.arange(250.0, 560.0)   # beyond ~560 nm the beta-carotene spectrum is baseline noise
    a.semilogy(nc, np.maximum(to.beta_carotene_extinction(nc), 1), c="#2a9d8f", lw=1.2,
               label=r"$\beta$-carotene (PhotochemCAD)")
    a.set(xlabel="Wavelength (nm)", ylabel=r"Molar extinction $\varepsilon$ (cm$^{-1}$ M$^{-1}$)",
          xlim=(250, 1000), ylim=(1e2, 1e6), title="(a) Pigment extinction")
    a.legend(frameon=False, fontsize=8, loc="upper right")
    a.grid(alpha=0.25, which="both", lw=0.4)
    # (b) absorption coefficients of the bulk chromophores at physiological content
    b.semilogy(nm, to.whole_blood_mua_per_mm(nm, 1.0) * 0.01, c="#c1121f", lw=1.5, label="1 % blood, S = 1")
    b.semilogy(nm, to.whole_blood_mua_per_mm(nm, 0.0) * 0.01, c="#1d4e89", lw=1.5, label="1 % blood, S = 0")
    b.semilogy(nm, to.chromophore_mua_per_mm("water", nm, 0.65), c="#4895ef", lw=1.5, label="65 % water")
    b.semilogy(nm, to.chromophore_mua_per_mm("lipid", nm, 0.8), c="#bc6c25", lw=1.5, label="80 % lipid")
    b.semilogy(nm, to.chromophore_mua_per_mm("melanosome", nm, 0.025), c="#4a4e69", lw=1.5, label="2.5 % melanosomes")
    b.set(xlabel="Wavelength (nm)", ylabel=r"Absorption coefficient $\mu_a$ (mm$^{-1}$)", xlim=(250, 1000),
          ylim=(1e-5, 1e2), title="(b) Contribution to tissue $\\mu_a$ (Jacques 2013, eq. 1)")
    b.legend(frameon=False, fontsize=8, loc="upper right")
    b.grid(alpha=0.25, which="both", lw=0.4)
    fig.savefig(OUT / "fig20_chromophores.png")
    plt.close(fig)


PAPER_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
            "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6, "savefig.dpi": 300}


def fig21() -> None:
    with plt.rc_context(PAPER_RC):
        _fig21()


def _fig21() -> None:
    FL = {f.name: f for f in builtin_fluorophores()}
    nm = np.arange(380.0, 900.0, 2.0)
    mats = presets_tissue.tissue_materials()
    cmap = plt.get_cmap("tab20")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.9), constrained_layout=True)
    for i, m in enumerate(mats):
        o = optics.material_optics(m, FL, nm)
        c = cmap(i % 20)
        a.semilogy(nm, o.mua_per_mm, c=c, lw=1.0, label=m.name.replace("Epidermis (","Epidermis, ").replace(" skin, ", " ").split(" (")[0].rstrip(")"))
        if not o.clear:
            b.semilogy(nm, o.mus_per_mm * (1 - o.g), c=c, lw=1.0)
    a.set(xlabel="Wavelength (nm)", ylabel=r"$\mu_a$ (mm$^{-1}$)", xlim=(380, 900), title="(a) Absorption of the tissue library")
    b.set(xlabel="Wavelength (nm)", ylabel=r"$\mu_s'$ (mm$^{-1}$)", xlim=(380, 900), title="(b) Reduced scattering")
    for ax in (a, b):
        ax.grid(alpha=0.25, which="both", lw=0.4)
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    fig.savefig(OUT / "fig21_tissue_optics.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig20()
    fig21()
    print("wrote", OUT)
