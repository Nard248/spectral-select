#!/usr/bin/env python3
"""
Render an A0-portrait wireframe mockup for the v2 poster proposal.
Outputs:
  Showcase_Poster/POSTER_v2_wireframe.png
  Showcase_Poster/POSTER_v2_wireframe.pdf
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe

OUT_DIR = Path(__file__).resolve().parent.parent / "Showcase_Poster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A0 portrait at 1/10 scale (84.1 x 118.9 cm -> 8.41 x 11.89 inches)
W_IN, H_IN = 8.41, 11.89
fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.invert_yaxis()
ax.axis("off")

# Color palette
C = {
    "title":  "#1a2332",
    "intake": "#56b870",
    "compute":"#3a86c8",
    "output": "#7a4ec8",
    "highlight":"#dff5e0",
    "rule":   "#9aa6b5",
    "fig":    "#fdebd0",
    "table":  "#fffbe7",
    "text":   "#1a2332",
}

def block(x, y, w, h, label, *, fc="#f3f5f9", ec="#1a2332", lw=1.2,
          fontsize=8, weight="normal", align="center"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.008",
                          fc=fc, ec=ec, lw=lw)
    ax.add_patch(rect)
    if align == "center":
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, color=C["text"], weight=weight, wrap=True)
    elif align == "topleft":
        ax.text(x + 0.005, y + 0.012, label, ha="left", va="top",
                fontsize=fontsize, color=C["text"], weight=weight, wrap=True)

# ----------- TITLE BAR (0.00 - 0.10) -----------
block(0.02, 0.01, 0.96, 0.085,
      ("PICKING THE WAVELENGTHS THAT MATTER\n"
       "Autoencoder-Driven Band Selection for 4D Multi-Excitation Hyperspectral Imaging\n"
       "Narek Meloyan · Narine Sarvazyan   |   AUA · GWU · L.A. Orbeli Institute   |   QR · Repo"),
      fc="#0a5dab", ec="#0a5dab", fontsize=8.5, weight="bold")
ax.text(0.5, 0.0625,
        "9-13 selected (excitation, emission) bands recover full-spectrum classification accuracy on biological samples — no labels for selection",
        ha="center", va="center", fontsize=6.0, color="white", style="italic")

# ----------- ROW 1: WHY / PROBLEM / CONTRIBUTION (0.10-0.21) -----------
y0, h0 = 0.105, 0.10
block(0.02, y0, 0.31, h0,
      "WHY\n\n• ME-HSI = 4D cubes (x,y,λex,λem)\n• 200+ band combinations / pixel\n• Most bands redundant or noisy\n• Acquisition slow + correlated\n  features for downstream classifiers",
      fc="#f6f8fc", fontsize=6.5, align="topleft")

block(0.345, y0, 0.31, h0,
      "PROBLEM\n\n• Existing pickers built for 3D HSI\n• Variance/PCA → single bright excitation\n• SPA / MCUVE → adjacent redundant bands\n• None model nonlinear ex-em coupling",
      fc="#fff8ec", fontsize=6.5, align="topleft")

block(0.67, y0, 0.31, h0,
      "CONTRIBUTION  (3 stages)\n\n1. 3D CAE w/ parallel excitation branches\n2. Latent perturbation → influence map\n3. MMR selection (relevance + diversity)\n\nSelf-supervised. KNN validates only.",
      fc="#eaf6ec", fontsize=6.5, align="topleft", weight="bold")

# ----------- ROW 2: METHOD (0.21 - 0.42) -----------
ym, hm = 0.215, 0.20
ax.add_patch(Rectangle((0.02, ym), 0.96, hm, fc="#fafbfd", ec="#9aa6b5", lw=0.6))
ax.text(0.5, ym + 0.012, "METHOD — single horizontal pipeline",
        ha="center", va="center", fontsize=10.5, color=C["title"], weight="bold")

# Pipeline boxes
pipe_y = ym + 0.030
pipe_h = 0.040
boxes = [
    ("ME-HSI cube\n(x,y,λex,λem)",    C["intake"]),
    ("Preprocess\nRayleigh + mask",    "#cfe6f4"),
    ("3D CAE\nparallel branches",      C["compute"]),
    ("Perturb latent\n±15/30/45 % σ",   C["compute"]),
    ("MMR\nλ = 0.5",                   C["compute"]),
    ("K bands\n(λex, λem)",            C["output"]),
]
n = len(boxes)
gap = 0.014
total_w = 0.94
box_w = (total_w - gap*(n-1)) / n
x = 0.03
for label, color in boxes:
    block(x, pipe_y, box_w, pipe_h, label, fc=color, ec="#1a2332", fontsize=7,
          weight="bold")
    if x + box_w + gap < 0.97:
        ax.annotate("", xy=(x + box_w + gap, pipe_y + pipe_h/2),
                    xytext=(x + box_w, pipe_y + pipe_h/2),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#1a2332"))
    x += box_w + gap

# Inset A: CAE arch
inset_y = pipe_y + pipe_h + 0.020
inset_h = 0.105
block(0.025, inset_y, 0.46, inset_h,
      "Inset A — 3D CAE architecture\n\n(parallel encoders → AVG → latent z ∈ R²⁰\n  → shared decoder → parallel decoders)\n\n[redraw in Publisher per spec §4D]",
      fc=C["fig"], fontsize=6.5, align="center")

# Inset B: perturbation
block(0.515, inset_y, 0.46, inset_h,
      "Inset B — Perturbation → Influence\n\nlatent dim d   ●●●●●○○○○○\n   ↓ perturb · decode\n|Δ X̂(ex,em)|   →   Σ over d\n[heatmap]                        [heatmap]",
      fc=C["fig"], fontsize=6.5, align="center")

# Acquisition micro-caption
ax.text(0.5, ym + hm - 0.010,
        "Acquisition + preprocessing condensed: 7-8 excitations × ~24-31 emission bands; dark / Rayleigh / mask. Not the contribution.",
        ha="center", va="center", fontsize=5.5, style="italic", color="#5a6675")

# ----------- ROW 3: RESULTS (0.42 - 0.78) -----------
yr, hr = 0.425, 0.355
ax.add_patch(Rectangle((0.02, yr), 0.96, hr, fc="#ffffff", ec="#9aa6b5", lw=0.6))
ax.text(0.5, yr + 0.012, "RESULTS — two datasets, same pipeline",
        ha="center", va="center", fontsize=10.5, color=C["title"], weight="bold")

# Dataset A: Lichens (left half)
dA_x, dA_w = 0.025, 0.475
ax.text(dA_x + dA_w/2, yr + 0.030,
        "Dataset A · Lichens (TPAMI submission)",
        ha="center", va="center", fontsize=8.5, weight="bold", color="#0a5dab")
fig_y = yr + 0.045
fig_h = 0.085
fig_w = (dA_w - 0.020) / 3
for i, (lbl, must) in enumerate([
    ("sample / labels", False),
    ("ACCURACY\nENVELOPE\n[must-have]", True),
    ("WAVELENGTH\nHEATMAP\n[must-have]", True),
]):
    fc = "#fff5cc" if must else C["fig"]
    block(dA_x + i*(fig_w + 0.005), fig_y, fig_w, fig_h, lbl,
          fc=fc, fontsize=6.2, weight="bold" if must else "normal")
# Lichens body text
block(dA_x, fig_y + fig_h + 0.005, dA_w, 0.040,
      "Lichens Dataset 1 — 4 species, 192 bands. CAE selects 13 (ex,em) pairs "
      "matching baseline (86.1 % vs 85.5 %). Robustness 100th percentile vs 10 000 random.",
      fc="#fffef6", fontsize=5.7, align="topleft")

# Dataset B: Pepsin (right half)
dB_x, dB_w = 0.520, 0.475
ax.text(dB_x + dB_w/2, yr + 0.030,
        "Dataset B · Collagen Pepsin (IASIM 2026)",
        ha="center", va="center", fontsize=8.5, weight="bold", color="#7a4ec8")
for i, (lbl, must) in enumerate([
    ("sample", False),
    ("ACCURACY\nENVELOPE\n[must-have]", True),
    ("WAVELENGTH\nHEATMAP\n[must-have]", True),
]):
    fc = "#fff5cc" if must else C["fig"]
    block(dB_x + i*(fig_w + 0.005), fig_y, fig_w, fig_h, lbl,
          fc=fc, fontsize=6.2, weight="bold" if must else "normal")
block(dB_x, fig_y + fig_h + 0.005, dB_w, 0.040,
      "Same pipeline, no retuning, chemically distinct sample. CAE matches the "
      "158-band KNN baseline at 5 bands; +5.8 pp at 30 bands. Selection lands "
      "on 440-490 nm = collagen autofluorescence.",
      fc="#faf6ff", fontsize=5.7, align="topleft")

# Merged results table
tbl_y = yr + 0.235
tbl_h = 0.115
block(0.025, tbl_y, 0.95, tbl_h,
      ("MERGED RESULTS TABLE  (single artifact spanning both datasets)\n"
       "─────────────────────────────────────────────────────────────────────\n"
       "Configuration         Bands  Reduction   Accuracy   F1(w)    κ      Notes\n"
       "─────────────────────────────────────────────────────────────────────\n"
       "LICHENS\n"
       " Baseline (full)        192      —        0.855     0.859   0.807   KNN-5 reference\n"
       " Selected (best)         13    93.2 %     0.861     0.864   0.814   ★ EXCEEDS baseline\n"
       " Selected                 9    95.3 %     0.818     0.822   0.758\n"
       " Selected                 5    97.4 %     0.812     0.814   0.749\n"
       "─────────────────────────────────────────────────────────────────────\n"
       "COLLAGEN PEPSIN\n"
       " Baseline (full)        158      —        0.798       —     0.696   KNN-5 reference\n"
       " Selected (best)         30    81.0 %     0.856       —       —     ★ +5.8 pp\n"
       " Selected                10    93.7 %     0.817       —       —     ★ EXCEEDS\n"
       " Selected                 5    96.8 %     0.808       —       —     ★ EXCEEDS"),
      fc=C["table"], fontsize=5.4, align="topleft")

# ----------- ROW 4: ROBUSTNESS / CONCLUSIONS (0.78 - 0.93) -----------
yc, hc = 0.785, 0.135
block(0.02, yc, 0.31, hc,
      "ROBUSTNESS\n\n[robustness histogram]\n\nMethod 86.1 % vs random max 57.9 %.\nThe selection is not 'any 13 bands' —\nit's the right 13.",
      fc="#f6f8fc", fontsize=6.5, align="topleft")
block(0.345, yc, 0.31, hc,
      "OBJECT-LEVEL\n\n[per-ROI accuracy overlay]\n\nDifficult-to-classify objects gain up\nto +21 pp with selected bands.",
      fc="#f6f8fc", fontsize=6.5, align="topleft")
block(0.67, yc, 0.31, hc,
      "CONCLUSIONS  /  WHAT'S NEXT\n\n• 90-97 % band reduction, accuracy held\n• Pipeline transfers across samples\n• Selection without supervised labels\n\nNext: blind drop array — pipeline\nrecovering 3 spectral types from cubes alone.",
      fc="#eaf6ec", fontsize=6.5, align="topleft", weight="normal")

# ----------- FOOTER (0.93 - 1.00) -----------
block(0.02, 0.927, 0.96, 0.06,
      ("References:  Bank et al. (2020) Autoencoders · Carbonell & Goldstein (1998) MMR · "
       "Cohen (1960) κ · Meloyan & Sarvazyan (2025) TPAMI under review.\n"
       "Acknowledgments:  EU ERA Chair NAS-SAR award.   "
       "Code: github.com/narekmeloyan/spectral-select   ·   Zenodo DOI 10.5281/zenodo.18640119"),
      fc="#1a2332", ec="#1a2332", fontsize=5.5, align="center")
# Force footer text white
ax.text(0.5, 0.957,
        "References:  Bank et al. (2020) Autoencoders · Carbonell & Goldstein (1998) MMR · "
        "Cohen (1960) κ · Meloyan & Sarvazyan (2025) TPAMI under review.",
        ha="center", va="center", fontsize=5.5, color="white")
ax.text(0.5, 0.975,
        "Acknowledgments: EU ERA Chair NAS-SAR award.    "
        "Code: github.com/narekmeloyan/spectral-select    ·    "
        "Zenodo DOI 10.5281/zenodo.18640119",
        ha="center", va="center", fontsize=5.5, color="white")

# Outer border + cut marks
ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec="#9aa6b5", lw=0.6))

fig.savefig(OUT_DIR / "POSTER_v2_wireframe.png", dpi=240,
            facecolor="white", bbox_inches=None)
fig.savefig(OUT_DIR / "POSTER_v2_wireframe.pdf",
            facecolor="white", bbox_inches=None)
plt.close(fig)
print(f"wrote {OUT_DIR / 'POSTER_v2_wireframe.png'}")
print(f"wrote {OUT_DIR / 'POSTER_v2_wireframe.pdf'}")
