# CODASSCA 2026 — Short Paper Submission

**Title:** *Dependency-Aware Dimensionality Reduction for Complex Sensor Data: Capturing Conditional Channel Relevance*
**Authors:** Narek Meloyan, Aleksandr Hayrapetyan, Narine Sarvazyan (AUA)
**Target track:** Track 1 — Data Science and Information-Theoretic Approaches (secondary: Track 4 — AI / Neural Networks / Deep Learning)
**Category:** Short paper (3-page limit)

## Files
| File | Purpose |
|---|---|
| `build_codassca_shortpaper.py` | Generator for the DOCX (edit content here, re-run to rebuild) |
| `make_architecture_figure.py` | Renders the generic architecture diagram (`fig_architecture.png`) |
| `fig_architecture.png` | Fig. 1 — de-branded encoder/fusion/decoder diagram |
| `CODASSCA2026_Meloyan_ShortPaper.docx` | The submission draft (~3 pp, IEEE-style two-column) |

## Figures (all de-branded / benchmark-consistent)
- **Fig. 1** Architecture — parallel per-channel-group encoders → average fusion → latent `z` → parallel decoders (generic labels; generated locally).
- **Fig. 2** Conditional-relevance map — reuses the benchmark wavelength-importance heatmap (`MasterThesis_Narek_Meloyan/figures/wavelength_heatmap.png`); axes are excitation/emission wavelength, consistent with the stated fluorescence benchmark.
- **Fig. 3** Accuracy vs. retained channels — reuses `accuracy_envelope.png`; shows above-baseline accuracy and the non-monotonic peak.
- **Table I** Reduction-vs-accuracy operating points (biological + materials sets).

## Framing decision (2026-05-23)

This submission deliberately **generalizes away** from the ME-HSI / perturbation-autoencoder branding used in the TPAMI journal paper and the IASIM 2026 poster, to (a) present a distinct general-method contribution and (b) avoid dual-submission overlap. Hyperspectral fluorescence imaging appears only as a **benchmark/stress test**, not the subject.

The headline angle is **information-theoretic conditional relevance** — chosen because Track 1's chairs are information theorists (Yanling Chen, Han Vinck) and because "a channel matters more next to another" is literally a conditional-mutual-information idea.

### De-branding map (specific → general)
| TPAMI / IASIM term | CODASSCA framing |
|---|---|
| Multi-Excitation Hyperspectral Imaging (ME-HSI) | complex multi-channel sensor data with coupled channel groups |
| Perturbation-based autoencoder | sensitivity-based attribution on a learned joint representation |
| MMR band selection | relevance–redundancy selection (surrogate for conditional MI) |
| Cross-excitation correlation | conditional / dependent channel relevance |
| Lichens dataset | "biological-morphology set (192 channels, 4 classes)" |
| Collagen Sponges dataset | "materials-chemistry set (158 channels, 3 classes)" |

### Novel contribution at the general level
The empty cell in the design space: **unsupervised + conditional + discrete + interpretable** channel selection. Marginal methods (PCA, filters) are unsupervised but marginal; information-theoretic selectors (mRMR/CMIM/JMI) are conditional but supervised and need density estimation.

## ⚠️ Before submitting
1. **IEEE template is mandatory** — CODASSCA states submissions deviating from the official IEEE proceedings template "may be automatically rejected." Pour this content into the official IEEE Word/LaTeX two-column template; this DOCX is a content draft, not the final formatted file.
2. **AI-disclosure is required and is already included** in the Acknowledgements (CODASSCA policy). Adjust the tool name/version to match your actual usage before submitting.
3. **Confirm author affiliations / emails** — placeholders use `@aua.am`; verify Sarvazyan's affiliation (lab vs AUA).
4. **Deadlines:** submission was May 1 → extended May 15, 2026; verify any further extension with organizers. Camera-ready July 20, 2026.
5. **Page limit is 3 pages (hard).** With all 3 figures + table, this draft sits at roughly 3 pages. Exact pagination depends on the official IEEE template's metrics — **verify in the template and trim if it spills over.** Easiest levers if over: shrink Fig. 2 to column width, or move Table I inline with Fig. 3.
