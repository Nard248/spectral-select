# SpectraForge — big review paper (internal, for supervisors & peers)

A comprehensive, story-first write-up of the whole program: **SpectraForge** (the
physics-grounded 4D fluorescence generator — photophysics, forward-model math, the
`variance ≠ informativeness` design, artifacts/nonlinear physics/clutter, regimes, API,
ground truth, FPbase import, GUI) **and** the complete CAE journey it enabled (chance →
competitive debugging story, the significance-tested verdict, the nonlinear tests, the
`selection ≠ classification` insight, the honest boundary, and the one structural
spatial-CAE win).

This is the **fully honest, comprehensive** version. The WHISPERS papers in
`publications/whispers2026/` are the positive-framed, venue-length distillations of it.

## Build
```
pdflatex main.tex && pdflatex main.tex     # -> main.pdf (~11 pp)
```
Figures in `figures/` (from `paper/figures/`, regenerable via `reports/paper_figures.py`).

## Provenance
Consolidates `docs/spectraforge/FINAL-REPORT.md`, docs 01/06/13/14/16/17/18,
`OVERNIGHT-LOG.md`, and `reports/exp_records/metric_suite.csv`. Every number is
significance-tested where a verdict is claimed.
