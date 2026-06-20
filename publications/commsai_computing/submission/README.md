# Communications AI & Computing — submission draft

**Title:** *Unsupervised deep learning for wavelength selection in multi-excitation fluorescence imaging*
**Authors:** Narek Meloyan, Narine Sarvazyan
**Target venue:** Communications AI & Computing (Nature Portfolio) — primary research **Article**
**Status:** compiles to `main.pdf` (38 pp, Nature double-spaced / line-numbered review format)

## Build
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Approach: VERBATIM original text + surgical collagen additions
This draft is built from the **original TPAMI-revised section sources**
(`~/Downloads/paper 2/sections/`, identical to the "(2)" PDF) used **verbatim**.
The original wording, voice (including first-person "we"), figures (the TikZ
pipeline, architecture, and training-curve diagrams), tables, and equations are
preserved. The manuscript was **not** paraphrased.

The only changes are:
1. **Structure → Nature order.** Sections reordered to Introduction → Related Work
   → Results → Discussion → Conclusion → **Methods** (= original *Methodology* +
   *Experimental Setup*, moved to the end per Nature convention). Two-column
   `figure*`/`table*` floats converted to single-column. Added Data/Code
   availability, Author contributions, Competing interests. naturemag references.
2. **Collagen sponge dataset woven in surgically** (the only content additions):
   | Location | Change |
   |---|---|
   | Abstract | +1 sentence on collagen generalization |
   | Introduction | +1 sentence noting two datasets |
   | Experimental Setup (Methods) | + "Collagen Sponge Dataset" subsection + spec table + ROI figure; lichen part given a matching subsubsection heading |
   | Results | + **new subsection "Cross-Dataset Validation: Collagen Sponges"** (envelope + wavelength-importance figures) |
   | Discussion | "Multi-Dataset Generalization" bullet updated (cross-domain now demonstrated, not just proposed) |
   | Conclusion | +1 sentence on collagen result |

Everything else is the original paper, unchanged.

## Lengths (Communications AI & Computing limits)
- **Main text** (Intro+Related+Results+Discussion+Conclusion): **~4,300 words** — under the ~5,000 ceiling ✅
- **Methods** (excluded from limit): ~2,300 words
- **Title:** 10 words, no abbreviations/active verb ✅
- **Abstract:** **198 words** — within the 150–200 cap ✅ (trimmed from the original ~218 and converted to impersonal voice)

## Tone pass (done)
Per instruction, a light, grounded tone pass was applied to the verbatim text:
- **All 14 first-person instances removed** ("we developed" → "a framework was
  developed"; "Our approach" → "The approach"; "To the best of our knowledge" →
  "To the best of current knowledge"; etc.). **Zero `we/our/us` remain.**
- **Factual, data-backed claims kept** — e.g. the "Key Finding: Performance Exceeds
  Baseline" subsection and all "exceeded the baseline" statements stay, because they
  reflect the results.
- **Only mild intensifiers grounded** (the text had no "exceptional/best/life-changing"
  to begin with): "transformative technology" → "established technology";
  "markedly improved" → "improved"; "has transformed" → "has substantially advanced";
  "selection intelligence" → "selectivity" / "a genuine, non-random selection".
- Nothing else was reworded; the original phrasing and substance are intact.

## Verified numbers used for collagen
- 6 excitations (310–400 nm), **158** valid EEM pairs (confirmed canonical by the
  author, matching the reference manuscript), 3 crosslinker classes, 39,970 annotated
  pixels.
- Baseline 79.8%; peak **85.6% at 30 bands (81% reduction, +5.8 pp)**. Source: the
  vetted accuracy-envelope figure. (The 10-classifier comparison is still omitted — its
  per-classifier numbers are inconsistent across runs; see prior note.)

## ⚠️ Before submitting
1. **Collagen band count = 158** (confirmed canonical by the author; the draft uses
   158 throughout, with K=30 → 81% reduction). Note: some local result CSVs report 149
   under a different preprocessing run — if those are ever used for a fresh figure,
   keep the 158 convention.
2. **Related Work** is kept as its own section. Nature usually folds background into
   the Introduction; not required at submission, but the editor may ask. Easy to fold
   later.
3. Confirm author list / affiliations / ORCIDs; add corresponding-author email.
4. AI-use disclosure is in Acknowledgements — adjust to your actual usage.
5. Architecture figure is the **original paper's TikZ** (decoder-side shared conv
   only). The earlier "symmetric" PNG correction is available at
   `../CODASSCA2026_Submission/fig_architecture.png` if you want to swap it in.
6. **Optional, not added:** the reference manuscript frames collagen with a
   ten-classifier evaluation and includes a SOTA comparison table. Both were left out
   here because their numbers are inconsistent across result runs; they can be added
   after one clean re-run.
