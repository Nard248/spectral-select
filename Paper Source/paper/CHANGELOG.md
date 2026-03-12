# Paper Revision Changelog — Response to Reviewer (Aram) Feedback

## Feedback Summary and Action Plan

### F1: K-NN placement is confusing
**Feedback**: K-NN appears in methodology (II-E) but not in abstract/conclusion. It appears frequently between main results and is confusing.
**Assessment**: Valid. The KNN validation is woven throughout the paper and distracts from the main unsupervised contribution.
**Action**: Compress Validation Protocol (II-E) from 3 subsections into a single compact paragraph. Remove standalone Baseline section (IV-B), merge baseline context into Wavelength Selection Results (IV-C). Keep abstract/conclusion focused on the framework (not the evaluation tool). Add roadmap paragraph at start of Results.

### F2: Dataset spatial dimensions missing in intro
**Feedback**: 200 million data points mentioned in intro but spatial dimensions (1040x925) only appear on page 5.
**Assessment**: Valid. Reader cannot verify the number until much later.
**Action**: Add spatial dimensions parenthetically in the introduction when the 200M number is mentioned.

### F3: Evaluation metrics defined in multiple places with too much detail
**Feedback**: Metrics described in II-E-3 and again with full descriptions in III-D-1. PCA appears without introduction.
**Assessment**: Valid duplication.
**Action**: Remove Metrics subsection from Methodology (II-E) entirely — fold into a brief sentence. Keep full metric definitions only in Experimental Setup (III-D). Trim III-D for conciseness. Add brief PCA context where first used.

### F4: Excessive citation repetition
**Feedback**: Cohen's Kappa [22] cited in II-E-3, III-D-1, IV-B.
**Assessment**: Valid. Cite at first introduction only.
**Action**: Keep citation at first mention in Methodology. Remove from all subsequent uses.

### F5: Repetitive text (λ=0.5)
**Feedback**: "MMR diversity was fixed at λ = 0.5" appears in ~5 places.
**Assessment**: Valid. Excessive repetition.
**Action**: Define once in Methodology, state value once in Experimental Setup, reference briefly in Results/Discussion without full re-explanation.

### F6: Tables IV and V overlap; per-class info not in tables
**Feedback**: Table IV (baseline) and Table V (main results with baseline row) are redundant. Per-class accuracies in text not backed by a table.
**Assessment**: Valid. Table V already contains the baseline row.
**Action**: Remove Table IV entirely. Fold baseline discussion into the Wavelength Selection Results section with brief prose. Convert per-class bullet list into compact prose.

### F7: Confusing notation (k_1, k_3; K in eq.14 vs KNN)
**Feedback**: Why k_1 and k_3 (no k_2)? What are K-s in equation 14?
**Assessment**: Valid. Naming convention not explained; K overloaded between selected bands and KNN neighbors.
**Action**: Add parenthetical explanation for k_1/k_3 naming. Remove K variable from KNN description (just say "5 neighbors").

### F8: Band count transition 232→192→180 unclear
**Feedback**: 232→192 transition clear, but 180 appears without explanation.
**Assessment**: Valid. 180 is the sweep upper bound, not a data processing step.
**Action**: Add clarifying note that 180 is the sweep maximum (the 192-band baseline is evaluated separately).

### F9: Inconsistent figure title formatting
**Feedback**: Figures 4, 6, 7, 8 have embedded titles in different formats; Figures 3, 5 look better without.
**Assessment**: Valid. Titles are baked into PNG files.
**Action**: NOTE — requires regenerating PNG figures without embedded titles. Flag for figure update. Captions already provide appropriate titles.

### F10: 3072 experiment breakdown unclear
**Feedback**: Cannot see how the number 3072 is derived from the listed parameters.
**Assessment**: Valid. Multiplication not shown explicitly.
**Action**: Add explicit calculation: 2 × 2 × 2 × 2 × 3 = 48 configurations per band count × 64 band counts = 3,072.

### F11: Best 80-band configuration not fully specified
**Feedback**: Want to see exactly which configuration produces the peak 95.2% at 80 bands.
**Assessment**: Valid. Table V only shows "PCA, 3-dim, 80 bands" without perturbation method, magnitude, or normalization.
**Action**: Add full configuration details for key operating points (either in table note or text). TODO: Verify exact parameters with experimental logs.

### F12: General structural complexity
**Feedback**: Too many sections/subsections/sub-subsections and bullet points. Prefer clear data-methodology-results flow with ablation studies separated.
**Assessment**: Valid. Paper has deep nesting that can be hard to follow.
**Action**: Add roadmap paragraph at start of Results. Merge short subsections where possible. Rename "Configuration Parameter Analysis" to "Sensitivity Analysis" to signal its ablation nature. Reduce bullet point usage — convert to prose where possible.

---

## Changes Applied (by file)

### introduction.tex
- [x] C1: Added spatial dimensions "(e.g., $1040 \times 925$ pixels)" when mentioning megapixel image and ~200M data points

### methodology.tex
- [x] C2: Added explanation for k_1, k_3 naming convention (subscripts correspond to convolutional layer indices; averaging step has no learned parameters)
- [x] C3: Compressed Validation Protocol (II-E) from 3 subsections into single flowing subsection
- [x] C4: Removed Metrics subsection; folded into brief sentence referencing Experimental Setup
- [x] C5: Removed K variable from KNN description to avoid notation collision with band count K
- [x] C6: Kept first Cohen's Kappa citation

### experimental_setup.tex
- [x] C7: Removed duplicate λ=0.5 statement from Experimental Configurations (line 152)
- [x] C8: Trimmed Evaluation Metrics section for conciseness; removed Cohen's Kappa citation (not first mention)
- [x] C9: Added explicit 3,072 calculation breakdown
- [x] C10: Added explanation for 180 upper bound in band count sweep
- [x] C11: Removed K variable from KNN Classification Setup (use "5 neighbors" directly)
- [x] C12: Added brief PCA introduction where it first appears

### results.tex
- [x] C13: Removed Table IV (baseline) — baseline info merged into Wavelength Selection Results
- [x] C14: Removed standalone Baseline Classification Performance subsection (IV-B)
- [x] C15: Added baseline context as opening of Wavelength Selection Results
- [x] C16: Converted per-class analysis from bullet list to compact prose
- [x] C17: Removed Cohen's Kappa duplicate citation in results
- [x] C18: Reduced λ=0.5 repetition in MMR Trade-off subsection
- [x] C19: Renamed "Configuration Parameter Analysis" to "Sensitivity Analysis"
- [x] C20: Added roadmap paragraph at start of Results section
- [x] C21: Added full configuration detail note for best 80-band result

### discussion.tex
- [x] C22: Reduced λ=0.5 verbosity — reference established value rather than restating
- [x] C23: Removed duplicate Cohen's Kappa citation if present

### results.tex (additional)
- [x] C24: Fixed pre-existing typo "lear advantage" → "clear advantage"

### methodology.tex (additional)
- [x] C25: Used lowercase $k$ in "$k$-nearest neighbors (KNN)" to distinguish from band-count $K$

### Figures (DONE — regenerated without embedded titles)
- [x] F1: robustness_histogram.png (Figure 8) — regenerated from CSV data, no title
- [x] F2: accuracy_envelope.png (Figure 6) — regenerated from results.csv, no title
- [x] F3: wavelength_heatmap.png (Figure 9) — regenerated from experiment data, no title
- [x] F4: roi_overlay.png (Figure 4) — cropped title from existing render
- [x] F5: classification_192bands.png, classification_80bands.png, classification_9bands.png (Figure 7) — cropped titles from existing renders
- Script: scripts/regenerate_figures_no_titles.py
- Output: Paper Source/paper/figures-updated/

---

## Summary of Impact

| Feedback | Status | Type |
|----------|--------|------|
| F1: K-NN placement | ADDRESSED | Structural: compressed validation, merged baseline into selection results |
| F2: Spatial dimensions | ADDRESSED | Clarity: added (1040×925) to introduction |
| F3: Metrics duplication | ADDRESSED | Redundancy: single definition point, trimmed |
| F4: Citation repetition | ADDRESSED | Redundancy: single citation at first mention |
| F5: λ=0.5 repetition | ADDRESSED | Redundancy: reduced from ~6 to 4 purposeful mentions |
| F6: Table IV/V overlap | ADDRESSED | Structural: Table IV removed, baseline in Table V row |
| F7: Notation confusion | ADDRESSED | Clarity: k_1/k_3 explained, KNN uses lowercase k |
| F8: 232→192→180 | ADDRESSED | Clarity: 180 upper bound explained |
| F9: Figure titles | ADDRESSED | Figures regenerated without titles in figures-updated/ |
| F10: 3072 breakdown | ADDRESSED | Clarity: explicit 48×64=3072 calculation |
| F11: Best config detail | PARTIALLY | Table note expanded; exact perturbation/normalization params need verification from logs |
| F12: Structural complexity | ADDRESSED | Roadmap paragraph, renamed to "Sensitivity Analysis", reduced nesting |

### Net effect on paper:
- Removed ~1 table (Table IV), 3 subsections (from II-E), and ~15 lines of redundant text
- Added ~8 lines of clarifying content (roadmap, calculations, explanations)
- Overall: more focused, less repetitive, clearer navigation
