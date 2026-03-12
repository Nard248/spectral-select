# Paper Revision Change Log

**Paper:** Deep Learning for Dimensionality Reduction in Multi-Excitation Hyperspectral Imaging
**Revision Date:** February 2026
**Feedback Sources:** NAS figures PDF, NAS Word document (yellow/turquoise/red markups), Gemini suggestions

---

## Legend
- **TEXT** = Wording change (yellow highlight in Word doc)
- **DELETE** = Removed content (red strikethrough in Word doc)
- **FIGURE** = Figure regenerated or TikZ redrawn
- **ADD** = New content added per reviewer request
- **REWORD** = Phrasing improved (turquoise in Word doc)
- **REDUNDANCY** = Trimmed per Gemini/reviewer redundancy concerns

---

| # | Section | File:Line | Change | Reason | Type |
|---|---------|-----------|--------|--------|------|
| 1 | Abstract | abstract.tex:2 | "Traditional" -> "Conventional" | Reviewer request (yellow) | TEXT |
| 2 | Abstract | abstract.tex:8 | Added "which corresponds to only 9 out of 192" after 9-band mention | Reviewer request (yellow) -- clarifies the band count | ADD |
| 3 | Abstract | abstract.tex:10-11 | Removed "Multi-excitation hyperspectral imaging," and "dimensionality reduction," from Index Terms; added "spectral" before "band selection" | Reviewer request (strikethrough + yellow) | TEXT |
| 4 | Introduction | introduction.tex:7 | Added "based on reflected light spectrum" to HSI description | Reviewer request (yellow) | ADD |
| 5 | Introduction | introduction.tex:17-21 | Rewrote benefits as numbered list: (i) sensor design, (ii) real-time processing, (iii) system deployment | Reviewer request (yellow) | TEXT |
| 6 | Introduction | introduction.tex:18 | Added "The developed methods work albeit with notable limitations." | Reviewer request (yellow) | ADD |
| 7 | Introduction | introduction.tex:20 | Fixed "ME-HIS" -> "ME-HSI datasets" | Typo fix (yellow) | TEXT |
| 8 | Introduction | introduction.tex:36-39 | Trimmed "This paper presents..." paragraph by ~50%, removed repetition of abstract | Gemini + reviewer redundancy | REDUNDANCY |
| 9 | Methodology | methodology.tex:83-87 | Updated Rayleigh section to mention BOTH 1st and 2nd order scattering removal | Reviewer request -- figure showed only 1st order | TEXT |
| 10 | Methodology | methodology.tex:89-118 | Replaced TikZ Rayleigh figure: Y-axis 310-430nm, added 2nd order cutoff band, larger fonts, proper valid region | Critical figure fix from NAS | FIGURE |
| 11 | Methodology | methodology.tex:116 | Updated caption to describe both 1st and 2nd order cutoffs | Follows from figure fix | TEXT |
| 12 | Experimental Setup | experimental_setup.tex:36-38 | Replaced generic instrument description with specific names: WheeLED (Mightex), Nuance FX (PerkinElmer) | Reviewer request (yellow) | TEXT |
| 13 | Experimental Setup | experimental_setup.tex:40 | Added "lichen obtained from the Takhtajyan Institute of Botany" | Reviewer request (yellow) | ADD |
| 14 | Experimental Setup | experimental_setup.tex:42-47 | Class labels changed: Class 0->1, Class 1->2, Class 2->3, Class 3->4 | Reviewer request (yellow) - 0-indexed confusing | TEXT |
| 15 | Experimental Setup | experimental_setup.tex:49 | DELETED "The class distribution was relatively balanced..." | Reviewer request (red) | DELETE |
| 16 | Experimental Setup | experimental_setup.tex:51 | DELETED "This dataset provided the primary evaluation..." | Reviewer request (red strikethrough) | DELETE |
| 17 | Experimental Setup | experimental_setup.tex:69 | Fixed "perregion" -> "per-region" | Typo fix (yellow) | TEXT |
| 18 | Experimental Setup | experimental_setup.tex | Added sentence explaining WHY 192 bands (not 234) | Reviewer question (turquoise) | ADD |
| 19 | Results | results.tex:97-102 | Changed Class 0->1, Class 1->2, Class 2->3, Class 3->4 in per-class analysis | Consistent class relabeling | TEXT |
| 20 | Results | results.tex:104 | Updated confusion description with new class numbers | Consistent class relabeling | TEXT |
| 21 | Results | results.tex:178,198-202 | Updated figure labels and captions from Class 0-3 to Class 1-4 | Consistent class relabeling | TEXT |
| 22 | Results | results.tex:300-307 | DELETED entire "Class-Level Patterns" subsection | Reviewer: "will change with each sample set" (turquoise "can be removed") | DELETE |
| 23 | Results | results.tex:224 | Updated learned selection accuracy from 0.8606 -> 0.9016 | Master run best 13-band from PCA config | TEXT |
| 24 | Results | results.tex:240 | Updated caption: "86.1%" -> "90.2%" | Follows from updated robustness data | TEXT |
| 25 | Results | results.tex:244-249 | Updated robustness comparison text with new numbers | Follows from updated data | TEXT |
| 26 | Results | results.tex:253-257 | Updated z-score: (0.9016 - 0.4606)/0.0358 = 12.3 sigma | Follows from updated data | TEXT |
| 27 | Results | results.tex:163-174 | Trimmed "Key Observations" bullets that duplicate Table V data | Gemini redundancy reduction | REDUNDANCY |
| 28 | Discussion | discussion.tex:51 | Updated robustness reference from 86.1% -> 90.2% | Follows from updated data | TEXT |
| 29 | Discussion | discussion.tex:55 | Updated "100th percentile" text | Follows from updated data | TEXT |
| 30 | Discussion | discussion.tex:69 | "particularly harmful" -> "detrimental" | Reviewer reword request (turquoise) | REWORD |
| 31 | Discussion | discussion.tex:70 | Removed "problematic"; "disproportionately" -> "significantly" | Reviewer request (red + turquoise) | REWORD |
| 32 | Discussion | discussion.tex:104 | "consumed three slots in the budget" -> more formal wording | Reviewer: "too informal" (turquoise) | REWORD |
| 33 | Discussion | discussion.tex:85-94 | Condensed PCA vs Variance section | Gemini: partially duplicates Results | REDUNDANCY |
| 34 | Conclusion | conclusion.tex:10 | "carefully" -> "strategically" | Reviewer reword (turquoise) | REWORD |
| 35 | Conclusion | conclusion.tex:14 | Updated robustness text with new percentile/accuracy data | Follows from updated data | TEXT |
| 36 | Conclusion | conclusion.tex:36 | DELETED "by over 7 percentage points while reducing data by 58%---or to exceed baseline with 95% data reduction---" | Reviewer request (red strikethrough) | DELETE |
| 37 | main.tex | main.tex:91 | Updated Acknowledgment: added grant "101087403", added "Dr. Arsen Gasparyan from the Takhtajyan Institute of Botany, NAS Armenia for providing lichen samples" | Reviewer request (yellow) | ADD |
| 38 | Figures | accuracy_envelope.png | Increased font sizes, tightened Y-axis, added random chance baseline at 0.25, DPI=300 | NAS figure feedback | FIGURE |
| 39 | Figures | wavelength_heatmap.png | Flipped Y-axis (310 at bottom), increased font sizes, DPI=300 | NAS figure feedback | FIGURE |
| 40 | Figures | classification maps | Class labels 0-3 -> 1-4, increased font sizes, DPI=300 | NAS figure feedback | FIGURE |
| 41 | main.tex | main.tex:72-73 | Removed "Multi-excitation hyperspectral imaging," and "dimensionality reduction," from IEEEkeywords | Consistent with Index Terms change | TEXT |

---

## Items NOT Changed (with explanations for Zoom)

| # | Section | Item | Explanation |
|---|---------|------|-------------|
| A | Methodology:80 | tref, Pref reference values | These are arbitrary reference values for normalization consistency. They cancel out when comparing pixels within same acquisition. For our dataset with consistent parameters, we use global min-max instead. |
| B | Heatmap x-axis | "Intermediate" emission values (e.g. 615nm) | These ARE correct -- Ex 415nm starts emission at 455nm due to Rayleigh cutoff. The "unusual" values come from different excitation starting points. Verbal explanation prepared. |
| C | Results | 100th percentile among 10,000 | This means our learned selection outperformed ALL 10,000 random combinations. It's the empirical percentile from the random sampling distribution. |
| D | Author list | Alexandr addition | Flagged for user decision -- needs discussion during Zoom |
