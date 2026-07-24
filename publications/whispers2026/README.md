# WHISPERS 2026 submission package — SpectraForge

**Venue:** 16th IEEE GRSS Workshop on Hyperspectral Imaging and Signal Processing
(WHISPERS), Glasgow, Scotland, **17–19 November 2026**.
**Submission:** EasyChair. **Deadline (as listed):** **28 July 2026** — verify on the
site for any extension.

This folder holds **both** submission options so we can decide with the supervisor:

| folder | what it is | length | IEEE Xplore |
|---|---|---|---|
| `full_paper/` | Full paper, IEEE double-column (`IEEEtran`) | **5 pages** | **yes** (submitted for inclusion) |
| `abstract/`   | One-page abstract | **1 page** | no (program + on-site) |

Registration note: one registration covers 2 papers per author.

## Framing (both options)
Leads with **SpectraForge as an application/benchmark** for the field (an open,
physics-grounded, controllable, ground-truth-labeled 4D fluorescence generator that
fills the labeled-data gap), then the **honest evaluation methodology**, then the
**CAE case study** as the demonstration: the debugging recovery (chance → 85–97 % of
oracle), the significance-tested nonlinear-reabsorption win, and the spatial-CAE
structural result. Positive, and honest — the benchmark's boundary findings are
presented as *actionable guidance* (when unsupervised band selection helps), not as a
method failure. The full, unvarnished narrative is `publications/spectraforge_review/`.

## Build
```
cd full_paper && pdflatex main.tex && pdflatex main.tex     # -> main.pdf (5 pp)
cd ../abstract && pdflatex main.tex                          # -> main.pdf (1 pp)
```
Figures are in each folder's `figures/` (staged from `paper/figures/`, regenerable via
`reports/paper_figures.py`).

## Open decisions before submitting
1. **Full paper vs abstract** — decide with the supervisor (both are ready).
2. **Template** — currently `IEEEtran` conference (standard IEEE, 2-column). If WHISPERS
   mandates the ICASSP-style `spconf` template, it is a minor reformat.
3. **Authors / affiliations / acknowledgment** — mirror the CODASSCA submission
   (Meloyan, Sarvazyan); confirm the funding acknowledgment text and the AI-use
   disclosure.
4. **Real data** — synthetic-only as it stands; Lichens/Collagen validation is the
   stated next step (data was not on the training machine).
5. **Numbers** — all figures/tables trace to `reports/exp_records/metric_suite.csv` and
   `docs/spectraforge/` (13,14,16,17,18, OVERNIGHT-LOG).
