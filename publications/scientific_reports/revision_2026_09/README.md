# Scientific Reports revision — 16 September 2026

Manuscript: "Unsupervised Deep Learning for Wavelength Selection in Multi-Excitation Fluorescence Imaging".
Two reviewers, 4 + 7 points. Route: sentence-level edits following the authors' hand-written responses and the supervisor's response
letter (~/Downloads/Response_NAS.docx); seven references added, none removed; no new experiments; no figure
or results table changed.

## Submission package
`Change_Index_2026-09-16.docx` — point-by-point change locations and the new reference numbers, to paste into
the supervisor's response letter (Response_NAS.docx; its numbers [4]-[32] shift by +1 after Tran 2026 became [4]).

Word versions (`build_docx.py`, pandoc from the LaTeX source; equations as Word math, figures embedded,
IEEE-numbered references): `Revised_Manuscript_2026-09-16.docx`, `Revised_Manuscript_2026-09-16_changes_marked.docx`
(blue underline = added, red strikethrough = deleted, as Word character styles DIFadd / DIFdel) and
`Original_Manuscript_v6.docx` (for Word's Review > Compare, which yields true tracked changes).

- `Revised_Manuscript_2026-09-16.pdf` — compiled from `manuscript/revised/` (17 pp).
- `Revised_Manuscript_2026-09-16_changes_marked.pdf` — latexdiff (additions blue, deletions red).
- `Response_to_Reviewers_2026-09-14 (superseded_2026-09-14/).docx` / `.pdf` — point-by-point response with page / column / line
  for every change (lines counted from the top of the column; floats and captions not counted), and the
  reasons for keeping [1], [2] and the section structure.

## Manuscript sources
- `manuscript/original_v6/` — the submitted LaTeX, untouched.
- `manuscript/revised/` — 21 text edits (V1-V21) applied by `manuscript/apply_edits_2026-09-16.py`
  (= the 09-14 edits + Tran et al. 2026 as reference [4] + the R1.4 'seven sources / two bands at 430 nm' wording);
  `references.bib` +7 entries. `abstract.tex`, `main.tex`, all figures and all results tables unchanged.
- `manuscript/revised_linenumbered/` — internal `lineno` build (identical pagination) used by
  `locate_changes.py` to produce `change_locations.json`; NOT a submission file.
- `manuscript/diff/` — latexdiff source of the change-marked PDF.
- Archived earlier versions: `manuscript/revised_extended_2026-09-13/`, `manuscript/revised_minimal_2026-09-13/`,
  and the planning documents in `superseded_2026-09-13/`.

## Rebuild
1. `cp -R manuscript/original_v6 manuscript/revised && python manuscript/apply_edits_2026-09-14.py`
2. latexdiff (before compiling `revised`, so no .bbl is inlined):
   `latexdiff --flatten --math-markup=0 --graphics-markup=none --config="PICTUREENV=(?:picture|DIFnomarkup|tabular|algorithmic|tikzpicture)[\w\d*@]*" original_v6/main.tex revised/main.tex > diff/main.tex`
3. `latexmk -pdf` in `revised`, `diff`, and `revised_linenumbered` (copy of `revised` + `\usepackage[pagewise,mathlines]{lineno}` and `\linenumbers`; keep its .aux).
4. `python locate_changes.py && python build_response_letter_docx.py`.
