"""Build the point-by-point Response to Reviewers (Word) for the revision of 2026-09-14.

Every "Changes in the manuscript" entry carries page / column / line numbers from change_locations.json
(locate_changes.py, run on a line-numbered build with identical pagination). Lines are counted from the
top of the column; figures, tables and captions are not counted. Output: Response_to_Reviewers.docx.
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
LOCJ = HERE / "change_locations.json"
OUT = HERE / "Response_to_Reviewers.docx"

NAVY = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x55, 0x55, 0x55)
FONT = "Calibri"

LOC = json.load(open(LOCJ, encoding="utf-8")) if LOCJ.exists() else {}
REFS = LOC.get("_refs", {})
COLNAME = {"left": "left column", "right": "right column", "full": "full width"}


def loc(key: str) -> str:
    d = LOC.get(key)
    if not d or not d.get("found"):
        return "location to be inserted"
    page, col = d["page"], COLNAME.get(d["col"], d["col"])
    if d["kind"] == "float":
        return f"p. {page}, {col}, {d['label']}"
    s, e = d.get("start"), d.get("end")
    if s is None:
        return f"p. {page}, {col}"
    if e is None:
        return f"p. {page}, {col}, line {s}"
    if d.get("end_page") == page and d.get("end_col") == d["col"]:
        return f"p. {page}, {col}, lines {s}-{e}"
    return f"p. {page}, {col}, line {s} to p. {d['end_page']}, {COLNAME.get(d['end_col'], d['end_col'])}, line {e}"


def refnum(key: str) -> str:
    n = REFS.get(key)
    return f"[{n}]" if n else "[-]"


# ----------------------------------------------------------------------------- helpers
def _shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def _run(p, text, *, size=11, bold=False, italic=False, color=None):
    r = p.add_run(text)
    r.font.name = FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if color is not None:
        r.font.color.rgb = color
    return r


def H(doc, text, level=1):
    p = doc.add_heading(level=level)
    r = p.add_run(text)
    r.font.name = FONT
    r.font.color.rgb = NAVY
    r.font.size = Pt({1: 15, 2: 12.5, 3: 11.5}[level])
    p.paragraph_format.space_before = Pt({1: 16, 2: 12, 3: 10}[level])
    p.paragraph_format.space_after = Pt(4)
    return p


def P(doc, text="", *, lead=None, size=11, italic=False, color=None, after=6, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if lead:
        _run(p, lead + " ", size=size, bold=True)
    if text:
        _run(p, text, size=size, italic=italic, color=color)
    return p


def COMMENT(doc, text):
    t = doc.add_table(rows=1, cols=1)
    c = t.cell(0, 0)
    _shade(c, "F2F2F2")
    c.text = ""
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    _run(p, "Reviewer comment. ", size=10.5, bold=True, color=GREY)
    _run(p, text, size=10.5, italic=True, color=GREY)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def CHANGES(doc, items):
    P(doc, "", lead="Changes in the manuscript.", after=2)
    for text, key in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        _run(p, text, size=10.5)
        if key:
            _run(p, f" [{loc(key)}]", size=10.5, color=NAVY)


def T(doc, headers, rows, *, widths=None, size=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        _shade(c, "1F4E79")
        c.text = ""
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        _run(p, h, size=size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            c = cells[i]
            if ri % 2 == 1:
                _shade(c, "F7F9FC")
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            _run(p, str(val), size=size)
    if not widths:
        widths = [6.5 / len(headers)] * len(headers)
    t.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ----------------------------------------------------------------------------- document
def build():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(0.9)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _run(p, "Response to Reviewers", size=18, bold=True, color=NAVY)
    P(doc, "Scientific Reports — manuscript [ID to be inserted] — revision of 14 September 2026", align="center", size=11, color=GREY, after=2)
    P(doc, "“Unsupervised Deep Learning for Wavelength Selection in Multi-Excitation Fluorescence Imaging”",
      align="center", size=11.5, italic=True, after=2)
    P(doc, "Narek Meloyan and Narine Sarvazyan", align="center", size=11, after=14)

    P(doc, "Dear Editor,", after=8)
    P(doc, "We thank the editor and both reviewers for their careful reading and for comments that have improved the manuscript. "
           "We have addressed every point; where we chose a different way of addressing a suggestion than the one proposed, "
           "we give our reasons below. For each change we give its location in the revised manuscript as page, column and "
           "line numbers (lines are counted from the top of the column; figures, tables and their captions are not counted). "
           "A change-marked version (additions in blue, deletions in red) accompanies this letter. No experiments were re-run "
           "and no figure or table of results was changed; the few numbers added to the text are read from the experiments "
           "already reported.")
    P(doc, "In summary, the revision (i) makes Section III-C consistent with the reported results (PCA-based selection with "
           "k = 3; the percentile and absolute-range perturbation formulas and the magnitude sets actually used) and states in "
           "Section IV-B which settings produced the reported configurations; (ii) states in Section III-E the validating role "
           "of the labels in choosing among attribution configurations; (iii) lists the nine excitation-emission pairs of the "
           "9-band configuration and corrects the sensor-design sentences; (iv) adds a rationale for the attribution stage and "
           "for the fixed MMR trade-off; and (v) adds six recent references (2020-2024), keeping the foundational ones, and "
           "labels the band-selection families of Section II-B point by point.")

    H(doc, "Summary of changes", 1)
    T(doc, ["Location (revised manuscript)", "Change", "Point"], [
        (loc("intro_apps"), "Third opening reference added (Hong et al. 2024); two sentences on recent applications, citing the two suggested references", "R2.1, R2.2"),
        (loc("intro_p3"), "Recent review added (Rasti et al. 2020); embedded-method citation corrected", "R2.3"),
        (loc("rw_roadmap"), "Sentence stating the order and role of the four parts of Section II", "R2.6"),
        (f"{loc('rw_B_intro')}; {loc('rw_B_unsup')}", "Section II-B: families labelled point by point; recent review cited; new paragraph on unsupervised deep methods (BS-Nets)", "R2.4, R2.5"),
        (f"{loc('IIIC1')}; {loc('IIIC1b')}", "Section III-C.1: PCA-based selection with k = 3 stated as optimal; consistency sentence corrected", "R1.1"),
        (f"{loc('IIIC2')}; {loc('IIIC3')}", "Section III-C.2 and III-C.3: percentile and absolute-range formulas; magnitude sets as used; sensitivity equation adjusted", "R1.2"),
        (loc("IVB2_settings"), "Section IV-B.2: settings behind the reported configurations", "R1.2"),
        (loc("IIIE"), "Section III-E: validating role of the labels in choosing the configuration", "R1.3"),
        (loc("VG"), "Section V-G: retuning sentence made consistent", "R1.3"),
        (f"{loc('VF')}; {loc('VIH')}; {loc('concl')}", "The nine excitation-emission pairs listed; sensor-design and Conclusion sentences corrected", "R1.4"),
        (f"{loc('IIIC_rationale')}; {loc('IIID')}; {loc('VIE')}", "Rationale for the attribution stage; MMR trade-off stated as fixed a priori", "R2.7"),
        (loc("VII_I"), "Section VI-I: citations corrected", "R2.5"),
        ("Reference list", "Six references added (numbers at the end of this letter); none removed", "R2.1-R2.6"),
    ], widths=[2.4, 3.2, 0.9], size=9)

    # ------------------------------------------------------------------ Reviewer 1
    H(doc, "Reviewer 1", 1)
    P(doc, "We thank the reviewer for noting that the manuscript has improved and for the four precise points.", italic=True, color=GREY)

    H(doc, "Point 1 — Variance-based selection with k = 1 versus PCA-based selection with k = 3", 2)
    COMMENT(doc, "Section III-C states that variance-based latent dimension selection with k=1 gives the best performance, whereas the results show that PCA-based selection, including k=3, performs best. These descriptions should be made consistent throughout the manuscript.")
    P(doc, "", lead="Response.", after=2)
    P(doc, "The reviewer is right; the sentence in Section III-C was left over from an earlier stage of the work. It now reads "
           "that PCA-based dimension selection (defined in Section IV-B) with k = 3 achieved optimal performance, which is what "
           "Table V and Section V-E report. The following sentence, which attributed the most consistent rankings to "
           "variance-based scoring, was corrected in the same way so that Sections III-C, IV-B, V-E and VI-D now agree.")
    CHANGES(doc, [
        ("Section III-C.1: 'PCA-based dimension selection (Section IV-B) with k = 3 achieved optimal performance'.", "IIIC1"),
        ("Section III-C.1: 'PCA-based scoring yielded the most consistent wavelength rankings ...'.", "IIIC1b"),
    ])

    H(doc, "Point 2 — Perturbation settings", 2)
    COMMENT(doc, "The perturbation settings in the methodology do not match those reported in the experimental setup. The manuscript should clearly state which perturbation strategy and magnitude settings were actually used to obtain the reported results.")
    P(doc, "", lead="Response.", after=2)
    P(doc, "The reviewer is correct. Section III-C.2 described a standard-deviation-scaled perturbation with magnitudes 15, 30 "
           "and 45, whereas the experiments used the percentile and absolute-range schemes with the magnitude sets listed in "
           "Section IV-B.2. Section III-C.2 now gives the two schemes as formulas (Eqs. 11-12) with the magnitude sets "
           "{30, 40, 50} and {50, 60, 70}, using the same symbol as Section IV-B and Table IV, and the sensitivity definition of "
           "Section III-C.3 was adjusted so that it no longer refers to the removed quantities. Section IV-B.2 now also states "
           "which settings produced the reported results: the percentile scheme with the medium magnitude set for all "
           "configurations in Table V except the 30- and 9-band rows (absolute range, medium) and the 50-band row "
           "(percentile, high).")
    CHANGES(doc, [
        ("Section III-C.2 'Systematic Perturbation': percentile and absolute-range formulas; magnitude sets.", "IIIC2"),
        ("Section III-C.3 'Sensitivity Mapping': equation and sentence adjusted.", "IIIC3"),
        ("Section IV-B.2: sentence stating the settings behind the reported configurations.", "IVB2_settings"),
    ])

    H(doc, "Point 3 — Configuration selection uses labelled evaluation results", 2)
    COMMENT(doc, "The final results are selected from 3,072 configurations according to supervised classification accuracy. Although each wavelength ranking is generated without labels, the choice of the best configuration uses labeled evaluation results. This distinction should be clarified, preferably with a separate configuration-selection and evaluation protocol.")
    P(doc, "", lead="Response.", after=2)
    P(doc, "We agree that the distinction should be explicit. Section III-E already stated that the selection pipeline "
           "operates without labels and that the classifier serves only to evaluate the selected wavelengths; we have added "
           "a sentence stating that the labels play the same validating role in choosing among the attribution "
           "configurations of Section IV-C: every configuration produces its wavelength ranking without labels, and the "
           "classifier is then used to determine which configuration yielded the most informative selection, in the same way "
           "that a validation set is used to choose hyperparameters. We preferred this clarification to a separate protocol "
           "section because the sweep (Section IV-C) and the evaluation (Sections III-E and IV-D) are already described in "
           "full, the reported results are unchanged, and a new section would restate them. For consistency, the sentence in "
           "Section V-G that listed 'attribution settings' among the things not retuned for the collagen dataset was corrected, "
           "since the attribution configuration was chosen for that dataset in the same validated way.")
    CHANGES(doc, [
        ("Section III-E: sentence on the validating role of the labels in choosing the configuration.", "IIIE"),
        ("Section V-G: 'attribution settings' removed from the retuning sentence.", "VG"),
    ])

    H(doc, "Point 4 — Nine bands from eight excitation sources", 2)
    COMMENT(doc, "The manuscript states that the 9-band solution selects exactly one wavelength from each of eight excitation sources. These numbers are not consistent. The authors should report the actual nine excitation-emission pairs and revise the related sensor-design discussion accordingly.")
    P(doc, "", lead="Response.", after=2)
    P(doc, "A fair point; we apologise for the inconsistency. Section V-F now lists the nine pairs in selection order "
           "(excitation/emission, nm): 310/520, 385/500, 325/570, 430/610, 400/510, 340/500, 365/520, 415/615 and 430/680, "
           "i.e. one band from each of the eight excitation sources and a second band from the 430 nm source. The "
           "interpretation was adjusted to this near one-per-excitation pattern, which reflects the per-excitation "
           "normalisation of this configuration, and the sensor-design sentence in Section VI-H and the corresponding sentence "
           "in the Conclusion were corrected (all eight sources retained; one or two emission bands per source).")
    CHANGES(doc, [
        ("Section V-F: the nine pairs and the adjusted interpretation.", "VF"),
        ("Section VI-H.1: sensor-design sentence corrected.", "VIH"),
        ("Section VII: '(two for one source)' added.", "concl"),
    ])

    # ------------------------------------------------------------------ Reviewer 2
    H(doc, "Reviewer 2", 1)
    P(doc, "We thank the reviewer for the positive assessment of the topic and writing, and for the suggestions on the literature and the presentation of the first part of the paper.", italic=True, color=GREY)

    H(doc, "Point 1 — References [1] and [2]", 2)
    COMMENT(doc, "References [1] and [2] are relatively old. I suggest replacing them with more recent and up-to-date references.")
    P(doc, "", lead="Response.", after=2)
    P(doc, f"We agree that the opening references should also point to current work, and we have added the 2024 IEEE TPAMI "
           f"paper on spectral foundation models (Hong et al., now {refnum('hong2024spectralgpt')}) alongside them. We have kept "
           f"[1] (Plaza et al. 2009) and [2] (Bioucas-Dias et al. 2013), because they are the foundational surveys of hyperspectral "
           f"image processing on which the field's terminology rests (about 1,700 and 2,400 citations respectively), and the "
           f"sentence they support is a statement of established capability rather than of recent progress. Recent progress is "
           f"now covered by the added reference and by the application examples of Point 2.")
    CHANGES(doc, [("Introduction, first sentence: third citation added.", "intro_apps")])

    H(doc, "Point 2 — Applications of HSI", 2)
    COMMENT(doc, "The authors briefly mention several applications of HSI, but this part would benefit from additional references and examples. I suggest enriching the background/context of HSI applications and including the relevant references provided below [1], [2].")
    P(doc, "", lead="Response.", after=2)
    P(doc, f"Done. Two sentences were added to the first paragraph of the Introduction. They cite the two references kindly "
           f"suggested by the reviewer (Jörges et al. 2023 on the detection of photovoltaic power plants, now "
           f"{refnum('joerges2023photovoltaic')}; Deville et al. 2023 on unsupervised unmixing of oceanic data, now "
           f"{refnum('deville2023oceanic')}) as Earth-observation examples, contrast them with laboratory and clinical settings in "
           f"which fluorescence rather than reflectance is measured, and connect the breadth of applications to the recurring "
           f"cost of acquiring many bands, which is the problem the paper addresses.")
    CHANGES(doc, [("Introduction, paragraph 1: two sentences added.", "intro_apps")])

    H(doc, "Point 3 — Recent reviews on band selection", 2)
    COMMENT(doc, "In the third paragraph of the Introduction, the references related to the identification of relevant spectral bands are relatively old. I suggest including some recent review articles to provide a more up-to-date overview of this topic.")
    P(doc, "", lead="Response.", after=2)
    P(doc, f"Done. The paragraph now cites, next to the band-selection review of Sun and Du (2019), the widely cited overview of "
           f"Rasti et al. (2020, {refnum('rasti2020featureextraction')}) on hyperspectral feature extraction from shallow to deep "
           f"methods, which covers band selection among the recent developments. In the same sentence the citation for "
           f"attention-based embedded methods was corrected to Lorenzo et al. (2020, {refnum('lorenzo2020attention')}); the "
           f"reference previously cited there describes a recurrent classifier rather than band weighting.")
    CHANGES(doc, [("Introduction, paragraph 3.", "intro_p3")])

    H(doc, "Point 4 — Point-by-point description of each technique", 2)
    COMMENT(doc, "For the different approaches mentioned in this section, I suggest clearly describing the specific contribution of each introduced technique. A point-by-point description would help readers better understand the contribution and relevance of each technique.")
    P(doc, "", lead="Response.", after=2)
    P(doc, "Done. Section II-B now announces that the contribution of each family, and its limitation for ME-HSI data, are "
           "summarised in turn, and the four families are labelled point by point (filter, wrapper, embedded, and unsupervised "
           "deep methods). Each paragraph already stated what the family contributes and where it falls short for "
           "multi-excitation data, so we kept the existing text and added only the labels and the new fourth paragraph "
           "(Point 5), rather than expanding the section.")
    CHANGES(doc, [
        ("Section II-B, opening sentence and labels.", "rw_B_intro"),
        ("Section II-B: filter methods.", "rw_B_filter"),
        ("Section II-B: wrapper methods.", "rw_B_wrapper"),
        ("Section II-B: embedded methods.", "rw_B_embedded"),
        ("Section II-B: unsupervised deep methods (new paragraph).", "rw_B_unsup"),
    ])

    H(doc, "Point 5 — References in Section II-B", 2)
    COMMENT(doc, "Similarly, in Section II-B, some of the references are relatively old. I suggest updating this section by including more recent and relevant references to better reflect the current state of research.")
    P(doc, "", lead="Response.", after=2)
    P(doc, f"Done. Section II-B now cites the recent overview of Rasti et al. (2020, {refnum('rasti2020featureextraction')}) in its "
           f"opening sentence, attention-based band selection (Lorenzo et al. 2020, {refnum('lorenzo2020attention')}) under the "
           f"embedded methods, and, in a new paragraph, the unsupervised deep methods represented by BS-Nets (Cai et al. 2020, "
           f"{refnum('cai2020bsnets')}), which are the closest relatives of the present framework. The classical references were "
           f"kept because they are the original sources of each family (variance, entropy and correlation criteria; sequential "
           f"and evolutionary search; LASSO). Two citations were corrected in the process: the reference previously cited for "
           f"attention-based band weighting is a recurrent classifier, and the one cited for reinforcement-learning band "
           f"selection (HybridSN) is a classification architecture; the latter clause was removed from Sections II-B and VI-I "
           f"rather than replaced, to keep the reference list focused.")
    CHANGES(doc, [("Section II-B.", "rw_B_intro"), ("Section VI-I: citation corrected.", "VII_I")])

    H(doc, "Point 6 — Organisation of the Related Work section", 2)
    COMMENT(doc, "I suggest reorganizing the 'Related Work' section, as its current structure is not very clear. In addition, many of the cited references are relatively old. Updating the references with more recent and relevant studies would help provide a clearer and more up-to-date overview of the existing approaches.")
    P(doc, "", lead="Response.", after=2)
    P(doc, "We have kept the four-part structure but made its logic explicit with a sentence at the start of the section: the "
           "parts are ordered from the data to the method, the modality and its 4D structure motivating the problem (II-A), "
           "existing band selection methods defining the gap (II-B), deep representation learning (II-C) and perturbation-based "
           "attribution (II-D) supplying the two tools the framework combines, and the final part positioning the framework "
           "against them (II-E). We believe this ordering is the natural one for the paper and that stating it removes the "
           "unclarity rather than a reordering of the material. The references were updated as described in Points 3 and 5; "
           "the older references in II-A, II-C and II-D are the original sources for the systems, architectures and attribution "
           "methods they support, and we kept them for that reason.")
    CHANGES(doc, [("Section II, opening: sentence added.", "rw_roadmap")])

    H(doc, "Point 7 — Justification of the methodological choices", 2)
    COMMENT(doc, "The criteria and methodological choices presented in the Methodology section need to be better justified. The authors should provide more details and explanations regarding the rationale behind their choices and the criteria used at each step of the proposed approach.")
    P(doc, "", lead="Response.", after=2)
    P(doc, "We have added the rationale that connects the steps. At the start of Section III-C two sentences now explain why "
           "perturbing the latent space identifies informative wavelengths: a latent space trained to reconstruct the full 4D "
           "cube must retain whatever information the reconstruction requires, so wavelengths that are redundant or dominated "
           "by noise contribute little to it; tracing which excitation-emission pairs respond most strongly when a latent "
           "dimension is displaced therefore reveals the wavelengths on which the representation depends, after which MMR "
           "removes redundancy among them. The MMR trade-off is now stated as fixed a priori at lambda = 0.5 (equal weight to "
           "relevance and diversity) in Sections III-D and VI-E, instead of being described as an empirical optimum. The remaining criteria were "
           "already justified in the text (Rayleigh cutoffs and normalisation in III-A, the parallel-branch design and feature "
           "averaging in III-B, the variance criterion in III-C, the scaling of the perturbation to each dimension's "
           "variation in III-C.2, and the choice of a non-parametric classifier in III-E) and in the rationale column of "
           "Table III.")
    CHANGES(doc, [
        ("Section III-C, opening: two rationale sentences.", "IIIC_rationale"),
        ("Section III-D.1: lambda fixed a priori.", "IIID"),
        ("Section VI-E: sentence on lambda aligned.", "VIE"),
    ])

    # ------------------------------------------------------------------ index of changes
    H(doc, "Index of changes", 1)
    P(doc, "All edits in reading order. Line numbers count the text lines of the column from its top; figures, tables and captions are not counted.", size=10, color=GREY)
    rows = [(d.get("eids", ""), d.get("desc", k), d.get("point", ""), loc(k)) for k, d in LOC.items() if k != "_refs"]
    T(doc, ["Edit", "Description", "Point", "Location"], rows, widths=[0.6, 3.2, 0.7, 2.0], size=8.5)

    # ------------------------------------------------------------------ references added
    H(doc, "References added in the revision", 1)
    added = [
        ("hong2024spectralgpt", "Hong et al., SpectralGPT: Spectral remote sensing foundation model, IEEE Trans. Pattern Anal. Mach. Intell. 46(8), 5227-5244, 2024"),
        ("joerges2023photovoltaic", "Jörges, Vidal, Hank and Bach, Detection of solar photovoltaic power plants using satellite and airborne hyperspectral imaging, Remote Sens. 15(13), 3403, 2023"),
        ("deville2023oceanic", "Deville et al., Modeling and unsupervised unmixing based on spectral variability for hyperspectral oceanic remote sensing data with adjacency effects, Remote Sens. 15(18), 4583, 2023"),
        ("rasti2020featureextraction", "Rasti et al., Feature extraction for hyperspectral imagery: the evolution from shallow to deep, IEEE Geosci. Remote Sens. Mag. 8(4), 60-88, 2020"),
        ("lorenzo2020attention", "Lorenzo, Tulczyjew, Marcinkiewicz and Nalepa, Hyperspectral band selection using attention-based convolutional neural networks, IEEE Access 8, 42384-42403, 2020"),
        ("cai2020bsnets", "Cai, Liu and Cai, BS-Nets: an end-to-end framework for band selection of hyperspectral image, IEEE Trans. Geosci. Remote Sens. 58(3), 1969-1984, 2020"),
    ]
    T(doc, ["No.", "Reference"], [(refnum(k), txt) for k, txt in added], widths=[0.6, 5.9], size=9)

    P(doc, "We hope the revised manuscript is now suitable for publication in Scientific Reports and thank the reviewers again for their help.", after=8)
    P(doc, "Sincerely,", after=2)
    P(doc, "Narek Meloyan and Narine Sarvazyan", after=2)

    doc.save(OUT)
    missing = [k for k, d in LOC.items() if k != "_refs" and not d.get("found")]
    unnumbered = [k for k, _ in added if k not in REFS]
    print("wrote", OUT, "| unresolved locations:", missing, "| references without a number:", unnumbered)


if __name__ == "__main__":
    build()
