"""Locate every manuscript change in the line-numbered revised PDF and report page / column / line.

Reads manuscript/revised_linenumbered/main.pdf (lineno, pagewise: numbers restart on every page and run
down the left column, then the right column). Writes change_locations.json, consumed by
build_response_letter_docx.py. Run after any rebuild of the manuscript.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent
PDF = HERE / "manuscript" / "revised_linenumbered" / "main.pdf"
AUX = HERE / "manuscript" / "revised_linenumbered" / "main.aux"
OUT = HERE / "change_locations.json"

LIG = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl", "\u2013": "-", "\u2014": "-", "\u2019": "'"}


def clean(s: str) -> str:
    for a, b in LIG.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------------- page model
class Page:
    def __init__(self, page, pno):
        self.pno = pno
        W = page.rect.width
        self.mid = W / 2
        words = page.get_text("words")
        # candidate line-number tokens: pure integers
        nums = [w for w in words if re.fullmatch(r"\d{1,3}", w[4])]
        text = [w for w in words if not re.fullmatch(r"\d{1,3}", w[4])]
        # text left edges per column (robust: 5th percentile of x0 of alphabetic words)
        def edge(col):
            xs = sorted(w[0] for w in text if (w[0] < self.mid) == (col == "left") and re.search(r"[A-Za-z]{3,}", w[4]))
            return xs[int(0.05 * len(xs))] if xs else None
        self.edge = {"left": edge("left"), "right": edge("right")}
        # line numbers: integers whose right edge is left of (edge - 1pt) and within 18pt of it
        self.linenos = {"left": [], "right": []}
        for w in nums:
            for col in ("left", "right"):
                e = self.edge[col]
                if e is not None and w[2] <= e - 0.5 and w[0] >= e - 18:
                    self.linenos[col].append(((w[1] + w[3]) / 2, int(w[4])))
        # text lines grouped by (block, line)
        lines = {}
        for w in words:
            key = (w[5], w[6])
            lines.setdefault(key, []).append(w)
        self.lines = []
        for key, ws in lines.items():
            ws.sort(key=lambda w: w[0])
            x0 = min(w[0] for w in ws)
            x1 = max(w[2] for w in ws)
            yc = sum((w[1] + w[3]) / 2 for w in ws) / len(ws)
            txt = clean(" ".join(w[4] for w in ws))
            if re.fullmatch(r"\d{1,3}", txt):
                continue  # a bare line number
            col = "full" if (x0 < self.mid - 20 and x1 > self.mid + 20) else ("left" if (x0 + x1) / 2 < self.mid else "right")
            self.lines.append({"y": yc, "x0": x0, "x1": x1, "col": col, "text": txt})
        # merge fragments of one justified line (same column, same baseline) into a single line
        self.lines.sort(key=lambda l: (l["col"], l["y"], l["x0"]))
        merged = []
        for l in self.lines:
            if merged and merged[-1]["col"] == l["col"] and abs(merged[-1]["y"] - l["y"]) < 1.5:
                m = merged[-1]
                m["text"] = clean(m["text"] + " " + l["text"])
                m["x1"] = max(m["x1"], l["x1"])
                m["x0"] = min(m["x0"], l["x0"])
            else:
                merged.append(dict(l))
        self.lines = merged
        # reading order: left column top-down, then right column, full-width lines by y within their band
        self.lines.sort(key=lambda l: (0 if l["col"] == "left" else (1 if l["col"] == "right" else 0), l["y"]))

    def lineno_of(self, line):
        """Line number counted within the column (pagewise lineno numbering runs left column then right
        column; the right column's numbers are re-based to start at 1)."""
        col = line["col"]
        if col == "full":
            return None
        cands = [(abs(y - line["y"]), n) for y, n in self.linenos[col]]
        if not cands:
            return None
        d, n = min(cands)
        if d > 2.5:
            return None
        offset = (min(k for _, k in self.linenos["right"]) - 1) if (col == "right" and self.linenos["right"]) else 0
        return n - offset


def load_pages():
    doc = fitz.open(PDF)
    return [Page(p, i + 1) for i, p in enumerate(doc)]


def compact(s: str) -> str:
    return re.sub(r"\s+", "", clean(s))


def join_lines(texts):
    """Concatenate line texts, dropping end-of-line hyphenation, with all whitespace removed."""
    out = ""
    for t in texts:
        t = compact(t)
        if out.endswith("-"):
            out = out[:-1]
        out += t
    return out


def find(pages, phrase, occurrence=1, after=None):
    """Return (page, col, lineno, line_text, page_index, line_index) of the n-th line (reading order)
    on which `phrase` BEGINS. The phrase may continue over the next two lines of the same column
    (hyphenation-tolerant). `after` = (page_index, line_index) restricts the search to later lines."""
    cp = compact(phrase)
    seen = 0
    for pi, pg in enumerate(pages):
        if after and pi < after[0]:
            continue
        L = pg.lines
        for i, line in enumerate(L):
            if after and pi == after[0] and i <= after[1]:
                continue
            texts = [line["text"]]
            for k in (1, 2):
                if i + k < len(L) and L[i + k]["col"] == line["col"]:
                    texts.append(L[i + k]["text"])
            joined = join_lines(texts)
            pos = joined.find(cp)
            if pos != -1 and pos < max(1, len(compact(line["text"]))):
                seen += 1
                if seen == occurrence:
                    return pg.pno, line["col"], pg.lineno_of(line), line["text"], pi, i
    return None


# ----------------------------------------------------------------------------- change inventory
# key: (edit ids, description, reviewer point, start phrase, end phrase or None, kind)
#   kind "text"  -> page / column / lines a-b
#   kind "float" -> page / column, label only (captions, tables, figures are not line-numbered)
CHANGES_V3 = [
    ("intro_apps", "V1", "Introduction, paragraph 1: third opening reference added; two application sentences added", "R2.1, R2.2", "imaging has become an established", "small informative subset", "text", None),
    ("intro_p3", "V2", "Introduction, paragraph 3: recent review added; embedded-method citation corrected", "R2.3", "The challenge of identifying informative", "attention mechanisms", "text", None),
    ("rw_roadmap", "V3", "Section II, opening: sentence stating the order and role of the four parts", "R2.6", "The four parts are ordered", "positions the framework against them", "text", None),
    ("rw_B_intro", "V4", "Section II-B, opening: recent review added; point-by-point framing", "R2.4, R2.5", "Band selection methods aim", "summarized in turn", "text", None),
    ("rw_B_filter", "V4", "Section II-B: filter methods (labelled)", "R2.4", "Filter methods rank bands", None, "text", None),
    ("rw_B_wrapper", "V5", "Section II-B: wrapper methods (labelled)", "R2.4", "Wrapper methods evaluate band subsets", None, "text", None),
    ("rw_B_embedded", "V6", "Section II-B: embedded methods (labelled); attention citation corrected; mis-cited reinforcement-learning clause removed", "R2.4, R2.5", "Embedded methods integrate band selection", "expensive to obtain", "text", None),
    ("rw_B_unsup", "V6", "Section II-B: new paragraph on unsupervised deep methods (BS-Nets)", "R2.4, R2.5", "Unsupervised deep methods learn band weights", "fixed at training time", "text", None),
    ("IIIC_rationale", "V9", "Section III-C, opening: two rationale sentences", "R2.7", "The rationale follows from the autoencoder objective", "wavelengths it identifies", "text", None),
    ("IIIC1", "V7", "Section III-C.1: PCA-based selection with k = 3 stated as optimal", "R1.1", "dimensions were then selected for perturbation", "achieved optimal performance", "text", None),
    ("IIIC1b", "V8", "Section III-C.1: 'most consistent rankings' sentence made consistent (PCA)", "R1.1", "PCA-based scoring yielded the most consistent", "distinguish different materials", "text", None),
    ("IIIC2", "V10", "Section III-C.2: percentile and absolute-range formulas; magnitude sets as used", "R1.2", "Systematic Perturbation", "destructively large perturbations", "text", None),
    ("IIIC3", "V11", "Section III-C.3: sensitivity equation without the removed symbols", "R1.2", "Sensitivity Mapping", "displacements of latent dimension", "text", None),
    ("IIID", "V12", "Section III-D.1: lambda = 0.5 fixed a priori", "R2.7", "was fixed a priori", None, "text", None),
    ("IIIE", "V13", "Section III-E: sentence on the validating role of labels in choosing the configuration", "R1.3", "The labels play the same validating role", "most informative selection", "text", None),
    ("IVB2_settings", "V14", "Section IV-B.2: settings behind the reported configurations of Table V", "R1.2", "magnitude levels. The best", "(percentile, high)", "text", None),
    ("VF", "V16", "Section V-F: the nine pairs; interpretation adjusted", "R1.4", "comprises the following excitation/emission", "strongest band of every excitation first", "text", None),
    ("VG", "V17", "Section V-G: 'attribution settings' removed from the retuning sentence", "R1.3", "No hyperparameters or selection parameters were retuned", None, "text", None),
    ("VIE", "V18", "Section VI-E: lambda sentence aligned", "R2.7", "The balanced setting", "diversity term matters", "text", None),
    ("VIH", "V19", "Section VI-H.1: sensor-design sentence corrected", "R1.4", "The near one-per-excitation pattern observed", "effective discrimination", "text", None),
    ("VII_I", "V20", "Section VI-I: attention citation corrected; mis-cited reinforcement-learning clause removed", "R2.5", "attention-based networks", None, "text", None),
    ("concl", "V21", "Section VII: 9-band sentence corrected", "R1.4", "two for one source", None, "text", None),
]

CHANGES_MINIMAL = [
    ("intro_apps", "M1", "Introduction, paragraph 1: two opening references replaced; applications sentence added", "R2.1, R2.2", "Its applications now span", "image-guided surgery", "text", None),
    ("intro_p3", "M2", "Introduction, paragraph 3: recent reviews added; embedded-method citation corrected", "R2.3", "The challenge of identifying informative", "attention mechanisms", "text", None),
    ("rw_roadmap", "M3", "Section II, opening roadmap sentence", "R2.6", "This section proceeds from", "positioning the framework", "text", None),
    ("rw_A_add", "M4", "Section II-A: recent multi-excitation autofluorescence references", "R2.6", "Recent reviews of multi", "exploration of biological samples", "text", None),
    ("rw_B", "M5", "Section II-B: point-by-point (contribution / limitation) list; recent references; two mis-citations corrected", "R2.4, R2.5", "Band selection methods aim", "requires retraining", "text", None),
    ("rw_C_add", "M6", "Section II-C: recent surveys cited", "R2.6", "recent surveys trace", None, "text", None),
    ("rw_D_add", "M7", "Section II-D: perturbation-attribution references added", "R2.6", "Meaningful-perturbation methods", "explanation methods is given", "text", None),
    ("IIIC1", "M8, M9", "Section III-C.1: k in {1, 3}; PCA-based scoring introduced and stated as best; 'variance most consistent' sentence removed", "R1.1", "dimensions were then selected for perturbation", "remaining band counts", "text", None),
    ("IIIC2", "M10", "Section III-C.2 'Systematic Perturbation': displacement schemes and magnitude sets as used", "R1.2", "Systematic Perturbation", "destructively large perturbations", "text", None),
    ("IIIC3", "M11", "Section III-C.3 'Sensitivity Mapping': sensitivity equation without the removed symbols", "R1.2", "Sensitivity Mapping", "changes in that dimension", "text", None),
    ("IIID", "M12", "Section III-D.1: lambda = 0.5 fixed a priori", "R2.7", "was fixed a priori", None, "text", None),
    ("IVB1", "M13", "Section IV-B.1: sentence introducing the extended Table III", "R2.7", "together with the attribution and selection settings", None, "text", None),
    ("tabIII", "M14-M16", "Table III retitled and extended with attribution and selection settings and rationale", "R2.7", "Attribution and Selection Settings", None, "float", "Table III"),
    ("IVB2", "M17", "Section IV-B.2: magnitude symbol; reference to the Table V note; normalisation rationale", "R1.2, R2.7", "Both methods were explored at", "intensity variance", "text", None),
    ("tabIV", "M18", "Table IV: magnitude symbol", "R1.2", "Configuration Parameter Space", None, "float", "Table IV"),
    ("IVC", "M19", "Section IV-C: configuration selection vs evaluation; envelope; fixed configuration", "R1.3", "Two uses of the class labels", "no configuration choice was made there", "text", None),
    ("tabV", "M20", "Table V note: envelope stated; scheme, magnitude set and normalisation of each row", "R1.2, R1.3", "an envelope over the 48 attribution", None, "float", "Table V note"),
    ("VB_b", "M21", "Section V-B.1: fixed-configuration results (lichens)", "R1.3", "Under the configuration fixed in Section", "90.2% and 95.2%", "text", None),
    ("VF", "M22", "Section V-F: the nine pairs and the origin of the near one-per-excitation pattern", "R1.4", "comprises the following excitation/emission", "each excitation contributes", "text", None),
    ("VG_a", "M23", "Section V-G: retuning sentence corrected", "R1.3", "No autoencoder hyperparameters", "as for the lichen dataset", "text", None),
    ("VG_b", "M24", "Section V-G.1: fixed-configuration results (collagen)", "R1.3", "Under the configuration fixed on the lichen", "on this dataset", "text", None),
    ("VIE", "M25", "Section VI-E: lambda sentence aligned", "R2.7", "The MMR parameter was fixed at", "fluorescence spectra", "text", None),
    ("VIH", "M26", "Section VI-H.1: sensor-design sentence rewritten", "R1.4", "all eight excitation sources are retained", "nine detection channels", "text", None),
    ("VII_I", "M27", "Section VI-I: attention and reinforcement-learning citations corrected", "R2.5", "attention-based networks", None, "text", None),
    ("concl", "M28", "Section VII: 9-band sentence corrected", "R1.4", "retained all eight excitation sources", "sensor design guidelines", "text", None),
]

CHANGES_EXTENDED = [
    ("intro_apps", "E14, E15", "Introduction, paragraph 1: two opening references replaced; applications sentence added", "R2.1, R2.2", "Its applications now span", "image-guided surgery", "text", None),
    ("intro_p3", "E16", "Introduction, paragraph 3: recent reviews added; embedded-method citations corrected; label clause qualified", "R2.3", "The challenge of identifying informative", "at training time", "text", None),
    ("rw_roadmap", "E19a", "Section II, opening roadmap sentence", "R2.6", "This section proceeds from", "framework addresses", "text", None),
    ("rw_A_title", "E19b", "Section II-A retitled", "R2.6", "Dimensionality Problem", None, "text", None),
    ("rw_A_add", "E19c", "Section II-A: recent multi-excitation autofluorescence references", "R2.6", "Recent reviews of multi", "exploration of biological samples", "text", None),
    ("rw_B", "E17, E19d", "Section II-B retitled and rewritten as an enumerated point-by-point survey", "R2.4, R2.5", "Band selection methods aim", "4D imaging case", "text", None),
    ("rw_C_add", "E19e, E19f", "Section II-C retitled; recent surveys cited", "R2.6", "recent surveys trace", None, "text", None),
    ("rw_D_add", "E19g, E19h", "Section II-D retitled; perturbation-attribution references added", "R2.6", "Meaningful-perturbation methods", "explanation methods is given", "text", None),
    ("rw_E", "E19i, E19j", "Section II-E retitled; gap stated with reference to the numbered families", "R2.6", "Gap Addressed by This Work", "emission bands per excitation", "text", None),
    ("fig1", "T1a, T1b", "Figure 1: latent-space label and caption ('coordinates')", "R1.1", "Stage 1: Representation Learning", None, "float", "Fig. 1"),
    ("IIIC_intro", "T1c", "Section III-C, opening paragraph ('coordinate')", "R1.1", "Different latent coordinates encode", "encoded by that coordinate", "text", None),
    ("IIIC1", "E1, T1", "Section III-C.1 'Latent Coordinate Scoring' rewritten (scoring rules, k, PCA best)", "R1.1, R2.7", "Latent Coordinate Scoring", "not used for the reported results", "text", None),
    ("IIIC2", "E2", "Section III-C.2 'Systematic Perturbation' rewritten (schemes, magnitude sets)", "R1.2", "Systematic Perturbation", "settings behind each reported", "text", None),
    ("IIIC3", "E3", "Section III-C.3 'Sensitivity Mapping' rewritten", "R1.2", "Sensitivity Mapping", "changes in that coordinate", "text", None),
    ("IIIC4", "T1d, E4", "Section III-C.4: accumulation wording", "R1.2", "weighted by coordinate score", "three perturbation magnitudes", "text", None),
    ("IIID", "T3, E21", "Section III-D.1: MMR score as implemented; lambda fixed a priori", "R2.7", "influence score divided by its maximum", "it was not tuned", "text", None),
    ("alg1", "T3d", "Algorithm 1, line 8: MMR score as implemented", "R2.7", "Algorithm 1 MMR Wavelength Selection", None, "float", "Algorithm 1"),
    ("tabI", "T6", "Table I: spectral resolution 10 nm", "-", "Lichen Dataset Specifications", None, "float", "Table I"),
    ("IVB1", "E20a", "Section IV-B.1: sentence introducing the extended Table III", "R2.7", "together with the attribution and selection settings", None, "text", None),
    ("tabIII", "E20", "Table III retitled and extended with attribution and selection settings and rationale", "R2.7", "Attribution and Selection Settings", None, "float", "Table III"),
    ("IVB2", "E5, T1, T2, E21", "Section IV-B.2: coordinates, magnitude symbol, reference to Table V, patch set, normalisation rationale", "R1.2, R2.7", "For latent space analysis", "intensity variance", "text", None),
    ("IVB4", "T7", "Section IV-B.4: training pixels contained in the evaluated set", "-", "11,519 pixels", None, "text", None),
    ("tabIV", "T1e, T1f", "Table IV: row labels ('coordinate', magnitude symbol)", "R1.1", "Configuration Parameter Space", None, "float", "Table IV"),
    ("IVC", "T1g", "Section IV-C, attribution-parameters bullet ('coordinate')", "R1.1", "Two coordinate scoring", None, "text", None),
    ("IVD", "E7", "New Section IV-D 'Configuration Selection and Evaluation Protocol'", "R1.3", "Configuration Selection and Evaluation Protocol", "no configuration choice was made there", "text", None),
    ("VA3", "T1h", "Section V-A.3: last sentence ('coordinates')", "R1.1", "top-scoring latent coordinates", None, "text", None),
    ("VB_a", "E8a", "Section V-B: Table V introduced as the best-of-48 envelope with the fixed configuration", "R1.3", "best-of-48 envelope at key band counts", "at the same band counts", "text", None),
    ("tabV", "E6", "Table V: columns 'Scheme / magn.', 'Norm.', 'Fixed config.' added; caption and note revised", "R1.2, R1.3", "Best-of-48 Envelope and Fixed Configuration", None, "float", "Table V"),
    ("VB_b", "E8b", "Section V-B.1: fixed-configuration sentence after the 9-band result", "R1.3", "Under the fixed configuration of Section", "80-band optimum", "text", None),
    ("fig7", "E8c", "Figure 7 caption: 'Best-of-48 ... envelope'", "R1.3", "Best-of-48 classification accuracy envelope as a func", None, "float", "Fig. 7 caption"),
    ("VD_bullet", "T5a", "Section V-D, Object 8 bullet: four declines", "-", "Declined from 72.8%", "spectral overlap", "text", None),
    ("VD_text", "T5b", "Section V-D: '12 of 16 specimens'", "-", "12 of 16 specimens", None, "text", None),
    ("tabVII", "T5c, T5d", "Table VII: object counts and maximum", "-", "Object-Level Performance Summary", None, "float", "Table VII"),
    ("VE", "E1b, T4", "Section V-E: 'coordinate'; k sentence; 'best PCA configuration'", "R1.1", "PCA-based latent coordinate scoring", "exceeded the baseline)", "text", None),
    ("VF", "E11", "Section V-F: the nine pairs and the origin of the one-per-excitation coverage", "R1.4", "comprises the following excitation/emission", "are placed", "text", None),
    ("fig10", "E10a", "Figure 10 caption: descriptive, label-based grouping stated", "R1.3", "plays no role in selection", None, "float", "Fig. 10 caption"),
    ("VG_a", "E9a", "Section V-G: 'without retuning' sentence corrected", "R1.3", "No autoencoder hyperparameters", "applied unchanged", "text", None),
    ("VG_b", "E9b", "Section V-G.1: fixed-configuration collagen results", "R1.3", "Under the fixed configuration selected on the lichen", "on this dataset", "text", None),
    ("fig11", "E10b", "Figure 11 caption: descriptive, label-based grouping stated", "R1.3", "plays no role in selection", None, "float", "Fig. 11 caption"),
    ("fig12", "E8d", "Figure 12 caption: 'Best-of-48 ... envelope'", "R1.3", "for the collagen sponge dataset (Section", None, "float", "Fig. 12 caption"),
    ("VID", "T1i, T1j", "Section VI-D retitled; 'coordinates'", "R1.1", "Variance Coordinate Scoring", "coordinates independently", "text", None),
    ("VIE", "C1", "Section VI-E: lambda = 1 sentence replaced", "R2.7", "The MMR parameter was fixed at", "fluorescence spectra", "text", None),
    ("VIH", "E12", "Section VI-H.1: sensor-design sentence rewritten", "R1.4", "all eight excitation sources are retained", "nine detection channels", "text", None),
    ("VII_I", "E18", "Section VI-I: attention and reinforcement-learning citations corrected", "R2.5", "attention-based networks", None, "text", None),
    ("concl_a", "T1k", "Section VII: 'latent coordinate importance'", "R1.1", "tracing latent coordinate importance", None, "text", None),
    ("concl_b", "E13", "Section VII: 9-band sentence corrected", "R1.4", "retained all eight excitation sources", "sensor design guidelines", "text", None),
]

NEW_REF_KEYS_ALL = ["hong2026hyperspectral", "hong2024spectralgpt", "joerges2023photovoltaic", "deville2023oceanic", "lu2020agriculture",
                "lekha2026environmental", "sun2024foodindustry", "yoon2022clinical", "campbell2023beyondoncology", "sawant2021survey",
                "rasti2020featureextraction", "ahmad2022survey", "lorenzo2020attention", "cai2020bsnets", "balin2019concrete",
                "ghaffari2019multiexcitation", "dou2021attentionae", "roy2021darecnet", "nandi2023tattmsrecnet", "sun2022concrete",
                "mou2022reinforcement", "guo2025twostage", "wang2020onr", "fong2017perturbation", "covert2021removing", "ivanovs2021perturbation"]

NEW_REF_KEYS = ["hong2024spectralgpt", "tran2026medical", "joerges2023photovoltaic", "deville2023oceanic", "rasti2020featureextraction", "lorenzo2020attention", "cai2020bsnets"]


def main():
    pages = load_pages()
    out = {}
    fig_occ = {"fig10": 1, "fig11": 2}
    rows = []
    CHANGES = CHANGES_V3
    for key, eids, desc, point, start, end, kind, label in CHANGES:
        occ = fig_occ.get(key, 1)
        s = find(pages, start, occ)
        if s is None:
            out[key] = {"eids": eids, "desc": desc, "point": point, "found": False}
            rows.append((key, "NOT FOUND", start))
            continue
        e = find(pages, end, 1, after=(s[4], s[5] - 1)) if end else None
        loc = {"eids": eids, "desc": desc, "point": point, "found": True, "kind": kind, "label": label,
               "page": s[0], "col": s[1], "start": s[2], "end": (e[2] if e else None), "end_page": (e[0] if e else None), "end_col": (e[1] if e else None)}
        out[key] = loc
        rows.append((key, f"p.{s[0]} {s[1]} L{s[2]}" + (f" - p.{e[0]} {e[1]} L{e[2]}" if e else ""), s[3][:60]))
    # reference numbers of the added entries
    refs = {}
    if AUX.exists():
        for m in re.finditer(r"\\bibcite\{([^}]+)\}\{(\d+)\}", AUX.read_text(encoding="utf-8", errors="ignore")):
            if m.group(1) in NEW_REF_KEYS:
                refs[m.group(1)] = int(m.group(2))
    out["_refs"] = refs
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r[0]:12s} {r[1]:40s} {r[2]}")
    print("new reference numbers:", sorted(refs.values()))
    missing = [k for k, v in out.items() if k != "_refs" and not v.get("found")]
    if missing:
        print("MISSING:", missing)
        sys.exit(1)


if __name__ == "__main__":
    main()
