"""Build the CODASSCA 2026 presenter guide (Word document).

    .venv/bin/python publications/codassca2026/poster/build_presenter_guide.py

Output: publications/codassca2026/poster/CODASSCA2026_presenter_guide.docx
Content: datasets in plain English, algorithms compared, timed presentation script,
speaker notes per poster section, likely questions, glossary, and an appendix with
every poster visual and what to say about it.
"""
from __future__ import annotations
import re
from pathlib import Path
from PIL import Image
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FIG = ROOT / "figures"
OUT = ROOT / "CODASSCA2026_presenter_guide.docx"
EXTRA = {
    "lichens_labels": REPO / "publications/iasim_poster/02_lichens_TPAMI/02_lichens_labels_and_ROI.png",
    "lichens_maps": REPO / "publications/iasim_poster/02_lichens_TPAMI/13_classification_comparison_3panel.png",
    "lichens_heatmap": REPO / "publications/iasim_poster/02_lichens_TPAMI/09_wavelength_heatmap.png",
    "collagen_roi": REPO / "publications/iasim_poster/03_collagen_pepsin_IASIM/11_roi_overlay_pepsin.png",
    "collagen_classifiers": REPO / "results/Pepsin_Paper_Figures/classifier_curves.png",
    "collagen_heatmap": REPO / "results/Pepsin_Paper_Figures/wavelength_heatmap_v2.png",
    "poster": ROOT / "CODASSCA2026_poster_A0_preview.png",
}
NAVY = RGBColor(0x1A, 0x23, 0x32)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x52, 0x51, 0x4E)
TEXT_W_IN = 6.7

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
for lvl, size in ((1, 20), (2, 15), (3, 12.5)):
    hs = doc.styles[f"Heading {lvl}"]; hs.font.name = "Calibri"; hs.font.size = Pt(size); hs.font.bold = True
    hs.font.color.rgb = NAVY; hs.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")


# ---------------------------------------------------------------- helpers
def _runs(par, text, size=None, color=None, italic=False):
    """Mini-markup: **bold** segments."""
    for i, chunk in enumerate(re.split(r"(\*\*.+?\*\*)", text)):
        if not chunk:
            continue
        bold = chunk.startswith("**") and chunk.endswith("**")
        run = par.add_run(chunk[2:-2] if bold else chunk)
        run.bold = bold; run.italic = italic
        if size: run.font.size = Pt(size)
        if color: run.font.color.rgb = color
    return par


def H(text, level=1):
    h = doc.add_heading(text, level=level); h.paragraph_format.keep_with_next = True; return h


def P(text, size=None, color=None, italic=False, align=None, space_after=6):
    par = doc.add_paragraph(); _runs(par, text, size, color, italic)
    par.paragraph_format.space_after = Pt(space_after)
    if align: par.alignment = align
    return par


def SAY(text):
    """A 'say this' block: shaded, italic."""
    par = doc.add_paragraph()
    r = par.add_run("Say: "); r.bold = True; r.font.color.rgb = RED
    _runs(par, text, italic=True)
    par.paragraph_format.left_indent = Cm(0.6); par.paragraph_format.space_after = Pt(6)
    _shade(par, "FBEFEA")
    return par


def NOTE(text, label="Note"):
    par = doc.add_paragraph()
    r = par.add_run(f"{label}: "); r.bold = True; r.font.color.rgb = NAVY
    _runs(par, text)
    par.paragraph_format.left_indent = Cm(0.6); par.paragraph_format.space_after = Pt(6)
    _shade(par, "EAF1F9")
    return par


def _shade(par, hex_fill):
    pPr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    pPr.append(shd)


def BULLETS(items, style="List Bullet"):
    for it in items:
        par = doc.add_paragraph(style=style); _runs(par, it); par.paragraph_format.space_after = Pt(2)


def NUMBERS(items):
    BULLETS(items, style="List Number")


def TABLE(rows, col_widths=None, header=True, font=10, highlight=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j); cell.text = ""
            par = cell.paragraphs[0]; _runs(par, str(val), size=font); par.paragraph_format.space_after = Pt(0)
            if header and i == 0:
                for r in par.runs: r.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                _cell_fill(cell, "1A2332")
            elif highlight is not None and i == highlight:
                _cell_fill(cell, "FBEFEA")
                for r in par.runs: r.bold = True; r.font.color.rgb = RED
            elif i % 2 == 0:
                _cell_fill(cell, "F3F5F8")
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if i < len(t.rows) - 1:
            for cell in row.cells:
                for par in cell.paragraphs: par.paragraph_format.keep_with_next = True
    if col_widths:
        t.autofit = False
        tblPr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        for j, w in enumerate(col_widths):
            t.columns[j].width = Inches(w)
            for i in range(len(rows)): t.cell(i, j).width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _cell_fill(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def FIGURE(path, caption, width_in=TEXT_W_IN, max_h_in=8.0):
    path = Path(path)
    iw, ih = Image.open(path).size
    w = min(width_in, TEXT_W_IN); h = w * ih / iw
    if h > max_h_in: h = max_h_in; w = h * iw / ih
    w = max(w, 3.0) if iw / ih > 4 else w   # very wide strips stay readable
    doc.add_picture(str(path), width=Inches(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    par = doc.add_paragraph(); _runs(par, caption, size=9.5, color=GREY)
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER; par.paragraph_format.space_after = Pt(10)


def PAGEBREAK():
    doc.add_page_break()


# ================================================================ TITLE PAGE
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.space_before = Pt(120)
r = t.add_run("Presenter Guide"); r.bold = True; r.font.size = Pt(30); r.font.color.rgb = NAVY
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Label-Free, Dependency-Aware Channel Selection\nfor Coupled Multi-Channel Sensor Data"); r.font.size = Pt(16); r.bold = True
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("CODASSCA 2026 poster (paper 122)\nNarek Meloyan, Narine Sarvazyan"); r.font.size = Pt(13)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.space_before = Pt(30)
r = t.add_run("Datasets explained in simple English · algorithms compared · full presentation script\n"
              "speaker notes for every poster section · likely questions · every visual with what to say"); r.font.size = Pt(11); r.font.color.rgb = GREY
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.space_before = Pt(200)
r = t.add_run("Built 2026-09-08 from the poster sources in publications/codassca2026/poster/"); r.font.size = Pt(9); r.font.color.rgb = GREY
PAGEBREAK()

# ================================================================ CONTENTS
H("Contents", 1)
NUMBERS([
    "Quick orientation: what this guide is, the poster at a glance, the one-sentence message, numbers to remember",
    "The idea in simple words: the problem, why the usual methods fail, our method in three steps, how we judge a selection",
    "The datasets: controlled synthetic benchmark; motor-imagery EEG; lichens fluorescence imaging; collagen sponges fluorescence imaging; the three boundary datasets (PAMAP2, Opportunity, Indian Pines)",
    "The algorithms we compare against, and the classifiers that score them",
    "Presentation script: 30-second hook, 2-minute walk-through, 5-minute full version, closing",
    "Speaker notes for each poster section",
    "Likely questions and short answers",
    "Glossary in plain English",
    "Appendix A: every poster visual, what it shows, what to say",
    "Appendix B: extra visuals not on the poster (useful backup on a tablet)",
    "Appendix C: where the data and figure files live",
])
PAGEBREAK()

# ================================================================ 1. QUICK ORIENTATION
H("1. Quick orientation", 1)
H("1.1 What this guide is", 2)
P("This is the companion to the A0 poster. It explains, in simple English, everything a visitor might ask: "
  "what each dataset is, where it comes from, how big it is, what a \"channel\" means in it, which channels the method "
  "picked, which other algorithms we compared against, and what every picture on the poster shows. It also contains a "
  "script you can rehearse and a set of short answers for questions. Poster sessions run in parallel, so most visitors "
  "will give you 30 seconds to 2 minutes; the script is built in layers so you can stop at any point.")
H("1.2 The poster at a glance", 2)
FIGURE(EXTRA["poster"], "The full poster. Reading order: title → three concept columns (Introduction, Problem, Novelty) → "
       "Method band → one band per dataset (synthetic, EEG, imaging) → Results band → Conclusion, References, Acknowledgments.",
       width_in=5.2, max_h_in=7.4)
H("1.3 The one-sentence message", 2)
SAY("We pick a handful of real sensor channels without any labels, by learning how the channels depend on each other, "
    "and the channels we pick are as good as the ones a supervised method picks with labels.")
P("If you only have five seconds: **\"Label-free channel selection that matches a supervised selector.\"**")
H("1.4 Numbers to remember", 2)
TABLE([
    ["Dataset", "Budget K", "Proposed (no labels)", "Supervised MI (uses labels)", "PCA", "Variance", "Random-K", "All channels"],
    ["Synthetic (kNN accuracy)", "8 of 64", "0.94", "0.99", "0.95", "0.24", "0.25", "0.87"],
    ["Motor-imagery EEG (macro-F1)", "6 of 22", "0.52", "0.53", "0.48", "0.47", "0.49", "0.58"],
    ["Lichens imaging (pixel accuracy)", "9 of 192", "89.4 %", "-", "-", "-", "-", "88.2 %"],
    ["Collagen imaging (pixel accuracy)", "30 of 158", "85.6 %", "-", "-", "-", "-", "79.8 %"],
    ["PAMAP2 wearables (macro-F1, LOSO)", "7 of 27", "0.63", "0.61", "0.56", "0.54", "0.59 to 0.62", "0.72"],
    ["Opportunity wearables (kNN acc.)", "10 of 113", "0.84", "0.88", "0.85", "0.83", "0.85", "0.90"],
    ["Indian Pines HSI (kNN acc.)", "20 of 200", "0.65", "0.73", "0.58", "0.69", "0.76", "0.75"],
], col_widths=[1.7, 0.7, 0.9, 0.9, 0.55, 0.65, 0.75, 0.7], font=9)
P("Three phrases carry the whole story: **\"matches supervised\"** (synthetic, EEG), **\"removes 95 % of the bands without loss\"** "
  "(lichens), and **\"nothing beats random on redundant data, and we can tell in advance\"** (PAMAP2, Opportunity, Indian Pines).")
PAGEBREAK()

# ================================================================ 2. THE IDEA
H("2. The idea in simple words", 1)
H("2.1 The problem", 2)
P("Modern sensors record many signals at once. An EEG cap has 22 electrodes, a wearable set-up has three motion units with "
  "nine axes each, and a fluorescence camera records hundreds of wavelength bands. We call each recorded signal a **channel**. "
  "Channels arrive in natural **groups**: all electrodes of one scalp region, all axes of one motion unit, all emission bands "
  "recorded under one excitation light. The group structure is known from the hardware. It is not a label.")
P("Often we want to keep only a few of the real channels, not a mathematical mixture of them. Fewer channels means cheaper "
  "sensors, less power, less data, and a result a person can read (\"use electrodes FC3, Cz and CP3\"). The difficulty is "
  "choosing **which** channels to keep when we have **no class labels**.")
H("2.2 Why the usual label-free methods fail", 2)
P("The standard label-free tools score each channel on its own. **Variance ranking** keeps the channels that move the most. "
  "**PCA loadings** keep the channels that contribute most to the biggest principal components. Both are fast, but both look at "
  "one channel at a time and both favour the **loudest** signal. If the information sits in a quiet channel, they miss it. "
  "In EEG, for example, eye movements and the resting alpha rhythm are much louder than the motor rhythm we actually want.")
P("Methods that model how channels depend on each other exist (mRMR, JMI), but they need labels. Our method fills the gap: "
  "it is **label-free**, it is **dependency-aware**, and it returns **real channels**, not a projection.")
H("2.3 Our method in three steps", 2)
FIGURE(FIG / "fig_pipeline.png", "Figure 1 on the poster. The three stages, drawn with real intermediate outputs from the synthetic benchmark.")
P("**Step 1: learn the joint structure (a compressive group autoencoder).** Every group of channels goes through its own small "
  "encoder. The per-group features are averaged into one short code, called z, that has no time or space extent. Decoders then try "
  "to rebuild every channel from z. The network is trained only to reconstruct the data, so it never sees a label. Because z is "
  "small (a bottleneck), it cannot store noise; it has to store the shared structure that explains many channels at once.")
P("**Step 2: perturb the code and watch what changes.** We take the coordinates of z with the largest variance, shift each one by a "
  "few amounts, decode again, and measure how much each channel's reconstruction changed. A channel that reacts strongly is a "
  "channel the shared representation depends on. Summing those reactions gives every channel a **relevance score**, still with no labels.")
P("**Step 3: pick a diverse shortlist.** Channels are chosen greedily by maximal marginal relevance (MMR): the next pick has high "
  "relevance and low similarity to the channels already picked. The output is an ordered list that can be cut at any budget K.")
FIGURE(FIG / "fig_attribution.png", "Figure 2 on the poster. Left: the latent coordinates ranked by variance. Middle: how much each "
       "channel changes when each coordinate is shifted. Right: the accumulated relevance and the eight MMR picks.")
NOTE("Only the encoder changes between data types: 1-D convolutions for signals over time, 2-D convolutions for images. "
     "Everything after the encoder is identical for every dataset.")
H("2.4 How we judge a selection", 2)
P("Selection itself never uses labels. To measure whether a selection is good, we then train a simple classifier on the selected "
  "channels only, using labels, and compare its score against the same classifier on the channels chosen by four other selectors: "
  "**variance ranking**, **PCA loadings**, a **supervised mutual-information selector** (which does use labels, so it is our upper "
  "reference), and **random-K** (a random subset of the same size, the floor every selector must beat).")
PAGEBREAK()

# ================================================================ 3. DATASETS
H("3. The datasets", 1)
P("Four datasets carry the main results and three more mark the boundary of the method. For each one you will find: what it is, "
  "where it comes from and in which format, how big it is, what a channel means, which channels we highlighted, which algorithms "
  "were compared, the numbers, and the figures with what to say about them.")

# ---------------------------------------------------------------- 3.1 SYNTHETIC
H("3.1 Controlled synthetic benchmark", 2)
H("What it is and what it represents", 3)
P("A dataset we generate ourselves so that we know the truth. It imitates a multi-sensor recording in which the class of each "
  "window depends on **four hidden factors**. Each factor is expressed in a **cluster of five correlated channels**, so those five "
  "are near-copies of one another. The other 44 channels are pure noise. To classify a window correctly you must cover **all four** "
  "factors; keeping several channels from the same cluster gains nothing. Cluster amplitudes decrease (factor 1 is loudest, factor 4 "
  "is quietest), which is exactly the situation where variance-based methods pile into the loud cluster.")
H("Source and format", 3)
BULLETS(["Generated in code (`src/channel_select/synthetic.py`) with a fixed random seed, so it is fully reproducible.",
         "Format: NumPy/PyTorch arrays, one array per group of shape windows × time × channels.",
         "No files to download; the poster figures were made from a saved dump of one run (`data/synthetic_attribution.npz`)."])
H("Size and attributes", 3)
TABLE([
    ["Property", "Value"],
    ["Rows (samples)", "600 windows"],
    ["Time steps per window", "64"],
    ["Channels", "64, in 4 groups of 16"],
    ["Informative channels", "20 = 4 factors × 5 correlated channels each (amplitudes 2.0, 1.8, 1.6, 1.5)"],
    ["Noise channels", "44 (Gaussian noise, standard deviation 0.15)"],
    ["Classes", "16 = 2 states per factor, 4 factors"],
    ["What a channel is", "One simulated sensor signal over time; a cluster channel carries its factor's waveform plus noise"],
    ["Groups used by the method", "The 4 groups of 16 channels (fixed by construction, no labels)"],
], col_widths=[2.0, 4.7])
H("What we highlighted as important", 3)
P("At a budget of **8 channels** the label-free method picks channels from **all four clusters and no noise channel**. The supervised "
  "selector and PCA also cover all four factors here; variance ranking covers only two (it piles into the loud clusters); random "
  "mostly grabs noise. This run is one draw of the benchmark; over many random draws the method covers the factors with a mean "
  "score of 0.70 ± 0.29, because the bottleneck occasionally under-encodes one factor.")
H("Algorithms compared and the classifier", 3)
BULLETS(["Autoencoder: bottleneck temporal autoencoder, latent size 8, 100 epochs, trained on all 600 windows without labels.",
         "Attribution: shifts of the latent coordinates to the 50th, 70th and 90th percentile positions; MMR with redundancy weight 0.3.",
         "Baselines: variance ranking, PCA loadings (8 components), supervised mutual information, random-K (mean of many draws).",
         "Downstream classifier: k-nearest neighbours (k = 5) on the mean and standard deviation of each selected channel; 70/30 split."])
H("Results", 3)
TABLE([
    ["Selector", "K=2", "K=4", "K=6", "K=8", "K=12", "K=16", "K=20"],
    ["Proposed (label-free)", "0.267", "0.278", "0.494", "0.944", "0.994", "1.000", "1.000"],
    ["Supervised MI", "0.272", "0.450", "0.994", "0.994", "0.950", "1.000", "1.000"],
    ["PCA", "0.144", "0.517", "0.956", "0.950", "0.994", "0.994", "1.000"],
    ["Variance", "0.106", "0.139", "0.283", "0.244", "0.583", "0.911", "1.000"],
    ["Random-K", "0.100", "0.140", "0.234", "0.254", "0.295", "0.493", "0.536"],
], col_widths=[1.9, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68], font=9.5, highlight=1)
P("All 64 channels together score **0.87**. Eight selected channels score **0.94**, so removing the noise channels actually helps.")
H("Figures and what to say", 3)
FIGURE(FIG / "fig_syn_dataset.png", "Poster panel (a): one window (channels sorted by role) and the channel-by-channel correlation matrix. "
       "The four dark blocks on the diagonal are the four correlated clusters; the rest is noise.")
SAY("Here is one window of the synthetic data. The rows are the 64 channels, sorted so you can see the structure: four small blocks "
    "of five correlated channels, one per hidden factor, and 44 rows of noise. The correlation matrix on the right shows the same "
    "thing: four dark squares, nothing else.")
FIGURE(FIG / "fig_syn_curve.png", "Poster panel (b): kNN accuracy against the number of channels kept.")
SAY("With eight of 64 channels our method reaches 0.94, above the full set, because the full set includes 44 noise channels. "
    "The supervised selector, which uses labels, reaches 0.99. PCA also gets there because on this dataset the informative channels "
    "happen to be the loudest. Variance ranking piles all its picks into one cluster and stays at 0.24; random is at 0.25.")
FIGURE(FIG / "fig_syn_selection.png", "Poster panel (c): which channels each selector keeps at K = 8. Black bars are the picks; the "
       "background colour shows the channel's role.")
SAY("This strip shows exactly which channels each method picked. Ours: two from each cluster, no noise. Variance: only two clusters. "
    "Random: four noise channels.")
NOTE("If asked why PCA also works here: PCA recovers the factors because they are the highest-variance channels in this benchmark. "
     "The EEG dataset is the case where that assumption breaks.")
doc.add_paragraph().paragraph_format.space_after = Pt(10)

# ---------------------------------------------------------------- 3.2 EEG
H("3.2 Motor-imagery EEG (BCI Competition IV, dataset 2a)", 2)
H("What it is and what it represents", 3)
P("A public brain-computer-interface benchmark. Nine healthy people wore a 22-electrode EEG cap and, on a cue, **imagined** moving "
  "their left hand, right hand, both feet or tongue, without moving. Each imagined movement changes the 8 to 30 Hz rhythm over the "
  "motor cortex slightly. The task is to tell the four imagined movements apart from the EEG alone. The signal we want is weak; eye "
  "movements and the resting alpha rhythm over the back of the head are much stronger. That makes it the ideal test of "
  "\"the informative channel is not the loudest one\".")
H("Source and format", 3)
BULLETS(["Public dataset BNCI2014_001 (Tangermann et al., 2012), downloaded through the MOABB library (original GDF files).",
         "In memory: one array of trials × 22 electrodes × 1001 time samples (4 seconds at 250 Hz), band-pass filtered to 8 to 30 Hz.",
         "Per subject: 2 recording sessions on different days, 288 trials each (72 per class)."])
H("Size and attributes", 3)
TABLE([
    ["Property", "Value"],
    ["Subjects", "9"],
    ["Rows (samples)", "576 trials per subject (2 sessions × 288); 5,184 in total"],
    ["Time samples per trial", "1001 (4 s at 250 Hz); decimated by 3 to 334 for autoencoder training only"],
    ["Channels", "22 EEG electrodes (10-20 positions Fz to POz)"],
    ["Classes", "4: left hand, right hand, feet, tongue (balanced)"],
    ["What a channel is", "The voltage over time at one scalp electrode"],
    ["Groups used by the method", "4 scalp regions from the montage: frontal (Fz, FC3, FC1, FCz, FC2, FC4), central (C5, C3, C1, Cz, C2, C4, C6), "
     "centro-parietal (CP3, CP1, CPz, CP2, CP4), parietal (P1, Pz, P2, POz)"],
    ["Split", "Within subject: selection and classifier fitted on session 1, tested on session 2"],
], col_widths=[2.0, 4.7])
H("What we highlighted as important", 3)
P("At a budget of six electrodes, across the nine subjects, the label-free method most often picked **POz (7 of 9 subjects), FC3 (6), "
  "Cz (5), C6, CP3 and CP1 (4 each)**: a spread over the sensorimotor strip plus one midline parietal electrode. PCA most often picked "
  "**C5 (all 9), POz (8), CP3 (7), C3, C6, CP4**: the loud lateral and occipital electrodes. Variance picked the parietal row **Pz, P2, "
  "POz (7 each)**. The supervised selector picked **C3 (7), P2, CP3, CP4, P1**.")
H("Algorithms compared and the classifier", 3)
BULLETS(["Autoencoder: bottleneck temporal autoencoder, latent size 8, 20 epochs, trained on session-1 trials (up to 1,500 trials, decimated).",
         "Attribution: percentile shifts (50, 70, 90); MMR redundancy weight 0.4.",
         "Baselines: variance ranking, PCA loadings, supervised mutual information (on band power), random-K.",
         "Downstream classifier: common spatial patterns (CSP) followed by linear discriminant analysis (LDA) on the selected electrodes, "
         "the standard motor-imagery pipeline; score is macro-F1 on session 2, averaged over the 9 subjects."])
H("Results", 3)
TABLE([
    ["Selector", "K=3", "K=4", "K=6", "K=8", "K=10"],
    ["Proposed (label-free)", "0.399", "0.426", "0.517", "0.533", "0.544"],
    ["Supervised MI", "0.419", "0.456", "0.527", "0.540", "0.561"],
    ["PCA", "0.340", "0.412", "0.484", "0.538", "0.568"],
    ["Variance", "0.372", "0.402", "0.466", "0.525", "0.524"],
    ["Random-K", "0.396", "0.427", "0.486", "0.501", "0.535"],
], col_widths=[2.2, 0.9, 0.9, 0.9, 0.9, 0.9], font=9.5, highlight=1)
P("All 22 electrodes score **0.578** (± 0.143 across subjects). At K = 6 our six electrodes keep **89 %** of that. The method is at or above "
  "random for **8 of the 9 subjects**. PCA is below random at K = 3 to 6 and only catches up at K = 8 and 10.")
H("Figures and what to say", 3)
FIGURE(FIG / "fig_eeg_montage.png", "Poster panel (a): the 22-electrode montage coloured by the four region groups.", width_in=4.6)
SAY("These are the 22 electrodes, seen from above with the nose at the top. The four colours are the four scalp regions we use as "
    "groups. The grouping comes from the cap layout, so it needs no labels.")
FIGURE(FIG / "fig_eeg_topomaps.png", "Poster panel (b): signal power, label-free relevance, and the electrodes picked by our method and by PCA "
       "(darker = picked for more of the nine subjects).")
SAY("The first head shows where the signal is loud: the sides and the back, which is muscle, alpha and eye activity. That is what "
    "variance and PCA rank by, so PCA picks C5, C6 and POz. The second head is our relevance score; it lights up over the motor strip. "
    "The third and fourth heads show the actual picks: ours cluster on FC3, Cz, CP1, CP3; PCA's sit on the loud rim.")
FIGURE(FIG / "fig_eeg_curve.png", "Poster panel (c): macro-F1 against the number of electrodes kept, mean over nine subjects.")
SAY("With six of 22 electrodes we reach 0.52, which is 89 percent of the full-cap score and within 0.01 of the supervised selector. "
    "PCA and variance are no better than random at small budgets.")
NOTE("Be honest if pressed: the margin over random at K = 6 is about 0.03 macro-F1, and subject-to-subject variation is large. The "
     "strong claim is \"matches the supervised selector without labels while PCA collapses to random\", not \"a huge gain over random\".")
doc.add_paragraph().paragraph_format.space_after = Pt(10)

# ---------------------------------------------------------------- 3.3 LICHENS
H("3.3 Lichens: multi-excitation fluorescence imaging", 2)
H("What it is and what it represents", 3)
P("Lichens are organisms made of a fungus and an alga living together; they are used as indicators of air quality. Under ultraviolet "
  "and blue light they **fluoresce**: they absorb the light and re-emit it at longer wavelengths, with a pattern that depends on their "
  "chemistry. We imaged 16 lichen specimens (four species, four specimens each, laid out in a 4 × 4 grid) with a camera that records a "
  "full emission spectrum at every pixel, and we repeated that under **eight different excitation lights**. The result is a 4-D cube: "
  "two spatial axes, an emission axis and an excitation axis. Every (excitation, emission) pair is one image, and one **channel**.")
H("Source and format", 3)
BULLETS(["Our own acquisition: Mightex WheeLED multi-LED excitation source (310 to 440 nm) and a PerkinElmer Nuance FX hyperspectral camera "
         "(emission 420 to 720 nm in 10 nm steps). Species labels from the Takhtajyan Institute of Botany, NAS Armenia.",
         "Raw camera files are one cube per excitation; after pre-processing they are stored as a Python pickle holding, for each excitation, "
         "a cube of height × width × emission bands, the valid-pixel mask, plus a class-mask image and the training ROI boxes.",
         "Bands where the excitation light itself leaks into the emission (Rayleigh scatter, first and second order) are removed, which is why "
         "each excitation has a different number of usable emission bands and the excitation × emission grid is not a full rectangle."])
H("Size and attributes", 3)
TABLE([
    ["Property", "Value"],
    ["Image size", "1040 × 925 pixels; 191,046 pixels fall on a specimen (the rest is background and is masked out)"],
    ["Rows (samples)", "191,046 labelled pixels; 11,519 of them (one 50 × 60-pixel box per class, first row of the grid) are the training set; "
     "all 191,046 are scored"],
    ["Excitations", "8: 310, 325, 340, 365, 385, 400, 415, 430 nm"],
    ["Emission bands per excitation", "22, 22, 22, 27, 28, 25, 24, 22 (ranges from 420-720 nm down to 470-680 nm after Rayleigh removal)"],
    ["Channels", "192 valid (excitation, emission) pairs"],
    ["Classes", "4 lichen classes; pixels per class 57,947 / 37,852 / 56,090 / 39,157"],
    ["What a channel is", "The fluorescence intensity image at one excitation and one emission wavelength"],
    ["Groups used by the method", "The 8 excitations (all emission bands recorded under one light form a group)"],
], col_widths=[2.0, 4.7])
H("What we highlighted as important", 3)
P("The best 9-band selection, which scores above the full 192-band set, is: **(310, 520), (385, 500), (325, 570), (430, 610), (400, 510), "
  "(340, 500), (365, 520), (415, 615), (430, 680)** in (excitation, emission) nanometres. Note that it takes **one emission band under "
  "almost every excitation**: the method spreads its picks across excitations instead of stacking them on the brightest one. That is a "
  "direct design hint for a cheaper instrument: a few LEDs, one filter each.")
H("Algorithms compared and the classifier", 3)
BULLETS(["Autoencoder: the same engine with a 2-D spatial encoder (per-excitation branches, averaged into one latent code).",
         "A sweep of 3,072 configurations (band counts 1 to 180, dimension ranking, perturbation type and magnitude, normalisation, MMR settings); "
         "the poster reports the best configuration at each band count.",
         "Downstream classifier: k-nearest neighbours on the per-pixel spectrum of the selected bands, trained on the four ROI boxes, scored on every specimen pixel.",
         "Extra validation (thesis / journal paper, not on the poster): the 13-band selection scored 90.2 % against a mean of 46.1 % for 10,000 random 13-band subsets."])
H("Results", 3)
TABLE([
    ["Bands kept", "192 (all)", "4", "9", "13", "30", "80 (peak)"],
    ["Pixel accuracy", "88.2 %", "88.1 %", "89.4 %", "90.2 %", "93.0 %", "95.2 %"],
    ["Reduction", "0 %", "98 %", "95 %", "93 %", "84 %", "58 %"],
], col_widths=[1.5, 0.9, 0.8, 0.8, 0.8, 0.8, 1.0], font=9.5)
H("Figures and what to say", 3)
FIGURE(FIG / "fig_hsi_dataset.png", "Poster panel (a): photo, class map and one selected band image for lichens (left) and collagen (right).")
SAY("This is what the imaging data looks like. A photo of the 16 lichen pieces, the four classes as colours, and one of the 192 band "
    "images. Every excitation-emission pair gives one such image, so one pair is one channel.")
FIGURE(FIG / "fig_hsi_eem.png", "Poster panel (b): the excitation × emission grids with the selected bands marked as white dots; "
       "background colour is the mean emission intensity per excitation.")
SAY("Each row is one excitation light, each column an emission wavelength. Grey cells are bands we cannot use because the excitation "
    "light leaks in. The white dots are the nine bands we keep. They are spread over the excitations, not stacked on the brightest region.")
FIGURE(FIG / "fig_hsi_bars.png", "Poster panel (c): per-pixel accuracy with all bands versus the selected subset.", width_in=4.2)
SAY("Nine of 192 bands give 89.4 percent against 88.2 with everything: a 95 percent reduction with no loss. Dropping the uninformative "
    "bands also removes their noise, which is why the accuracy goes up rather than down.")
doc.add_paragraph().paragraph_format.space_after = Pt(10)

# ---------------------------------------------------------------- 3.4 COLLAGEN
H("3.4 Collagen sponges: multi-excitation fluorescence imaging", 2)
H("What it is and what it represents", 3)
P("Collagen is the main structural protein of skin, tendon and cartilage. Collagen sponges are porous scaffolds used in tissue "
  "engineering; they are chemically **cross-linked** to make them stronger, and the amount of cross-linker changes how the collagen "
  "fluoresces. Nine sponge specimens were prepared at **three cross-linker concentrations** (three specimens each, arranged in a 3 × 3 "
  "grid with one concentration per column) and imaged with the same camera under **six excitation lights**. The task is to tell the "
  "three concentrations apart from the fluorescence at each pixel. It tests whether the method transfers to a chemically different "
  "sample without any retuning.")
H("Source and format", 3)
BULLETS(["Our own acquisition with the same instrument as the lichens; six excitations (310 to 400 nm) because 415 and 430 nm added nothing in pilot tests.",
         "Stored in the same pickle format as the lichens: per-excitation cubes, valid-pixel mask, class mask, ROI boxes; exposure times and "
         "LED power per excitation are recorded for intensity normalisation.",
         "Labels come from the known preparation of each specimen (the column of the grid)."])
H("Size and attributes", 3)
TABLE([
    ["Property", "Value"],
    ["Image size", "256 × 348 pixels; 39,970 pixels fall on a sponge"],
    ["Rows (samples)", "39,970 labelled pixels; 5,934 of them (one small box per class in the first row) are the training set; all are scored"],
    ["Excitations", "6: 310, 325, 340, 365, 385, 400 nm"],
    ["Emission bands per excitation", "24, 24, 24, 28, 31, 27 (420-720 nm in 10 nm steps, minus Rayleigh-affected bands)"],
    ["Channels", "158 valid (excitation, emission) pairs"],
    ["Classes", "3 cross-linker concentrations; pixels per class 12,473 / 13,652 / 13,845"],
    ["What a channel is", "The fluorescence intensity image at one excitation and one emission wavelength"],
    ["Groups used by the method", "The 6 excitations"],
], col_widths=[2.0, 4.7])
H("What we highlighted as important", 3)
P("The 30-band selection that scores 85.6 % concentrates on two regions: the **red emission window 620 to 720 nm under 365 to 400 nm excitation**, "
  "which is where the fluorescence of cross-link products sits and therefore scales with concentration, and a **short-wavelength shoulder at "
  "420 to 440 nm under every excitation**, which is native collagen fluorescence. Top picks in order: (365, 660), (365, 420), (385, 420), "
  "(325, 430), (340, 420), (310, 420), (385, 720), (325, 420), (385, 710), (385, 430) and then the 640 to 720 nm bands under 365, 385 and 400 nm.")
H("Algorithms compared and the classifier", 3)
BULLETS(["Exactly the same pipeline and hyper-parameter family as the lichens, unchanged; sweep of 432 configurations.",
         "Downstream classifier on the poster: k-nearest neighbours per pixel, trained on the three ROI boxes.",
         "Extra validation (thesis / journal paper): ten classifiers (kNN variants, LDA, linear and RBF SVM, MLP, random forests, gradient boosting) "
         "on the same selected bands; nine of ten benefit from selection, and LDA reaches 92.5 % at 50 bands against 92.8 % on all 158."])
H("Results", 3)
TABLE([
    ["Bands kept", "158 (all)", "5", "10", "20", "30", "50"],
    ["Pixel accuracy (kNN)", "79.8 %", "80.9 %", "81.7 %", "85.0 %", "85.6 %", "85.1 %"],
    ["Reduction", "0 %", "97 %", "94 %", "87 %", "81 %", "68 %"],
], col_widths=[1.6, 0.95, 0.8, 0.8, 0.8, 0.8, 0.8], font=9.5)
SAY("Same pipeline, no retuning, on a chemically different sample: 30 of 158 bands raise accuracy from 79.8 to 85.6 percent. "
    "The selected bands sit exactly where the chemistry says the cross-link signal should be, in the red, plus the native collagen "
    "shoulder in the blue.")
NOTE("The lichens are a 4-class species problem; the collagen sponges are a 3-class concentration problem. Together they show the "
     "method is not tuned to one kind of sample.")
doc.add_paragraph().paragraph_format.space_after = Pt(10)

# ---------------------------------------------------------------- 3.5 BOUNDARY
H("3.5 The boundary datasets: where no selector beats random", 2)
P("These three datasets are on the poster to show honesty and a useful diagnostic. On all three, channels are near-duplicates of one "
  "another (neighbouring wavelengths, or accelerometers on the same body), so any subset carries about the same information. "
  "There, **no unsupervised selector, ours or PCA, separates from a random subset**, and even the supervised selector barely does. "
  "A simple **label-using check** (score of the best K channels under a supervised ranking minus the score of random K channels) "
  "flags such datasets before any modelling.")
FIGURE(FIG / "fig_boundary_doc.png", "Poster Figure 3, split in two rows for readability: gain over a random subset of the same size for the label-free method, PCA and the supervised selector. Top: the two datasets where selection pays. Bottom: the three redundant datasets.")
H("PAMAP2: wearable human-activity recognition", 3)
TABLE([
    ["Property", "Value"],
    ["What it represents", "People doing 12 everyday activities (walking, running, cycling, ironing, ...) while wearing three motion units on hand, chest and ankle"],
    ["Source and format", "Public UCI dataset in the MONSTER pre-windowed version (Hugging Face): a NumPy array of 38,856 windows × 52 raw channels × 100 samples (1 s at 100 Hz), CSV labels and subject ids"],
    ["Rows", "38,856 windows; 9 subjects (subjects 1 to 8 used in leave-one-subject-out evaluation)"],
    ["Channels used", "27: for each of the 3 units, accelerometer, gyroscope and magnetometer x/y/z (temperature, heart rate, low-range accelerometer and orientation columns dropped)"],
    ["Groups", "The 3 body-worn units (hand, chest, ankle)"],
    ["Classifier", "k-nearest neighbours (k = 5) on the mean and standard deviation of each selected channel; macro-F1, leave-one-subject-out"],
    ["Result at K = 7", "All 27: 0.72. Proposed 0.63 vs random 0.62 in the same run; supervised MI 0.61, PCA 0.56, variance 0.54, random 0.59 in the baseline run"],
], col_widths=[1.8, 4.9], font=9.5)
H("Opportunity: dense wearable activity recognition", 3)
TABLE([
    ["Property", "Value"],
    ["What it represents", "Locomotion (stand, walk, sit, lie, and a null class) recorded with a very dense set of body-worn sensors"],
    ["Source and format", "MONSTER version on Hugging Face: 17,386 windows × 113 channels × 100 samples"],
    ["Rows / channels / classes", "17,386 windows; 113 channels; 5 classes"],
    ["Groups", "8 equal blocks of channels"],
    ["Classifier", "k-nearest neighbours (k = 5) on mean and standard deviation features; within-distribution 70/30 split"],
    ["Result at K = 10", "All 113: 0.90. Proposed 0.84, supervised MI 0.88, PCA 0.85, variance 0.83, random 0.85"],
], col_widths=[1.8, 4.9], font=9.5)
H("Indian Pines: remote-sensing hyperspectral image", 3)
TABLE([
    ["Property", "Value"],
    ["What it represents", "A classic airborne hyperspectral scene of farmland in Indiana (AVIRIS sensor); the task is to label each pixel with one of 16 crop or land-cover classes"],
    ["Source and format", "MATLAB .mat files (corrected cube and ground truth) from Hugging Face"],
    ["Size", "145 × 145 pixels × 200 reflectance bands; 10,249 labelled pixels; 16 classes"],
    ["Groups", "5 equal blocks of adjacent bands; 7 × 7-pixel patches through the 2-D encoder"],
    ["Classifier", "k-nearest neighbours (k = 5) on the pixel spectrum; 50/50 stratified split"],
    ["Result at K = 20", "All 200: 0.75. Proposed 0.65, supervised MI 0.73, PCA 0.58, variance 0.69, random 0.76"],
], col_widths=[1.8, 4.9], font=9.5)
SAY("The right side of Figure 3 is the honest part. On wearable sensors and on a remote-sensing image, neighbouring channels are copies of "
    "each other, so a random subset is as good as any selection. Not even the supervised selector helps. The useful lesson is that a cheap "
    "label-using check tells you this in advance, so you do not waste time on selection where it cannot pay.")
NOTE("The message is a positive one: the method pays where the informative channels are a small, non-redundant part of the input, which "
     "is exactly the situation in the EEG cap and the fluorescence camera, and that advantage grows with the number of uninformative channels.")
PAGEBREAK()

# ================================================================ 4. ALGORITHMS
H("4. The algorithms we compare against", 1)
TABLE([
    ["Selector", "Needs labels?", "Sees dependencies between channels?", "How it ranks channels", "Where it fails"],
    ["Proposed (ours)", "No", "Yes, through the shared latent code", "Reconstruction change when latent factors are perturbed, then MMR for diversity",
     "Data where channels are near-duplicates (nothing beats random there); occasional under-encoding of a factor"],
    ["Supervised mutual information (MI)", "Yes", "No (one channel at a time, but against the label)", "Mutual information between each channel's feature and the class label",
     "Needs labels; used here only as the upper reference"],
    ["PCA loadings", "No", "Partly (through covariance), but variance-driven", "Sum of absolute loadings of each channel on the top principal components",
     "Picks loud channels; on EEG it collapses to random at small budgets"],
    ["Variance ranking", "No", "No", "Variance of each channel", "Piles into the loudest cluster; misses quiet informative channels"],
    ["Random-K", "No", "No", "Random subset of size K, averaged over draws", "The floor: any selector must beat it to be useful"],
], col_widths=[1.25, 0.7, 1.35, 1.8, 1.6], font=9)
H("The classifiers that score a selection", 2)
BULLETS(["**k-nearest neighbours (k = 5):** synthetic benchmark, both imaging datasets, PAMAP2, Opportunity and Indian Pines. Simple, no tuning, so the score reflects the channels, not the classifier.",
         "**CSP + LDA:** the EEG dataset. Common spatial patterns is the standard spatial filter for motor imagery; LDA is the standard linear classifier after it.",
         "**Ten-classifier check (collagen, thesis only):** kNN variants, LDA, SVM linear and RBF, MLP, random forests, gradient boosting; nine of ten benefit from the selection."])
H("The label-using headroom check", 2)
P("Before running any selector on a new dataset, rank channels by a supervised score (mutual information with the label), and compare the "
  "downstream accuracy of the best K channels with the accuracy of random K channels. If the difference is near zero, the dataset is "
  "\"saturated\": every subset is as good as any other and no selection method can help. This check is cheap and it correctly flags "
  "PAMAP2, Opportunity and Indian Pines.")
PAGEBREAK()

# ================================================================ 5. SCRIPT
H("5. Presentation script", 1)
P("The script is layered. Start with the hook. If the visitor stays, continue with the walk-through. If they ask questions, use the full "
  "version and the answers in Section 7. Pointing cues are in brackets.")
H("5.1 The 30-second hook (someone slows down in front of the poster)", 2)
SAY("Hi. This is about choosing a few real sensor channels, like electrodes or camera bands, when you have no labels at all. "
    "[point to the Method band] We train a small autoencoder to rebuild all channels from a compressed code, then we poke each part "
    "of the code and watch which channels react. The channels that react are the ones the whole recording depends on. "
    "[point to the EEG band] On motor-imagery EEG, six of 22 electrodes chosen this way match a supervised selector, while PCA "
    "picks the loud electrodes and is no better than random. [point to the imaging band] On fluorescence imaging we drop 95 percent "
    "of the bands with no loss of accuracy. Happy to walk you through it.")
H("5.2 The 2-minute walk-through", 2)
NUMBERS([
    "**Problem** [Introduction and Problem columns]. \"Sensors give us dozens or hundreds of channels that arrive in groups: electrodes per scalp region, "
    "bands per excitation light. We want to keep a handful of the real channels, for cost and for interpretability, and we have no labels. "
    "The usual label-free tools, variance and PCA, look at one channel at a time and prefer the loudest one. When the useful signal is quiet, they miss it.\"",
    "**Method** [Figure 1]. \"Three steps. One: each group of channels gets its own encoder, the features are averaged into a short code, and decoders "
    "rebuild every channel; training uses reconstruction only. Two: we shift each coordinate of the code and measure how much every channel's "
    "reconstruction changes; that is a label-free relevance score. Three: a greedy rule picks channels with high relevance and low similarity "
    "to what is already picked. Only the encoder changes between signals and images.\"",
    "**Synthetic** [synthetic band]. \"We first check it on data where we know the truth: four hidden factors, each in a cluster of five correlated "
    "channels, plus 44 noise channels. With eight of 64 channels we score 0.94, above the full set, and we cover all four factors with no noise. "
    "Variance ranking gets stuck in one loud cluster.\"",
    "**EEG** [EEG band]. \"The real test: nine subjects imagining four movements. The signal we need is weaker than eye and alpha activity. "
    "Power is loud at the sides and the back, so PCA picks those and is no better than random. Our relevance lights up over the motor strip, "
    "and six electrodes give 89 percent of the full-cap score, within 0.01 of the supervised selector.\"",
    "**Imaging** [imaging band]. \"Same engine with a 2-D encoder on fluorescence cubes. Nine of 192 lichen bands beat all 192; 30 of 158 collagen "
    "bands raise accuracy by six points. The picks spread across excitation lights, which is a direct design hint for a cheaper instrument.\"",
    "**Boundary** [Figure 3]. \"And the honest part: when channels are near-duplicates, as in wearable sensors or neighbouring bands of a remote-sensing "
    "image, nothing beats a random subset, not even the supervised method. A cheap label-using check tells you that in advance.\"",
    "**Close** [Conclusion, QR codes]. \"So: label-free, dependency-aware, real channels, matching a supervised selector where it matters. "
    "Paper and code are on the QR codes.\"",
])
H("5.3 The 5-minute full version", 2)
P("Use the 2-minute walk-through as the spine and add these expansions where the visitor shows interest.")
BULLETS([
    "**After the Problem:** \"Dependency-aware selectors exist, mRMR and JMI, but they need labels. Few methods are label-free, dependency-aware and return "
    "real channels at the same time. That is the gap.\"",
    "**After Method step 1:** \"The bottleneck matters. The code has no time or space extent, so it cannot memorise per-channel noise. Without that, "
    "the attribution in step 2 stops being a reliable importance signal on signal data.\"",
    "**After Method step 2 [Figure 2]:** \"Here it is on the synthetic data. Left: the latent coordinates ranked by variance. Middle: each row is one "
    "coordinate shifted, each column a channel, colour is how much the channel changed. Right: the sum, our relevance. The four clusters are "
    "high, the 44 noise channels are near zero, and the triangles are the eight picks.\"",
    "**After Synthetic:** \"PCA also works on this benchmark, and that is expected: here the informative channels are the loudest. The EEG is the "
    "case where that assumption breaks. Also, over repeated random draws of the benchmark the bottleneck sometimes under-encodes a factor, "
    "so the mean coverage is 0.70 with a wide spread; we say so on the poster.\"",
    "**After EEG:** \"Protocol: within subject, selection and CSP-LDA fitted on session one, tested on session two, standard for this dataset "
    "because motor-imagery patterns are subject-specific. We are at or above random for eight of nine subjects.\"",
    "**After Imaging:** \"Every excitation-emission pair is one image. Bands where the excitation light leaks in are removed, which is why the "
    "grid is ragged. The 9-band lichen selection takes one band under almost every excitation.\"",
    "**After Boundary:** \"The advantage grows with the fraction of uninformative channels, so these modest benchmarks are a conservative test. "
    "The method is aimed at much larger channel arrays.\"",
    "**Future work:** \"Tighten the bottleneck, make the attribution less sensitive to channel variance, and scale to larger arrays.\"",
])
H("5.4 Closing lines", 2)
SAY("Thanks for stopping by. The DOI on the QR code goes live when the proceedings appear on IEEE Xplore; the code is public now. "
    "If you have a multi-channel dataset with no labels, I would be glad to try this on it.")
PAGEBREAK()

# ================================================================ 6. SPEAKER NOTES
H("6. Speaker notes for each poster section", 1)
sections = [
    ("Title and authors", ["Read the title slowly once; \"coupled\" means the channels are dependent on each other.",
                           "Affiliations: AUA (college and research center), Orbeli Institute, GWU for Dr Sarvazyan."]),
    ("Introduction", ["Three sentences: many channels in groups; we want a few real ones; no labels.",
                      "Give one concrete example for the visitor's field if you can guess it (electrodes, bands, sensors)."]),
    ("Problem Statement", ["Key word: **marginal**. Variance and PCA score channels one at a time and prefer the loudest.",
                           "Supervised dependency-aware selectors exist but need labels. The gap is label-free + dependency-aware + real channels."]),
    ("Novelty", ["The three bold phrases are the three stages: compressive bottleneck, perturb each latent factor, relevance-redundancy rule.",
                 "\"Only the encoder changes between modalities\" is the portability claim."]),
    ("Method band, Figure 1", ["Walk left to right. Stage 1 boxes: encoder per group, mean fusion, latent z, decoders. Stage 2: shift, decode, measure change, accumulate. Stage 3: MMR shortlist.",
                               "The tiny bar chart in stage 3 is a real relevance vector; the coloured chips are the real first eight picks."]),
    ("Method band, Figure 2", ["Three panels = three sub-steps of attribution. Point to the noise block on the right of panel 3: near zero."]),
    ("Equations", ["Equation 1: encode per group, average, decode, reconstruction loss only.",
                   "Equation 2: relevance = sum over perturbed coordinates and magnitudes of the absolute reconstruction change, averaged over windows and time, weighted by the coordinate's variance rank.",
                   "Equation 3: greedy MMR; lambda trades relevance against redundancy; relevances are normalised within each group."]),
    ("Protocol line", ["Selection never sees labels; labels are used only by the downstream classifier that scores the selected K channels."]),
    ("Synthetic band", ["Numbers: 64 channels, 4 factors × 5, 44 noise, 16 classes, 600 windows. K = 8: 0.94 vs 0.87 full; supervised 0.99; PCA 0.95; variance 0.24; random 0.25.",
                        "Panel (c) is the most convincing picture: point to the noise picks of random and the single-cluster picks of variance."]),
    ("EEG band", ["Numbers: 22 electrodes, 4 regions, 4 classes, 9 subjects, 250 Hz, 8 to 30 Hz. K = 6: 0.517 vs 0.578 full (89 %); supervised 0.527; PCA 0.484; variance 0.466; random 0.486.",
                  "Topomaps: power is loud at the rim; relevance is on the motor strip; PCA picks C5, C6, POz.",
                  "8 of 9 subjects at or above random. Do not overclaim the margin over random."]),
    ("Imaging band", ["Lichens: 192 bands = 8 excitations; 4 classes; 9 bands 89.4 % vs 88.2 %. Collagen: 158 bands = 6 excitations; 3 classes; 30 bands 85.6 % vs 79.8 %.",
                      "\"Every excitation-emission pair is one image and one channel.\" Grey cells in the grid are Rayleigh-removed bands.",
                      "Selection spreads across excitations: instrument-design hint."]),
    ("Results band, Table 1", ["Read the Proposed row: 0.94 and 0.52; within 0.05 and 0.01 of supervised. All-channels row: 0.87 and 0.58."]),
    ("Results band, Figure 3", ["Left of the dashed line: gains of +0.69 to +0.74 on synthetic, +0.03 on EEG. Right: everything within ±0.03 of zero or negative, including supervised MI.",
                                "Indian Pines: random beats every selector; that is the extreme of redundancy."]),
    ("Conclusion", ["Three bullets: label-free real channels, matches supervised and beats PCA/variance where the signal is not the loudest, the check tells in advance when selection is pointless.",
                    "Next: tighter bottleneck, attribution less sensitive to variance, larger arrays."]),
    ("References and acknowledgments", ["Funding: EU NAS SAR-101087403 and US R44 HL120511. Thanks to Narek Chilingaryan and Kristina Ghahramanyan for the labelled samples.",
                                        "QR codes: paper DOI (live once on IEEE Xplore) and the GitHub repository."]),
]
for title, notes in sections:
    H(title, 3); BULLETS(notes)
PAGEBREAK()

# ================================================================ 7. Q&A
H("7. Likely questions and short answers", 1)
qa = [
    ("Why not just use PCA?", "PCA gives you a mixture of channels, not real channels, and its ranking is driven by variance. On EEG the informative rhythm is quieter than eye and alpha activity, so PCA picks the loud electrodes and is no better than random at small budgets. Our method models how channels depend on each other through the reconstruction, so it finds the quiet informative ones."),
    ("Where do the groups come from? Is that not a label?", "The groups are acquisition metadata: which scalp region an electrode sits in, which excitation light a band was recorded under, which body-worn unit an axis belongs to. They are known before any data is collected. The method also runs with a single group if no grouping exists."),
    ("How do you use labels at all, then?", "Only to score a selection afterwards. A simple classifier is trained on the selected channels with labels and compared against the same classifier on channels chosen by the baselines. Selection itself never sees a label."),
    ("What is the supervised MI selector?", "For each channel we compute the mutual information between a summary feature of that channel and the class label, and keep the top K. It uses labels, so it is our upper reference, not a competitor."),
    ("Why does the synthetic score beat the full set?", "The full set includes 44 pure-noise channels that hurt a k-nearest-neighbour classifier. Dropping them helps. The same happens on the fluorescence images: removing uninformative bands also removes their noise."),
    ("Your EEG gain over random is only 0.03. Is that significant?", "The margin over random is small and subject variation is large; we say so. The claim is that six label-free electrodes match the supervised selector within 0.01 while PCA and variance collapse to random. We are at or above random for eight of nine subjects."),
    ("Why does PCA work on the synthetic data?", "Because there the informative clusters are the highest-variance channels by construction. That is exactly the assumption PCA relies on, and the EEG shows what happens when it breaks."),
    ("What does perturbing the latent code actually measure?", "Sensitivity. If shifting a latent coordinate changes a channel's reconstruction a lot, the shared representation depends on that channel. Channels that never react are ones the representation ignores, which is what noise looks like."),
    ("Why the bottleneck? Why not a normal autoencoder?", "A code with time or space extent can memorise per-channel noise, and then the perturbation attribution stops being a reliable importance signal on signal data. Pooling the regular axis away forces the code to hold the shared structure."),
    ("What is MMR and why do you need it?", "Maximal marginal relevance: each next pick maximises relevance minus lambda times its cosine similarity to the channels already picked. Without it, the method would pick five near-identical channels from the loudest cluster."),
    ("How do you choose K?", "The output is an ordered list, so K is a budget the user sets: number of electrodes, number of filters. On the poster we show the whole curve against K, and the operating points we quote (8, 6, 9, 30) are where the curve reaches the full-set score."),
    ("Does it work on wearable sensors?", "Not better than random, and neither does anything else, including the supervised selector. Accelerometers on the same body are near-duplicates. Our label-using check flags that in advance, which we consider a useful result."),
    ("What about the Indian Pines result, where random beats you?", "Adjacent reflectance bands are almost perfect copies, so any 20 bands carry the whole spectrum, and a random spread of bands happens to sample it well. Our selector, and PCA, chase structure that is not there. It is the extreme of redundancy."),
    ("How long does it take to run?", "The autoencoder is small: seconds to minutes on a laptop GPU for the EEG and synthetic data; the imaging sweeps take longer because they train on patches, but a single configuration is minutes."),
    ("Is the code available?", "Yes, the GitHub repository on the QR code. The engine, the adapters for each dataset, and the figure scripts are all there."),
    ("What is next?", "Tighten the bottleneck so a factor is never under-encoded, make the attribution less sensitive to raw channel variance, and scale to much larger channel arrays, where the advantage should grow."),
]
for q, a in qa:
    par = doc.add_paragraph(); r = par.add_run("Q: " + q); r.bold = True; r.font.color.rgb = NAVY; par.paragraph_format.space_after = Pt(1)
    par.paragraph_format.keep_with_next = True
    par = doc.add_paragraph(); _runs(par, "A: " + a); par.paragraph_format.left_indent = Cm(0.6); par.paragraph_format.space_after = Pt(7)
PAGEBREAK()

# ================================================================ 8. GLOSSARY
H("8. Glossary in plain English", 1)
TABLE([
    ["Term", "Plain meaning"],
    ["Channel", "One recorded signal: an electrode, one sensor axis, one image band."],
    ["Group", "A set of channels that come from the same physical source: one scalp region, one excitation light, one motion unit."],
    ["Label", "The class of a sample (imagined movement, lichen species, cross-linker concentration). Our selection never uses them."],
    ["Label-free / unsupervised", "Works without labels."],
    ["Dependency-aware", "Takes into account how channels relate to each other, instead of scoring each one alone."],
    ["Marginal method", "Scores each channel on its own statistics (variance, loadings)."],
    ["Autoencoder", "A network that compresses data into a short code and rebuilds the data from that code."],
    ["Bottleneck / latent code z", "The short code in the middle of the autoencoder; here it has no time or space extent."],
    ["Perturbation / attribution", "Shifting the code and measuring which channels change; assigning importance from that."],
    ["Relevance", "The label-free importance score of a channel."],
    ["MMR (maximal marginal relevance)", "Greedy rule: pick high relevance, low similarity to what is already picked."],
    ["Budget K", "How many channels to keep."],
    ["Supervised MI selector", "Ranks channels by mutual information with the label; our upper reference."],
    ["PCA loadings", "How much each channel contributes to the main principal components."],
    ["Random-K", "A random subset of K channels; the floor."],
    ["kNN", "k-nearest neighbours; classifies a sample by the majority class of its nearest training samples."],
    ["CSP + LDA", "The standard EEG motor-imagery pipeline: spatial filters that maximise class contrast, then a linear classifier."],
    ["Macro-F1", "Average of per-class F1 scores; fair when classes are unbalanced."],
    ["Within-subject", "Train and test on the same person, on different sessions."],
    ["LOSO", "Leave one subject out: train on all subjects but one, test on the left-out one."],
    ["Excitation / emission", "The light we shine on the sample / the light the sample gives back."],
    ["Multi-excitation hyperspectral imaging", "A full emission spectrum at every pixel, repeated under several excitation lights."],
    ["Rayleigh scatter", "Excitation light leaking into the camera; those bands are removed."],
    ["Headroom check", "Best-K under a supervised ranking minus random-K; near zero means selection cannot help."],
], col_widths=[2.0, 4.7], font=9.5)
PAGEBREAK()

# ================================================================ APPENDIX A
H("Appendix A. Every poster visual, what it shows, what to say", 1)
visuals = [
    ("fig_pipeline.png", "Figure 1: the three-stage pipeline", "Real intermediate outputs: input tiles, latent bar chart, per-channel change, relevance vector, first eight picks.",
     "Walk it left to right: encode per group, average, decode; shift the code and measure the change; accumulate and pick a diverse shortlist."),
    ("fig_attribution.png", "Figure 2: perturbation attribution", "(1) latent coordinates by variance; (2) per-channel change per coordinate; (3) accumulated relevance with the eight picks.",
     "Point to the near-zero noise block in panel 3: the method sees the 44 noise channels as irrelevant without any label."),
    ("eq1.png", "Equation 1", "Encode each group, average into z, decode per group, reconstruction loss only.", "\"Trained on reconstruction only, no labels.\""),
    ("eq2.png", "Equation 2", "Relevance of channel c in group g: accumulated absolute reconstruction change over perturbed coordinates j and magnitudes delta, averaged over windows and time, weighted by variance rank.",
     "\"A channel whose reconstruction reacts is one the joint representation depends on.\""),
    ("eq3.png", "Equation 3", "Greedy MMR pick: relevance minus lambda times the maximum cosine similarity to already-picked channels; relevance normalised within group.",
     "\"High relevance, low redundancy; the list is ordered so you can cut it at any K.\""),
    ("fig_syn_dataset.png", "Synthetic (a): one window and the correlation matrix", "Channels sorted by role: four correlated clusters, 44 noise rows; four dark diagonal blocks in the correlation matrix.",
     "\"Four hidden factors, each in five near-copy channels, plus noise. You must cover all four to classify.\""),
    ("fig_syn_curve.png", "Synthetic (b): accuracy vs K", "kNN accuracy for five selectors; dashed line = all 64 channels (0.87); shaded K = 8.",
     "\"Eight channels: 0.94, above the full set; supervised 0.99; PCA 0.95; variance 0.24; random 0.25.\""),
    ("fig_syn_selection.png", "Synthetic (c): picks at K = 8", "Black bars are picks; background colour is the channel's role; right margin counts factors covered and noise picked.",
     "\"Ours: all four factors, no noise. Variance: two factors. Random: four noise channels.\""),
    ("fig_eeg_montage.png", "EEG (a): montage", "22 electrodes coloured by the four region groups.", "\"The grouping comes from the cap layout; it is metadata, not labels.\""),
    ("fig_eeg_topomaps.png", "EEG (b): power, relevance, picks", "Signal power (what PCA and variance rank by); our relevance; electrodes picked by ours and by PCA, darker = more of the 9 subjects.",
     "\"Power is loud at the rim; relevance is on the motor strip; PCA picks C5, C6, POz.\""),
    ("fig_eeg_curve.png", "EEG (c): macro-F1 vs K", "Mean over nine subjects; dashed line = all 22 electrodes (0.58); shaded K = 6.",
     "\"Six electrodes: 0.52, 89 percent of the full cap, within 0.01 of supervised; PCA and variance at random level.\""),
    ("fig_hsi_dataset.png", "Imaging (a): the two cubes", "Photo / false colour, class map, one selected band image, for lichens and collagen.",
     "\"Every excitation-emission pair is one image and one channel: 192 for lichens, 158 for collagen.\""),
    ("fig_hsi_eem.png", "Imaging (b): where the picks sit", "Excitation × emission grids; white dots are the selected bands; grey cells are Rayleigh-removed.",
     "\"The picks spread across excitations rather than stacking on the brightest region.\""),
    ("fig_hsi_bars.png", "Imaging (c): accuracy bars", "All bands vs selected subset for both datasets, with band counts on the bars.",
     "\"9 of 192: 89.4 vs 88.2. 30 of 158: 85.6 vs 79.8. Pruning also denoises.\""),
    ("fig_boundary_doc.png", "Figure 3: gain over random (shown here in two rows)", "Score minus random-K for ours, PCA and supervised MI on five datasets; on the poster the two rows sit side by side with a dashed line between the regimes.",
     "\"Left: selection pays. Right: near-duplicate channels, nothing beats random, not even supervised; the label-using check flags it in advance.\""),
    ("qr_paper.png", "QR: paper", "DOI 10.1109/CODASSCA69992.2026.00059 (resolves once the proceedings are on IEEE Xplore).", "\"The paper is on this QR; it goes live with the proceedings.\""),
    ("qr_code.png", "QR: code", "github.com/Nard248/spectral-select", "\"Engine, dataset adapters and figure scripts are all public.\""),
]
for fname, title, shows, say in visuals:
    H(title, 3)
    p = FIG / fname
    iw, ih = Image.open(p).size
    w = 6.7 if iw / ih > 2.2 else (4.8 if iw / ih > 1.2 else 3.0)
    if fname.startswith("eq"): w = 6.0
    if fname.startswith("qr"): w = 1.4
    FIGURE(p, f"{fname}", width_in=w, max_h_in=4.5)
    P(f"**Shows:** {shows}", space_after=2)
    SAY(say)
PAGEBREAK()

# ================================================================ APPENDIX B
H("Appendix B. Extra visuals not on the poster", 1)
P("Useful on a tablet or as backup if someone asks for more detail on the imaging datasets. All come from the thesis / journal-paper figure sets.")
extras = [
    ("lichens_labels", "Lichens: the 16 specimens with their class labels and the four training boxes (ROIs).",
     "\"Sixteen pieces of four lichen species, laid out by class. The small squares are the only pixels the classifier trains on; everything else is scored.\""),
    ("lichens_maps", "Lichens: per-pixel classification maps with all 192 bands, with the best 80 bands, and with 9 bands.",
     "\"Same picture with 192, 80 and 9 bands. The 9-band map is at least as clean as the full one.\""),
    ("lichens_heatmap", "Lichens: wavelength importance map across the full sweep (mean normalised rank of each band across above-baseline configurations).",
     "\"Bright cells are bands that rank high in almost every good configuration. The importance is spread over several excitations.\""),
    ("collagen_roi", "Collagen sponges: the nine specimens, class layout and training boxes.",
     "\"Three concentrations in three columns; the boxes in the first row are the training pixels.\""),
    ("collagen_heatmap", "Collagen sponges: wavelength importance map.",
     "\"The hot region is emission 600 to 690 nm under 365 to 400 nm excitation, where cross-link fluorescence sits.\""),
    ("collagen_classifiers", "Collagen sponges: accuracy against K for ten classifier families on the same selected bands.",
     "\"Nine of ten classifiers benefit from the selection, so the bands are informative in themselves, not tuned to kNN.\""),
]
for key, shows, say in extras:
    p = EXTRA[key]
    if not p.exists():
        continue
    H(shows.split(".")[0], 3)
    iw, ih = Image.open(p).size
    FIGURE(p, p.name, width_in=6.7 if iw / ih > 1.6 else 4.8, max_h_in=4.6)
    P(f"**Shows:** {shows}", space_after=2); SAY(say)
PAGEBREAK()

# ================================================================ APPENDIX C
H("Appendix C. Where the data and figure files live", 1)
TABLE([
    ["Item", "Location (repository root = 4D-Hyperspectral-Unsupervised-Clustering)"],
    ["Poster (editable)", "publications/codassca2026/poster/CODASSCA2026_poster_A0.pptx; builder build_poster.py"],
    ["Poster figures and scripts", "publications/codassca2026/poster/figures/ and scripts/ (one script per row: fig_synthetic.py, fig_eeg.py, fig_hsi.py, fig_pipeline.py, fig_attribution.py, fig_boundary.py)"],
    ["Synthetic generator and run", "src/channel_select/synthetic.py; experiments/synthetic/run_synthetic_recovery.py; dump for figures experiments/synthetic/dump_synthetic_attribution.py"],
    ["EEG adapter and runs", "src/channel_select/adapters/bci_eeg.py (MOABB BNCI2014_001); experiments/eeg/run_bci_loso.py and dump_bci_selections.py; per-subject selections in poster/data/eeg_selections.json"],
    ["Lichens data", "Data/processed/Lichens Dataset 1/ (spectra_masked.pkl, class_mask.png, roi_regions.json); results in results/Lichens_Dataset_1_MasterRun/"],
    ["Collagen data", "Data/processed/Collagen Pepsin/ (same layout); results in results/Collagen_Pepsin_Normalized/ and results/Pepsin_Paper_Figures/"],
    ["PAMAP2 / Opportunity / Indian Pines", "Data/Raw/PAMAP2_MONSTER/; Hugging Face monster-monash/Opportunity and danaroth/indian_pines (cached); screens experiments/opportunity_screen.py, hsi_benchmark_screen.py, pamap2/*.py"],
    ["Reports with the numbers", "publications/generalization/reports/*.txt; poster/data/*.log"],
    ["Camera-ready paper", "publications/generalization/paper/main.tex (paper 122, DOI 10.1109/CODASSCA69992.2026.00059)"],
], col_widths=[1.8, 4.9], font=9.5)

doc.save(OUT)
print("wrote", OUT)
