#!/usr/bin/env python3
"""Build the plain-English Word version of the RL / group-importance brainstorm.

Usage:
    python docs/research/build_rl_brainstorm_docx.py [output.docx]

The Markdown note (2026-09-11-rl-and-group-importance-brainstorm.md) is the compact,
technical version; this document explains the same ideas in natural English with
more detail, for the whole team. New .docx files are git-ignored, so this script is
what gets versioned; rebuild the document from it.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
    "2026-09-11-rl-and-group-importance-brainstorm.docx"
)

doc = Document()

# ---------------------------------------------------------------- page + fonts
for section in doc.sections:
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(11)
styles["Normal"].paragraph_format.space_after = Pt(6)
styles["Normal"].paragraph_format.line_spacing = 1.12
for lvl, size in ((1, 18), (2, 14), (3, 12)):
    st = styles[f"Heading {lvl}"]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    st.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    st.paragraph_format.space_after = Pt(4)


def _shade(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


# ---------------------------------------------------------------- helpers
def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)
def h3(t): doc.add_heading(t, level=3)


def p(t, bold_lead: str | None = None, italic: bool = False):
    para = doc.add_paragraph()
    if bold_lead:
        r = para.add_run(bold_lead + " ")
        r.bold = True
    r = para.add_run(t)
    r.italic = italic
    return para


def bullets(items, style="List Bullet"):
    for it in items:
        if isinstance(it, tuple):
            para = doc.add_paragraph(style=style)
            r = para.add_run(it[0] + " ")
            r.bold = True
            para.add_run(it[1])
        else:
            doc.add_paragraph(it, style=style)


def numbered(items):
    bullets(items, style="List Number")


def box(title: str, text: str, fill: str = "EAF1FB"):
    """A one-cell shaded table used as a call-out box."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    _shade(cell, fill)
    cell.paragraphs[0].add_run(title).bold = True
    for line in text.strip().split("\n"):
        cell.add_paragraph(line)
    doc.add_paragraph()


def table(headers, rows, widths_cm=None):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Light Grid Accent 1"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = tbl.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].add_run(h).bold = True
    for row in rows:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    if widths_cm:
        for row in tbl.rows:
            for i, w in enumerate(widths_cm):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()


# ================================================================ TITLE PAGE
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("New Research Directions for Wavelength Selection")
r.bold = True
r.font.size = Pt(26)
r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s.add_run("Reinforcement learning, wavelength combinations, and other ideas, explained in plain English")
r.font.size = Pt(14)
r.italic = True

m = doc.add_paragraph()
m.alignment = WD_ALIGN_PARAGRAPH.CENTER
m.add_run(f"spectral-select project  |  {date(2026, 9, 11).strftime('%d %B %Y')}").font.size = Pt(11)
m2 = doc.add_paragraph()
m2.alignment = WD_ALIGN_PARAGRAPH.CENTER
m2.add_run("Companion to docs/research/2026-09-11-rl-and-group-importance-brainstorm.md").font.size = Pt(9)

doc.add_paragraph()
box(
    "How to read this document",
    "Every idea is explained in three parts: what it is, why it could help us, and how we would actually do it "
    "with the data and code we already have. Technical words are explained the first time they appear, and there "
    "is a glossary at the end. Nothing here is a decision. It is a menu of things to try, with a suggested order "
    "in Section 7 and an honest list of risks in Section 8.",
)

p("Contents", bold_lead=None).runs[0].bold = True
for line in [
    "1. Where we are today",
    "2. A short introduction to reinforcement learning",
    "3. Part 1: Reinforcement-learning ideas (A to E)",
    "4. Part 2: The importance of wavelength combinations (ideas 1 to 7)",
    "5. Part 3: Broader ideas (3.1 to 3.7)",
    "6. What each idea gives us, at a glance",
    "7. Suggested order of work",
    "8. Risks and honest expectations",
    "9. Glossary",
    "Appendix: where things are in the repository",
]:
    doc.add_paragraph(line, style="List Bullet")

doc.add_page_break()

# ================================================================ 1. WHERE WE ARE
h1("1. Where we are today")

h2("1.1 The problem we are solving")
p("Our instrument records, for every pixel of an image, how brightly the sample fluoresces at each of 31 emission "
  "wavelengths, and it repeats that for eight different excitation wavelengths. After removing the parts of the "
  "spectrum that are contaminated by scattered excitation light, about 192 usable excitation-emission pairs remain "
  "per pixel. We call each pair a wavelength band, or simply a band.")
p("Most of those 192 numbers say nearly the same thing. Neighbouring emission wavelengths look alike, and several "
  "excitations light up the same molecules. The scientific and practical question of the paper is: which small "
  "set of bands carries the information that distinguishes the materials in the sample? If we know that, a cheaper "
  "instrument with a handful of light sources and filters could do the job, acquisition would be faster, and any "
  "classifier working on the data would see less noise.")
p("The important constraint is that we want to find those bands without using labels. Labelled pixels (pixels where "
  "somebody has written down which material they are) are scarce and expensive in biomedical imaging, so a method "
  "that needs them is much less useful. Labels are only used at the very end, to check whether the selection was good.")

h2("1.2 How the current method works, in four steps")
p("The method in the paper is a pipeline of four steps. Here they are in everyday language.")
numbered([
    ("Compress the data with an autoencoder.",
     "An autoencoder is a neural network with a narrow waist. It must squeeze the 192 numbers of every pixel "
     "(together with the pixel's neighbourhood) into a short summary of 20 numbers, and then rebuild the original "
     "192 numbers from that summary. It is trained by being punished for rebuilding errors. To rebuild well from "
     "so little, it is forced to learn what really varies in the data and to ignore what is redundant or noisy. Our "
     "autoencoder has one branch per excitation, because each excitation has a different number of usable bands, "
     "and the branches are averaged into the shared 20-number summary."),
    ("Nudge the summary and watch what changes.",
     "Once the autoencoder is trained, we take the 20-number summary of the data, increase one of the 20 numbers a "
     "little, decrease it a little, and rebuild the image each time. Some bands barely change; others change a lot. "
     "A band that reacts strongly to a nudge is a band the compressor relies on to describe the data, and we count "
     "that as evidence that it is informative. We do this for the most important summary numbers (chosen by how much "
     "they vary across the image), with several nudge sizes, and add everything up into one influence score per band."),
    ("Rank the bands by influence.",
     "This gives a list of all 192 bands from most to least influential."),
    ("Pick a spread-out subset.",
     "Neighbouring bands tend to have similar influence, and picking three neighbours wastes two of the slots we "
     "have. A rule borrowed from search engines, called maximal marginal relevance (MMR), adds bands one at a time, "
     "each time preferring a band that is influential but not too similar to the ones already chosen."),
])
p("The paper calls step 2 perturbation-based attribution. The word perturbation simply means nudge, and attribution "
  "means giving credit to the bands.")

h2("1.3 How we know it works")
p("To measure whether a selected set of bands is good, we use a deliberately simple classifier called k-nearest "
  "neighbours (KNN). It labels a pixel by looking at the five labelled pixels whose band values are most similar to "
  "it and taking a majority vote. We use it because it has no knobs to tune, so the result reflects the bands, not "
  "the classifier.")
p("On the lichen dataset, using all 192 bands gives 88.2 percent accuracy. Using the 80 bands our method chose gives "
  "95.2 percent; using only 9 bands still gives 89.4 percent. So dropping the right bands removes noise and helps. "
  "The strongest evidence that the method finds something real is this: we drew 10 000 random sets of 13 bands and "
  "scored them all; they averaged 46 percent and the best of them reached 58 percent, while our 13-band set scored "
  "90 percent, better than every random draw. On the collagen dataset the same pipeline, unchanged, also beats the "
  "all-bands result, and on the droplet dataset, where nobody has labels at all, it picks bands exactly where the "
  "three types of droplets differ and stays silent at the two excitations where they do not.")

h2("1.4 What the method cannot do yet")
p("Reading the paper and the reviewers' comments together, these are the honest gaps:")
bullets([
    ("It scores bands one at a time.",
     "The influence score is a property of a single band. The method never asks whether two or three bands are "
     "informative together, or whether they say the same thing. The spread-out rule (MMR) only avoids look-alikes "
     "after the fact, using how similar their spectra look, not how they behave together."),
    ("The number of bands, K, is chosen by hand.",
     "On the lichen dataset we chose the headline K by looking at labelled accuracy, which a reviewer rightly noted "
     "weakens the claim of being fully unsupervised."),
    ("It is slow.",
     "About three quarters of the two-hour pipeline is the nudging step, because it is done by brute force: rebuild "
     "the image again and again for every summary number and nudge size."),
    ("It is a chain of separate tricks rather than one objective.",
     "The reviewers asked for a single principle that ties compression, nudging and picking together. Our planned "
     "answer is that compression keeps the information, nudging reads out how much each band contributes to it, and "
     "the spread-out rule avoids counting the same information twice. Several ideas below turn that answer into code."),
    ("It says which bands, not why.",
     "The scores do not tell us which molecule (collagen, NADH, chlorophyll, and so on) a band is reading."),
    ("It gives no uncertainty.",
     "We cannot say how confident we are that band 47 belongs in the top ten."),
])

h2("1.5 Two hard lessons from this summer")
p("Two findings from the last months shape everything that follows, so they deserve their own paragraphs.")
p("The spatial autoencoder can cheat.", bold_lead=None).runs[0].bold = True
p("When we built SpectraForge, our simulator that paints known fluorescent molecules onto a canvas and renders the "
  "cube our instrument would record, we found that on those synthetic cubes the autoencoder converges to a lazy "
  "solution: it predicts roughly the average value of each band everywhere. The rebuilt image is nearly "
  "uncorrelated with the input, yet the training error is small, because most of the synthetic cube is empty. A "
  "compressor that has learned nothing produces nudging scores that are noise; on those cubes our selection was "
  "worse than random. A simpler autoencoder that works one pixel at a time on the spectrum, without the spatial "
  "convolutions, does not cheat and recovers the informative bands. Real data has so far avoided the problem because "
  "every band carries signal everywhere, but the lesson is that new objectives must be tested on both kinds of "
  "autoencoder, and that the per-pixel spectral one is the safer backbone for anything trained on synthetic data.")
p("On some datasets, nothing beats random.", bold_lead=None).runs[0].bold = True
p("When we generalised the method to wearable-sensor and EEG data, and when we screened public hyperspectral "
  "benchmarks, we learned that many datasets are saturated: their channels are so redundant that any ten of them "
  "work as well as the best ten, and even a supervised selector that uses the labels cannot beat a random pick. "
  "The droplet dataset is like that too. On such data no method can look good, and the honest conclusion is a "
  "dataset property, not a method weakness. The lichen dataset is the opposite: random 13-band sets score 46 percent "
  "and the right set scores 90, so there is enormous room for a clever method to show its worth. Every experiment "
  "below should therefore start on lichens (and collagen), and every dataset should first pass our saturation "
  "diagnostic, which measures the gap between the best labelled selection and a random one.")

doc.add_page_break()

# ================================================================ 2. RL INTRO
h1("2. A short introduction to reinforcement learning")
p("Reinforcement learning (RL) is the branch of machine learning where a program learns by trial and error, the way "
  "a player learns a game. The vocabulary is small:")
bullets([
    ("Agent.", "The program that makes decisions."),
    ("State.", "What the agent can see at a given moment; the current situation of the game."),
    ("Action.", "One of the moves the agent may choose in that state."),
    ("Reward.", "A number the environment hands back after an action, or at the end of the game, saying how well "
               "things went. The agent's whole purpose is to collect as much reward as possible."),
    ("Episode.", "One complete game, from the first move to the end."),
    ("Policy.", "The agent's strategy: a rule, usually a neural network, that turns a state into a choice of action. "
               "Training means adjusting the policy so that actions which led to high reward become more likely."),
])
p("Translated to our problem: the state is the set of bands chosen so far, an action is adding one more band, an "
  "episode is building one complete set of K bands, and the reward is a score of how good that set is. The agent "
  "plays this game thousands of times and gradually learns which bands to add after which.")
p("This sounds attractive, but there is a trap that we need to name before listing ideas. If the reward is just a "
  "score computed on one fixed image, then RL is nothing more than a search procedure, and older search procedures "
  "such as genetic algorithms (which breed and mutate candidate band sets) or simulated annealing (which makes random "
  "changes and keeps the good ones) will find equally good sets with the same computing budget. RL is worth the extra "
  "complexity only in three situations:")
numbered([
    "The reward is cheap to compute but cannot be differentiated, so ordinary neural-network training cannot use it "
    "directly. Classification accuracy, clustering quality, hardware costs and integer constraints are all like that.",
    "The decisions are genuinely sequential, meaning each choice changes what can be observed next. Choosing what to "
    "measure next on the instrument is exactly that.",
    "The policy is trained once and then reused on new samples without retraining. This is called amortisation, and "
    "it is where RL can remove the cost that the reviewers complained about.",
])
p("There is also a differentiable cousin worth knowing about, because it is the natural competitor to RL for our "
  "problem: the Concrete Autoencoder. It puts a soft switch on every input band, lets the switches be partly on "
  "during training so that gradients can flow, and slowly forces them to become fully on or off. Our own research "
  "log already names it as a mandatory baseline. Any RL selector we build must be compared with it.")

doc.add_page_break()

# ================================================================ 3. PART 1 RL IDEAS
h1("3. Part 1: Reinforcement-learning ideas")

# ---- Idea A
h2("Idea A. A trained band selector that works on new samples without retraining")
h3("What it is")
p("We turn band selection into a game the agent plays on many images, and we train one policy that, given a new "
  "image of the same kind, chooses a good set of bands directly, one band per step, without running the two-hour "
  "nudging pipeline again.")
h3("The game in detail")
bullets([
    ("State.", "A list of all 192 bands, each with a few numbers describing it: its excitation and emission "
              "wavelengths, its average brightness and its variability in this image, the influence score from our "
              "existing method (a useful hint), and a flag saying whether it has already been chosen."),
    ("Action.", "Pick one band that is not yet chosen."),
    ("Episode length.", "K steps, for example ten."),
    ("Reward.", "Given at the end of the episode: a score of the whole set. Optionally, a small reward after each "
               "step equal to how much the score improved, which helps the agent learn faster."),
])
h3("The reward without labels: 'can we guess the summary from the chosen bands?'")
p("This is the key design choice, because we want selection to stay unsupervised. The trick is to use the "
  "autoencoder we already train. Its 20-number summary of each pixel is, by construction, a compact version of "
  "everything the data contains. So we can ask: if I only give you the values of the chosen bands for each pixel, "
  "how well can you predict the pixel's 20-number summary? If a small set of bands lets you predict the summary "
  "almost perfectly, then that set carries the information; if it does not, something important is missing.")
p("Computing this is cheap. We take a few thousand masked pixels, precompute their summaries once, and for any "
  "candidate set of bands fit a simple linear predictor from the chosen band values to the summary. The quality of "
  "that prediction, a number between 0 and 1 known as R-squared, is the reward. A closed-form formula gives it in "
  "well under a millisecond, so the agent can play millions of games. Nicely, this is precisely the theoretical "
  "story we are writing for the reviewers: the summary holds the information, and the best set of bands is the one "
  "from which the summary can be recovered.")
p("A second unsupervised reward is possible: how well can all 192 bands be rebuilt from the chosen ones (again with "
  "a simple linear model, or with the autoencoder's own decoder). This is the same objective the Concrete "
  "Autoencoder optimises, just evaluated on whole sets rather than through soft switches.")
p("When labels exist, we can swap in the KNN accuracy on the labelled pixels as the reward. That makes the agent a "
  "supervised method, which is fine as long as we say so; it should not be presented as the unsupervised result.")
h3("The agent's brain")
p("The policy is a small neural network that looks at all 192 band descriptions at once, including which ones are "
  "already chosen, and outputs a probability for each unchosen band. Because it sees what has already been picked, "
  "it can learn interactions: after choosing a band at 365 nm excitation and 480 nm emission, it might learn to prefer "
  "a band under a different excitation rather than the neighbour at 490 nm. Our greedy spread-out rule cannot learn "
  "anything of the sort; it only knows that neighbours look alike.")
h3("Training")
p("We use a standard policy-training algorithm (PPO, or the simpler REINFORCE). In plain words: play many episodes, "
  "compare each episode's reward with the average, and make the actions of above-average episodes a little more "
  "likely and those of below-average episodes a little less likely. The crucial ingredient is variety: each episode "
  "is played on a different random crop of the image, a different random subset of pixels, or a synthetic scene "
  "from SpectraForge. Otherwise the agent memorises one image instead of learning a strategy.")
h3("What we gain")
p("Once trained on, say, crops of the small lichen image, the policy can select bands for the large lichen image in "
  "a few forward passes of a small network. No nudging sweep, no ninety minutes. That answers the cost objection "
  "directly, and it is a capability our current method simply does not have.")
h3("How to test it fairly")
bullets([
    "Against our current greedy method at the same K.",
    "Against the Concrete Autoencoder.",
    "Against a genetic algorithm given the same number of reward evaluations. If the agent does not beat it on a fixed "
    "image, the only honest advantage left is the transfer to new images, so the experiment must measure that.",
    "Against the exhaustive best set at K equals 2 and 3, which Section 4 shows we can actually compute.",
    "Against 10 000 random sets, as in the paper.",
])
h3("What could go wrong")
p("The linear predictor might miss information that only a nonlinear model would see; that is why the exhaustive "
  "KNN check at small K matters. The agent might overfit to the training images; that is why variety in episodes "
  "matters. And on a saturated dataset the reward landscape is flat and nothing can be learned; that is why the "
  "saturation diagnostic comes first.")

# ---- Idea B
h2("Idea B. An agent that decides what to measure next on the instrument")
h3("What it is")
p("This is the idea that is unique to our instrument. Today every acquisition follows a fixed recipe: eight "
  "excitations, 31 emission bands each, fixed exposure times. But the hardware is inherently sequential: the light "
  "source steps from one excitation to the next, and the tunable filter steps through the emission bands, one "
  "exposure at a time. Nothing forces the recipe to be fixed. An agent could decide, as the measurement proceeds, "
  "what to record next and when to stop.")
h3("The game in detail")
bullets([
    ("State.", "Everything measured so far for the current sample (a partly filled excitation-emission table, per "
              "pixel or per region), plus the current belief of a classifier or clustering model about what the "
              "sample contains."),
    ("Action.", "Measure a particular excitation-emission band; or measure a whole excitation at a chosen exposure "
               "time; or stop."),
    ("Reward.", "A payment for getting the answer right at the end (or, without labels, for how confident and stable "
               "the clustering has become), minus a cost for every measurement taken. The cost can include acquisition "
               "time, the light dose delivered to a photosensitive sample, and a penalty when the chosen exposure "
               "saturates the detector."),
])
h3("Why this is interesting")
p("On the droplet dataset our method selects nothing at the 310 and 340 nm excitations because they carry no "
  "discriminative information. A measuring agent would learn to skip them entirely and spend the saved time on "
  "exposure brackets at the excitations that matter. The exposure choice is important in its own right: the first "
  "droplet analysis was wrong because exposure brackets and lamp power were ignored, and including a saturation "
  "penalty in the reward turns that painful lesson into a rule the agent follows automatically.")
p("The result is a dynamic selection: different specimens get different measurement sequences, and the recipe is "
  "as short as it can be for the question being asked. No current band-selection paper tells that story, and it fits "
  "our grant's device objective, where acquisition speed and light dose are explicit goals.")
h3("How we would build it")
p("Training in the real world is impossible, because every episode would require a new acquisition. But we have two "
  "substitutes. SpectraForge can simulate any acquisition order and any exposure, so the agent can be trained on "
  "thousands of synthetic samples. And for the real cubes we already recorded every band, so the agent can be tested "
  "offline: it simply reveals the recorded bands in the order it chooses, and we score the result as if it had "
  "measured only those. The same offline replay works for exposure brackets on the droplet data, where several "
  "exposures per excitation were recorded.")
h3("Challenges")
p("The agent needs a model in the loop that turns partial measurements into a belief (a classifier when labels exist, "
  "a clustering model otherwise), and the unsupervised reward must be a per-pixel quantity, because we learned on the "
  "droplet data that cluster-level scores can be gamed. The simulator-to-reality gap is real, and we must be careful "
  "not to repeat the earlier SpectraForge mistake of judging success with a metric that a random policy also passes; "
  "a random measuring policy and a fixed full-recipe policy are the mandatory baselines.")

# ---- Idea C
h2("Idea C. Sharing a band budget between excitations")
p("From an instrument designer's point of view, excitations are the expensive dimension: each one is a lamp line or an "
  "LED. Emission bands are cheap, because the tunable filter can select them at will. So the practical question is "
  "often: how many bands should each excitation get, and which ones?")
p("This is a two-level decision. First decide how to split a budget of K bands among the eight excitations; then, "
  "within each excitation, decide which emission bands to take. It can be set up as one agent making the first "
  "decision and a second agent making the second, or as eight small agents, one per encoder branch, bidding for a "
  "shared budget. The reward is the same as in Idea A. The output is exactly what a designer needs, for example "
  "three excitations with four filters each, and the per-excitation budgets are themselves a clean statement of "
  "which excitations matter, which connects to the group-importance ideas in Part 2.")

# ---- Idea D
h2("Idea D. Learning the nudging settings themselves (low priority)")
p("Our paper swept 3 072 configurations of the nudging step: which summary numbers to nudge, how hard, how to "
  "normalise the results. An agent, or more simply a bandit algorithm, could learn which settings produce the most "
  "useful influence maps, using the unsupervised reward from Idea A as feedback. We list it for completeness, but it "
  "is really a hyperparameter search, and Bayesian optimisation would probably do it with fewer evaluations. Do this "
  "only if the cheaper tools fail.")

# ---- Idea E
h2("Idea E. Searching for formulas, not only bands")
p("In remote sensing, the most useful features are often not single bands but simple formulas of two bands, such as "
  "the vegetation index that divides the difference of two bands by their sum. Formulas like ratios cancel out "
  "things we do not care about, such as how bright the lamp was or how thick the sample is, and keep what we do care "
  "about, such as the relative amount of two molecules. Our grant's objective O4 asks precisely for such pair "
  "combinations and the formulas that combine them.")
p("An agent can search for formulas: it builds small expression trees whose leaves are bands from our shortlist and "
  "whose branches are plus, minus, times and divide, and it is rewarded by how well the resulting feature separates "
  "the materials (or predicts the summary, in the unsupervised version). Monte Carlo tree search, the technique behind "
  "game-playing programs, fits this well. Genetic programming is the classic alternative and is a fair baseline. "
  "Starting from the top twenty bands of our ranking keeps the search space small enough to explore thoroughly.")

doc.add_page_break()

# ================================================================ 4. PART 2 COMBINATIONS
h1("4. Part 2: The importance of wavelength combinations")

h2("4.1 Why single scores are not enough")
p("Think of a football team. A list of the eleven best players in the world is not the best team, because a team "
  "needs different roles and because two brilliant strikers may get in each other's way. Bands behave the same way. "
  "Two bands can each be mediocre on their own and excellent together: for example, when the ratio of the two "
  "cancels a brightness variation that neither can escape alone. And two bands can each be excellent on their own "
  "and useless together, because they say the same thing and the second one adds nothing. Our current score sees "
  "only individuals.")
p("The question you asked, 'show the importance when three bands are selected together', is a question about sets. "
  "Everything in this part is a way of scoring sets, of measuring how much a set is worth beyond its members, or of "
  "letting the autoencoder tell us which bands act as a group.")

h2("4.2 Scoring a team of bands")
p("We need a way to give a number to any set of bands. Call it the score of the set. Three candidates, all of which "
  "we can compute quickly:")
bullets([
    ("With labels: KNN accuracy.", "Take the labelled pixels, keep only the bands in the set, run the five-neighbour "
                                   "classifier, report the accuracy. This is the score the paper uses."),
    ("Without labels: summary predictability.", "How well can the autoencoder's 20-number summary be predicted from "
                                                "the bands in the set (the reward from Idea A)."),
    ("Without labels: rebuild quality.", "How well can all 192 bands be rebuilt from the bands in the set."),
])
p("The first is the most trustworthy and the two others are the most useful, because they allow scoring millions of "
  "sets without labels. We will use the labelled score to check the unlabelled ones wherever we can.")

# ---- Idea 1
h2("Idea 1. Score every pair and every triplet (do this first)")
h3("What it is")
p("For sets of two and three bands, we do not need to search at all. We can score every possible set. There are "
  "18 336 different pairs of bands and 1 161 280 different triplets. That sounds like a lot, but scoring one "
  "triplet with the KNN classifier on the 4 000 balanced lichen pixels we already use for the baseline comparison "
  "is one distance table of 4 000 by 4 000 numbers, about a millisecond on a graphics card. All triplets take twenty "
  "to forty minutes; with cross-validation, a few hours; with the unlabelled scores, minutes.")
h3("Why it matters")
p("This gives us something nobody has for a real multi-excitation dataset: the complete landscape of how good every "
  "small combination is. From it we learn:")
bullets([
    "How many near-perfect triplets exist, and how rugged the landscape is. If thousands of triplets are within a "
    "point of the best one, the exact choice matters little and robustness matters more. If only a few are, finding "
    "them is the whole game.",
    "Which bands are team players and which are soloists: for every band we compare its best triplet score with its "
    "score alone. A band that is weak alone but appears in many of the best triplets is a team player, and our "
    "current ranking would have missed it.",
    "Where our ranked bands sit in that landscape. Do the top nine bands from the paper contain the best triplets, or "
    "do they miss them? This single check tells us how much there is to gain from all the other ideas in this part.",
])
p("It also turns the paper's robustness argument (better than 10 000 random 13-band sets) into a complete "
  "enumeration at small K, and it gives an exact answer against which every selector, including the RL agent, can be "
  "judged at K equals 3.")
h3("The figure")
p("A 192 by 192 map, arranged on the excitation-emission grid, where the colour of each cell is the score of the best "
  "triplet containing that pair. Bright cells are pairs that belong to great teams. It is easy to read, and it will "
  "immediately show whether the great teams span different excitations, which is the whole premise of "
  "multi-excitation imaging.")

# ---- Idea 2
h2("Idea 2. Synergy, redundancy and fair credit (Shapley values)")
h3("Synergy")
p("With set scores in hand, we can measure how much two bands add to each other. The synergy of bands A and B is the "
  "score of the pair minus the score of A alone minus the score of B alone. If the pair scores 80 while each alone "
  "scores 50, the synergy is minus 20: they are redundant, because together they add less than the sum of their "
  "parts. If the pair scores 90 while each alone scores 30, the synergy is plus 30: they complement each other. A "
  "192 by 192 map of synergy, again drawn on the excitation-emission grid, is the 'combination importance map' you "
  "asked for. Its block structure will show which regions of the grid cooperate and which merely repeat each other.")
h3("Fair credit: Shapley values")
p("When bands interact, how much credit does each one deserve? The fair answer comes from game theory. Imagine "
  "adding bands to the set one at a time in a random order, and each time recording how much the score improved "
  "when a particular band was added. Average that improvement over all possible orders, and you get the band's "
  "Shapley value. It is the only credit-sharing rule that is fair in a precise sense (the credits add up to the "
  "total score, identical bands get identical credit, a band that never helps gets zero). Shapley values replace our "
  "individual ranking with one that accounts for interactions.")
p("The same theory defines interaction indices for pairs, and so-called dividends for triplets, which isolate the "
  "part of a triplet's score that none of its pairs or singles can explain. A table of the ten triplets with the "
  "largest dividends is, quite literally, the list of three-band combinations whose importance only shows when all "
  "three are present.")
h3("Cost")
p("Everything here is a matter of evaluating set scores many times. With the unlabelled linear scores we can afford "
  "tens of millions of evaluations, so exact values at small K and well-sampled estimates at larger K are within "
  "reach on one machine.")

# ---- Idea 3
h2("Idea 3. Letting the autoencoder tell us which bands act as a group")
p("Ideas 1 and 2 treat the scorer as a black box. The autoencoder itself can also reveal groups, without labels and "
  "without enumerating sets. Three ways:")
h3("Knock out a group and see what the summary does")
p("Take a set of bands, blank them out in the input (set them to zero or to their average), feed the image through "
  "the encoder, and see how far the 20-number summary moves. A large move means the group carries information the "
  "summary needs. Then ask the decoder to rebuild the blanked bands from the rest: if it can, the group was "
  "redundant; if it cannot, the group carries unique information. When the effect of blanking a group is larger than "
  "the sum of blanking its members one by one, the group has synergy in the eyes of the model. This costs nothing but "
  "forward passes of the network.")
h3("Read the whole response table, not only its column sums")
p("Our nudging step produces, in effect, a table with one row per band and one column per summary number: how much "
  "each band responds to a nudge of each summary number (mathematicians call this table the Jacobian). Today we only "
  "use the size of each row, summed up. But the rows themselves are informative: two bands with similar rows respond "
  "to the same summary numbers in the same way, so they are functionally redundant, whatever their spectra look like. "
  "Grouping bands by their rows gives functional groups, and it gives a better notion of similarity for the "
  "spread-out rule than the spectral similarity we use now. A standard decomposition of the table (singular value "
  "decomposition) also produces 'band modes': patterns of bands that move together, ranked by how much of the "
  "table they explain. The top three modes are three combinations of bands, each with a weight for every band, and "
  "they can be read off without any labels. They are also, in the theoretical language of our rebuttal, the main "
  "directions along which the compressed information depends on the bands.")
h3("Nudge two summary numbers at once")
p("Nudge two summary numbers together and compare the response with the sum of the two single nudges. The difference "
  "is a second-order effect, and mapped through the response table it tells us which pairs of bands respond jointly. "
  "This is the model-based counterpart of the synergy map of Idea 2. It was out of reach in the paper because "
  "brute-force nudging was already slow; with the automatic differentiation tools built into PyTorch, both the table "
  "and the second-order terms come from exact, batched computations that are much faster than finite differences.")

# ---- Idea 4
h2("Idea 4. Which excitations matter, exactly")
p("Our autoencoder has one branch per excitation. That makes 'how important is excitation 365 nm as a whole' a "
  "question we can answer exactly. First we retrain the model with branch dropout: during training, whole excitation "
  "branches are randomly switched off, so the model learns to cope when an excitation is missing. Then, at test "
  "time, we can switch any combination of branches off and score the result. With eight excitations there are only "
  "256 combinations, so we can evaluate all of them and compute exact Shapley values and exact interaction terms for "
  "excitations, no sampling needed.")
p("This gives the cleanest possible answer to 'which lamp lines does the instrument need', it reproduces the "
  "'silent excitation' finding on the droplet data as a number rather than an observation, and as a bonus the "
  "model becomes robust to a missing lamp line at deployment.")

# ---- Idea 5
h2("Idea 5. Choosing sets instead of ranking bands")
p("The spread-out rule adds one band at a time and never reconsiders. Two established tools choose whole sets:")
bullets([
    ("Determinantal point processes.", "Think of a lottery over sets of K bands in which a set's chance of being drawn "
                                       "grows with how relevant its members are and shrinks when members resemble each "
                                       "other. Drawing from that lottery gives several good, different sets rather than one "
                                       "ranking, and the frequency with which a band appears in drawn sets is an "
                                       "interaction-aware importance score. The similarity between bands can come from the "
                                       "response-table rows of Idea 3. Our spread-out rule is, in fact, a greedy shortcut "
                                       "to this lottery."),
    ("Submodular optimisation.", "Many sensible set scores have a diminishing-returns property: adding a band to a small "
                                 "set helps more than adding it to a large one. For such scores, the simple greedy "
                                 "procedure is guaranteed to reach at least 63 percent of the best possible score. "
                                 "Phrasing our selection as maximising such a score, with the greedy rule as the solver, "
                                 "gives the single objective the reviewers asked for, with the current method recovered as "
                                 "a special case."),
])

# ---- Idea 6
h2("Idea 6. Mining the results we already have (one afternoon)")
p("The big lichen sweep produced 3 072 selections, and 16 groups of configurations beat the all-bands baseline. We "
  "can count, across those good selections, how often each pair of bands is chosen together. That gives a network "
  "in which bands are nodes and 'often chosen together' is a link. Standard community detection on that network "
  "finds groups of bands that travel together, and a simple count gives the most frequent triplets. This needs no "
  "training and no graphics card, it can be done today from the files in the results folder, and it produces the "
  "first 'combinations, not rankings' figure. It also feeds the idea of choosing K by stability in Part 3.")

# ---- Idea 7
h2("Idea 7. Combinations with physical meaning")
p("Beyond sets, the physics suggests derived features. The ratio of two emission bands under the same excitation "
  "does not depend on how bright the pixel is, only on the relative amounts of the molecules emitting at those two "
  "wavelengths. The ratio of the same emission band under two different excitations reads how the molecule absorbs, "
  "which is another fingerprint. Including such ratios and differences over our shortlisted bands as extra candidates "
  "in Ideas 1 and 2 lets us find combinations that are not only informative but interpretable. Together with the "
  "physics-based decompositions of Part 3, we could then say 'this triplet reads the ratio of collagen-like to "
  "NADH-like emission'. That is the concrete route to the grant's 'formulas that combine them'.")

doc.add_page_break()

# ================================================================ 5. PART 3 BROADER
h1("5. Part 3: Broader ideas")

h2("5.1 One objective instead of a chain of steps")
p("The reviewers' strongest complaint was that the method is a sequence of separate tricks. Two ways to make it one "
  "objective in code, not only in prose:")
bullets([
    ("A soft switch on every band.", "Put a trainable switch on each of the 192 inputs (the Concrete Autoencoder idea), "
                                     "train the autoencoder to rebuild the data through the switches with a penalty for "
                                     "keeping too many on, and start the switches from our nudging scores. If the switches "
                                     "end up agreeing with the nudging scores, we have shown that our method is a cheap "
                                     "approximation of an explicit objective. If they disagree, the disagreement is itself "
                                     "a finding worth a paper."),
    ("Summary predictability as the objective.", "Choose the set of bands that best predicts the autoencoder's summary, "
                                                 "with a penalty for redundancy. Steps 2 and 3 of the pipeline become one "
                                                 "optimisation problem, and the RL agent, the set lottery and the greedy rule "
                                                 "are simply different ways of solving it."),
])

h2("5.2 Fixing the autoencoder where it breaks")
p("The summer finding stands: the spatial autoencoder can cheat on sparse data. The planned ladder of fixes, in "
  "order, is a per-pixel spectral autoencoder, then a hybrid that adds a little spatial context after the spectral "
  "encoder, a signed summary instead of a positive-only one, and a genuine bottleneck. Two additions help every idea "
  "in this document. Branch dropout (Idea 4) makes the model tolerate missing excitations. Exposure-invariance "
  "augmentation, meaning that during training we randomly rescale each excitation's brightness and add noise, "
  "prevents the summary from encoding lamp brightness at all; it is the model-side version of the lesson that "
  "brightness is not fluorescence. A further step is to replace the rebuild-the-input objective by a contrastive "
  "one, where the model learns to recognise the same pixel under different brightness and noise, which makes the "
  "invariance explicit rather than hoped for.")

h2("5.3 Attribution with physical meaning")
p("An excitation-emission table is a three-way object (pixel by excitation by emission), and there is a classical "
  "statistical method built precisely for such tables: PARAFAC, with its non-negative cousin MCR-ALS. They decompose "
  "the table into a handful of components, each with an excitation profile, an emission profile and a concentration "
  "map, which very often correspond to individual fluorescent molecules. Fitting them to our data and matching the "
  "components against the molecule library that SpectraForge already contains lets us say what each selected band "
  "or triplet is reading. It also gives a physics baseline for the autoencoder: do the 20 summary numbers correlate "
  "with the component concentrations? A further variant is a physics-informed autoencoder whose decoder is a fixed "
  "dictionary of known molecule spectra and whose encoder learns only concentrations.")

h2("5.4 Choosing K without labels, and saying how sure we are")
bullets([
    ("K from stability.", "Repeat the selection with different random seeds and image crops and measure how much the "
                          "chosen sets overlap. The smallest K at which the overlap stops changing is a natural, label-free "
                          "choice. Alternatively, stop when adding a band improves summary predictability by less than the "
                          "noise level. Either removes the reviewer's objection about choosing K with labels."),
    ("Confidence for free.", "The model already uses dropout, which randomly switches off half of its internal units "
                             "during training. Leaving dropout on at test time and repeating the nudging a few dozen times "
                             "gives a spread of influence maps at no extra engineering cost. Bootstrap ensembles do the "
                             "same for Shapley values. We could then report bands as 'selected in 94 percent of resamples'."),
])

h2("5.5 Speed")
p("Replacing brute-force nudging by the exact derivative tools in PyTorch (Jacobian-vector products) should shrink "
  "the ninety-minute step to minutes, and it is what makes the response table and the second-order analysis of Idea 3 "
  "affordable. This is engineering rather than research, but it unlocks several of the research items.")

h2("5.6 Training once, applying everywhere")
p("If we train one encoder on all our datasets plus SpectraForge scenes, with branch dropout and exposure "
  "augmentation, we can attribute bands for a new sample without retraining, and with the amortised agent of Idea A, "
  "selection for a new specimen takes seconds. Transfer must be tested honestly: from the small lichen image to the "
  "large one (same specimens, different resolution), from one collagen treatment to the other, and from simulation "
  "to reality.")

h2("5.7 Evaluation habits that should become standard")
bullets([
    "Run the saturation diagnostic before any experiment; if the gap between the best labelled selection and a random "
    "one is under five points, no selector can show anything on that dataset.",
    "Always report random, variance, PCA loading, supervised mutual information, and where possible the exhaustive best "
    "set, next to our method.",
    "Use per-pixel scores; cluster-level scores only as a secondary check, because we showed they can be gamed.",
    "Report both normalisations of the influence scores, since one of them inverts the ranking on some datasets.",
    "Repeat with several seeds and give confidence intervals.",
])

doc.add_page_break()

# ================================================================ 6. AT A GLANCE
h1("6. What each idea gives us, at a glance")
table(
    ["Idea", "Needs labels?", "Answers which gap", "Main risk"],
    [
        ["A. Trained selector (RL)", "No (labels optional)", "Speed; interactions; single objective", "May not beat a genetic algorithm on one image"],
        ["B. Measuring agent (RL)", "No", "Adaptive acquisition; device design", "Simulator-to-reality gap; needs a belief model"],
        ["C. Budget per excitation (RL)", "No", "Instrument recipe; excitation importance", "Small gain over Idea 4"],
        ["D. Learned nudging settings", "No", "Hyperparameter search", "Bayesian optimisation is simpler"],
        ["E. Formula search", "Either", "Interpretable indices; grant O4", "Search space explodes without a shortlist"],
        ["1. Exhaustive pairs and triplets", "Yes for KNN; no for proxies", "Ground truth for combinations", "Only small K"],
        ["2. Synergy and Shapley", "Either", "Combination importance; fair credit", "Depends on the quality of the set score"],
        ["3. Groups from the autoencoder", "No", "Label-free groups; better similarity for MMR", "Inherits the autoencoder's flaws"],
        ["4. Exact excitation Shapley", "Either", "Which lamp lines to build", "Requires retraining with branch dropout"],
        ["5. Set lottery / submodular", "No", "Sets not rankings; single objective", "New machinery to explain"],
        ["6. Co-selection mining", "No", "Immediate combinations figure", "Descriptive only"],
        ["7. Physical combinations", "Either", "Interpretability; grant O4", "Needs component decomposition first"],
        ["3.1 to 3.7 broader ideas", "Mostly no", "Reviewer points; robustness; speed", "Spread of effort"],
    ],
    widths_cm=[4.6, 3.0, 5.2, 4.6],
)

# ================================================================ 7. ORDER OF WORK
h1("7. Suggested order of work")
p("The order below is chosen so that each step either de-risks the next or produces something publishable on its own.")
table(
    ["Step", "What", "Effort", "Depends on", "Why in this position"],
    [
        ["1", "Exhaustive pairs and triplets on lichens (Idea 1), with KNN and the unlabelled proxies", "2 to 3 days", "Balanced 4 000-pixel subset; a graphics card", "Ground truth for every later claim; a strong figure by itself"],
        ["2", "Co-selection network and frequent triplets from the existing sweep (Idea 6)", "1 day", "Results folder of the lichen sweep", "Free and immediate"],
        ["3", "Exact response table via automatic differentiation; band modes; row clustering as the MMR similarity (Idea 3, 5.5)", "3 to 5 days", "A trained model", "Speeds up the pipeline and gives model-based groups"],
        ["4", "Synergy map, Shapley values, triplet dividends (Idea 2)", "3 days", "Steps 1 and 3", "The requested combination-importance analysis"],
        ["5", "Branch dropout and exact excitation Shapley (Idea 4)", "1 week", "Retraining", "Instrument-level result; feeds the device objective"],
        ["6", "Summary-predictability objective with set lottery or submodular solver (Idea A reward, Idea 5, 5.1)", "1 to 2 weeks", "Steps 1 and 3", "The single objective; baseline for RL"],
        ["7", "Amortised RL selector trained on crops and SpectraForge, tested by transfer (Idea A)", "3 to 4 weeks", "Step 6; spectral autoencoder backbone", "The RL paper, if it beats a genetic algorithm at equal budget or shows transfer"],
        ["8", "Measuring agent in SpectraForge, then offline on real cubes (Idea B)", "4 to 6 weeks", "SpectraForge acquisition simulator; step 5", "The instrument-native story; grant O5"],
        ["9", "PARAFAC/MCR attribution and formula search (Idea 7, Idea E, 5.3)", "2 to 3 weeks", "Steps 1 and 4", "Grant O4; interpretability"],
        ["10", "Label-free K, uncertainty, exposure-invariant pre-training (5.4, 5.2, 5.6)", "ongoing", "", "Closes the remaining reviewer points"],
    ],
    widths_cm=[1.0, 6.0, 2.2, 3.6, 4.6],
)
box(
    "The decision point after step 4",
    "Steps 1 to 4 form a self-contained short paper: 'How much of band importance in multi-excitation imaging is "
    "interaction?' They also tell us whether interactions are large enough to justify the RL work. If the exhaustive "
    "landscape shows that the best triplets are simply the three best individual bands, then set-level methods and "
    "RL have little to gain on this data, and the effort should go to the measuring agent (Idea B) and the physics "
    "(5.3) instead. If the best triplets contain team players that the ranking missed, the whole of Part 2 and "
    "Idea A are justified.",
    fill="FFF4E0",
)

# ================================================================ 8. RISKS
h1("8. Risks and honest expectations")
bullets([
    ("RL as a wrapper.", "On a fixed dataset, a genetic algorithm or simulated annealing with the same number of reward "
                         "evaluations is the fair comparison and may win. We should design the RL experiments around "
                         "transfer to new images and sequential acquisition, where RL has a structural advantage, and say "
                         "plainly if it does not win on a single image."),
    ("Supervised rewards contradict the unsupervised story.", "Keep the two tiers explicit: unlabelled rewards for "
                                                                "selection, labels only for evaluation. Otherwise the "
                                                                "reviewers' objection about hidden supervision returns."),
    ("Interaction analyses are only as good as the set score.", "The linear proxies are fast but can miss nonlinear "
                                                                 "synergies, so they must be anchored to the KNN result at "
                                                                 "small K, where we can afford the exact computation."),
    ("Saturated datasets show nothing.", "Use lichens and collagen. Use SpectraForge only with the discriminability-based "
                                          "ground truth that is planned, not the broad 'informative band' mask that we "
                                          "withdrew in July after a random selector matched the autoencoder on it."),
    ("The autoencoder's own flaws propagate.", "Everything in Idea 3 inherits whatever the autoencoder learned. Check "
                                                "reconstruction quality per band (the correlation between input and "
                                                "rebuilt values) before trusting any model-based group."),
    ("Too many directions.", "The list is long on purpose. The order in Section 7 is the antidote: steps 1 to 4 are cheap, "
                             "decisive and publishable, and they tell us which of the expensive directions deserve the "
                             "time."),
])

# ================================================================ 9. GLOSSARY
h1("9. Glossary")
table(
    ["Term", "Plain-English meaning"],
    [
        ["Agent, state, action, reward, episode, policy", "The vocabulary of reinforcement learning; see Section 2."],
        ["Amortisation", "Paying a training cost once and reusing the result on many new samples without retraining."],
        ["Autoencoder", "A neural network that compresses its input into a short summary and rebuilds the input from it; trained by penalising rebuild errors."],
        ["Band", "One excitation-emission pair; one number per pixel."],
        ["Bottleneck / summary / latent", "The short (20-number) representation in the middle of the autoencoder. 'Latent' is the technical word."],
        ["Branch dropout", "Randomly switching off whole excitation branches during training so the model tolerates missing excitations."],
        ["Concrete Autoencoder", "A method that puts soft switches on the inputs so that band selection can be trained with ordinary gradients."],
        ["Determinantal point process (DPP)", "A lottery over sets that favours relevant, mutually different members."],
        ["Excitation / emission", "The wavelength of the light we shine on the sample / the wavelength of the light it gives back."],
        ["Excitation-emission table (EEM)", "The grid of intensities over all excitation and emission wavelengths, for one pixel or region."],
        ["Genetic algorithm, simulated annealing", "Classic search methods that improve candidate solutions by mutation and selection, or by random changes that are kept when they help."],
        ["Jacobian / response table", "The table of how much each output (band) responds to a small change in each summary number."],
        ["KNN (k-nearest neighbours)", "A classifier that labels a pixel by majority vote of its k most similar labelled pixels; we use k equals 5."],
        ["MMR (maximal marginal relevance)", "The greedy rule that adds relevant but not-too-similar bands one at a time."],
        ["PARAFAC / MCR-ALS", "Classical methods that decompose an excitation-emission table into a few components, often one per fluorescent molecule."],
        ["Perturbation / nudging", "Changing one summary number slightly and observing how the rebuilt bands respond."],
        ["PPO, REINFORCE", "Standard algorithms for training a policy from rewards."],
        ["R-squared", "A number between 0 and 1 saying how well a predictor explains its target; 1 is perfect."],
        ["Saturation (of a dataset)", "The situation where channels are so redundant that any selection performs as well as the best one."],
        ["Shapley value", "A fair share of credit for a band, averaged over all orders in which bands could have been added."],
        ["SpectraForge", "Our simulator: paints known fluorescent molecules onto a canvas and renders the cube the instrument would record, with known ground truth."],
        ["Submodular", "A set score with diminishing returns; the greedy rule is guaranteed to do well on it."],
        ["Synergy", "Score of a pair minus the scores of its members alone; positive means complementary, negative means redundant."],
    ],
    widths_cm=[5.0, 12.4],
)

# ================================================================ APPENDIX
h1("Appendix: where things are in the repository")
table(
    ["What", "Where"],
    [
        ["The compact technical version of this note", "docs/research/2026-09-11-rl-and-group-importance-brainstorm.md"],
        ["The paper (method, results, discussion)", "publications/tpami/paper/sections/"],
        ["Reviewer comments and our planned responses", "publications/tpami/revision/MITIGATION_TABLE.md"],
        ["Baseline methods, defined with formulas", "publications/tpami/revision/SOTA_METHODS_EXPLAINED.md"],
        ["The selection engine (nudging, influence, normalisation)", "src/selection_core/engine.py"],
        ["The autoencoder, training and the Analyzer", "src/spectral_select/ (models/, analyzer.py)"],
        ["The 3 072-configuration lichen sweep outputs (for Idea 6)", "results/Lichens_Dataset_1_MasterRun/"],
        ["The SpectraForge findings on the cheating autoencoder", "docs/spectraforge/02-investigation-and-findings.md"],
        ["The saturation diagnostic", "experiments/saturation_diagnostic.py"],
        ["The generalisation research log (saturation, supervised MI comparisons)", "publications/generalization/RESEARCH_LOG.md"],
        ["Onboarding material for new team members", "docs/onboarding/"],
    ],
    widths_cm=[7.0, 10.4],
)

# ---------------------------------------------------------------- footer
for section in doc.sections:
    f = section.footer.paragraphs[0]
    f.text = "spectral-select  |  Research directions: RL, wavelength combinations, and beyond  |  September 2026"
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in f.runs:
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
words = sum(len(pp.text.split()) for pp in doc.paragraphs) + sum(
    len(c.text.split()) for tb in doc.tables for row in tb.rows for c in row.cells
)
print(f"wrote {OUT} ({words} words, {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables)")
