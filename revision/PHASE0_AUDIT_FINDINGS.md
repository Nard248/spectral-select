# Phase 0 Audit — Critical Findings

**Date:** 2026-05-12
**Status:** ⚠️  Requires user decision before Phase 1 proceeds.

The Phase 0 audit (consistency between manuscript and code/data) surfaced discrepancies that go beyond R1.9 and R1.10 — the paper's empirical claims do not match the data on disk. The discrepancies are *favorable* (reality is better than claimed), but they need a deliberate decision about how to handle them.

---

## Finding #1 — Perturbation magnitudes (R1.10 confirmed real)

| Source | Value |
|---|---|
| `archive/paper/sections/methodology.tex` | `ε ∈ {15, 30, 45}` |
| `archive/paper/sections/experimental_setup.tex` | `ε ∈ {15, 30, 45}` |
| `spectral_select/config.py:127` default | `[10, 20, 30]` |
| `experiments/run_master_experiment.py:106` `PERTURBATION_MAGNITUDE_VARIANTS` | `{"medium": [30,40,50], "high": [50,60,70]}` |
| `experiments/collagen_*.py` scripts | `[50,60,70]` |
| **Lichens MasterRun `results.csv` rows** | All rows use `[30,40,50]` or `[50,60,70]` |

**The values `{15, 30, 45}` in the paper appear nowhere in the codebase.** This is a real R1.10 hit — the paper either contains a typo or was written from an earlier experimental setup that has since been overwritten.

**Recommended fix:** Update paper text to reflect the actual experiment: `ε ∈ {30, 40, 50}` (the "medium" variant which is the default used in the MasterRun configurations that produce the best results) and note in the ablation that `{50, 60, 70}` was also tested and produced essentially identical selections.

---

## Finding #2 — Baseline accuracy off by 2.7 pp

| Source | Baseline (192 bands) accuracy |
|---|---|
| Paper Abstract & §V.B Table III | **0.8554** |
| `paper_metrics.json` | **0.8815** |
| MasterRun `results.csv` BASELINE row | **0.8815** |

Same K=192, same KNN classifier (k=5). Different number. The paper's baseline is **understated by 2.7 percentage points**.

Possible reasons:
- Different label set (paper claims 223,597 labeled pixels; the MasterRun confusion matrix sums to 191,046).
- Different train/test split.
- Different preprocessing.

**Implication:** Every "Rel. Perf." percentage in Table IV and every "% of baseline" claim is computed against the wrong baseline.

---

## Finding #3 — k=1 optimality claim is wrong (R1.9 confirmed real, paper-level)

Paper §V.F Table V (Dimension Selection Methods):

| Paper claim | Variance(1) | Variance(3) | Variance(5) | PCA |
|---|---|---|---|---|
| Reported accuracy at K=13 | **0.8606** | 0.8547 | 0.8489 | 0.7654 |

Reality on the MasterRun `results.csv` at K=13:

| Config (dim-method × k × normalization) | Mean accuracy |
|---|---|
| **PCA, k=3, max_per_excitation** | **0.901** ⭐ optimal |
| PCA, k=1, max_per_excitation | 0.856 |
| PCA, k=1, none | 0.818 |
| PCA, k=3, variance | 0.694 |
| variance, k=1, variance | **0.694** ← paper's claimed "optimal" |
| variance, k=1, max_per_excitation | 0.637 |
| variance, k=3, max_per_excitation | 0.637 |

**The paper's claimed "optimal" config (variance + k=1) scores 0.694, not 0.861.** The actual optimal config at K=13 is **PCA + k=3 + max_per_excitation** at **0.902**.

The paper's Table V appears to either:
- Have labeled the rows wrong (these may actually be PCA results mis-labeled as Variance), or
- Have been computed on a different (pre-fix) version of the codebase where `dimension_selection_method='variance'` did something different than it does now, or
- Have been transcribed from a stale or different run.

---

## Finding #4 — Headline result understated

Paper headline: **"13 bands → 86.1%, exceeding 85.5% baseline (101% relative performance)"**

Reality (MasterRun, optimal config):

| K | Best accuracy | vs baseline (88.2%) |
|---|---|---|
| 9 | 0.894 | +1.2 pp (101.4%) |
| 13 | **0.902** | **+2.1 pp (102.4%)** |
| 30 | 0.930 | +4.8 pp (105.5%) |
| 50 | 0.940 | +5.9 pp (106.6%) |
| **80** | **0.952** ⭐ | **+7.0 pp (108.0%)** |
| 100 | 0.939 | +5.7 pp (106.6%) |
| 192 | 0.882 | baseline |

The K-vs-accuracy curve has a clear peak at **K=80** with **95.2%** — this is the actual peak of the method on Lichens, and it sits **7 percentage points above baseline**, not 0.6 as the paper currently claims.

K=13 is still a respectable trade-off (90.2%, 2.1pp above baseline), but it is not the "optimal" point.

---

## Implications for the revision

The revision is now significantly easier in some ways and harder in others:

**Easier:**
- Reviewer R1.5/R2.2 (single small dataset) is partly addressed by the *actual* numbers being more impressive. 95.2% at 80 bands (vs 88.2% baseline) is a stronger headline than 86.1% at 13.
- Reviewer R1.6 (unsupervised claim) is helped because the best configurations (PCA + max_per_excitation) are configuration choices, not labels.
- Reviewer R1.3/R2.3 (no strong baselines) becomes addressable: when we add SOTA baselines, our actual numbers are higher than the paper claimed, so the comparison is cleaner.

**Harder:**
- Every numerical table in the paper needs to be rebuilt from `results.csv`.
- Every figure plotting accuracy vs K needs to be regenerated.
- The narrative shifts: "13 bands at 86.1%" was a clean punchline; "80 bands at 95.2% with a smaller-K sweet spot at K=18 (92%) and K=13 (90.2%)" is more complex.
- Reviewer R1.9 was *partly* right — k=1 is not optimal in the actual data. We need to either (a) admit this and recast the optimal config as PCA + k=3, or (b) explain why the paper's claim used a different definition.

---

## Decisions needed

I have **stopped before touching any paper text** because these changes affect the headline claims. Three options:

1. **(Recommended) Rebuild the paper from the MasterRun data.** Headline becomes "80 bands → 95.2%, sweet spot at 13 bands → 90.2%." Update Table III/IV/V. Recompute random-baseline percentile (likely still 100th, given a larger gap). Update Discussion accordingly. Time cost: ~1 day of figure regeneration + text updates.

2. **Investigate whether the paper's numbers came from a specific older run.** If so, decide whether to (a) re-anchor on the older run for continuity, or (b) move to the MasterRun as the new canonical. Time cost: ~2 hours investigation, then path (1) or accept old numbers.

3. **Run a fresh full experiment on Lichens** with the exact protocol described in the paper (variance, k=1, ε={15,30,45}) and see what numbers actually come out. If 86.1% emerges, the paper's claim was reproducible — the MasterRun just used a different (better) config. Time cost: ~1 day of compute (Lichens is the largest dataset).

My recommendation is **path 1**. The MasterRun is the most recent canonical run, the numbers are better, and "we updated to a better configuration in revision" is a defensible story. Path 3 would *also* be valuable as a side-experiment to confirm the paper's old number is reproducible, but it shouldn't gate the revision.

---

## Other Phase 0 checks not yet done

- Paper claim "10,000 random combinations, mean 46.1%, max 57.9%" — needs verification against `results/Lichens_Dataset_1_*` random-baseline outputs.
- Paper claim "Object 14 improved from 36.6% to 44.4%" — needs verification.
- Collagen Sponges and Drop Data have their own numerical claims (e.g. F=89.9 at K=5 on `full_cr`) — those are recent and reliable per the conversation memory.

These can wait until the Lichens reanchoring decision is made.

---

## Finding #5 — Lichens excitation grid + class labels also wrong

Added 2026-05-12 after running the Lichens adapter.

| Source | Excitation grid | Class labels | Total labeled pixels |
|---|---|---|---|
| Paper §IV.A.1 | 8 (310, 320, 330, 340, 345, 350, 355, 365) | "4 lichen morphological types: Type 0/1/2/5" | 223,597 |
| `spectra_masked.pkl` and `class_mask.png` | 8 (310, **325**, 340, **365**, **385**, **400**, **415**, **430**) | Classes 1/3/6/7 (RGB-encoded) | **191,046** |

The paper's excitation grid does not match the actual experiment grid. Five of the eight values differ. The class IDs and counts also differ. This is consistent with the Finding-#3 pattern: paper text was written from an earlier version of the experiment and never re-anchored.

**Implication for revision:** §IV.A.1 needs full rewrite to reflect actual excitations and class semantics. The "4 classes" claim is still defensible (it's 4 classes), but the IDs and counts need to come from the current data.
