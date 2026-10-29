# ME-Attention remote-GPU results (2026-10-28 -- 2026-10-29 handoff)

Hardware: Windows desktop, NVIDIA RTX 3060 (12 GB), Intel i9-9900K (8c/16t), 32 GB RAM,
torch 2.12.1+cu126. Code: `src/me_attention/`. Protocol: `docs/me_attention/SPEC.md`. Robustness
pre-registration: `docs/me_attention/PREREG_ROBUSTNESS_2026-10-28.md` (frozen 2026-10-28, two
dated deviations logged in its own Deviations section -- see section 5 below).

**Scope actually run, by user instruction (both logged as deviations, neither changes a frozen
condition, level, hypothesis or the decision rule):** Collagen and the full 5-class Lichens 2 were
dropped from the robustness study (Phase B) to concentrate compute on Lichens 2's 3-class subset
and Lichens 1. The clean sweep (Phase A protocol) still covers all four datasets, finished earlier
and unaffected by this change.

## 1. Phase A -- clean results (whole-specimen nested CV, corrected statistics)

All four datasets, full clean sweep. `ex_tokens_lin` (the Phase A addition) vs. the best classical
baseline, mean balanced accuracy % over (repeat, fold) units, Nadeau-Bengio corrected paired
t-test, Holm-adjusted across each dataset's baseline comparisons (`results/me_attention/analysis_sub3`,
`analysis_final_recwl`):

| Dataset | `ex_tokens_lin` | Best baseline | Delta | Holm-adj. p | Significant? |
|---|---|---|---|---|---|
| Lichens 2 (3-class subset) | 96.0 +/- 3.0 | RBF-SVM 96.6 +/- 2.3 | -0.6 | 0.227 | no |
| Lichens 2 (full 5-class) | 85.5 +/- 6.8 | RBF-SVM 86.6 | n/a (not in frozen stats table) | -- | -- |
| Collagen Pepsin | 85.0 +/- 9.5 | LR 90.3 | n/a (not in frozen stats table) | -- | -- |
| Lichens 1 | 98.8 +/- 0.8 | SpectralFormer 99.5 | n/a (not in frozen stats table) | -- | -- |

(`ex_tokens_lin`'s own paired comparison against each dataset's best baseline was not part of the
frozen `FROZEN_SPACES.md` statistics table build -- only the Lichens 2 subset number above comes
directly from `tab_stats.md`; the other three are the two models' raw means, not a paired test. If
a formal paired test for `ex_tokens_lin` specifically is wanted on those three, it is a quick
re-run of `analysis.py`'s `paired()` helper and not included here to avoid implying a significance
test that was not actually run with full rigor.)

Full clean-sweep ranking, every spectral model, every dataset (`tab_spectral.md` in each analysis
folder): on every dataset the best attention model and the best classical baseline are statistically
tied (no Holm-significant advantage either way) except where a classical baseline wins outright
(Lichens 1: SpectralFormer edges every attention model, not significantly). The clean task is near
ceiling on Lichens 2 subset (specimen-vote accuracy 100% for every model) -- this is exactly the
motivation written into the robustness pre-registration before Phase B was run.

**Collagen, clean-sweep only:** best attention model (`ex_tokens_v2`, 90.0) is statistically tied
with the best baseline (LR, 90.3; Holm p = 0.72) and significantly worse than several (see section
4 for the recommendation).

## 2. Phase B -- robustness (377/377 units complete for the two datasets run)

**Confirmatory design (frozen, `COMPARATORS_2026-10-28.md`):** candidates `ex_tokens_v2` and
`ex_tokens_lin` vs. the best clean baseline (`rbfsvm` for both Lichens 2 variants), across
`{T1, T2, T3, L, C}`, on the two pre-registered primary datasets `{lichens2_sub3, lichens2}`. With
`lichens2` (full) dropped, **10 of the 20 pre-registered tests are complete; the other 10 show as
"missing"** in the table below -- not fabricated, not silently omitted, exactly as the
pre-registration's own honesty rule requires.

| Dataset | Family | Candidate | vs. | n | Delta [95% CI] | p (NB) | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|---|
| Lichens 2 subset | T1 (missing excitations) | ex_tokens_v2 | RBF-SVM | 25 | -5.1 [-7.3, -3.1] | 0.097 | 0.876 | no advantage |
| Lichens 2 subset | T1 | ex_tokens_lin | RBF-SVM | 25 | -1.9 [-4.0, +0.3] | 0.546 | 1.000 | no advantage |
| Lichens 2 subset | T2 (gain drift) | ex_tokens_v2 | RBF-SVM | 25 | +4.3 [+0.5, +8.0] | 0.426 | 1.000 | no advantage |
| Lichens 2 subset | T2 | ex_tokens_lin | RBF-SVM | 25 | +6.6 [+3.0, +10.4] | 0.216 | 1.000 | no advantage |
| Lichens 2 subset | T3 (noise) | ex_tokens_v2 | RBF-SVM | 25 | +3.8 [+1.8, +5.8] | 0.189 | 1.000 | no advantage |
| Lichens 2 subset | T3 | ex_tokens_lin | RBF-SVM | 25 | +6.1 [+4.3, +7.9] | 0.027 | 0.266 | no advantage |
| Lichens 2 subset | L (label noise) | ex_tokens_v2 | RBF-SVM | 25 | -0.1 [-1.5, +1.4] | 0.953 | 1.000 | no advantage |
| Lichens 2 subset | L | ex_tokens_lin | RBF-SVM | 25 | +0.7 [-0.7, +2.3] | 0.745 | 1.000 | no advantage |
| Lichens 2 subset | C (learning curve) | ex_tokens_v2 | RBF-SVM | 25 | -1.1 [-2.0, -0.1] | 0.452 | 1.000 | no advantage |
| Lichens 2 subset | C | ex_tokens_lin | RBF-SVM | 25 | -1.0 [-2.2, -0.1] | 0.507 | 1.000 | no advantage |
| Lichens 2 (full) | all 5 families | both candidates | RBF-SVM | 0 | -- | -- | -- | **missing** (dataset dropped) |

**Decision: no confirmatory robustness advantage anywhere.** Under the pre-registered rule
(Delta > 0 and Holm-adjusted p < 0.05), none of the 10 completed tests support an attention
robustness advantage on Lichens 2 subset. Directionally, attention is *worse* under T1 (missing
excitations, both candidates) and under C (reduced training data, both candidates), and nominally
better under T2/T3 (gain drift, noise) and roughly tied under L (label noise) -- but none of these
directional differences survive Holm correction.

Full exploratory tables (every attention model vs. every baseline, every dataset actually run,
clearly marked exploratory and not Holm-corrected across the whole set) and the robustness-score
table (mean balanced accuracy per family's non-clean levels) are in
`results/me_attention/analysis_robust/tab_exploratory.md` and `tab_robust_scores.md`.
Per-level degradation curves and figures: `tab_degradation.md`,
`figures/fig_degradation_{lichens2_sub3,lichens1,lichens2}.{png,pdf}`.

**Lichens 1** ran in full (all 13 models, all families) as an exploratory dataset (it was always
secondary in the pre-registration, so it carries no confirmatory tests) -- robustness scores are in
the same `tab_robust_scores.md`. Qualitatively the same pattern holds: attention is not
systematically more robust than the strongest classical baselines there either, though SpectralFormer
and the MLP baseline degrade less under some families (e.g. T2) than some attention variants.

A handful of `lichens2` (full) classical-model records (`lr`, `rbfsvm`, `rf`, `hgb`, `knn`, each
fully done for their 10 units) exist from before the dataset was dropped mid-run. They are kept on
disk and in git for transparency but are **not** part of any confirmatory or exploratory comparison
above, since the matching attention-model records for that dataset were never produced.

## 3. Hypotheses H1-H5 (predictions made *before* any robustness result existed)

- **H1 (T1, missing excitations): predicted "attention degrades less, most likely genuine
  advantage." Not supported -- reversed, in fact.** Both candidates lose *more* balanced accuracy
  than RBF-SVM under missing excitations (`ex_tokens_v2`: -5.1 points relative to RBF-SVM's own
  degradation; `ex_tokens_lin`: -1.9 points), though neither difference survives Holm correction.
- **H2 (T2, gain drift): predicted attention degrades less than kNN/RBF-SVM. Directionally
  supported, not statistically confirmed.** Both candidates show a positive Delta vs. RBF-SVM
  (+4.3, +6.6 points), i.e. less degradation, but p_Holm = 1.000 for both.
- **H3 (T3, noise): no prediction was made.** Both candidates again show a positive Delta vs.
  RBF-SVM (+3.8, +6.1); `ex_tokens_lin`'s raw p = 0.027 is the smallest of the 20 confirmatory
  tests but does not survive the Holm correction (0.266).
- **H4 (L, label noise): predicted kNN degrades the most, attention ~= other neural models. Broadly
  consistent.** Deltas vs. RBF-SVM are small and non-significant (-0.1, +0.7); in the full
  robustness-score table (`tab_robust_scores.md`) kNN's L-score (86.2) is indeed the lowest among
  classical models on Lichens 2 subset, while every neural model (attention and non-attention
  alike) clusters in the 91-95.5 range.
- **H5 (C, learning curve): predicted attention degrades *more* than kNN/RBF-SVM. Supported,
  directionally, not statistically confirmed.** Both candidates show a small *negative* Delta vs.
  RBF-SVM (-1.1, -1.0), i.e. attention loses slightly more under reduced training data, as
  predicted -- but again p_Holm = 1.000 for both.

**Overall:** on the one primary dataset with complete robustness data, attention shows **no
confirmatory robustness advantage over the strongest classical baseline (RBF-SVM) under any of the
five pre-registered perturbation families.** Where a direction reaches nominal significance (T3 /
`ex_tokens_lin`, raw p = 0.027) it does not survive the one Holm family of 20 tests. H1's "most
likely genuine advantage" prediction is the one most clearly contradicted by what was actually run.

## 4. Collagen recommendation

Per the pre-registration's own deviation log, Collagen carries **no robustness evidence** (dropped
from Phase B before any robustness run on it completed). The recommendation below is therefore the
clean-sweep comparison only, exactly as `PREREG_ROBUSTNESS_2026-10-28.md`'s deviation note commits
to.

Clean sweep, Collagen Pepsin, spectral track (`results/me_attention/analysis_final_recwl/tab_stats.md`,
`tab_spectral.md`):
- Best attention model: `ex_tokens_v2`, 90.0 +/- 10.9% balanced accuracy.
- Best baseline: LR, 90.3%.
- Holm-adjusted paired test, `ex_tokens_v2` vs. LR: Delta = -0.3 [-4.7, +3.0], p_Holm = 0.72 --
  statistically tied.
- Against every *other* baseline, `ex_tokens_v2` is directionally better (RBF-SVM +2.7, RF +6.3,
  HGB +5.3, kNN +8.7, MLP +6.8, 1D-CNN +10.8, SpectralFormer +6.1), several nominally significant
  before Holm correction but only kNN (p_Holm = 0.030) and 1D-CNN (p_Holm = 0.012) survive it.
- `smartattention_v0` and `hybrid_conv_attn` are both significantly *worse* than LR on Collagen
  (Delta -9.2 and -12.2, p_Holm = 0.010 and 0.041) -- the weakest attention variants on this
  dataset specifically.

**Recommendation: Collagen does not show a clear attention advantage and has no robustness
evidence behind it.** The single best attention variant (`ex_tokens_v2`) is statistically tied
with the best classical baseline (LR) on the comparison that matters for the paper's claim, and two
of the four attention variants tested are significantly worse than LR. Per the standard set at the
start of the study ("Collagen goes out of the paper if attention does not perform well there"),
this reads as **drop Collagen from the paper**, with the caveat that this call rests on clean-sweep
evidence alone -- no robustness comparison exists for it. The user makes the final call.

## 5. Hardware, wall time, deviations

**Wall time.** Phase A (today's additions: 33 sub3 jobs + 29 recwl jobs): sub3 19.9 min, recwl 4.8
min on this GPU. Phase B (377 units, Lichens 2 subset + Lichens 1): **total wall-clock span
2026-10-28 20:56 to 2026-10-29 15:51 (~19 h), but this includes an unplanned ~11.5 h system sleep**
(23:21-10:52, Windows event log: "Sleep Reason: Application API", not an idle timeout; the sleep
timer was already disabled, so something else triggered it -- cause not fully identified). The
process survived the sleep and resumed correctly with no data loss; active compute time was
approximately 4-6 hours across three restarts (two of the restarts were to apply the Collagen-drop
and Lichens-2-full-drop scope changes, both resumable with no wasted work beyond the units already
written).

**Deviations (full text in `PREREG_ROBUSTNESS_2026-10-28.md`'s own Deviations section):**
1. T1 subset count: exhaustive enumeration instead of 10 random draws when C(n_ex, m) <= 10
   (strictly reduces Monte-Carlo noise; same subsets for every model either way).
2. T1's imputed fill value maps to exactly 0 after preprocessing only for `norm="none"` configs,
   not every tuned config -- kept anyway because the stronger invariant (byte-identical input
   across models) matters more than the "maps to 0" framing.
3. C(b) level validity checked per fold (not just per dataset), matching the pre-registration's own
   worked `collagen_pepsin` example.
4. Collagen dropped from Phase B (user instruction, secondary dataset, no effect on the
   confirmatory design).
5. Lichens 2 (full 5-class) dropped from Phase B (user instruction) -- removes 10 of the 20
   pre-registered confirmatory tests; they show as "missing," not fabricated or silently dropped.

**One operational note, not a pre-registration deviation:** a single unit
(`lichens1/smartattention_v0`, repeat 0 fold 3) was absent after a run that otherwise reported zero
errors across exactly as many records as units submitted -- most likely a job-accounting edge case
at launch rather than a crash, since the run's own error counter stayed at zero. It was re-run in
isolation afterward (resumability confirmed it was the only gap) and is included in every result
above.

**Code review.** `robustness.py`, `robust_analysis.py` and their 22 unit tests were written by a
Sonnet subagent against a lead-written implementation spec (`PHASEB_IMPL_SPEC.md`), then reviewed
line-by-line before any run: two issues were found and fixed before launch (a T1 record's confusion
matrix was summed instead of averaged across subsets, leaving `sum(confusion) != n`; GPU memory was
not freed between the ~10 sequential fits per unit, unlike the existing harness convention). A
mandated smoke test (2 models, 1 fold, real GPU/data) reproduced the clean record's balanced
accuracy to within 0.14 points before the full run was launched. Separately, a pre-existing
encoding bug was found and fixed: every committed `.md` table (both Phase A and Phase B) was being
written with Windows' default (non-UTF-8) encoding, silently corrupting "+-" and "-" characters;
all affected tables were regenerated.

Full test suite: 153 passed (131 existing + 22 new), both before and after Phase B.
