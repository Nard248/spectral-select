# Generalization Changelog

Chronological record of every action touching the general channel-selection effort.

---

## 2026-05-23 [P0 · Lit scan]
**Action:** Background researcher mapped the channel-selection prior art.
**Why:** Position the paper correctly before committing to the novelty framing.
**Result:** "Unsupervised conditional discrete" cell is not empty; Concrete Autoencoder is
the must-beat baseline; report LOSO macro-F1. See RESEARCH_LOG 2026-05-23.

## 2026-05-23 [Brainstorming · Spec + Plan]
**Action:** Wrote + committed design spec and P1+P2 implementation plan via the superpowers
brainstorming -> writing-plans workflow.
**Files:** `docs/superpowers/specs/2026-05-23-general-channel-selection-design.md`,
`docs/superpowers/plans/2026-05-23-general-channel-selection.md`
**Why:** Standalone-paper effort needs a durable, reviewed design before code.

## 2026-05-23 [P1 · Package + engine]
**Action:** Built and tested `channel_select/` (protocols, data, engine, temporal AE,
training, PAMAP2 raw + MONSTER adapters). 20 tests passing.
**Files:** `channel_select/**`, `tests/channel_select/**`
**Why:** Shared domain-agnostic engine + HAR adapters. `spectral_select/` core untouched.
**Result:** Engine generalizes the HSI perturbation/influence/MMR math over arbitrary
group/channel/axis layouts. Two TDD-surfaced fixes: float64 variance clamp; MMR test
redesigned to isolate redundancy as the deciding factor.

## 2026-05-23 [P2 · PAMAP2 data + slice]
**Action:** UCI unreachable from environment; downloaded ungated HF MONSTER PAMAP2
(monster-monash/PAMAP2, 1.6 GB, (38856,52,100)). Ran `experiments/general_pamap2_slice.py`.
**Files:** `Data/Raw/PAMAP2_MONSTER/` (gitignored — large), `experiments/general_pamap2_slice.py`
**Why:** First real cross-domain validation; P2 go/no-go gate.
**Result:** Engine transfers to HAR. P2 GATE PASSED — AE-perturb beats variance baseline in
mid-K regime (K=7: 0.696 vs 0.574; K=10: 0.743 vs 0.659) and beats random; variance < random
at K=7 (over-picks redundant channels), illustrating the dependency-aware advantage. Loses at
K=3 (normalization hypothesis logged). Ceiling (all 27) = 0.828. See RESEARCH_LOG 2026-05-23.

## 2026-05-23 [P2 -> P3 · Full LOSO robustness]
**Action:** Launched `experiments/general_pamap2_loso.py` (8 subjects, AE/variance/random at
K=5/7/10, mean+/-std; one greedy-MMR selection per fold, evaluate K prefixes).
**Why:** Confirm the mid-K advantage is robust, not a subject-5 artifact, before building
the heavy P3 baselines.
**Result:** Mixed. AE-perturb robustly beats variance (+0.09 at K=7/10) and is much more
stable across subjects (std ~0.07 vs ~0.15), but TIES random under full LOSO (the subject-5
win was a favorable fold). Likely causes: crude mean+std feature + PAMAP2 redundancy. Next:
diagnostic Concrete-AE/mRMR on PAMAP2, learned downstream features, prioritize Opportunity.
See RESEARCH_LOG 2026-05-23. Do not headline "beats random on PAMAP2".

## 2026-05-23 [Workspace]
**Action:** Created `generalization/` workspace (MASTER_PLAN, RESEARCH_LOG, CHANGELOG,
baselines/, figures/, reports/), mirroring `revision/`.
**Why:** Durable paper-effort record.
