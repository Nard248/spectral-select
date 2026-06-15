# Repository Cleanup & Reorganization — Design Spec

- **Date:** 2026-06-15
- **Author:** Narek Meloyan (sole author)
- **Status:** Approved design — pending spec review → implementation plan
- **Repo:** `4D-Hyperspectral-Unsupervised-Clustering` (package name `spectral-select`)
- **Working branch at start:** `ExperimentsOnDrops` (local-only)

## 1. Goal

Turn a 27 GB, 947 MB-`.git`, multi-branch research monorepo that has accreted across many
iterations into a **well-defined, industry-standard, easy-to-scale repository**, by:

1. Reorganizing into a clean, conventional top-level layout (one repo).
2. Corralling junk into a single gitignored `archive/` (move, do not delete).
3. Shrinking `.git` (947 MB → ~300 MB) by purging large tracked binaries from history.
4. Unifying the two parallel selection engines onto a shared core.
5. Consolidating branches and the standalone preprocessor app, with full provenance preserved.

Non-goal: changing the science, the results, or any published numbers.

## 2. Findings that motivate this (from the 6-facet read-only audit)

1. **Git bloat ≠ disk bloat.** 27 GB on disk is mostly correctly-gitignored `Data/` (13 G),
   `.venv` (4 G), `archive/` (7.1 G). The 947 MB `.git` is caused by **tracked binaries in
   history**: `Archive.zip` (564 M), `Showcase_Poster.zip`, `MasterThesis.zip`, `paper.zip`,
   `revision/baselines/*.npz` (~100 M). Only 316 files are tracked total.
2. **Two engines, duplicated math.** `spectral_select.Analyzer` (class API, in `pyproject`) and
   `channel_select.engine.run_selection` (functional API, **not** in `pyproject`) implement the
   same perturbation/selection algorithm with mirrored config fields. ~200 LOC shared logic in
   two places.
3. **Publications scattered + duplicated.** 7 venues at root; four folder+zip pairs; two competing
   CommsAIComputing copies (`_Submission` tracked vs `_Overleaf` untracked but newer); legacy
   `Paper Source/`.
4. **Data redundancy (~13 G).** Four overlapping Lichens processings (~9 G), three Drop Data
   variants, orphaned `Lime` raw (107 M), `Sample Processed`/`Sample Export 2` (330 M, no code
   refs), empty placeholders, a stray root `spectra_masked.pkl`. No data contract.
5. **Branches are mostly ancestors.** `main`, `main-cleanup`, `publication/cleanup`,
   `archive/March2026` are ancestors of `ExperimentsOnDrops`. Unique work only on
   `collagen-experiments` (tuning scripts) and `information-detection/initial-setup` (Kiwi/LIME
   notebooks). **`ExperimentsOnDrops` is local-only** — data-loss risk.
6. **Loose cruft.** `TM.py` (unrelated Turing-machine artifact), `Untitled.ipynb`, redundant
   `run_preprocessor.py`, empty `scripts/`, `notebooks/` holding only cached outputs, `.remember/`
   not gitignored, regenerable `htmlcov/`/`coverage*`.

## 3. Decisions (locked with owner)

| # | Decision | Choice |
|---|---|---|
| 1 | Target shape | **Organized monorepo** (one repo) |
| 2 | Two engines | **Unify on a shared `selection_core`** |
| 3 | Git history | **Rewrite + force-push** (mirror-backup first) |
| 4 | "cversion" software | = **MEHSI Preprocessor PyQt app**; consolidate canonical, tag rest |
| 5 | Package layout | **`src/` layout** |
| 6 | Data cleanup | **Archive, don't delete** (MANIFEST; confirm each move) |

## 4. Target top-level structure

```
4D-Hyperspectral-Unsupervised-Clustering/
├── README.md                 # rewritten: what it is + repo map + quickstart
├── LICENSE  ·  pyproject.toml  ·  .gitignore  ·  .github/workflows/
├── src/
│   ├── selection_core/        # NEW: shared perturbation/selection algorithm
│   ├── spectral_select/       # HSI library (depends on selection_core)
│   ├── channel_select/        # general/temporal engine (depends on selection_core)
│   └── mehsi_preprocessor/    # PyQt6 GUI app (canonical preprocessor)
├── experiments/
│   ├── README.md              # run order, dataset requirements, lifecycle
│   ├── <~12 reusable drivers>
│   ├── pamap2/                # general_pamap2_*.py
│   └── _archive/2026_paper_runs/   # ~31 one-off paper/figure scripts
├── examples/                  # 3 notebooks (kept)
├── tests/                     # kept; coverage extended to all packages
├── docs/                      # kept; + ARCHITECTURE.md, data contract
├── publications/
│   ├── tpami/  commsai_computing/  codassca2026/
│   └── master_thesis/  iasim_poster/  generalization/
├── data/                      # gitignored; data/README.md = dataset contract
└── archive/                   # gitignored junk drawer + MANIFEST.md
    ├── legacy_wavelength_analysis/   (existing 7.1 G)
    ├── redundant_data/  ·  snapshots/  ·  misc/
    └── MANIFEST.md
```

## 5. Detailed migration map

### Packages → `src/`
- `git mv` `spectral_select/`, `channel_select/`, `mehsi_preprocessor/` → `src/`.
- Create `src/selection_core/` (Phase 7).
- `pyproject.toml`: `where = ["src"]`; `include = ["selection_core*","spectral_select*","channel_select*","mehsi_preprocessor*"]`.
- `python -m mehsi_preprocessor` remains the launch entry; remove redundant `run_preprocessor.py` (→ `archive/misc/`).

### Experiments
- Keep at top: `run_master_experiment.py`, `analyze_results.py`, `extract_wavelengths.py`,
  `export_tiffs.py`, `generate_figures.py`, `export_wavelength_combinations.py`, `rerun_knn.py`.
- `general_pamap2_*.py` → `experiments/pamap2/`.
- ~31 one-offs (`drop_data_*` ×16, `collagen_*` ×3, `pepsin_*` ×6, `poster_*` ×6) →
  `experiments/_archive/2026_paper_runs/`.
- Commit the 2 currently-untracked active scripts (`drop_data_radiometric_rerun.py`,
  `drop_data_radiometric_knn.py`).
- Add `experiments/README.md` documenting lifecycle + dataset requirements.

### Publications → `publications/`
- `revision/` + `paper/` → `publications/tpami/`.
- `CommsAIComputing_Submission/` + `CommsAIComputing_Overleaf/` → `publications/commsai_computing/`
  (Overleaf = canonical source since newer; keep Submission as frozen snapshot subdir; drop the dup zip).
- `CODASSCA2026_Submission/` → `publications/codassca2026/`.
- `MasterThesis_Narek_Meloyan/` → `publications/master_thesis/`.
- `Showcase_Poster/` → `publications/iasim_poster/`.
- `generalization/` (paper, docs, figures, reports) → `publications/generalization/`.
- `Paper Source/` (legacy) → `archive/misc/`.

### Archive (move, never delete data)
- `TM.py`, `run_preprocessor.py`, `model_output/`, `visualizations/`, stray `Data/processed/spectra_masked.pkl`
  → `archive/misc/` (after verifying no live code refs).
- Orphan/redundant datasets → `archive/redundant_data/` **after per-item confirmation**, logged in MANIFEST.
- Large zips/binaries staged in `archive/snapshots/` before history purge (so content survives if needed).

### Cruft (delete — regenerable)
- `htmlcov/`, `.coverage`, `coverage.json`, `.pytest_cache/`, `*.egg-info/`, `Untitled.ipynb`,
  `texput.log`, empty `scripts/`, `.ipynb_checkpoints/`.
- Add `.remember/`, `*.npz`, `*.zip`, `*.docx`, `*.pptx` rules to `.gitignore` to block *future*
  additions. (Gitignore does not untrack already-committed files; existing small `.docx/.pptx`
  deliverables stay tracked — see §9 for the Phase-6 purge list.)

### Branches
- **Backup:** `git clone --mirror` to a safe external path before anything.
- Push `ExperimentsOnDrops` → origin immediately (independent data-loss fix).
- Tag-then-retire: `archive/March2026` → tag `v1.0-preprocessor`; `main-cleanup`,
  `publication/cleanup`, `archive/pre-publication-cleanup` → tag (if unique) then delete.
- `collagen-experiments`: cherry-pick `collagen_tuning.py` + `reprocess_with_metadata.py` into
  `experiments/_archive/`, then tag + delete.
- `information-detection/initial-setup`: tag `archive-kiwi-lime-exploration`, then delete.
- End state: cleaned `ExperimentsOnDrops` becomes the new `main`.

## 6. Phased execution plan

Each phase = one reviewable checkpoint (commit/branch); tests stay green throughout.

| Phase | Work | Acceptance criteria | Risk | Rollback |
|---|---|---|---|---|
| **0 Backup** | mirror-clone; record `.git` size + object list; push `ExperimentsOnDrops` | backup verified restorable; branch on origin | low | n/a |
| **1 Cruft sweep** | delete regenerable artifacts; gitignore `.remember/` | `git status` clean of cruft; tests green | none | git restore |
| **2 archive/ consolidation** | move junk + (confirmed) orphan data → `archive/`; write MANIFEST | MANIFEST lists every moved item w/ original path | low (moves) | move back per MANIFEST |
| **3 publications/ regroup** | consolidate venues; remove folder+zip dupes | one subdir per venue; no dup zips in tree | low | git mv back |
| **4 src/ layout + experiments split** | `git mv` packages → `src/`; fix `pyproject`/imports; split experiments | `pip install -e .` works; full test suite green | medium | git revert |
| **5 Branch consolidation** | tags + cherry-picks + retire stale branches | all unique work preserved (tag or merged); namespace clean | medium | tags are permanent refs |
| **6 History rewrite** | `git filter-repo` purge `Archive.zip`+zips+npz; force-push | `.git` ≤ ~350 MB; clone test passes; origin updated | **high** | restore from Phase-0 mirror |
| **7 Engine unify (TDD)** | characterization tests → extract `selection_core` → both engines use it | identical numeric output pre/post; coverage on core | medium | git revert (isolated) |

Phase 7 is **separable** — the clean repo is fully delivered after Phase 6; the unify can run as
its own session.

## 7. Safety rails (non-negotiable)
- No `Data/` file is deleted — only moved to `archive/redundant_data/`, confirmed per item, logged.
- History rewrite only after Phase-0 mirror backup, on its own checkpoint, with before/after object
  list shown for approval.
- Tests green before and after Phase 4; engine unify (Phase 7) must produce byte-identical results.
- Every phase is an independent commit/branch the owner can review and revert.

## 8. Risks & mitigations
- **History rewrite breaks existing clones / origin refs** → sole author; mirror backup; force-push
  done deliberately; document the new clone command.
- **Hidden code refs to moved data/paths** → grep for path strings before each move; tests + a
  smoke run of `run_master_experiment.py` after Phase 4 (data permitting).
- **Wrong "canonical" dataset archived** → present candidates (esp. the 4 Lichens variants) and
  confirm before moving; nothing irreversible.
- **Engine unify regression** → TDD characterization tests pin current outputs first.

## 9. To verify during execution (not blocking design)
- Which Lichens processing is canonical for the record: `Lichens Dataset 1` (TPAMI-named) and/or
  `Lichens_2` (matches raw source name)? Others → archive after confirm.
- Whether all three Drop Data variants stay pending the revision's Phase 4, or non-publication
  variants archive now.
- Exact binary list to purge in Phase 6 (proposed: `Archive.zip`, all `*.zip` snapshots, the
  `revision/**/*.npz` cubes; **keep** small `.docx/.pptx` deliverables tracked, gitignore future).
- Newest/most-complete preprocessor copy across `ExperimentsOnDrops` vs `collagen-experiments`.

## 10. Out of scope / future
- Renaming the GitHub repo to match `spectral-select` (owner's call; cosmetic).
- DVC / external data-versioning (documented as a future option in `data/README.md`).
- Publishing `spectral-select` to PyPI (the `src/` layout makes this straightforward later).
