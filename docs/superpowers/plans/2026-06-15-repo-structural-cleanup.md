# Repository Structural Cleanup Implementation Plan (Phases 0–6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the 4D-Hyperspectral research monorepo into an industry-standard, scalable layout — `src/` packages, `publications/` by venue, a single gitignored `archive/` for junk — and shrink `.git` from 947 MB to ~300 MB by purging large binaries from history.

**Architecture:** Mechanical, test-gated, back-loaded risk. Phases 0–3 are zero/low-risk (backup, delete-regenerable, `git mv`). Phase 4 is test-gated (`src/` move). Phases 5–6 are irreversible (branch deletion, history rewrite) and run only after a verified mirror backup. Each phase is one or more commits the owner reviews before the next.

**Tech Stack:** git 2.50, `git-filter-repo` (installed in Phase 6), Python 3.11 + setuptools src-layout, pytest.

**Source spec:** `docs/superpowers/specs/2026-06-15-repo-cleanup-reorg-design.md`

**Engine unification (was Phase 7) is deferred to its own plan** — written after Phase 6, once the engines are re-read in `src/`. Acceptance for that later plan: a `src/selection_core/` extracted from `spectral_select/analyzer.py` + `channel_select/engine.py`, both engines delegating to it, with characterization tests proving byte-identical numeric output.

---

## Conventions

- All paths are relative to the repo root: `/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering`.
- `archive/` is **gitignored** — anything moved there leaves git tracking but stays on disk (recoverable).
- "Moved (untracked)" = plain `mv` (item is gitignored/untracked). "Moved (tracked)" = `cp` to archive then `git rm`, or `git mv` for in-tree relocations that stay tracked.
- Provenance of every archived item is recorded in `docs/ARCHIVE_MANIFEST.md` (tracked).
- Run all commands from the repo root. Activate the venv first: `source .venv/bin/activate`.

---

## Phase 0 — Backup & Safety

### Task 0.1: Mirror-clone backup

**Files:** none (creates external backup)

- [ ] **Step 1: Record current state**

Run:
```bash
du -sh .git && git ls-files | wc -l && git rev-parse HEAD && git branch -a
```
Expected: `.git` ≈ 947M; ~316 tracked files; current HEAD on `ExperimentsOnDrops`. Note these numbers.

- [ ] **Step 2: Create a mirror backup outside the repo**

Run:
```bash
git clone --mirror . ~/4d-hyperspectral-backup-20260615.git
du -sh ~/4d-hyperspectral-backup-20260615.git
```
Expected: a `.git` mirror at `~/4d-hyperspectral-backup-20260615.git` (≈ same size as `.git`).

- [ ] **Step 3: Verify the backup is restorable**

Run:
```bash
git -C ~/4d-hyperspectral-backup-20260615.git rev-list --count --all && git -C ~/4d-hyperspectral-backup-20260615.git for-each-ref | wc -l
```
Expected: nonzero commit count; all refs present. **Do not proceed past Phase 0 unless this succeeds.**

### Task 0.2: Push the local-only working branch to origin — DEFERRED to Phase 6.4

**Files:** none (remote push)

> **Amended 2026-06-15:** The push fails at GitHub's 100MB pre-receive hook because
> `Archive.zip` (538MB) and `revision/baselines/lichens_cube.npz` (62MB) are in this branch's
> history — which is precisely why `ExperimentsOnDrops` was never pushed. The origin push is
> therefore impossible until Phase 6 purges those blobs, and is folded into **Task 6.4**
> (force-push after the rewrite). Until then, the **local mirror (Task 0.1)** is the backup.
> Owner confirmed (2026-06-15) to proceed on the local mirror.

### Task 0.3: Establish the green-test + size baseline

**Files:** none

- [ ] **Step 1: Run the test suite (fast subset) to record a baseline**

Run:
```bash
pip install -e '.[dev]' >/dev/null 2>&1; pytest -q -m "not slow and not notebook" | tail -20
```
Expected: all selected tests PASS. **Record the pass count** — Phase 4 must match it.

- [ ] **Step 2: Record working-tree size**

Run:
```bash
du -sh . --exclude=.git --exclude=.venv 2>/dev/null || du -sh .
```
Expected: a baseline number to compare against after the archive moves.

---

## Phase 1 — Cruft Sweep (regenerable artifacts + .gitignore)

### Task 1.1: Delete regenerable/untracked cruft

**Files:**
- Delete: `htmlcov/`, `.coverage`, `coverage.json`, `.pytest_cache/`, `spectral_select.egg-info/`, `Untitled.ipynb`, `texput.log`, `.ipynb_checkpoints/`, `experiments/__pycache__/`, `scripts/` (empty), all `**/.DS_Store`

- [ ] **Step 1: Confirm none are tracked (safety)**

Run:
```bash
git ls-files htmlcov coverage.json .coverage Untitled.ipynb texput.log scripts | wc -l
```
Expected: `0` (all untracked). If nonzero, stop and inspect.

- [ ] **Step 2: Delete them**

Run:
```bash
rm -rf htmlcov .coverage coverage.json .pytest_cache spectral_select.egg-info \
       Untitled.ipynb texput.log .ipynb_checkpoints experiments/__pycache__ scripts
find . -name .DS_Store -not -path './.venv/*' -not -path './.git/*' -delete
```
Expected: no errors.

- [ ] **Step 3: Verify**

Run: `ls htmlcov scripts Untitled.ipynb 2>&1 | head`
Expected: "No such file or directory" for each. No commit needed (all were untracked).

### Task 1.2: Extend `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append tool-state + binary rules**

Append to `.gitignore`:
```gitignore

# Tool state
.remember/

# Binary artifacts that should not be version-controlled going forward
# (existing tracked .docx/.pptx deliverables remain tracked; these block NEW additions)
*.zip
*.npz
*.docx
*.pptx
```

- [ ] **Step 2: Verify the ignore takes effect for new files**

Run: `git check-ignore -v <(echo) >/dev/null 2>&1; git status --porcelain | grep -E '\.zip$|\.npz$' || echo "no new zip/npz staged"`
Expected: no new `.zip`/`.npz` show as untracked-and-stageable.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore .remember/ and block new binary artifacts (zip/npz/docx/pptx)"
```

---

## Phase 2 — `archive/` Consolidation

### Task 2.1: Create archive structure + manifest scaffold

**Files:**
- Create: `archive/redundant_data/`, `archive/snapshots/`, `archive/misc/`, `docs/ARCHIVE_MANIFEST.md`

- [ ] **Step 1: Create dirs**

Run:
```bash
mkdir -p archive/redundant_data archive/snapshots archive/misc
```

- [ ] **Step 2: Create the tracked manifest**

Create `docs/ARCHIVE_MANIFEST.md`:
```markdown
# Archive Manifest

Everything under the gitignored `archive/` directory, with its original path, date moved, and reason.
Items here are NOT deleted — they remain on disk and are recoverable.

| Original path | Archived to | Date | Reason |
|---|---|---|---|
```

- [ ] **Step 3: Commit the scaffold**

```bash
git add docs/ARCHIVE_MANIFEST.md
git commit -m "docs: add archive manifest scaffold"
```

### Task 2.2: Archive unrelated/superseded code & outputs (no data)

**Files:**
- Move (tracked → untrack): `run_preprocessor.py`
- Move (untracked): `TM.py`, `Paper Source/`, `model_output/`, `visualizations/`

- [ ] **Step 1: Move untracked items**

Run:
```bash
mv TM.py "Paper Source" model_output visualizations archive/misc/ 2>/dev/null
```
Expected: no errors (each exists).

- [ ] **Step 2: Archive the tracked redundant launcher**

`run_preprocessor.py` is redundant with `python -m mehsi_preprocessor`.
Run:
```bash
cp run_preprocessor.py archive/misc/ && git rm run_preprocessor.py
```
Expected: staged deletion of `run_preprocessor.py`; copy now in `archive/misc/`.

- [ ] **Step 3: Append to manifest**

Add these rows to `docs/ARCHIVE_MANIFEST.md`:
```markdown
| TM.py | archive/misc/TM.py | 2026-06-15 | Unrelated Turing-machine class artifact |
| Paper Source/ | archive/misc/Paper Source/ | 2026-06-15 | Legacy paper wrapper, superseded by paper/ |
| model_output/ | archive/misc/model_output/ | 2026-06-15 | Old .pth/.npy training artifacts, superseded by results/ |
| visualizations/ | archive/misc/visualizations/ | 2026-06-15 | Old viz outputs, superseded by results/ |
| run_preprocessor.py | archive/misc/run_preprocessor.py | 2026-06-15 | Redundant launcher; use `python -m mehsi_preprocessor` |
```

- [ ] **Step 4: Verify no live code imports the moved launcher**

Run: `grep -rn "run_preprocessor" src experiments tests 2>/dev/null || echo "no refs"`
Expected: "no refs" (or only doc references).

- [ ] **Step 5: Commit**

```bash
git add -A docs/ARCHIVE_MANIFEST.md run_preprocessor.py
git commit -m "chore: archive unrelated/superseded code & outputs (TM.py, Paper Source, model_output, visualizations, run_preprocessor.py)"
```

### Task 2.3: Archive orphan/redundant DATA — CHECKPOINT (owner confirms each)

**Files:** moves within gitignored `Data/` → `archive/redundant_data/` (untracked; no git ops)

> **STOP — owner confirmation required before running.** None of these are referenced by code per the audit, but they are irreproducible. Present this list; move only the confirmed items. Empty dirs are safe.

- [ ] **Step 1: Show candidates with sizes**

Run:
```bash
du -sh "Data/Raw/Lime" "Data/Raw/Lichens" "Data/processed/Sample Processed" \
       "Data/processed/Sample Export 2" "Data/processed/YourSample" \
       "Data/processed/spectra_masked.pkl" "Data/processed/spectra_unmasked.pkl" \
       "Data/processed/LichensProcessed" "Data/processed/Lichhens 2 - Processed" \
       "Data/processed/lichens_data_cropped" 2>/dev/null
```
Expected: sizes for each. **Confirm with owner which to archive.** Canonical-to-KEEP per spec: `Data/processed/Lichens Dataset 1`, `Data/processed/Lichens_2`, all `Drop Data*`, `Collagen*`, `Sponges*`, `PAMAP2_MONSTER`.

- [ ] **Step 2: Remove confirmed-empty placeholders**

Run (only the confirmed-empty ones):
```bash
rmdir "Data/Raw/Lichens" "Data/processed/YourSample" 2>/dev/null && echo "removed empties"
```

- [ ] **Step 3: Move confirmed orphans/redundant data**

Run ONLY on owner-confirmed items, e.g.:
```bash
mv "Data/Raw/Lime" "Data/processed/Sample Processed" "Data/processed/Sample Export 2" \
   "Data/processed/spectra_masked.pkl" "Data/processed/spectra_unmasked.pkl" \
   archive/redundant_data/
# Lichens redundant variants — move ONLY those the owner names as non-canonical:
# mv "Data/processed/LichensProcessed" "Data/processed/Lichhens 2 - Processed" "Data/processed/lichens_data_cropped" archive/redundant_data/
```

- [ ] **Step 4: Append every moved item to the manifest & commit it**

Update `docs/ARCHIVE_MANIFEST.md` with one row per moved item, then:
```bash
git add docs/ARCHIVE_MANIFEST.md
git commit -m "docs: record archived orphan/redundant datasets in manifest"
```
(The data files themselves are gitignored — only the manifest is committed.)

### Task 2.4: Write the dataset contract (`docs/DATA.md`)

**Files:**
- Create: `docs/DATA.md`

- [ ] **Step 1: Document canonical vs archived datasets**

Create `docs/DATA.md` describing: the gitignored `Data/` layout (`Data/Raw/` + `Data/processed/`); the canonical datasets to keep (`Lichens Dataset 1`, `Lichens_2`, `Drop Data` + Cropped + Radiometric, `Collagen_Acetic_Acid`, `Collagen Pepsin`, `Sponges Acid Group 1`, `PAMAP2_MONSTER`); which experiment/script consumes each; what was moved to `archive/redundant_data/` (cross-reference `ARCHIVE_MANIFEST.md`); and a note that data is not version-controlled (future option: DVC / external storage).

- [ ] **Step 2: Commit**

```bash
git add docs/DATA.md
git commit -m "docs: add dataset contract (canonical vs archived, code references)"
```

---

## Phase 3 — `publications/` Regroup

### Task 3.1: Move tracked venue folders into `publications/`

**Files:**
- Move (tracked, `git mv`): `paper/` → `publications/tpami/paper/`; `revision/` → `publications/tpami/revision/`; `Showcase_Poster/` → `publications/iasim_poster/`; `MasterThesis_Narek_Meloyan/` → `publications/master_thesis/`; `CODASSCA2026_Submission/` → `publications/codassca2026/`; `generalization/` → `publications/generalization/`

- [ ] **Step 1: Create and move**

Run:
```bash
mkdir -p publications/tpami
git mv paper publications/tpami/paper
git mv revision publications/tpami/revision
git mv Showcase_Poster publications/iasim_poster
git mv MasterThesis_Narek_Meloyan publications/master_thesis
git mv CODASSCA2026_Submission publications/codassca2026
git mv generalization publications/generalization
```
Expected: renames staged, no errors.

- [ ] **Step 2: Remove the stale Word lock file**

`git status` shows `CODASSCA2026_Submission/~$DASSCA2026_Meloyan_ShortPaper.docx` as deleted. Confirm it's gone:
```bash
git status --porcelain | grep '~\$' || echo "no lock files"
```
If a `~$` lock file is still staged, `git rm` it.

- [ ] **Step 3: Verify build scripts' relative paths still resolve**

Run: `grep -rn "\.\./\.\./Data\|results/" publications/*/build*.py publications/**/build*.py 2>/dev/null | head`
Expected: note any absolute/relative path assumptions to fix; most build scripts use repo-root-relative or their own dir. Flag breakages but they don't block the move.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: consolidate publications into publications/<venue>/"
```

### Task 3.2: Remove tracked LaTeX backup files

**Files:**
- Delete (tracked): `publications/tpami/paper/sections/*.bak.20260512` (8 files)

- [ ] **Step 1: Remove the .bak files**

Run:
```bash
git rm publications/tpami/paper/sections/*.bak.20260512
```
Expected: 8 files staged for deletion.

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: drop stale LaTeX .bak.20260512 section backups"
```

### Task 3.3: Consolidate the two CommsAIComputing copies (untracked)

**Files:**
- Move (untracked): `CommsAIComputing_Overleaf/` → `publications/commsai_computing/overleaf/`; `CommsAIComputing_Submission/` → `publications/commsai_computing/submission/`; `CommsAIComputing_Overleaf.zip` → `archive/snapshots/`

- [ ] **Step 1: Move them**

Run:
```bash
mkdir -p publications/commsai_computing
mv CommsAIComputing_Overleaf publications/commsai_computing/overleaf
mv CommsAIComputing_Submission publications/commsai_computing/submission
mv CommsAIComputing_Overleaf.zip archive/snapshots/
```
Expected: Overleaf (canonical editing copy) and the frozen Submission now live side by side; both remain untracked (active drafts).

- [ ] **Step 2: Record in manifest + commit**

Add a manifest row for `CommsAIComputing_Overleaf.zip → archive/snapshots/`. Then:
```bash
git add docs/ARCHIVE_MANIFEST.md
git commit -m "chore: group CommsAIComputing overleaf+submission under publications/"
```
(Note for owner: if you want the `.tex` sources version-controlled, force-add them later with `git add -f publications/commsai_computing/overleaf/*.tex`.)

### Task 3.4: Remove tracked zips from the tree (content preserved)

**Files:**
- Move-then-untrack (tracked): `Archive.zip`, `paper.zip`, `MasterThesis_Narek_Meloyan.zip`, `Showcase_Poster.zip`, `publications/iasim_poster/dataset_plots.zip`, `publications/generalization/paper/Meloyan_GeneralChannelSelection_LaTeX.zip`

- [ ] **Step 1: Preserve content locally, then untrack from the tree**

Run:
```bash
cp Archive.zip paper.zip MasterThesis_Narek_Meloyan.zip Showcase_Poster.zip archive/snapshots/ 2>/dev/null
git rm Archive.zip paper.zip MasterThesis_Narek_Meloyan.zip Showcase_Poster.zip
git rm publications/iasim_poster/dataset_plots.zip
git rm "publications/generalization/paper/Meloyan_GeneralChannelSelection_LaTeX.zip"
```
Expected: 6 zips staged for deletion; the 4 root zips copied into `archive/snapshots/`.

- [ ] **Step 2: Verify no `.zip` remains tracked**

Run: `git ls-files '*.zip' | head` → Expected: empty output.

- [ ] **Step 3: Record in manifest + commit**

Add manifest rows for the 4 root zips. Then:
```bash
git add docs/ARCHIVE_MANIFEST.md
git commit -m "chore: untrack zip snapshots from tree (content preserved in archive/snapshots/; history purged in Phase 6)"
```

---

## Phase 4 — `src/` Layout + Experiments Split + pyproject + CI

### Task 4.1: Move packages into `src/`

**Files:**
- Move (tracked, `git mv`): `spectral_select/` → `src/spectral_select/`; `channel_select/` → `src/channel_select/`; `mehsi_preprocessor/` → `src/mehsi_preprocessor/`
- Modify: `pyproject.toml`

- [ ] **Step 1: Move the three packages**

Run:
```bash
mkdir -p src
git mv spectral_select src/spectral_select
git mv channel_select src/channel_select
git mv mehsi_preprocessor src/mehsi_preprocessor
```
Expected: renames staged.

- [ ] **Step 2: Update `pyproject.toml` package discovery**

In `pyproject.toml`, replace the `[tool.setuptools.packages.find]` block:
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["spectral_select*", "channel_select*", "mehsi_preprocessor*"]
```

- [ ] **Step 3: Reinstall editable and verify imports resolve from src/**

Run:
```bash
pip install -e '.[dev]' >/dev/null && python -c "import spectral_select, channel_select, mehsi_preprocessor; print('ok')"
```
Expected: `ok`.

- [ ] **Step 4: Run the full fast test suite — must match Phase 0 baseline**

Run:
```bash
pytest -q -m "not slow and not notebook" | tail -20
```
Expected: same pass count as Task 0.3 Step 1. If any fail, inspect import paths before committing.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: adopt src/ layout for spectral_select, channel_select, mehsi_preprocessor"
```

### Task 4.2: Split `experiments/` into tiers

**Files:**
- Create: `experiments/pamap2/`, `experiments/_archive/2026_paper_runs/`
- Move: 5 general scripts → `pamap2/`; 31 one-off scripts → `_archive/2026_paper_runs/`
- First track 2 untracked scripts

- [ ] **Step 1: Track the 2 active untracked scripts**

Run:
```bash
git add experiments/drop_data_radiometric_rerun.py experiments/drop_data_radiometric_knn.py
```

- [ ] **Step 2: Move the 5 generalization/PAMAP2 scripts**

Run:
```bash
mkdir -p experiments/pamap2
git mv experiments/general_make_figures.py experiments/general_pamap2_baseline_diag.py \
       experiments/general_pamap2_loso.py experiments/general_pamap2_richfeat_diag.py \
       experiments/general_pamap2_slice.py experiments/pamap2/
```

- [ ] **Step 3: Move the 31 one-off paper/figure scripts**

Run:
```bash
mkdir -p experiments/_archive/2026_paper_runs
git mv experiments/collagen_classifier_comparison.py experiments/collagen_full_analysis.py \
  experiments/collagen_hyperparameter_tuning.py \
  experiments/drop_data_cropped_pipeline.py experiments/drop_data_dim_sweep.py \
  experiments/drop_data_export_slides.py experiments/drop_data_export_slides_cropped.py \
  experiments/drop_data_influence_vs_fratio.py experiments/drop_data_inspect.py \
  experiments/drop_data_montage.py experiments/drop_data_norm_fix.py \
  experiments/drop_data_post_analysis.py experiments/drop_data_preprocess.py \
  experiments/drop_data_radiometric_knn.py experiments/drop_data_radiometric_rerun.py \
  experiments/drop_data_selection_sweep.py experiments/drop_data_smoke_test.py \
  experiments/drop_data_snr_rerank.py experiments/drop_data_spectra_explore.py \
  experiments/pepsin_docx_report.py experiments/pepsin_export_tiffs_range.py \
  experiments/pepsin_fix_embedded_figures.py experiments/pepsin_fix_heatmap_and_export_tiffs.py \
  experiments/pepsin_iasim_abstract.py experiments/pepsin_paper_figures.py \
  experiments/poster_arch_ushape_clean.py experiments/poster_dataset_plots_unified.py \
  experiments/poster_v2_architecture_figures.py experiments/poster_v2_pepsin_roi_overlay.py \
  experiments/poster_v2_results_table_xlsx.py experiments/poster_v2_wireframe_mockup.py \
  experiments/_archive/2026_paper_runs/
```

- [ ] **Step 4: Verify the active tier is exactly the 7 reusable drivers**

Run: `ls -1 experiments/*.py`
Expected: `analyze_results.py`, `export_tiffs.py`, `export_wavelength_combinations.py`, `extract_wavelengths.py`, `generate_figures.py`, `rerun_knn.py`, `run_master_experiment.py` (7 files).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: tier experiments/ (reusable drivers / pamap2 / _archive paper runs)"
```

### Task 4.3: Rewrite `experiments/README.md`

**Files:**
- Modify: `experiments/README.md`

- [ ] **Step 1: Replace contents**

Write `experiments/README.md`:
```markdown
# Experiments

Three tiers:

## Reusable drivers (top level)
The maintained, reusable pipeline scripts. Run against canonical datasets.
- `run_master_experiment.py` — main selection+validation pipeline driver
- `analyze_results.py`, `extract_wavelengths.py`, `export_wavelength_combinations.py`
- `export_tiffs.py`, `generate_figures.py`, `rerun_knn.py`

## pamap2/ — generalization domain
Domain-agnostic channel selection on PAMAP2 (wearables). Uses `channel_select`.
- `general_pamap2_*.py`, `general_make_figures.py`

## _archive/2026_paper_runs/ — one-off paper & figure scripts
Dataset-specific, run-once scripts kept for provenance (collagen/drop_data/pepsin/poster).
Not part of the maintained surface; runnable but tied to specific data paths & 2026 submissions.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/README.md
git commit -m "docs: document experiments/ three-tier structure"
```

### Task 4.4: Extend CI coverage to channel_select

**Files:**
- Modify: `.github/workflows/test.yml`

- [ ] **Step 1: Update the pytest coverage flags**

In `.github/workflows/test.yml`, change the pytest step's coverage target from `--cov=spectral_select` to:
```yaml
        run: pytest --cov=spectral_select --cov=channel_select --cov-report=xml
```

- [ ] **Step 2: Verify locally**

Run: `pytest -q --cov=spectral_select --cov=channel_select -m "not slow and not notebook" | tail -5`
Expected: runs and reports coverage for both packages.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: include channel_select in coverage"
```

---

## Phase 5 — Branch Consolidation

### Task 5.1: Recover the 2 unique collagen scripts

**Files:**
- Create (from branch): `experiments/_archive/2026_paper_runs/collagen_tuning.py`, `experiments/_archive/2026_paper_runs/reprocess_with_metadata.py`

- [ ] **Step 1: Check out only the two unique files from `collagen-experiments`**

Run:
```bash
git checkout collagen-experiments -- experiments/collagen_tuning.py experiments/reprocess_with_metadata.py
git mv experiments/collagen_tuning.py experiments/reprocess_with_metadata.py experiments/_archive/2026_paper_runs/
```
Expected: two files now under `_archive/2026_paper_runs/`.

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "chore: recover unique collagen scripts from collagen-experiments branch"
```

### Task 5.2: Tag everything that will be deleted (provenance)

**Files:** none (git tags)

- [ ] **Step 1: Create annotated tags**

Run:
```bash
git tag -a v1.0-preprocessor archive/March2026 -m "MEHSI Preprocessor app v1.0 (commit 5dd0583)"
git tag -a archive-pre-publication-cleanup archive/pre-publication-cleanup -m "Pre-publication state snapshot"
git tag -a archive-kiwi-lime-exploration information-detection/initial-setup -m "Kiwi/LIME unsupervised 4D clustering exploration"
git tag -a archive-collagen-experiments collagen-experiments -m "Collagen tuning branch snapshot"
```
Expected: 4 tags created.

- [ ] **Step 2: Verify tags resolve to commits**

Run: `git tag -l 'v1.0-preprocessor' 'archive-*' | cat && git rev-parse v1.0-preprocessor`
Expected: tags listed, SHA printed.

### Task 5.3: Delete stale branches — CHECKPOINT (remote deletion is outward-facing)

**Files:** none (branch deletion)

> **STOP — confirm with owner before deleting remote branches.** Local deletions are recoverable via tags/reflog; remote deletions are visible to anyone watching the repo.
>
> **Amended 2026-06-15 (owner choice):** Local branch deletion done now (Step 1). The **remote**
> branch deletions (Step 2) and **tag push** (Step 3) are **deferred to Task 6.4** so all
> outward-facing mutations happen together after the history rewrite. Steps 2–3 below are executed
> as part of 6.4, not here.

- [ ] **Step 1: Delete local stale branches**

Run:
```bash
git branch -D main-cleanup publication/cleanup archive/March2026 \
  archive/pre-publication-cleanup collagen-experiments information-detection/initial-setup
```
Expected: 6 branches deleted locally (all preserved via tags or as ancestors of `main`/`ExperimentsOnDrops`).

- [ ] **Step 2: Delete the corresponding remote branches**

Run:
```bash
git push origin --delete main-cleanup publication/cleanup collagen-experiments information-detection/initial-setup
```
Expected: remote branches deleted. (`archive/*` were local-only; skip any that error with "remote ref does not exist".)

- [ ] **Step 3: Push the provenance tags to origin**

Run:
```bash
git push origin v1.0-preprocessor archive-pre-publication-cleanup archive-kiwi-lime-exploration archive-collagen-experiments
```
Expected: 4 tags on origin.

- [ ] **Step 4: Verify final branch set**

Run: `git branch -a && git tag -l | cat`
Expected: local `main`, `ExperimentsOnDrops`; origin `main`, `ExperimentsOnDrops`; 4 archive tags.

---

## Phase 6 — History Rewrite — CHECKPOINT (destructive, irreversible)

> **STOP — owner must explicitly authorize this phase.** Requires the Phase 0 mirror backup to exist. Rewrites every commit SHA and force-pushes to origin.

### Task 6.1: Install git-filter-repo

**Files:** none

- [ ] **Step 1: Install into the venv**

Run:
```bash
pip install git-filter-repo && git filter-repo --version
```
Expected: a version string (e.g., `git filter-repo 2.x`). If `git filter-repo` isn't found on PATH after pip install, use `python -m git_filter_repo --version` and substitute that form below.

- [ ] **Step 2: Re-confirm the backup exists**

Run: `du -sh ~/4d-hyperspectral-backup-20260615.git`
Expected: the mirror from Task 0.1. **Do not proceed without it.**

### Task 6.2: Preview the blobs that will be purged

**Files:** none

- [ ] **Step 1: List the large blobs across all history**

Run:
```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $2>1000000 {print $2, $3}' | sort -rn | head -20
```
Expected: `Archive.zip` (~564M), the three `*.npz` cubes, and the `*.zip` snapshots dominate. Confirm this matches the intended purge set (all `*.zip` + all `*.npz`).

### Task 6.3: Rewrite history to drop all `.zip` and `.npz`

**Files:** none (rewrites history)

- [ ] **Step 1: Run filter-repo**

Run:
```bash
git filter-repo --path-glob '*.zip' --path-glob '*.npz' --invert-paths --force
```
Expected: filter-repo rewrites all commits, removing those blobs. It will also **remove the `origin` remote** (its safety behavior).

- [ ] **Step 2: Verify `.git` shrank**

Run:
```bash
git reflog expire --expire=now --all && git gc --prune=now --aggressive && du -sh .git
```
Expected: `.git` down to roughly ~300 MB (from 947 MB).

- [ ] **Step 3: Confirm no zip/npz remain in any history**

Run:
```bash
git rev-list --objects --all | grep -E '\.zip$|\.npz$' || echo "clean: no zip/npz in history"
```
Expected: `clean: no zip/npz in history`.

### Task 6.4: Re-add origin and force-push — CHECKPOINT

**Files:** none

- [ ] **Step 1: Re-add the remote**

Run:
```bash
git remote add origin https://github.com/Nard248/4D-Hyperspectral-Unsupervised-Clustering.git
```

- [ ] **Step 2: Delete stale remote branches, then force-push all branches and tags**

Run (deferred from Task 5.3):
```bash
git push origin --delete main-cleanup publication/cleanup collagen-experiments information-detection/initial-setup
git push origin --force --all && git push origin --force --tags
```
Expected: stale origin branches removed; origin updated to the rewritten history (only `main` +
`ExperimentsOnDrops` + 4 archive tags remain). (This breaks any other existing clone — expected;
sole author.)

- [ ] **Step 3: Verify with a fresh clone**

Run:
```bash
rm -rf /tmp/clonetest && git clone https://github.com/Nard248/4D-Hyperspectral-Unsupervised-Clustering.git /tmp/clonetest && du -sh /tmp/clonetest/.git
```
Expected: fresh clone `.git` ≈ ~300 MB; repo structure is the cleaned layout.

### Task 6.5: Promote the cleaned state to `main` — CHECKPOINT

**Files:** none

> Confirm with owner: make the cleaned `ExperimentsOnDrops` the new `main`?

- [ ] **Step 1: Fast-forward main to the cleaned working branch**

Run:
```bash
git branch -f main ExperimentsOnDrops && git push origin --force main
```
Expected: `main` now points at the cleaned history.

- [ ] **Step 2: Final verification**

Run:
```bash
git log --oneline -8 && du -sh .git && git branch -a
```
Expected: clean history, ~300 MB `.git`, `main` + `ExperimentsOnDrops` aligned.

---

## Post-Phase-6: Top-level README refresh (recommended)

### Task 7.1: Rewrite root `README.md` as the repo map

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write a top-level map**

Rewrite `README.md` to cover: what the project is; the layout (`src/` packages, `experiments/`, `publications/`, `data/` contract, `archive/`); install (`pip install -e .[dev]`); quickstart (`spectral_select.Analyzer`); how to launch the preprocessor (`python -m mehsi_preprocessor`); pointer to `docs/`.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as repo map for the cleaned layout"
```

---

## Self-Review checklist (run before execution)

- **Spec coverage:** Phases map to spec §6 Phases 0–6; engine-unify (spec Phase 7) explicitly deferred to its own plan. Data `data/README.md` contract and detailed dataset documentation will be authored within Phase 2/Task 2.3 follow-up or the README refresh.
- **Risk ordering:** zero/low-risk first (0–3), test-gated (4), irreversible last (5–6), all behind the Phase 0 backup.
- **Reversibility:** every deletion of a tracked file is either a `git mv`/`git rm` (recoverable from history until Phase 6) or preserved in `archive/` + manifest; branch deletions preserved via tags; history rewrite behind a mirror backup.
- **Checkpoints:** data archival (2.3), remote branch deletion (5.3), and the entire history rewrite (6) require explicit owner go-ahead.
