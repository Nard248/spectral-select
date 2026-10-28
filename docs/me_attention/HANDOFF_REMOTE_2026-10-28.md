# Handoff — ME-Attention training on the remote GPU machine (2026-10-28)

You are a Claude session on a GPU machine continuing the ME-Attention study that was started on the laptop (Apple M4
Pro, too slow for this scale). Read this file fully, then `docs/me_attention/SPEC.md` (protocol and fairness rules) and
`docs/me_attention/PREREG_ROBUSTNESS_2026-10-28.md` (frozen robustness plan) before doing anything.

## 1. Where the study stands

Goal set by the user: a supervised attention-based pixel classifier for multi-excitation (ME) fluorescence
hyperspectral images that clearly beats existing classifiers, **measured under a fair, pre-specified protocol and
reported honestly**. If there is no margin, we say so.

Evaluation: whole-specimen class-covering nested CV (k = min specimens per class, R repeats), successive halving
tuning (T = 3 configs, E_max = 25 epochs), shared preprocessing, Wilcoxon/Holm + Nadeau–Bengio corrected t-test,
bootstrap CIs. Code in `src/me_attention/` (`harness.py` runs sweeps, `analysis.py` makes tables/stats).

Clean results so far (balanced accuracy %, mean over folds):

| Dataset | Best attention | Best baseline | Significant? |
|---|---|---|---|
| Collagen Pepsin (recorded wl) | ex_tokens 90.0 | LR 90.3 | no |
| Lichens 1 (recorded wl) | smartattention_v0 99.2 | SpectralFormer 99.5 | no |
| Lichens 2, all 5 classes (recorded wl) | ex_tokens 86.1 | RBF-SVM 86.6 | no |
| **Lichens 2, 3-class subset** (25 folds) | ex_tokens_v2 95.7 | RBF-SVM 96.6 | not yet tested |

Lichens 2 subset, all models: RBF-SVM 96.6, SpectralFormer 95.9, HGB 95.8, ex_tokens_v2 95.7, MLP 95.6, kNN 95.3,
smartattention_v0 94.9, RF 94.9, hybrid_conv_attn 94.5, 1D-CNN 93.5, LR 93.4; meas_tokens 17/25 folds done.
Specimen-level majority vote is 100% for every model, so the clean task is close to ceiling.

### Datasets — facts that must not be changed or re-guessed

- **`lichens2_sub3`** (new, 2026-10-28): the user found part of Lichens 2 mislabelled. Keep only the last 3 rows ×
  first 5 columns of the 4 × 7 piece layout: classes Green / Blue / Yellow (original classes 2, 3, 4, relabelled 0..2),
  5 pieces each, 13,530 px. Recorded emission wavelengths and the 10× fix of the 400 nm exposure are built into its
  config in `src/me_attention/data.py` (`DATASETS["lichens2_sub3"]`: region rows 70+, columns 0–260, classes 1–3).
- The other datasets run with `ME_ATTENTION_RECORDED_WL=1` (recorded emission wavelengths; the original processed
  files mislabel some emission axes). Provenance and the bug are documented in `docs/datasets/audit_2026-10-22/`.
- Lichens 2 is **unpublished** data, and this GitHub repo is **public**. Never commit `Data/`, the
  `results/me_attention/cache/*.npz` files, or anything that contains raw spectra.

## 2. Setup

```bash
git pull
python3.11 -m venv .venv && .venv/bin/pip install -e .     # torch, numpy, scipy, scikit-learn; no new deps allowed
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If pip installs a CPU-only torch, reinstall the CUDA build from pytorch.org that matches the driver. On Windows, run
the shell scripts in Git Bash or WSL and pass `PY=.venv/Scripts/python.exe`.

Data: the feature caches are in the user's Google Drive (synced on this machine) at
**`My Drive/Orbeli/Data/me_attention_cache_2026-10-28/`**: `lichens2_sub3.npz`, `lichens2_recwl.npz`,
`collagen_pepsin_recwl.npz`, `lichens1_recwl.npz` and `SHA256SUMS`. Find where Drive is mounted on this machine. On
Windows it is usually a drive letter such as `G:\My Drive\...`; on macOS it is
`~/Library/CloudStorage/GoogleDrive-*/My Drive/...`. Copy (don't symlink) the four `.npz` files into
`results/me_attention/cache/`, and verify them there against `SHA256SUMS` (`shasum -a 256 -c`, or `certutil -hashfile
<f> SHA256` on Windows). If Drive streams files on demand, make sure they are fully downloaded first. The loader uses
these caches and does not need `Data/`. If the folder is missing, stop and ask the user.

Fallback, only if a cache checksum fails: the same Drive folder also holds the processed datasets
(`processed/Collagen Pepsin`, `Collagen_Acetic_Acid`, `Lichens Dataset 1`, `Lichens_2`) and the raw acquisitions
(`raw/`). Copy `processed/*` into `<repo>/Data/processed/`, delete the bad cache file, and let the loader rebuild it.
Use `ME_ATTENTION_RECORDED_WL=1` for the `*_recwl` caches and no variable for `lichens2_sub3`. Compare the rebuilt
cache's `X.shape`, `y` counts and specimen counts with the laptop values: `lichens2_sub3` is 13,530 px, 5/5/5
specimens, 5385/3608/4537 px. Note the rebuild in the report.

Sanity checks before training:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/me_attention -o addopts="" -q          # expect 131 passed
```

Then count the remaining Phase A jobs (expect sub3: 33 = meas_tokens 8 + ex_tokens_lin 25; recwl: 29):

```bash
PYTHONPATH=src .venv/bin/python -c "
from me_attention import harness as h
M='lr,rbfsvm,rf,hgb,knn,mlp,cnn1d,spectralformer,smartattention_v0,ex_tokens_v2,hybrid_conv_attn,meas_tokens,ex_tokens_lin'.split(',')
print(len(h.build_jobs(h.RESULTS_ROOT,'sub3',['lichens2_sub3'],M,{'lichens2_sub3':5})[0]))"
```

If the count is 325, the committed `results/me_attention/sub3/` records are not being found. Fix that before
training; never rerun finished jobs.

## 3. Phase A — finish the clean runs (no code changes)

```bash
DEVICE=cuda WORKERS=8 GPU_WORKERS=2 nohup bash experiments/me_attention/run_remote_phaseA.sh > results/me_attention/phaseA.log 2>&1 &
```

Tune `WORKERS` to the CPU count and `GPU_WORKERS` to GPU memory (each neural worker used 8–10 GB on the laptop; with
24 GB+ use 2–3). Progress is in `results/me_attention/sub3/harness.log` and `results/me_attention/recwl/harness.log`.
The script prints `PHASE_A_DONE` at the end.

Then the clean analysis:

```bash
PYTHONPATH=src .venv/bin/python -m me_attention.analysis --sweep sub3 --out results/me_attention/analysis_sub3
PYTHONPATH=src .venv/bin/python -m me_attention.analysis --sweep final_recwl --out results/me_attention/analysis_final_recwl
```

Check `python -m me_attention.analysis -h` for the exact flags first; adapt if they differ, but don't change the
statistics.

## 4. Phase B — robustness study (implement, then run)

Implement exactly `docs/me_attention/PREREG_ROBUSTNESS_2026-10-28.md`:

- `src/me_attention/robustness.py`: a runner that, per (dataset, model, repeat, fold), reads `best_config` and `seed`
  from the clean record and rebuilds the same outer split (reuse `protocol.py` / `harness.py` helpers; do not
  re-implement fold logic). It retrains once for the test-time families T1–T3 and evaluates every level. It refits per
  level for L and C. Output goes to `results/me_attention/robust/<dataset>/<model>.jsonl`, one record per
  (repeat, fold, family, level[, subset]), with the same metric block as the clean records. It is resumable, skipping
  done keys, parallel like the harness, and writes a `harness.log`-style progress log.
- Perturbation helpers are pure functions of the raw `X` plus a seed derived from (dataset, repeat, fold, family,
  level), never from the model. Unit tests in `tests/me_attention/test_robustness.py` must cover: identical perturbed
  input for two different models; level 0 equals clean; T1 imputed value maps to 0 after preprocessing; T2 constant
  within (specimen, excitation); label flips never keep the original label; subsampling respects per-specimen counts.
- First run a smoke test on `lichens2_sub3`, repeat 0, fold 0, for 2 models. Check the clean level reproduces the
  clean record's test balanced accuracy to within about 1 point (same config and seed). If not, find out why before
  the full run.
- Run order: `lichens2_sub3` → `lichens2` → `collagen_pepsin` → `lichens1`, all 13 spectral models.
- Statistics in `src/me_attention/robust_analysis.py` (reuse `analysis.py` statistics functions): degradation curves
  per family (figure), robustness score table, the 20 confirmatory tests with Holm, and exploratory tables.

Anything that cannot follow the pre-registration goes into its *Deviations* section, with a reason. The conditions,
levels, hypotheses and decision rule must not change.

## 5. Phase C — report

Write `docs/me_attention/RESULTS_REMOTE_<date>.md`:

1. Phase A clean results, with `ex_tokens_lin` vs the best baseline per dataset (corrected statistics).
2. Phase B confirmatory table (20 tests, Holm), then exploratory tables and figures.
3. Hypotheses H1–H5: supported or not, in plain words.
4. **Collagen recommendation.** The user said Collagen goes out of the paper if attention does not perform well
   there. State the evidence: does any attention model beat the best Pepsin baseline in the clean or the robustness
   results? Make a recommendation. The user makes the final decision.
5. Hardware, wall time, deviations.

Do not edit the papers in `publications/`; the user decides on the paper after reading the report.

## 6. Commit and push

`results/` is git-ignored, so results must be force-added. Commit only result records, logs, figures and docs:

```bash
git add -f results/me_attention/sub3/{COMMAND,RUN_INFO.md,harness.log} results/me_attention/sub3/lichens2_sub3/*.jsonl \
           results/me_attention/recwl results/me_attention/final_recwl results/me_attention/robust \
           results/me_attention/analysis_*
git add src/me_attention tests/me_attention docs/me_attention experiments/me_attention
git status   # confirm: no *.npz, no Data/, no PID / nohup files
git pull --rebase && git commit -m "..." && git push origin main
```

Commit after Phase A and again after Phase C, so partial progress is not lost.

## 7. Working rules from the user

- Delegate coding to subagents running Sonnet or Haiku, never Opus. Keep design decisions short, and review their code
  before running it.
- Honesty over a positive story: no test-set tuning, no picking conditions or levels after seeing results, every model
  under every condition reported, and losses stated plainly. If the result is "no advantage", say that.
- Ask the user before going beyond this scope: no new datasets, models or conditions without their approval.
- If a run crashes, fix the cause, resume (finished jobs are skipped), and note it in the report.
