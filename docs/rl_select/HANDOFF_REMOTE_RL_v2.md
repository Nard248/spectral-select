# Handoff — reversible-RL band-selection rerun (v2) on the remote machine (2026-10-28)

You are a Claude session on a remote CPU/GPU machine. Rerun the whole `rl_select` study (spec: `docs/rl_select/SPEC.md`,
paper: `publications/rl_band_selection/main.tex`) with the corrected data. Nothing from the 2026-10-19 results is reused;
everything goes to `results/rl_select_v2/`.

## What changed versus the 2026-10-19 results
1. **Lichens 2 = `lichens2_sub3`** (3 classes x 5 specimens, 13,530 px, 210 pairs; the same cleaned subset as the ME-Attention
   study). Dataset list = `lichens1, lichens2_sub3, collagen_acetic, collagen_pepsin` (`RL_SELECT_DATASETS` overrides; the full
   `lichens2` stays loadable for comparison). Outer folds k = 4, 5, 3, 3 (derived from the data, nothing hard-coded).
2. **Recorded emission wavelengths** in all caches. Feature values X are bit-identical to the old caches (verified); only
   `em_nm` / `emissions` / `grid_index` labels differ. `lichens2_sub3` also has the 400 nm exposure fix (that block is x0.1).
3. **3 policy seeds** (0,1,2) for every policy variant (6 non-augmented variants x 6 source pools, and 4 augmented variants x
   6 pools); the S1 wrappers keep their 3 search seeds.

## 1. Setup
```bash
git pull
python3.11 -m venv .venv && .venv/bin/pip install -e .          # torch, numpy, scipy, scikit-learn, matplotlib, pandas
```
Caches come from Google Drive `My Drive/Orbeli/Data/me_attention_cache_2026-10-28/` (synced on the machine; find the mount:
Windows `G:\My Drive\...`, macOS `~/Library/CloudStorage/GoogleDrive-*/My Drive/...`). It holds `lichens2_sub3.npz`,
`lichens2_recwl.npz`, `collagen_pepsin_recwl.npz`, `lichens1_recwl.npz`, `collagen_acetic_recwl.npz` and `SHA256SUMS`.
Copy (do not symlink) all five `.npz` into `results/me_attention/cache/` and check them there (`shasum -a 256 -c SHA256SUMS`
or `certutil -hashfile <f> SHA256`). If Drive streams on demand, make sure the files are fully downloaded. If the folder is
missing, stop and ask the user.

> **NOTE (done on the laptop 2026-10-28):** `collagen_acetic_recwl.npz` was not in the folder at first. It was copied from
> `results/me_attention/cache/collagen_acetic_recwl.npz` into the Drive folder and its SHA-256
> (`79bf8b439d12a7f0a7a8d0d95cce6c89b3da567a1373b8dd4d637dd2fad14cfc`) appended to `SHA256SUMS`. If it is missing on your side, copy it from the laptop cache the
> same way. Path on the Mac: `/Users/narekmeloyan/Library/CloudStorage/GoogleDrive-narek.h.meloyan@gmail.com/My Drive/Orbeli/Data/me_attention_cache_2026-10-28/`.

Fallback only if a checksum fails: the Drive folder also holds `processed/` (copy to `Data/processed/`); delete the bad cache and
let `me_attention.data.load` rebuild it (`ME_ATTENTION_RECORDED_WL=1` for `*_recwl`, none for `lichens2_sub3`).

Build the RL caches (the script also does this as its first stage) and sanity-check:
```bash
PYTHONPATH=src .venv/bin/python experiments/rl_select/build_caches.py     # -> results/rl_select/cache_recwl/{lichens1,lichens2_sub3,collagen_acetic,collagen_pepsin,lichens2}.npz
PYTHONPATH=src .venv/bin/python -m pytest tests/rl_select -o addopts="" -q                   # needs the caches above; expect 85 passed on the laptop, 75 passed + 10 skipped without results/rl_select/cache (those 10 compare with the old caches)
PYTHONPATH=src .venv/bin/python experiments/rl_select/smoke_v2.py                            # ~20 s end-to-end smoke into a temp dir, prints SMOKE_OK
```
Expected cache facts: `lichens2_sub3` X (13530, 210), classes 3, specimens 5/5/5, pixels 5385/3608/4537, k = 5; `lichens1` (191046,192) 4x4
specimens; `collagen_acetic` (46670,149) 3x3; `collagen_pepsin` (39970,158) 3x3.

## 2. Run
```bash
WORKERS=15 nohup bash experiments/rl_select/run_remote_v2.sh > /dev/null 2>&1 &
tail -f results/rl_select_v2/run.log            # ends with RL_V2_DONE (or RL_V2_INCOMPLETE if a stage failed)
```
`WORKERS` = number of worker processes (default cores - 1). Stages, in order (each leaves `results/rl_select_v2/.done/<stage>`;
re-running the script resumes: finished stages are skipped, drivers skip finished rows/policies; `FORCE=1` ignores markers,
`STAGES="rl_v3 report"` runs a subset, `SEEDS=0,1,2`):

| stage | what | driver |
|---|---|---|
| `caches` | RL caches from the ME-attention loader | `build_caches.py` |
| `baselines` | S1/S2/S3 baselines, then exact n=2,3 | `run_baselines.py --phase main`, `--phase exact` |
| `rl_s2_s3` | train 108 policies (6 variants x 6 pools x 3 seeds), each followed at once by S2 label-free deployment (3 repeats) and, for LODO pools, S3 few-shot | `run_rl.py s2` |
| `rl_s1_abl_s4` | per-instance RL search S1 (n=5,13), collagen-acetic ablation (n=8, B=20000), hardware-cost S4 (Lichens 1) | `run_rl.py s1 abl s4` |
| `rl_v3` | augmented-task policies (add, add/remove, CSS-start, mix) x 6 pools x 3 seeds, train + S2 | `run_rl_v3.py run` |
| `report` | digests (`summary/`), `paper_figures.py` (-> `results/rl_select_v2/paper/{figures,tables}`, the `publications/` folder is NOT touched), `decompose_gain.py`, count check | `aggregate*.py`, `digest_rl_v3.py`, `paper_figures.py`, `decompose_gain.py`, `verify_v2.py` |

The paper cites S1, S2, S3, S4 (hardware-cost paragraph) and the ablation table, so all are run. Training (stage `rl_s2_s3`) and S2/S3 deployment are
one pipeline in the driver, not separate passes.

**Parallelism / GPU.** PPO is `torch`, but the policy has ~38k parameters and tiny batches (<= 64 episodes x ~200 candidates x 64
units); about 40% of a training update is numpy environment/feature/JSup work that a GPU cannot take, and the policy code has no
device handling. The script therefore runs CPU-only with one thread per process (it sets `CUDA_VISIBLE_DEVICES=""`) and
parallelises over (pool, variant, seed) jobs: throughput scales linearly with cores. A GPU was not benchmarked and is not needed.

**Time (from the laptop logs, CPU-hours = sum of single-process job times; wall ~ CPU-h / WORKERS, +10-20% for tail effects).**
Old per-job means (4-8 concurrent jobs on an M4 Pro, may be faster on a server): policy training 676-862 s (250 updates, ~2.7 s/update),
S2 deployment job 183 s, S3 150 s, S1 52 s, ablation 100 s, S4 22 s; augmented training 1095 s (250 updates), augmented S2 job 302 s;
baselines 2581 s wall on 6 workers for 144 units, exact 300 s.

| stage | jobs | CPU-h |
|---|---|---|
| baselines | 180 units + 30 exact rows | ~5 |
| `rl_s2_s3` | 108 train + 432 S2 + 24 S3 | ~46 (23 + 22 + 1) |
| `rl_s1_abl_s4` | 60 + 27 + 40 | ~2 |
| `rl_v3` | 72 train + 288 S2 | ~46 (22 + 24) |
| report | | < 0.5 (`decompose_gain.py` ~10-20 min) |
| **total** | | **~100** -> ~7 h at 15 workers, ~3.5 h at 31 |

`run_rl_v3.py` uses 400 PPO updates by default (the 2026-10-19 code default; the paper text says this made augmented and non-augmented
training lengths unequal, a stated confound). To train them for 250 like the others run with `V3_UPDATES=250`. Ask the user if unsure.

## 3. Monitor and verify
```bash
tail -f results/rl_select_v2/run.log                      # stage lines; per-stage detail in results/rl_select_v2/logs/<stage>.log
PYTHONPATH=src RL_SELECT_RESULTS=results/rl_select_v2 .venv/bin/python experiments/rl_select/verify_v2.py   # expected vs actual
```
Expected (k = 4/5/3/3, seeds 0-2): policies 108 + 72 (augmented); S2 rows 12,960 (non-aug) + 8,640 (augmented); S3 rows <= 1,800 (a few budgets
can be absent per row); S1 RL 60, ablation 27, S4 40; baselines S1 1,260 / S2 3,420 (+45 `all_pairs`) / S3 3,600, exact 30.
`verify_v2.py` exits 0 and prints `VERIFY_OK` only when nothing is missing. A job that raised prints `FAILED ...` in the stage log: rerun the script.

## 4. What to commit (ask the user first; the repo is PUBLIC and Lichens 2 is unpublished)
`results/` is git-ignored: force-add only digests and result rows.
```bash
git add -f results/rl_select_v2/{S1,S2,S3,S4}/*.jsonl results/rl_select_v2/summary results/rl_select_v2/paper results/rl_select_v2/run.log results/rl_select_v2/logs
```
Never add `results/rl_select/cache_recwl/*.npz`, `results/me_attention/cache/*.npz`, `Data/`, or the policy checkpoints unless the user asks
(`policies*/` hold only network weights and training logs, no spectra, but are not needed). The rows contain pixel-free metrics and selected pair indices only.
Do not edit `publications/rl_band_selection/` or `docs/rl_select/SPEC.md`; report numbers to the user instead.

## 5. Order relative to the ME-attention handoff
`docs/me_attention/HANDOFF_REMOTE_2026-10-28.md` (GPU, ~8-10 GB per neural worker, uses `WORKERS`/`GPU_WORKERS`) and this run are independent.
Run this one AFTER it, or alongside it only on a machine with enough cores: this run is pure CPU and wants all `WORKERS` cores, so give each
job a disjoint share (for example ME-attention `WORKERS=8`, this run `WORKERS=<cores - 10>`); sharing cores slows both and distorts the
per-job time estimates above. Both write to different result roots and share only the read-only `results/me_attention/cache/` files.
