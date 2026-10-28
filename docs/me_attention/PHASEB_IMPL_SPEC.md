# Phase B implementation spec (lead-written, 2026-10-28)

Implementation contract for `src/me_attention/robustness.py`, `tests/me_attention/test_robustness.py` and
`src/me_attention/robust_analysis.py`. The science is frozen in `PREREG_ROBUSTNESS_2026-10-28.md`; this file only
pins down the engineering so the code is unambiguous. Where the two disagree, the pre-registration wins and the
difference is logged in its *Deviations* section.

## 0. Verified facts you may rely on (checked by the lead, do not re-derive)

A clean record is reproducible **exactly** from its own fields. This was confirmed on
`results/me_attention/sub3/lichens2_sub3/rbfsvm.jsonl`, repeat 0 fold 0 (delta = 0.000000):

```python
from me_attention import data as D
from me_attention.models.registry import MODELS
from me_attention.protocol import outer_folds, metrics

d    = D.load(rec["dataset"])          # for *_recwl datasets: os.environ["ME_ATTENTION_RECORDED_WL"]="1" FIRST
meta = d.meta()
spec = MODELS[rec["model"]]
tr, te = outer_folds(d, rec["repeat"])[rec["fold"]]     # == rec["n_train"], rec["n_test"]
cfg  = dict(rec["best_config"])
if spec.family != "neural-ish":                        # see note below
    cfg["n_jobs"] = <threads>                          # harness strips n_jobs from the record; classical models need it
est = spec.build(cfg, meta, rec["seed"], device)
est.fit(d.X[tr], d.y[tr], d.groups[tr])                # NO X_val/y_val -> no early stopping, exactly cfg["epochs"]
m   = metrics(d.y[te], est.predict_proba(d.X[te]), d.groups[te], d.n_classes, whole_specimen=True)
```

- Use `me_attention.harness.is_neural(spec)` to decide neural vs classical. Add `cfg["n_jobs"] = threads` only for
  classical models (that is what the harness does).
- `rec["best_config"]` already carries the tuned `epochs` and the adjusted `warmup_epochs` for neural models. Never
  re-tune, never pass a validation set, never early-stop.
- `rec["seed"]` equals `harness.job_seed(dataset, model, repeat, fold)`; reuse the stored value.
- Reuse `outer_folds` / `metrics` / `stats_paired` / `holm` from `protocol.py`. Do not reimplement fold logic,
  metrics or statistics.
- Spectral track only. Never build patches.

Dataset structure (measured; `avail` = training specimens per class on a fold):

| Key | env | N px | F | E | C | k | specimens/class | units (R x k) |
|---|---|---|---|---|---|---|---|---|
| `lichens2_sub3` | none | 13,530 | 210 | 8 | 3 | 5 | 5/5/5 | 5 x 5 = 25 |
| `lichens2` | `ME_ATTENTION_RECORDED_WL=1` | 22,562 | 210 | 8 | 5 | 2 | 4/10/7/5/2 | 5 x 2 = 10 |
| `collagen_pepsin` | `ME_ATTENTION_RECORDED_WL=1` | 39,970 | 158 | 6 | 3 | 3 | 3/3/3 | 5 x 3 = 15 |
| `lichens1` | `ME_ATTENTION_RECORDED_WL=1` | 191,046 | 192 | 8 | 4 | 4 | 4/4/4 | 1 x 4 = 4 |

## 1. Module layout

### `src/me_attention/robustness.py`

Public surface (keep these names; the tests and the lead's review use them):

```python
PERTURB_SEED_SALT = "robust-2026-10-28"

def perturb_seed(dataset: str, repeat: int, fold: int, family: str, level) -> int
def impute_values(X_train: np.ndarray) -> np.ndarray            # [F], T1 fill value
def ex_blocks(data) -> list[np.ndarray]                         # feature indices per excitation
def t1_subsets(n_ex: int, m: int, seed: int, n_draw: int = 10) -> list[tuple[int, ...]]
def apply_t1(X_test, blocks, drop_ex, impute) -> np.ndarray
def apply_t2(X_test, blocks, groups_test, sigma, seed) -> np.ndarray
def apply_t3(X_test, std_train, s, seed) -> np.ndarray
def flip_labels(y_train, n_classes, p, seed) -> np.ndarray
def subsample_pixels(idx, groups, n, seed) -> np.ndarray         # C(a)
def subsample_specimens(idx, groups, specimen_classes, s, seed) -> np.ndarray   # C(b)
def clip_config(cfg: dict, model: str, n_train: int) -> tuple[dict, list[str]]  # returns (cfg, notes)
def levels_for(family: str, data, train_idx) -> list
def run_unit(job: dict) -> list[dict]                            # all records for one (ds, model, repeat, fold)
def build_jobs(...) -> tuple[list[dict], int]                    # resumable, mirrors harness.build_jobs
def main(argv=None)                                              # CLI, mirrors harness CLI
```

**Purity rule (tested).** Every perturbation is a pure function of the raw `X` (or raw `y`) plus a seed derived from
`(dataset, repeat, fold, family, level)` only. Nothing may depend on the model name, the config, or the estimator.
`perturb_seed` must not take a model argument.

`perturb_seed` is md5-based, like `harness.job_seed`:

```python
def perturb_seed(dataset, repeat, fold, family, level):
    key = f"{PERTURB_SEED_SALT}|{dataset}|{repeat}|{fold}|{family}|{level}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
```

### Perturbations

**T1 — missing excitations** (test-time). `impute_values(X_train)` is the pre-registered per-feature value

```python
impute = np.expm1(np.log1p(np.maximum(X_train, 0.0)).mean(axis=0))      # [F], float32
```

For each `m in (1, 2, 3)` with `m <= n_ex - 1`, `t1_subsets` returns the excitation index tuples to remove: if
`C(n_ex, m) <= n_draw` return **all** of them sorted; else draw `n_draw` distinct subsets with
`perturb_seed(dataset, repeat, fold, "T1", m)`. Record `n_subsets` on every T1 record. `apply_t1` writes
`impute[f]` into every feature `f` of every removed excitation block, for all test rows. Average balanced accuracy
over the subsets for the level's summary record, and also store the per-subset values.

**T2 — per-excitation gain drift** (test-time). For `sigma in (0.1, 0.25, 0.5)`: one factor
`g = exp(N(0, sigma^2))` per `(test specimen, excitation)`, identical for every pixel of that specimen, multiplying
that excitation's block. Seed `perturb_seed(ds, r, f, "T2", sigma)`.

**T3 — measurement noise** (test-time). For `s in (0.1, 0.25, 0.5, 1.0)`: add `N(0, (s * std_train[f])^2)` per
feature in raw space, then `np.maximum(out, 0.0)`. `std_train` = per-feature std of the fold's raw training `X`.
Seed `perturb_seed(ds, r, f, "T3", s)`.

**L — label noise** (training-time, refit). For `p in (0.1, 0.2, 0.3, 0.4)`: choose `round(p * n_train)` training
pixels uniformly without replacement and reassign each to a class drawn uniformly from the **other** `C - 1`
classes. A flipped label is never the original. Test set stays clean. Seed `perturb_seed(ds, r, f, "L", p)`.

**C — learning curve** (training-time, refit). Two sub-families, recorded with `subset` set:

- `subset="a"`, `n in (25, 100, 400)`: keep `min(n, count_s)` pixels of each training specimen `s`, drawn uniformly
  without replacement. `n="all"` is the clean level and is **not** refit.
- `subset="b"`, `s in (1, 2)`: keep `min(s, avail_c)` training specimens of each class `c`, drawn uniformly. Skip the
  level when it removes nothing, i.e. when `min(s, avail_c) == avail_c` for every class (this is why
  `collagen_pepsin`, with `avail = [2,2,2]`, gets only `s = 1`). `s="all"` is the clean level and is not refit.

`levels_for` returns the valid levels of a family for a given fold, so the skipping rule lives in one place.

**Config clipping.** `clip_config` fixes hyperparameters that become invalid on a shrunken training set and returns
human-readable notes that go into the record's `clip_notes`:

- `knn`: `n_neighbors = min(n_neighbors, n_train)` (and at least 1).
- any model with `max_train`: `min(max_train, n_train)`.
- any neural model: `batch_size = min(batch_size, max(8, n_train))`.
Leave everything else alone; never change `epochs`.

### Work unit and record format

One job = one `(dataset, model, repeat, fold)`. `run_unit` does:

1. Rebuild `tr, te` and `cfg` / `seed` from the clean record (section 0). If the clean record is missing or is an
   error record, return a single `{"error": ...}` record and move on — never train without a tuned config.
2. **One clean fit.** Emit `family="clean", level=0`. Then, reusing that same fitted estimator, evaluate every
   test-time level: T1 (per `m`, averaged over its subsets), T2 (per `sigma`), T3 (per `s`).
3. **Refits** for every level of `L`, `C/a`, `C/b` from `levels_for`. Each refit gets a freshly built estimator with
   the same `cfg` and `seed`, trained on the perturbed training data, evaluated on the **clean** test fold.

Every record is one JSON line in `results/me_attention/robust/<dataset>/<model>.jsonl` with

```json
{"dataset": ..., "model": ..., "track": "spectral", "family": ..., "level": ..., "subset": null,
 "repeat": ..., "fold": ..., "seed": ..., "n_train": ..., "n_test": ..., "n_subsets": null,
 "best_config": {...}, "clip_notes": [], "test": { <the full protocol.metrics block> },
 "subset_baccs": null, "device": ..., "fit_s": ..., "predict_s": ..., "wall_s": ...}
```

`test` must be the **same metric block** as the clean records (`protocol.metrics(..., whole_specimen=True)`), so the
two are directly comparable. Use `harness._jsonable` before writing.

**Record key** for resumability: `(repeat, fold, family, str(level), subset)`. `build_jobs` reads the existing
`.jsonl`, collects done keys, and skips a unit only when **all** of its expected keys are present; otherwise the unit
re-runs and the writer must not duplicate keys already present (drop records whose key is already done). Writes go
through the parent process only, whole line + `flush` + `fsync`, exactly like `harness.append_record`.

### CLI and parallelism

Mirror `harness` so the operator experience is identical:

```
python -m me_attention.robustness --datasets lichens2_sub3 --models lr,rbfsvm,... \
    --repeats lichens2_sub3=5 --device cuda --workers 8 --gpu-workers 2 \
    [--families clean,T1,T2,T3,L,C] [--only-repeat 0 --only-fold 0] [--dry-run] [--results-root ...]
```

Reuse the harness's process-pool pattern: neural units capped at `--gpu-workers`, classical ones on the remaining
CPU workers, `--threads` per worker, and a `harness.log`-style progress line per finished unit
(`results/me_attention/robust/harness.log`) with elapsed and ETA. `--only-repeat` / `--only-fold` / `--families`
exist for the smoke test. Errors become records, never crashes (copy `harness.run_job`'s try/except shape).

### `tests/me_attention/test_robustness.py`

Use the existing toy fixtures in `tests/me_attention/toy.py` / `conftest.py` (look at
`tests/me_attention/test_harness.py` for how a tiny `MEData` and a monkeypatched registry are set up). Required
tests, all of which must actually assert the property:

1. **Model independence** — the perturbed `X` for two different model names is byte-identical
   (`np.array_equal`) for one representative level of T1, T2 and T3.
2. **Level 0 == clean** — `apply_t1` with an empty drop set, `apply_t2` with `sigma=0`, `apply_t3` with `s=0` all
   return arrays equal to the input.
3. **T1 imputed value maps to 0 after preprocessing** — with `Preprocessor(norm="none", log1p=True,
   standardize=True)` fitted on `X_train` whose fitted `scale_ == 1.0`, the row `impute_values(X_train)` transforms
   to approximately the zero vector (`atol=1e-5`). Construct `X_train` so that `scale_` is 1 (median positive
   normalised value equal to 1); assert `pre.scale_ == 1.0` in the test so the premise is explicit. See the
   deviation note in section 2.
4. **T2 constant within (specimen, excitation)** — the ratio perturbed/clean is constant across all pixels of a
   given (specimen, excitation) block and differs between blocks.
5. **Label flips never keep the original label** — exactly `round(p*n)` labels change, every changed one differs
   from the original, and all values stay in `0..C-1`.
6. **Subsampling respects per-specimen counts** — `subsample_pixels` keeps `min(n, count_s)` per specimen;
   `subsample_specimens` keeps `min(s, avail_c)` specimens per class and never drops a class.

Also test `perturb_seed` determinism, `t1_subsets` exhaustive-vs-sampled behaviour (`n_ex=6, m=1` -> 6 subsets;
`n_ex=8, m=3` -> 10 distinct), `clip_config` on kNN with `n_train=3`, and `levels_for`'s `C/b` skipping on an
`avail=[2,2,2]` fold. The whole file must run in seconds (no real dataset, no GPU).

### `src/me_attention/robust_analysis.py`

Reads `results/me_attention/robust/`, reuses `protocol.stats_paired` / `protocol.holm` and `analysis.py`'s
label/colour/`_save` helpers. Writes to `--out` (default `results/me_attention/analysis_robust`):

- `tab_robust_scores.md|tex` — robustness score per (dataset, model, family): mean balanced accuracy over that
  family's **non-clean** levels (T1 already averaged over subsets; C over both sub-families' non-clean levels),
  plus the clean value for reference.
- `tab_degradation.md|tex` — the full per-level curve, every model, every dataset, every family.
- `tab_confirmatory.md|tex` — **the 20 pre-registered tests**: `{lichens2_sub3, lichens2}` x `{T1,T2,T3,L,C}` x
  `{ex_tokens_v2, ex_tokens_lin}`, `Delta = candidate - rbfsvm` (comparators are frozen in
  `COMPARATORS_2026-10-28.md`), paired by `(repeat, fold)`, Nadeau-Bengio corrected t-test via `stats_paired` with
  `test_train_ratio = n_test/n_train` of that dataset, **one Holm family of all 20**, 95% bootstrap CI. A column
  must state the decision: advantage iff `Delta > 0` and Holm-adjusted `p < 0.05`.
- `tab_exploratory.md|tex` — everything else, clearly labelled exploratory: secondary datasets, the other three
  attention models, comparisons against every baseline, and the best baseline *per condition*.
- `figures/fig_degradation_<dataset>.pdf|png` — one panel per family, balanced accuracy vs level, attention models
  in the `OURS_COLORS` warm palette and baselines in `BASE_COLORS`, as `analysis.py` does.
- `summary_robust.json` — machine-readable version of all of the above.

Never drop a model or a condition because it looks bad; the table must show every one that was run, and mark
missing cells as missing rather than omitting the row.

## 2. Deviations to append to the pre-registration (lead-decided, do not re-litigate)

Append these to the *Deviations* section of `PREREG_ROBUSTNESS_2026-10-28.md` verbatim, dated 2026-10-28:

1. **T1 subset count when fewer than 10 subsets exist.** The plan says "draw 10 random subsets of excitations".
   For `m = 1` there are only `E` (8, or 6 for `collagen_pepsin`) distinct subsets, and for `collagen_pepsin`
   `m = 2` only 15. Where `C(E, m) <= 10` all distinct subsets are enumerated instead of drawing 10 with
   replacement; otherwise 10 distinct subsets are drawn as planned. `n_subsets` is recorded per record. This
   strictly reduces Monte-Carlo noise and keeps the subsets identical across models.
2. **T1 imputation value and the "maps to 0" property.** The plan fixes the fill value to
   `expm1(mean(log1p(X_train)))` per feature *and* describes it as the value that maps to 0 after preprocessing.
   Those two coincide only when a model's preprocessing uses `norm="none"` and the fitted `log1p` scale is 1.
   `preprocess.Preprocessor` normalises **per pixel** (`norm` in {`l2`, `area`}) and divides by a fitted
   `scale_` before `log1p`, so for most tuned configs the fill value maps to approximately, not exactly, 0. The
   frozen formula is kept, because the pre-registration's stronger requirement is that every model receives
   byte-identical inputs: making the fill value config-dependent would break that. Consequence: T1 is "replace the
   block by its training-set typical value", which is the same input for every model and the same *kind* of input
   the `ex_drop` augmentation produces, but it is not exactly post-preprocessing zero for every config.
3. **C(b) level validity.** The plan says to use "only values smaller than the available count". Implemented as:
   level `s` keeps `min(s, avail_c)` specimens per class and is skipped when it removes nothing from any class.
   This reproduces the plan's stated `collagen_pepsin` case (`s in {1, all}`) and extends it per fold, which
   matters for `lichens2`, where class 4 contributes only one training specimen.

## 3. Working rules

- No new third-party dependencies: torch / numpy / scipy / scikit-learn / pandas / matplotlib only.
- Do not touch `types.py`, `protocol.py`, `preprocess.py`, `train.py`, `harness.py`, `data.py`, the models, or any
  existing test. Phase B is additive: three new files only. If you believe an existing file must change, stop and
  say so instead.
- Do not run any training, any real dataset, or anything on the GPU: the GPU is busy with Phase A. Run only
  `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/me_attention/test_robustness.py -o addopts="" -q`
  and confirm the existing suite still collects.
- Match the surrounding code's style: terse module docstring stating the contract, `from __future__ import
  annotations`, type hints on public functions, no decorative comments.
