# Native Apple-Silicon wavelength selector

This is a standalone Apple-Silicon-oriented implementation. It does not import
or call the Python Monte-Carlo selector in `personal/setup`.

The default run selects exactly 13 of 210 wavelength pairs using native beam
search followed by iterative one-pair swap refinement. For `n=3`, it instead
exhaustively evaluates all `210 choose 3 = 1,521,520` triples. The score is mean
three-fold, object-grouped balanced accuracy from a native standardized
nearest-centroid classifier. Complete connected lichen objects, not individual
pixels, are assigned to folds.

## Why it is fast

For each CV fold, the program calculates class means and pooled variance once.
It precomputes the standardized squared-distance contribution of every
`(feature, validation pixel, class)` tuple. Subsets then require only additions,
comparisons and confusion counts. Search evaluations run concurrently using
standard C++ threads, supported by Apple Clang without Homebrew OpenMP.

At each of 13 steps, beam search expands the current 128 best unique partial
sets with every remaining pair, evaluates them, and retains the best 128.
After reaching 13 pairs, swap refinement repeatedly tests replacing each
selected pair with every unselected pair until no replacement improves the
score. This is fast and interaction-aware, but it is not a global-optimum proof.

## Build and run

From the repository root:

```bash
chmod +x personal/native_exact_selector/run.sh
PYTHON=.venv-1/bin/python personal/native_exact_selector/run.sh
```

Or run each stage separately:

```bash
.venv-1/bin/python personal/native_exact_selector/export_dataset.py
mkdir -p personal/native_exact_selector/build
clang++ -std=c++20 -O3 -ffast-math -mcpu=native -pthread \
  personal/native_exact_selector/src/main.cpp \
  -o personal/native_exact_selector/build/lichen_exact
mkdir -p personal/native_exact_selector/results
personal/native_exact_selector/build/lichen_exact \
  personal/native_exact_selector/data/lichens2.bin \
  personal/native_exact_selector/results/top_13.csv 13
```

The native executable accepts optional thread and top-result counts:

```text
lichen_exact DATASET OUTPUT.csv N THREADS BEAM_WIDTH TOP_K
```

It defaults to `N=13`, all logical CPU cores, beam width 128, and the top 100
sets. The result reports indices, excitation/emission wavelengths, grouped-CV
balanced accuracy, and rank. A wider beam explores more alternatives:

```bash
personal/native_exact_selector/build/lichen_exact \
  personal/native_exact_selector/data/lichens2.bin \
  personal/native_exact_selector/results/top_13_wide.csv 13 10 512 100
```

## Scope

The 13-pair result is a heuristic optimum, not an exhaustive global optimum.
The executable remains exact for three pairs under its nearest-centroid CV
objective. Different classifiers define different objectives, so the winning
set should be validated with a second classifier before a scientific claim.
