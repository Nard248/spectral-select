# Native Monte-Carlo REINFORCE selector

This is a standalone C++-accelerated reinforcement-learning wavelength
selector. It selects exactly 13 excitation/emission pairs by default.

## Run

From the repository root:

```bash
PYTHON=.venv-1/bin/python personal/native_mc_rl_selector/run.sh
```

The default is 30 training episodes and three Monte-Carlo completions per
candidate. On the tested MacBook Air M4, native training takes about six
seconds; export and compilation add about two seconds.

Manual invocation:

```bash
personal/native_mc_rl_selector/build/native_mc_rl \
  personal/native_mc_rl_selector/data/lichens2.bin \
  personal/native_mc_rl_selector/results \
  13 100 5 10
```

The final arguments are `N`, `EPISODES`, `ROLLOUTS`, and `THREADS`. A stronger
100-episode/five-rollout run is expected to take roughly 30–60 seconds on an
M4. Runtime is approximately linear in episodes and rollouts.

## What is reinforcement learning here?

The state is the current subset, actions are unused wavelength pairs, and an
episode contains exactly `N` actions. Its terminal reward is mean three-fold
object-grouped balanced accuracy. A step-specific softmax policy has learned
weights for nine normalized action features:

1. individual gain;
2. marginal gain in the current set;
3. Monte-Carlo completed-set mean;
4. Monte-Carlo standard deviation;
5. Monte-Carlo minimum;
6. Monte-Carlo maximum;
7. mean absolute correlation redundancy;
8. excitation wavelength;
9. emission wavelength.

The initial weights encode the earlier comprehensive-score prior. After every
episode, REINFORCE updates them using
`(terminal reward - exponential moving baseline) * gradient(log policy)`.
Adam stabilizes these policy-gradient updates. Temperature decay controls
exploration. The weights are separate for each selection position, so the
model can learn that early and late decisions require different criteria.

Monte-Carlo rollouts complete every tentative action to exactly `N` pairs and
score the completed subsets in the native grouped-CV environment. Completions
use a lightweight policy-guided sampler. Candidate metric calculation is
parallelized across all CPU threads.

## Native score engine

For each fold and class, C++ calculates feature centroids and pooled within-
class variances. Standardized squared-distance contributions are precomputed
for every feature, validation pixel, and class. Subset scoring then consists
mostly of additions and comparisons. Whole lichen objects remain within one
fold to prevent pixel leakage.

The score is optimized for this nearest-centroid environment. The resulting
set should still be validated using the intended final classifier and outer
held-out lichen objects.

## Outputs

Results are written under `personal/native_mc_rl_selector/results`:

- `best_set.csv`: selected indices and physical wavelengths;
- `episodes.csv`: reward, baseline, advantage, and selected indices per episode;
- `policy_weights.csv`: learned comprehensive-score weights at every step;
- `summary.txt`: configuration, runtime, deterministic policy score, and best score.

The saved best set is the better of the strongest exploratory episode and a
final deterministic rollout of the trained policy.
