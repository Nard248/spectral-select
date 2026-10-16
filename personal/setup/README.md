# Small RL wavelength selector

This folder is a self-contained reinforcement-learning baseline for
the labelled Lichens_2 image. Each action is one `(excitation_nm, emission_nm)` plane.
The policy samples a fixed number of unique actions and is optimized with REINFORCE.

## Reward and data split

The class mask contains multiple disconnected lichen specimens per class. Whole
connected specimens are assigned to train, validation, or test; adjacent pixels from
one specimen never cross partitions. Training pixels define standardized class
centroids. The terminal reward is:

```text
validation balanced accuracy - redundancy_weight * mean absolute band correlation
```

The policy sees validation reward. The test specimens are used once after selection.
This is preferable to a random-pixel split, but one image is still only a pilot study.

## Run

From the repository root, install this mini-project into the existing environment:

```bash
.venv-1/bin/pip install -e personal/setup --no-build-isolation
.venv-1/bin/lichen-rl-select --budget 10 --updates 100
```

Defaults use the exposure- and lamp-power-normalized processed cube under
`personal/Processed/Lichens_2`. Outputs are written inside this setup folder at
`personal/setup/results/rl_lichens2/`:

- `policy.pth`: learned per-band policy logits and selected action indices.
- `selection.json`: wavelengths, split sizes, training history, metrics, and random,
  variance, Fisher-score, and full-band baselines.

The model is deliberately small: it has one trainable logit per candidate band. It is
best understood as a sequential combinatorial-bandit baseline. A later version can
replace the nearest-centroid reward with held-out CNN performance without changing
the action space or selection policy.

## Version 2: exact triple oracle and transfer

The separate `lichen-rl-oracle-transfer` command enumerates all three-band
combinations, retains the top combinations, derives per-band and pairwise priors,
and uses those priors to initialize a state-dependent policy for a larger budget.
It also trains the original policy from scratch for a direct transfer comparison.

For 210 candidates, the complete oracle evaluates 1,521,520 triples:

```bash
.venv-1/bin/lichen-rl-oracle-transfer \
  --budget 13 \
  --updates 150 \
  --output personal/setup/results/oracle_transfer
```

The oracle periodically writes `triple_oracle_checkpoint.json`. Re-running the
same command resumes an interrupted enumeration. Once complete, the reusable
`triple_oracle.npz` contains the top triples plus band and pair priors.

For a quick functional check without enumerating the complete search space:

```bash
.venv-1/bin/lichen-rl-oracle-transfer \
  --max-combinations 1000 \
  --top-k 100 \
  --budget 13 \
  --updates 5 \
  --output personal/setup/results/oracle_transfer_smoke
```

Main outputs:

- `triple_oracle.npz`: top triples, exact rewards, band prior, pair prior.
- `triple_oracle_checkpoint.json`: resumable enumeration state.
- `oracle_transfer_policy.pth`: larger-budget state-dependent transferred policy.
- `scratch_policy.pth`: same-budget policy trained without oracle initialization.
- `oracle_transfer_report.json`: wavelengths, histories, and comparison metrics.

“Exact” refers to the configured fixed validation reward and validation sample. The
default uses up to 1,000 validation pixels per class. Use
`--oracle-pixels-per-class 0` to use every validation pixel.

### Completed reference run (seed 42)

The complete 210-choose-3 run evaluated all 1,521,520 triples. Its optimal triple
under the configured validation reward was:

```text
Ex 400 / Em 460 nm
Ex 415 / Em 665 nm
Ex 430 / Em 590 nm
```

It achieved test balanced accuracy 0.7793. For a 13-band budget, the transferred
policy achieved 0.8058 versus 0.7269 for the same REINFORCE training from scratch.
These are results for one object split and seed, not yet a multi-seed significance
claim. The full details are in `results/oracle_transfer/oracle_transfer_report.json`.

## Important interpretation

The selected wavelengths are optimized for the five labels in
`lichens_2_class_mask.npy` and for this reward model. They are not yet evidence of a
generally optimal lichen wavelength set. That claim requires repeated seeds and,
ideally, held-out images or specimens from separate acquisitions.

## Select a different number of bands

Use `--budget` for the exact number of unique bands. Keep separate output folders
when comparing budgets:

```bash
.venv-1/bin/lichen-rl-select --budget 3  --output personal/setup/results/n3
.venv-1/bin/lichen-rl-select --budget 5  --output personal/setup/results/n5
.venv-1/bin/lichen-rl-select --budget 13 --output personal/setup/results/n13
```
# Monte-Carlo wavelength selection (separate version)

The `wavelength_selection` package performs leakage-safe, grouped-CV feature
selection over excitation/emission pairs. It audits the raw wavelength grid,
applies first- and second-order Rayleigh rejection, uses the normalized
processed cube, and selects exactly `n` unique pairs using marginal reward,
Monte-Carlo future completions, individual usefulness, and redundancy.

Quick smoke run from `personal/setup`:

```bash
python -m wavelength_selection.run_experiment --n-pairs 3 --episodes 2 --rollouts 2
```

For a substantive run, increase `--episodes` and `--rollouts`. Results are
written to `results/monte_carlo_selection`: `summary.json` contains the best
pairs and baseline scores; `episodes.json` contains every action and rollout
statistic; CSV files contain pair-level metrics and rejection reasons; PNG
files show scores, selection frequency, and the selected EEM positions.
