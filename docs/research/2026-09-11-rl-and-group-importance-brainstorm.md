# Beyond individual band rankings: RL, group importance, and next research directions

Brainstorm on the perturbation-autoencoder wavelength-selection method (TPAMI manuscript, `publications/tpami/`). Written 2026-09-11 against the current code and the revision notes. Three parts, as asked: (1) reinforcement-learning formulations and how each would work on our data, (2) importance of *combinations* of wavelengths rather than individual ranks, (3) broader directions. A prioritised roadmap and evaluation guardrails close the note.

## 0. What any new idea has to respect

Facts from the paper, the revision file, and the July investigations that constrain the design space:

| Fact | Where it comes from | Consequence |
|---|---|---|
| The method is three loosely coupled stages: 3D CAE (parallel excitation branches, average merge, 20-channel latent) -> first-order latent perturbation -> greedy MMR. Reviewer R1.1 asked for a unified objective; the planned rebuttal frames it as an information-bottleneck readout. | `methodology.tex`, `MITIGATION_TABLE.md` R1.1 | Anything that turns "rank then greedy" into an explicit set objective directly answers the strongest critique. |
| Influence is a **first-order** quantity: a finite-difference Jacobian column per latent dimension, summed with importance weights. Interactions between bands never enter; MMR only subtracts spectral cosine similarity afterwards. | `methodology.tex` eq. (sensitivity), (influence), (mmr) | Group importance needs second-order or set-function machinery; the current code cannot express "these three bands are informative together". |
| 75 % of the 2 h pipeline is the perturbation sweep, done by finite differences. | `discussion.tex` Limitations | Autograd Jacobian-vector products would make interaction analysis affordable. |
| K is user-chosen; the headline K on Lichens was picked with labels (R1.6). | `discussion.tex`, R1.6 | Set-level criteria (marginal gain, stability) can make K unsupervised. |
| On Lichens, the learned 13-band set beats all 10 000 random 13-band sets (90.2 % vs random mean 46 %, 12 sigma). Drop Data and PAMAP2 are saturated: random is within noise of everything, including supervised MI. | `results.tex`, `generalization/RESEARCH_LOG.md`, `saturation_diagnostic.py` | Lichens (and Collagen) are the testbeds with headroom. Run the saturation diagnostic before spending compute on any dataset. |
| The spatial CAE collapses the emission axis in the encoder and, on sparse synthetic cubes, converges to "predict the per-band mean" (reconstruction R about 0), making its influence noise; a per-pixel spectral MLP autoencoder does not. | `docs/spectraforge/02-investigation-and-findings.md` | New objectives should be tested on both encoders; the spectral MLP-AE is the safer backbone for synthetic training grounds. |
| Influence normalisation is a scope axis: `variance` inverts the ranking on Drop Data, `max_per_excitation` fixes it. Cluster-recovery metrics are gameable; per-pixel classification is the trustworthy criterion. | `discussion.tex` | Rewards and set utilities must be per-pixel, and should be reported under both normalisations. |
| The instrument acquires excitations sequentially (TLS steps the lamp, the LCTF steps emission); exposure brackets matter (Drop Data lesson). The grant's device objective (O5) and the pair-combination objective (O4) are live. | `DATA_GUIDE.md`, grant memory | Sequential-acquisition RL and combination search have hardware and funding relevance, not only methodological. |

## 1. Reinforcement learning: five formulations that fit the data

The common trap: RL as a wrapper around a fixed dataset is an expensive way to do what a genetic algorithm or simulated annealing does with the same reward budget. RL earns its place when (a) the reward is non-differentiable but cheap, (b) the decision is genuinely sequential (acquisition), or (c) the policy is trained once and **amortised** across samples so selection no longer needs per-sample retraining. Each formulation below states which of the three it relies on.

### 1.1 Band-subset selection as an episodic MDP with an unsupervised reward (amortised selector)

**Episode.** State = the set of bands chosen so far (as a binary vector over the 192 excitation-emission cells) plus a fixed per-band feature table. Action = add one unselected band. Length = K. Reward only at the end (or shaped as marginal gain per step).

**Reward without labels.** Two candidates, both cheap enough for millions of episodes:

* **Latent predictability**: with the trained autoencoder's latent `z` precomputed for the masked pixels, reward = R² of a ridge regression `X_S -> z` (closed form, sub-millisecond on 4 000 pixels). This is exactly the information-bottleneck reading the R1.1 rebuttal proposes: the subset from which the compressed representation is most recoverable is the subset that carries the mutual information `I(z; X)`. It also unifies stages 2 and 3 into one objective.
* **Sensing reconstruction**: reward = R² of reconstructing all 192 bands from `X_S` with a linear decoder (or the CAE decoder with the other inputs zeroed). This is the objective of the Concrete Autoencoder, evaluated combinatorially instead of through a Gumbel relaxation.

When labels exist, swap in per-pixel KNN-5 accuracy on the labelled ROI pixels; this yields a supervised RL wrapper that must be reported as such.

**Policy.** A small transformer or DeepSets over the 192 band tokens; token features = excitation, emission, per-band mean and variance, the current AE influence score, a selected flag. Output = softmax over unselected bands. Because the policy conditions on the *set already chosen*, it learns interactions ("after picking 365/480, prefer 415/560 rather than 365/490"), which the greedy MMR cannot.

**Training.** PPO or REINFORCE with a moving baseline; episodes sampled over random spatial crops, random pixel subsets, and SpectraForge scenes, so that the policy generalises rather than memorises one image. Amortisation is the payoff: a trained policy selects bands for a new sample of the same type in one forward pass per step, no perturbation sweep, which removes the 90-minute cost reviewers complained about.

**What to compare against.** Greedy MMR (the paper), Concrete Autoencoder (the closest differentiable relative, already flagged as a mandatory baseline in the research log), a genetic algorithm with the same reward budget, random-10k, and the exhaustive oracle at K = 2 and 3 (section 2.1 shows it is computable). If RL does not beat the GA at equal budget on a *fixed* image, its only defensible advantage is amortisation; design the experiment so that amortisation is measured (train on Lichens_2 crops, deploy on Lichens Dataset 1 without retraining).

### 1.2 Sequential acquisition: choose the next measurement (active sensing)

This is the formulation that is unique to our instrument and to the grant's device objective. A multi-excitation acquisition *is* a sequence of decisions: which excitation next, which emission window, which exposure. Today the protocol is fixed (8 excitations, 31 bands each, fixed exposures). An agent could instead:

* **State**: the measurements taken so far (a partially filled EEM per pixel or per region) and the running posterior of a classifier or clustering.
* **Action**: next (excitation, emission band) or next whole excitation with an exposure time; or "stop".
* **Reward**: terminal classification correctness (or, unsupervised, the confidence/entropy reduction of the full-spectrum clustering posterior), minus a per-measurement cost (acquisition time, photobleaching dose, saturation penalty if the chosen exposure clips).

Why it matters: on Drop Data the method is "silent" on the 310 and 340 nm excitations because they carry no discriminative information; an acquisition agent would learn to skip them and spend the time on brackets at 365 to 415 nm. The saturation penalty makes exposure bracketing part of the policy, which is precisely the Drop Data lesson turned into a decision rule. Related literature: active feature acquisition, "learning to acquire" in medical imaging, hard attention. Training ground: SpectraForge can simulate any acquisition order and exposure, so the policy can be trained in simulation and tested on the real cubes offline (all bands exist, the agent just reveals them in its own order). This produces a *dynamic* selection: different specimens receive different band sets, a story no current band-selection paper tells.

### 1.3 Budget allocation across excitations (hierarchical or multi-agent)

The parallel-branch architecture suggests a two-level decision: first how many bands each excitation gets (the expensive dimension: each excitation is a lamp setting or LED), then which emission bands within it (cheap: the LCTF). Formulate as a hierarchical MDP or as one agent per excitation branch that bids for a shared budget. Reward as in 1.1. The output is exactly the structured selection an instrument designer needs ("three LEDs, four filters each"), and the per-excitation budgets are an interpretable group-importance statement in their own right (compare with section 2.4).

### 1.4 Learned perturbation (RL inside the attribution step)

Replace the 3 072-configuration sweep (which latent dimensions, which magnitudes, which normalisation) by a bandit that learns which perturbations produce the most *discriminative* influence maps, with the unsupervised reward of 1.1 as feedback. This is more honestly a hyper-parameter search; Bayesian optimisation would likely do it with fewer evaluations. Listed for completeness, low priority.

### 1.5 RL over formulas, not only bands (links to grant objective O4)

Grant O4 asks for "pair combinations and the formulas that combine them" (ratios, differences, normalised indices such as NDVI-style contrasts). Monte Carlo tree search over expression trees whose leaves are the shortlisted bands and whose internal nodes are `+ - * /`, with the reward of 1.1 or a supervised reward, discovers interpretable indices. Symbolic regression toolkits do this with genetic programming; an RL/MCTS variant is a legitimate alternative when the reward is a classification metric. Start from the AE shortlist (top 20 bands) so the search space is tiny.

## 2. From individual rankings to combination importance

The question "which three bands, together, are important" is a set-function question. Write `v(S)` for the utility of a band subset (per-pixel KNN accuracy when labelled; latent-predictability R² or reconstruction R² when not). Everything below is a way of computing, approximating, or learning structure in `v`.

### 2.1 The exhaustive small-K oracle (do this first: it is the ground truth for everything else)

On the balanced 4 000-pixel Lichens subset used for the SOTA comparison:

* all pairs: C(192, 2) = 18 336 subsets;
* all triplets: C(192, 3) = 1 161 280 subsets.

A KNN-5 evaluation on 4 000 pixels with 3 features is one 4 000 x 4 000 distance matrix, about a millisecond on a GPU with batched `torch.cdist`; the full triplet landscape is therefore 20 to 40 minutes, and a few hours with 5-fold cross-validation. With the ridge proxies it is minutes. This gives, for the first time, the **true landscape** of combination importance on a real ME-HSI dataset:

* the distribution of triplet accuracies (how many near-optimal triplets exist; how rugged the landscape is);
* for every band, its best-triplet accuracy versus its solo accuracy: "team players" (weak alone, strong in a set) versus "soloists";
* where the AE-ranked bands sit in that landscape (do the top-9 bands contain the best triplets, or do they miss them?).

It also validates any RL or DPP selector at K = 3 exactly, and it turns the paper's "10 000 random 13-band sets" robustness argument into a complete enumeration at small K. The figure practically writes itself: a 192 x 192 heatmap of best-triplet accuracy for each pair, drawn on the EEM grid.

### 2.2 Synergy and redundancy from the set function: Shapley and Möbius terms

With `v` available (exhaustively at small K, by sampling at larger K), compute:

* **pairwise synergy** `syn(i, j) = v({i, j}) - v({i}) - v({j})`: positive means complementary, negative means redundant. The 192 x 192 synergy matrix, displayed on the EEM grid, is the "combination importance EEM" the question asks for; its block structure shows which excitation regions cooperate.
* **Shapley values** of bands under `v` (Monte Carlo over random orderings) as an interaction-aware replacement for the individual ranking, and **Shapley interaction indices** or **Möbius (Harsanyi) dividends** for pairs and triplets, which isolate the part of `v({i, j, k})` that none of the sub-coalitions explain. Reporting the top-10 triplet dividends *is* the requested "combinations of three that reveal importance together".

Cost is dominated by evaluations of `v`; with the ridge proxy tens of millions of evaluations are affordable.

### 2.3 Group importance from the autoencoder itself (no labels, no enumeration)

The current attribution reads only the diagonal of the problem. Three model-based routes to sets:

* **Input-space group knockout.** Zero (or mean-fill) a band subset `S` in the input, re-encode, and measure the latent displacement `‖z(X) - z(X_{-S})‖` and the reconstruction error on the *other* bands. A large displacement means the group carries information the latent needs; a subset whose bands the decoder can reconstruct from the complement is redundant. Superadditivity of knockout effects (effect of `S` larger than the sum of its members) is model-based synergy. This needs only forward passes.
* **Full decoder Jacobian, not its column norms.** `J` is the 192 x 20 matrix of band responses to latent directions (autograd, exact, one backward pass per latent dimension). Bands with similar Jacobian rows respond to the same latent directions and are functionally redundant; clustering the rows gives *latent-functional* groups, a better similarity for the MMR step than raw spectral cosine. The singular vectors of `J` are "band modes": each right singular vector is a group of bands that moves together, ranked by singular value. The top three modes are three interpretable combinations, and they answer R1.1 halfway: they are the principal directions of `I(z; X)` restricted to bands.
* **Second-order sensitivity.** Perturb two latent dimensions jointly and compare with the sum of single perturbations; map the residual through `J` to band pairs. This is the Hessian of the decoder and is the model-based counterpart of 2.2. Autograd makes it cheap; finite differences made it prohibitive, which is why the paper stopped at first order.

### 2.4 Excitation-level groups: exact Shapley over eight branches

Because each excitation has its own encoder branch, the contribution of an *excitation* is a branch-ablation question. Train the CAE with **branch dropout** (randomly zero whole excitation branches during training, as in modality dropout) so it tolerates missing branches; then evaluate `v(E)` for every subset `E` of the 8 excitations: 2^8 = 256 forward passes. Exact Shapley values and all interaction terms for excitations follow with no sampling. This gives the cleanest possible statement of "which excitations are worth an LED", it recovers the "silent excitation" finding on Drop Data quantitatively, and it makes the model robust to a missing lamp line at deployment.

### 2.5 Set-level selection instead of greedy ranking

Two principled replacements for MMR that output *sets* with set scores:

* **k-DPP (determinantal point process)** over bands with a kernel built from Jacobian rows (or spectral profiles) scaled by relevance. A k-DPP assigns a probability to every K-subset that rewards both relevance and diversity, sampling gives several high-quality alternative combinations instead of one ranking, and the marginal probability of each band within sampled sets is an interaction-aware importance. MMR is a greedy approximation to this.
* **Submodular maximisation** of a set utility (coverage of the latent modes, or the knockout displacement of 2.3) with the greedy algorithm's (1 - 1/e) guarantee. This gives the "unified objective" language reviewers wanted, with MMR recovered as a special case.

### 2.6 Co-selection statistics from what already exists

The MasterRun produced 3 072 configurations, each with a ranked selection, and 16 above-baseline configuration groups. Build the co-occurrence graph of bands across the above-baseline selections, detect communities, and count the most frequent triplets. This costs an afternoon, needs no GPU, and is the first "combinations, not rankings" figure; it also supports the stability-based K selection of section 3.

### 2.7 Physical combinations

Beyond sets, the physics suggests *derived* features: the ratio of two emission bands under one excitation is independent of local intensity and often maps to a concentration ratio of two fluorophores; the ratio of the same emission under two excitations reads the excitation spectrum. Include such ratios and differences over the shortlisted bands as candidate features in 2.1 and 2.2, and use PARAFAC/MCR-ALS components (section 3.3) to interpret which triplets correspond to which fluorophore pairs. This is the concrete path to the grant's "formulas that combine them".

## 3. Broader directions

### 3.1 A unified objective (answers R1.1 in code, not only in prose)

* **Differentiable selection layer**: a Concrete/Gumbel gate on the 192 inputs, trained with the reconstruction loss plus a budget penalty, initialised from the perturbation influence. Compare against the perturbation readout; report the gap. If they agree, the perturbation method is a cheap approximation of an explicit objective; if they disagree, the disagreement is the paper.
* **Latent-predictability as the single objective** (from 1.1): select `S` to maximise R²(`X_S -> z`) under a diversity regulariser. Both stage 2 and stage 3 become one optimisation problem, and RL, DPP and greedy are then just solvers.

### 3.2 Fix the backbone where it breaks

The SpectraForge finding stands: the spatial CAE can degenerate on sparse cubes. Candidates, in the order of the existing C0-C4 ladder plan: per-pixel spectral MLP-AE, then a hybrid (spectral encoder per pixel followed by shallow spatial context), signed latent, real bottleneck. Add **branch dropout** (2.4) and an **exposure-invariance augmentation** (random per-excitation gain and additive noise) so the latent cannot encode lamp brightness; that is the model-side version of the radiometric lesson. A contrastive/self-supervised objective (augmentations = gain jitter, noise, spatial crops) is a natural alternative to reconstruction and would make the latent explicitly invariant to the nuisances that fooled Drop Data v1.

### 3.3 Physics-grounded attribution

The EEM is trilinear (pixel x excitation x emission), which is precisely what **PARAFAC** decomposes; MCR-ALS is its non-negative cousin. Fit K components, match them to the SpectraForge fluorophore library, and express each selected band or triplet as "reads component 2 (collagen-like) against component 3". This delivers the interpretable attribution the paper lists as future work and gives a physics baseline for the AE latent (do the latent dimensions correlate with PARAFAC scores?). A dictionary decoder (fixed fluorophore spectra, learned concentrations) is the fully physics-informed autoencoder variant.

### 3.4 Unsupervised K and uncertainty

* **K from stability**: the size at which bootstrap selections (seeds, crops) stop changing (Jaccard stability curve) or at which the marginal latent-predictability gain drops below the noise floor. Removes the R1.6 objection.
* **Uncertainty on ranks and sets**: the model already has dropout 0.5; Monte Carlo dropout at inference gives a distribution over influence maps for free, and bootstrap ensembles give confidence intervals on Shapley values. Report bands as "selected in 94 % of resamples".

### 3.5 Speed

Replace finite-difference perturbation by autograd Jacobian-vector products: exact, batched, and it unlocks the second-order analysis of 2.3. Expect the 90-minute step to drop to minutes.

### 3.6 Transfer and a pre-trained encoder

Train the encoder once on all lab datasets plus SpectraForge scenes with branch dropout and exposure augmentation, then attribute per sample without retraining. Combined with the amortised RL policy of 1.1, selection for a new specimen becomes seconds. Test transfer honestly: Lichens_2 to Lichens Dataset 1 (same specimens, different resolution), Collagen acetic to pepsin, and simulation to real.

### 3.7 Evaluation hygiene that should become standard

* Run `saturation_diagnostic.py` (headroom = best supervised K-subset minus random K-subset) before any experiment; if the headroom is under 5 points, no selector can show anything.
* Always report random, variance, PCA-loading, supervised MI and, where available, the exhaustive oracle.
* Per-pixel metrics only; cluster-recovery metrics as secondary.
* Both influence normalisations (`variance`, `max_per_excitation`).
* Multiple seeds with confidence intervals (R1.8).

## 4. Prioritised roadmap

| Priority | Item | Effort | Depends on | Why first |
|---|---|---|---|---|
| 1 | Exhaustive pair and triplet oracle on Lichens (2.1) with KNN and the ridge proxies | 2 to 3 days | balanced 4 000-pixel subset, GPU | ground truth for every later claim; a strong figure on its own |
| 2 | Co-selection graph and frequent triplets from the MasterRun (2.6) | 1 day | `results/Lichens_Dataset_1_MasterRun` | free, immediate "combinations" result |
| 3 | Autograd Jacobian + band modes + Jacobian-row clustering as the MMR similarity (2.3, 3.5) | 3 to 5 days | trained model | speeds the pipeline and gives model-based groups |
| 4 | Synergy matrix, Shapley values and triplet dividends (2.2) | 3 days | items 1 and 3 | the requested group-importance analysis |
| 5 | Branch dropout + exact excitation Shapley (2.4) | 1 week | retraining | instrument-level result; feeds the device objective |
| 6 | Latent-predictability objective + k-DPP / submodular solver (1.1 reward, 2.5, 3.1) | 1 to 2 weeks | items 1 and 3 | the unified objective; baseline for RL |
| 7 | Amortised RL selector (1.1) trained on crops and SpectraForge, tested by transfer | 3 to 4 weeks | item 6, spectral MLP-AE backbone | the RL paper, if it beats GA at equal budget or shows amortisation |
| 8 | Sequential-acquisition agent in SpectraForge, then offline on real cubes (1.2) | 4 to 6 weeks | SpectraForge acquisition simulator, item 5 | the instrument-native RL story; grant O5 |
| 9 | PARAFAC/MCR attribution and formula search (2.7, 3.3, 1.5) | 2 to 3 weeks | items 1 and 4 | grant O4; interpretability |
| 10 | Unsupervised K, MC-dropout uncertainty, exposure-invariant pre-training (3.4, 3.2, 3.6) | ongoing | | closes the remaining reviewer points |

Items 1 to 4 are a self-contained short paper ("How much of ME-HSI band importance is interaction?"), and they de-risk the RL work by telling us whether interactions are large enough to matter on real data. If the exhaustive landscape shows that the best triplets are simply the top-3 individual bands, RL and DPP have little to gain and the effort should go to acquisition (1.2) and physics (3.3) instead.

## 5. Risks and honest expectations

* RL against a fixed dataset is a wrapper; a GA or simulated annealing with the same reward calls is the fair comparison and may win. Design for amortisation and sequential acquisition, where RL has a structural advantage.
* Supervised rewards contradict the unsupervised positioning; keep the two tiers explicit (unsupervised proxy rewards for selection, labels only for evaluation) or the R1.6 critique returns.
* Interaction analyses are only as good as `v`; the ridge proxies are linear and can miss nonlinear synergies, so anchor them to the KNN oracle at small K.
* Saturated datasets (Drop Data, PAMAP2) will show nothing; use Lichens and Collagen, and use SpectraForge only with the discriminability-grounded ground truth that is planned, not the broad informative-band mask that was retracted.
