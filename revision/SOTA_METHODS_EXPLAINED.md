# SOTA Methods — Definitions and Math

**Companion document to the spectral-select paper revision.**

**SOTA** = **State-Of-The-Art**. In the band-selection literature it is shorthand for "the published methods researchers compare against." In our paper, the SOTA comparison means: against the methods most commonly cited as standards in the hyperspectral band-selection community.

We compared 9 methods (8 baselines + ours), spanning the four standard families used in the field.

---

## Setup notation

For all methods:

- $X \in \mathbb{R}^{N\times B}$ — feature matrix; $N$ pixels × $B$ bands
- $X_j \in \mathbb{R}^N$ — column $j$ (the values of band $j$ across all pixels)
- $y \in \{1,\dots,C\}^N$ — class labels (only for supervised methods)
- $K$ — desired number of selected bands
- $S$ — the selected set, $|S| = K$

Each method outputs an ordered list of $K$ indices from $\{1,\dots,B\}$.

---

## Family 1: Filter methods

Cheap, label-free, per-band statistic.

### 1. Variance

For each band, compute its variance across pixels and rank descending.

$$\sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N}(x_{ij} - \bar{x}_j)^2, \qquad S = \text{top-}K\bigl(\sigma_1^2,\dots,\sigma_B^2\bigr)$$

**Intuition.** Bands with more variation are presumed more informative. Cheap ($O(NB)$). Fails when noise bands have high variance, or when discriminative bands have low variance.

### 2. PCA-loading

Compute PCA of $X$; rank bands by their summed absolute loadings across top components, weighted by explained variance.

$$X^TX = V\Lambda V^T \quad (\text{eigendecomposition}),\qquad \text{score}_j = \sum_{k=1}^{r}\frac{\lambda_k}{\sum_l \lambda_l}\,|V_{jk}|$$

where $V$ is the loading matrix and $\lambda_k$ are eigenvalues (variance per component). Select top-$K$ bands by score.

**Intuition.** A band that loads strongly on a high-variance principal component is "explaining" a lot of the data's spread. Like variance but multi-dimensional.

---

## Family 2: Wrapper-unsupervised

Iterative, label-free, residual-driven.

### 3. SAM-greedy (Spectral Angle Mapper, greedy)

Greedy selection where each new band maximizes the minimum spectral angle to the already-selected set.

$$\theta(X_j, X_k) = \arccos\!\left(\frac{|\langle X_j, X_k\rangle|}{\|X_j\|\,\|X_k\|}\right)\in[0,\tfrac{\pi}{2}]$$

Selection:

- **Init:** chosen $= \{\arg\max_j \operatorname{Var}(X_j)\}$
- **At step $t+1$:** for each unselected $j$, compute $\min_k \theta(X_j, X_k)$ over $k \in$ chosen
- **Pick:** $j^* = \arg\max_j\,\min_k \theta(X_j, X_k)$

**Intuition.** Each new band points in a maximally different spectral direction. Pure-diversity method (no relevance signal). Fails as $K$ grows because eventually only low-signal bands are left.

### 4. SPA (Successive Projections Algorithm)

Araújo et al. 2001. Classic chemometrics method. Iteratively project residual onto the orthogonal complement of selected bands; pick the band with maximum residual norm.

- **Init:** $j_1 = \arg\max_j \|X_j\|$
- **Project away** the chosen bands' span: $R = (I - P_{X_S})\,X$, where $P_{X_S}$ is the projection onto the column space of $X$ restricted to $S$
- **Pick:** $j_{t+1} = \arg\max_j \|R_j\|$
- **Update** $S \leftarrow S \cup \{j_{t+1}\}$; repeat until $|S| = K$

Equivalently:

$$j_{t+1} = \arg\max_j \;\bigl\|X_j - X_S(X_S^T X_S)^{-1} X_S^T X_j\bigr\|^2$$

**Intuition.** Each new band brings the maximum energy *not yet captured* by previous selections. Approximately a basis-selection in the EEM space.

### 5. MCUVE (Monte Carlo Uninformative Variable Elimination)

Centner et al. 1996. Bootstrap stability of regression coefficients. In unsupervised mode, the target is PC1 of $X$.

- **Target:** $y_{\text{PC1}}$ = first principal component score of $X$
- **For $m = 1, \ldots, M$ (bootstrap iterations):**
  - sample $N/2$ pixel indices uniformly without replacement
  - fit per-band regressions: $\beta_j^{(m)} = \langle X_j[\text{subsample}], y_{\text{PC1}}[\text{subsample}]\rangle / \|X_j[\text{subsample}]\|^2$
- **Stability score** per band:

$$\text{stability}_j = \frac{\bigl|\,\mathbb{E}_m\,\beta_j^{(m)}\,\bigr|}{\sigma_m\!\bigl(\beta_j^{(m)}\bigr)}$$

Rank descending, pick top $K$.

**Intuition.** Bands whose regression coefficient is *consistent across resamples* are informative; bands with high coefficient variance are noise.

---

## Family 3: Cluster-based unsupervised

### 6. ISSC (Improved Sparse Subspace Clustering)

Adapted from Elhamifar & Vidal 2013 (Sparse Subspace Clustering). Treats each band as a data point in pixel-space, learns sparse self-representation, spectral-clusters bands, then picks one representative per cluster.

**Step 1 — Self-expression (lasso per band):**

$$\min_{c_j \in \mathbb{R}^{B-1}}\;\|X_j - X_{\setminus j}\,c_j\|_2^2 + \alpha\,\|c_j\|_1$$

For each band $j$, solve a sparse regression of $X_j$ onto all other bands. Stack the coefficient vectors into matrix $C$ with $C_{jj} = 0$.

**Step 2 — Build affinity matrix:**

$$W = |C| + |C|^T$$

**Step 3 — Spectral clustering** on $W$ into $K$ clusters (eigendecompose graph Laplacian, k-means in the spectral embedding).

**Step 4 — Representatives:** for each cluster, pick the highest-variance band.

**Intuition.** Bands that lie in the same low-dimensional subspace as some other band are redundant; one representative per "subspace cluster" gives a diverse $K$-band set.

---

## Family 4: Deep unsupervised

### 7. BS-Net-FC (Band Selection Network — Fully Connected)

Cai, Liu, Chanussot 2020 (IEEE TGRS). End-to-end neural network with attention + reconstruction loss. Bands are ranked by their average attention weight after training.

**Architecture:**

$$w = \sigma\!\bigl(W_3\,\operatorname{ReLU}(W_2\,\operatorname{ReLU}(W_1 x + b_1) + b_2) + b_3\bigr) \quad \text{(attention subnet, } w \in [0,1]^B\text{)}$$

$$\tilde{x} = w \odot x \quad \text{(element-wise gated input)}$$

$$\hat{x} = W_6\,\operatorname{ReLU}(W_5\,\operatorname{ReLU}(W_4\,\tilde{x} + b_4) + b_5) + b_6 \quad \text{(reconstruction)}$$

**Loss:**

$$\mathcal{L} = \underbrace{\mathbb{E}_x\,\|x - \hat{x}\|_2^2}_{\text{reconstruction MSE}} + \lambda\,\underbrace{\mathbb{E}_x\,\overline{w(x)}}_{\text{sparsity penalty}}$$

After training via Adam, compute per-band mean attention:

$$\bar{w}_j = \mathbb{E}_x\,w_j(x), \qquad S = \text{top-}K\,(\bar{w}_1, \dots, \bar{w}_B)$$

**Intuition.** The attention mask learns to mask out bands that don't help reconstruction; the sparsity penalty pushes most weights to zero so a clear top-$K$ emerges.

### 8. AE-perturb (OURS — proposed method)

Latent-space perturbation attribution. The autoencoder is 3D-convolutional with parallel branches per excitation; perturbation analysis traces reconstruction sensitivity back to individual $(\lambda_{\text{ex}}, \lambda_{\text{em}})$ cells.

**Step 1 — Train autoencoder** to minimize masked MSE: $F: x \to z$ encoder, $G: z \to \hat{x}$ decoder.

$$\min_{F,G}\;\mathbb{E}_x\,\|M \odot (x - G(F(x)))\|_2^2$$

where $M$ is a foreground mask.

**Step 2 — Rank latent dimensions.** For $d = 1, \dots, \dim(z)$, score dim $d$ by either $\operatorname{Var}_x(z_d)$ or by PCA loading on the latent activations. Take top-$k$ dims.

**Step 3 — Finite-difference perturbation.** For each top-$k$ dim $d$:

$$z^{\pm}_d = z \pm \epsilon\,\sigma_d\,e_d$$

($e_d$ is one-hot; $\sigma_d$ is the standard deviation of dim $d$ across samples)

Influence at band $(\lambda_{\text{ex}}, \lambda_{\text{em}})$:

$$\Delta_d(\lambda_{\text{ex}}, \lambda_{\text{em}}) = \frac{\bigl|G(z^{+}_d)_{\lambda_{\text{ex}}, \lambda_{\text{em}}} - G(z^{-}_d)_{\lambda_{\text{ex}}, \lambda_{\text{em}}}\bigr|}{2\,\epsilon\,\sigma_d}$$

This is a **finite-difference approximation of $|\partial G / \partial z_d|$** at the operating point — i.e., how much reconstruction at this band changes when we wiggle latent dim $d$.

**Step 4 — Aggregate** across dims weighted by dim score, and across $\epsilon$ values:

$$\text{Influence}(\lambda_{\text{ex}}, \lambda_{\text{em}}) = \sum_{d \in \text{top-}k} \text{score}(d)\,\overline{\Delta_d(\lambda_{\text{ex}}, \lambda_{\text{em}})}_{\epsilon}$$

**Step 5 — MMR diversity selection.** Greedy:

$$w^* = \arg\max_{w \notin S}\;\Bigl[\,\lambda\,\text{Rel}(w) - (1 - \lambda)\,\max_{w' \in S}\,\text{Sim}(w, w')\,\Bigr]$$

with $\text{Rel}$ = normalized influence, $\text{Sim}$ = cosine similarity between full spectral profiles at the two bands, $\lambda = 0.5$.

**Intuition.** Influence is a *causal* attribution: it measures what the autoencoder's compressed representation actually depends on. Combined with MMR, the result is diverse + relevant. Distinct from BS-Net because attribution is post-hoc on a model trained for reconstruction only, not jointly optimized with a sparsity penalty.

---

## Family 5: Embedded supervised

### 9. Sparse-LASSO

L1-penalized multinomial logistic regression. Uses labels.

**Optimization:**

$$\min_{W \in \mathbb{R}^{C \times B}}\;\sum_{i=1}^{N}\,\ell_{\text{CE}}\!\bigl(y_i,\,Wx_i\bigr)\;+\;\frac{1}{C}\,\|W\|_1$$

where $\ell_{\text{CE}}$ is cross-entropy. L1 regularization induces *column sparsity* in $W$; many bands end up with all-zero coefficients across all classes.

**Importance per band:**

$$\text{importance}_j = \max_{c \in \{1, \dots, C\}}|W_{cj}|$$

**Hyperparameter sweep.** L1 strength $\lambda_{L1}$ is swept over a small grid; smallest $\lambda_{L1}$ that leaves $\geq K$ non-zero importance scores is kept.

Top-$K$ by importance.

**Intuition.** Direct supervised band selection. The L1 norm gives a "discrete" feature selector via column-zeroing.

---

## Baseline: Random

$$S \sim \text{Uniform}\{\,K\text{-subsets of }\{1, \dots, B\}\,\}$$

Sampled $R$ times to compute mean $\pm$ std. The benchmark every method must beat.

---

## Summary table

| Method | Family | Uses labels? | Cost | Selection mechanism |
|---|---|---|---|---|
| Variance | Filter | No | $O(NB)$ | Per-band statistic |
| PCA-loading | Filter | No | $O(NB^2 + B^3)$ | Per-band statistic from loadings |
| SAM-greedy | Wrapper | No | $O(K \cdot NB)$ | Iterative diversity-only |
| SPA | Wrapper | No | $O(K \cdot NB^2)$ | Iterative residual-norm |
| MCUVE | Wrapper | No (or yes) | $O(M \cdot NB)$ | Bootstrap stability |
| ISSC | Cluster | No | $O(B \cdot (N \cdot B + B^3))$ | Sparse self-expression + spectral clustering |
| BS-Net-FC | Deep | No | $O(N \cdot B \cdot \text{epochs})$ | Learned attention + sparsity penalty |
| **AE-perturb (ours)** | Deep | No | $O(N \cdot B \cdot \text{epochs} + \text{perturbations})$ | **Reconstruction-sensitivity attribution + MMR** |
| Sparse-LASSO | Embedded | **Yes** | $O(N \cdot C \cdot B \cdot \text{iter})$ | L1 logistic coefficients |
| Random | Baseline | No | $O(1)$ | Uniform sampling |

---

## Why we picked these eight

Together they cover the four standard categorizations in the band-selection literature (filter / wrapper / embedded / deep) and span supervised and unsupervised regimes. Comparing against all of them lets us claim that the method's gains do not come from:

- **Generic dimensionality reduction** — refuted by *not losing to Variance*
- **Being a deep method** — *BS-Net-FC is the deep-method control*
- **Having labels** — *Sparse-LASSO is the supervised control*
- **Pure orthogonal projection** — *SPA is the chemometrics workhorse*
- **Variable-stability** — *MCUVE is the classical statistical control*
- **Subspace clustering** — *ISSC covers this family*
- **Diversity-only heuristics** — *SAM-greedy is the failure-mode control*

For a "why not method X?" question, the answer is: *X falls into one of these four families, which the comparison already covers; we picked the most-cited representative of each family.*

---

## References

- **SPA**: Araújo, M. C. U., Saldanha, T. C. B., Galvão, R. K. H., Yoneyama, T., Chame, H. C., & Visani, V. (2001). The successive projections algorithm for variable selection in spectroscopic multicomponent analysis. *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73.

- **MCUVE**: Centner, V., Massart, D. L., de Noord, O. E., de Jong, S., Vandeginste, B. M., & Sterna, C. (1996). Elimination of uninformative variables for multivariate calibration. *Analytical Chemistry*, 68(21), 3851-3858.

- **Sparse Subspace Clustering** (ISSC basis): Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(11), 2765-2781.

- **BS-Net-FC**: Cai, Y., Liu, X., & Cai, Z. (2020). BS-Nets: An end-to-end framework for band selection of hyperspectral image. *IEEE Transactions on Geoscience and Remote Sensing*, 58(3), 1969-1984.

- **Sparse-LASSO / L1 logistic regression**: Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

- **MMR (used inside AE-perturb)**: Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98*.
