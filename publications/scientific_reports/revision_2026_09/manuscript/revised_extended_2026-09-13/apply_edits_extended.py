"""Apply the agreed text edits to manuscript/revised/ (never touches original_v6/).

Every edit is an exact-string replacement that must match exactly once; the script aborts otherwise,
so nothing is changed silently. Edit IDs follow SciRep_Minimal_Change_Guide.docx (E1-E21) plus the
Tier B corrections agreed on 2026-09-12 (T1-T7) and two consistency edits forced by E21 (C1, C2).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE / "revised"
SEC = REV / "sections"

EDITS: list[tuple[str, str, str, str]] = []  # (edit id, file, old, new)


def E(eid, fname, old, new):
    EDITS.append((eid, fname, old, new))


# =============================================================================== introduction.tex
E("E14", "introduction.tex",
  r"exceeding the discriminative capabilities of RGB or multispectral data~\cite{plaza2009recent,bioucas2013hyperspectral}.",
  r"exceeding the discriminative capabilities of RGB or multispectral data~\cite{hong2026hyperspectral,hong2024spectralgpt}." + "\n"
  r"Its applications now span Earth observation, where it supports the detection of photovoltaic installations from satellite and airborne sensors~\cite{joerges2023photovoltaic} and the unsupervised unmixing of oceanic scenes affected by spectral variability and adjacency effects~\cite{deville2023oceanic}; agriculture and environmental monitoring~\cite{lu2020agriculture,lekha2026environmental}; food quality control~\cite{sun2024foodindustry}; and clinical imaging, where reflectance and autofluorescence hyperspectral data support diagnosis and image-guided surgery~\cite{yoon2022clinical,campbell2023beyondoncology}.")

E("E16a", "introduction.tex",
  r"The challenge of identifying informative spectral bands has been extensively studied for conventional 3D HSI~\cite{sun2019hyperspectral,jia2020survey,asfour2018,guan2018}, with approaches spanning filter-based methods (variance, entropy), wrapper methods (sequential forward selection, genetic algorithms), and embedded methods (sparse regression, attention mechanisms)~\cite{martinez2006comparison,chang2000constrained,pudil1994floating,pal2010feature,tibshirani1996regression,mou2017deep}.",
  r"The challenge of identifying informative spectral bands has been extensively studied for conventional 3D HSI~\cite{sun2019hyperspectral,sawant2021survey,rasti2020featureextraction,ahmad2022survey,jia2020survey,asfour2018,guan2018}, with approaches spanning filter-based methods (variance, entropy), wrapper methods (sequential forward selection, genetic algorithms), and embedded methods (sparse regression, attention mechanisms)~\cite{martinez2006comparison,chang2000constrained,pudil1994floating,pal2010feature,tibshirani1996regression,lorenzo2020attention,cai2020bsnets}.")

E("E16b", "introduction.tex",
  r"wrapper methods suffer from prohibitive computational cost when the search space is large, and embedded methods require labeled training data.",
  r"wrapper methods suffer from prohibitive computational cost when the search space is large, and embedded methods either require labeled training data or fix the number of selected bands at training time~\cite{balin2019concrete,cai2020bsnets}.")

# =============================================================================== related_work.tex
E("E19a", "related_work.tex",
  r"This section reviews prior work in four areas relevant to the present framework: hyperspectral imaging systems and data characteristics, band selection methods for hyperspectral data, deep learning approaches for spectral analysis, and explainability methods for feature attribution.",
  r"This section proceeds from the imaging modality and its dimensionality problem (Section~\ref{sec:related_hsi}), to the band-selection methods available for conventional cubes (Section~\ref{sec:related_band}), to the deep-learning and attribution tools on which the framework builds (Sections~\ref{sec:related_dl} and~\ref{sec:related_explain}), and closes with the gap the framework addresses (Section~\ref{sec:related_summary}).")

E("E19b", "related_work.tex",
  r"\subsection{Multi-Excitation Hyperspectral Imaging Systems}",
  r"\subsection{Multi-Excitation Fluorescence Imaging and the Dimensionality Problem}")

E("E19c", "related_work.tex",
  r"Such systems have proven particularly valuable for biological specimens, including lichens and plant tissues, where autofluorescence signatures encode information about molecular composition that reflectance imaging cannot capture~\cite{croce2019autofluorescence}.",
  r"Such systems have proven particularly valuable for biological specimens, including lichens and plant tissues, where autofluorescence signatures encode information about molecular composition that reflectance imaging cannot capture~\cite{croce2019autofluorescence}." + "\n"
  r"Recent reviews of multi- and hyperspectral autofluorescence imaging document a growing clinical and biological application base~\cite{campbell2023beyondoncology}, and multi-excitation autofluorescence hyperspectral imaging has been demonstrated for the exploration of biological samples~\cite{ghaffari2019multiexcitation}.")

E("E19d", "related_work.tex",
  r"\subsection{Band Selection for Hyperspectral Data}",
  r"\subsection{Band Selection for Hyperspectral Cubes}")

E("E17", "related_work.tex",
  r"""Band selection methods aim to identify a subset of spectral bands that retain discriminative information while reducing dimensionality~\cite{sun2019hyperspectral}.
These methods are typically categorized as filter, wrapper, or embedded approaches.

Filter methods rank bands based on statistical properties computed without reference to any downstream task.
Variance-based selection prioritizes bands with high variance across samples~\cite{chang2000constrained}, entropy-based methods favor bands with maximum information content~\cite{guo2006band}, and correlation-based approaches remove redundant bands that are highly correlated with others~\cite{martinez2006comparison}.
While computationally efficient, filter methods lack any notion of discriminative value---a band with high variance may capture noise rather than meaningful variation.
Moreover, they treat bands independently, ignoring the joint information captured by band combinations.

Wrapper methods evaluate band subsets based on the performance of a classifier trained on the selected features.
Sequential forward selection (SFS) greedily adds bands that most improve classification accuracy~\cite{pudil1994floating}, while genetic algorithms and particle swarm optimization explore the combinatorial search space more globally~\cite{pal2010feature}.
These methods can identify task-relevant bands but suffer from computational cost that grows combinatorially with the number of bands.
For a 192-band ME-HSI dataset, exhaustive evaluation of all possible subsets is intractable.

Embedded methods integrate band selection into model training.
Sparse regression (LASSO) induces feature selection through $L_1$ regularization~\cite{tibshirani1996regression}, attention mechanisms in neural networks learn to weight input bands~\cite{mou2017deep}, and reinforcement learning formulations treat band selection as a sequential decision process~\cite{roy2020hybridsn}.
While elegant, these approaches typically require labeled training data, limiting their applicability when ground truth is scarce or expensive to obtain.

Critically, nearly all existing band selection methods assume a 3D data structure where bands can be evaluated independently.
For ME-HSI, where information lies in excitation-emission \emph{pairs} and cross-excitation patterns may be discriminative, methods that explicitly model the 4D structure are needed.
Recent work on tensor decomposition for EEM data~\cite{murphy2013fluorescence} and multi-way analysis~\cite{bro1997parafac} addresses spectroscopic rather than imaging data.
The present framework bridges this gap by extending deep learning band selection to the 4D imaging case.""",
  r"""Band selection methods aim to identify a subset of spectral bands that retain discriminative information while reducing dimensionality~\cite{sun2019hyperspectral,sawant2021survey}.
The main families, their specific contribution and their limitation for ME-HSI data are as follows.
\begin{enumerate}
\item \textbf{Filter methods} rank bands by statistics computed without reference to any downstream task: variance~\cite{chang2000constrained}, entropy or mutual information~\cite{guo2006band}, and inter-band correlation~\cite{martinez2006comparison}.
  Contribution: negligible cost and no labels.
  Limitation: no notion of discriminative value (a high-variance band may capture noise) and independent treatment of bands, so excitation-emission combinations are never assessed jointly.
\item \textbf{Wrapper methods} score band subsets by the performance of a classifier trained on them: sequential forward selection~\cite{pudil1994floating} and genetic or particle-swarm search~\cite{pal2010feature}.
  Contribution: task-relevant subsets.
  Limitation: labels are required and the cost grows combinatorially with the number of bands; exhaustive evaluation of a 192-band ME-HSI grid is intractable.
\item \textbf{Sparse and embedded supervised methods} fold selection into model training through $L_1$ regularization~\cite{tibshirani1996regression} or attention layers trained for classification~\cite{lorenzo2020attention}.
  Contribution: selection and prediction optimized jointly.
  Limitation: labeled training data are required, limiting applicability when ground truth is scarce or expensive to obtain.
\item \textbf{Attention--reconstruction networks} learn band weights jointly with a reconstruction network and read the selection off the weights: BS-Nets~\cite{cai2020bsnets}, attention-based autoencoders~\cite{dou2021attentionae}, DARecNet-BS~\cite{roy2021darecnet} and TAttMSRecNet~\cite{nandi2023tattmsrecnet}.
  Contribution: label-free, end-to-end band weighting.
  Limitation: a single attention vector over a single spectral axis presumes one rectangular cube, and the selection is fixed at training time.
\item \textbf{Concrete-autoencoder selectors} make subset selection differentiable through a relaxed categorical layer~\cite{balin2019concrete,sun2022concrete}.
  Contribution: exact-$K$ unsupervised selection with a reconstruction objective.
  Limitation: $K$ is an architectural constant and the selector layer again assumes a single spectral axis.
\item \textbf{Reinforcement-learning selectors} cast band selection as a sequential decision process~\cite{mou2022reinforcement,guo2025twostage}.
  Contribution: explicit search with a learned policy.
  Limitation: the reward is a classification score or a per-step proxy, and the search cost grows with the number of candidate bands.
\item \textbf{Reconstruction-based combinatorial selection} chooses bands that best reconstruct their neighbours~\cite{wang2020onr}.
  Contribution: an unsupervised, interpretable objective.
  Limitation: band-to-band linear reconstruction of a 3D cube, without cross-excitation structure.
\item \textbf{Wavelength selection for fluorescence imaging} has been addressed for autofluorescence tissue imaging~\cite{asfour2018}, on the same modality family as the present work.
  Contribution: task-driven selection of acquisition wavelengths.
  Limitation: supervised and single-excitation.
\end{enumerate}

Critically, all of these methods assume a 3D data structure in which bands can be evaluated independently or along a single spectral axis.
For ME-HSI, where information lies in excitation-emission \emph{pairs} and cross-excitation patterns may be discriminative, methods that explicitly model the 4D structure are needed.
Recent work on tensor decomposition for EEM data~\cite{murphy2013fluorescence} and multi-way analysis~\cite{bro1997parafac} addresses spectroscopic rather than imaging data, and the closest imaging work, multi-excitation autofluorescence hyperspectral imaging of biological samples~\cite{ghaffari2019multiexcitation}, analyses the data by chemometric decomposition rather than by band selection.
The present framework bridges this gap by extending unsupervised deep band selection to the 4D imaging case.""")

E("E19e", "related_work.tex",
  r"\subsection{Deep Learning for Hyperspectral Analysis}",
  r"\subsection{Deep Learning and Autoencoders for Hyperspectral Data}")

E("E19f", "related_work.tex",
  r"Deep learning has substantially advanced hyperspectral image analysis, with convolutional neural networks achieving state-of-the-art results in classification, segmentation, and anomaly detection~\cite{li2019deep,paoletti2019deep}.",
  r"Deep learning has substantially advanced hyperspectral image analysis, with convolutional neural networks achieving state-of-the-art results in classification, segmentation, and anomaly detection~\cite{li2019deep,paoletti2019deep}; recent surveys trace the evolution from shallow to deep feature extraction~\cite{rasti2020featureextraction,ahmad2022survey}.")

E("E19g", "related_work.tex",
  r"\subsection{Explainability and Attribution Methods}",
  r"\subsection{Perturbation-Based Attribution}")

E("E19h", "related_work.tex",
  r"This approach has been applied to image classification by occluding image regions~\cite{zeiler2014visualizing}, to text analysis by removing words~\cite{li2016understanding}, and to tabular data by shuffling feature values~\cite{fisher2019all}.",
  r"This approach has been applied to image classification by occluding image regions~\cite{zeiler2014visualizing}, to text analysis by removing words~\cite{li2016understanding}, and to tabular data by shuffling feature values~\cite{fisher2019all}." + "\n"
  r"Meaningful-perturbation methods~\cite{fong2017perturbation} and the removal-based view that unifies occlusion, permutation and Shapley-type attributions~\cite{covert2021removing} provide the general framework; a survey of perturbation-based explanation methods is given in~\cite{ivanovs2021perturbation}.")

E("E19i", "related_work.tex",
  r"\subsection{Summary and Positioning}",
  r"\subsection{Gap Addressed by This Work}")

E("E19j", "related_work.tex",
  r"In summary, existing band selection methods predominantly assume a 3D hyperspectral data structure and rely on either statistical properties (filter methods), supervised classification performance (wrapper methods), or labeled training data (embedded methods).",
  r"In summary, the band-selection families of Section~\ref{sec:related_band} (items 1--8) either rely on labels (items 2, 3, 6 and 8), fix the number of bands at training time (items 4 and 5), or evaluate bands along a single spectral axis (all items), and none accommodates a variable number of emission bands per excitation.")

# =============================================================================== methodology.tex
E("T1a", "methodology.tex",
  r"\node[block, right=of encoder] (latent) {Shared\\Latent Space\\$\vect{z} \in \R^{20}$};",
  r"\node[block, right=of encoder] (latent) {Shared\\Latent Space\\$\vect{z} \in \R^{20 \times h \times w}$};")

E("T1b", "methodology.tex",
  r"(2) perturbation analysis attributes latent dimensions to individual wavelengths;",
  r"(2) perturbation analysis attributes latent coordinates to individual wavelengths;")

E("T1c", "methodology.tex",
  r"""Different latent dimensions encode different aspects of this structure; the goal was to attribute each latent dimension's influence back to specific excitation-emission wavelength pairs.
If perturbing a latent dimension significantly changed the reconstruction of a particular wavelength, that wavelength must be encoded by that dimension.""",
  r"""Different latent coordinates encode different aspects of this structure; the goal was to attribute each latent coordinate's influence back to specific excitation-emission wavelength pairs.
If perturbing a latent coordinate significantly changed the reconstruction of a particular wavelength, that wavelength must be encoded by that coordinate.""")

E("E1+T1", "methodology.tex",
  r"""\subsubsection{Latent Dimension Scoring}
Not all latent dimensions carry equal information.
The $k_3 = 20$ latent dimensions were first ranked by importance using variance-based scoring:
\begin{equation}
\text{score}(d) = \text{Var}_{\text{samples}}\left(\vect{z}[\cdot, d, \cdot, \cdot, \cdot]\right)
\label{eq:dim_score}
\end{equation}
where $\vect{z}[\cdot, d, \cdot, \cdot, \cdot]$ denotes dimension $d$ across all samples and spatial locations.
Higher variance indicated that the dimension captured meaningful variation rather than constant or near-zero values.
The top $k$ dimensions were then selected for perturbation analysis, where $k$ is treated as a hyperparameter; empirical evaluation (Section~\ref{sec:results}) revealed that $k=1$ (using only the highest-variance dimension) achieved optimal performance.

Alternative scoring methods include activation magnitude (mean absolute value) and reconstruction sensitivity (MSE when dimension is zeroed).
Variance-based scoring yielded the most consistent wavelength rankings, likely because high-variance dimensions captured the primary modes of spectral variation that distinguish different materials.""",
  r"""\subsubsection{Latent Coordinate Scoring}
Encoding an analysis patch $p$ (Section~\ref{sec:implementation}) yields a latent map $\vect{z}_p \in \R^{k_3 \times h \times w}$; a latent \emph{coordinate} $d = (c, y, x)$ denotes one channel at one spatial position, and the attribution operates on these $k_3 h w$ coordinates.
Not all latent coordinates carry equal information.
Coordinates were first ranked by importance using variance-based scoring across the analysis patches:
\begin{equation}
\text{score}(d) = \text{Var}_{p}\left(z_p[d]\right)
\label{eq:dim_score}
\end{equation}
so that coordinates with constant or near-zero values are excluded, while higher variance indicates that the coordinate captures meaningful variation across the data.
Because the decoder is fully convolutional, the response of the reconstruction to a displacement of a single coordinate is, to first order, independent of its position, so a single-coordinate displacement probes the decoding pathway of channel $c$ with a bounded spatial footprint.
The top $k$ coordinates were then selected for perturbation analysis, where $k$ is treated as a hyperparameter with $k \in \{1, 3\}$.
Two scoring strategies were compared: the variance-based ranking of Eq.~\eqref{eq:dim_score}, and a ranking by contribution to a principal component analysis (PCA) of the latent activations, in which each coordinate is scored by the sum of the absolute loadings it receives on the leading components.
Empirical evaluation (Section~\ref{sec:results_params}) showed that PCA-based scoring outperformed variance-based scoring at every band count on the lichen dataset; both $k = 1$ and $k = 3$ appear among the best configurations, with $k = 3$ at the 9--13-band operating points and at the 80-band optimum, where the two settings tie (Table~\ref{tab:main_results}).
PCA-based scoring is therefore the recommended default; variance-based scoring is retained as a parameter-light alternative.
Scoring by activation magnitude (mean absolute value) was also available but was not used for the reported results.""")

E("E2", "methodology.tex",
  r"""\subsubsection{Systematic Perturbation}
For each selected latent dimension $d$, the standard deviation $\sigma_d$ was computed across spatial locations and samples, and then the perturbed representations were generated:
\begin{align}
\vect{z}^{+}_d &= \vect{z} + \epsilon \cdot \sigma_d \cdot \vect{e}_d \label{eq:perturb_pos}\\
\vect{z}^{-}_d &= \vect{z} - \epsilon \cdot \sigma_d \cdot \vect{e}_d \label{eq:perturb_neg}
\end{align}
where $\vect{e}_d$ is a one-hot vector selecting dimension $d$, and $\epsilon$ controls perturbation magnitude.
Values of $\epsilon \in \{15, 30, 45\}$ were used, and results across scales were aggregated to ensure robustness.
The $\sigma$-based scaling ensured perturbations were proportional to the natural variation in each dimension, avoiding both negligibly small and destructively large perturbations.""",
  r"""\subsubsection{Systematic Perturbation}
For each selected latent coordinate $d$, perturbed representations were generated in both directions:
\begin{equation}
\vect{z}^{\pm}_d = \vect{z} \pm \delta_d \, \vect{e}_d
\label{eq:perturb}
\end{equation}
where $\vect{e}_d$ is a one-hot vector selecting coordinate $d$ and the displacement $\delta_d$ is scaled to the empirical distribution of that coordinate across the analysis patches.
Two displacement schemes were used.
In the \emph{percentile} scheme, the coordinate is displaced from its mean to the $(50 \pm m/2)$-th percentile of its distribution, rounded to the nearest of the tabulated percentiles $\{5, 10, 25, 50, 75, 90, 95\}$.
In the \emph{absolute-range} scheme, the displacement is the fraction $m/100$ of the observed range $(\max - \min)$ of the coordinate.
The magnitude parameter $m$ was taken from a medium set $\{30, 40, 50\}$ or a high set $\{50, 60, 70\}$, and the sensitivities obtained at the three magnitudes were accumulated.
Scaling the displacement to the empirical distribution ensured perturbations proportional to the natural variation in each coordinate, avoiding both negligibly small and destructively large perturbations.
All results in Section~\ref{sec:results} were obtained with these two schemes and magnitude sets; the settings behind each reported configuration are listed in Table~\ref{tab:main_results}.""")

E("E3", "methodology.tex",
  r"""\subsubsection{Sensitivity Mapping}
Decoding the perturbed representations yielded reconstructions $\hat{\vect{X}}^{+}_d$ and $\hat{\vect{X}}^{-}_d$.
The sensitivity of each wavelength to perturbation in dimension $d$ was computed:
\begin{equation}
\Delta_d(\lambdaex, \lambdaem) = \frac{\left| \hat{X}^{+}_d(\lambdaex, \lambdaem) - \hat{X}^{-}_d(\lambdaex, \lambdaem) \right|}{2\epsilon \cdot \sigma_d}
\label{eq:sensitivity}
\end{equation}
where averaging was performed over spatial locations and samples.
This finite difference approximation estimates the partial derivative of reconstruction with respect to latent dimension $d$, quantifying how strongly each wavelength's reconstruction responds to changes in that dimension.""",
  r"""\subsubsection{Sensitivity Mapping}
Decoding the perturbed representations yielded reconstructions $\hat{\vect{X}}^{+}_d$ and $\hat{\vect{X}}^{-}_d$.
The sensitivity of each wavelength to the perturbation of coordinate $d$ was computed as the mean absolute change of its reconstruction, averaged over the two directions:
\begin{equation}
\begin{split}
\Delta_d(\lambdaex, \lambdaem) = \tfrac{1}{2}\Big( &\left| \hat{X}^{+}_d(\lambdaex, \lambdaem) - \hat{X}(\lambdaex, \lambdaem) \right| \\
&+ \left| \hat{X}^{-}_d(\lambdaex, \lambdaem) - \hat{X}(\lambdaex, \lambdaem) \right| \Big)
\end{split}
\label{eq:sensitivity}
\end{equation}
where $\hat{X}$ denotes the unperturbed reconstruction and averaging was performed over spatial locations and samples.
This quantifies how strongly each wavelength's reconstruction responds to changes in that coordinate.""")

E("T1d", "methodology.tex",
  r"Sensitivity was accumulated across latent dimensions, weighted by dimension importance:",
  r"Sensitivity was accumulated across the selected latent coordinates, weighted by coordinate score:")

E("E4", "methodology.tex",
  r"where $\overline{\Delta_d}$ denotes the mean sensitivity across perturbation magnitudes.",
  r"where $\overline{\Delta_d}$ denotes the sensitivity summed across the three perturbation magnitudes.")

E("T3a", "methodology.tex",
  r"\text{MMR}(w_i) = \lambda \cdot \text{Rel}(w_i) - (1 - \lambda) \cdot \max_{w_j \in \mathcal{S}} \text{Sim}(w_i, w_j)",
  r"\text{MMR}(w_i) = \text{Rel}(w_i) - \lambda \cdot \max_{w_j \in \mathcal{S}} \left| \text{Sim}(w_i, w_j) \right|")

E("T3b", "methodology.tex",
  r"where $\text{Rel}(w_i)$ is the normalized influence score, $\text{Sim}(w_i, w_j)$ is the cosine similarity between spectral profiles at wavelengths $w_i$ and $w_j$, and $\lambda \in [0, 1]$ controls the relevance-diversity trade-off.",
  r"where $\text{Rel}(w_i)$ is the influence score divided by its maximum over all wavelength pairs, $\text{Sim}(w_i, w_j)$ is the cosine similarity between the spatial intensity images of the two bands (two bands are similar when they render the scene similarly), and $\lambda \ge 0$ controls the relevance-diversity trade-off.")

E("T3c+E21", "methodology.tex",
  r"""At $\lambda = 1$, MMR reduces to pure relevance ranking; at $\lambda = 0$, selection maximizes diversity without regard for informativeness.
Empirical evaluation (Section~\ref{sec:results}) identified $\lambda = 0.5$ as optimal, providing balanced consideration of both criteria.""",
  r"""At $\lambda = 0$, MMR reduces to pure relevance ranking; larger values of $\lambda$ penalize candidates that resemble already-selected bands more strongly.
The value $\lambda = 0.5$ was fixed a priori, giving equal weight to relevance and diversity~\cite{carbonell1998use}; it was not tuned.""")

E("T3d", "methodology.tex",
  r"            \STATE $\text{MMR}(w) \gets \lambda \cdot \text{Rel}(w) - (1-\lambda) \cdot \max_{w_j \in \mathcal{S}} \text{Sim}(w, w_j)$",
  r"            \STATE $\text{MMR}(w) \gets \text{Rel}(w) - \lambda \cdot \max_{w_j \in \mathcal{S}} |\text{Sim}(w, w_j)|$")

# =============================================================================== experimental_setup.tex
E("T6", "experimental_setup.tex",
  r"Spectral resolution & $\sim$12.5\,nm \\",
  r"Spectral resolution & 10\,nm \\")

E("E20a", "experimental_setup.tex",
  r"Table~\ref{tab:hyperparams} lists the autoencoder configuration.",
  r"Table~\ref{tab:hyperparams} lists the autoencoder configuration together with the attribution and selection settings and the rationale for each.")

E("E20b", "experimental_setup.tex",
  r"\caption{Autoencoder Hyperparameters}",
  r"\caption{Autoencoder, Attribution and Selection Settings}")

E("E20c", "experimental_setup.tex",
  r"""Early stopping patience & 5 epochs & Prevent overfitting \\
\bottomrule""",
  r"""Early stopping patience & 5 epochs & Prevent overfitting \\
\midrule
Rayleigh cutoff & 40\,nm (1st, 2nd) & Scatter exclusion \\
Feature merging & Averaging & Size indep.\ of $N_{\text{ex}}$ \\
Coordinate scoring & PCA / variance & Leading latent modes \\
Coordinates ($k$) & 3 / 1 & Tie at the optimum \\
Displacement & Percentile & Bounded; range alt. \\
Magnitudes ($m$) & \{30, 40, 50\} & Measurable, in-regime \\
Directions & $\pm$ & No sign bias \\
Normalization & Max per ex. & Coverage if $K{<}2N_{\text{ex}}$ \\
MMR ($\lambda$) & 0.5 (fixed) & Equal weights~\cite{carbonell1998use} \\
Similarity & Cosine (images) & Redundancy criterion \\
Classifier & KNN-5 & Feature quality only \\
\bottomrule""")

E("E20d", "experimental_setup.tex",
  r"""\label{tab:hyperparams}
\begin{tabular}{@{}lll@{}}""",
  r"""\label{tab:hyperparams}
\footnotesize
\begin{tabular}{@{}lll@{}}""")

E("E5+T1+T2+E21", "experimental_setup.tex",
  r"""For latent space analysis, dimensions were ranked by either variance (as in Section~\ref{sec:perturbation}) or by their contribution to a Principal Component Analysis (PCA) decomposition of the latent space.
Both methods were explored at $k \in \{1, 3\}$ top dimensions. Perturbations were applied using two methods, percentile-based and absolute-range, at medium ($\epsilon \in \{30, 40, 50\}$) and high ($\epsilon \in \{50, 60, 70\}$) magnitude levels.
Sensitivity values were averaged across 1,000 randomly sampled patches to ensure robust estimation.
Additionally, three normalization strategies for influence scores were compared: no normalization, max-per-excitation, and variance normalization (see Section~\ref{sec:results}).""",
  r"""For latent space analysis, latent coordinates were ranked by either variance (as in Section~\ref{sec:perturbation}) or by their contribution to a Principal Component Analysis (PCA) decomposition of the latent activations (sum of absolute loadings on the leading components).
Both methods were explored at $k \in \{1, 3\}$ top coordinates. Perturbations were applied using two schemes, percentile-based and absolute-range, at medium ($m \in \{30, 40, 50\}$) and high ($m \in \{50, 60, 70\}$) magnitude levels.
The scheme and magnitude set behind each reported configuration are given in Table~\ref{tab:main_results}; the best configurations used the percentile scheme with the medium set except where noted there.
Sensitivity values were averaged across 50 analysis patches of $32 \times 32$ pixels, extracted in raster order with a stride of 16 pixels and retained when more than half of their pixels were foreground.
Additionally, three normalization strategies for influence scores were compared: no normalization, max-per-excitation, and variance normalization (Section~\ref{sec:configurations}).
Per-excitation normalization makes influence scores comparable across excitations that differ in absolute reconstruction error, whereas variance normalization discounts bands whose reconstruction changes merely track their intensity variance.""")

E("T7", "experimental_setup.tex",
  r"For evaluation, all annotated pixels were used, with four regions of interest (ROIs) providing class exemplars (Figure~\ref{fig:roi_overlay}); the same classifier configuration was applied consistently across all experiments for fair comparison.",
  r"For evaluation, all annotated pixels were used, with four regions of interest (ROIs) providing class exemplars (Figure~\ref{fig:roi_overlay}); the ROI pixels used to train the classifier (11,519 pixels, 6.0\% of the annotated pixels) are contained in the evaluated set. The same classifier configuration was applied consistently across all experiments for fair comparison.")

E("T1e", "experimental_setup.tex",
  r"""Dimension selection method & Variance, PCA \\
Number of dimensions ($k$) & 1, 3 \\""",
  r"""Coordinate scoring method & Variance, PCA \\
Number of coordinates ($k$) & 1, 3 \\""")

E("T1f", "experimental_setup.tex",
  r"Perturbation magnitudes ($\epsilon$) & Medium \{30,40,50\}, High \{50,60,70\} \\",
  r"Perturbation magnitudes ($m$) & Medium \{30,40,50\}, High \{50,60,70\} \\")

E("T1g", "experimental_setup.tex",
  r"    \item \textbf{Attribution parameters:} Two dimension selection methods $\times$ two dimension counts $\times$ two perturbation methods $\times$ two magnitude levels $= 16$ configurations.",
  r"    \item \textbf{Attribution parameters:} Two coordinate scoring methods $\times$ two coordinate counts $\times$ two perturbation schemes $\times$ two magnitude levels $= 16$ configurations.")

E("E7", "experimental_setup.tex",
  r"Each configuration produced a wavelength ranking through the unsupervised pipeline; classification accuracy was measured post-hoc to assess quality.",
  r"""Each configuration produced a wavelength ranking through the unsupervised pipeline; classification accuracy was measured post-hoc to assess quality.

\subsection{Configuration Selection and Evaluation Protocol}
\label{sec:protocol}
Two uses of the class labels must be distinguished.
The wavelength ranking produced by any configuration is computed without labels (Section~\ref{sec:validation}).
Labels are used, first, to evaluate a ranking through the KNN classifier and, second, to identify the best of the 48 attribution configurations at each band count in the sweep of Table~\ref{tab:configs}.
Results of the second kind are reported as the \emph{best-of-48 envelope} (Table~\ref{tab:main_results}, Figures~\ref{fig:accuracy_vs_bands} and~\ref{fig:collagen_envelope}); the envelope characterizes the configuration space and is an upper bound on the accuracy that configuration tuning can reach, not an estimate for a configuration chosen without labels.
To separate configuration selection from evaluation, a single configuration was fixed on the lichen dataset and then applied unchanged to the collagen dataset: PCA-based scoring with $k = 3$, the percentile scheme with the medium magnitude set, and per-excitation normalization for $K < 2N_{\text{ex}}$ with no normalization otherwise; the $K$-dependent normalization reflects that coverage of the excitation axis is the binding constraint only when $K$ is small relative to $N_{\text{ex}}$.
The accuracy of this fixed configuration is reported alongside the envelope for both datasets; on the collagen dataset it constitutes a held-out test, since no configuration choice was made there.""")

# =============================================================================== results.tex
E("T1h", "results.tex",
  r"This supported using top-$k$ dimension selection for perturbation analysis.",
  r"This supported restricting the perturbation analysis to a small number of top-scoring latent coordinates.")

E("E8a", "results.tex",
  r"""Table~\ref{tab:main_results} presents the best-performing configurations at key band counts, all of which used PCA-based dimension selection (see Section~\ref{sec:results_params} for method comparison).""",
  r"""Table~\ref{tab:main_results} presents the best-of-48 envelope at key band counts (Section~\ref{sec:protocol}), all of which used PCA-based coordinate scoring (see Section~\ref{sec:results_params} for method comparison), together with the accuracy of the fixed configuration at the same band counts.""")

E("E6", "results.tex",
  r"""\begin{table*}[!t]
\centering
\caption{Wavelength Selection Performance Across Band Counts (Best Configuration per Band Count)}
\label{tab:main_results}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Configuration} & \textbf{Bands} & \textbf{Reduction} & \textbf{Accuracy} & \textbf{F1 (wtd)} & \textbf{$\kappa$} & \textbf{Rel.\ Perf.} \\
\midrule
Baseline (all 192 bands) & 192 & 0.0\% & 0.882 & 0.882 & 0.842 & 100.0\% \\
\midrule
\textbf{PCA, 3-dim, 80 bands} & \textbf{80} & \textbf{58.3\%} & \textbf{0.952} & \textbf{0.952} & \textbf{0.936} & \textbf{108.0\%} \\
PCA, 1-dim, 60 bands & 60 & 68.8\% & 0.945 & 0.945 & 0.926 & 107.2\% \\
PCA, 3-dim, 50 bands & 50 & 74.0\% & 0.940 & 0.941 & 0.920 & 106.7\% \\
PCA, 1-dim, 40 bands & 40 & 79.2\% & 0.932 & 0.932 & 0.908 & 105.7\% \\
PCA, 1-dim, 30 bands & 30 & 84.4\% & 0.930 & 0.931 & 0.906 & 105.5\% \\
PCA, 1-dim, 20 bands & 20 & 89.6\% & 0.917 & 0.918 & 0.889 & 104.0\% \\
PCA, 3-dim, 13 bands & 13 & 93.2\% & 0.902 & 0.902 & 0.868 & 102.3\% \\
PCA, 3-dim, 10 bands & 10 & 94.8\% & 0.897 & 0.898 & 0.863 & 101.8\% \\
\textbf{PCA, 3-dim, 9 bands} & \textbf{9} & \textbf{95.3\%} & \textbf{0.894} & \textbf{0.895} & \textbf{0.859} & \textbf{101.5\%} \\
PCA, 1-dim, 5 bands & 5 & 97.4\% & 0.870 & 0.871 & 0.826 & 98.7\% \\
PCA, 1-dim, 3 bands & 3 & 98.4\% & 0.827 & 0.827 & 0.768 & 93.8\% \\
\bottomrule
\vspace{0.2em}
\end{tabular}
\parbox{\textwidth}{\small\textit{Note:} Best configuration at each band count shown; all use PCA-based dimension selection with MMR $\lambda = 0.5$. Bold rows indicate key operating points. Rel.\ Perf.\ = accuracy/baseline accuracy. The ``dim'' attribute indicates the number of latent dimensions used ($k$); remaining parameters (perturbation method, magnitude, normalization) varied per row. Normalization combinations were evaluated at each band count.}
\end{table*}""",
  r"""\begin{table*}[!t]
\centering
\caption{Wavelength Selection Performance Across Band Counts: Best-of-48 Envelope and Fixed Configuration}
\label{tab:main_results}
\footnotesize
\begin{tabular}{@{}lccllccccc@{}}
\toprule
\textbf{Configuration} & \textbf{Bands} & \textbf{Reduction} & \textbf{Scheme / magn.} & \textbf{Norm.} & \textbf{Accuracy} & \textbf{F1 (wtd)} & \textbf{$\kappa$} & \textbf{Rel.\ Perf.} & \textbf{Fixed config.} \\
\midrule
Baseline (all 192 bands) & 192 & 0.0\% & -- & -- & 0.882 & 0.882 & 0.842 & 100.0\% & 0.882 \\
\midrule
\textbf{PCA, 3-dim, 80 bands} & \textbf{80} & \textbf{58.3\%} & perc.\ / med. & none & \textbf{0.952} & \textbf{0.952} & \textbf{0.936} & \textbf{108.0\%} & \textbf{0.952} \\
PCA, 1-dim, 60 bands & 60 & 68.8\% & perc.\ / med. & none & 0.945 & 0.945 & 0.926 & 107.2\% & 0.944 \\
PCA, 3-dim, 50 bands & 50 & 74.0\% & perc.\ / high & none & 0.940 & 0.941 & 0.920 & 106.7\% & 0.940 \\
PCA, 1-dim, 40 bands & 40 & 79.2\% & perc.\ / med. & none & 0.932 & 0.932 & 0.908 & 105.7\% & 0.931 \\
PCA, 1-dim, 30 bands & 30 & 84.4\% & abs.\ / med. & none & 0.930 & 0.931 & 0.906 & 105.5\% & 0.923 \\
PCA, 1-dim, 20 bands & 20 & 89.6\% & perc.\ / med. & none & 0.917 & 0.918 & 0.889 & 104.0\% & 0.906 \\
PCA, 3-dim, 13 bands & 13 & 93.2\% & perc.\ / med. & max/ex. & 0.902 & 0.902 & 0.868 & 102.3\% & 0.902 \\
PCA, 3-dim, 10 bands & 10 & 94.8\% & perc.\ / med. & max/ex. & 0.897 & 0.898 & 0.863 & 101.8\% & 0.897 \\
\textbf{PCA, 3-dim, 9 bands} & \textbf{9} & \textbf{95.3\%} & abs.\ / med. & max/ex. & \textbf{0.894} & \textbf{0.895} & \textbf{0.859} & \textbf{101.5\%} & \textbf{0.874} \\
PCA, 1-dim, 5 bands & 5 & 97.4\% & perc.\ / med. & max/ex. & 0.870 & 0.871 & 0.826 & 98.7\% & 0.859 \\
PCA, 1-dim, 3 bands & 3 & 98.4\% & perc.\ / med. & max/ex. & 0.827 & 0.827 & 0.768 & 93.8\% & 0.827 \\
\bottomrule
\vspace{0.2em}
\end{tabular}
\parbox{\textwidth}{\small\textit{Note:} Envelope = best of the 48 attribution configurations at each band count (Section~\ref{sec:protocol}); all use PCA-based coordinate scoring with MMR $\lambda = 0.5$. Bold rows indicate key operating points. The ``dim'' attribute is the number of latent coordinates $k$ (at 80 bands, $k = 1$ and $k = 3$ tie); ``Scheme / magn.'' gives the displacement scheme (percentile or absolute range) and magnitude set (medium or high), and ``Norm.'' the influence normalization (none or max-per-excitation) of that configuration. Rel.\ Perf.\ = envelope accuracy / baseline accuracy. Fixed config.\ = accuracy of the single configuration of Section~\ref{sec:protocol} at the same band count.}
\end{table*}""")

E("E8b", "results.tex",
  r"Even with only 9 bands, accuracy remained at 89.4\%, still exceeding the baseline.",
  r"""Even with only 9 bands, accuracy remained at 89.4\%, still exceeding the baseline.
Under the fixed configuration of Section~\ref{sec:protocol}, which involves no label-based choice, accuracy exceeds the baseline from 10 bands upward (89.7\%), reaches 90.2\% at 13 bands and coincides with the envelope at the 80-band optimum (Table~\ref{tab:main_results}).""")

E("E8c", "results.tex",
  r"\caption{Classification accuracy envelope as a function of band count.}",
  r"\caption{Best-of-48 classification accuracy envelope as a function of band count (Section~\ref{sec:protocol}).}")

E("T5a", "results.tex",
  r"    \item \textbf{Object 8} (Class 2): Improved from 72.8\% to 68.9\%---the only object showing slight decline, reflecting the persistent Class 2/3 spectral overlap.",
  r"    \item \textbf{Object 8} (Class 2): Declined from 72.8\% to 68.9\%, the largest of four declines (Objects 4, 8, 14 and 16; the other three are below 1.3 percentage points), reflecting the persistent Class 2/3 spectral overlap.")

E("T5b", "results.tex",
  r"Wavelength selection removed these harmful bands, with 15 of 16 specimens showing improvement, and performance standard deviation decreasing from 0.172 to 0.079.",
  r"Wavelength selection removed these harmful bands, with 12 of 16 specimens showing improvement, and performance standard deviation decreasing from 0.172 to 0.079.")

E("T5c", "results.tex",
  r"Maximum & 0.998 (Obj.\ 4) & 1.000 (Obj.\ 1) \\",
  r"Maximum & 0.998 (Obj.\ 4) & 1.000 (Obj.\ 2) \\")

E("T5d", "results.tex",
  r"""Objects Improved & -- & 15/16 (94\%) \\
Objects Maintained & -- & 0/16 (0\%) \\
Objects Declined & -- & 1/16 (6\%) \\""",
  r"""Objects Improved & -- & 12/16 (75\%) \\
Objects Maintained & -- & 0/16 (0\%) \\
Objects Declined & -- & 4/16 (25\%) \\""")

E("E1b+T4", "results.tex",
  r"""PCA-based latent dimension selection outperformed variance-based selection (95.2\% vs.\ 87.0\% best accuracy).
This finding demonstrated that PCA decomposition of the learned latent space more effectively identified the dimensions encoding discriminative information.""",
  r"""PCA-based latent coordinate scoring outperformed variance-based scoring (95.2\% vs.\ 87.0\% best accuracy).
This finding demonstrated that PCA decomposition of the learned latent space more effectively identified the coordinates encoding discriminative information.
Both $k = 1$ and $k = 3$ appear among the best configurations and tie at the 80-band optimum; the two settings gave identical accuracy in 63\% of paired runs.""")

E("T4", "results.tex",
  r"The variance-based method did not exceed the 88.2\% baseline at any band count, while every PCA configuration from 9 bands upward surpassed it.",
  r"The variance-based method did not exceed the 88.2\% baseline at any band count, while the best PCA configuration surpassed it at every band count from 9 upward (761 of the 1,344 PCA runs exceeded the baseline).")

E("E11", "results.tex",
  r"""The 9-band optimal configuration exemplifies the framework's selection strategy: it selects exactly one wavelength from each of the 8 excitation sources (310--430\,nm), with emission peaks spanning 500--680\,nm.
This one-per-excitation pattern suggests that the framework identifies the single most informative emission band at each excitation, consistent with the parallel-branch architecture that processes each excitation independently before feature averaging.""",
  r"""The 9-band configuration of Table~\ref{tab:main_results} comprises the following excitation/emission pairs (nm), in selection order: 310/520, 385/500, 325/570, 430/610, 400/510, 340/500, 365/520, 415/615 and 430/680.
One band is drawn from each of the eight excitation sources and a second band (680\,nm) from the 430\,nm source; the selected emission bands lie at 500--570\,nm for the 310--400\,nm excitations and at 610--680\,nm for the 415--430\,nm excitations.
The one-band-per-excitation coverage follows from the per-excitation normalization used in this configuration, which ranks the strongest band of every excitation first; the framework then determines which emission band each excitation contributes and where the bands beyond the first $N_{\text{ex}}$ are placed.""")

E("E10a", "results.tex",
  r"\caption{Mean wavelength importance score for the lichen dataset, across the 16 above-baseline configuration groups.",
  r"\caption{Mean wavelength importance score for the lichen dataset, across the 16 configuration groups whose envelope accuracy exceeded the baseline (a descriptive grouping that uses the evaluation labels and plays no role in selection).")

E("E9a", "results.tex",
  r"No hyperparameters, attribution settings, or selection parameters were retuned; only the input data and the number of excitation branches changed, the latter determined automatically by the dataset.",
  r"No autoencoder hyperparameters or selection parameters were retuned; the number of excitation branches was determined automatically by the dataset. The attribution sweep of Table~\ref{tab:configs} was repeated on this dataset to obtain its envelope, and the fixed configuration of Section~\ref{sec:protocol} was applied unchanged.")

E("E9b", "results.tex",
  r"Peak accuracy reached 85.6\% at 30 bands, 5.8 percentage points higher than the 79.8\% full-spectrum baseline (i.e., 85.6\% versus 79.8\% accuracy), while retaining only 30 of the 158 bands (81\% data reduction).",
  r"""Peak accuracy reached 85.6\% at 30 bands, 5.8 percentage points higher than the 79.8\% full-spectrum baseline (i.e., 85.6\% versus 79.8\% accuracy), while retaining only 30 of the 158 bands (81\% data reduction).
Under the fixed configuration selected on the lichen dataset (Section~\ref{sec:protocol}), accuracy was 83.6\% at 30 bands and 85.1\% at 50 bands, confirming the gain over the 79.8\% baseline without any configuration choice on this dataset.""")

E("E10b", "results.tex",
  r"\caption{Mean wavelength importance score for the collagen sponge dataset across the above-baseline configuration groups, in the same format as",
  r"\caption{Mean wavelength importance score for the collagen sponge dataset across the configuration groups whose envelope accuracy exceeded the baseline (a descriptive grouping that uses the evaluation labels and plays no role in selection), in the same format as")

E("E8d", "results.tex",
  r"\caption{Classification accuracy envelope as a function of band count for the collagen sponge dataset. The best selection exceeds the full-spectrum baseline (79.8\%) across a wide range and peaks at 85.6\% with 30 of 158 bands.",
  r"\caption{Best-of-48 classification accuracy envelope as a function of band count for the collagen sponge dataset (Section~\ref{sec:protocol}). The best selection exceeds the full-spectrum baseline (79.8\%) across a wide range and peaks at 85.6\% with 30 of 158 bands.")

# =============================================================================== discussion.tex
E("T1i", "discussion.tex",
  r"\subsection{PCA vs.\ Variance Dimension Selection}",
  r"\subsection{PCA vs.\ Variance Coordinate Scoring}")

E("T1j", "discussion.tex",
  r"""PCA identifies orthogonal directions in the autoencoder's learned representation, capturing structured variation through a linear combination of dimensions.
Variance-based selection ranks individual dimensions independently, missing these interactions, hence its lower performance ceiling and need for more bands to reach it.""",
  r"""PCA identifies orthogonal directions in the autoencoder's learned representation, capturing structured variation through a linear combination of coordinates.
Variance-based scoring ranks individual coordinates independently, missing these interactions, hence its lower performance ceiling and need for more bands to reach it.""")

E("C1", "discussion.tex",
  r"The selection with MMR parameter of $\lambda = 1$ underperformed the balanced setting ($\lambda = 0.5$) used throughout the experiments.",
  r"The MMR parameter was fixed at $\lambda = 0.5$ throughout the experiments; the role of the diversity term is explained by the redundancy structure of fluorescence spectra.")

E("E12", "discussion.tex",
  r"The one-per-excitation pattern observed in the 9-band configuration (Section~\ref{sec:results_wavelengths}) directly informs sensor design: coverage across the full excitation range is essential, but only a single emission band per excitation can be needed for effective discrimination.",
  r"The 9-band configuration (Section~\ref{sec:results_wavelengths}) directly informs sensor design: all eight excitation sources are retained, and one emission band per source (two for the 430\,nm source) suffices for above-baseline discrimination, so a compact instrument would require eight excitation sources and nine detection channels.")

E("E18", "discussion.tex",
  r"For 3D HSI band selection, deep learning approaches include attention-based networks~\cite{mou2017deep} and reinforcement learning approaches~\cite{roy2020hybridsn}.",
  r"For 3D HSI band selection, deep learning approaches include attention-based networks~\cite{cai2020bsnets,lorenzo2020attention} and reinforcement learning approaches~\cite{mou2022reinforcement}.")

# =============================================================================== conclusion.tex
E("T1k", "conclusion.tex",
  r"perturbation-based attribution for tracing latent dimension importance to individual excitation-emission wavelength pairs",
  r"perturbation-based attribution for tracing latent coordinate importance to individual excitation-emission wavelength pairs")

E("E13", "conclusion.tex",
  r"The 9-band optimal configuration selected one wavelength per excitation source, providing directly actionable sensor design guidelines.",
  r"The 9-band configuration retained all eight excitation sources with one emission band per source (two for one source), providing directly actionable sensor design guidelines.")


# =============================================================================== apply
def main():
    applied = []
    for eid, fname, old, new in EDITS:
        path = SEC / fname
        text = path.read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1:
            print(f"ABORT: edit {eid} in {fname}: expected exactly 1 match, found {n}")
            sys.exit(1)
        path.write_text(text.replace(old, new), encoding="utf-8")
        applied.append((eid, fname))
    # bibliography: append the 26 verified entries
    bib = REV / "references.bib"
    add = (HERE.parent / "minimal_additions.bib").read_text(encoding="utf-8")
    btxt = bib.read_text(encoding="utf-8")
    if "hong2026hyperspectral" not in btxt:
        bib.write_text(btxt.rstrip("\n") + "\n\n" + add, encoding="utf-8")
        applied.append(("BIB", "references.bib"))
    print(f"applied {len(applied)} edits:")
    for eid, fname in applied:
        print(f"  {eid:12s} {fname}")


if __name__ == "__main__":
    main()
