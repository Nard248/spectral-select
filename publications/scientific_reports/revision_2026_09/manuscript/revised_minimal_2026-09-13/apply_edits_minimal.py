"""MINIMAL revision: apply only the edits required by the eleven reviewer points to manuscript/revised/.

Each edit is an exact-string replacement that must match exactly once (the script aborts otherwise).
Numbers added to the text are read from the existing sweeps (results/Lichens_Dataset_1_MasterRun/results.csv,
results/Collagen_Pepsin_Normalized/results.csv); nothing was re-run. Edit IDs: M1-M27.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE / "revised"
SEC = REV / "sections"
EDITS: list[tuple[str, str, str, str]] = []


def E(eid, fname, old, new):
    EDITS.append((eid, fname, old, new))


# =============================================================================== R2.1 / R2.2 / R2.3 — introduction.tex
E("M1 (R2.1, R2.2)", "introduction.tex",
  r"exceeding the discriminative capabilities of RGB or multispectral data~\cite{plaza2009recent,bioucas2013hyperspectral}.",
  r"exceeding the discriminative capabilities of RGB or multispectral data~\cite{hong2026hyperspectral,hong2024spectralgpt}." + "\n"
  r"Its applications now span Earth observation, where it supports the detection of photovoltaic installations from satellite and airborne sensors~\cite{joerges2023photovoltaic} and the unsupervised unmixing of oceanic scenes affected by spectral variability and adjacency effects~\cite{deville2023oceanic}; agriculture and environmental monitoring~\cite{lu2020agriculture,lekha2026environmental}; food quality control~\cite{sun2024foodindustry}; and clinical imaging, where reflectance and autofluorescence hyperspectral data support diagnosis and image-guided surgery~\cite{yoon2022clinical,campbell2023beyondoncology}.")

E("M2 (R2.3)", "introduction.tex",
  r"The challenge of identifying informative spectral bands has been extensively studied for conventional 3D HSI~\cite{sun2019hyperspectral,jia2020survey,asfour2018,guan2018}, with approaches spanning filter-based methods (variance, entropy), wrapper methods (sequential forward selection, genetic algorithms), and embedded methods (sparse regression, attention mechanisms)~\cite{martinez2006comparison,chang2000constrained,pudil1994floating,pal2010feature,tibshirani1996regression,mou2017deep}.",
  r"The challenge of identifying informative spectral bands has been extensively studied for conventional 3D HSI~\cite{sun2019hyperspectral,sawant2021survey,rasti2020featureextraction,ahmad2022survey,jia2020survey,asfour2018,guan2018}, with approaches spanning filter-based methods (variance, entropy), wrapper methods (sequential forward selection, genetic algorithms), and embedded methods (sparse regression, attention mechanisms)~\cite{martinez2006comparison,chang2000constrained,pudil1994floating,pal2010feature,tibshirani1996regression,lorenzo2020attention}.")

# =============================================================================== R2.4 / R2.5 / R2.6 — related_work.tex
E("M3 (R2.6)", "related_work.tex",
  r"This section reviews prior work in four areas relevant to the present framework: hyperspectral imaging systems and data characteristics, band selection methods for hyperspectral data, deep learning approaches for spectral analysis, and explainability methods for feature attribution.",
  r"This section proceeds from the imaging modality and its data characteristics (Section~\ref{sec:related_hsi}), to band selection methods for conventional hyperspectral cubes (Section~\ref{sec:related_band}), to the deep learning and attribution tools on which the framework builds (Sections~\ref{sec:related_dl} and~\ref{sec:related_explain}), and closes by positioning the framework against them (Section~\ref{sec:related_summary}).")

E("M4 (R2.6)", "related_work.tex",
  r"Such systems have proven particularly valuable for biological specimens, including lichens and plant tissues, where autofluorescence signatures encode information about molecular composition that reflectance imaging cannot capture~\cite{croce2019autofluorescence}.",
  r"Such systems have proven particularly valuable for biological specimens, including lichens and plant tissues, where autofluorescence signatures encode information about molecular composition that reflectance imaging cannot capture~\cite{croce2019autofluorescence}." + "\n"
  r"Recent reviews of multi- and hyperspectral autofluorescence imaging document a growing clinical and biological application base~\cite{campbell2023beyondoncology}, and multi-excitation autofluorescence hyperspectral imaging has been demonstrated for the exploration of biological samples~\cite{ghaffari2019multiexcitation}.")

E("M5 (R2.4, R2.5)", "related_work.tex",
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
While elegant, these approaches typically require labeled training data, limiting their applicability when ground truth is scarce or expensive to obtain.""",
  r"""Band selection methods aim to identify a subset of spectral bands that retain discriminative information while reducing dimensionality~\cite{sun2019hyperspectral,sawant2021survey}.
These methods are typically categorized as filter, wrapper, or embedded approaches; the specific contribution of each family, and its limitation for ME-HSI data, are as follows.
\begin{itemize}
\item \textbf{Filter methods} rank bands based on statistical properties computed without reference to any downstream task: variance-based selection prioritizes bands with high variance across samples~\cite{chang2000constrained}, entropy-based methods favor bands with maximum information content~\cite{guo2006band}, and correlation-based approaches remove redundant bands that are highly correlated with others~\cite{martinez2006comparison}.
Contribution: negligible computational cost and no need for labels.
Limitation: no notion of discriminative value---a band with high variance may capture noise rather than meaningful variation---and bands are treated independently, so excitation-emission combinations are never assessed jointly.
\item \textbf{Wrapper methods} evaluate band subsets based on the performance of a classifier trained on the selected features: sequential forward selection (SFS) greedily adds bands that most improve classification accuracy~\cite{pudil1994floating}, while genetic algorithms and particle swarm optimization explore the combinatorial search space more globally~\cite{pal2010feature}.
Contribution: task-relevant band subsets.
Limitation: labels are required and the cost grows combinatorially with the number of bands; for a 192-band ME-HSI dataset, exhaustive evaluation of all possible subsets is intractable.
\item \textbf{Embedded supervised methods} integrate band selection into model training: sparse regression (LASSO) induces feature selection through $L_1$ regularization~\cite{tibshirani1996regression}, attention layers trained for classification learn to weight input bands~\cite{lorenzo2020attention}, and reinforcement learning formulations treat band selection as a sequential decision process~\cite{mou2022reinforcement}.
Contribution: selection and prediction are optimized jointly.
Limitation: labeled training data are required, limiting applicability when ground truth is scarce or expensive to obtain.
\item \textbf{Unsupervised deep methods} learn band weights jointly with a reconstruction network and read the selection off the learned weights---BS-Nets~\cite{cai2020bsnets}, attention-based autoencoders~\cite{dou2021attentionae}, DARecNet-BS~\cite{roy2021darecnet} and TAttMSRecNet~\cite{nandi2023tattmsrecnet}---or make subset selection differentiable through a concrete selector layer~\cite{sun2022concrete}.
Contribution: label-free, end-to-end band selection.
Limitation: a single attention or selector layer over a single spectral axis presumes one rectangular cube, and the selection is fixed at training time, so a different number of bands requires retraining.
\end{itemize}""")

E("M6 (R2.6)", "related_work.tex",
  r"Deep learning has substantially advanced hyperspectral image analysis, with convolutional neural networks achieving state-of-the-art results in classification, segmentation, and anomaly detection~\cite{li2019deep,paoletti2019deep}.",
  r"Deep learning has substantially advanced hyperspectral image analysis, with convolutional neural networks achieving state-of-the-art results in classification, segmentation, and anomaly detection~\cite{li2019deep,paoletti2019deep}; recent surveys trace the evolution from shallow to deep feature extraction~\cite{rasti2020featureextraction,ahmad2022survey}.")

E("M7 (R2.6)", "related_work.tex",
  r"This approach has been applied to image classification by occluding image regions~\cite{zeiler2014visualizing}, to text analysis by removing words~\cite{li2016understanding}, and to tabular data by shuffling feature values~\cite{fisher2019all}.",
  r"This approach has been applied to image classification by occluding image regions~\cite{zeiler2014visualizing}, to text analysis by removing words~\cite{li2016understanding}, and to tabular data by shuffling feature values~\cite{fisher2019all}." + "\n"
  r"Meaningful-perturbation methods~\cite{fong2017perturbation} and the removal-based view that unifies occlusion, permutation and Shapley-type attributions~\cite{covert2021removing} provide the general framework; a survey of perturbation-based explanation methods is given in~\cite{ivanovs2021perturbation}.")

# =============================================================================== R1.1 — methodology.tex
E("M8 (R1.1)", "methodology.tex",
  r"The top $k$ dimensions were then selected for perturbation analysis, where $k$ is treated as a hyperparameter; empirical evaluation (Section~\ref{sec:results}) revealed that $k=1$ (using only the highest-variance dimension) achieved optimal performance.",
  r"""The top $k$ dimensions were then selected for perturbation analysis, where $k$ is treated as a hyperparameter with $k \in \{1, 3\}$.
As an alternative to variance-based scoring, dimensions were also ranked by their contribution to a principal component analysis (PCA) of the latent activations (Section~\ref{sec:implementation}); empirical evaluation (Section~\ref{sec:results_params}) showed that PCA-based scoring outperformed variance-based scoring at every band count, with $k = 3$ at the 80-band optimum and at the 9--13-band operating points and $k = 1$ at the remaining band counts of Table~\ref{tab:main_results}.""")

E("M9 (R1.1)", "methodology.tex",
  r"""Alternative scoring methods include activation magnitude (mean absolute value) and reconstruction sensitivity (MSE when dimension is zeroed).
Variance-based scoring yielded the most consistent wavelength rankings, likely because high-variance dimensions captured the primary modes of spectral variation that distinguish different materials.""",
  r"""Alternative scoring methods include activation magnitude (mean absolute value) and reconstruction sensitivity (MSE when dimension is zeroed).""")

# =============================================================================== R1.2 — methodology.tex
E("M10 (R1.2)", "methodology.tex",
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
For each selected latent dimension $d$, perturbed representations were generated in both directions:
\begin{equation}
\vect{z}^{\pm}_d = \vect{z} \pm \delta_d \, \vect{e}_d
\label{eq:perturb}
\end{equation}
where $\vect{e}_d$ is a one-hot vector selecting dimension $d$ and the displacement $\delta_d$ is scaled to the empirical distribution of that dimension across the analysis samples by one of two schemes: a \emph{percentile} scheme, which moves the dimension from its mean to the $(50 \pm m/2)$-th percentile of its distribution (rounded to the nearest tabulated percentile), and an \emph{absolute-range} scheme, which displaces it by the fraction $m/100$ of its observed range.
The magnitude $m$ was taken from a medium set $\{30, 40, 50\}$ or a high set $\{50, 60, 70\}$ (Section~\ref{sec:implementation}), and the sensitivities obtained at the three magnitudes were aggregated.
Scaling the displacement to the empirical distribution ensured perturbations proportional to the natural variation in each dimension, avoiding both negligibly small and destructively large perturbations.""")

E("M11 (R1.2)", "methodology.tex",
  r"""The sensitivity of each wavelength to perturbation in dimension $d$ was computed:
\begin{equation}
\Delta_d(\lambdaex, \lambdaem) = \frac{\left| \hat{X}^{+}_d(\lambdaex, \lambdaem) - \hat{X}^{-}_d(\lambdaex, \lambdaem) \right|}{2\epsilon \cdot \sigma_d}
\label{eq:sensitivity}
\end{equation}
where averaging was performed over spatial locations and samples.
This finite difference approximation estimates the partial derivative of reconstruction with respect to latent dimension $d$, quantifying how strongly each wavelength's reconstruction responds to changes in that dimension.""",
  r"""The sensitivity of each wavelength to perturbation in dimension $d$ was computed as the mean absolute change of its reconstruction, averaged over the two directions:
\begin{equation}
\begin{split}
\Delta_d(\lambdaex, \lambdaem) = \tfrac{1}{2}\Big( &\left| \hat{X}^{+}_d(\lambdaex, \lambdaem) - \hat{X}(\lambdaex, \lambdaem) \right| \\
&+ \left| \hat{X}^{-}_d(\lambdaex, \lambdaem) - \hat{X}(\lambdaex, \lambdaem) \right| \Big)
\end{split}
\label{eq:sensitivity}
\end{equation}
where $\hat{X}$ denotes the unperturbed reconstruction and averaging was performed over spatial locations and samples.
This quantifies how strongly each wavelength's reconstruction responds to changes in that dimension.""")

# =============================================================================== R2.7 — methodology.tex (lambda)
E("M12 (R2.7)", "methodology.tex",
  r"Empirical evaluation (Section~\ref{sec:results}) identified $\lambda = 0.5$ as optimal, providing balanced consideration of both criteria.",
  r"The value $\lambda = 0.5$ was fixed a priori, giving equal weight to relevance and diversity~\cite{carbonell1998use}; it was not tuned.")

# =============================================================================== R2.7 / R1.2 — experimental_setup.tex
E("M13 (R2.7)", "experimental_setup.tex",
  r"Table~\ref{tab:hyperparams} lists the autoencoder configuration.",
  r"Table~\ref{tab:hyperparams} lists the autoencoder configuration together with the attribution and selection settings and the rationale for each.")

E("M14 (R2.7)", "experimental_setup.tex",
  r"\caption{Autoencoder Hyperparameters}",
  r"\caption{Autoencoder, Attribution and Selection Settings}")

E("M15 (R2.7)", "experimental_setup.tex",
  r"""\label{tab:hyperparams}
\begin{tabular}{@{}lll@{}}""",
  r"""\label{tab:hyperparams}
\footnotesize
\begin{tabular}{@{}lll@{}}""")

E("M16 (R2.7)", "experimental_setup.tex",
  r"""Early stopping patience & 5 epochs & Prevent overfitting \\
\bottomrule""",
  r"""Early stopping patience & 5 epochs & Prevent overfitting \\
\midrule
Rayleigh cutoff & 40\,nm (1st, 2nd) & Scatter exclusion \\
Feature merging & Averaging & Size indep.\ of $N_{\text{ex}}$ \\
Dimension scoring & PCA / variance & Leading latent modes \\
Perturbed dims.\ ($k$) & 3 / 1 & Best in Table~\ref{tab:main_results} \\
Displacement & Percentile & Bounded; range alt. \\
Magnitudes ($m$) & \{30, 40, 50\} & Measurable, in-regime \\
Directions & $\pm$ & No sign bias \\
Normalization & Max per ex. & Coverage if $K{<}2N_{\text{ex}}$ \\
MMR ($\lambda$) & 0.5 (fixed) & Equal weights~\cite{carbonell1998use} \\
Similarity & Cosine & Redundancy criterion \\
Classifier & KNN-5 & Feature quality only \\
\bottomrule""")

E("M17 (R1.2, R2.7)", "experimental_setup.tex",
  r"""Both methods were explored at $k \in \{1, 3\}$ top dimensions. Perturbations were applied using two methods, percentile-based and absolute-range, at medium ($\epsilon \in \{30, 40, 50\}$) and high ($\epsilon \in \{50, 60, 70\}$) magnitude levels.
Sensitivity values were averaged across 1,000 randomly sampled patches to ensure robust estimation.
Additionally, three normalization strategies for influence scores were compared: no normalization, max-per-excitation, and variance normalization (see Section~\ref{sec:results}).""",
  r"""Both methods were explored at $k \in \{1, 3\}$ top dimensions. Perturbations were applied using two schemes, percentile-based and absolute-range (Section~\ref{sec:perturbation}), at medium ($m \in \{30, 40, 50\}$) and high ($m \in \{50, 60, 70\}$) magnitude levels; the scheme and magnitude set behind each reported configuration are stated in the note to Table~\ref{tab:main_results}.
Sensitivity values were averaged across 1,000 randomly sampled patches to ensure robust estimation.
Additionally, three normalization strategies for influence scores were compared: no normalization, max-per-excitation, and variance normalization (Section~\ref{sec:configurations}).
Per-excitation normalization makes influence scores comparable across excitations that differ in absolute reconstruction error, whereas variance normalization discounts bands whose reconstruction changes merely track their intensity variance.""")

E("M18 (R1.2)", "experimental_setup.tex",
  r"Perturbation magnitudes ($\epsilon$) & Medium \{30,40,50\}, High \{50,60,70\} \\",
  r"Perturbation magnitudes ($m$) & Medium \{30,40,50\}, High \{50,60,70\} \\")

# =============================================================================== R1.3 — experimental_setup.tex (protocol paragraph, inside IV-C)
E("M19 (R1.3)", "experimental_setup.tex",
  r"Each configuration produced a wavelength ranking through the unsupervised pipeline; classification accuracy was measured post-hoc to assess quality.",
  r"""Each configuration produced a wavelength ranking through the unsupervised pipeline; classification accuracy was measured post-hoc to assess quality.
Two uses of the class labels are therefore to be distinguished: labels evaluate a ranking, and, in the sweep, they also identify the best of the 48 attribution configurations at each band count.
The best-configuration results of Table~\ref{tab:main_results} and Figures~\ref{fig:accuracy_vs_bands} and~\ref{fig:collagen_envelope} are thus an envelope over the configuration space---an upper bound on what configuration tuning can reach---rather than an estimate for a configuration chosen without labels.
To separate configuration selection from evaluation, a single configuration was additionally fixed on the lichen dataset and applied unchanged to the collagen dataset: PCA-based scoring with $k = 3$, the percentile scheme with the medium magnitude set, and per-excitation normalization for $K < 2N_{\text{ex}}$ with no normalization otherwise, coverage of the excitation axis being the binding constraint only when $K$ is small relative to $N_{\text{ex}}$.
Its accuracy is reported alongside the envelope in Sections~\ref{sec:results_selection} and~\ref{sec:results_collagen}; on the collagen dataset it constitutes a held-out test, since no configuration choice was made there.""")

# =============================================================================== R1.2 / R1.3 — results.tex (Table V note)
E("M20 (R1.2, R1.3)", "results.tex",
  r"\parbox{\textwidth}{\small\textit{Note:} Best configuration at each band count shown; all use PCA-based dimension selection with MMR $\lambda = 0.5$. Bold rows indicate key operating points. Rel.\ Perf.\ = accuracy/baseline accuracy. The ``dim'' attribute indicates the number of latent dimensions used ($k$); remaining parameters (perturbation method, magnitude, normalization) varied per row. Normalization combinations were evaluated at each band count.}",
  r"\parbox{\textwidth}{\small\textit{Note:} Best configuration at each band count shown (an envelope over the 48 attribution configurations, Section~\ref{sec:configurations}); all use PCA-based dimension selection with MMR $\lambda = 0.5$. Bold rows indicate key operating points. Rel.\ Perf.\ = accuracy/baseline accuracy. The ``dim'' attribute indicates the number of latent dimensions used ($k$). The remaining attribution settings of the listed configurations are the percentile scheme with the medium magnitude set for all rows except the 30- and 9-band rows (absolute range, medium) and the 50-band row (percentile, high), with no influence normalization for 20 bands and above and max-per-excitation normalization for 13 bands and below.}")

# =============================================================================== R1.3 — results.tex (fixed configuration, lichens)
E("M21 (R1.3)", "results.tex",
  r"Even with only 9 bands, accuracy remained at 89.4\%, still exceeding the baseline.",
  r"""Even with only 9 bands, accuracy remained at 89.4\%, still exceeding the baseline.
Under the configuration fixed in Section~\ref{sec:configurations}, which involves no label-based choice, accuracy exceeds the baseline from 10 bands upward (89.7\%) and coincides with the best configuration at 10, 13 and 80 bands (89.7\%, 90.2\% and 95.2\%).""")

# =============================================================================== R1.4 — results.tex
E("M22 (R1.4)", "results.tex",
  r"""The 9-band optimal configuration exemplifies the framework's selection strategy: it selects exactly one wavelength from each of the 8 excitation sources (310--430\,nm), with emission peaks spanning 500--680\,nm.
This one-per-excitation pattern suggests that the framework identifies the single most informative emission band at each excitation, consistent with the parallel-branch architecture that processes each excitation independently before feature averaging.""",
  r"""The 9-band configuration of Table~\ref{tab:main_results} comprises the following excitation/emission pairs (nm), in selection order: 310/520, 385/500, 325/570, 430/610, 400/510, 340/500, 365/520, 415/615 and 430/680, i.e., one band from each of the eight excitation sources (310--430\,nm) and a second band from the 430\,nm source, with emission bands spanning 500--680\,nm.
This near one-per-excitation pattern reflects the per-excitation normalization of this configuration, which ranks the strongest band of every excitation first; the framework then determines which emission band each excitation contributes.""")

# =============================================================================== R1.3 — results.tex (collagen)
E("M23 (R1.3)", "results.tex",
  r"No hyperparameters, attribution settings, or selection parameters were retuned; only the input data and the number of excitation branches changed, the latter determined automatically by the dataset.",
  r"No autoencoder hyperparameters or selection parameters were retuned; the number of excitation branches was determined automatically by the dataset, and the attribution configuration was chosen per band count as for the lichen dataset (Section~\ref{sec:configurations}).")

E("M24 (R1.3)", "results.tex",
  r"Peak accuracy reached 85.6\% at 30 bands, 5.8 percentage points higher than the 79.8\% full-spectrum baseline (i.e., 85.6\% versus 79.8\% accuracy), while retaining only 30 of the 158 bands (81\% data reduction).",
  r"""Peak accuracy reached 85.6\% at 30 bands, 5.8 percentage points higher than the 79.8\% full-spectrum baseline (i.e., 85.6\% versus 79.8\% accuracy), while retaining only 30 of the 158 bands (81\% data reduction).
Under the configuration fixed on the lichen dataset (Section~\ref{sec:configurations}), accuracy was 83.6\% at 30 bands and 85.1\% at 50 bands, confirming the gain over the 79.8\% baseline without any configuration choice on this dataset.""")

# =============================================================================== R2.7 / R1.4 / R2.5 — discussion.tex
E("M25 (R2.7)", "discussion.tex",
  r"The selection with MMR parameter of $\lambda = 1$ underperformed the balanced setting ($\lambda = 0.5$) used throughout the experiments.",
  r"The MMR parameter was fixed at $\lambda = 0.5$ throughout the experiments; the value of the diversity term is explained by the redundancy structure of fluorescence spectra.")

E("M26 (R1.4)", "discussion.tex",
  r"The one-per-excitation pattern observed in the 9-band configuration (Section~\ref{sec:results_wavelengths}) directly informs sensor design: coverage across the full excitation range is essential, but only a single emission band per excitation can be needed for effective discrimination.",
  r"The 9-band configuration (Section~\ref{sec:results_wavelengths}) directly informs sensor design: all eight excitation sources are retained, and one emission band per source (two for the 430\,nm source) suffices for above-baseline discrimination, so a compact instrument would require eight excitation sources and nine detection channels.")

E("M27 (R2.5)", "discussion.tex",
  r"For 3D HSI band selection, deep learning approaches include attention-based networks~\cite{mou2017deep} and reinforcement learning approaches~\cite{roy2020hybridsn}.",
  r"For 3D HSI band selection, deep learning approaches include attention-based networks~\cite{cai2020bsnets,lorenzo2020attention} and reinforcement learning approaches~\cite{mou2022reinforcement}.")

# =============================================================================== R1.4 — conclusion.tex
E("M28 (R1.4)", "conclusion.tex",
  r"The 9-band optimal configuration selected one wavelength per excitation source, providing directly actionable sensor design guidelines.",
  r"The 9-band configuration retained all eight excitation sources with one emission band per source (two for one source), providing directly actionable sensor design guidelines.")


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
    bib = REV / "references.bib"
    add = (HERE.parent / "minimal_additions.bib").read_text(encoding="utf-8")
    btxt = bib.read_text(encoding="utf-8")
    if "hong2026hyperspectral" not in btxt:
        bib.write_text(btxt.rstrip("\n") + "\n\n" + add, encoding="utf-8")
        applied.append(("BIB", "references.bib"))
    print(f"applied {len(applied)} edits:")
    for eid, fname in applied:
        print(f"  {eid:18s} {fname}")


if __name__ == "__main__":
    main()
