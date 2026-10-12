"""Revision of 2026-09-16: the 2026-09-14 edits plus the two changes in the supervisor's response (Response_NAS.docx):
(a) a seventh reference, Tran et al. 2026 (J. Biomed. Opt.), cited with [1]-[3] in the opening sentence (R2.1);
(b) R1.4 wording: one band from each of seven excitation sources and two bands from the 430 nm source.
(c) Funding Declaration: the NIH grant clause removed (main.tex), at the authors' request; not part of the reviewer responses.

Applied to manuscript/revised/ (a fresh copy of original_v6). Each edit is an exact-string replacement
that must match exactly once. Six references are added (references.bib); none removed.
Edit IDs V1-V24.
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


# =============================================================================== Reviewer 2, points 1-3 — introduction.tex
E("V1 (R2.1, R2.2)", "introduction.tex",
  r"exceeding the discriminative capabilities of RGB or multispectral data~\cite{plaza2009recent,bioucas2013hyperspectral}.",
  r"exceeding the discriminative capabilities of RGB or multispectral data~\cite{plaza2009recent,bioucas2013hyperspectral,hong2024spectralgpt,tran2026medical}." + "\n"
  r"Recent applications range from Earth observation---for example the detection of photovoltaic power plants from satellite and airborne sensors~\cite{joerges2023photovoltaic} and the unsupervised unmixing of oceanic scenes affected by spectral variability and adjacency effects~\cite{deville2023oceanic}---to laboratory and clinical imaging, where the measured signal is increasingly fluorescence rather than reflectance." + "\n"
  r"In all of these settings the number of acquired bands is a recurring cost, which motivates the selection of a small informative subset.")

E("V2 (R2.3)", "introduction.tex",
  r"The challenge of identifying informative spectral bands has been extensively studied for conventional 3D HSI~\cite{sun2019hyperspectral,jia2020survey,asfour2018,guan2018}, with approaches spanning filter-based methods (variance, entropy), wrapper methods (sequential forward selection, genetic algorithms), and embedded methods (sparse regression, attention mechanisms)~\cite{martinez2006comparison,chang2000constrained,pudil1994floating,pal2010feature,tibshirani1996regression,mou2017deep}.",
  r"The challenge of identifying informative spectral bands has been extensively studied for conventional 3D HSI~\cite{sun2019hyperspectral,rasti2020featureextraction,jia2020survey,asfour2018,guan2018}, with approaches spanning filter-based methods (variance, entropy), wrapper methods (sequential forward selection, genetic algorithms), and embedded methods (sparse regression, attention mechanisms)~\cite{martinez2006comparison,chang2000constrained,pudil1994floating,pal2010feature,tibshirani1996regression,lorenzo2020attention}.")

# =============================================================================== Reviewer 2, points 4-6 — related_work.tex
E("V3 (R2.6)", "related_work.tex",
  r"This section reviews prior work in four areas relevant to the present framework: hyperspectral imaging systems and data characteristics, band selection methods for hyperspectral data, deep learning approaches for spectral analysis, and explainability methods for feature attribution.",
  r"This section reviews prior work in four areas relevant to the present framework: hyperspectral imaging systems and data characteristics, band selection methods for hyperspectral data, deep learning approaches for spectral analysis, and explainability methods for feature attribution." + "\n"
  r"The four parts are ordered from the data to the method: the modality and its 4D structure motivate the problem (Section~\ref{sec:related_hsi}), existing band selection methods define the gap (Section~\ref{sec:related_band}), deep representation learning (Section~\ref{sec:related_dl}) and perturbation-based attribution (Section~\ref{sec:related_explain}) supply the two tools that the framework combines, and the final part positions the framework against them (Section~\ref{sec:related_summary}).")

E("V4 (R2.4, R2.5)", "related_work.tex",
  r"""Band selection methods aim to identify a subset of spectral bands that retain discriminative information while reducing dimensionality~\cite{sun2019hyperspectral}.
These methods are typically categorized as filter, wrapper, or embedded approaches.

Filter methods rank bands based on statistical properties computed without reference to any downstream task.""",
  r"""Band selection methods aim to identify a subset of spectral bands that retain discriminative information while reducing dimensionality~\cite{sun2019hyperspectral,rasti2020featureextraction}.
These methods are typically categorized as filter, wrapper, or embedded approaches; the specific contribution of each family, and its limitation for ME-HSI data, are summarized in turn.

\textbf{Filter methods} rank bands based on statistical properties computed without reference to any downstream task.""")

E("V5 (R2.4)", "related_work.tex",
  r"Wrapper methods evaluate band subsets based on the performance of a classifier trained on the selected features.",
  r"\textbf{Wrapper methods} evaluate band subsets based on the performance of a classifier trained on the selected features.")

E("V6 (R2.4, R2.5)", "related_work.tex",
  r"""Embedded methods integrate band selection into model training.
Sparse regression (LASSO) induces feature selection through $L_1$ regularization~\cite{tibshirani1996regression}, attention mechanisms in neural networks learn to weight input bands~\cite{mou2017deep}, and reinforcement learning formulations treat band selection as a sequential decision process~\cite{roy2020hybridsn}.
While elegant, these approaches typically require labeled training data, limiting their applicability when ground truth is scarce or expensive to obtain.""",
  r"""\textbf{Embedded methods} integrate band selection into model training.
Sparse regression (LASSO) induces feature selection through $L_1$ regularization~\cite{tibshirani1996regression}, and attention mechanisms in neural networks trained for classification learn to weight input bands~\cite{lorenzo2020attention}.
While elegant, these approaches typically require labeled training data, limiting their applicability when ground truth is scarce or expensive to obtain.

\textbf{Unsupervised deep methods} learn band weights jointly with a reconstruction network and read the selection off the learned weights, as in BS-Nets~\cite{cai2020bsnets}.
Their contribution is label-free, end-to-end selection; their limitation is that a single attention layer over a single spectral axis presumes one rectangular cube, and that the selection is fixed at training time.""")

# =============================================================================== Reviewer 1, point 1 — methodology.tex
E("V7 (R1.1)", "methodology.tex",
  r"The top $k$ dimensions were then selected for perturbation analysis, where $k$ is treated as a hyperparameter; empirical evaluation (Section~\ref{sec:results}) revealed that $k=1$ (using only the highest-variance dimension) achieved optimal performance.",
  r"The top $k$ dimensions were then selected for perturbation analysis, where $k$ is treated as a hyperparameter; empirical evaluation (Section~\ref{sec:results}) revealed that PCA-based dimension selection (Section~\ref{sec:implementation}) with $k = 3$ achieved optimal performance.")

E("V8 (R1.1)", "methodology.tex",
  r"Variance-based scoring yielded the most consistent wavelength rankings, likely because high-variance dimensions captured the primary modes of spectral variation that distinguish different materials.",
  r"PCA-based scoring yielded the most consistent wavelength rankings, likely because its leading components captured the primary modes of spectral variation that distinguish different materials.")

# =============================================================================== Reviewer 2, point 7 — methodology.tex (rationale)
E("V9 (R2.7)", "methodology.tex",
  r"If perturbing a latent dimension significantly changed the reconstruction of a particular wavelength, that wavelength must be encoded by that dimension.",
  r"""If perturbing a latent dimension significantly changed the reconstruction of a particular wavelength, that wavelength must be encoded by that dimension.
The rationale follows from the autoencoder objective: a latent space trained to reconstruct the full 4D cube must retain whatever information the reconstruction requires~\cite{hinton2006reducing}, so wavelengths that are redundant or dominated by noise contribute little to it.
Tracing which excitation-emission pairs respond most strongly when a latent dimension is displaced therefore reveals the wavelengths on which the learned representation depends; the perturbation mechanism below performs this tracing, and the MMR step (Section~\ref{sec:mmr}) then removes redundancy among the wavelengths it identifies.""")

# =============================================================================== Reviewer 1, point 2 — methodology.tex
E("V10 (R1.2)", "methodology.tex",
  r"""For each selected latent dimension $d$, the standard deviation $\sigma_d$ was computed across spatial locations and samples, and then the perturbed representations were generated:
\begin{align}
\vect{z}^{+}_d &= \vect{z} + \epsilon \cdot \sigma_d \cdot \vect{e}_d \label{eq:perturb_pos}\\
\vect{z}^{-}_d &= \vect{z} - \epsilon \cdot \sigma_d \cdot \vect{e}_d \label{eq:perturb_neg}
\end{align}
where $\vect{e}_d$ is a one-hot vector selecting dimension $d$, and $\epsilon$ controls perturbation magnitude.
Values of $\epsilon \in \{15, 30, 45\}$ were used, and results across scales were aggregated to ensure robustness.
The $\sigma$-based scaling ensured perturbations were proportional to the natural variation in each dimension, avoiding both negligibly small and destructively large perturbations.""",
  r"""For each selected latent dimension $d$, the perturbed representations were generated in both directions by one of two schemes, a percentile scheme and an absolute-range scheme:
\begin{align}
\vect{z}^{\pm}_d &= \vect{z} + \left( Q_{50 \pm \epsilon/2}(z_d) - \bar{z}_d \right) \vect{e}_d \label{eq:perturb_pos}\\
\vect{z}^{\pm}_d &= \vect{z} \pm \tfrac{\epsilon}{100} \left( \max z_d - \min z_d \right) \vect{e}_d \label{eq:perturb_neg}
\end{align}
where $\vect{e}_d$ is a one-hot vector selecting dimension $d$, $Q_p(z_d)$ (rounded to the nearest tabulated percentile) and $\bar{z}_d$ are the $p$-th percentile and the mean of dimension $d$ across the analysis samples, $\max z_d - \min z_d$ is its observed range, and $\epsilon$ controls perturbation magnitude.
Values of $\epsilon \in \{30, 40, 50\}$ (medium) and $\epsilon \in \{50, 60, 70\}$ (high) were used, and results across the three magnitudes were aggregated to ensure robustness.
Scaling the displacement to the empirical distribution of each dimension ensured perturbations proportional to its natural variation, avoiding both negligibly small and destructively large perturbations.""")

E("V11 (R1.2)", "methodology.tex",
  r"""\begin{equation}
\Delta_d(\lambdaex, \lambdaem) = \frac{\left| \hat{X}^{+}_d(\lambdaex, \lambdaem) - \hat{X}^{-}_d(\lambdaex, \lambdaem) \right|}{2\epsilon \cdot \sigma_d}
\label{eq:sensitivity}
\end{equation}
where averaging was performed over spatial locations and samples.
This finite difference approximation estimates the partial derivative of reconstruction with respect to latent dimension $d$, quantifying how strongly each wavelength's reconstruction responds to changes in that dimension.""",
  r"""\begin{equation}
\begin{split}
\Delta_d(\lambdaex, \lambdaem) = \tfrac{1}{2}\Big( &\left| \hat{X}^{+}_d(\lambdaex, \lambdaem) - \hat{X}(\lambdaex, \lambdaem) \right| \\
&+ \left| \hat{X}^{-}_d(\lambdaex, \lambdaem) - \hat{X}(\lambdaex, \lambdaem) \right| \Big)
\end{split}
\label{eq:sensitivity}
\end{equation}
where $\hat{X}$ is the unperturbed reconstruction and averaging was performed over spatial locations and samples.
This quantifies how strongly each wavelength's reconstruction responds to displacements of latent dimension $d$.""")

# =============================================================================== Reviewer 2, point 7 — methodology.tex (lambda)
E("V12 (R2.7)", "methodology.tex",
  r"Empirical evaluation (Section~\ref{sec:results}) identified $\lambda = 0.5$ as optimal, providing balanced consideration of both criteria.",
  r"The value $\lambda = 0.5$ was fixed a priori, giving equal weight to both criteria~\cite{carbonell1998use}.")

# =============================================================================== Reviewer 1, point 3 — methodology.tex (III-E)
E("V13 (R1.3)", "methodology.tex",
  r"A supervised classification algorithm is used to quantify how much discriminative power remains in the compressed features; the classification described below serves \emph{only} to evaluate the quality of selected wavelengths---it does not influence the selection process.",
  r"""A supervised classification algorithm is used to quantify how much discriminative power remains in the compressed features; the classification described below serves \emph{only} to evaluate the quality of selected wavelengths---it does not influence the selection process.
The labels play the same validating role in choosing among the attribution configurations of Section~\ref{sec:configurations}: every configuration produces its wavelength ranking without labels, and the classifier is then used to determine which configuration yielded the most informative selection.""")

# =============================================================================== Reviewer 1, point 2 / Reviewer 2, point 7 — experimental_setup.tex
E("V14 (R1.2)", "experimental_setup.tex",
  r"Both methods were explored at $k \in \{1, 3\}$ top dimensions. Perturbations were applied using two methods, percentile-based and absolute-range, at medium ($\epsilon \in \{30, 40, 50\}$) and high ($\epsilon \in \{50, 60, 70\}$) magnitude levels.",
  r"""Both methods were explored at $k \in \{1, 3\}$ top dimensions. Perturbations were applied using two methods, percentile-based and absolute-range (Section~\ref{sec:perturbation}), at medium ($\epsilon \in \{30, 40, 50\}$) and high ($\epsilon \in \{50, 60, 70\}$) magnitude levels.
The best configurations of Table~\ref{tab:main_results} were obtained with the percentile scheme and the medium magnitude set, except the 30- and 9-band rows (absolute range, medium) and the 50-band row (percentile, high).""")

# =============================================================================== Reviewer 1, point 4 — results.tex
E("V16 (R1.4)", "results.tex",
  r"""The 9-band optimal configuration exemplifies the framework's selection strategy: it selects exactly one wavelength from each of the 8 excitation sources (310--430\,nm), with emission peaks spanning 500--680\,nm.
This one-per-excitation pattern suggests that the framework identifies the single most informative emission band at each excitation, consistent with the parallel-branch architecture that processes each excitation independently before feature averaging.""",
  r"""The 9-band configuration of Table~\ref{tab:main_results} comprises the following excitation/emission pairs (nm), in selection order: 310/520, 385/500, 325/570, 430/610, 400/510, 340/500, 365/520, 415/615 and 430/680, i.e., one band from each of the seven excitation sources between 310 and 415\,nm and two bands from the 430\,nm source, with emission bands spanning 500--680\,nm.
This near one-per-excitation pattern reflects the per-excitation normalization of this configuration, which ranks the strongest band of every excitation first.""")

# =============================================================================== Reviewer 1, point 3 — results.tex (consistency)
E("V17 (R1.3)", "results.tex",
  r"No hyperparameters, attribution settings, or selection parameters were retuned; only the input data and the number of excitation branches changed, the latter determined automatically by the dataset.",
  r"No hyperparameters or selection parameters were retuned; only the input data and the number of excitation branches changed, the latter determined automatically by the dataset.")

# =============================================================================== discussion.tex
E("V18 (R2.7)", "discussion.tex",
  r"The selection with MMR parameter of $\lambda = 1$ underperformed the balanced setting ($\lambda = 0.5$) used throughout the experiments.",
  r"The balanced setting $\lambda = 0.5$ was used throughout the experiments; the redundancy structure of fluorescence spectra explains why the diversity term matters.")

E("V19 (R1.4)", "discussion.tex",
  r"The one-per-excitation pattern observed in the 9-band configuration (Section~\ref{sec:results_wavelengths}) directly informs sensor design: coverage across the full excitation range is essential, but only a single emission band per excitation can be needed for effective discrimination.",
  r"The near one-per-excitation pattern observed in the 9-band configuration (Section~\ref{sec:results_wavelengths}) directly informs sensor design: all eight excitation sources are retained, but only one or two emission bands per excitation are needed for effective discrimination.")

E("V20 (R2.5)", "discussion.tex",
  r"For 3D HSI band selection, deep learning approaches include attention-based networks~\cite{mou2017deep} and reinforcement learning approaches~\cite{roy2020hybridsn}.",
  r"For 3D HSI band selection, deep learning approaches include attention-based networks~\cite{lorenzo2020attention}.")

# =============================================================================== conclusion.tex
E("V21 (R1.4)", "conclusion.tex",
  r"The 9-band optimal configuration selected one wavelength per excitation source, providing directly actionable sensor design guidelines.",
  r"The 9-band optimal configuration selected one wavelength per excitation source (two for one source), providing directly actionable sensor design guidelines.")


NEW_BIB = r"""
%% ---- added in the revision of 2026-09-16 (7 entries) ----
@article{tran2026medical,
  author  = {Tran, Minh Ha and Ma, Ling and Yuan, Mandy and others},
  title   = {Medical hyperspectral imaging: an updated review of technology advancements and biomedical applications},
  journal = {Journal of Biomedical Optics},
  volume  = {31},
  number  = {3},
  pages   = {030901},
  year    = {2026},
  doi     = {10.1117/1.JBO.31.3.030901}
}
@article{hong2024spectralgpt,
  author  = {Hong, Danfeng and Zhang, Bing and Li, Xuyang and Li, Yuxuan and Li, Chenyu and Yao, Jing and Yokoya, Naoto and Li, Hao and Ghamisi, Pedram and Jia, Xiuping and Plaza, Antonio and Gamba, Paolo and Benediktsson, Jon Atli and Chanussot, Jocelyn},
  title   = {{SpectralGPT}: Spectral remote sensing foundation model},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {46},
  number  = {8},
  pages   = {5227--5244},
  year    = {2024},
  doi     = {10.1109/TPAMI.2024.3362475}
}
@article{joerges2023photovoltaic,
  author  = {J{\"o}rges, Christoph and Vidal, Hedwig Sophie and Hank, Tobias and Bach, Heike},
  title   = {Detection of solar photovoltaic power plants using satellite and airborne hyperspectral imaging},
  journal = {Remote Sensing},
  volume  = {15},
  number  = {13},
  pages   = {3403},
  year    = {2023},
  doi     = {10.3390/rs15133403}
}
@article{deville2023oceanic,
  author  = {Deville, Yannick and Brezini, Salah Eddine and Benhalouche, Fatima Zohra and Karoui, Moussa Sofiane and Guillaume, Mireille and Lenot, Xavier and Lafrance, Bruno and Chami, Malik and Jay, Sylvain and Minghelli, Audrey and Briottet, Xavier and Serfaty, V{\'e}ronique},
  title   = {Modeling and unsupervised unmixing based on spectral variability for hyperspectral oceanic remote sensing data with adjacency effects},
  journal = {Remote Sensing},
  volume  = {15},
  number  = {18},
  pages   = {4583},
  year    = {2023},
  doi     = {10.3390/rs15184583}
}
@article{rasti2020featureextraction,
  author  = {Rasti, Behnood and Hong, Danfeng and Hang, Renlong and Ghamisi, Pedram and Kang, Xudong and Chanussot, Jocelyn and Benediktsson, Jon Atli},
  title   = {Feature extraction for hyperspectral imagery: The evolution from shallow to deep: Overview and toolbox},
  journal = {IEEE Geoscience and Remote Sensing Magazine},
  volume  = {8},
  number  = {4},
  pages   = {60--88},
  year    = {2020},
  doi     = {10.1109/MGRS.2020.2979764}
}
@article{lorenzo2020attention,
  author  = {Lorenzo, Pablo Ribalta and Tulczyjew, Lukasz and Marcinkiewicz, Michal and Nalepa, Jakub},
  title   = {Hyperspectral band selection using attention-based convolutional neural networks},
  journal = {IEEE Access},
  volume  = {8},
  pages   = {42384--42403},
  year    = {2020},
  doi     = {10.1109/ACCESS.2020.2977454}
}
@article{cai2020bsnets,
  author  = {Cai, Yaoming and Liu, Xiaobo and Cai, Zhihua},
  title   = {{BS-Nets}: An end-to-end framework for band selection of hyperspectral image},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {58},
  number  = {3},
  pages   = {1969--1984},
  year    = {2020},
  doi     = {10.1109/TGRS.2019.2951433}
}
"""


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
    mt = REV / "main.tex"
    mtext = mt.read_text(encoding="utf-8")
    fund_old = r"This work was supported by the European Union (ERA Chair NAS-SAR award, grant no.\ 101087403) and by the United States National Institutes of Health (grant no.\ R44HL120511)."
    fund_new = r"This work was supported by the European Union (ERA Chair NAS-SAR award, grant no.\ 101087403)."
    if mtext.count(fund_old) != 1:
        print("ABORT: funding sentence not found exactly once in main.tex"); sys.exit(1)
    mt.write_text(mtext.replace(fund_old, fund_new), encoding="utf-8")
    applied.append(("FUND (main.tex)", "main.tex"))
    bib = REV / "references.bib"
    btxt = bib.read_text(encoding="utf-8")
    if "tran2026medical" not in btxt:
        bib.write_text(btxt.rstrip("\n") + "\n" + NEW_BIB, encoding="utf-8")
        applied.append(("BIB (+6)", "references.bib"))
    print(f"applied {len(applied)} edits:")
    for eid, fname in applied:
        print(f"  {eid:18s} {fname}")


if __name__ == "__main__":
    main()
