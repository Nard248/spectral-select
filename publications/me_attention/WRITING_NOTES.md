# Notes for authors (pass 3, 2026-10-19)

Remaining facts to confirm (red `\RESULT` in main.tex):
1. Co-authors, affiliations, funding/acknowledgements (title block).
2. Reason for the removed 70-90 nm emission block per excitation (Introduction).
3. Radiometric correction (exposure + lamp power) actually applied to each dataset (Sec. Data, Acquisition).
4. Count of dropped components (<100 px) per dataset: 0/0 Pepsin and Lichens 1; 1 component/18 px Lichens 2 (Sec. Data).
5. Repository URL and commit; which Zenodo record holds which dataset, and whether Lichens 2 is released (Data and Code Availability).
Also unverified: the 5 nm offset of the 415 nm excitation grid in Lichens 1 (processed vectors show 10 nm steps, union grid M=55); whether regions of one class in Lichens 2 belong to one thallus; refit uses best inner epoch (>=3) without train-size scaling (harness.py, differs from SPEC.md).

Pass 3 decisions: "no reliable win" framing everywhere; ablation table and top-K selection table removed (not run; stated in Limitations); CD diagram not shown (3 datasets, CD=9.62 uninformative; average ranks quoted in text); Fig. 5 = rollout panels (Pepsin, Lichens 1) cropped from generated importance PDFs; Lichens 1 spatial attention rerun (after OOM) included. Regenerate with `PYTHONPATH=src python -m me_attention.analysis --sweep night --out publications/me_attention/generated`.
"Protocol effect larger than model effect" is stated as: 10-12 points vs <1 point between best attention and best baseline; comparable to the best-to-worst spread on Pepsin (11.9).

# WRITING_NOTES: remaining placeholders in main.tex (pass 2)

Conventions: `\RESULT{...}` (red) = number/text to fill from results or to confirm; `\TBD` (red [--]) = table cell; `\ALT{A}{B}` (blue) = choose framing A (attention wins) or B (no reliable win), delete the other. Search main.tex for `\RESULT`, `\TBD`, `\ALT` until none remain, then remove the three macro definitions.

Already filled in pass 2: T=3, E_max=25, repeats (6/4/4 units initial; spectral extended to 15/10/4, stated as post-hoc), baseline list (8 spectral + 5 spatial = 13), model list, dev findings subsection, training objective, ex_tokens_v2 / hybrid_conv_attn descriptions + Fig. extokens, random-pixel vs specimen numbers, hardware/compute-constraint statements, dropped-model statements.

## Fill from the final sweep (results/me_attention/night/<dataset>/<model>.jsonl)

- Tab. spectral (12 models: LR, RBF-SVM, RF, HGB, kNN, MLP, 1D-CNN, SpectralFormer; prototype, meas_tokens, ex_tokens_v2, hybrid_conv_attn) and Tab. spatial (patch-mean LR/RF combined row: split or report best, say which; 2D-CNN, SpectralFormer-patch, SSFTT, spatial_attn): mean +/- std balanced accuracy, bold best, Delta row from `protocol.stats_paired`. Note unit counts differ per track (spectral 15/10/4; spatial 6/4/4).
- Tab. stats: per dataset x track `stats_paired` + `protocol.holm`.
- Fig. box (replace `\fbox`), CD diagrams, per-class recall comment, specimen majority-vote x/y.
- Tab. ablation: keep only ablations actually run (axial row already removed); shuffled-wavelength control needs a run; interpretation text.
- Fig. importance + Tab. select (rollout/gate; gates are NOT in the frozen meas_tokens space, so gate importance needs a separate run or should be dropped; K values; Spearman); `\fbox` placeholders.
- Choose `\ALT` branches (abstract, intro contributions, spectral/spatial results text, discussion, conclusion) and write the `\RESULT` interpretation items in Discussion; the abstract still has `\RESULT{x--y}`, `\RESULT{k}`, `\RESULT{x}`.
- Discussion sentence on protocol vs model choice: now supportable by dev gap (10-12 points) but verify against final numbers.
- Results header says "placeholders until the final sweep is complete": remove when done.

## Facts to confirm (not result-dependent)

- Authors/affiliations/funding; repository URL and commit; which Zenodo record holds which dataset (Lichens 2 and Acetic may not be released).
- Sec. Introduction: reason for the removed 70-90 nm block per excitation.
- Sec. Data: radiometric correction applied to each dataset; 5 nm offset claim for Lichens 1 (processed vectors show 10 nm steps, union grid M=55); dropped-component counts (0/0 Pepsin and Lichens 1, 1 component/18 px Lichens 2).
- Lichens 2: whether regions of one class belong to the same thallus (Limitations).
- Refit uses best inner epoch (>=3) without train-size scaling (follows harness.py, differs from SPEC.md).
- "Distinct specimens 9, 16, 28" in Limitations: re-check order vs the new unit counts (15, 4, 10).
- Method text for the measurement-token family describes options (Fourier/learned, rel. bias, gates, axial); only the frozen-space options were evaluated (see text).
