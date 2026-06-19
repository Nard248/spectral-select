# SpectraForge — Roadmap (next increments)

> **STATUS (2026-06-20):** Increments **A, B, C, E, D** built; harness then **corrected** after
> adversarial review. The earlier "method recovers on realistic spectra, not on Gaussians" headline
> is **RETRACTED** — it was an artifact of a saturated ground-truth mask (83-93% of the grid) with no
> chance baseline; a uniformly-random selector matched/beat the AE. The harness now has a random
> baseline + tight `peak_recovery`/`mask_coverage` metrics. On peak_recovery the AE scores ~0 vs
> random ~0.33, but this is **INCONCLUSIVE** (peaks need not be the discriminative target); a
> discriminability-grounded ground truth is the real open task. No claim went into any paper.
> See `reports/fpbase_validation.py` and `reports/spectraforge_validation_report.py`.

**North star:** rigorous, scalable validation of unsupervised band-selection methods using
chemically-faithful synthetic ME-HSI data with perfect ground truth. Every increment must serve
**realism**, **fidelity**, **scale**, or **usability** toward that.

**Where we are (done):** engine (fluorophores → materials → scene → `render()` → `SpectraData` +
`GroundTruth`, linear EEM model, scatter lines + noise) and the Forge GUI workbench
(library/spectra + define-fluorophore, material composer, layered canvas with brush/eraser/rect/
circle painting, layer visibility/reorder, render + slice-preview, export). ~436 tests; binds to
spectral-select; surfaced + fixed a real `selection_core` NaN bug.

Each increment below is its own spec → plan → TDD → verify → push cycle (the pipeline).

---

## Increment A — Persistence & brush control  *(usability; unblocks iterative use)*
- **Project save/load** (`.forge` = a single compressed `.npz`): serialize `ForgeState` — custom
  fluorophores, materials, layers (incl. `amount_map` arrays), acquisition, artifacts, seed — and
  reload it identically. GUI **File menu**: New / Open… / Save…
- **Brush value control**: a spin box for the painted amount (currently fixed 1.0); enables
  graded concentrations.
- **DoD:** `save_project`/`load_project` round-trip (pure, tested); File menu wired; brush value
  adjustable; offscreen smoke + screenshot.
- **Effort:** S · **Risk:** low. *(Pre-approved as the next batch.)*

## Increment B — Validation harness  *(the research payoff; small, highest-leverage)*
- A pure `validate_selection(ground_truth, selected_bands, tol_nm) → {recall, precision, f1,
  per_fluorophore}`: did the selection recover the bands the ground truth says carry signal?
- A "Validate" action: render → run `Analyzer.fit/select` → score → report. Closes the loop
  **synthetic → select → auto-score** — this is literally "verify our methods."
- **DoD:** `validate_selection` tested on a 2-fluorophore scene (selection at the planted peaks
  scores high recall); CLI/GUI report.
- **Effort:** S–M · **Risk:** low. **Recommendation: do this right after A** — it's the smallest
  piece that directly delivers the project's purpose.

## Increment C — Richer physics  *(realism; the meaty scientific increment)*
- **Beer–Lambert inner-filter**: opt-in non-linear regime `F ∝ (1 − 10^(−Σ εcl))` + secondary
  self-absorption; linear remains the default (the `render(A+B)==render(A)+render(B)` invariant
  stays the guard for linear mode).
- **PSF blur**: per-band Gaussian spatial convolution (`scipy.ndimage`), configurable σ.
- **Autofluorescence background**: a low, broad baseline term / background material.
- **DoD:** each opt-in; linear default byte-unchanged (existing tests pass); new tests (inner-filter
  saturates high-conc signal sub-linearly; PSF blurs a point source; background raises the floor);
  screenshot. *(Pre-approved to follow A.)*
- **Effort:** M · **Risk:** medium (guarded by keeping linear the default).

## Increment D — Real fluorophore data  *(fidelity)*
- Extend `Fluorophore` to optionally carry **measured** ex/em arrays (interpolated) alongside the
  parametric Gaussian; CSV import (`wavelength,intensity`); optional **FPbase** fetch with local
  cache + offline fallback (network-guarded).
- **DoD:** import a CSV spectrum → used by `render`; tests with a synthetic CSV; FPbase mocked/optional.
- **Effort:** M · **Risk:** low–medium.

## Increment E — Batch / sweep generation  *(scale; "experiment a lot")*
- A batch spec (base scene + parameter grid: noise levels, n_bands, excitation sets, seeds,
  concentration scales) → N datasets + a manifest; with B, auto-run validation per dataset → an
  aggregate report (selection accuracy vs noise/bands). CLI `spectraforge-batch` (+ optional GUI tab).
- **DoD:** a small sweep (e.g. 3 noise × 2 seeds) → 6 datasets + manifest; with B, a CSV of recall vs noise.
- **Effort:** M · **Risk:** low. Builds on B.

## Increment F — Advanced (deferred / optional)
FRET coupling between proximal fluorophores; pH/solvent spectral shifts; bi-Gaussian asymmetric
spectra; soft (falloff) brushes; richer multi-channel display. Niche; pick up if a specific
experiment needs them.

---

## Recommended sequence & rationale
**A → B → C → E → D → F.**
- **A** first: makes the tool usable for real iterative work (don't lose a composition) — cheap, unblocks everything.
- **B** next (inserted before C): it is the *smallest* increment that directly delivers the north
  star — automatically scoring whether band selection recovers the known-informative bands. High
  value per unit effort; everything after it can report validation numbers.
- **C** (richer physics): makes synthetic realism transfer to real instruments; biggest scientific lift.
- **E** (batch + auto-validate): turns the tool into an experiment engine — sweep noise/bands and
  chart how selection degrades. The combination B+C+E is the actual research instrument.
- **D** (real spectra): fidelity boost; independent, can slot in whenever real-dye studies are needed.
- **F**: only when a concrete experiment demands it.

*(You pre-approved A then C. I recommend slotting B between them — it's small and it's the payoff.
Happy to keep strict A→C if you'd rather.)*
