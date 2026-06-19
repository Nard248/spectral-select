"""Validation harness: score a band selection against SpectraForge ground truth.

Given a ``GroundTruth`` (which knows exactly which excitation/emission bands carry fluorophore
signal) and a list of selected bands (``WavelengthBand`` objects or ``(excitation_nm, emission_nm)``
tuples), report how well the selection recovered the informative bands.

- ``precision`` — fraction of selected bands that land on a genuinely informative band.
- ``recall``    — fraction of informative bands the selection covered.
- ``f1``        — harmonic mean of the two.
- ``per_fluorophore`` / ``fluorophores_recovered`` — did the selection find *each* material's
  emission signature? (the intuitive "did we detect collagen / NADH / …" question).
"""
from __future__ import annotations

import numpy as np


def _normalize_selected(selected):
    """Accept WavelengthBand objects or (excitation_nm, emission_nm) tuples."""
    out = []
    for s in selected:
        if hasattr(s, "excitation_nm") and hasattr(s, "emission_nm"):
            out.append((float(s.excitation_nm), float(s.emission_nm)))
        else:
            ex, em = s
            out.append((float(ex), float(em)))
    return out


def validate_selection(ground_truth, selected, tol_nm: float = 10.0, threshold: float = 0.01) -> dict:
    grid = np.asarray(ground_truth.emission_grid, dtype=float)
    informative = ground_truth.informative_bands(threshold)       # {ex: bool array over grid}
    excis = list(informative.keys())
    sel = _normalize_selected(selected)
    n_selected = len(sel)

    def matched_ex(ex):
        """Nearest acquired excitation, or None if the selection's excitation isn't one we acquired."""
        if not excis:
            return None
        gex = min(excis, key=lambda e: abs(e - ex))
        return gex if abs(gex - ex) <= 1.0 else None

    def band_index(em_nm):
        """Nearest emission band, or None if outside tolerance."""
        j = int(np.argmin(np.abs(grid - em_nm)))
        return j if abs(grid[j] - em_nm) <= tol_nm else None

    hits = 0
    covered = set()
    for ex, em_nm in sel:
        gex = matched_ex(ex)
        if gex is None:
            continue
        j = band_index(em_nm)
        if j is not None and informative[gex][j]:
            hits += 1
            covered.add((gex, j))

    n_informative = int(sum(int(m.sum()) for m in informative.values()))
    precision = hits / n_selected if n_selected else 0.0
    recall = len(covered) / n_informative if n_informative else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Per-fluorophore recovery: did any selected band land on this fluorophore's signature?
    per_fluor_info = ground_truth.informative_bands_per_fluorophore(threshold)
    per_fluorophore = {}
    for fname, perex in per_fluor_info.items():
        recovered = False
        for ex, em_nm in sel:
            gex = matched_ex(ex)
            if gex is None or gex not in perex:
                continue
            j = band_index(em_nm)
            if j is not None and perex[gex][j]:
                recovered = True
                break
        per_fluorophore[fname] = bool(recovered)
    frac = (sum(per_fluorophore.values()) / len(per_fluorophore)) if per_fluorophore else 0.0

    # TIGHT, artifact-free recovery: did a selected band hit each fluorophore's TRUE emission peak
    # (argmax of its own per-pixel-max spectrum, at its best excitation), not just land anywhere on
    # the broad informative mask? This is the metric that actually discriminates a real selector
    # from chance — the broad-mask `fluorophores_recovered`/`precision` saturate (see mask_coverage).
    peak_hits = {}
    for fname, perex in ground_truth.per_fluorophore_spectra.items():
        if not perex:
            peak_hits[fname] = False
            continue
        best_ex = max(perex, key=lambda e: float(np.max(perex[e])))
        spec = np.asarray(perex[best_ex], dtype=float)
        if spec.max() <= 0:
            peak_hits[fname] = False
            continue
        peak_nm = float(grid[int(np.argmax(spec))])
        peak_hits[fname] = any(abs(ex - best_ex) <= 1.0 and abs(em_nm - peak_nm) <= tol_nm
                               for ex, em_nm in sel)
    peak_recovery = (sum(peak_hits.values()) / len(peak_hits)) if peak_hits else 0.0

    n_grid = sum(int(m.size) for m in informative.values())
    mask_coverage = (n_informative / n_grid) if n_grid else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits": hits,
        "n_selected": n_selected,
        "n_informative": n_informative,
        "per_fluorophore": per_fluorophore,
        "fluorophores_recovered": frac,
        "peak_hits": {k: bool(v) for k, v in peak_hits.items()},
        "peak_recovery": peak_recovery,
        "mask_coverage": mask_coverage,
    }
