"""Publication-ready Methods paragraph generator.

``methods_text(exp, study=None, style="markdown"|"latex", software=None)`` renders a deterministic description of a
simulated experiment: geometry, the optical properties of every material at the excitation wavelengths (absorption
mu_a, reduced scattering mu_s' = mu_s (1 - g), anisotropy g, refractive index n, dye concentrations - evaluated with
the same :func:`optics.material_optics` the simulator uses), source, camera, spectral filter, sensor, the physics
toggles that are switched OFF, Monte Carlo photon counts, seeds, software version / commit and a citation line.
With a :class:`StudySpec` a "Study design" paragraph (design, parameters, replicates, metrics, sensitivity
estimators) is appended.  Everything is derived from the data model, so the text cannot drift from what was run.

``software`` (dict with ``package_version``, ``git`` ...) defaults to :func:`provenance.provenance`; pass a fixed
dict for reproducible snapshots.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from .provenance import experiment_hash, provenance

CITATION = ("N. Meloyan, \"SpectralForge: A Physics-Based Simulator for Fluorescence Hyperspectral Imaging,\" "
            "16th IEEE GRSS Workshop on Hyperspectral Imaging and Signal Processing (WHISPERS), Glasgow, 2026.")

_LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


class _Style:
    def __init__(self, style: str):
        if style not in ("markdown", "latex"):
            raise ValueError("style must be 'markdown' or 'latex'")
        self.latex = style == "latex"

    def esc(self, s: Any) -> str:
        s = str(s)
        return "".join(_LATEX_SPECIAL.get(c, c) for c in s) if self.latex else s

    def h(self, title: str) -> str:
        return f"\\paragraph{{{title}.}}" if self.latex else f"**{title}.**"

    def code(self, s: str) -> str:
        return f"\\texttt{{{self.esc(s)}}}" if self.latex else f"`{s}`"

    def unit(self, v: str, u: str) -> str:
        return f"{v}\\,{u}" if self.latex else f"{v} {u}"

    def nm(self, v: float) -> str:
        return self.unit(f"{v:g}", "nm")

    def mm(self, v: float) -> str:
        return self.unit(f"{v:g}", "mm")

    @property
    def mua(self) -> str:
        return r"$\mu_a$" if self.latex else "μa"

    @property
    def musp(self) -> str:
        return r"$\mu_s'$" if self.latex else "μs'"

    @property
    def per_mm(self) -> str:
        return r"mm$^{-1}$" if self.latex else "mm⁻¹"

    @property
    def um(self) -> str:
        return r"\textmu{}M" if self.latex else "µM"


def _join(items: list[str], last: str = "and") -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {last} " + items[-1]


def _fmt(v: float) -> str:
    return f"{v:.3g}" if v != 0 else "0"


def _collect_materials(exp) -> list[tuple[str, str]]:
    """(material id, where used) in order of first appearance."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(mid, where):
        if mid and mid not in seen:
            seen.add(mid)
            out.append((mid, where))
    for o in exp.objects:
        shape = o.shape
        if shape.kind == "layered_block":
            for layer in shape.layers:
                add(layer.material_id, f"{o.name}: layer '{layer.name}'")
            for inc in getattr(shape, "inclusions", []) or []:
                add(inc.material_id, f"{o.name}: inclusion '{inc.name}'")
        else:
            add(o.material_id, o.name)
            add(getattr(o, "vessel_material_id", None) or ("glass" if shape.kind in ("petri_dish", "cuvette") else None),
                f"{o.name}: vessel")
    if exp.background.kind != "none":
        add(exp.background.material_id, "background")
    return out


def _geometry(exp, S: _Style) -> str:
    parts = []
    for o in exp.objects:
        s = o.shape
        pos = tuple(round(v, 3) for v in o.transform.position_mm)
        if s.kind == "layered_block":
            lay = []
            for l in s.layers:
                t = l.thickness
                prof = (S.mm(t.thickness_mm) if t.kind == "uniform"
                        else f"{S.mm(t.thickness_mm)} to {S.mm(t.end_thickness_mm)} along {t.axis}")
                lay.append(f"{S.esc(l.name)} ({S.esc(l.material_id)}, {prof})")
            txt = (f"{S.esc(o.name)}: a layered block of {s.footprint_mm[0]:g} × {s.footprint_mm[1]:g} "
                   f"{S.esc('mm')} footprint with layers (bottom to top) " + _join(lay, "then"))
            incs = getattr(s, "inclusions", []) or []
            if incs:
                txt += "; inclusions: " + _join([f"{S.esc(i.name)} ({i.kind}, {S.esc(i.material_id)})" for i in incs])
        else:
            dims = {k: v for k, v in s.model_dump().items() if k != "kind"}
            txt = f"{S.esc(o.name)}: {s.kind.replace('_', ' ')} ({S.esc(', '.join(f'{k}={v}' for k, v in dims.items()))})"
        parts.append(txt + f", centred at {S.esc(pos)} mm")
    bg = ("no background plane" if exp.background.kind == "none"
          else f"a {exp.background.extent_mm:g} mm background plane of {S.esc(exp.background.material_id)}")
    return (f"The scene consisted of {len(exp.objects)} object(s) on {bg}. " +
            ("; ".join(p for p in parts) + "." if parts else ""))


def _optics_table(exp, S: _Style, ex_list: list[float]) -> str:
    from .optics import material_optics, merged_fluorophores
    fl = merged_fluorophores(exp)
    tg = exp.acquisition.physics
    rows = []
    for mid, where in _collect_materials(exp):
        try:
            m = exp.material(mid)
            o = material_optics(m, fl, toggles=tg)
        except Exception:  # noqa: BLE001 - unknown material: skip, validation reports it elsewhere
            continue
        mua = [float(np.interp(w, o.wavelengths_nm, o.mua_per_mm)) for w in ex_list]
        musp = [float(np.interp(w, o.wavelengths_nm, o.mus_per_mm)) * (1 - o.g) for w in ex_list]
        dyes = _join([f"{S.esc(d.fluorophore)} {d.concentration_uM:g} {S.um}" for d in m.dyes]) or "none"
        rows.append((mid, where, o.n, o.g, mua, musp, dyes, bool(o.clear)))
    if not rows:
        return ""
    head = ["material", "n", "g"] + [f"{S.mua} / {S.musp} at {w:g} nm ({S.per_mm})" for w in ex_list] + ["dyes"]
    lines = []
    if S.latex:
        lines.append(r"\begin{tabular}{l" + "r" * (2 + len(ex_list)) + "l}")
        lines.append(r"\hline " + " & ".join(head) + r" \\ \hline")
        for mid, where, n, g, mua, musp, dyes, clear in rows:
            cells = [S.esc(mid), f"{n:.3g}", "--" if clear else f"{g:.3g}"] + \
                    [f"{_fmt(a)} / {'--' if clear else _fmt(b)}" for a, b in zip(mua, musp)] + [dyes]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline \end{tabular}")
    else:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        for mid, where, n, g, mua, musp, dyes, clear in rows:
            cells = [f"{mid}", f"{n:.3g}", "-" if clear else f"{g:.3g}"] + \
                    [f"{_fmt(a)} / {'-' if clear else _fmt(b)}" for a, b in zip(mua, musp)] + [dyes]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _source(exp, S: _Style) -> tuple[str, list[float]]:
    src = exp.source()
    steps = [s for s in src.sequence if s.enabled]
    exs = sorted({float(s.wavelength_nm) for s in steps})
    seq = _join([f"{s.wavelength_nm:g} nm ({s.power_mW:g} mW, {s.exposure_ms:g} ms)" for s in steps])
    drift = f", step-to-step power drift {src.power_drift_pct:g} % (1σ)" if src.power_drift_pct and \
        exp.acquisition.physics.power_drift else ""
    txt = (f"Excitation was provided by {S.esc(src.name)} ({src.kind.replace('_', ' ')}; line width "
           f"{src.line_fwhm_nm:g} nm FWHM; {src.beam.profile.replace('_', ' ')} beam of "
           f"{src.beam.aperture_diameter_mm:g} mm diameter, {src.beam.divergence_half_angle_deg:g}° half-angle "
           f"divergence; clean-up filter OD {src.excitation_filter_od:g}{drift}) at "
           f"{S.esc(tuple(round(v, 1) for v in src.position_mm))} mm aimed at {S.esc(tuple(round(v, 1) for v in src.target_mm))} mm. "
           f"The excitation sequence (wavelength, delivered power, exposure) was {S.esc(seq)}.")
    return txt, exs


def _camera(exp, S: _Style) -> str:
    cam = exp.camera()
    f, sn, ln = cam.filter, cam.sensor, cam.lens
    eff = sn.pixel_pitch_um * sn.binning
    bands = f"{f.n_bands} bands from {f.band_start_nm:g} nm in {f.band_step_nm:g} nm steps"
    if f.band_start_policy == "excitation_offset":
        bands = f"{f.n_bands} bands in {f.band_step_nm:g} nm steps starting {f.start_offset_nm:g} nm above each excitation line"
    lp = f", {f.emission_longpass_nm:g} nm long-pass (OD {f.longpass_od:g})" if f.emission_longpass_nm else ""
    return (f"Images were recorded with {S.esc(cam.name)}: a {ln.focal_length_mm:g} mm f/{ln.f_number:g} lens focused at "
            f"{ln.focus_distance_mm:g} mm, a {sn.width_px}×{sn.height_px} px sensor ({sn.pixel_pitch_um:g} µm pitch, "
            f"{sn.binning}×{sn.binning} binning, effective {eff:g} µm; {sn.qe_curve} quantum-efficiency curve, "
            f"{sn.bit_depth}-bit, gain {sn.gain_e_per_adu:g} e⁻/ADU, offset {sn.offset_adu:g} ADU, read noise "
            f"{sn.read_noise_e:g} e⁻, dark current {sn.dark_current_e_per_s:g} e⁻/s, full well {sn.full_well_e:g} e⁻, "
            f"PRNU {sn.prnu_pct:g} %, DSNU {sn.dsnu_e:g} e⁻) and a {f.kind.replace('_', ' ')} spectral filter "
            f"({bands}; {f.fwhm_nm:g} nm FWHM{lp}).")


def _physics(exp, S: _Style) -> str:
    tg = exp.acquisition.physics
    off = [k for k, v in tg.model_dump().items() if v is False]
    if not off:
        return "All physics modules of the simulator were enabled."
    return ("The following physics modules were switched OFF: " + ", ".join(S.code(k) for k in off) +
            "; all others were enabled.")


def _mc(exp, S: _Style) -> str:
    a, v = exp.acquisition, exp.acquisition.volume_mc
    has_block = any(o.shape.kind == "layered_block" for o in exp.objects)
    txt = f"The per-material transport Monte Carlo used {a.mc_photons:,} photons per bake. "
    if has_block:
        extra = f", inclusion voxel {v.inclusion_voxel_mm:g} mm" if getattr(v, "inclusion_voxel_mm", None) else ""
        txt += (f"Layered blocks were solved with a 3D layered Monte Carlo ({v.excitation_photons:,} excitation and "
                f"{v.emission_photons:,} emission photons per step, {v.sublayers} sublayers per layer, lateral cell "
                f"{v.lateral_cell_mm:g} mm, exit cell {v.exit_cell_mm:g} mm, emission nodes every "
                f"{v.emission_node_step_nm:g} nm{extra}).")
    return txt


def methods_text(exp, study=None, *, style: str = "markdown", software: dict | None = None) -> str:
    """Methods paragraph(s) for ``exp`` (and optionally a :class:`StudySpec`); see the module docstring."""
    S = _Style(style)
    sw = software or provenance()
    git = sw.get("git") or {}
    commit = (git.get("commit") or "unknown")[:10] + (" (modified working tree)" if git.get("dirty") else "")
    src_txt, exs = _source(exp, S)
    sections = [
        S.h("Simulation") + f" Multi-excitation fluorescence hyperspectral images were simulated with SpectralForge Studio "
        f"(version {S.esc(sw.get('package_version', '?'))}, commit {S.esc(commit)}), a physics-based forward model that "
        f"combines ray tracing of the scene, Monte Carlo light transport, dye absorption, fluorescence emission and a "
        f"detector model. The experiment {S.esc(repr(exp.name))} (SHA-256 {S.code(experiment_hash(exp)[:16])}) was "
        f"simulated in {S.esc(exp.acquisition.mode)} mode.",
        S.h("Scene") + " " + _geometry(exp, S),
    ]
    tbl = _optics_table(exp, S, exs) if exs else ""
    musp_def = r"$\mu_s(1-g)$" if S.latex else "μs(1 − g)"
    sections.append(S.h("Optical properties") +
                    f" Optical properties at the excitation wavelengths ({_join([S.nm(w) for w in exs])}) are listed below: "
                    f"refractive index n, scattering anisotropy g, absorption coefficient {S.mua} and reduced scattering "
                    f"coefficient {S.musp} = {musp_def} in {S.per_mm}, and dye "
                    "concentrations. Non-scattering (clear) materials have no g or " + S.musp + "." + ("\n\n" + tbl if tbl else ""))
    sections += [S.h("Source") + " " + src_txt, S.h("Detection") + " " + _camera(exp, S),
                 S.h("Physics and Monte Carlo") + " " + _physics(exp, S) + " " + _mc(exp, S) +
                 f" Random numbers were seeded with {exp.acquisition.seed} (the identical seed reproduces the identical result)."]
    if study is not None:
        sections.append(S.h("Study design") + " " + _study_text(study, S))
    sections.append(S.h("Software") + f" Simulations ran with Python {S.esc(sw.get('python', '?'))}, NumPy "
                    f"{S.esc(sw.get('numpy', '?'))} and Numba {S.esc((sw.get('numba') or {}).get('version', '?'))} on "
                    f"{S.esc(sw.get('platform', '?'))}. If you use this simulator please cite: {S.esc(CITATION)}")
    return "\n\n".join(sections) + "\n"


def _study_text(study, S: _Style) -> str:
    d = study.design
    plist = []
    for p in study.parameters:
        if p.kind == "choice":
            rng = "one of " + _join([S.esc(v) for v in p.values], "or")
        else:
            rng = f"{p.low:g} to {p.high:g}" + (" (log-uniform)" if p.scale == "log" else "")
        plist.append(f"{S.esc(p.label)} ({S.code(p.pointer)}: {rng})")
    k = len(study.parameters)
    kinds = {
        "grid": f"a full-factorial grid with {d.levels} levels per continuous parameter",
        "random": f"{d.n} uniformly random points",
        "lhs": f"a Latin hypercube sample of {d.n} points (McKay et al., 1979)",
        "sobol": f"{d.n} points of a scrambled Sobol' sequence (Owen, 1998)",
        "saltelli": (f"Saltelli's sampling scheme (Saltelli, 2010) with N = {d.n} base samples from a scrambled Sobol' "
                     f"sequence, i.e. N(k + 2) = {d.n * (k + 2)} design points, analysed with the Saltelli (first-order) "
                     "and Jansen (total-order) estimators (Jansen, 1999) with bootstrap 95 % confidence intervals"),
        "morris": (f"{d.n} Morris trajectories on a {d.levels}-level grid (Morris, 1991; Campolongo et al., 2007), "
                   f"i.e. {d.n * (k + 1)} design points, summarised by the mean absolute elementary effect μ* and σ"),
    }
    mets = _join([f"{S.esc(m.name)} ({m.kind.replace('_', ' ')})" for m in study.metrics])
    mode = "preview-resolution single-excitation" if study.mode == "preview" else "full-resolution"
    return (f"The study varied {_join(plist)} using {kinds[d.kind]} (design seed {d.seed}). Each design point was "
            f"simulated {study.replicates} time(s) in {mode} mode with Monte Carlo seeds base+0…base+{study.replicates - 1}. "
            f"The recorded metrics were {mets}.")
