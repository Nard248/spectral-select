r"""Capture orchestration: experiment -> hyperspectral 4D cube (excitation x emission x H x W).

Pipeline per capture (see ``contracts.py`` for the radiometric definitions)::

    trace_scene()  -> HitBuffer (rays, layers, irradiance)        [tracer.py]
    material_optics() -> MaterialOptics per material               [optics.py]
    for each pixel chunk, for each excitation step, in sequence order:
        spectral radiance L(lam) of every component              [this module + transport kernels]
        project onto bands  ->  radiance / photo-electrons       [sensor.py]
    per step: subsurface spread -> optical blur -> detector -> ADU [sensor.py]

Components of the spectral radiance leaving a pixel (names as reported by :func:`explain_pixel`):

* ``dye:<f>``       fluorescence ``L_k = E <mu_a,k>_line Phi_k e_k(lam) G(lam) b_k V`` (+ FRET-sensitised
  emission of acceptors); ``E`` includes drifted power, beam irradiance, shadow transmission and
  Beer-Lambert absorption by clear media in front; ``b_k`` is the photobleaching survival; ``V`` the
  view-path transmission through clear layers in front (``T_view`` x Beer).
* ``secondary:<f>`` re-emission of fluorescence reabsorbed by dye ``f`` (one generation).
* ``elastic``       diffuse remission ``E rho line(lam)``.
* ``glint``         specular reflection ``glint_per_mW P line(lam)``.
* ``raman``         water Raman band for aqueous media.
* ``leak_source``   excitation-filter leakage (out-of-band source continuum) reflected/glinting.
* ``background:<material>``  everything diffuse emitted by the background slab.
* ``dye:<f>@<layer>``, ``raman@<layer>``, ``elastic``, ``glint``, ``leak_source``  pixels of a
  ``layered_block`` (3D Monte Carlo, see :mod:`spectraforge.studio.block_capture`).
* ``room_light``    ambient light (elastic + induced fluorescence).

Layered blocks (v1.1)
---------------------
Pixels that see a ``layered_block`` are not evaluated by the 1-D kernels: before the radiance stage the
engine runs the 3D Monte Carlo of every visible block (:func:`block_capture.get_runs`; excitation pass per
step, photobleaching/quenching/FRET/Raman source volumes, emission pass per spectral node) and reads the face
tallies at each pixel's hit point. Results are cached (small LRU keyed by the experiment JSON, mode, photon
fraction and step set) so ``explain_pixel`` / ``preview`` do not repeat the simulation. Documented
approximations: ``subsurface_spread`` is inherent in 3D (ignored for blocks), ``secondary_emission`` is not
modelled for blocks, room light is not applied to block pixels, one camera direction per block (acceptance
cone), clear objects in front of a block attenuate through the usual view transmission ``V`` and the
launch's Fresnel/Beer factors. Preview (``scale < 1``) uses ``volume_mc.preview_fraction`` of the photons.
Extra ground-truth maps (NaN off-block): ``layer_thickness:<layer>``, ``layer_depth:<layer>`` (depth of that
layer's TOP interface below the block's top surface), ``block_face`` (0 top, 1 +x, 2 -x, 3 +z, 4 -z),
``origin_depth_mm``, ``origin_layer_fraction:<layer>``; ``conc:<dye>`` / ``bleached:<dye>`` are NaN there.

Progress messages (stable format, parsed by the web UI)
-----------------------------------------------------
``capture(..., progress=fn)`` calls ``fn(fraction, message)`` with ``fraction`` monotonically
non-decreasing in ``[0, 1]`` and a ``message`` that is exactly one of::

    "tracing rays"
    "simulating <ex> nm (<i>/<n>) · radiance"     # radiance stage of excitation i of n (1-based)
    "sensor <ex> nm (<i>/<n>)"                    # optical blur + detector of excitation i of n
    "simulating <ex> nm (<i>/<n>) · 3D excitation"  # layered blocks only, before the radiance stage
    "emission node <lam> nm (<j>/<m>) · 3D"         # layered blocks only
    "done"

``<ex>`` is the excitation wavelength formatted with ``f"{ex:g}"`` (``365``, ``488``, ``532.5``) and
``·`` is U+00B7 with one space on each side.  Regex: ``^simulating ([\d.]+) nm \((\d+)/(\d+)\) · radiance$``
and ``^sensor ([\d.]+) nm \((\d+)/(\d+)\)$``.  Stages are sequential (all steps are simulated first, in
sequence order, then all sensor stages), so the fraction budget is 0-0.08 tracing/setup, 0.08-0.70
radiance, 0.70-0.99 sensor (with blocks: 0.08-0.52 3D Monte Carlo, 0.52-0.70 radiance).

Approximations (all documented where they occur)
------------------------------------------------
* Excitation is evaluated at the nominal line centre for transport kernels; absorption is averaged
  over the line shape (factorised: ``E`` Beer-averaged over the line, ``<mu_a>`` separately).
* Leakage uses the elastic ``rho`` of the line centre.
* Secondary emission: one generation, isotropic re-emission; the mean emission path is the
  fluorescence-weighted mean origin distance to the exit surface, the same rule for every geometry
  (closed-form depth profile for front-lit clear media, cell-integrated quadrature for side-lit clear
  media, kernel mean depth for turbid media), see :func:`_secondary_emission`.
* Side-lit clear media (laser through a cuvette, viewed at 90 deg): the tracer's volume samples are
  integrated cell by cell exactly (:meth:`_Engine._volumetric_G`) and the Beer-Lambert absorption of the
  clear media crossed before the cuvette (cuvettes in a row) is applied from a ray march done here
  (:meth:`_Engine._prior_occluder_paths`).  The ground-truth ``irradiance@ex`` of such pixels is the
  chord-mean beam flux density inside the medium.
* Specular glint warning: per fluorescent object pixel the band reached by the specular reflection is
  compared with the brightest fluorescence band (:func:`_glint_stats`).
* Photobleaching: dose at the illuminated surface (collimated intensity ``E / cos(theta_i')``,
  times ``1 + 2 pi rho`` surface build-up in turbid media), the *whole* sequence exposure
  ``n_bands x exposure`` per step, survival averaged over the step; no within-step spectral drift.
* The pixel's state (bleaching) is local, so sequential per-pixel recomputation in
  :func:`explain_pixel` is exact (except for blur, subsurface spread and fixed-pattern noise).
* Preview/scale < 1: the photo-electron count per output pixel is unchanged (a coarser sampling of
  the same detector), blur sigma is scaled to output pixels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from spectraforge.artifacts import raman_center_nm
from spectraforge.studio import optics as optics_mod
from spectraforge.studio import sensor as sensor_mod
from spectraforge.studio import source as srcmod
from spectraforge.studio import spectra
from spectraforge.studio.contracts import MAX_LAYERS, SPECTRAL_GRID_NM, CaptureResult, HitBuffer, ProgressFn
from spectraforge.studio.schema import (
    EXTINCTION_REFERENCE_M_CM,
    Background,
    BoxShape,
    ExcitationStep,
    Experiment,
    LightSourceSpec,
    PhysicsToggles,
    SceneObject,
    Transform,
)

GRID = SPECTRAL_GRID_NM
_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
CHUNK = 4096

#: static ground-truth keys; dynamic families: ``irradiance@<ex>``, ``conc:<dye>``,
#: ``bleached:<dye>``, ``absorbed_photons:<dye>@<ex>``  (``<ex>`` formatted with ``f"{ex:g}"``).
GROUND_TRUTH_KEYS = ("material", "object", "depth_mm", "cos_light", "cos_view", "path_mm",
                     "thickness_mm", "shadow")

# Raman: water O-H stretch, differential cross-section at 488 nm (Walrafen; Faris & Copeland 1997)
RAMAN_CROSS_SECTION_488_CM2_SR = 8e-30
RAMAN_FWHM_CM = 480.0
WATER_MOLECULES_PER_MM3 = 3.343e19
# room light: kernel evaluation bins (centres, nm)
ROOM_BIN_CENTRES = np.arange(310.0, 751.0, 40.0)
ROOM_BIN_HALF = 20.0
ROOM_COS_IN = 2.0 / 3.0       # mean incidence cosine of a cosine-weighted hemisphere
_SIGMA_CM2_PER_EPS = np.log(10.0) * 1000.0 / 6.02214076e23   # absorption cross-section per unit eps
_MODE_ID = {"sample": 0, "dark": 1, "white_reference": 2}


# --------------------------------------------------------------------------------------------
# source model (spectral parts come from source.py)
# --------------------------------------------------------------------------------------------
def _line_shape(src: LightSourceSpec, ex_nm: float, grid: np.ndarray = GRID) -> np.ndarray:
    """Normalised (integral 1, 1/nm) spectral line of the source at ``ex_nm``."""
    return srcmod.line_shape(src, ex_nm, grid)


def _leak_spectrum(src: LightSourceSpec, ex_nm: float, grid: np.ndarray = GRID) -> np.ndarray:
    """Out-of-band leakage of the clean-up filter per unit in-line power (1/nm)."""
    return srcmod.leakage_spectrum(src, ex_nm, grid)


def _white_continuum(src: LightSourceSpec, grid: np.ndarray = GRID) -> np.ndarray:
    """Broadband white illumination used for the white reference (integral 1 over 380-800 nm)."""
    x = srcmod.xenon_relative_spectrum(grid) if src.lamp_curve == "xenon" else np.ones_like(grid)
    x = x * ((grid >= 380.0) & (grid <= 800.0))
    return x / (x.sum() * float(grid[1] - grid[0]))


def _step_powers(exp: Experiment, mode: str, enabled_steps: list[ExcitationStep]) -> dict[int, float]:
    """Delivered (drifted) power per enabled step, keyed by ``id(step)``; deterministic in (seed, mode)."""
    src, tg = exp.source(), exp.acquisition.physics
    rng = np.random.default_rng(np.random.SeedSequence([int(exp.acquisition.seed) & 0xFFFFFFFF, 0xD21F7,
                                                       _MODE_ID[mode]]))
    pw = srcmod.drifted_powers(src, enabled_steps, rng, enabled=tg.power_drift)
    return {id(s): p for s, p in zip(enabled_steps, pw)}


def _raman_shape(ex_nm: float, grid: np.ndarray = GRID) -> np.ndarray | None:
    """Raman band density (1/nm, integral 1): Gaussian in wavenumber, FWHM 480 cm^-1 (~16 nm)."""
    centre = raman_center_nm(ex_nm)
    if not (grid[0] + 20 < centre < grid[-1] - 20):
        return None
    nu = 1e7 / grid
    s = RAMAN_FWHM_CM * _FWHM_TO_SIGMA
    g = np.exp(-0.5 * ((nu - 1e7 / centre) / s) ** 2) * (1e7 / grid ** 2)
    return g / (g.sum() * float(grid[1] - grid[0]))


def raman_mu_phi_per_mm(ex_nm: float) -> float:
    """Raman "mu_a * Phi" equivalent (1/mm) of water: ``n 4 pi sigma_R(ex) * (lam_ex / lam_R)``.

    ``sigma_R`` scales as ``lam^-4`` from 8e-30 cm^2/sr at 488 nm; the factor ``lam_ex/lam_R`` converts
    photon yield to energy yield.  Multiplying by ``E G`` gives the Raman radiance (same formalism
    as fluorescence, where ``mu_a Phi`` is the energy-converted emission rate per volume)."""
    sig_mm2 = RAMAN_CROSS_SECTION_488_CM2_SR * (488.0 / ex_nm) ** 4 * 100.0
    return WATER_MOLECULES_PER_MM3 * 4.0 * np.pi * sig_mm2 * ex_nm / raman_center_nm(ex_nm)


# --------------------------------------------------------------------------------------------
# small physics helpers (also used directly by tests)
# --------------------------------------------------------------------------------------------
def fluorescence_spectrum(E, mua_line, phi, emission_density, G):
    """``L(lam) = E <mu_a> Phi e(lam) G(lam)`` -- the contract's fluorescence radiance.

    ``E`` (mW/mm^2) scalar or (n,), ``G`` (W,) or (n, W) mm/sr, ``emission_density`` (W,) 1/nm."""
    E = np.asarray(E, dtype=float)
    amp = E * mua_line * phi
    return (amp[..., None] if amp.ndim else amp) * emission_density * G


def dye_spectra(opt, line: np.ndarray, E, G) -> dict[str, np.ndarray]:
    """Noise-free per-dye spectral radiance of a uniform medium (no FRET / bleaching / secondary).

    The *internal* function behind the dilute-limit equivalence with the legacy 2-D engine."""
    out = {}
    for dye, mua in opt.mua_dyes_per_mm.items():
        mua_line = optics_mod.average_over_line(mua, line)
        out[dye] = fluorescence_spectrum(E, mua_line, opt.quantum_yields[dye], opt.emission_density[dye], G)
    return out


def optics_mod_mut(opt):
    return opt.mua_per_mm + opt.mus_per_mm


def _line_samples(line: np.ndarray, n: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature of the line shape: ``n`` grid indices and weights (sum 1) for Beer averaging."""
    cdf = np.cumsum(line)
    cdf = cdf / cdf[-1]
    idx = np.unique(np.searchsorted(cdf, (np.arange(n) + 0.5) / n))
    idx = np.clip(idx, 0, len(line) - 1)
    w = line[idx]
    return idx, w / w.sum()


def _subset_hit(hit: HitBuffer, pix: np.ndarray) -> HitBuffer:
    """HitBuffer restricted to pixel indices ``pix`` (layer arrays are (K, P, ...), pixel arrays (P, ...))."""
    layer = {"material", "object_index", "point_mm", "normal", "path_mm", "cos_view", "cos_view_internal",
             "cos_light_internal", "cos_light", "thickness_mm", "T_view", "T_light", "n_outside",
             "irradiance_per_mW", "light_clear_path_mm"}
    pixel = {"depth_mm", "glint_per_mW", "pixel_footprint_mm", "view_dir", "ray_dir", "field_angle_rad"}
    kw = {}
    for name in layer:
        kw[name] = np.asarray(getattr(hit, name))[:, pix]
    for name in pixel:
        kw[name] = np.asarray(getattr(hit, name))[pix]
    for name in ("vol_s_mm", "vol_weight_mm", "vol_fluence_per_mW", "vol_light_path_mm"):
        arr = getattr(hit, name, None)
        kw[name] = None if arr is None else np.asarray(arr)[:, pix]
    arr = getattr(hit, "vol_T_out", None)
    kw["vol_T_out"] = None if arr is None else np.asarray(arr)[:, pix]
    info = dict(hit.info)
    if "block_hits" in info:                     # remap pixel positions into the subset
        inv = np.full(hit.material.shape[1], -1, dtype=np.int64)
        inv[pix] = np.arange(len(pix))
        nb = {}
        for oi, d in info["block_hits"].items():
            newp = inv[d["pixel"]]
            ok = newp >= 0
            if ok.any():
                nb[oi] = {k_: (newp[ok] if k_ == "pixel" else np.asarray(v_)[ok]) for k_, v_ in d.items()}
        info["block_hits"] = nb
    return HitBuffer(width=hit.width, height=hit.height, focus_depth_mm=hit.focus_depth_mm,
                     info=info, **kw)


# --------------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------------
@dataclass
class _Step:
    index: int
    ex_nm: float
    power_mW: float
    exposure_ms: float
    centres: np.ndarray
    responses: sensor_mod.BandResponses
    wsel: np.ndarray
    line: np.ndarray
    leak: np.ndarray
    raman: np.ndarray | None
    beer_idx: np.ndarray
    beer_w: np.ndarray
    rad_d: np.ndarray = None
    el_d: np.ndarray = None
    rad_s: np.ndarray = None
    el_s: np.ndarray = None
    comps: dict | None = None
    irr: np.ndarray = None                     # (P,) E at the first surface (mW/mm^2)
    dose: dict = field(default_factory=dict)   # dye -> (P,) absorbed photons per molecule (layer 0)
    radial: dict = field(default_factory=dict)  # material idx -> radial kernel tuple
    ring: np.ndarray = None                    # (P,) field-angle ring index
    wsel_idx: np.ndarray = None                # indices of the step's spectral window on the fine grid
    fl_b: np.ndarray = None                    # (P, B) fluorescence-only band radiance (glint metric)


class _Engine:
    """Computes band radiance / electrons for a HitBuffer over a list of excitation steps."""

    def __init__(self, experiment: Experiment, hit: HitBuffer, transport, mode: str):
        self.exp = experiment
        self.cam = experiment.camera()
        self.src = experiment.source()
        self.tg: PhysicsToggles = experiment.acquisition.physics
        self.hit = hit
        self.transport = transport
        self.mode = mode
        self.P = hit.material.shape[1]
        self.K = hit.material.shape[0]
        self.fl = optics_mod.merged_fluorophores(experiment)
        self.mats = experiment.materials
        self.optics = [optics_mod.material_optics(m, self.fl, GRID, self.tg) for m in self.mats]
        self.mua_all = np.stack([o.mua_per_mm for o in self.optics]).astype(np.float64) if self.optics \
            else np.zeros((0, len(GRID)))
        dyes = sorted({d for o in self.optics for d in o.mua_dyes_per_mm})
        self.dyes = dyes
        self.survival = {d: np.ones((self.K, self.P), dtype=np.float32) for d in dyes}
        self.phi_b = {d: self.fl[d].bleach_quantum_yield for d in dyes}
        self.eps_line_cache: dict[tuple, float] = {}
        self._keep = False
        self._em_ok: bool | None = None
        self._ek_cache: dict[tuple, np.ndarray] = {}
        self.room_cols: np.ndarray | None = None     # union spectral window the room-light cache keeps
        self._prior_cache: dict[int, np.ndarray] = {}
        self._scene = None
        self.blocks: list = []                       # block_capture.BlockRun (layered blocks, 3D MC)
        self.mat_eff = hit.material                  # material per layer slot with block pixels masked (-1)

    def attach_blocks(self, runs) -> None:
        """Register the 3D block runs: their pixels bypass the 1-D kernels (see block_capture)."""
        import copy
        # shallow copies: the cached runs are shared between captures / previews / explain calls (possibly
        # on different threads) and ``attach_hits`` stores the per-capture pixel look-up on the object
        self.blocks = [copy.copy(r) for r in runs]
        K, P = self.hit.material.shape
        eff = self.hit.material.copy()
        for r in self.blocks:
            r.attach_hits(self.hit.info, K, P)
            eff[r.mask] = -1
        self.mat_eff = eff

    # ---------------------------------------------------------------- transport helper
    def _kernels(self, opt, ex_nm, k, gidx, cos_in=None, em_index=None):
        """Transport kernels; ``em_index`` (indices into the fine grid) restricts G to those columns when
        the transport model supports it (``kernels(..., em_index=)``), else the full grid is returned."""
        h = self.hit
        kw = {}
        if em_index is not None and self._em_supported():
            kw["em_index"] = em_index
        return self.transport.kernels(
            opt, float(ex_nm), h.cos_light[k, gidx] if cos_in is None else cos_in, h.cos_view[k, gidx],
            h.thickness_mm[k, gidx], n_outside=h.n_outside[k, gidx], path_mm=h.path_mm[k, gidx], **kw)

    def _em_supported(self) -> bool:
        if self._em_ok is None:
            import inspect
            try:
                self._em_ok = "em_index" in inspect.signature(self.transport.kernels).parameters
            except (TypeError, ValueError):
                self._em_ok = False
        return self._em_ok

    def _view_transmission(self, k, gidx, wsel) -> np.ndarray:
        """``V(lam)`` for layer k: T_view x Beer through the clear layers in front."""
        h = self.hit
        v = np.repeat(h.T_view[k, gidx][:, None].astype(np.float32), int(wsel.sum()), axis=1)
        if k > 0 and self.tg.absorption:
            tau = np.zeros_like(v)
            mua_w = self.mua_all[:, wsel]
            for j in range(k):
                mj = h.material[j, gidx]
                ok = mj >= 0
                if ok.any():
                    tau[ok] += (mua_w[mj[ok]] * h.path_mm[j, gidx][ok, None]).astype(np.float32)
            v = v * np.exp(-tau)
        return v

    def _skippable(self, opt, mspec, white: bool) -> bool:
        """Layers that cannot emit anything: dye-less clear media (glass) that are not aqueous."""
        if white:
            return False
        has_dyes = bool(opt.mua_dyes_per_mm) and self.tg.fluorescence
        has_raman = mspec.aqueous and self.tg.raman
        has_el = (not opt.clear) and self.tg.elastic_leak
        return not (has_dyes or has_raman or has_el)

    # ---------------------------------------------------------------- one pixel chunk x one step
    def step_chunk(self, st: _Step, a: int, b: int, room: np.ndarray | None, keep: bool) -> None:
        h, tg = self.hit, self.tg
        n = b - a
        ws = st.wsel
        wn = int(ws.sum())
        spec_d = np.zeros((n, wn), dtype=np.float32)
        spec_s = np.zeros((n, wn), dtype=np.float32)
        spec_f = np.zeros((n, wn), dtype=np.float32) if st.fl_b is not None else None
        comps: dict[str, np.ndarray] = {}
        line_w, leak_w = st.line[ws], st.leak[ws]
        P = st.power_mW
        white = self.mode == "white_reference"

        self._keep = keep

        def add(name: str, arr: np.ndarray, sel: np.ndarray, specular: bool = False, fl_arr=None):
            (spec_s if specular else spec_d)[sel] += arr
            if spec_f is not None:
                if fl_arr is not None:
                    spec_f[sel] += fl_arr
                elif name[:4] in ("dye:", "seco"):
                    spec_f[sel] += arr
            if keep:
                c = comps.setdefault(name, np.zeros((n, wn), dtype=np.float32))
                c[sel] += arr

        if room is not None:
            room_ws = room[:, ws[self.room_cols]] if self.room_cols is not None else room[:, ws]
            spec_d += room_ws
            if keep:
                comps["room_light"] = room_ws.copy()

        if P > 0:
            for k in range(self.K):
                matk = self.mat_eff[k, a:b]
                isbg = h.object_index[k, a:b] == -2
                for m in np.unique(matk[matk >= 0]):
                    for flag in (False, True):
                        sel = np.nonzero((matk == m) & (isbg == flag))[0]
                        if sel.size:
                            with np.errstate(all="ignore"):     # Accelerate BLAS raises spurious FP flags
                                self._layer(st, k, int(m), a + sel, sel, line_w, leak_w, ws, add, white, flag)
            if self.blocks:
                self._block_chunk(st, a, b, line_w, leak_w, ws, add)
            # specular glint at the first interface
            if tg.specular_glints:
                m0 = h.material[0, a:b] >= 0
                sel = np.nonzero(m0)[0]
                if sel.size:
                    g = (h.glint_per_mW[a + sel] * P * h.T_light[0, a + sel]).astype(np.float32)
                    add("glint", g[:, None] * line_w[None, :], sel, specular=True)
                    if tg.elastic_leak and not white:
                        add("leak_source", g[:, None] * leak_w[None, :], sel, specular=True)

        rd, ed = sensor_mod.project_bands(spec_d, st.responses, st.ring[a:b], ws)
        rs, es = sensor_mod.project_bands(spec_s, st.responses, st.ring[a:b], ws)
        st.rad_d[a:b], st.el_d[a:b], st.rad_s[a:b], st.el_s[a:b] = rd, ed, rs, es
        if spec_f is not None:
            st.fl_b[a:b], _ = sensor_mod.project_bands(spec_f, st.responses, st.ring[a:b], ws)
        if keep:
            if st.comps is None:
                st.comps = {}
            for name, arr in comps.items():
                r, _ = sensor_mod.project_bands(arr, st.responses, st.ring[a:b], ws)
                if name not in st.comps:
                    st.comps[name] = np.zeros((self.P, len(st.centres)), dtype=np.float32)
                st.comps[name][a:b] = r

    def _block_chunk(self, st: _Step, a: int, b: int, line_w, leak_w, ws, add) -> None:
        """Radiance of the layered-block pixels of the chunk from the 3D Monte Carlo face tallies."""
        h, tg = self.hit, self.tg
        for run in self.blocks:
            for k in range(self.K):
                msk = run.mask[k, a:b]
                if not msk.any():
                    continue
                sel = np.nonzero(msk)[0]
                gidx = a + sel
                if k == 0:       # ground-truth surface irradiance of the first surface
                    st.irr[gidx] = st.power_mW * h.irradiance_per_mW[0, gidx].astype(np.float64) * h.T_light[0, gidx]
                V = self._view_transmission(k, gidx, ws)
                parts = run.components(st.ex_nm, run.face_code[k, gidx], run.hit_u[k, gidx], run.hit_v[k, gidx],
                                       st.wsel_idx, line_w, leak_w, self._keep, tg.fluorescence, tg.elastic_leak)
                if self._keep:
                    for name, arr, _fl in parts:
                        add(name, (arr * V).astype(np.float32), sel)
                else:
                    tot = np.zeros((len(sel), line_w.size), np.float32)
                    fl = np.zeros_like(tot)
                    for _n, arr, is_fl in parts:
                        tot += arr
                        if is_fl:
                            fl += arr
                    add("layer", tot * V, sel, fl_arr=fl * V)

    def _layer(self, st: _Step, k: int, m: int, gidx: np.ndarray, sel: np.ndarray, line_w, leak_w, ws,
               add, white: bool, bg: bool) -> None:
        h, tg = self.hit, self.tg
        opt, mspec = self.optics[m], self.mats[m]
        if self._skippable(opt, mspec, white):
            return
        label = (lambda base: f"background:{mspec.id}") if bg else (lambda base: base)
        P = st.power_mW
        # ---- excitation irradiance at the entry point (mW/mm^2), Beer over clear media in front
        e0 = P * h.irradiance_per_mW[k, gidx].astype(np.float64) * h.T_light[k, gidx]
        if tg.absorption and h.light_clear_path_mm.shape[2] > 0:
            cp = h.light_clear_path_mm[k, gidx]                                  # (n, M)
            if np.any(cp > 0):
                with np.errstate(all="ignore"):   # spurious BLAS FP flags
                    tau = cp @ self.mua_all[:, st.beer_idx]                      # (n, S)
                    e0 = e0 * (np.exp(-tau) @ st.beer_w)
        E = e0
        if k == 0:
            st.irr[gidx] = E
        V = self._view_transmission(k, gidx, ws)
        em_idx = st.wsel_idx
        kern = self._kernels(opt, st.ex_nm, k, gidx, em_index=em_idx)
        if k == 0 and kern.radial_profile is not None and m not in st.radial:
            st.radial[m] = (kern.radial_r_mm, kern.radial_profile, kern.radial_lambda_nm)
        G = np.asarray(kern.fluorescence_G, dtype=np.float32)
        if G.shape[1] != em_idx.size:           # transport without em_index support: full grid -> window
            G = G[:, ws]
        E_fl, E_bl = E, E
        vp = self._volumetric_mask(opt, k, gidx) if not white else None
        if vp is not None and vp.any():
            G = G.copy()
            G[vp] = self._volumetric_G(opt, st, k, gidx[vp], ws)
            # side-lit clear media: the tracer's quadrature already carries the beam fluence and the
            # Fresnel T of every crossed interface; Beer-Lambert absorption by the clear media the
            # beam crossed *before* reaching this medium (cuvettes in series) is applied here.
            beer = np.ones(len(gidx))
            if tg.absorption:
                prior = self._prior_occluder_paths(k)
                if prior is not None:
                    with np.errstate(all="ignore"):   # spurious BLAS FP flags
                        tau = prior[gidx] @ self.mua_all[:, st.beer_idx]             # (n, S)
                        beer = np.exp(-tau) @ st.beer_w
            E_fl = np.where(vp, P * beer, E)           # G_vol already contains the beam fluence
            wq = h.vol_weight_mm[k, gidx] * h.vol_fluence_per_mW[k, gidx]
            fmean = wq.sum(1) / np.maximum(h.vol_weight_mm[k, gidx].sum(1), 1e-30)
            E_vol = P * beer * fmean                   # chord-mean beam flux density in the medium
            E_bl = np.where(vp, E_vol * np.maximum(h.cos_light_internal[k, gidx], 0.05), E)
            if k == 0:                                 # ground-truth irradiance of side-lit media
                st.irr[gidx[vp]] = E_vol[vp]
        GV = G * V
        rho = np.asarray(kern.elastic_rho, dtype=np.float64)
        n = len(gidx)

        keep = self._keep
        wn = int(ws.sum())
        # non-keep fast path: every component of the layer is accumulated into one (n, wn) array and written
        # to the chunk spectrum with a single fancy-index add; ``keep`` (explain_pixel) records each one.
        acc = None if keep else np.zeros((n, wn), dtype=np.float32)
        acc_f = np.zeros((n, wn), dtype=np.float32) if (not keep and st.fl_b is not None) else None

        def put(name: str, arr: np.ndarray, fluo: bool = False) -> None:
            if keep:
                add(name, arr, sel)
            else:
                acc.__iadd__(arr)
                if fluo and acc_f is not None:
                    acc_f.__iadd__(arr)

        # ---- elastic remission and leakage
        if tg.elastic_leak:
            if white:
                rho_w = self._rho_spectrum(opt, k, gidx, None)[:, ws]
                put(label("elastic"), (E[:, None] * rho_w * line_w[None, :] * V).astype(np.float32))
            elif keep:
                put(label("elastic"), ((E * rho)[:, None] * line_w[None, :] * V).astype(np.float32))
                put("leak_source", ((E * rho)[:, None] * leak_w[None, :] * V).astype(np.float32))
            else:      # line + leakage share the same factor E rho V
                put("elastic", ((E * rho)[:, None] * (line_w + leak_w)[None, :] * V).astype(np.float32))

        # ---- fluorescence
        if tg.fluorescence and opt.mua_dyes_per_mm:
            dyes = list(opt.mua_dyes_per_mm)
            col = {d: i for i, d in enumerate(dyes)}
            mua_line = {d: optics_mod.average_over_line(a_, st.line) for d, a_ in opt.mua_dyes_per_mm.items()}
            b_eff = self._bleach(st, k, m, gidx, opt, mspec, E_bl, rho, kern)
            A = np.empty((n, len(dyes)), dtype=np.float64)       # per-dye emission amplitude E <mu_a> Phi b
            for d in dyes:
                donor = np.ones(n)
                if tg.fret:
                    conc = {dd.fluorophore: dd.concentration_uM for dd in mspec.dyes}
                    for pair in mspec.fret:
                        if pair.donor == d and pair.acceptor in mua_line:
                            kc = pair.k_per_uM * conc.get(pair.acceptor, 0.0)
                            donor = donor * (1.0 - kc / (1.0 + kc))
                A[:, col[d]] = E_fl * mua_line[d] * opt.quantum_yields[d] * b_eff[d] * donor
            if tg.fret:        # FRET-sensitised acceptor emission (donor energy absorbed, transferred)
                conc = {dd.fluorophore: dd.concentration_uM for dd in mspec.dyes}
                for pair in mspec.fret:
                    if pair.donor in mua_line and pair.acceptor in mua_line:
                        kc = pair.k_per_uM * conc.get(pair.acceptor, 0.0)
                        eff = kc / (1.0 + kc)
                        A[:, col[pair.acceptor]] += (E_fl * mua_line[pair.donor] * eff * b_eff[pair.donor]
                                                     * opt.quantum_yields[pair.acceptor])
            EK = self._emission_matrix(opt, m, dyes, ws)                       # (D, wn) float32
            A32 = A.astype(np.float32)
            fl = (A32 @ EK) * GV                                               # (n, wn) primary fluorescence
            if keep:
                for d in dyes:
                    put(label(f"dye:{d}"), (A32[:, col[d], None] * EK[col[d]][None, :]) * GV)
            else:
                put("dye", fl, fluo=True)
            if tg.secondary_emission and tg.absorption:
                sec = self._secondary_emission(opt, mspec, fl, b_eff, kern, k, gidx, ws, st, vp, dyes)
                if sec is not None:
                    coef, expneg = sec                                             # (n, D), (n, wn)
                    if keep:
                        for d in dyes:
                            put(label(f"secondary:{d}"), (coef[:, col[d], None] * EK[col[d]][None, :]) * expneg)
                    else:
                        put("secondary", (coef @ EK) * expneg, fluo=True)

        # ---- Raman
        if mspec.aqueous and tg.raman and st.raman is not None:
            amp = E_fl * raman_mu_phi_per_mm(st.ex_nm) * mspec.raman_strength
            put(label("raman"), (amp[:, None] * st.raman[ws][None, :]).astype(np.float32) * GV)

        if not keep:
            add("layer", acc, sel, fl_arr=acc_f)

    # ---------------------------------------------------------------- volumetric (side-lit clear media)
    def _volumetric_mask(self, opt, k, gidx):
        h = self.hit
        if not opt.clear or getattr(h, "vol_weight_mm", None) is None or h.vol_fluence_per_mW is None:
            return None
        return (h.vol_weight_mm[k, gidx] * h.vol_fluence_per_mW[k, gidx]).sum(axis=1) > 0

    def _volumetric_G(self, opt, st, k, gidx, ws) -> np.ndarray:
        """Equivalent kernel (mm/sr, per unit source power) from the tracer's volume samples:
        ``G = T_out/(4 pi n^2) * integral F(s) exp(-mu_t,ex l(s)) exp(-mu_t(lam) s) ds`` over the view chord.

        The tracer samples ``F`` (beam fluence) and ``l`` (excitation path inside the medium) at ``Q``
        Gauss-Legendre nodes ``s_q`` with weights ``w_q``.  A plain point rule ``sum w_q f(s_q)`` fails
        when the exponentials decay faster than the node spacing (strongly absorbing front-lit media:
        the first node sits ~0.2 mm deep for a 30 mm chord, ~-70 % error at 1 mM fluorescein).  Instead
        each node's cell ``[a_q, b_q]`` (cumulative weights) is integrated *exactly* with ``F`` constant and
        ``l(s)`` linear (slope from the neighbouring nodes): with ``E(s) = -mu_t,ex l(s) - mu_t(lam) s`` the cell
        integral is ``F_q (e^{E_a} - e^{E_b}) / (-(dE/ds))``.  This is the point rule in the optically thin
        limit and the correct ``1/kappa`` surface layer in the thick limit."""
        h = self.hit
        mut_ex = float(opt.interp(optics_mod_mut(opt), st.ex_nm))
        mut_em = optics_mod_mut(opt)[ws].astype(np.float32)
        wn = int(ws.sum())
        out = np.zeros((len(gidx), wn), dtype=np.float32)
        pre = h.vol_T_out[k, gidx] / opt.n ** 2 / (4.0 * np.pi)
        s_ = h.vol_s_mm[k, gidx].astype(np.float64)
        w_ = h.vol_weight_mm[k, gidx].astype(np.float64)
        F_ = h.vol_fluence_per_mW[k, gidx].astype(np.float64)
        l_ = h.vol_light_path_mm[k, gidx].astype(np.float64)
        Q = s_.shape[1]
        a_ = np.cumsum(w_, axis=1) - w_
        b_ = a_ + w_
        slope = np.zeros_like(s_)
        if Q > 1:
            ds, dl = np.diff(s_, axis=1), np.diff(l_, axis=1)
            sg = np.where(ds > 1e-12, dl / np.maximum(ds, 1e-12), 0.0)
            slope[:, 0], slope[:, -1] = sg[:, 0], sg[:, -1]
            if Q > 2:
                slope[:, 1:-1] = 0.5 * (sg[:, :-1] + sg[:, 1:])
        for q in range(Q):
            ok = (w_[:, q] * F_[:, q]) > 0
            if not ok.any():
                continue
            la = np.maximum(l_[ok, q] + slope[ok, q] * (a_[ok, q] - s_[ok, q]), 0.0)
            lb = np.maximum(l_[ok, q] + slope[ok, q] * (b_[ok, q] - s_[ok, q]), 0.0)
            Ea = (-mut_ex * la)[:, None] - mut_em[None, :] * a_[ok, q, None]
            Eb = (-mut_ex * lb)[:, None] - mut_em[None, :] * b_[ok, q, None]
            x = Ea - Eb
            dx = np.abs(x) < 1e-4
            val = np.where(dx, w_[ok, q, None] * np.exp(0.5 * (Ea + Eb)),
                           (np.exp(Ea) - np.exp(Eb)) * (w_[ok, q, None] / np.where(dx, 1.0, x)))
            out[ok] += ((F_[ok, q] * pre[ok]).astype(np.float32)[:, None] * val.astype(np.float32))
        return out

    # ---------------------------------------------------------------- beam attenuation by prior clear media
    def _scene_geometry(self):
        if self._scene is None:
            from spectraforge.studio.geometry import build_scene
            self._scene = build_scene(self.exp)
        return self._scene

    def _prior_occluder_paths(self, k: int) -> np.ndarray | None:
        """Per-material path length (mm) the beam travelled through *other* clear media before it
        reached the mid-chord point of every side-lit clear layer-``k`` pixel -> ``(P, M)`` or None.

        The tracer's volumetric samples carry the beam fluence and the Fresnel T of every crossing but
        not the Beer-Lambert absorption of media crossed before the sample's own cuvette (e.g. the first
        cuvette of a row absorbing the laser that then reaches the second).  The mid-chord point is
        rebuilt from the layer entry point and the Snell chain of the view ray (exact for planar
        parallel faces, first-order for curved ones); the straight line to the source aperture (as in the
        tracer) is marched through the exact solids and the chord inside every clear medium except the
        sample's own is accumulated.  Computed once per layer (geometry only, wavelength independent)."""
        if k in self._prior_cache:
            return self._prior_cache[k]
        h = self.hit
        out = None
        wf = None if getattr(h, "vol_weight_mm", None) is None or h.vol_fluence_per_mW is None \
            else (h.vol_weight_mm[k] * h.vol_fluence_per_mW[k]).sum(1)
        if wf is not None and wf.max() > 0 and self.mats and len(self.exp.objects) + 1 > 1:
            try:
                out = self._march_prior(k, np.nonzero(wf > 0)[0])
            except Exception:                    # pragma: no cover - never let a diagnostic kill a capture
                out = None
        self._prior_cache[k] = out
        return out

    def _march_prior(self, k: int, sel: np.ndarray) -> np.ndarray:
        from spectraforge.studio import surface
        from spectraforge.studio.geometry import EPS_T_MM
        h, M = self.hit, len(self.mats)
        scene = self._scene_geometry()
        mat_n = np.array([m.refractive_index for m in self.mats], float)
        solid_mat = np.array([sd.material_index for sd in scene.solids], dtype=np.int64)
        clear_m = np.array([o.clear for o in self.optics], bool)
        # view direction inside layer k (Snell chain over the layers 0..k)
        d_prev = np.asarray(h.ray_dir, float)[sel]
        d_in = d_prev
        for j in range(k + 1):
            mj = h.material[j, sel]
            ok = mj >= 0
            n2 = np.where(ok, mat_n[np.maximum(mj, 0)], 1.0)
            n1 = np.asarray(h.n_outside[j, sel], float)
            if j > 0:
                d_prev = np.where((n1 > 1.0 + 1e-9)[:, None], d_in, np.asarray(h.ray_dir, float)[sel])
            d_in, _ = surface.refract(d_prev, np.asarray(h.normal[j, sel], float), n1, n2)
        x = np.asarray(h.point_mm[k, sel], float) + d_in * (0.5 * h.path_mm[k, sel])[:, None]
        src = self.src
        pos = np.asarray(src.position_mm, float)
        # straight line to the beam aperture: same construction as the tracer's shadow rays
        from spectraforge.studio import source as _src
        bs = _src.beam_samples(src, x)
        u, dist = bs.u_to_source, bs.dist_mm
        n = len(sel)
        path = np.zeros((n, M), dtype=np.float32)
        t_prev = np.zeros(n)
        cur = scene.locate(x)
        act = np.arange(n)
        for it in range(48):
            if act.size == 0:
                break
            hh = scene.first_hit(x[act], u[act], t_prev[act] + EPS_T_MM)
            crossed = hh.found & (hh.t < dist[act])
            t_end = np.where(crossed, hh.t, dist[act])
            seg = np.maximum(t_end - t_prev[act], 0.0)
            c = cur[act]
            if it > 0:                                  # segment 0 is inside the sample's own medium
                m_ = np.where(c >= 0, solid_mat[np.maximum(c, 0)], -1)
                good = (m_ >= 0)
                good[good] = clear_m[m_[good]]
                np.add.at(path, (act[good], m_[good]), seg[good].astype(np.float32))
            ci = np.nonzero(crossed)[0]
            if ci.size == 0:
                break
            ids = act[ci]
            t_prev[ids] = hh.t[ci]
            cur[ids] = scene.locate(x[ids] + u[ids] * (hh.t[ci] + 2.0 * EPS_T_MM)[:, None])
            act = ids
        full = np.zeros((self.P, M), dtype=np.float32)
        full[sel] = path
        return full

    # ---------------------------------------------------------------- bleaching
    def _bleach(self, st: _Step, k, m, gidx, opt, mspec, E, rho, kern) -> dict[str, np.ndarray]:
        """Survival used for this step's fluorescence (mean over the exposure); updates the state."""
        out = {}
        h = self.hit
        active = self.tg.photobleaching and self.mode == "sample" and st.power_mW > 0
        for d in opt.mua_dyes_per_mm:
            s0 = self.survival[d][k, gidx].astype(np.float64)
            if not active:
                out[d] = s0
                continue
            key = (d, round(st.ex_nm, 3), st.index)
            eps = self.eps_line_cache.get(key)
            if eps is None:
                spec = self.fl[d]
                eps = spec.extinction * EXTINCTION_REFERENCE_M_CM * float(
                    optics_mod.average_over_line(optics_mod.to_legacy(spec).excitation(GRID), st.line))
                self.eps_line_cache[key] = eps
            cos_i = np.maximum(h.cos_light_internal[k, gidx], 0.05)
            intensity = E / cos_i                                              # mW/mm^2 along the beam
            build = 1.0 if opt.clear else np.clip(1.0 + 2.0 * np.pi * rho, 1.0, 6.0)
            photons_cm2_s = intensity * build * 100.0 * st.ex_nm * sensor_mod.PHOTONS_PER_MW_S_PER_NM
            t_total = st.exposure_ms * 1e-3 * len(st.centres)
            dose = _SIGMA_CM2_PER_EPS * eps * photons_cm2_s * t_total          # absorbed photons / molecule
            x = self.phi_b[d] * dose
            mean_f = np.where(x > 1e-9, -np.expm1(-x) / np.maximum(x, 1e-300), 1.0 - 0.5 * x)
            out[d] = s0 * mean_f
            self.survival[d][k, gidx] = (s0 * np.exp(-x)).astype(np.float32)
            if k == 0:
                st.dose.setdefault(d, np.zeros(self.P, dtype=np.float32))[gidx] = dose
        return out

    # ---------------------------------------------------------------- secondary emission
    def _secondary_emission(self, opt, mspec, total, b_eff, kern, k, gidx, ws, st, vp, dyes):
        """One-generation re-emission of fluorescence reabsorbed by dyes in the same medium.

        ``total`` (n, wn) is the primary fluorescence radiance that left the medium.  With
        ``tau(lam) = mu_a,tot(lam) * l`` (``l`` = fluorescence-weighted mean distance from the emission
        point to the exit surface along the view ray, see :meth:`_mean_emission_path`: the exact mean of the
        closed-form depth profile for front-lit clear media, the quadrature mean for side-lit (volumetric)
        clear media, the kernels' mean origin depth / cos(theta_o') for turbid media) the radiance that
        would have been seen without self-absorption is ``L / exp(-tau)``; the part absorbed by dye ``k``
        is ``L (e^tau - 1) mu_a,k / mu_a,tot``.  Dye ``k`` re-emits it with ``Phi_k`` and the Stokes factor
        ``lam_abs / lam_em`` (energy), shape ``e_k``, escaping with ``exp(-tau)``.

        Returns ``(coef (n, D), exp(-tau) (n, wn))`` -- the secondary radiance of dye ``d`` is
        ``coef[:, d] * e_d(lam) * exp(-tau)`` -- or None if negligible.  Two small GEMMs replace a loop
        over dyes of (n, wn) temporaries."""
        h = self.hit
        if opt.clear:
            ell = self._mean_emission_path(opt, st, k, gidx, ws, vp, total)
        elif kern.mean_depth_mm is not None:
            ell = np.asarray(kern.mean_depth_mm) / np.maximum(h.cos_view_internal[k, gidx], 0.2)
        else:
            mut_ex = float(opt.interp(opt.mut_per_mm, 400.0))
            ell = np.minimum(h.thickness_mm[k, gidx], 1.0 / max(mut_ex, 1e-6)) / np.maximum(
                h.cos_view_internal[k, gidx], 0.2)
        ell = np.clip(np.nan_to_num(ell, nan=0.0, posinf=100.0), 0.0, 100.0)
        mua_tot = opt.mua_per_mm[ws]
        if float(mua_tot.max()) * float(ell.max()) < 1e-7:
            return None
        tau = np.clip(mua_tot[None, :] * ell[:, None], 0.0, 4.0).astype(np.float32)
        X = total * np.expm1(tau)                                              # absorbed radiance per band
        lam = GRID[ws].astype(np.float32)
        frac = np.stack([(opt.mua_dyes_per_mm[d][ws] / mua_tot) for d in dyes], axis=1).astype(np.float32)  # (wn, D)
        a = X @ frac                                                           # (n, D) mW/(mm^2 sr)
        if not np.any(a > 0):
            return None
        lam_abs = ((X * lam[None, :]) @ frac) / np.maximum(a, 1e-30)           # (n, D)
        ek = np.stack([opt.emission_density[d][ws] for d in dyes])             # (D, wn)
        lam_em = (ek * GRID[ws][None, :]).sum(1) / np.maximum(ek.sum(1), 1e-30)
        coef = a * np.array([opt.quantum_yields[d] for d in dyes])[None, :] * np.stack(
            [b_eff[d] for d in dyes], axis=1) * (lam_abs / lam_em[None, :])
        return coef.astype(np.float32), np.exp(-tau)

    def _emission_matrix(self, opt, m: int, dyes: list[str], ws: np.ndarray) -> np.ndarray:
        """(D, wn) float32 emission densities on the step window (cached per material and window)."""
        key = (m, ws.tobytes().__hash__(), tuple(dyes))
        ek = self._ek_cache.get(key)
        if ek is None:
            ek = np.stack([opt.emission_density[d][ws] for d in dyes]).astype(np.float32)
            self._ek_cache[key] = ek
        return ek

    def _mean_emission_path(self, opt, st, k, gidx, ws, vp, prim_total) -> np.ndarray:
        """Mean distance (mm) from the fluorescence origin to the exit surface along the view chord of a
        CLEAR layer, weighted by the local emission rate -- one rule for both illumination geometries.

        * front-lit (excitation enters through the view face; closed-form kernel): the origin density is
          ``exp(-kappa s)`` on ``s in [0, L]`` with ``kappa = mu_t,ex cos(o')/cos(i') + mu_t,em`` (the
          exponent of the contract's closed form), whose mean is ``1/kappa - L e^{-kappa L}/(1 - e^{-kappa L})``
          (``L/2`` in the dilute limit);
        * side-lit (tracer volume quadrature): ``sum_q w_q F_q e^{-mu_t,ex l_q} e^{-mu_t,em s_q} s_q`` over the
          same weights without the factor ``s_q``.

        ``mu_t,em`` is evaluated at the per-pixel spectral centroid of the primary fluorescence."""
        h = self.hit
        mut = optics_mod_mut(opt)
        lam = GRID[ws].astype(np.float64)
        tot = np.asarray(prim_total, dtype=np.float64)
        cen = (tot * lam[None, :]).sum(1) / np.maximum(tot.sum(1), 1e-300)
        cen = np.where(tot.sum(1) > 0, cen, 550.0)
        mut_em = np.interp(cen, GRID, mut)
        mut_ex = float(np.interp(st.ex_nm, GRID, mut)) if st is not None else 0.0
        L = np.asarray(h.path_mm[k, gidx], dtype=np.float64)
        ratio = np.maximum(h.cos_view_internal[k, gidx], 0.05) / np.maximum(h.cos_light_internal[k, gidx], 0.05)
        kap = np.maximum(mut_ex * ratio + mut_em, 1e-12)
        x = kap * L
        with np.errstate(all="ignore"):
            ell = np.where(x > 1e-4, 1.0 / kap - L * np.exp(-x) / (-np.expm1(-x)), 0.5 * L * (1.0 - x / 6.0))
        if vp is not None and np.any(vp):
            vi = np.nonzero(vp)[0]
            g = gidx[vi]
            s_ = h.vol_s_mm[k, g].astype(np.float64)
            w_ = h.vol_weight_mm[k, g].astype(np.float64)
            F_ = h.vol_fluence_per_mW[k, g].astype(np.float64)
            l_ = h.vol_light_path_mm[k, g].astype(np.float64)
            a_ = np.cumsum(w_, axis=1) - w_
            b_ = a_ + w_
            slope = np.zeros_like(s_)
            if s_.shape[1] > 1:      # same exact cell integration as _volumetric_G (F const, l linear per cell)
                ds, dl = np.diff(s_, axis=1), np.diff(l_, axis=1)
                sg = np.where(ds > 1e-12, dl / np.maximum(ds, 1e-12), 0.0)
                slope[:, 0], slope[:, -1] = sg[:, 0], sg[:, -1]
                if s_.shape[1] > 2:
                    slope[:, 1:-1] = 0.5 * (sg[:, :-1] + sg[:, 1:])
            la = np.maximum(l_ + slope * (a_ - s_), 0.0)
            kap_q = mut_ex * slope + mut_em[vi, None]
            Ea = -mut_ex * la - mut_em[vi, None] * a_
            x = kap_q * w_
            small = np.abs(x) < 1e-4
            ksafe = np.where(small, 1.0, kap_q)
            ex_ = np.exp(-np.clip(x, -50, 50))
            z0 = np.where(small, w_, -np.expm1(-np.clip(x, -50, 50)) / ksafe)
            m1 = np.where(small, w_ * 0.5 * (a_ + b_),
                          (a_ / ksafe + 1.0 / ksafe ** 2) - ex_ * (b_ / ksafe + 1.0 / ksafe ** 2))
            wt = F_ * np.exp(Ea)
            sw = (wt * z0).sum(1)
            ell_v = (wt * m1).sum(1) / np.maximum(sw, 1e-300)
            ell[vi] = np.where(sw > 0, ell_v, ell[vi])
        return ell

    # ---------------------------------------------------------------- spectral rho (white ref, room light)
    def _rho_spectrum(self, opt, k, gidx, cos_in_const: float | None) -> np.ndarray:
        """Elastic ``rho(lam)`` on the full grid, from kernels at the room-bin wavelengths."""
        nb = len(ROOM_BIN_CENTRES)
        rho = np.empty((len(gidx), nb))
        cin = None if cos_in_const is None else np.full(len(gidx), cos_in_const)
        for i, c in enumerate(ROOM_BIN_CENTRES):
            rho[:, i] = np.asarray(self._kernels(opt, c, k, gidx, cos_in=cin).elastic_rho)
        pos = np.interp(GRID, ROOM_BIN_CENTRES, np.arange(nb))
        lo = np.minimum(np.floor(pos).astype(int), nb - 1)
        hi = np.minimum(lo + 1, nb - 1)
        t = pos - lo
        return rho[:, lo] * (1 - t) + rho[:, hi] * t

    # ---------------------------------------------------------------- room light
    def room_chunk(self, a: int, b: int) -> np.ndarray | None:
        """Ambient-light spectral radiance (n, W): elastic remission + induced fluorescence."""
        rl = self.exp.background.room_light
        if not rl.enabled or rl.illuminance_lux <= 0:
            return None
        h, tg = self.hit, self.tg
        full = np.ones(len(GRID), dtype=bool)
        erm = spectra.room_light_irradiance_mW_mm2_nm(rl.spectrum, rl.illuminance_lux, GRID)
        out = np.zeros((b - a, len(GRID)), dtype=np.float32)
        for k in range(self.K):
            matk = self.mat_eff[k, a:b]
            for m in np.unique(matk[matk >= 0]):
                opt, mspec = self.optics[int(m)], self.mats[int(m)]
                if self._skippable(opt, mspec, False):
                    continue
                sel = np.nonzero(matk == m)[0]
                gidx = a + sel
                V = self._view_transmission(k, gidx, full)
                acc = np.zeros((len(gidx), len(GRID)), dtype=np.float32)
                if tg.elastic_leak:
                    rho_w = self._rho_spectrum(opt, k, gidx, ROOM_COS_IN)
                    acc += (erm[None, :] * rho_w).astype(np.float32)
                if tg.fluorescence and opt.mua_dyes_per_mm:
                    for c in ROOM_BIN_CENTRES:
                        inb = np.abs(GRID - c) <= ROOM_BIN_HALF
                        e_i = float(erm[inb].sum())
                        if e_i <= 0:
                            continue
                        g = np.asarray(self._kernels(opt, c, k, gidx, cos_in=np.full(len(gidx), ROOM_COS_IN)
                                                     ).fluorescence_G, dtype=np.float32)
                        for d, mua in opt.mua_dyes_per_mm.items():
                            mua_i = float((erm[inb] * mua[inb]).sum() / e_i)
                            amp = e_i * mua_i * opt.quantum_yields[d]
                            acc += np.float32(amp) * opt.emission_density[d].astype(np.float32)[None, :] * g
                out[sel] += acc * V
        return out


# --------------------------------------------------------------------------------------------
# step preparation / result assembly
# --------------------------------------------------------------------------------------------
def _prepare_steps(exp: Experiment, hit: HitBuffer, mode: str, steps: list[ExcitationStep],
                   eng: _Engine, keep: bool, max_angle: float | None = None) -> list[_Step]:
    cam, src, tg = exp.camera(), exp.source(), exp.acquisition.physics
    powers = _step_powers(exp, mode, [s for s in src.sequence if s.enabled])
    if max_angle is None:
        max_angle = float(np.max(hit.field_angle_rad)) if hit.field_angle_rad.size else 0.0
    out = []
    for i, st in enumerate(steps):
        centres = sensor_mod.band_centres(cam.filter, st.wavelength_nm)
        resp = sensor_mod.band_responses(cam, centres, tg, max_angle)
        power = 0.0 if mode == "dark" else powers[id(st)]
        if mode == "white_reference":
            line, leak = _white_continuum(src), np.zeros_like(GRID)
        else:
            line, leak = _line_shape(src, st.wavelength_nm), _leak_spectrum(src, st.wavelength_nm)
        bi, bw = _line_samples(line)
        P, B = eng.P, len(centres)
        s = _Step(index=i, ex_nm=float(st.wavelength_nm), power_mW=power, exposure_ms=float(st.exposure_ms),
                  centres=centres, responses=resp, wsel=resp.window.copy(), line=line, leak=leak,
                  raman=_raman_shape(st.wavelength_nm) if mode == "sample" else None,
                  beer_idx=bi, beer_w=bw)
        s.rad_d, s.el_d, s.rad_s, s.el_s = (np.zeros((P, B), dtype=np.float32) for _ in range(4))
        s.irr = np.zeros(P, dtype=np.float64)
        if mode == "sample" and power > 0 and tg.specular_glints and tg.fluorescence:
            s.fl_b = np.zeros((P, B), dtype=np.float32)
        s.ring = resp.ring_index(hit.field_angle_rad)
        if mode == "white_reference":    # the continuum spans everything the bands see
            s.wsel = s.wsel | (line > 0)
        s.wsel_idx = np.nonzero(s.wsel)[0]
        out.append(s)
    return out


def _run_engine(eng: _Engine, steps: list[_Step], keep: bool, progress, cancel, p0: float, p1: float) -> bool:
    """Step-outer / chunk-inner loop.  Bleaching state is pixel-local and every pixel still sees the
    steps in sequence order, so this is equivalent to chunk-outer order.  Returns False if cancelled.

    Progress: ``simulating <ex> nm (<i>/<n>) · radiance`` (see the module docstring)."""
    nchunk = max(1, -(-eng.P // CHUNK))
    nst = len(steps)
    rl = eng.exp.background.room_light
    room_cache: dict[int, np.ndarray] | None = None
    if rl.enabled and rl.illuminance_lux > 0:
        room_cache = {}
        u = np.zeros(len(GRID), dtype=bool)
        for s_ in steps:
            u |= s_.wsel
        eng.room_cols = u
    last = -1.0
    for si, st in enumerate(steps):
        msg = f"simulating {st.ex_nm:g} nm ({si + 1}/{nst}) · radiance"
        for ci in range(nchunk):
            a, b = ci * CHUNK, min((ci + 1) * CHUNK, eng.P)
            if cancel is not None and cancel():
                return False
            room = None
            if room_cache is not None:
                if ci not in room_cache:
                    r_ = eng.room_chunk(a, b)
                    room_cache[ci] = None if r_ is None else np.ascontiguousarray(r_[:, eng.room_cols])
                room = room_cache[ci]
            eng.step_chunk(st, a, b, room, keep)
            if progress is not None:
                f = p0 + (p1 - p0) * (si * nchunk + ci + 1) / (nst * nchunk)
                if f - last >= 0.004 or ci == nchunk - 1:
                    progress(f, msg)
                    last = f
    return True


def _spread_subsurface(eng: _Engine, st: _Step, h: HitBuffer) -> None:
    """Convolve each turbid material's diffuse band images with its radial kernel (in place)."""
    H, W = h.height, h.width
    if not st.radial or eng.P != H * W:
        return
    mat0 = eng.mat_eff[0].reshape(H, W)          # block pixels (-1) are not spread
    foot = h.pixel_footprint_mm.reshape(H, W)
    centres = st.centres
    for m, (r_mm, prof, lam_nodes) in st.radial.items():
        mask = mat0 == m
        if not mask.any():
            continue
        f_mm = float(np.median(foot[mask]))
        if not np.isfinite(f_mm) or f_mm <= 0:
            continue
        ys, xs = np.nonzero(mask)
        rad_px = int(min(25, max(1, np.ceil(float(r_mm[-1]) / f_mm))))
        y0, y1 = max(ys.min() - rad_px, 0), min(ys.max() + rad_px + 1, H)
        x0, x1 = max(xs.min() - rad_px, 0), min(xs.max() + rad_px + 1, W)
        yy, xx = np.mgrid[-rad_px:rad_px + 1, -rad_px:rad_px + 1]
        rr = np.hypot(yy, xx) * f_mm
        node_of_band = (np.argmin(np.abs(np.asarray(lam_nodes)[None, :] - centres[:, None]), axis=1)
                        if lam_nodes is not None and len(lam_nodes) > 1 else np.zeros(len(centres), int))
        m_sub = mask[y0:y1, x0:x1].astype(np.float32)
        for node in np.unique(node_of_band):
            kern = np.interp(rr, r_mm, np.asarray(prof)[node]).astype(np.float32)
            if kern.sum() <= 0:
                continue
            kern /= kern.sum()
            bands = np.nonzero(node_of_band == node)[0]
            nb = len(bands)
            cube_d, cube_e = st.rad_d.reshape(H, W, -1), st.el_d.reshape(H, W, -1)
            # radiance and electron images share the kernel: one batched FFT convolution of both
            sub = np.concatenate([cube_d[y0:y1, x0:x1][:, :, bands], cube_e[y0:y1, x0:x1][:, :, bands]], axis=2)
            sub = sub * m_sub[:, :, None]
            conv = _fft_same(sub, kern)
            keep_m = m_sub[:, :, None] > 0
            for cube, part in ((cube_d, conv[:, :, :nb]), (cube_e, conv[:, :, nb:])):
                region = cube[y0:y1, x0:x1]
                region[:, :, bands] = np.where(keep_m, part, region[:, :, bands])


def _fft_same(sub: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """``scipy.signal.fftconvolve(sub, kern[:, :, None], mode='same', axes=(0, 1))`` with multithreaded
    FFTs (``sub`` (h, w, c) float32, ``kern`` (kh, kw) odd-sized); identical to ~1e-7 relative."""
    from scipy import fft as sfft
    h, w = sub.shape[:2]
    kh, kw = kern.shape
    shape = (sfft.next_fast_len(h + kh - 1, real=True), sfft.next_fast_len(w + kw - 1, real=True))
    F = sfft.rfft2(sub, s=shape, axes=(0, 1), workers=-1)
    K = sfft.rfft2(kern, s=shape, workers=-1)
    out = sfft.irfft2(F * K[:, :, None], s=shape, axes=(0, 1), workers=-1)
    r0, c0 = (kh - 1) // 2, (kw - 1) // 2
    return np.ascontiguousarray(out[r0:r0 + h, c0:c0 + w, :], dtype=np.float32)


def _gt_maps(eng: _Engine, steps: list[_Step], h: HitBuffer) -> dict[str, np.ndarray]:
    H, W = h.height, h.width
    f32 = lambda x: np.asarray(x, dtype=np.float32).reshape(H, W)   # noqa: E731
    valid = h.material[0] >= 0
    gt = {
        "material": f32(h.material[0]), "object": f32(h.object_index[0]),
        "depth_mm": f32(np.where(np.isfinite(h.depth_mm), h.depth_mm, np.nan)),
        "cos_light": f32(h.cos_light[0]), "cos_view": f32(h.cos_view[0]),
        "path_mm": f32(h.path_mm[0]), "thickness_mm": f32(np.where(np.isfinite(h.thickness_mm[0]),
                                                                    h.thickness_mm[0], np.nan)),
        "shadow": f32(valid & (h.T_light[0] <= 0)),
    }
    for s in steps:
        gt[f"irradiance@{s.ex_nm:g}"] = f32(s.irr)
        for d, dose in s.dose.items():
            gt[f"absorbed_photons:{d}@{s.ex_nm:g}"] = f32(dose)
    for d in eng.dyes:
        conc = np.zeros(h.material.shape[1])
        for mi, m in enumerate(eng.mats):
            c = sum(x.concentration_uM for x in m.dyes if x.fluorophore == d)
            if c > 0:
                conc[eng.mat_eff[0] == mi] = c
        bl = np.asarray(eng.survival[d][0], dtype=np.float64).copy()
        for run in eng.blocks:                       # concentration / survival vary with depth in blocks
            conc[run.mask[0]] = np.nan
            bl[run.mask[0]] = np.nan
        gt[f"conc:{d}"] = f32(conc)
        gt[f"bleached:{d}"] = f32(bl)
    gt.update(_merge_block_gt(eng, h))
    return gt


def _block_labels(eng: _Engine) -> list[str]:
    """Unique per-block qualifier: the scene-object name, or ``<name>#<object id>`` when names repeat."""
    names = [str(eng.exp.objects[r.object_index].name) for r in eng.blocks]
    return [n if names.count(n) == 1 else f"{n}#{r.object_id}" for n, r in zip(names, eng.blocks)]


def _merge_block_gt(eng: _Engine, h: HitBuffer) -> dict[str, np.ndarray]:
    """Ground-truth maps of all blocks. Un-namespaced keys (``layer_thickness:cover``, ...) are the MERGED map: each
    pixel takes the value of the visible block (the one whose surface the view ray meets first, i.e. the lowest
    camera layer slot). With several blocks also ``<family>:<block>/<name>`` qualified keys (own block only)."""
    if len(eng.blocks) == 1:
        return _block_gt(eng.blocks[0], h)
    H, W = h.height, h.width
    merged: dict[str, np.ndarray] = {}
    rank: dict[str, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    for run, label in zip(eng.blocks, _block_labels(eng)):
        maps = _block_gt(run, h)
        first = np.where(run.mask.any(axis=0), run.mask.argmax(axis=0), np.iinfo(np.int32).max).reshape(H, W)
        for key, arr in maps.items():
            if ":" in key:
                fam, nm = key.split(":", 1)
                out[f"{fam}:{label}/{nm}"] = arr
            m = merged.setdefault(key, np.full((H, W), np.nan, dtype=np.float32))
            r = rank.setdefault(key, np.full((H, W), np.iinfo(np.int32).max, dtype=np.int64))
            take = np.isfinite(arr) & (first < r)
            m[take] = arr[take]
            r[take] = first[take]
    out.update(merged)
    return out


def _block_gt(run, h: HitBuffer) -> dict[str, np.ndarray]:
    """Block ground-truth maps (NaN off the block; every camera layer slot that hits the block)."""
    H, W = h.height, h.width
    K = run.mask.shape[0]
    maps: dict[str, np.ndarray] = {}
    for k in range(K - 1, -1, -1):          # layer 0 (first surface) wins over deeper slots
        sel = np.nonzero(run.mask[k])[0]
        if sel.size == 0:
            continue
        vals = run.gt_values(run.face_code[k, sel].astype(int), run.hit_u[k, sel], run.hit_v[k, sel],
                             run.hit_x[k, sel], run.hit_z[k, sel])
        for key, v in vals.items():
            arr = maps.setdefault(key, np.full(H * W, np.nan, dtype=np.float32))
            arr[sel] = v
    return {k_: v.reshape(H, W) for k_, v in maps.items()}


def _block_gt_pixel(run, h: HitBuffer) -> dict[str, float]:
    """Block ground truth of the single traced pixel of :func:`explain_pixel` (NaN-free scalars)."""
    for k in range(run.mask.shape[0]):
        if run.mask[k, 0]:
            vals = run.gt_values(run.face_code[k, :1].astype(int), run.hit_u[k, :1], run.hit_v[k, :1],
                                 run.hit_x[k, :1], run.hit_z[k, :1])
            return {key: (None if not np.isfinite(v[0]) else float(v[0])) for key, v in vals.items()}
    return {}


def _run_blocks(eng: _Engine, steps: list[_Step], traced: Experiment, mode: str, scale: float, progress, cancel):
    """3D Monte Carlo of every visible block (cached); ``None`` if cancelled."""
    if not eng.hit.info.get("block_hits"):
        return []
    from spectraforge.studio import block_capture
    frac = _block_fraction(traced, scale)
    return block_capture.get_runs(eng, steps, eng._scene_geometry(), mode, frac, progress, cancel)


def _block_fraction(exp: Experiment, scale: float) -> float:
    """Photon fraction of the block Monte Carlo: full at scale >= 1, ``preview_fraction`` otherwise."""
    return 1.0 if scale >= 1.0 else float(exp.acquisition.volume_mc.preview_fraction)


def _restrict_sequence(exp: Experiment, excitations) -> Experiment:
    """``exp`` whose active source only keeps the enabled steps at ``excitations`` (what ``preview`` does)."""
    src = exp.source()
    want = [float(e) for e in excitations]
    seq = [s for s in src.sequence if s.enabled and any(abs(s.wavelength_nm - w) < 1e-6 for w in want)]
    if not seq or len(seq) == sum(1 for s in src.sequence if s.enabled):
        return exp
    new_src = src.model_copy(update={"sequence": seq})
    return exp.model_copy(update={"sources": [new_src if s.id == src.id else s for s in exp.sources]})


def _block_results(res: CaptureResult, eng: _Engine, steps: list[_Step]) -> None:
    """Fill ``CaptureResult.blocks`` / ``.volumes`` from the block runs."""
    for run in eng.blocks:
        res.blocks.append({"object_id": run.object_id, "layer_names": list(run.layer_names),
                           "footprint_mm": [float(run.footprint_mm[0]), float(run.footprint_mm[1])],
                           "world_from_local": np.asarray(run.world_from_local, float).tolist(),
                           "inclusions": [dict(m) for m in getattr(run, "incl_meta", [])]})
        res.volumes[run.object_id] = {
            "grid": run.grid, "layer_names": list(run.layer_names),
            "fluence": {ex: run.fluence[ex] for ex in run.ex_list},
            "absorbed": {ex: run.absorbed[ex] for ex in run.ex_list},
            "source": dict(run.sources),
            "units": {"fluence": "mW/mm^2", "absorbed": "mW", "source": "mW"}}
        imap = getattr(run, "imap", None)
        if imap is not None:       # v1.2: inclusion tallies on the inclusion map (axis order x, y, z)
            res.volumes[run.object_id]["inclusions"] = {
                "names": list(imap.names), "voxel_mm": imap.voxel_mm, "origin_mm": list(imap.origin_mm),
                "shape": list(imap.shape), "region": imap.region,
                "fluence": {ex: run.incl_fluence[ex] for ex in run.ex_list if ex in run.incl_fluence},
                "absorbed": {ex: run.incl_absorbed[ex] for ex in run.ex_list if ex in run.incl_absorbed},
                "source": dict(run.incl_sources)}


# --------------------------------------------------------------------------------------------
# white reference geometry
# --------------------------------------------------------------------------------------------
def _euler_xyz_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """three.js 'XYZ' intrinsic Euler angles (degrees) of rotation matrix ``m`` (columns = local axes)."""
    b = np.arcsin(np.clip(m[0, 2], -1, 1))
    if abs(m[0, 2]) < 0.9999999:
        a = np.arctan2(-m[1, 2], m[2, 2])
        c = np.arctan2(-m[0, 1], m[0, 0])
    else:
        a, c = np.arctan2(m[2, 1], m[1, 1]), 0.0
    return tuple(float(np.degrees(v)) for v in (a, b, c))


def white_reference_experiment(exp: Experiment) -> Experiment:
    """Experiment with the objects replaced by a Spectralon plate filling the field of view.

    The plate is a thin box facing the camera, centred on the camera's target (so it also works for
    side views); the background slab is removed.  The room light setting is preserved."""
    from spectraforge.studio.presets import builtin_materials

    cam = exp.camera()
    mats = list(exp.materials)
    if not any(m.id == "spectralon" for m in mats):
        mats.append(next(m for m in builtin_materials() if m.id == "spectralon"))
    pos, tgt = np.array(cam.position_mm, float), np.array(cam.target_mm, float)
    z = pos - tgt
    dist = float(np.linalg.norm(z))
    z = z / max(dist, 1e-9)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = _euler_xyz_from_matrix(np.column_stack([x, y, z]))
    thick = 5.0
    size = max(4.0 * dist, 200.0)
    centre = tgt - z * thick / 2.0
    plate = SceneObject(id="__white_reference__", name="Spectralon plate",
                        shape=BoxShape(size_mm=(size, size, thick)), material_id="spectralon",
                        transform=Transform(position_mm=tuple(float(v) for v in centre), rotation_deg=rot))
    bg = Background(kind="none", material_id=None, room_light=exp.background.room_light)
    return exp.model_copy(update={"materials": mats, "objects": [plate], "background": bg})


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------
def _get_tracer(tracer):
    if tracer is not None:
        return tracer
    from spectraforge.studio import tracer as _tracer_mod
    return _tracer_mod.trace_scene


def _get_transport(transport, exp: Experiment):
    if transport is not None:
        return transport
    from spectraforge.studio.transport import get_transport_model
    acq = exp.acquisition
    try:
        return get_transport_model(n_photons=acq.mc_photons, seed=acq.seed, cache_dir=None)
    except TypeError:
        return get_transport_model(acq.mc_photons, acq.seed, None)


def _select_steps(exp: Experiment, excitations) -> list[ExcitationStep]:
    src = exp.source()
    steps = [s for s in src.sequence if s.enabled]
    if excitations is not None:
        want = [float(e) for e in excitations]
        steps = [s for s in steps if any(abs(s.wavelength_nm - w) < 1e-6 for w in want)]
    if not steps:
        raise ValueError("no enabled excitation steps to capture")
    return steps


def _mode_experiment(exp: Experiment, mode: str) -> Experiment:
    return white_reference_experiment(exp) if mode == "white_reference" else exp


def capture(experiment: Experiment, *, mode: str | None = None, scale: float = 1.0, excitations=None,
            progress: ProgressFn | None = None, cancel=None, transport=None, tracer=None) -> CaptureResult:
    """Run the full simulation and return the 4-D cube plus ground truth.

    ``mode``: ``sample`` | ``dark`` (source off; room light and detector terms remain) |
    ``white_reference`` (objects replaced by a Spectralon plate lit by the broadband continuum of the
    source: one frame per excitation step with the *same band set* as the sample frames, so
    ``sample[ex] / white[ex]`` is the lab's Whitelight normalisation; no leakage, no Raman).
    ``scale``: resolution scale of the sensor (preview < 1).  ``excitations``: subset of the
    sequence (nm).  ``cancel()`` -> True aborts: a partial (empty) result with a warning is returned.
    Deterministic: ``acquisition.seed`` + mode + step index fix every random draw."""
    t_start = time.perf_counter()
    mode = mode or experiment.acquisition.mode
    if mode not in _MODE_ID:
        raise ValueError(f"unknown mode '{mode}'")
    exp = experiment
    cam, tg, acq = exp.camera(), exp.acquisition.physics, exp.acquisition
    steps_spec = _select_steps(exp, excitations)
    width = max(8, int(round(cam.sensor.width_px * scale)))
    height = max(8, int(round(cam.sensor.height_px * scale)))
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def prog(f, msg):
        if progress is not None:
            progress(float(min(max(f, 0.0), 1.0)), msg)

    prog(0.0, "tracing rays")
    t0 = time.perf_counter()
    traced = _mode_experiment(exp, mode)
    hit = _get_tracer(tracer)(traced, width, height, toggles=tg)
    timings["trace"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tr = _get_transport(transport, traced)
    eng = _Engine(traced, hit, tr, mode)
    steps = _prepare_steps(traced, hit, mode, steps_spec, eng, keep=False)
    timings["setup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rad_p0 = 0.08
    ok = True
    if hit.info.get("block_hits"):                  # 3D Monte Carlo of the layered blocks (cached)
        runs = _run_blocks(eng, steps, traced, mode, scale, lambda f, m: prog(0.08 + 0.44 * f, m), cancel)
        timings["mc3d"] = time.perf_counter() - t0
        ok = runs is not None
        if ok:
            eng.attach_blocks(runs)
            rad_p0 = 0.52
        t0 = time.perf_counter()
    if ok:
        ok = _run_engine(eng, steps, False, progress, cancel, rad_p0, 0.7)
    timings["radiance"] = time.perf_counter() - t0
    if not ok:
        res = CaptureResult(experiment_json=experiment.model_dump_json(), mode=mode, width=width, height=height,
                            excitations_nm=[], emission_nm={}, adu={}, radiance={}, electrons_mean={},
                            exposure_ms={}, power_mW={}, ground_truth={}, warnings=["capture cancelled"],
                            timings_s=timings)
        return res

    # ---- sensor stage -------------------------------------------------------------------------
    t0 = time.perf_counter()
    copt = sensor_mod.CameraOptics.from_camera(cam, hit.focus_depth_mm if hit.focus_depth_mm > 0 else None)
    fp = sensor_mod.make_fixed_pattern(cam.sensor, height, width, acq.seed)
    vign = sensor_mod.vignetting(hit.field_angle_rad, tg.vignetting).astype(np.float32)
    axial = hit.depth_mm * np.cos(hit.field_angle_rad)
    res = CaptureResult(experiment_json=experiment.model_dump_json(), mode=mode, width=width, height=height,
                        excitations_nm=[], emission_nm={}, adu={}, radiance={}, electrons_mean={},
                        exposure_ms={}, power_mW={}, ground_truth={})
    sat_note = []
    low_snr = []
    glint_note = []
    any_lit = False
    for si, st in enumerate(steps):
        if tg.subsurface_spread:
            _spread_subsurface(eng, st, hit)
        rad = (st.rad_d + st.rad_s)
        e_geo = (st.el_d + st.el_s) * (copt.electron_scale(st.exposure_ms) * vign)[:, None]
        e_cube = e_geo.reshape(height, width, -1)
        sig = sensor_mod.blur_sigma_px(axial, cam, st.centres, tg, hit.focus_depth_mm or None)
        sig = (sig * scale).reshape(height, width, -1).astype(np.float32)
        e_blur = sensor_mod.blur_cube(e_cube, sig)
        rng = np.random.default_rng(np.random.SeedSequence([int(acq.seed) & 0xFFFFFFFF, _MODE_ID[mode], si, 0xC4A7]))
        adu, sat = sensor_mod.detect(e_blur, st.exposure_ms, cam.sensor, tg, fp, rng)
        ex = st.ex_nm
        res.excitations_nm.append(ex)
        res.emission_nm[ex] = st.centres
        res.adu[ex] = adu
        res.radiance[ex] = rad.reshape(height, width, -1).astype(np.float32)
        res.electrons_mean[ex] = e_blur.astype(np.float32)
        res.exposure_ms[ex] = st.exposure_ms
        res.power_mW[ex] = st.power_mW
        if sat > 5e-4:
            sat_note.append(f"{ex:g} nm: {100 * sat:.2f}% of samples saturated")
        objpix = hit.object_index[0] >= 0
        if mode == "sample" and st.power_mW > 0:
            vf = getattr(hit, "vol_fluence_per_mW", None)
            if np.any(st.irr[objpix] > 0) or (vf is not None and np.any(vf[:, objpix] > 0)):
                any_lit = True
            g = _glint_stats(st, objpix)
            if g is not None:
                glint_note.append(g)
            st.fl_b = None
        if objpix.any():
            snr = sensor_mod.snr_estimate(e_blur.reshape(-1, e_blur.shape[-1])[objpix], st.exposure_ms,
                                          cam.sensor, tg).max(axis=1)
            if np.median(snr) < 3.0:
                low_snr.append(f"{ex:g} nm (median peak-band SNR {np.median(snr):.1f})")
        prog(0.7 + 0.29 * (si + 1) / len(steps), f"sensor {ex:g} nm ({si + 1}/{len(steps)})")
    timings["sensor"] = time.perf_counter() - t0

    res.ground_truth = _gt_maps(eng, steps, hit)
    _block_results(res, eng, steps)
    # ---- warnings -------------------------------------------------------------------------------
    if sat_note:
        warnings.append("saturation: " + "; ".join(sat_note))
    if mode == "sample" and (hit.object_index[0] >= 0).any() and not any_lit:
        warnings.append("beam misses every object (no object pixel receives excitation light)")
    if glint_note:
        warnings.append("specular glint: " + "; ".join(glint_note))
    if low_snr and mode != "dark":
        warnings.append("very low SNR at " + "; ".join(low_snr))
    warnings.extend(_dof_warnings(traced, hit, cam, tg))
    res.warnings = warnings
    timings["total"] = time.perf_counter() - t_start
    res.timings_s = timings
    prog(1.0, "done")
    return res


#: glint warning thresholds (see :func:`_glint_stats`)
GLINT_RATIO_THRESHOLD = 1.0       # glint band signal / peak fluorescence band signal
GLINT_MIN_FRACTION = 0.05         # fraction of lit object pixels that must exceed it


def _glint_stats(st: _Step, objpix: np.ndarray) -> str | None:
    """Describe the specular (glint) contamination of one excitation step, or None if negligible.

    Only *fluorescent* lit object pixels are judged (peak fluorescence band above 1 % of the step's
    99th-percentile fluorescence): glass walls and inert objects have no fluorescence to be contaminated.
    Per such pixel ``R = max_b L_spec,b / max_b L_fluor,b`` compares the band that the specular reflection of
    the excitation (plus specularly reflected filter leakage) reaches with the brightest fluorescence band,
    both *as recorded* (after band pass-band, long-pass and angle shift).  The warning is raised when more
    than 5 % of those pixels have ``R > 1`` and reports that fraction, the pixel count and the median / 95th
    percentile of ``R`` among them."""
    if st.fl_b is None:
        return None
    spec = st.rad_s.max(axis=1)
    fl = st.fl_b.max(axis=1)
    cand = objpix & (fl > 0)
    if not cand.any():
        return None
    fluor = cand & (fl > 1e-2 * np.percentile(fl[cand], 99))
    if fluor.sum() < 20:
        return None
    ratio = spec[fluor] / fl[fluor]
    bad = ratio > GLINT_RATIO_THRESHOLD
    frac = float(bad.mean())
    if frac < GLINT_MIN_FRACTION:
        return None
    fmt = lambda v: ">1000x" if v > 1000 else (f"{v:.0f}x" if v >= 10 else f"{v:.1f}x")   # noqa: E731
    rb = ratio[bad]
    return (f"{st.ex_nm:g} nm: {100 * frac:.0f}% of fluorescent object pixels ({int(bad.sum())} px) have glint above "
            f"their peak fluorescence band (median {fmt(np.median(rb))}, 95th percentile {fmt(np.percentile(rb, 95))})")


def _dof_warnings(exp: Experiment, hit: HitBuffer, cam, tg: PhysicsToggles) -> list[str]:
    if not tg.defocus:
        return []
    axial = hit.depth_mm * np.cos(hit.field_angle_rad)
    t2 = tg.model_copy(update={"diffraction": False, "chromatic_aberration": False})
    sig = sensor_mod.blur_sigma_px(axial, cam, np.array([550.0]), t2, hit.focus_depth_mm or None)[:, 0]
    out = []
    for oi, obj in enumerate(exp.objects):
        sel = hit.object_index[0] == oi
        if sel.sum() >= 20 and np.median(sig[sel]) > 1.5:
            out.append(f"object '{obj.name}' is outside the depth of field (blur sigma ~{np.median(sig[sel]):.1f} px)")
    return out


def preview(experiment: Experiment, excitation_nm: float | None = None, scale: float | None = None,
            transport=None, tracer=None, **kw) -> CaptureResult:
    """One excitation at ``scale`` (default ``acquisition.preview_scale``).

    ``excitation_nm`` defaults to the first enabled step; a wavelength not in the sequence reuses
    the power/exposure of the nearest step."""
    src = experiment.source()
    enabled = [s for s in src.sequence if s.enabled] or list(src.sequence)
    if not enabled:
        raise ValueError("the light source has no excitation steps")
    if excitation_nm is None:
        step = enabled[0]
    else:
        step = min(enabled, key=lambda s: abs(s.wavelength_nm - excitation_nm))
        if abs(step.wavelength_nm - excitation_nm) > 1e-6:
            step = step.model_copy(update={"wavelength_nm": float(excitation_nm)})
    step = step.model_copy(update={"enabled": True})
    new_src = src.model_copy(update={"sequence": [step]})
    exp = experiment.model_copy(update={
        "sources": [new_src if s.id == src.id else s for s in experiment.sources]})
    return capture(exp, scale=scale or experiment.acquisition.preview_scale, transport=transport,
                   tracer=tracer, **kw)


def explain_pixel(experiment: Experiment, x: int, y: int, *, scale: float = 1.0, mode: str | None = None,
                  transport=None, tracer=None, excitations=None) -> dict:
    """Per-pixel decomposition of the captured signal (see :class:`PixelExplain` in ``api/types.ts``).

    The pixel is recomputed exactly via ``trace_scene(pixels=[idx])`` and the same engine
    (bleaching included, since it is pixel-local).  Blur, subsurface spread and fixed-pattern noise
    are *not* applied (they need neighbours), so ``radiance`` equals the capture's noise-free band
    radiance except near edges of turbid materials; ``adu`` is the noise-free expected value.
    Excitation keys are formatted ``f"{ex:g}"``; component values are per-band radiance.

    Agreement with ``capture().radiance`` (QA, all presets, 24 random pixels each): at *interior* pixels
    (9x9 neighbourhood of one object and material) the band-summed radiance matches to < 0.01 % for clear
    and diffuse-only media and to a median 0.04 % / max 0.9 % for turbid media with subsurface spread;
    only the radial subsurface-spread convolution (within ~ the spread radius of a material edge, or
    over a whole small turbid object such as a 14 mm agarose bead: ~25 %) is absent here, because it needs
    neighbouring pixels.  At edge pixels the difference is the lateral light exchange (median 0.14 %, up
    to ~7 % at material boundaries).  Optical blur and fixed-pattern noise are not applied either (only
    ``radiance`` is compared; ``adu`` is the noise-free expectation without blur).  The filter-angle rings
    are laid out up to the full-frame corner angle, exactly as in :func:`capture`.

    Layered blocks: the 3D Monte Carlo is cached per (experiment, mode, photon fraction, excitation steps)
    (:mod:`block_capture`).  ``excitations`` (wavelengths in nm) selects the steps to explain; by default the
    steps of the most recent cached block run of this experiment are used, so a pixel of a preview explains
    the PREVIEWED excitation and re-uses its tallies (~20-60 ms per pixel instead of re-simulating the whole
    sequence, ~10 s).  Pass the same ``scale`` as the capture (it selects the photon fraction)."""
    mode = mode or experiment.acquisition.mode
    exp = experiment
    cam, tg, acq = exp.camera(), exp.acquisition.physics, exp.acquisition
    width = max(8, int(round(cam.sensor.width_px * scale)))
    height = max(8, int(round(cam.sensor.height_px * scale)))
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"pixel ({x}, {y}) outside {width}x{height}")
    idx = y * width + x
    if any(o.visible and o.shape.kind == "layered_block" for o in exp.objects):
        # layered blocks: explain the excitation(s) that were previewed / captured, so that the cached 3D
        # Monte Carlo is reused instead of re-simulating the whole sequence (``excitations`` overrides)
        from spectraforge.studio import block_capture
        sel = excitations if excitations is not None else block_capture.cached_excitations(
            _mode_experiment(exp, mode), mode, _block_fraction(exp, scale))
        if sel is not None:
            exp = _restrict_sequence(exp, sel)
    traced = _mode_experiment(exp, mode)
    hit = _get_tracer(tracer)(traced, width, height, pixels=[idx], toggles=tg)
    if hit.material.shape[1] != 1:
        hit = _subset_hit(hit, np.array([idx]))
    eng = _Engine(traced, hit, _get_transport(transport, traced), mode)
    # the filter-angle rings are laid out up to the largest field angle of the FULL frame (corner pixels),
    # exactly as in capture(); a single-pixel trace would otherwise quantise the angle shift differently
    corners = np.array(sorted({0, width - 1, (height - 1) * width, height * width - 1}))
    hc = _get_tracer(tracer)(traced, width, height, pixels=corners, toggles=tg)
    steps = _prepare_steps(traced, hit, mode, _select_steps(exp, None), eng, keep=True,
                           max_angle=float(np.max(hc.field_angle_rad)))
    runs = _run_blocks(eng, steps, traced, mode, scale, None, None)
    eng.attach_blocks(runs)
    _run_engine(eng, steps, True, None, None, 0.0, 1.0)
    copt = sensor_mod.CameraOptics.from_camera(cam, hit.focus_depth_mm if hit.focus_depth_mm > 0 else None)
    vign = sensor_mod.vignetting(hit.field_angle_rad, tg.vignetting)
    fp = sensor_mod.make_fixed_pattern(cam.sensor, height, width, acq.seed)
    out: dict = {"x": int(x), "y": int(y), "mode": mode, "emission_nm": {}, "adu": {}, "radiance": {},
                 "components": {}, "ground_truth": {}}
    gt: dict = {}
    for st in steps:
        key = f"{st.ex_nm:g}"
        out["emission_nm"][key] = [float(v) for v in st.centres]
        rad = (st.rad_d + st.rad_s)[0]
        out["radiance"][key] = [float(v) for v in rad]
        e = (st.el_d + st.el_s)[0] * copt.electron_scale(st.exposure_ms) * vign[0]
        dark = float(np.mean(sensor_mod.mean_dark_electrons(cam.sensor, st.exposure_ms, tg, None)))
        adu = (e + dark) / cam.sensor.gain_e_per_adu + cam.sensor.offset_adu
        if tg.saturation:
            adu = np.clip(adu, 0.0, 2 ** cam.sensor.bit_depth - 1)
        out["adu"][key] = [float(v) for v in adu]
        out["components"][key] = {name: [float(v) for v in arr[0]] for name, arr in sorted((st.comps or {}).items())}
        gt[f"irradiance@{key}"] = float(st.irr[0])
        for d, dose in st.dose.items():
            gt[f"absorbed_photons:{d}@{key}"] = float(dose[0])
    m0, o0 = int(hit.material[0, 0]), int(hit.object_index[0, 0])
    gt.update({
        "material": eng.mats[m0].id if m0 >= 0 else None,
        "object": (exp.objects[o0].id if o0 >= 0 else ("background" if o0 == -2 else None)),
        "depth_mm": float(hit.depth_mm[0]) if np.isfinite(hit.depth_mm[0]) else None,
        "cos_light": float(hit.cos_light[0, 0]), "cos_view": float(hit.cos_view[0, 0]),
        "path_mm": float(hit.path_mm[0, 0]), "thickness_mm": (float(hit.thickness_mm[0, 0])
                                                              if np.isfinite(hit.thickness_mm[0, 0]) else None),
        "shadow": bool(m0 >= 0 and hit.T_light[0, 0] <= 0),
        "field_angle_deg": float(np.degrees(hit.field_angle_rad[0])),
    })
    is_block = any(r.mask[0, 0] for r in eng.blocks)
    for d in eng.dyes:
        conc = sum(dd.concentration_uM for dd in eng.mats[m0].dyes if dd.fluorophore == d) if m0 >= 0 else 0.0
        gt[f"conc:{d}"] = None if is_block else float(conc)
        gt[f"bleached:{d}"] = None if is_block else float(eng.survival[d][0, 0])
    hit_blocks = [(int(r.mask[:, 0].argmax()), r, lab) for r, lab in zip(eng.blocks, _block_labels(eng))
                  if r.mask[:, 0].any()]
    for _k, r, lab in sorted(hit_blocks, key=lambda t: -t[0]):      # topmost (lowest slot) block applied last = wins
        vals = _block_gt_pixel(r, hit)
        if len(eng.blocks) > 1:
            gt.update({f"{k.split(':', 1)[0]}:{lab}/{k.split(':', 1)[1]}": v for k, v in vals.items() if ":" in k})
        gt.update(vals)
    out["ground_truth"] = gt
    return out
