"""3D Monte Carlo orchestration for layered blocks (called by :mod:`spectraforge.studio.capture`).

For every ``layered_block`` object the capture engine replaces the 1-D transport kernel by an explicit
3D photon-transport simulation (:mod:`spectraforge.studio.transport3d`). This module builds the
inputs of that simulation from the experiment and turns its tallies back into spectral radiance.

Pipeline for one block (all quantities in the block-local frame, mm / nm / mW)::

    launch   : jittered grid of points on the faces that face the source (top + sides); weight per mW
               = beam flux S x cos(inc) x dA x T_other (T_other = Fresnel/Beer of CLEAR objects in the
               way, 0 if a turbid/opaque object shadows the point; the block itself is excluded)
    for each enabled excitation step, in sequence order:
        excitation_pass  -> absorbed power A and fluence Phi per voxel, elastic face radiance
        per layer / dye:
            absorbed by the dye      A_d = A x <mu_a,d>/<mu_a,tot>        (absorption on; the MC used
                                     the line-averaged mu_a)  or  Phi x V x <mu_a,d>  (absorption off)
            photobleaching           dose = sigma(eps) x Phi x t_exposure, per voxel and dye, state kept
                                     across steps (survival volumes); mean survival over the exposure
                                     is (1 - e^-x)/x with x = phi_bleach x dose  (same model as v1)
            quenching                Phi_eff from optics (Stern-Volmer, per layer)
            FRET                     donor x (1 - E), acceptor channel += donor-absorbed x E x Phi_acc
            source channel (ex, layer, dye) = A_d x Phi_eff x survival x donor-factor  [mW]
        Raman channel (ex, aqueous layer)  = Phi x V x mu_R(ex) x strength          [mW]
    emission nodes (every ``emission_node_step_nm`` over the union of the camera bands):
        emission_pass with ALL channels of ALL steps at once -> face radiance (nu, nv, Ne, C)

Spectral radiance leaving a block pixel (face cell bilinearly interpolated at the pixel's hit point)::

    L(lam) = sum_c  T_c(lam; face,u,v) e_c(lam)       + L_el(face,u,v) (line(lam) + leak(lam))

``T_c`` is the emission-pass tally interpolated linearly in lambda between the nodes (it is smooth;
the sharp structure is in ``e_c``, the area-normalised emission density, applied on the fine grid) and
``L_el`` the elastic excitation-wavelength radiance of the excitation pass.

Approximations (documented in the module docstring of ``capture``)
-----------------------------------------------------------------
* Fluorescence yield uses the contract's energy form (no extra Stokes factor), as in v1; Raman includes
  ``lam_ex/lam_R`` like v1.
* ``subsurface_spread`` is inherent in the 3D transport and ignored for blocks; ``secondary_emission``
  (re-emission of reabsorbed fluorescence) is NOT modelled for blocks in v1.1 (the emission pass does
  reabsorb light, but does not re-emit it). Room light on block pixels is not modelled.
* The camera is represented by ONE direction per block (from the block centre to the camera); the
  radiance is tallied in a cone of ``acceptance_half_angle_deg`` around it (wide-field cameras close to
  the block therefore see a small angular error across the field of view).
* The camera's incoming ray bends in clear objects in front of a block are ignored for the face/cell
  look-up (the tracer's refracted hit point is used); their Beer/Fresnel attenuation is applied through
  the usual view transmission ``V`` of the capture engine.
* The excitation launch ignores refraction of the beam by clear objects in front of the block (straight
  line to the source, Fresnel and Beer-Lambert attenuation of the crossed clear media applied).
* Top-face tallies are NOT smoothed: the next-event (peel-off) estimator of ``transport3d`` is smooth enough
  (measured 2.4-2.6 % rms per 0.5 mm cell at 5e5 photons, versus 7-8 % for the old cone estimator at 2e6
  photons smoothed by a 0.75 mm Gaussian), so lateral structure (wedges, edges) is resolved to the exit cell.
  Because it needs ~5x fewer packets, the photon counts of ``volume_mc`` are scaled by ``PEEL_PHOTON_FACTOR``.
  Side faces (rare camera views) still use the cone estimator and its adaptive smoothing
  (:func:`smooth_sigma_mm`). Origin diagnostics are unsmoothed.
* Emission nodes are adaptive (:func:`choose_emission_nodes`): dense where the transport factor bends
  (absorption bands), at most ``2 x emission_node_step_nm`` apart elsewhere.
* Origin-depth / layer-fraction ground truth come from the emission pass at the middle emission node.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from spectraforge.studio import geometry, optics as optics_mod, tracer
from spectraforge.studio.contracts import BlockLaunch, RegionOptics, SPECTRAL_GRID_NM

GRID = SPECTRAL_GRID_NM
FACE_NAMES = ("top", "+x", "-x", "+z", "-z")
FACE_CODE = {"top": 0, "+x": 1, "-x": 2, "+z": 3, "-z": 4, "bottom": 5}
CODE_FACE = {v: k for k, v in FACE_CODE.items()}
SMOOTH_MM_AT_2M = 0.75        # Gaussian sigma (mm) of the face-tally smoothing at 2e6 photons (see smooth_sigma_mm)
PEEL_OFF = True               # next-event (peel-off) estimator towards the camera (transport3d); False = cone only
PEEL_PHOTON_FACTOR = 0.2      # photons per pass relative to ``volume_mc`` when the peel-off estimator is on (it reaches
                              # the old 2e6-photon + 0.75 mm-smoothing noise with ~5x fewer packets and no blur)
ADAPTIVE_NODES = True         # adaptive emission-node placement (False = uniform ``emission_node_step_nm``)
SMOOTH_SIGMA_MM = 0.35        # optional lateral Gaussian smoothing of TOP-face tallies with peel-off on (0 = off;
                              # a sub-cell value <= 0.5 exit cell removes ~40 % of the noise at no visible blur)
LAUNCH_TOP = 160              # jittered launch grid: top face n x n
LAUNCH_SIDE = (96, 24)        # side faces: (along, up)
_MODE_ID = {"sample": 0, "dark": 1, "white_reference": 2}


def _t3():
    from spectraforge.studio import transport3d
    return transport3d


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _bilinear(arr: np.ndarray, u_edges: np.ndarray, v_edges: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Bilinear interpolation of cell-centred ``arr`` (nu, nv, ...) at face coordinates ``(u, v)``;
    coordinates outside the cell-centre hull are clamped."""
    nu, nv = arr.shape[:2]
    uc = 0.5 * (u_edges[1:] + u_edges[:-1])
    vc = 0.5 * (v_edges[1:] + v_edges[:-1])
    du = (uc[-1] - uc[0]) / max(nu - 1, 1) if nu > 1 else 1.0
    dv = (vc[-1] - vc[0]) / max(nv - 1, 1) if nv > 1 else 1.0
    fu = np.clip((u - uc[0]) / du, 0.0, nu - 1.0)
    fv = np.clip((v - vc[0]) / dv, 0.0, nv - 1.0)
    i0 = np.minimum(np.floor(fu).astype(np.int64), max(nu - 2, 0))
    j0 = np.minimum(np.floor(fv).astype(np.int64), max(nv - 2, 0))
    i1, j1 = np.minimum(i0 + 1, nu - 1), np.minimum(j0 + 1, nv - 1)
    a, b = fu - i0, fv - j0
    shp = (-1,) + (1,) * (arr.ndim - 2)
    a, b = a.reshape(shp), b.reshape(shp)
    return (arr[i0, j0] * (1 - a) * (1 - b) + arr[i1, j0] * a * (1 - b)
            + arr[i0, j1] * (1 - a) * b + arr[i1, j1] * a * b)


def smooth_sigma_mm(n_photons: int, face: str = "top") -> float:
    """Lateral smoothing (mm) of a face tally.

    With the peel-off estimator the TOP face is smooth enough to be used unsmoothed (``SMOOTH_SIGMA_MM``, default 0:
    a smoothing of 0.75 mm would erase exactly the lateral structure - wedges, edges - the block studies look
    at).  Side faces (and every face when ``PEEL_OFF`` is False) are scored by the acceptance-cone estimator,
    whose noise falls as ``1/(sigma sqrt(N))``: ``SMOOTH_MM_AT_2M`` mm at 2e6 photons gives ~8 % noise per channel."""
    if PEEL_OFF and face == "top":
        return float(SMOOTH_SIGMA_MM)
    return float(np.clip(SMOOTH_MM_AT_2M * np.sqrt(2.0e6 / max(n_photons, 1)), 0.25, 3.0))


def _smooth_face(arr: np.ndarray, fg, sigma_mm: float) -> np.ndarray:
    """Gaussian smoothing of a (nu, nv, ...) face tally with ``sigma_mm`` (edge-replicated)."""
    if sigma_mm <= 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return arr
    from scipy.ndimage import gaussian_filter
    du = float(np.mean(np.diff(fg.u_edges_mm)))
    dv = float(np.mean(np.diff(fg.v_edges_mm)))
    return gaussian_filter(arr, sigma=(sigma_mm / du, sigma_mm / dv) + (0.0,) * (arr.ndim - 2), mode="nearest")


def _interp_matrix(nodes: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """(Ne, W) linear-interpolation weights from node values to wavelengths ``lam`` (clamped)."""
    ne = len(nodes)
    m = np.zeros((ne, len(lam)))
    for j in range(ne):
        e = np.zeros(ne)
        e[j] = 1.0
        m[j] = np.interp(lam, nodes, e)
    return m


def block_object_indices(exp) -> list[int]:
    return [i for i, o in enumerate(exp.objects) if o.visible and o.shape.kind == "layered_block"]


@dataclass
class Channel:
    ex: float
    layer: int
    kind: str                     # dye name or "raman"
    label: str                    # "<layer name>|<dye>" (volumes / UI)
    component: str                # "dye:<dye>@<layer or inclusion>" | "raman@<layer or inclusion>"
    fluo: bool
    incl: int = -1                # inclusion index (0-based) for channels emitted inside an inclusion, else -1

    @property
    def mc_label(self) -> str:
        return f"{self.ex:g}|{self.label}"


@dataclass
class BlockRun:
    """All 3D Monte Carlo results of one layered block for one capture configuration."""

    object_index: int
    object_id: str
    geom: object
    grid: object
    faces: dict
    layer_names: list[str]
    layer_mats: list[int]
    plane_coeffs: np.ndarray                         # (L, 3)
    footprint_mm: tuple[float, float]
    world_from_local: np.ndarray
    camera_dir_local: np.ndarray
    ex_list: list[float] = field(default_factory=list)
    elastic: dict = field(default_factory=dict)      # ex -> face -> (nu, nv)
    fluence: dict = field(default_factory=dict)      # ex -> (nx, nz, L, S)
    absorbed: dict = field(default_factory=dict)
    channels: list[Channel] = field(default_factory=list)
    sources: dict = field(default_factory=dict)      # (ex, label) -> (nx, nz, L, S) float32
    nodes: np.ndarray = None
    rad: dict = field(default_factory=dict)          # face -> (nu, nv, Ne, C) float32
    origin_depth: dict | None = None
    origin_frac: dict | None = None
    em_density: np.ndarray | None = None             # (C, W) float32
    launch_power: dict = field(default_factory=dict) # ex -> total launched mW
    diag: dict = field(default_factory=dict)
    # hit look-up (set by the engine): dense per-slot arrays
    mask: np.ndarray | None = None                   # (K, P) bool
    face_code: np.ndarray | None = None              # (K, P) int8
    hit_u: np.ndarray | None = None
    hit_v: np.ndarray | None = None
    hit_x: np.ndarray | None = None
    hit_z: np.ndarray | None = None
    _sub: dict = field(default_factory=dict)
    # ---- v1.2: inclusions ---------------------------------------------------------------------
    imap: object = None                               # contracts.InclusionMap | None
    incl_shapes: list = field(default_factory=list)   # schema.Inclusion (analytic shapes, ground truth)
    incl_mats: list = field(default_factory=list)
    incl_meta: list = field(default_factory=list)     # JSON-able description for the web
    incl_fluence: dict = field(default_factory=dict)  # ex -> (nx, ny, nz) float32 over the inclusion map
    incl_absorbed: dict = field(default_factory=dict)
    incl_sources: dict = field(default_factory=dict)  # (ex, label) -> (nx, ny, nz) float32
    origin_incl: dict | None = None                   # face -> (nu, nv, R)

    # ---------------------------------------------------------------- hit look-up
    def attach_hits(self, hit_info: dict, K: int, P: int) -> None:
        d = hit_info.get("block_hits", {}).get(self.object_index)
        self.mask = np.zeros((K, P), bool)
        self.face_code = np.full((K, P), -1, np.int8)
        self.hit_u = np.zeros((K, P))
        self.hit_v = np.zeros((K, P))
        self.hit_x = np.zeros((K, P))
        self.hit_z = np.zeros((K, P))
        if d is None:
            return
        k, p = d["layer"], d["pixel"]
        self.mask[k, p] = True
        self.face_code[k, p] = [FACE_CODE[f] for f in d["face"]]
        self.hit_u[k, p], self.hit_v[k, p] = d["u"], d["v"]
        self.hit_x[k, p], self.hit_z[k, p] = d["x"], d["z"]

    # ---------------------------------------------------------------- spectral radiance
    def _ex_index(self, ex: float):
        for i, e in enumerate(self.ex_list):
            if abs(e - ex) < 1e-6:
                return i
        return None

    def _channel_idx(self, ex: float) -> list[int]:
        return [i for i, c in enumerate(self.channels) if abs(c.ex - ex) < 1e-6]

    def components(self, ex: float, codes: np.ndarray, u: np.ndarray, v: np.ndarray, ws_idx: np.ndarray,
                   line_w: np.ndarray, leak_w: np.ndarray, keep: bool, fluorescence: bool = True,
                   elastic: bool = True) -> list[tuple[str, np.ndarray, bool]]:
        """Spectral radiance components (name, (n, wn) float32, is_fluorescence) of block pixels hitting
        face ``codes`` at ``(u, v)`` for excitation ``ex``, on the fine-grid window ``ws_idx``.
        ``keep=False`` merges into ``elastic`` and one ``dye`` array (cheaper)."""
        n, wn = len(codes), len(ws_idx)
        lam = GRID[ws_idx]
        cidx = self._channel_idx(ex) if (fluorescence and self.nodes is not None) else []
        el_arr = np.zeros((n, 1), np.float32)
        comp = {c: np.zeros((n, wn), np.float32) for c in cidx} if keep else None
        tot_fl = np.zeros((n, wn), np.float32)
        wmat = _interp_matrix(self.nodes, lam) if cidx else None
        ek = self.em_density[cidx][:, ws_idx] if cidx else None
        el_n = np.zeros(n)
        for code in np.unique(codes):
            if code < 0 or code > 4:
                continue
            f = CODE_FACE[int(code)]
            sel = np.nonzero(codes == code)[0]
            fg = self.faces[f]
            if elastic and ex in self.elastic:
                el_n[sel] = _bilinear(self.elastic[ex][f], fg.u_edges_mm, fg.v_edges_mm, u[sel], v[sel])
            if cidx:
                key = (ex, f)
                sub = self._sub.get(key)
                if sub is None:
                    sub = self.rad[f][:, :, :, cidx]
                    self._sub[key] = sub
                t = _bilinear(sub, fg.u_edges_mm, fg.v_edges_mm, u[sel], v[sel])        # (m, Ne, Cx)
                a = np.einsum("nec,ew->ncw", t, wmat, optimize=True) * ek[None]          # (m, Cx, wn)
                if keep:
                    for ci, c in enumerate(cidx):
                        comp[c][sel] = a[:, ci]
                else:
                    tot_fl[sel] = a.sum(axis=1)
        out: list[tuple[str, np.ndarray, bool]] = []
        if elastic and ex in self.elastic:
            el = el_n[:, None]
            if keep:
                out.append(("elastic", (el * line_w[None, :]).astype(np.float32), False))
                out.append(("leak_source", (el * leak_w[None, :]).astype(np.float32), False))
            else:
                out.append(("elastic", (el * (line_w + leak_w)[None, :]).astype(np.float32), False))
        if keep:
            for c in cidx:
                out.append((self.channels[c].component, comp[c], True))
        elif cidx:
            out.append(("dye", tot_fl, True))
        return out

    # ---------------------------------------------------------------- ground truth
    def layer_geometry(self, x: np.ndarray, z: np.ndarray):
        """Interface heights (L, n) of every layer's top at block-local ``(x, z)``."""
        pc = self.plane_coeffs
        return pc[:, 0:1] + pc[:, 1:2] * x[None, :] + pc[:, 2:3] * z[None, :]

    def gt_values(self, codes: np.ndarray, u, v, x, z) -> dict[str, np.ndarray]:
        """Block ground truth at hit points (see capture.GROUND_TRUTH docs)."""
        tops = self.layer_geometry(x, z)
        n = len(x)
        out: dict[str, np.ndarray] = {}
        prev = np.zeros(n)
        for li, name in enumerate(self.layer_names):
            out[f"layer_thickness:{name}"] = np.maximum(tops[li] - prev, 0.0)
            out[f"layer_depth:{name}"] = np.maximum(tops[-1] - tops[li], 0.0)
            prev = tops[li]
        out["block_face"] = codes.astype(float)
        od = np.full(n, np.nan)
        of = np.full((n, len(self.layer_names)), np.nan)
        for code in np.unique(codes):
            if code < 0 or code > 4:
                continue
            f = CODE_FACE[int(code)]
            sel = np.nonzero(codes == code)[0]
            fg = self.faces[f]
            if self.origin_depth is not None and f in self.origin_depth:
                od[sel] = _bilinear(self.origin_depth[f], fg.u_edges_mm, fg.v_edges_mm, u[sel], v[sel])
            if self.origin_frac is not None and f in self.origin_frac:
                of[sel] = _bilinear(self.origin_frac[f], fg.u_edges_mm, fg.v_edges_mm, u[sel], v[sel])
        out["origin_depth_mm"] = od
        for li, name in enumerate(self.layer_names):
            out[f"origin_layer_fraction:{name}"] = of[:, li]
        if self.incl_shapes:
            from spectraforge.studio import inclusions as incl_geo
            top = tops[-1]
            oi = np.full((n, len(self.incl_shapes)), np.nan)
            for code in np.unique(codes):
                if code < 0 or code > 4 or self.origin_incl is None:
                    continue
                f = CODE_FACE[int(code)]
                sel = np.nonzero(codes == code)[0]
                if f in self.origin_incl:
                    fg = self.faces[f]
                    oi[sel] = _bilinear(self.origin_incl[f], fg.u_edges_mm, fg.v_edges_mm, u[sel], v[sel])
            for j, shp in enumerate(self.incl_shapes):
                mask, depth = incl_geo.footprint_depth(shp, x, z, top)
                out[f"inclusion_mask:{shp.name}"] = mask.astype(float)
                out[f"inclusion_depth_mm:{shp.name}"] = depth
                out[f"origin_fraction:{shp.name}"] = oi[:, j]
        return out

    def host_volumes(self) -> np.ndarray:
        """Volume (mm^3) of the HOST material of every layer voxel (voxel volume minus the inclusion voxels in it)."""
        vol = self.voxel_volumes()
        if self.imap is None:
            return vol
        from spectraforge.studio import transport3d
        return np.maximum(vol - transport3d.inclusion_covered_volume(self.geom, self.grid, self.imap), 0.0)

    def voxel_volumes(self) -> np.ndarray:
        """Exact voxel volumes (nx, nz, L, S) of the layer-conforming grid (mm³)."""
        g = self.grid
        xc = 0.5 * (g.x_edges_mm[1:] + g.x_edges_mm[:-1])
        zc = 0.5 * (g.z_edges_mm[1:] + g.z_edges_mm[:-1])
        dx, dz = np.diff(g.x_edges_mm), np.diff(g.z_edges_mm)
        X, Z = np.meshgrid(xc, zc, indexing="ij")
        pc = self.plane_coeffs
        tops = pc[:, 0, None, None] + pc[:, 1, None, None] * X[None] + pc[:, 2, None, None] * Z[None]   # (L,nx,nz)
        thick = np.diff(np.concatenate([np.zeros((1,) + X.shape), tops]), axis=0)
        vol = (np.maximum(thick, 0.0) * (dx[:, None] * dz[None, :])[None]) / g.sublayers      # (L, nx, nz)
        return np.broadcast_to(np.moveaxis(vol, 0, -1)[..., None], g.shape).copy()


# --------------------------------------------------------------------------------------------
# launch sampling
# --------------------------------------------------------------------------------------------
def _face_samples(shape, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jittered sample points on the top and side faces (local frame): points, unit normals, areas."""
    X, Z = shape.footprint_mm
    a, b, c = geometry.block_plane_coeffs(shape)[-1]
    nn = float(np.sqrt(1.0 + b * b + c * c))

    def top(x, z):
        return a + b * x + c * z

    pts, nrm, area = [], [], []
    n = LAUNCH_TOP
    ix, iz = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = -X / 2 + (ix + rng.random((n, n))) * X / n
    z = -Z / 2 + (iz + rng.random((n, n))) * Z / n
    pts.append(np.stack([x.ravel(), top(x, z).ravel(), z.ravel()], 1))
    nrm.append(np.tile([-b / nn, 1.0 / nn, -c / nn], (n * n, 1)))
    area.append(np.full(n * n, (X / n) * (Z / n) * nn))
    na, nu = LAUNCH_SIDE
    ia, iu = np.meshgrid(np.arange(na), np.arange(nu), indexing="ij")
    ja, ju = ia + rng.random((na, nu)), iu + rng.random((na, nu))
    for sgn in (1.0, -1.0):
        # +/-x faces: along z
        zz = -Z / 2 + ja * Z / na
        h = np.maximum(top(sgn * X / 2, zz), 0.0)
        pts.append(np.stack([np.full(zz.size, sgn * X / 2), (ju / nu * h).ravel(), zz.ravel()], 1))
        nrm.append(np.tile([sgn, 0.0, 0.0], (zz.size, 1)))
        area.append(((Z / na) * (h / nu)).ravel())
        # +/-z faces: along x
        xx = -X / 2 + ja * X / na
        h = np.maximum(top(xx, sgn * Z / 2), 0.0)
        pts.append(np.stack([xx.ravel(), (ju / nu * h).ravel(), np.full(xx.size, sgn * Z / 2)], 1))
        nrm.append(np.tile([0.0, 0.0, sgn], (xx.size, 1)))
        area.append(((X / na) * (h / nu)).ravel())
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(area)


@dataclass
class _LaunchSet:
    pts_local: np.ndarray
    dirs_local: np.ndarray
    w_per_mW: np.ndarray         # launched power per mW of source power (before Beer of clear media)
    clear_path: np.ndarray       # (N, M) shadow-ray path through each clear material


def build_launch(eng, ctx, oi: int, R: np.ndarray, p: np.ndarray, rng, n_top: float) -> _LaunchSet:
    shape = eng.exp.objects[oi].shape
    pl, nl, area = _face_samples(shape, rng)
    pw = geometry._rowmat(pl, R.T) + p
    nw = geometry._rowmat(nl, R.T)
    keep = area > 0
    pl, nl, pw, nw, area = pl[keep], nl[keep], pw[keep], nw[keep], area[keep]
    sh = tracer._shadow(ctx, eng.src, pw, nw, np.full(len(pw), -1, np.int64), np.ones(len(pw)),
                        np.full(len(pw), n_top))
    # beam flux along its direction x cos(incidence) x area x transmission of clear objects in front
    w = sh["E"] * area * sh["T"]
    u_w = sh["u"]
    return _LaunchSet(pl, -geometry._rowmat(u_w, R), w, sh["path"])


# --------------------------------------------------------------------------------------------
# running a block
# --------------------------------------------------------------------------------------------
def _region_optics(eng, layer_mats: list[int], lam_nm: float, line: np.ndarray | None,
                   incl_mats: list[int] | None = None) -> RegionOptics:
    def one(m):
        o = eng.optics[m]
        mua = (float(optics_mod.average_over_line(o.mua_per_mm, line)) if line is not None
               else float(o.interp(o.mua_per_mm, lam_nm)))
        return mua, float(o.interp(o.mus_per_mm, lam_nm)), o.g, o.n

    mua, mus, g, n = (list(t) for t in zip(*[one(m) for m in layer_mats]))
    if not incl_mats:
        return RegionOptics(float(lam_nm), np.array(mua), np.array(mus), np.array(g), np.array(n))
    im, is_, ig, _ = (list(t) for t in zip(*[one(m) for m in incl_mats]))
    return RegionOptics(float(lam_nm), np.array(mua), np.array(mus), np.array(g), np.array(n),
                        np.array(im), np.array(is_), np.array(ig))


def _seed(acq_seed: int, mode: str, *tags: int) -> int:
    ss = np.random.SeedSequence([int(acq_seed) & 0xFFFFFFFF, _MODE_ID[mode], 0x3D3D] + [int(t) for t in tags])
    return int(ss.generate_state(1)[0] & 0x7FFFFFFF)


def run_block(eng, steps, oi: int, scene, mode: str, frac: float, progress=None, cancel=None) -> BlockRun | None:
    """Run the full 3D simulation of block ``oi``; ``None`` if cancelled. ``progress(f in 0..1, msg)``."""
    import time
    from spectraforge.studio import capture as cap
    t_start = time.perf_counter()

    exp, tg = eng.exp, eng.tg
    acq = exp.acquisition
    vs = acq.volume_mc
    obj = exp.objects[oi]
    shape = obj.shape
    t3 = _t3()
    geom = t3.build_block_geometry(exp, oi)
    grid = t3.make_voxel_grid(geom, vs)
    faces = t3.make_face_grids(geom, vs)
    R, p = geometry.object_transform(obj.transform)
    coeffs = geometry.block_plane_coeffs(shape)
    X, Z = shape.footprint_mm
    centre_local = np.array([0.0, 0.5 * float(coeffs[-1, 0]), 0.0])       # mid-height above the block centre
    cam_vec = geometry._rowvec(np.asarray(exp.camera().position_mm, float) - (R @ centre_local + p), R)
    cam_dir = cam_vec / max(np.linalg.norm(cam_vec), 1e-12)
    W4 = np.eye(4)
    W4[:3, :3], W4[:3, 3] = R, p
    mat_ids = [m.id for m in eng.mats]
    layer_mats = [mat_ids.index(l.material_id) for l in shape.layers]
    run = BlockRun(oi, obj.id, geom, grid, faces, [l.name for l in shape.layers], layer_mats, coeffs,
                   (float(X), float(Z)), W4, cam_dir)
    L, S = len(layer_mats), grid.sublayers
    imap = None
    if shape.inclusions:
        imap = t3.make_inclusion_map(geom, vs)
        run.imap = imap
        run.incl_shapes = list(shape.inclusions)
        run.incl_mats = [mat_ids.index(i.material_id) for i in shape.inclusions]
        run.incl_meta = [{"name": i.name, "material_id": i.material_id, "kind": i.kind,
                          "center_mm": list(i.center_mm), "radius_mm": i.radius_mm,
                          "semi_axes_mm": list(i.semi_axes_mm), "size_mm": list(i.size_mm), "axis": i.axis,
                          "length_mm": i.length_mm, "region": k + 1} for k, i in enumerate(shape.inclusions)]
    ikw = {} if imap is None else {"inclusion_map": imap}
    vol = run.host_volumes()
    pf = PEEL_PHOTON_FACTOR if PEEL_OFF else 1.0
    n_exc = max(10_000, int(round(vs.excitation_photons * frac * pf)))
    n_em = max(10_000, int(round(vs.emission_photons * frac * pf)))
    nst = len(steps)

    ctx, _mc, _sb = tracer.make_context(exp, scene, tg)
    rng = np.random.default_rng(np.random.SeedSequence([int(acq.seed) & 0xFFFFFFFF, 0xB10C, oi]))
    ls = (build_launch(eng, ctx, oi, R, p, rng, float(eng.mats[layer_mats[-1]].refractive_index))
          if mode != "dark" else None)

    def prog(f, msg):                        # excitation = first half of the block's budget, emission = second
        if progress is not None:
            progress(min(max(f, 0.0), 1.0), msg)

    bleach_state: dict[tuple[int, str], np.ndarray] = {}
    src_parts: list[tuple[Channel, np.ndarray]] = []
    sensitive = mode == "sample"
    for si, st in enumerate(steps):
        if cancel is not None and cancel():
            return None
        prog(0.5 * si / max(nst, 1), f"simulating {st.ex_nm:g} nm ({si + 1}/{nst}) · 3D excitation")
        run.ex_list.append(st.ex_nm)
        P = st.power_mW
        if ls is None or P <= 0:
            run.launch_power[st.ex_nm] = 0.0
            run.fluence[st.ex_nm] = np.zeros(grid.shape, np.float32)
            run.absorbed[st.ex_nm] = np.zeros(grid.shape, np.float32)
            run.elastic[st.ex_nm] = {f: np.zeros((len(fg.u_edges_mm) - 1, len(fg.v_edges_mm) - 1))
                                     for f, fg in faces.items()}
            _zero_incl(run, st.ex_nm)
            continue
        w = ls.w_per_mW * P
        if tg.absorption and ls.clear_path.shape[1] > 0 and np.any(ls.clear_path > 0):
            tau = ls.clear_path @ eng.mua_all[:, st.beer_idx]
            w = w * (np.exp(-tau) @ st.beer_w)
        used = w > 0
        run.launch_power[st.ex_nm] = float(w.sum())
        if not used.any():
            run.fluence[st.ex_nm] = np.zeros(grid.shape, np.float32)
            run.absorbed[st.ex_nm] = np.zeros(grid.shape, np.float32)
            run.elastic[st.ex_nm] = {f: np.zeros((len(fg.u_edges_mm) - 1, len(fg.v_edges_mm) - 1))
                                     for f, fg in faces.items()}
            _zero_incl(run, st.ex_nm)
            continue
        launch = BlockLaunch(st.ex_nm, ls.pts_local[used], ls.dirs_local[used], w[used])
        ro = _region_optics(eng, layer_mats, st.ex_nm, st.line, run.incl_mats)
        exc = t3.excitation_pass(geom, ro, launch, cam_dir, vs, n_photons=n_exc,
                                 seed=_seed(acq.seed, mode, 1, si), toggles=tg, peel_off=PEEL_OFF, **ikw)
        fl = np.asarray(exc.fluence_mW_mm2, float)
        ab = np.asarray(exc.absorbed_mW, float)
        run.fluence[st.ex_nm] = fl.astype(np.float32)
        run.absorbed[st.ex_nm] = ab.astype(np.float32)
        run.elastic[st.ex_nm] = {f: _smooth_face(np.asarray(a, float), faces[f], smooth_sigma_mm(n_exc, f))
                                 for f, a in exc.elastic_radiance.items()}
        run.diag[f"exc_{st.ex_nm:g}"] = {"R": exc.diffuse_reflectance, "T": exc.transmittance,
                                         "A": exc.absorbed_fraction}
        fli = abi = None
        if imap is not None:
            run.incl_fluence[st.ex_nm] = np.asarray(exc.inclusion_fluence_mW_mm2, np.float32)
            run.incl_absorbed[st.ex_nm] = np.asarray(exc.inclusion_absorbed_mW, np.float32)
            fli = np.asarray(exc.inclusion_fluence_mW_mm2, float).reshape(-1)[imap.occ_flat]
            abi = np.asarray(exc.inclusion_absorbed_mW, float).reshape(-1)[imap.occ_flat]
            run.diag[f"exc_{st.ex_nm:g}"]["inclusion_absorbed_fraction"] = list(
                exc.info.get("inclusion_absorbed_fraction", []))
        _build_sources(eng, run, st, fl, ab, vol, bleach_state, sensitive, src_parts, cap, fli, abi)
    prog(0.5, f"simulating {steps[-1].ex_nm:g} nm ({nst}/{nst}) · 3D excitation")
    if cancel is not None and cancel():
        return None
    t_exc = time.perf_counter() - t_start

    # ---- emission pass ------------------------------------------------------------------------
    C = len(src_parts)
    union = np.zeros(len(GRID), bool)
    for st in steps:
        union |= st.wsel
    # emission nodes only where some channel actually emits (density > 1e-3 of its peak) inside the bands
    ek_all = {}
    for c in (c for c, _ in src_parts):
        e = (_raman_shape_for(steps, c.ex) if c.kind == "raman" else _chan_mat(eng, run, c).emission_density[c.kind])
        ek_all[(c.ex, c.kind, c.layer, c.incl)] = e
    emit = np.zeros(len(GRID), bool)
    for e in ek_all.values():
        emit |= e > 1e-3 * float(np.max(e))
    idx = np.nonzero(union & emit)[0]
    if idx.size == 0:
        idx = np.nonzero(union)[0]
    lo, hi = (float(GRID[idx[0]]), float(GRID[idx[-1]])) if idx.size else (450.0, 650.0)
    if ADAPTIVE_NODES:
        nodes = choose_emission_nodes(
            transport_proxy(eng, layer_mats, run.plane_coeffs, {c.layer for c, _ in src_parts if c.layer >= 0}), lo, hi,
            vs.emission_node_step_nm)
    else:
        nodes = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / vs.emission_node_step_nm)) + 1))
    ne = len(nodes)
    run.nodes = nodes
    run.channels = [c for c, _ in src_parts]
    for c, arr in src_parts:
        if c.incl >= 0:
            run.incl_sources[(c.ex, c.label)] = imap.dense(np.asarray(arr, np.float32))
            continue
        full = np.zeros(grid.shape, np.float32)
        full[:, :, c.layer, :] = arr
        run.sources[(c.ex, c.label)] = full
    if C == 0:
        run.rad = {f: np.zeros((len(fg.u_edges_mm) - 1, len(fg.v_edges_mm) - 1, ne, 0), np.float32)
                   for f, fg in faces.items()}
        run.em_density = np.zeros((0, len(GRID)), np.float32)
        return run
    srcarr = np.zeros((C,) + grid.shape)
    isrc = None if not any(c.incl >= 0 for c, _ in src_parts) else np.zeros((C,) + tuple(imap.shape))
    for ci, (c, arr) in enumerate(src_parts):
        if c.incl >= 0:
            isrc[ci].reshape(-1)[imap.occ_flat] = arr
        else:
            srcarr[ci, :, :, c.layer, :] = arr
    ekw = dict(ikw)
    if isrc is not None:
        ekw["inclusion_sources"] = isrc
    labels = [c.mc_label for c in run.channels]
    parts: dict[str, list[np.ndarray]] = {f: [] for f in faces}
    mid = ne // 2
    for j, lam in enumerate(nodes):
        if cancel is not None and cancel():
            return None
        prog(0.5 + 0.5 * j / ne, f"emission node {lam:g} nm ({j + 1}/{ne}) · 3D")
        em = t3.emission_pass(geom, [_region_optics(eng, layer_mats, float(lam), None, run.incl_mats)], srcarr, labels,
                              cam_dir, vs, n_photons=n_em, seed=_seed(acq.seed, mode, 2), toggles=tg,
                              peel_off=PEEL_OFF, **ekw)   # same seed per node = common random numbers
        for f in faces:
            parts[f].append(np.asarray(em.radiance[f], float))
        if j == mid:
            run.origin_depth = None if em.origin_depth_mm is None else {
                f: np.asarray(a, float) for f, a in em.origin_depth_mm.items()}
            run.origin_frac = None if em.origin_layer_fraction is None else {
                f: np.asarray(a, float) for f, a in em.origin_layer_fraction.items()}
            oif = getattr(em, "origin_inclusion_fraction", None)
            run.origin_incl = None if oif is None else {f: np.asarray(a, float) for f, a in oif.items()}
    prog(1.0, f"emission node {nodes[-1]:g} nm ({ne}/{ne}) · 3D")
    run.rad = {f: _smooth_face(np.concatenate(parts[f], axis=2), faces[f], smooth_sigma_mm(n_em, f)).astype(np.float32)
               for f in faces}
    ek = np.zeros((C, len(GRID)), np.float32)
    for ci, c in enumerate(run.channels):
        ek[ci] = (_raman_shape_for(steps, c.ex) if c.kind == "raman"
                  else _chan_mat(eng, run, c).emission_density[c.kind])
    run.em_density = ek
    run.diag["seconds"] = {"excitation": t_exc, "emission": time.perf_counter() - t_start - t_exc,
                           "n_nodes": int(ne), "n_channels": int(C), "n_photons_exc": int(n_exc),
                           "n_photons_em": int(n_em)}
    return run


def _chan_mat(eng, run: BlockRun, c: Channel):
    """MaterialOptics of the material that emits channel ``c`` (a layer's or an inclusion's)."""
    return eng.optics[run.incl_mats[c.incl] if c.incl >= 0 else run.layer_mats[c.layer]]


def _zero_incl(run: BlockRun, ex: float) -> None:
    if run.imap is not None:
        run.incl_fluence[ex] = np.zeros(run.imap.shape, np.float32)
        run.incl_absorbed[ex] = np.zeros(run.imap.shape, np.float32)


PROXY_TAU_CAP = 4.0           # per-layer optical depth beyond which the proxy stops changing (sources sit near the surface)


def transport_proxy(eng, layer_mats: list[int], plane_coeffs: np.ndarray, source_layers=None) -> np.ndarray:
    """Cheap wavelength dependence of the block's transport factor on the fine grid, shape ``(n, W)``.

    One row per layer that holds emitters: ``exp(-D(lambda))`` with ``D = sum mu_eff,j t_j`` over the half of
    the emitting layer itself and every layer ABOVE it (``mu_eff = sqrt(3 mu_a (mu_a + mu_s'))``, mean layer
    thickness, each term capped at ``PROXY_TAU_CAP``: deeper light does not reach the surface).  It tracks
    how fast the true (Monte-Carlo) transport factor of the channels in that layer varies between emission
    wavelengths and drives the adaptive node placement (:func:`choose_emission_nodes`)."""
    L = len(layer_mats)
    prev, thick = np.zeros(3), []
    for pc in plane_coeffs:
        thick.append(max(float(pc[0] - prev[0]), 0.0))
        prev = pc
    mu_eff = []
    for m in layer_mats:
        o = eng.optics[m]
        mua, mus = np.asarray(o.mua_per_mm, float), np.asarray(o.mus_per_mm, float)
        mu_eff.append(np.sqrt(3.0 * mua * (mua + mus * (1.0 - float(o.g)))))
    rows = []
    for c in (range(L) if source_layers is None else sorted(set(source_layers))):
        D = np.minimum(mu_eff[c] * 0.5 * thick[c], PROXY_TAU_CAP)
        for j in range(c + 1, L):
            D = D + np.minimum(mu_eff[j] * thick[j], PROXY_TAU_CAP)
        rows.append(np.exp(-D))
    return np.array(rows) if rows else np.ones((1, len(GRID)))


def choose_emission_nodes(f: np.ndarray, lo: float, hi: float, step_nm: float, tol: float = 0.03,
                          min_step_nm: float | None = None, max_step_nm: float | None = None,
                          budget: int | None = None) -> np.ndarray:
    """Adaptive emission-node wavelengths in ``[lo, hi]``, never more than a uniform ``step_nm`` grid.

    The per-channel transport factor ``T(lambda)`` is interpolated linearly between the nodes (the sharp
    spectral structure sits in the emission density, applied on the fine grid), so nodes are only needed
    where ``T`` bends.  Starting from the two ends, the interval whose linear chord misses the proxy ``f``
    (rows of :func:`transport_proxy`; relative to the larger of ``f`` and 5 % of its peak, at ANY fine-grid
    wavelength inside) by the most is bisected, until every interval is within ``tol`` or ``budget`` nodes
    (default: the node count of the uniform ``step_nm`` grid) are used.  Intervals wider than ``max_step_nm``
    (default ``2 step_nm``) are always split and none is split below ``min_step_nm`` (default ``step_nm / 4``;
    the bisection halves, so the finest interval is between half and one of it).  Smooth transport therefore
    needs fewer nodes than the uniform grid, structured transport (hemoglobin bands) gets the same number
    but placed where it matters."""
    import heapq

    f = np.atleast_2d(np.asarray(f, float))
    min_step = max(float(min_step_nm if min_step_nm is not None else step_nm / 4.0), float(GRID[1] - GRID[0]))
    max_step = float(max_step_nm if max_step_nm is not None else 2.0 * step_nm)
    if hi - lo < min_step:
        return np.array([lo, lo + min_step]) if hi <= lo else np.array([lo, hi])
    budget = int(budget) if budget is not None else int(np.ceil((hi - lo) / step_nm - 1e-9)) + 1
    budget = max(budget, int(np.ceil((hi - lo) / max_step - 1e-9)) + 1, 2)
    inb = (GRID >= lo) & (GRID <= hi)
    floor = 0.05 * f[:, inb].max(axis=1, keepdims=True)

    def err(a: float, b: float) -> float:
        sel = (GRID >= a) & (GRID <= b)
        lam = GRID[sel]
        fa = np.array([np.interp(a, GRID, r) for r in f])[:, None]
        fb = np.array([np.interp(b, GRID, r) for r in f])[:, None]
        chord = fa + (fb - fa) * ((lam - a) / (b - a))[None, :]
        return float(np.max(np.abs(f[:, sel] - chord) / np.maximum(f[:, sel], floor)))

    pts = {lo, hi}
    heap = [(-np.inf if hi - lo > max_step else -err(lo, hi), lo, hi)]
    while heap:
        e, a, b = heapq.heappop(heap)
        forced = e == -np.inf
        if not forced and (-e <= tol or len(pts) >= budget):
            break                                   # the worst interval is already within tolerance / budget used
        if b - a <= min_step + 1e-9:
            continue
        m = 0.5 * (a + b)
        pts.add(m)
        for lo2, hi2 in ((a, m), (m, b)):
            heapq.heappush(heap, (-np.inf if hi2 - lo2 > max_step else -err(lo2, hi2), lo2, hi2))
    return np.array(sorted(pts))


def _raman_shape_for(steps, ex: float) -> np.ndarray:
    for st in steps:
        if abs(st.ex_nm - ex) < 1e-6 and st.raman is not None:
            return st.raman
    return np.zeros(len(GRID))


def _build_sources(eng, run: BlockRun, st, fl, ab, vol, bleach_state, sensitive, out_parts, cap,
                   fl_i=None, ab_i=None) -> None:
    """Fluorescence / Raman source channels of one excitation step (also advances the bleaching state)."""
    tg, mats = eng.tg, eng.mats
    ex = st.ex_nm
    active = tg.photobleaching and sensitive and st.power_mW > 0
    t_total = st.exposure_ms * 1e-3 * len(st.centres)
    # regions: every layer (host material, arrays (nx, nz, S)), then every inclusion (arrays over the occupied voxels)
    regions = [(li, m, run.layer_names[li], li, -1, fl[:, :, li, :], ab[:, :, li, :], vol[:, :, li, :])
               for li, m in enumerate(run.layer_mats)]
    if run.imap is not None and fl_i is not None:
        for r, m in enumerate(run.incl_mats):
            msk = (run.imap.occ_region == r + 1).astype(float)
            regions.append((("incl", r), m, run.imap.names[r], -1, r, fl_i * msk, ab_i * msk,
                            run.imap.occ_volume_mm3 * msk))
    for key, m, name, li, ir, fl_l, ab_l, v_l in regions:
        opt, ms = eng.optics[m], mats[m]
        dyes = list(opt.mua_dyes_per_mm)
        if tg.fluorescence and dyes:
            mua_line = {d: float(optics_mod.average_over_line(a_, st.line)) for d, a_ in opt.mua_dyes_per_mm.items()}
            mua_tot = float(optics_mod.average_over_line(opt.mua_per_mm, st.line))
            absd, beff = {}, {}
            for d in dyes:
                absd[d] = ab_l * (mua_line[d] / mua_tot) if tg.absorption else fl_l * v_l * mua_line[d]
                s0 = bleach_state.get((key, d))
                if s0 is None:
                    s0 = np.ones_like(fl_l)
                if active:
                    ekey = (d, round(ex, 3), st.index)
                    eps = eng.eps_line_cache.get(ekey)
                    if eps is None:
                        spec = eng.fl[d]
                        eps = spec.extinction * cap.EXTINCTION_REFERENCE_M_CM * float(
                            optics_mod.average_over_line(optics_mod.to_legacy(spec).excitation(GRID), st.line))
                        eng.eps_line_cache[ekey] = eps
                    photons_cm2_s = fl_l * 100.0 * ex * cap.sensor_mod.PHOTONS_PER_MW_S_PER_NM
                    dose = cap._SIGMA_CM2_PER_EPS * eps * photons_cm2_s * t_total
                    x = eng.phi_b[d] * dose
                    mean_f = np.where(x > 1e-9, -np.expm1(-x) / np.maximum(x, 1e-300), 1.0 - 0.5 * x)
                    beff[d] = s0 * mean_f
                    bleach_state[(key, d)] = s0 * np.exp(-x)
                else:
                    beff[d] = s0
                    bleach_state[(key, d)] = s0
            conc = {dd.fluorophore: 0.0 for dd in ms.dyes}
            for dd in ms.dyes:
                conc[dd.fluorophore] += dd.concentration_uM
            srcs = {}
            for d in dyes:
                donor = 1.0
                if tg.fret:
                    for pair in ms.fret:
                        if pair.donor == d and pair.acceptor in mua_line:
                            kc = pair.k_per_uM * conc.get(pair.acceptor, 0.0)
                            donor = donor * (1.0 - kc / (1.0 + kc))
                srcs[d] = absd[d] * opt.quantum_yields[d] * beff[d] * donor
            if tg.fret:
                for pair in ms.fret:
                    if pair.donor in mua_line and pair.acceptor in mua_line:
                        kc = pair.k_per_uM * conc.get(pair.acceptor, 0.0)
                        eff = kc / (1.0 + kc)
                        srcs[pair.acceptor] = srcs[pair.acceptor] + (
                            absd[pair.donor] * eff * beff[pair.donor] * opt.quantum_yields[pair.acceptor])
            for d in dyes:
                if float(srcs[d].max()) > 0.0:
                    out_parts.append((Channel(ex, li, d, f"{name}|{d}", f"dye:{d}@{name}", True, ir), srcs[d]))
        if ms.aqueous and tg.raman and st.raman is not None and sensitive:
            sr = fl_l * v_l * cap.raman_mu_phi_per_mm(ex) * ms.raman_strength
            if float(sr.max()) > 0.0:
                out_parts.append((Channel(ex, li, "raman", f"{name}|raman", f"raman@{name}", False, ir), sr))


# --------------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------------
_CACHE: OrderedDict = OrderedDict()      # key -> (present object indices, runs, excitation list)
_CACHE_LOCK = threading.Lock()
CACHE_SIZE = 4


def exp_digest(exp) -> str:
    """Digest of everything that shapes a block run EXCEPT the source's excitation sequence (the steps
    are part of the cache key separately, so a single-excitation preview and the explain/ROI calls that
    follow it can find each other)."""
    srcs = [s.model_copy(update={"sequence": []}) for s in exp.sources]
    light = exp.model_copy(update={"sources": srcs, "name": "", "description": ""})
    return hashlib.sha1(light.model_dump_json().encode()).hexdigest()


def cache_key(exp, mode: str, frac: float, steps) -> tuple:
    """(experiment digest without the sequence, mode, photon fraction, per-step signature)."""
    return (exp_digest(exp), mode, round(float(frac), 6),
            tuple((round(s.ex_nm, 6), round(float(s.power_mW), 9), round(float(s.exposure_ms), 6)) for s in steps))


def cached_excitations(exp, mode: str, frac: float) -> list[float] | None:
    """Excitation wavelengths of the most recently used cached block run that belongs to ``exp`` (same
    experiment apart from the excitation sequence), ``mode`` and photon fraction; ``None`` if there is none.
    ``explain_pixel`` / ROI decompositions use it to explain exactly what was previewed or captured."""
    dg = exp_digest(exp)
    r6 = round(float(frac), 6)
    with _CACHE_LOCK:
        for key in reversed(_CACHE):
            if key[0] == dg and key[1] == mode and key[2] == r6:
                return [float(e[0]) for e in key[3]]
    return None


def get_runs(eng, steps, scene, mode: str, frac: float, progress=None, cancel=None):
    """Cached block runs for the blocks visible in ``eng.hit``; ``None`` if cancelled."""
    present = tuple(sorted(eng.hit.info.get("block_hits", {})))
    if not present:
        return []
    key = cache_key(eng.exp, mode, frac, steps)
    with _CACHE_LOCK:
        ent = _CACHE.get(key)
        if ent is not None and set(present) <= set(ent[0]):
            _CACHE.move_to_end(key)
            return [r for r in ent[1] if r.object_index in present]
    # a cached entry that covers fewer blocks is replaced by one covering the union
    todo = tuple(sorted(set(present) | (set(ent[0]) if ent is not None else set())))
    runs = []
    nblk = len(todo)
    for bi, oi in enumerate(todo):
        sub = None if progress is None else (lambda f, msg, bi=bi: progress((bi + f) / nblk, msg))
        r = run_block(eng, steps, oi, scene, mode, frac, sub, cancel)
        if r is None:
            return None
        runs.append(r)
    with _CACHE_LOCK:
        _CACHE[key] = (todo, runs, [s.ex_nm for s in steps])
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return [r for r in runs if r.object_index in present]


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
