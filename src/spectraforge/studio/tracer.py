"""Camera ray tracing → :class:`~spectraforge.studio.contracts.HitBuffer`.

Camera model
------------
Pinhole at the lens position with the thin-lens image distance ``v = f s / (s − f)``
(``s = focus_distance_mm``). The sensor measures ``width_px·pitch·binning`` × ``height_px·pitch·binning``
(µm) and the *requested* ``width × height`` may be a scaled-down grid over the same sensor area, so
the effective pixel pitch is ``sensor_size / requested_size``. Orientation = three.js
``Matrix4.lookAt(eye, target, up)`` with ``up = +Y`` (``(0,0,−1)`` when ``|dir.y| > 0.999``), then a
roll about the optical axis — identical to ``apps/studio-web/src/three/pose.ts``. Camera basis
(world): ``X`` image-right, ``Y`` image-up, ``Z`` backwards (optical axis = −Z). Row 0 is the image
top, column 0 the image left; the pixel (col,row) ray is ``(x_s X + y_s Y − v Z)`` normalised with
``x_s = (col + ½ − W/2) p_x``, ``y_s = (H/2 − row − ½) p_y`` (an upright, un-inverted image).
A camera above the origin looking down therefore has world +X on the image right and −Z at the top.

Layers
------
Layer 0 is the first surface the camera ray enters. If that medium is *clear* (``scattering.model ==
"none"``, ``mus'(500) == 0`` or ``toggles.scattering`` off) the ray refracts in (Snell, unpolarised
Fresnel), travels the exact chord, refracts out into whatever lies beyond (air or an adjacent solid —
a dish's liquid sits directly on its glass base) and continues: layer j+1. Turbid media and the
background slab terminate the chain. Whenever refraction at an exit interface would be totally
internally reflected, or the next entering interface cannot be crossed (TIR), the chain is
**terminated** (the reflected branch is not followed). At most ``MAX_LAYERS`` layers are produced.

Fresnel bookkeeping (see ``contracts.HitBuffer``): ``T_view`` / ``T_light`` multiply only the
interfaces crossed *before* the entry interface of the layer, the entry interface itself is applied by
the transport kernel; the exit interface of layer j into solid B *is* the entry interface of the next
layer, so ``T_view[j+1] = T_view[j] · T(entry of j)`` (and, if the ray goes through air in between,
``· T(exit of j)``).

Shadow rays / excitation
------------------------
From every layer entry point a shadow ray runs toward the source along the straight line to the
beam's aperture (:func:`spectraforge.studio.source.beam_samples`). Approximation: the line is *not*
bent by refraction (lateral shift through thin glass is ignored), but the directions inside the media
and each crossed interface's Fresnel transmittance are obtained by applying Snell's law successively
from the source side, so ``cos_light`` is the incidence cosine *in the medium outside the entry
interface*. Crossed clear media contribute their straight chord to ``light_clear_path_mm`` (for
Beer–Lambert); a crossed non-clear solid sets ``T_light = 0``. The surface irradiance
``irradiance_per_mW = S · cos(ŝ, n)`` uses the straight line direction, i.e. it is the irradiance of
the *air-side* beam on a surface with the entry normal, before any Fresnel transmission.
Volumetric samples (``vol_*``, clear layers only): ``Q`` Gauss-Legendre nodes ``s_q`` along the exact
(refracted) view chord. For each sample a straight shadow ray runs to the source (as above, no bending
of the path); the first crossing is the solid's own *light-entry face* (it may differ from the
view-entry face). ``vol_fluence_per_mW = S · T_total · cosθ_ext/cosθ_int`` where ``S`` is the beam flux
along its propagation direction (no surface cosine), ``T_total`` the product of the Fresnel T of every
crossing from the face outward to the source (non-clear occluder → 0), and ``cosθ_ext/cosθ_int`` the
beam-compression factor of refraction at the face (flux per unit cross-section inside = surface power
density / cosθ_int; needed to reproduce the closed-form slab kernel). ``vol_light_path_mm`` is the
in-medium excitation path ``t_face · |cos_u| / cosθ_int`` (perpendicular depth below the face divided by
the refracted cosine; exact for planar faces). Beer-Lambert absorption of *prior clear occluders* is
omitted (it is available per material in ``light_clear_path_mm`` for the entry point only).
``vol_T_out`` is the Fresnel T leaving through the view-entry face at the internal view angle
(0 on TIR). With ``toggles.shadows`` off only the own face is traced and nothing occludes; with
``toggles.fresnel`` off all T = 1 and no compression. Turbid layers/background: zeros.

Layered blocks
--------------
A ``layered_block`` is a single convex, non-clear solid (see :mod:`spectraforge.studio.geometry`): the
layer chain ends there (even if its top layer is a clear material) and it occludes shadow rays. Its
interior is NOT traced; the pixels that see it are reported in ``HitBuffer.info['block_hits']``::

    info['block_hits'][object_index] = {
        'layer': (n,) int   # which camera-ray layer slot j (0 = first surface) hit the block
        'pixel': (n,) int   # pixel position 0..P-1 within this buffer
        'face':  (n,) <U6   # 'top' | '+x' | '-x' | '+z' | '-z' | 'bottom'
        'u': (n,), 'v': (n,)  # face-local coordinates (mm, block frame; see contracts.FaceGrid)
        'x': (n,), 'z': (n,)  # block-local lateral position of the hit (for thickness ground truth)
    }

``material[j, pixel]`` is the TOP layer's material; the capture engine skips the 1-D kernel for these
pixels and reads the 3D Monte Carlo face tallies instead. The specular glint of the top/side face is
still computed here (ordinary GGX at the face normal).

Performance: the volumetric shadow rays (the dominant cost, ~16 per clear layer pixel) are traced by a
fused parallel numba kernel (``geometry_nb.volume_shadow_nb``: crossings, occlusion and the Fresnel
chain per sample, scalar port of ``_shadow(volume=True)``); the NumPy ``_shadow`` path remains as the
fallback / reference (``Scene.backend = "numpy"``). Results agree to ~1e-9 relative and are independent
of the thread count.

Entry points whose surface faces away from the source (``cos <= 0``) get ``irradiance = 0`` — the
slab-entry kernel model cannot represent light that reaches a volume through a different face
(e.g. a laser crossing a cuvette sideways and viewed at 90°).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from spectraforge.studio import source as srcmod
from spectraforge.studio import surface
from spectraforge.studio.contracts import MAX_LAYERS, HitBuffer
from spectraforge.studio.geometry import BACKGROUND_OBJECT, EPS_T_MM, PROBE_MM, Scene, _rowmat, build_scene
from spectraforge.studio.schema import CameraSpec, Experiment, PhysicsToggles

_MAX_SHADOW_CROSSINGS = 16   # three cuvettes in series need 10 crossings; ray beyond the cap counts as occluded
# default volume_samples stays 16: measured laser_line (348x256) 0.44 s at Q=16 vs 0.81 s at Q=32 (> 0.8 s budget)
_VOL_CHUNK = 16384          # shadow rays per batch for the volume samples (cache-friendly)


# --------------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------------
def camera_basis(position, target, roll_deg: float = 0.0) -> np.ndarray:
    """Rows = world vectors ``(X right, Y up, Z back)`` — three.js ``lookAt`` + roll (``pose.ts``)."""
    eye = np.asarray(position, float)
    tgt = np.asarray(target, float)
    dirv = tgt - eye
    dirv = dirv / max(np.linalg.norm(dirv), 1e-12)
    up = np.array([0.0, 0.0, -1.0]) if abs(dirv[1]) > 0.999 else np.array([0.0, 1.0, 0.0])
    z = eye - tgt
    if np.dot(z, z) == 0.0:
        z = np.array([0.0, 0.0, 1.0])
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.dot(x, x) == 0.0:                       # three.js: up parallel to z
        z = z + (np.array([1e-4, 0.0, 0.0]) if abs(up[2]) == 1.0 else np.array([0.0, 0.0, 1e-4]))
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.radians(roll_deg)
    c, s = np.cos(r), np.sin(r)
    return np.stack([c * x + s * y, -s * x + c * y, z])


def image_distance_mm(cam: CameraSpec) -> float:
    f, s = cam.lens.focal_length_mm, cam.lens.focus_distance_mm
    if s <= f:
        raise ValueError(f"focus_distance_mm ({s}) must exceed focal_length_mm ({f})")
    return f * s / (s - f)


def camera_rays(cam: CameraSpec, width: int, height: int, pixels: np.ndarray | None = None) -> dict:
    """Primary rays for the camera at ``width × height``. Returns a dict with ``origin`` (3,),
    ``dirs`` (P,3), ``field_angle`` (P,), ``pixel_angle`` (rad, on-axis), ``pitch_mm`` (px, py),
    ``v_mm`` and ``basis``."""
    v = image_distance_mm(cam)
    pitch = cam.sensor.pixel_pitch_um * cam.sensor.binning * 1e-3
    sw, sh = cam.sensor.width_px * pitch, cam.sensor.height_px * pitch
    px, py = sw / width, sh / height
    idx = np.arange(width * height) if pixels is None else np.asarray(pixels, dtype=np.int64)
    cols, rows = idx % width, idx // width
    xs = (cols + 0.5 - width / 2.0) * px
    ys = (height / 2.0 - rows - 0.5) * py
    b = camera_basis(cam.position_mm, cam.target_mm, cam.roll_deg)
    vec = xs[:, None] * b[0] + ys[:, None] * b[1] - v * b[2]
    norm = np.linalg.norm(vec, axis=1)
    return {"origin": np.asarray(cam.position_mm, float), "dirs": vec / norm[:, None],
            "field_angle": np.arccos(np.clip(v / np.sqrt(xs * xs + ys * ys + v * v), -1, 1)),
            "pixel_angle": px / v, "pitch_mm": (px, py), "v_mm": v, "basis": b, "sensor_mm": (sw, sh)}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
@dataclass
class _Ctx:
    scene: Scene
    solid_mat: np.ndarray
    solid_obj: np.ndarray
    solid_n: np.ndarray
    solid_clear: np.ndarray
    mat_rough: np.ndarray
    mat_n: np.ndarray
    M: int
    fresnel: bool
    shadows: bool

    def n_of(self, idx):
        return np.where(idx >= 0, self.solid_n[np.maximum(idx, 0)], 1.0)

    def clear_of(self, idx):
        return np.where(idx >= 0, self.solid_clear[np.maximum(idx, 0)], True)


def _forward(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Flip ``n`` so that ``d·n >= 0`` (pointing along the direction of travel)."""
    s = np.where(np.einsum("ij,ij->i", n, d) < 0.0, -1.0, 1.0)
    return n * s[:, None]


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _shadow(ctx: _Ctx, src, x: np.ndarray, nout: np.ndarray | None, start_solid: np.ndarray,
            n_outside: np.ndarray, n_layer: np.ndarray, volume: bool = False) -> dict:
    """Shadow-ray bookkeeping for points ``x`` (see module docstring).

    ``volume=True`` (sample points *inside* the solid ``start_solid``; ``nout`` is None): the first
    crossing along the ray is the solid's own light-entry face, which is always traced (even with
    shadows off, then it is the only crossing and nothing blocks); its geometry is returned in
    ``face_*``."""
    n = len(x)
    M = ctx.M
    res = {"E": np.zeros(n), "T": np.zeros(n), "cos": np.zeros(n), "cos_int": np.zeros(n),
           "path": np.zeros((n, M)), "u": np.zeros((n, 3)), "dist": np.full(n, np.inf), "lit": np.zeros(n, bool),
           "flux": np.zeros(n), "face_t": np.full(n, np.nan), "face_cos_u": np.zeros(n),
           "face_cos_ext": np.zeros(n), "face_cos_int": np.zeros(n)}
    if src is None or n == 0:
        return res
    bs = srcmod.beam_samples(src, x)
    u = bs.u_to_source
    dist = bs.dist_mm
    if nout is None:
        cos_inc = np.ones(n)
        lit = bs.flux_per_mW > 0.0
    else:
        cos_inc = _dot(u, nout)
        lit = (cos_inc > 1e-6) & (bs.flux_per_mW > 0.0)
    res["u"], res["dist"], res["flux"] = u, dist, bs.flux_per_mW
    dl = -u.copy()                                   # direction of the light traveling toward x
    T = np.where(lit, 1.0, 0.0)
    path = res["path"]
    blocked = np.zeros(n, bool)
    crossings = []
    trace = ctx.shadows or volume
    if trace:
        act = np.nonzero(lit)[0]
        cur = start_solid[act].copy()
        t_prev = np.zeros(act.size)
        max_it = _MAX_SHADOW_CROSSINGS if ctx.shadows else 1
        for _ in range(max_it):
            if act.size == 0:
                break
            h = ctx.scene.first_hit(x[act], u[act], t_prev + EPS_T_MM)
            cross = h.found & (h.t < dist[act])
            seg = np.maximum(np.where(cross, h.t, dist[act]) - t_prev, 0.0)
            inside = cur >= 0
            if inside.any():
                np.add.at(path, (act[inside], ctx.solid_mat[cur[inside]]), seg[inside])
            ci = np.nonzero(cross)[0]
            if ci.size == 0:
                act = act[:0]
                break
            ids = act[ci]
            after = ctx.scene.locate_after(x[ids], u[ids], h.t[ci])
            nrm_c = _forward(h.normal[ci], u[ids])
            crossings.append((ids, nrm_c, ctx.n_of(after), ctx.n_of(cur[ci])))
            if volume and len(crossings) == 1:
                res["face_t"][ids] = h.t[ci]
                res["face_cos_u"][ids] = np.abs(_dot(u[ids], nrm_c))
            if ctx.shadows:
                blk = ~ctx.clear_of(after)
                blocked[ids[blk]] = True
            else:
                blk = np.zeros(ids.size, bool)
            keep = ~blk
            act, cur, t_prev = ids[keep], after[keep], h.t[ci][keep]
        else:
            if ctx.shadows:
                blocked[act] = True                   # too many crossings: treat as occluded
        if ctx.fresnel:
            for k, (ids, nrm, n_src, n_x) in reversed(list(enumerate(crossings))):
                m = ~blocked[ids]
                if not m.any():
                    continue
                ids, nrm, n_src, n_x = ids[m], nrm[m], n_src[m], n_x[m]
                cos_i = np.abs(_dot(dl[ids], nrm))
                T[ids] *= 1.0 - surface.fresnel_reflectance(cos_i, n_src, n_x)
                dl[ids], _ = surface.refract(dl[ids], nrm, n_src, n_x)
                if volume and k == 0:
                    res["face_cos_ext"][ids] = cos_i
                    res["face_cos_int"][ids] = np.abs(_dot(dl[ids], nrm))
        elif volume and crossings:
            ids, nrm = crossings[0][0], crossings[0][1]
            c = np.abs(_dot(dl[ids], nrm))
            res["face_cos_ext"][ids] = c
            res["face_cos_int"][ids] = c
        T[blocked] = 0.0
        path[blocked] = 0.0
    if nout is None:
        res["T"] = T
        res["lit"] = lit
        return res
    cos_l = np.clip(-_dot(dl, nout), 0.0, 1.0)
    if ctx.fresnel:
        cos_int, _tir = surface.snell_cos_t(cos_l, n_outside, n_layer)
    else:
        cos_int = cos_l
    lit_e = lit & (cos_l > 1e-6)
    res["E"] = np.where(lit_e, bs.flux_per_mW * cos_inc, 0.0)
    res["T"] = np.where(lit_e, T, 0.0)
    res["cos"] = np.where(lit_e, cos_l, 0.0)
    res["cos_int"] = np.where(lit_e, cos_int, 0.0)
    res["lit"] = lit_e
    return res


def _volume_shadow_fast(ctx: _Ctx, src, pts: np.ndarray, start: np.ndarray, flat) -> dict:
    """Numba-fused equivalent of ``_shadow(..., volume=True)`` for all sample points at once
    (same crossings / Fresnel bookkeeping; see :mod:`spectraforge.studio.geometry_nb`)."""
    from spectraforge.studio import geometry_nb
    bs = srcmod.beam_samples(src, pts)
    lit = np.ascontiguousarray(bs.flux_per_mW > 0.0)
    T, ft, fcu, fce, fci = geometry_nb.volume_shadow_nb(
        np.ascontiguousarray(pts), np.ascontiguousarray(bs.u_to_source), np.ascontiguousarray(bs.dist_mm),
        lit, np.ascontiguousarray(start, dtype=np.int64), np.ascontiguousarray(ctx.solid_n, dtype=np.float64),
        np.ascontiguousarray(ctx.solid_clear, dtype=np.bool_), bool(ctx.fresnel), bool(ctx.shadows),
        EPS_T_MM, PROBE_MM, _MAX_SHADOW_CROSSINGS, *flat)
    return {"flux": bs.flux_per_mW, "T": T, "face_t": ft, "face_cos_u": fcu, "face_cos_ext": fce,
            "face_cos_int": fci, "dist": bs.dist_mm}


def _volume_samples(ctx: _Ctx, src, x, d_in, chord, S, n_S, n_out, nout, cosv_int, q_nodes, q_w):
    """Quadrature samples along the exact view chord of clear layers (see module docstring).
    Returns dict of (n, Q) arrays + ``T_out`` (n,)."""
    n, Q = len(x), len(q_nodes)
    s = chord[:, None] * q_nodes[None, :]                              # (n, Q)
    w = chord[:, None] * q_w[None, :]
    pts = (x[:, None, :] + d_in[:, None, :] * s[:, :, None]).reshape(-1, 3)
    start = np.repeat(S, Q)
    nS_rep = np.repeat(n_S, Q)
    flat = ctx.scene._nb()
    if flat is not None and len(pts):
        sh = _volume_shadow_fast(ctx, src, pts, start, flat)
    else:
        parts = [_shadow(ctx, src, pts[i:i + _VOL_CHUNK], None, start[i:i + _VOL_CHUNK], None,
                         nS_rep[i:i + _VOL_CHUNK], volume=True) for i in range(0, len(pts), _VOL_CHUNK)]
        sh = {k: np.concatenate([p_[k] for p_ in parts]) for k in ("flux", "T", "face_t", "face_cos_u",
                                                                   "face_cos_ext", "face_cos_int", "dist")}
    face_ok = np.isfinite(sh["face_t"]) & (sh["face_cos_int"] > 1e-6)
    comp = np.where(face_ok, sh["face_cos_ext"] / np.maximum(sh["face_cos_int"], 1e-6), 1.0)
    fluence = sh["flux"] * sh["T"] * comp
    lpath = np.where(face_ok, sh["face_t"] * sh["face_cos_u"] / np.maximum(sh["face_cos_int"], 1e-6),
                     np.minimum(sh["dist"], 1e6))
    if ctx.fresnel:
        T_out = 1.0 - surface.fresnel_reflectance(cosv_int, n_S, n_out)       # TIR → R = 1 → T = 0
    else:
        T_out = np.ones(n)
    return {"s": s, "w": w, "F": fluence.reshape(n, Q), "l": lpath.reshape(n, Q), "T_out": T_out}


def block_face_coords(local_pt: np.ndarray, local_n: np.ndarray):
    """Face id and face-local ``(u, v)`` of points ``local_pt`` (n,3) with outward normals ``local_n`` on
    a layered block: ``top`` (ny > 0.5; ``u = x, v = z``), ``+x/-x`` (``u = z, v = y``), ``+z/-z``
    (``u = x, v = y``), ``bottom`` (ny < -0.5)."""
    n = len(local_pt)
    face = np.full(n, "top", dtype="<U6")
    ny = local_n[:, 1]
    side = np.abs(ny) <= 0.5
    ax = np.abs(local_n[:, 0]) >= np.abs(local_n[:, 2])
    face[side & ax & (local_n[:, 0] > 0)] = "+x"
    face[side & ax & (local_n[:, 0] <= 0)] = "-x"
    face[side & ~ax & (local_n[:, 2] > 0)] = "+z"
    face[side & ~ax & (local_n[:, 2] <= 0)] = "-z"
    face[ny < -0.5] = "bottom"
    x, y, z = local_pt[:, 0], local_pt[:, 1], local_pt[:, 2]
    u = np.where(face == "top", x, np.where(np.isin(face, ("+x", "-x")), z, x))
    v = np.where(face == "top", z, y)
    return face, u, v


def _block_hits(scene: Scene, rec: list[tuple]) -> dict:
    out: dict[int, dict] = {}
    cat = [np.concatenate([r[i] for r in rec]) for i in (1, 2, 3, 4)]
    layer = np.concatenate([np.full(len(r[1]), r[0], dtype=np.int64) for r in rec])
    pix, sol, pw, nw = cat
    for si in np.unique(sol):
        sd = scene.solids[int(si)]
        m = sol == si
        lp = _rowmat(pw[m] - sd.p, sd.R)
        ln = _rowmat(nw[m], sd.R)
        face, u, v = block_face_coords(lp, ln)
        out[int(sd.object_index)] = {"layer": layer[m], "pixel": pix[m], "face": face, "u": u, "v": v,
                                     "x": lp[:, 0], "z": lp[:, 2]}
    return out


def make_context(experiment: Experiment, scene: Scene, toggles: PhysicsToggles):
    """Shadow-ray context of ``scene`` (also used by the capture engine to trace the excitation of
    layered blocks). Returns ``(ctx, mat_clear, solid_block)``; layered-block solids are never clear."""
    mats = experiment.materials
    mat_n = np.array([m.refractive_index for m in mats], float)
    mat_rough = np.array([m.surface_roughness for m in mats], float)
    mat_clear = np.array([(m.scattering.model == "none") or (m.scattering.mus_prime_500_per_mm == 0.0)
                          or (not toggles.scattering) for m in mats], bool)
    solid_mat = np.array([s.material_index for s in scene.solids], dtype=np.int64)
    solid_obj = np.array([s.object_index for s in scene.solids], dtype=np.int64)
    solid_block = np.array([s_.role == "block" for s_ in scene.solids], dtype=bool)
    ctx = _Ctx(scene, solid_mat, solid_obj, mat_n[solid_mat] if len(solid_mat) else np.zeros(0),
               (mat_clear[solid_mat] & (solid_obj != BACKGROUND_OBJECT) & ~solid_block)
               if len(solid_mat) else np.zeros(0, bool),
               mat_rough, mat_n, len(mats), toggles.fresnel, toggles.shadows)
    return ctx, mat_clear, solid_block


# --------------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------------
def trace_scene(experiment: Experiment, width: int, height: int, *, pixels: np.ndarray | None = None,
                toggles: PhysicsToggles | None = None, volume_samples: int = 16) -> HitBuffer:
    """Trace the active camera of ``experiment`` at ``width × height`` into a :class:`HitBuffer`.

    ``pixels``: optional flat pixel indices ``row*width + col`` of the *requested* grid. When given,
    only those rays are traced and the returned buffer has ``P = len(pixels)`` entries with
    ``width = len(pixels)`` and ``height = 1`` (so ``n_pixels`` matches the array length); the
    originals are kept in ``info['full_width' | 'full_height' | 'pixel_indices']``.

    ``volume_samples`` (Q): number of Gauss-Legendre nodes along the exact view chord of each CLEAR
    layer for the optional ``vol_*`` fields (0 disables; the fields stay None if no layer is clear).
    """
    t0 = time.perf_counter()
    toggles = toggles or experiment.acquisition.physics
    cam = experiment.camera()
    try:
        src = experiment.source()
    except ValueError:
        src = None
    scene = build_scene(experiment)
    ctx, mat_clear, solid_block = make_context(experiment, scene, toggles)
    mats, M, mat_n, mat_rough = experiment.materials, ctx.M, ctx.mat_n, ctx.mat_rough
    solid_mat, solid_obj = ctx.solid_mat, ctx.solid_obj

    rays = camera_rays(cam, width, height, pixels)
    dirs = rays["dirs"]
    P = len(dirs)
    K = MAX_LAYERS
    material = np.full((K, P), -1, dtype=np.int64)
    object_index = np.full((K, P), -1, dtype=np.int64)
    point = np.zeros((K, P, 3))
    normal = np.zeros((K, P, 3))
    path = np.zeros((K, P))
    cos_view = np.zeros((K, P))
    cos_view_int = np.zeros((K, P))
    cos_light_int = np.zeros((K, P))
    cos_light = np.zeros((K, P))
    thickness = np.zeros((K, P))
    T_view = np.zeros((K, P))
    T_light = np.zeros((K, P))
    n_outside = np.ones((K, P))
    irr = np.zeros((K, P))
    clear_path = np.zeros((K, P, M))
    depth = np.full(P, np.inf)
    glint = np.zeros(P)
    first_u = np.zeros((P, 3))
    first_dist = np.full(P, np.inf)
    first_lit = np.zeros(P, bool)

    alive = np.ones(P, bool)
    search = np.ones(P, bool)
    cpos = np.tile(rays["origin"], (P, 1))
    cdir = dirs.copy()
    cur_n = np.ones(P)
    cur_solid = np.full(P, -1, dtype=np.int64)
    tcum = np.ones(P)
    e_S = np.full(P, -1, dtype=np.int64)
    e_x = np.zeros((P, 3))
    e_n = np.zeros((P, 3))
    tir_term = 0
    block_rec: list[tuple] = []        # (layer j, pixel idx, solid idx, world point, world normal)
    vol = None
    q_nodes = q_w = None
    if volume_samples and volume_samples > 0:
        xg, wg = np.polynomial.legendre.leggauss(int(volume_samples))
        q_nodes, q_w = 0.5 * (xg + 1.0), 0.5 * wg                  # Gauss-Legendre on [0, 1]

    for j in range(K):
        s_idx = np.nonzero(alive & search)[0]
        if s_idx.size:
            h = scene.first_hit(cpos[s_idx], cdir[s_idx], EPS_T_MM)
            S = np.full(s_idx.size, -1, dtype=np.int64)
            f = np.nonzero(h.found)[0]
            if f.size:
                S[f] = scene.locate_after(cpos[s_idx][f], cdir[s_idx][f], h.t[f])
            good = h.found & (S >= 0)
            alive[s_idx[~good]] = False
            gi = s_idx[good]
            e_S[gi] = S[good]
            e_x[gi] = cpos[gi] + cdir[gi] * h.t[good][:, None]
            e_n[gi] = -_forward(h.normal[good], cdir[gi])
            if j == 0:
                depth[gi] = h.t[good]
        A = np.nonzero(alive)[0]
        if A.size == 0:
            break
        S = e_S[A]
        x = e_x[A]
        nout = e_n[A]
        d = cdir[A]
        mat = solid_mat[S]
        obj = solid_obj[S]
        n_S = mat_n[mat]
        n1 = cur_n[A]
        cosv = np.clip(-_dot(d, nout), 0.0, 1.0)
        if toggles.fresnel:
            d_in, tir = surface.refract(d, nout, n1, n_S)
            T_entry = 1.0 - surface.fresnel_reflectance(cosv, n1, n_S)
            cosv_int = np.clip(-_dot(d_in, nout), 0.0, 1.0)
        else:
            d_in, tir = d, np.zeros(A.size, bool)
            T_entry = np.ones(A.size)
            cosv_int = cosv
        if tir.any():
            tir_term += int(tir.sum())
            alive[A[tir]] = False
            k = ~tir
            A, S, x, nout, d, mat, obj, n_S, n1 = (a[k] for a in (A, S, x, nout, d, mat, obj, n_S, n1))
            d_in, T_entry, cosv, cosv_int = d_in[k], T_entry[k], cosv[k], cosv_int[k]
            if A.size == 0:
                break
        # --- geometry of the layer -------------------------------------------------------
        hth = scene.first_hit(x, -nout, EPS_T_MM)
        th = np.where(hth.found, hth.t, np.inf)
        hc = scene.first_hit(x, d_in, EPS_T_MM)
        chord = np.where(hc.found, hc.t, 0.0)
        is_bg = obj == BACKGROUND_OBJECT
        chord = np.where(is_bg, th, chord)
        # --- shadow / excitation -----------------------------------------------------------
        sh = _shadow(ctx, src, x, nout, cur_solid[A], n1, n_S)
        # --- record ------------------------------------------------------------------------
        material[j, A] = mat
        object_index[j, A] = obj
        point[j, A] = x
        normal[j, A] = nout
        path[j, A] = chord
        cos_view[j, A] = cosv
        cos_view_int[j, A] = cosv_int
        cos_light[j, A] = sh["cos"]
        cos_light_int[j, A] = sh["cos_int"]
        thickness[j, A] = th
        T_view[j, A] = tcum[A]
        T_light[j, A] = sh["T"]
        n_outside[j, A] = n1
        irr[j, A] = sh["E"]
        clear_path[j, A] = sh["path"]
        if q_nodes is not None and src is not None:
            vc = np.nonzero(mat_clear[mat] & ~is_bg & hc.found & (chord > 0.0))[0]
            if vc.size:
                if vol is None:
                    Q = len(q_nodes)
                    vol = {"s": np.zeros((K, P, Q), np.float32), "w": np.zeros((K, P, Q), np.float32),
                           "F": np.zeros((K, P, Q), np.float32), "l": np.zeros((K, P, Q), np.float32),
                           "T_out": np.zeros((K, P))}
                v = _volume_samples(ctx, src, x[vc], d_in[vc], chord[vc], S[vc], n_S[vc], n1[vc], nout[vc],
                                    cosv_int[vc], q_nodes, q_w)
                Av = A[vc]
                for key in ("s", "w", "F", "l"):
                    vol[key][j, Av] = v[key]
                vol["T_out"][j, Av] = v["T_out"]
        if j == 0:
            first_u[A], first_dist[A], first_lit[A] = sh["u"], sh["dist"], sh["lit"]
        # --- continue through clear solids ---------------------------------------------------
        if j == K - 1:
            break
        if solid_block.any():
            isb = solid_block[S]
            if isb.any():
                block_rec.append((j, A[isb], S[isb], x[isb], nout[isb]))
        cont = mat_clear[mat] & ~is_bg & hc.found & ~solid_block[S]
        alive[A[~cont]] = False
        if not cont.any():
            break
        C = np.nonzero(cont)[0]
        Ai, xi, di, ti = A[C], x[C], d_in[C], hc.t[C]
        nf = _forward(hc.normal[C], di)
        xe = xi + di * ti[:, None]
        B = scene.locate_after(xi, di, ti)
        nB = ctx.n_of(B)
        nS = n_S[C]
        if toggles.fresnel:
            d_new, tir2 = surface.refract(di, nf, nS, nB)
            T_exit = 1.0 - surface.fresnel_reflectance(np.abs(_dot(di, nf)), nS, nB)
        else:
            d_new, tir2, T_exit = di, np.zeros(C.size, bool), np.ones(C.size)
        if tir2.any():
            tir_term += int(tir2.sum())
            alive[Ai[tir2]] = False
        ok = ~tir2
        Ai, xe, di, d_new, nf, B, nS, T_exit, C = (a[ok] for a in (Ai, xe, di, d_new, nf, B, nS, T_exit, C))
        tc = tcum[Ai] * T_entry[C]
        solid_next = B >= 0
        sn = Ai[solid_next]
        e_S[sn] = B[solid_next]
        e_x[sn] = xe[solid_next]
        e_n[sn] = -nf[solid_next]
        cdir[sn] = di[solid_next]
        cur_n[sn] = nS[solid_next]
        cur_solid[sn] = S[C][solid_next]
        tcum[sn] = tc[solid_next]
        search[sn] = False
        an = Ai[~solid_next]
        cpos[an] = xe[~solid_next]
        cdir[an] = d_new[~solid_next]
        cur_n[an] = 1.0
        cur_solid[an] = -1
        tcum[an] = (tc * T_exit)[~solid_next]
        search[an] = True

    # --- per-pixel quantities ---------------------------------------------------------------
    cos0 = cos_view[0]
    footprint = np.where(np.isfinite(depth), depth * rays["pixel_angle"] / np.maximum(cos0, 1e-3), 0.0)
    view_dir = -dirs
    if src is not None and toggles.specular_glints:
        sel = np.nonzero((material[0] >= 0) & first_lit & (T_light[0] > 0))[0]
        if sel.size:
            m0 = material[0, sel]
            rho = np.arctan2(0.5 * src.beam.aperture_diameter_mm, first_dist[sel])
            alpha = np.sqrt(mat_rough[m0] ** 2 + (0.5 * rho) ** 2)
            f = surface.ggx_brdf(normal[0, sel], first_u[sel], view_dir[sel], alpha, mat_n[m0], 1.0)
            glint[sel] = f * irr[0, sel] * T_light[0, sel]

    used = [int((material[j] >= 0).sum()) for j in range(K)]
    info = {
        "n_solids": len(scene.solids),
        "layers_used": used,
        "fraction_hit": float(np.isfinite(depth).mean()) if P else 0.0,
        "tir_terminations": tir_term,
        "warnings": list(scene.warnings) + ([] if src is not None else ["experiment has no light source"]),
        "image_distance_mm": rays["v_mm"],
        "pixel_pitch_mm": rays["pitch_mm"],
        "sensor_mm": rays["sensor_mm"],
        "camera_basis": rays["basis"],
        "has_source": src is not None,
        "n_lit_layer0": int(first_lit.sum()),
        "trace_s": time.perf_counter() - t0,
    }
    if block_rec:
        info["block_hits"] = _block_hits(scene, block_rec)
    if pixels is not None:
        info.update(full_width=width, full_height=height, pixel_indices=np.asarray(pixels, dtype=np.int64))
        out_w, out_h = P, 1
    else:
        out_w, out_h = width, height
    return HitBuffer(
        width=out_w, height=out_h, material=material, object_index=object_index, point_mm=point,
        normal=normal, path_mm=path, cos_view=cos_view, cos_view_internal=cos_view_int,
        cos_light_internal=cos_light_int, cos_light=cos_light, thickness_mm=thickness, T_view=T_view,
        T_light=T_light, n_outside=n_outside, irradiance_per_mW=irr, light_clear_path_mm=clear_path,
        depth_mm=depth, glint_per_mW=glint, pixel_footprint_mm=footprint, view_dir=view_dir,
        ray_dir=dirs, field_angle_rad=rays["field_angle"],
        focus_depth_mm=float(cam.lens.focus_distance_mm), info=info,
        vol_s_mm=None if vol is None else vol["s"], vol_weight_mm=None if vol is None else vol["w"],
        vol_fluence_per_mW=None if vol is None else vol["F"],
        vol_light_path_mm=None if vol is None else vol["l"],
        vol_T_out=None if vol is None else vol["T_out"])
