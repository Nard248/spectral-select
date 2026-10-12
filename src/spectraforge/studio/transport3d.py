"""3D Monte Carlo for layered blocks (SpectralForge Studio v1.1) -- public API.

A ``layered_block`` is a stack of layers whose interfaces are exact planes
``y = h_i(x, z) = a_i + b_i x + c_i z`` (uniform or linear-wedge thickness), bounded by four
vertical side planes and the bottom face ``y = 0``; outside is air (n = 1).  Photon packets
random-walk through it (MCML generalised to tilted planar interfaces: implicit capture,
Henyey-Greenstein scattering, Russian roulette, Fresnel reflection/refraction/TIR with the true
plane normal, dimensionless-step bookkeeping across interfaces, ``mu_s = 0`` layers handled with
deterministic Beer-Lambert attenuation).  Kernels: :mod:`spectraforge.studio.transport3d_mc`.

Conventions
-----------
Local frame, mm / nm / mW.  ``camera_dir_local`` is the unit vector pointing FROM the block TO the
camera.  Faces: ``top`` (the top interface plane, (u, v) = (x, z)), ``+x``/``-x`` ((u, v) = (z, y))
and ``+z``/``-z`` ((u, v) = (x, y)).  The bottom face (``y = 0``) is a Fresnel interface to air, exactly
like the four sides and like the faces of the 1-D slab engine (``n_outside`` on BOTH faces): light reaching it
is split deterministically into ``w (1 - R)`` which leaves (absorbed by the stage, tallied as
``T_bottom``, never returns) and ``w R`` which is reflected back INTO the block (total internal reflection
included).  With ``toggles.fresnel = False`` nothing is reflected and everything reaching the bottom is lost.

Radiance normalisation (derived)
--------------------------------
A packet of power ``W`` (mW) leaves a face cell of true area ``A`` with refracted direction
``d`` inside the acceptance cone (half angle ``theta_a``) about the camera direction ``c``.  If the
exit radiance is constant over the cone, the power through the cell into the cone is::

    P_cone = L * A * integral_cone (n . d) dOmega = L * A * (n . c) * pi sin^2(theta_a)

(the vector flux of a spherical cap is ``pi sin^2(theta_a) c``).  Hence the unbiased estimator::

    L_cell = (sum of W in the cone) / ( A_cell (n . c) pi sin^2(theta_a) )        [mW/(mm^2 sr)]

(the brief's ``Omega cos(theta_out)`` with ``Omega = 2 pi (1 - cos theta_a)`` is the same to
``(1 + cos theta_a)/2``; the form above is exact for a constant-radiance cone).  For the top plane
``A (n . c) = dx dz (c_y - b c_x - c c_z)``, the area projected perpendicular to ``c``.  Side cells
use the part of the cell that lies below the top surface.  A face seen too obliquely
(``n . c < sin theta_a``) is reported as not visible (zeros).

Peel-off (next-event) estimator -- ``peel_off=True`` (default)
-------------------------------------------------------------
The cone estimator above scores only the rare packets that happen to leave inside the acceptance cone, so
a per-cell tally is very noisy.  The peel-off estimator instead scores, at EVERY scattering event (and at
the launch of an emission packet), the expected contribution of a straight line of sight to the camera.
Let a packet of weight ``w`` (after the implicit-capture absorption ``w mu_s/mu_t``) scatter at ``r`` in
layer ``i`` from direction ``u`` into direction ``d`` with probability density ``p_HG(u . d)`` per
steradian.  The power it sends into the solid angle ``dOmega_i`` around ``d`` is ``w p_HG dOmega_i``.  The
direction ``d`` that reaches the camera direction ``c`` (in air) through the chain of interfaces
``1 .. m`` above the point follows from Snell's law applied with each plane's own normal (the inverse of the
forward refraction, ``_peel_top``).  Of that power the fraction ``T exp(-tau)`` arrives at the exit:

* ``tau = sum_j mu_t,j s_j``: optical depth of the straight segments ``s_j`` in every layer ``j`` crossed
  (absorption AND scattering remove the packet from the direct path; clear layers have ``mu_s = 0``),
* ``T = prod_j (1 - R_j)``: unpolarised Fresnel transmittances of the interfaces crossed, incl. the exit.

Etendue (radiance / n^2) is conserved across a transmitting interface, ``n_a^2 cos(a) dOmega_a =
n_b^2 cos(b) dOmega_b`` (same area element, angles from that plane's normal), so the solid angle in air that
corresponds to ``dOmega_i`` is ``dOmega_air = dOmega_i prod_j (n_in,j^2 cos_in,j) / (n_up,j^2 cos_up,j)``.
Hence the power per unit solid angle in air towards the camera::

    dP/dOmega_air = w p_HG(u . d) T exp(-tau) J ,      J = dOmega_i/dOmega_air = prod_j n_up^2 cos_up/(n_in^2 cos_in)

and, dividing by the projected cell area ``A_cell (n . c)`` (the same area factor as the cone estimator),
the radiance contribution added to the TOP-face cell containing the exit point::

    delta L_cell = dP/dOmega_air / (A_cell (n . c))     = (w p_HG T J exp(-tau)) * [pi sin^2(theta_a)] * cnorm_cell

(``cnorm_cell = 1/(A (n.c) pi sin^2 theta_a)`` is the cone normalisation, so the peel-off and cone
estimators share every table).  For an isotropic emission packet ``p = 1/(4 pi)`` and ``w = 1``.  A single
line of sight replaces the cone average, which agrees to ``O(theta_a^2)`` (<~ 1 % at 12 deg).  The deterministic
exit of a single layer gives the textbook ``L = L_in T / n^2``.

No double counting: the peel-off covers exactly the paths whose last event is a scattering/emission event
followed by a straight, reflection-free line to the top face.  The packet itself keeps walking; a flag
``refl`` (True at the launch of an excitation packet and after any Fresnel reflection, False after a
scatter) selects the paths the peel-off does NOT cover, which the cone estimator still scores when they
leave through the top face.  Side faces are always scored by the cone estimator (the camera of a block
study looks at the top).  The chain of layers crossed may depend on where the ray crosses each plane (a
wedge layer can vanish); it is found iteratively from the previous crossing points (<= 4 passes).

Excitation: ``W`` is power in mW, so ``L`` is the elastic radiance for the launched power.  The
reflected part of the entry Fresnel split is NOT transported (the tracer renders the glint); it is
reported as ``info["specular"]``.  Fractions in the diagnostics are relative to the total incident
power: ``R + T + A + specular + lost = 1`` where the Russian-roulette gain/loss is booked
separately and exactly (``info["energy_balance"]``).

Fluence (track length): ``Phi_v = sum_segments w * l / V_v`` in mW/mm^2 for the launched power in mW
(``mu_a`` absorbed power is ``mu_a Phi V`` on average); ``V_v = dx dz t_l(x_c, z_c) / S`` is exact
since ``t`` is linear.  Absorption is tallied per collision (implicit capture) and, in clear
layers, per exact voxel piece of the path.

Emission: source voxels are sampled by stratified inverse-CDF on ``q(v) = sum_c sources[c, v]``
(lower variance than an alias table, same distribution), uniformly in the exact curved voxel
volume, isotropic direction.  A packet carries ``coef_c = sources[c,v] / q(v) * Q / N`` so one run
per wavelength node serves all channels.  ``sources`` are emitted POWERS in mW, therefore the
returned radiance is in mW/(mm^2 sr) for those powers; multiply by the area-normalised emission
density ``e_k(lambda)`` [1/nm] to get mW/(mm^2 sr nm).  Common random numbers are used across
emission nodes (same source samples and random streams), which smooths spectra.

Determinism: photons are cut into fixed batches (size depends only on N), every batch re-seeds the
generator from ``(seed, batch)`` and owns private tallies that are summed in batch order, so
results are bit-identical for any thread count.  float64 accumulation.
"""
from __future__ import annotations

import math
import time

import numba
import numpy as np

from spectraforge.studio.contracts import (
    BlockGeometry,
    BlockLaunch,
    EmissionTallies,
    ExcitationTallies,
    FaceGrid,
    RegionOptics,
    VoxelGrid,
)
from spectraforge.studio.schema import Experiment, LayeredBlockShape, PhysicsToggles, VolumeMCSettings

FACES = ("top", "+x", "-x", "+z", "-z")
EPS_T = 1.0e-9


# ============================================================================================
# Geometry
# ============================================================================================
def planes_from_layers(footprint_mm, layers) -> np.ndarray:
    """``(L, 3)`` ``[a, b, c]`` of every layer's TOP plane from ``ThicknessProfile`` objects.
    ``t(x, z) = t0 + (t1 - t0)(u/size + 1/2)`` is linear, so ``h_i = sum_{j<=i} t_j`` is a plane."""
    L = len(layers)
    out = np.zeros((L, 3))
    acc = np.zeros(3)
    for i, prof in enumerate(layers):
        if prof.kind == "uniform":
            t = np.array([prof.thickness_mm, 0.0, 0.0])
        else:
            size = footprint_mm[0] if prof.axis == "x" else footprint_mm[1]
            slope = (prof.end_thickness_mm - prof.thickness_mm) / size
            t = np.array([prof.thickness_mm + 0.5 * (prof.end_thickness_mm - prof.thickness_mm), 0.0, 0.0])
            t[1 if prof.axis == "x" else 2] = slope
        acc = acc + t
        out[i] = acc
    return out


def build_block_geometry(experiment: Experiment, object_index: int) -> BlockGeometry:
    """Planar-interface description of ``experiment.objects[object_index]`` (a ``layered_block``)."""
    from spectraforge.studio.geometry import object_transform

    obj = experiment.objects[object_index]
    shape = obj.shape
    if not isinstance(shape, LayeredBlockShape):
        raise TypeError(f"object {obj.id!r} is not a layered_block")
    ids = [m.id for m in experiment.materials]
    try:
        mats = [ids.index(l.material_id) for l in shape.layers]
    except ValueError as e:
        raise KeyError(f"layered block {obj.id!r} uses an unknown material: {e}") from None
    R, p = object_transform(obj.transform)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return BlockGeometry(
        object_index=object_index,
        footprint_mm=(float(shape.footprint_mm[0]), float(shape.footprint_mm[1])),
        layer_names=[l.name for l in shape.layers],
        layer_material=mats,
        plane_coeffs=planes_from_layers(shape.footprint_mm, [l.thickness for l in shape.layers]),
        world_from_local=T,
    )


def make_voxel_grid(geom: BlockGeometry, settings: VolumeMCSettings) -> VoxelGrid:
    X, Z = geom.footprint_mm
    nx = max(1, int(math.ceil(X / settings.lateral_cell_mm - 1e-9)))
    nz = max(1, int(math.ceil(Z / settings.lateral_cell_mm - 1e-9)))
    return VoxelGrid(nx=nx, nz=nz, n_layers=len(geom.layer_names), sublayers=int(settings.sublayers),
                     x_edges_mm=np.linspace(-X / 2, X / 2, nx + 1),
                     z_edges_mm=np.linspace(-Z / 2, Z / 2, nz + 1))


def layer_thickness(geom: BlockGeometry, layer: int, x, z):
    """Local thickness ``t_layer(x, z)`` (mm)."""
    cur = geom.plane_coeffs[layer]
    prev = geom.plane_coeffs[layer - 1] if layer > 0 else np.zeros(3)
    d = cur - prev
    return d[0] + d[1] * np.asarray(x) + d[2] * np.asarray(z)


def voxel_volumes(geom: BlockGeometry, grid: VoxelGrid) -> np.ndarray:
    """``(nx, nz, L, S)`` exact voxel volumes (mm^3): ``dx dz t(x_c, z_c) / S``."""
    xc = 0.5 * (grid.x_edges_mm[1:] + grid.x_edges_mm[:-1])
    zc = 0.5 * (grid.z_edges_mm[1:] + grid.z_edges_mm[:-1])
    dx = grid.x_edges_mm[1] - grid.x_edges_mm[0]
    dz = grid.z_edges_mm[1] - grid.z_edges_mm[0]
    vol = np.empty(grid.shape)
    for l in range(grid.n_layers):
        t = np.clip(layer_thickness(geom, l, xc[:, None], zc[None, :]), 0.0, None)
        vol[:, :, l, :] = (dx * dz * t / grid.sublayers)[:, :, None]
    return vol


def _top_plane(geom: BlockGeometry) -> np.ndarray:
    return geom.plane_coeffs[-1]


def _max_height(geom: BlockGeometry) -> float:
    X, Z = geom.footprint_mm
    a, b, c = _top_plane(geom)
    return float(max(a + b * sx * X / 2 + c * sz * Z / 2 for sx in (-1, 1) for sz in (-1, 1)))


def make_face_grids(geom: BlockGeometry, settings: VolumeMCSettings) -> dict[str, FaceGrid]:
    X, Z = geom.footprint_mm
    ec = settings.exit_cell_mm
    ymax = max(_max_height(geom), 1e-6)
    nxe = max(1, int(math.ceil(X / ec - 1e-9)))
    nze = max(1, int(math.ceil(Z / ec - 1e-9)))
    nye = max(1, int(math.ceil(ymax / ec - 1e-9)))
    ex = np.linspace(-X / 2, X / 2, nxe + 1)
    ez = np.linspace(-Z / 2, Z / 2, nze + 1)
    ey = np.linspace(0.0, ymax, nye + 1)
    return {"top": FaceGrid("top", ex, ez), "+x": FaceGrid("+x", ez, ey), "-x": FaceGrid("-x", ez, ey),
            "+z": FaceGrid("+z", ex, ey), "-z": FaceGrid("-z", ex, ey)}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("camera_dir_local must be a non-zero finite vector")
    return v / n


def _face_tables(geom: BlockGeometry, faces: dict[str, FaceGrid], cam: np.ndarray, cos_a: float):
    """Flat per-cell normalisation ``1 / (A (n.c) pi sin^2 theta_a)`` (0 where not visible / empty)
    plus the packed face-grid tables used by the kernel."""
    X, Z = geom.footprint_mm
    a, b, c = _top_plane(geom)
    sin2 = 1.0 - cos_a * cos_a
    fg = np.zeros((5, 4))
    fgi = np.zeros((5, 3), np.int64)
    chunks = []
    off = 0
    visible = {}
    for f, name in enumerate(FACES):
        g = faces[name]
        nu, nv = len(g.u_edges_mm) - 1, len(g.v_edges_mm) - 1
        du, dv = g.u_edges_mm[1] - g.u_edges_mm[0], g.v_edges_mm[1] - g.v_edges_mm[0]
        fg[f] = (g.u_edges_mm[0], du, g.v_edges_mm[0], dv)
        fgi[f] = (nu, nv, off)
        if name == "top":
            proj = cam[1] - b * cam[0] - c * cam[2]            # (n.c) * sqrt(1+b^2+c^2)
            nc = proj / math.sqrt(1 + b * b + c * c)
            vis = nc >= math.sqrt(sin2)
            norm = np.full((nu, nv), 1.0 / (math.pi * sin2 * du * dv * proj) if vis else 0.0)
        else:
            sgn = 1.0 if name[0] == "+" else -1.0
            nc = sgn * (cam[0] if name[1] == "x" else cam[2])
            vis = nc >= math.sqrt(sin2)
            norm = np.zeros((nu, nv))
            if vis:
                # covered area of each cell = part below the top surface at this side face
                sub = 8
                us = g.u_edges_mm[0] + du * (np.arange(nu * sub) + 0.5) / sub        # (nu*sub,)
                if name[1] == "x":
                    h = a + b * sgn * X / 2 + c * us
                else:
                    h = a + b * us + c * sgn * Z / 2
                vlo = g.v_edges_mm[:-1]
                cover = np.clip(h[:, None] - vlo[None, :], 0.0, dv)                  # (nu*sub, nv)
                area = cover.reshape(nu, sub, nv).sum(1) * (du / sub)
                with np.errstate(divide="ignore"):
                    norm = np.where(area > 1e-12, 1.0 / (math.pi * sin2 * np.maximum(area, 1e-300) * nc), 0.0)
        visible[name] = bool(vis)
        chunks.append(np.ascontiguousarray(norm).ravel())
        off += nu * nv
    return fg, fgi, np.concatenate(chunks), visible


def _split_faces(flat: np.ndarray, faces: dict[str, FaceGrid]) -> dict[str, np.ndarray]:
    out, off = {}, 0
    for name in FACES:
        g = faces[name]
        nu, nv = len(g.u_edges_mm) - 1, len(g.v_edges_mm) - 1
        out[name] = flat[off:off + nu * nv].reshape((nu, nv) + flat.shape[1:])
        off += nu * nv
    return out


# ============================================================================================
# Launch preparation (entry Fresnel) -- vectorised numpy
# ============================================================================================
def _fresnel_np(n1, n2, c1):
    c1 = np.clip(c1, 0.0, 1.0)
    s2 = (n1 / n2) ** 2 * (1 - c1 * c1)
    tir = s2 >= 1.0
    c2 = np.sqrt(np.clip(1 - s2, 0.0, None))
    rs = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2 + 1e-300)) ** 2
    rp = ((n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2 + 1e-300)) ** 2
    return np.where(tir, 1.0, 0.5 * (rs + rp))


def _plane_arrays(geom: BlockGeometry):
    L = len(geom.layer_names)
    pl = np.zeros((L + 1, 3))
    pl[1:] = geom.plane_coeffs
    pn = np.zeros((L + 1, 3))
    pn[:, 0] = -pl[:, 1]
    pn[:, 1] = 1.0
    pn[:, 2] = -pl[:, 2]
    pn /= np.linalg.norm(pn, axis=1, keepdims=True)
    return np.ascontiguousarray(pl), np.ascontiguousarray(pn)


def _prepare_launch(geom: BlockGeometry, launch: BlockLaunch, n_layers_idx: np.ndarray, fres: bool):
    """Entry Fresnel split. Returns transmitted (points, dirs, layer, power) and the specular power."""
    pts = np.asarray(launch.points_local, float).reshape(-1, 3)
    dirs = np.asarray(launch.dirs_local, float).reshape(-1, 3)
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-300)
    P = np.asarray(launch.power_mW, float).reshape(-1)
    X, Z = geom.footprint_mm
    pl, pn = _plane_arrays(geom)
    L = len(geom.layer_names)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    top = pl[L, 0] + pl[L, 1] * x + pl[L, 2] * z
    dist = np.stack([np.abs(y - top), np.abs(y), np.abs(x - X / 2), np.abs(x + X / 2),
                     np.abs(z - Z / 2), np.abs(z + Z / 2)], 1)
    face = np.argmin(dist, 1)
    outn = np.zeros_like(pts)
    outn[face == 0] = pn[L]
    outn[face == 1] = (0, -1, 0)
    outn[face == 2] = (1, 0, 0)
    outn[face == 3] = (-1, 0, 0)
    outn[face == 4] = (0, 0, 1)
    outn[face == 5] = (0, 0, -1)
    ci = -np.einsum("ij,ij->i", dirs, outn)
    enter = ci > 1e-9
    # entry layer
    heights = pl[None, :, 0] + pl[None, :, 1] * x[:, None] + pl[None, :, 2] * z[:, None]   # (M, L+1)
    thick = np.diff(heights, axis=1)                                                      # (M, L)
    nonzero = thick > EPS_T
    anyz = nonzero.any(1)
    first = np.argmax(nonzero, 1)
    last = L - 1 - np.argmax(nonzero[:, ::-1], 1)
    inside = nonzero & (y[:, None] <= heights[:, 1:] + 1e-7)
    mid = np.where(inside.any(1), np.argmax(inside, 1), last)
    layer = np.where(face == 0, last, np.where(face == 1, first, mid))
    ok = enter & anyz
    n2 = n_layers_idx[layer]
    if fres:
        R = _fresnel_np(1.0, n2, ci)
        eta = 1.0 / n2
        ct = np.sqrt(np.clip(1 - eta * eta * (1 - ci * ci), 0.0, None))
        m_in = -outn
        tdir = eta[:, None] * dirs + (ct - eta * ci)[:, None] * m_in
    else:
        R = np.zeros_like(ci)
        tdir = dirs
    Pt = np.where(ok, P * (1 - R), 0.0)
    spec = float(P.sum() - Pt.sum())
    return pts[ok], tdir[ok], layer[ok].astype(np.int64), Pt[ok], spec, float(P.sum())


def uniform_beam_launch(geom: BlockGeometry, ex_nm: float, total_power_mW: float, n: int,
                        beam_half_size_mm=None, direction=(0.0, -1.0, 0.0), seed: int = 0) -> BlockLaunch:
    """Uniform collimated beam over a rectangle centred on the top surface (helper for tests / capture
    previews). Samples are equal-power; ``total_power_mW`` is incident on the beam cross-section."""
    rng = np.random.default_rng(seed)
    X, Z = geom.footprint_mm
    hx, hz = beam_half_size_mm if beam_half_size_mm is not None else (X / 2, Z / 2)
    x = rng.uniform(-hx, hx, n)
    z = rng.uniform(-hz, hz, n)
    a, b, c = _top_plane(geom)
    pts = np.stack([x, a + b * x + c * z, z], 1)
    d = _unit(direction)
    return BlockLaunch(ex_nm, pts, np.tile(d, (n, 1)), np.full(n, total_power_mW / n))


# ============================================================================================
# Driver
# ============================================================================================
def _batch_size(n: int) -> int:
    return int(min(4096, max(256, n // 512)))


def _seeds(seed: int, n_batches: int) -> np.ndarray:
    return ((int(seed) * 1_000_003 + np.arange(n_batches, dtype=np.int64) * 7919 + 12345) % (2 ** 31 - 1)).astype(np.int64)


def _run(mode: int, geom: BlockGeometry, mua, mus, g, n, fres: bool, cam: np.ndarray, settings,
         faces, grid, n_photons: int, seed: int, pidx, lp, ld, llay, wph, src, qv, wscale: float,
         C: int, do_vox: bool, peel: bool = True):
    from spectraforge.studio import transport3d_mc as k

    L = len(geom.layer_names)
    cos_a = math.cos(math.radians(settings.acceptance_half_angle_deg))
    pl, pn = _plane_arrays(geom)
    fg, fgi, cnorm, visible = _face_tables(geom, faces, cam, cos_a)
    ncell = cnorm.size
    X, Z = geom.footprint_mm
    e0 = np.array([1.0, 0.0, 0.0]) if abs(cam[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    t1 = np.cross(cam, e0)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(cam, t1)
    fp = np.array([X / 2, Z / 2, X / grid.nx, Z / grid.nz, cam[0], cam[1], cam[2], cos_a,
                   2.0 * math.pi * (1.0 - cos_a), *t1, *t2])
    peel = bool(peel and visible["top"])
    ip = np.array([grid.nx, grid.nz, L, grid.sublayers, int(fres), mode, C, int(do_vox), int(peel)], np.int64)
    lay = np.ascontiguousarray(np.stack([mua, mus, g, n]).astype(float))
    nvox = grid.nx * grid.nz * L * grid.sublayers if do_vox else 1
    bs = _batch_size(n_photons)
    nb_total = int(math.ceil(n_photons / bs))
    seeds = _seeds(seed, nb_total)
    per_batch = 8 * (2 * nvox + ncell * C + ncell * (2 + L) + k.N_DIAG)
    chunk = int(min(128, max(8, 3.0e8 // max(per_batch, 1))))
    acc_abs = np.zeros(nvox)
    acc_flu = np.zeros(nvox)
    acc_rad = np.zeros((ncell, C))
    acc_orig = np.zeros((ncell, 2 + L))
    acc_diag = np.zeros(k.N_DIAG)
    for b0 in range(0, nb_total, chunk):
        nb = min(chunk, nb_total - b0)
        ab = np.zeros((nb, nvox))
        fl = np.zeros((nb, nvox))
        rd = np.zeros((nb, ncell, C))
        og = np.zeros((nb, ncell, 2 + L))
        dg = np.zeros((nb, k.N_DIAG))
        numba.set_parallel_chunksize(1)      # dynamic load balancing (P/E cores); results unaffected
        try:
            k.run_chunk(b0, nb, bs, n_photons, seeds, fp, ip, pl, pn, lay, fg, fgi, cnorm, pidx, lp,
                        ld, llay, float(wph), src, qv, float(wscale), ab, fl, rd, og, dg)
        finally:
            numba.set_parallel_chunksize(0)
        acc_abs += ab.sum(0)
        acc_flu += fl.sum(0)
        acc_rad += rd.sum(0)
        acc_orig += og.sum(0)
        acc_diag += dg.sum(0)
    return acc_abs, acc_flu, acc_rad, acc_orig, acc_diag, visible


def _check_optics(geom: BlockGeometry, o: RegionOptics, toggles: PhysicsToggles | None):
    L = len(geom.layer_names)
    mua = np.asarray(o.mua_per_mm, float).reshape(-1)
    mus = np.asarray(o.mus_per_mm, float).reshape(-1)
    g = np.asarray(o.g, float).reshape(-1)
    n = np.asarray(o.n, float).reshape(-1)
    if not (mua.size == mus.size == g.size == n.size == L):
        raise ValueError(f"RegionOptics must have one entry per layer ({L})")
    if (mua < 0).any() or (mus < 0).any() or (n < 1).any() or (np.abs(g) >= 1).any():
        raise ValueError("invalid optical properties (mua, mus >= 0; n >= 1; |g| < 1)")
    if toggles is not None and not toggles.scattering:
        mus = np.zeros_like(mus)
    return mua, mus, g, n


def excitation_pass(geom: BlockGeometry, optics: RegionOptics, launch: BlockLaunch, camera_dir_local,
                    settings: VolumeMCSettings, *, n_photons: int, seed: int,
                    toggles: PhysicsToggles | None = None, peel_off: bool = True) -> ExcitationTallies:
    """Transport the excitation light through the block. See the module docstring for units."""
    t0 = time.perf_counter()
    mua, mus, g, n = _check_optics(geom, optics, toggles)
    fres = True if toggles is None else bool(toggles.fresnel)
    cam = _unit(camera_dir_local)
    grid = make_voxel_grid(geom, settings)
    faces = make_face_grids(geom, settings)
    L = len(geom.layer_names)
    pts, tdir, layer, Pt, spec_P, P_inc = _prepare_launch(geom, launch, n, fres)
    if P_inc <= 0:
        raise ValueError("launch carries no power")
    n_photons = int(n_photons)
    Ptot = float(Pt.sum())
    zero = lambda shape: np.zeros(shape)
    if Ptot <= 0:
        res_abs = res_flu = zero(grid.shape)
        er = {f: zero((len(faces[f].u_edges_mm) - 1, len(faces[f].v_edges_mm) - 1)) for f in FACES}
        diag = np.zeros(7)
        visible = {}
    else:
        rng = np.random.default_rng(seed + 7)
        cdf = np.cumsum(Pt)
        cdf /= cdf[-1]
        u = (np.arange(n_photons) + rng.random(n_photons)) / n_photons
        pidx = np.minimum(np.searchsorted(cdf, u, side="right"), Pt.size - 1).astype(np.int64)
        res_abs, res_flu, rad, _, diag, visible = _run(
            0, geom, mua, mus, g, n, fres, cam, settings, faces, grid, n_photons, seed, pidx,
            np.ascontiguousarray(pts), np.ascontiguousarray(tdir), layer, Ptot / n_photons,
            np.zeros((1, 1)), np.ones(1), 1.0, 1, True, peel_off)
        vol = voxel_volumes(geom, grid).ravel()
        res_abs = res_abs.reshape(grid.shape)
        res_flu = np.where(vol > 0, res_flu / np.maximum(vol, 1e-300), 0.0).reshape(grid.shape)
        er = _split_faces(rad, faces)
        er = {f: v[..., 0] for f, v in er.items()}
    norm = 1.0 / P_inc
    info = {
        "specular": spec_P * norm, "lost_steps": float(diag[3]) * norm,
        "roulette_net": float(diag[4]) * norm, "visible_faces": visible,
        "R_top": float(diag[0]) * norm, "T_bottom": float(diag[1]) * norm, "T_sides": float(diag[2]) * norm,
        "layer_absorbed_fraction": [float(res_abs[:, :, l, :].sum()) * norm for l in range(L)],
        "seconds": time.perf_counter() - t0, "launch_samples": int(pts.shape[0]),
        "mean_steps_per_photon": float(diag[6]) / n_photons,
    }
    R, T = info["R_top"], info["T_bottom"] + info["T_sides"]
    A = float(diag[5]) * norm
    info["energy_balance"] = R + T + A + info["specular"] + info["lost_steps"] - info["roulette_net"]
    return ExcitationTallies(
        ex_nm=launch.ex_nm, grid=grid, absorbed_mW=res_abs, fluence_mW_mm2=res_flu, elastic_radiance=er,
        faces=faces, diffuse_reflectance=R, transmittance=T, absorbed_fraction=A, n_photons=n_photons,
        info=info)


def emission_pass(geom: BlockGeometry, optics_nodes: list[RegionOptics], sources: np.ndarray,
                  channels: list[str], camera_dir_local, settings: VolumeMCSettings, *, n_photons: int,
                  seed: int, toggles: PhysicsToggles | None = None, peel_off: bool = True) -> EmissionTallies:
    """Radiance leaving each face toward the camera per source channel, per emission node.

    ``sources``: ``(C, nx, nz, L, S)`` emitted power per voxel (mW, before the spectral shape).
    Returns radiance ``(nu, nv, Ne, C)`` per face in mW/(mm^2 sr) for those powers; multiply by
    ``e_k(lambda)`` [1/nm]. ``n_photons`` is per node.

    Normalisation (unbiasedness): photon ``j`` picks voxel ``v`` with probability ``q_v / Q`` and carries
    ``coef_c = s_{c,v} / q_v * Q / N``; with ``T_v`` the probability that an isotropic emitter in ``v``
    exits into the camera cone through a given cell, ``E[sum_j coef_c 1(exit)] = sum_v (q_v/Q)
    (s_{c,v}/q_v)(Q/N) N T_v = sum_v s_{c,v} T_v`` -- the exact power ``s`` (mW) reaching the cell,
    which the face normalisation of the module docstring turns into radiance."""
    t0 = time.perf_counter()
    cam = _unit(camera_dir_local)
    grid = make_voxel_grid(geom, settings)
    faces = make_face_grids(geom, settings)
    L = len(geom.layer_names)
    sources = np.asarray(sources, float)
    C = sources.shape[0]
    if sources.shape[1:] != grid.shape:
        raise ValueError(f"sources shape {sources.shape} does not match grid {(C,) + grid.shape}")
    if C != len(channels):
        raise ValueError("len(channels) must equal sources.shape[0]")
    if (sources < 0).any() or not np.isfinite(sources).all():
        raise ValueError("sources must be finite and non-negative")
    fres = True if toggles is None else bool(toggles.fresnel)
    nodes = list(optics_nodes)
    Ne = len(nodes)
    src = np.ascontiguousarray(np.moveaxis(sources, 0, -1).reshape(-1, C))       # (V, C)
    qv = src.sum(1)
    Q = float(qv.sum())
    ncell = sum((len(faces[f].u_edges_mm) - 1) * (len(faces[f].v_edges_mm) - 1) for f in FACES)
    rad_all = np.zeros((ncell, Ne, C))
    orig = np.zeros((ncell, 2 + L))
    diag_all = np.zeros((Ne, 7))
    n_photons = int(n_photons)
    visible = {}
    if Q > 0:
        cdf = np.cumsum(qv)
        cdf /= cdf[-1]
        rng = np.random.default_rng(seed + 11)
        u = (np.arange(n_photons) + rng.random(n_photons)) / n_photons
        pidx = np.minimum(np.searchsorted(cdf, u, side="right"), qv.size - 1).astype(np.int64)
        # never sample empty voxels (roundoff at cdf plateaus)
        bad = qv[pidx] <= 0
        if bad.any():
            nz = np.flatnonzero(qv > 0)
            pidx[bad] = nz[np.searchsorted(nz, pidx[bad]).clip(0, nz.size - 1)]
        for e, o in enumerate(nodes):
            mua, mus, g, n = _check_optics(geom, o, toggles)
            _, _, rad, og, dg, visible = _run(
                1, geom, mua, mus, g, n, fres, cam, settings, faces, grid, n_photons, seed, pidx,
                np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1, np.int64), 0.0, src, qv, Q / n_photons,
                C, False, peel_off)
            rad_all[:, e, :] = rad
            orig += og
            diag_all[e] = dg
    radiance = _split_faces(rad_all, faces)
    od, ol = {}, {}
    w = _split_faces(orig, faces)
    for f in FACES:
        tot = w[f][..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            od[f] = np.where(tot > 0, w[f][..., 1] / tot, np.nan)
            ol[f] = np.where(tot[..., None] > 0, w[f][..., 2:] / tot[..., None], np.nan)
    norm = 1.0 / Q if Q > 0 else 0.0
    info = {
        "visible_faces": visible, "seconds": time.perf_counter() - t0, "total_emitted_mW": Q,
        "escape_top": (diag_all[:, 0] * norm).tolist(), "escape_bottom": (diag_all[:, 1] * norm).tolist(),
        "escape_sides": (diag_all[:, 2] * norm).tolist(), "reabsorbed": (diag_all[:, 5] * norm).tolist(),
        "energy_balance": ((diag_all[:, :4].sum(1) + diag_all[:, 5] - diag_all[:, 4]) * norm).tolist(),
        "common_random_numbers": True,
        "mean_steps_per_photon": float(diag_all[:, 6].sum()) / max(n_photons * Ne, 1),
    }
    return EmissionTallies(
        em_nodes_nm=np.array([o.lambda_nm for o in nodes], float), channels=list(channels),
        radiance=radiance, faces=faces, origin_depth_mm=od, origin_layer_fraction=ol,
        n_photons=n_photons, info=info)
