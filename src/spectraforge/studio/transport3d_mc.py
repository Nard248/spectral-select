"""Numba kernels of the 3D layered-block Monte Carlo (see :mod:`spectraforge.studio.transport3d`).

One photon-packet engine serves both passes:

* ``mode 0`` (excitation): packets enter through the block surface (Fresnel already applied by the
  caller), carry power in mW, and tally absorbed power + track-length fluence per layer-conforming
  voxel and the elastic radiance leaving each face towards the camera;
* ``mode 1`` (emission): packets start isotropically inside source voxels with transport weight 1
  and carry a per-channel coefficient vector ``coef[c] = s[c,v]/q(v) * Q/N``; only escape tallies.

Geometry (local frame): layer ``i`` lies between planes ``k = i`` and ``k = i + 1`` of
``pl[k] = (a, b, c)``, ``y = a + b x + c z`` (``pl[0] = 0`` is the bottom face), plus four vertical
side planes.  Light leaving into air is split deterministically (``w(1-R)`` leaves, ``wR`` is
reflected); interior interfaces use the MCML probabilistic Fresnel decision (weight preserved).

Camera estimators (top face): ``I_PEEL`` switches the next-event (peel-off) estimator
(:func:`_peel_top`, derivation in :mod:`spectraforge.studio.transport3d`).  At every scattering event (and at the
launch of an emission packet) the contribution of the straight line of sight to the camera is added to the
top-face cell where it leaves; the acceptance-cone exit tally is then used only for packets that have no
peel-off pending (``refl``: launched excitation packets and packets reflected since their last scatter).
Side faces always use the cone estimator.  Events whose vertical optical depth to the top exceeds
``TAU_CUT`` are skipped (their contribution is < 1e-6 of the packet weight).

Determinism: the caller splits the photons into fixed batches; each batch re-seeds the (per-thread)
numba generator and writes into its own private tally row, so the result is independent of the
thread count.  Rows are reduced in batch order by the caller.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from spectraforge.studio._numba_env import parallel_njit

from spectraforge.studio.transport_mc import _hg, fresnel_R

EPS_T = 1.0e-9            # layers thinner than this (mm) are treated as absent at that point
TAU_CUT = 14.0            # peel-off is skipped when the vertical optical depth to the top exceeds this (e^-14 ~ 8e-7)
MAX_STEPS = 2_000_000     # per-photon safety cap (a photon trapped in a sliver)
BIG = 1.0e30

# diagnostics columns (weights in the units of the run: mW for excitation, emitted mW for emission)
D_R_TOP, D_T_BOT, D_T_SIDE, D_LOST, D_ROUL, D_ABS, D_STEPS = 0, 1, 2, 3, 4, 5, 6
N_DIAG = 7

# fp / ip packing
F_HX, F_HZ, F_DX, F_DZ, F_CX, F_CY, F_CZ, F_COSA, F_OM = 0, 1, 2, 3, 4, 5, 6, 7, 8
F_T1X, F_T1Y, F_T1Z, F_T2X, F_T2Y, F_T2Z = 9, 10, 11, 12, 13, 14       # orthonormal basis of the plane normal to the camera direction
I_NX, I_NZ, I_L, I_S, I_FRES, I_MODE, I_C, I_VOX, I_PEEL = 0, 1, 2, 3, 4, 5, 6, 7, 8


@njit(cache=True)
def _thick(pl, k, x, z):
    return (pl[k + 1, 0] - pl[k, 0]) + (pl[k + 1, 1] - pl[k, 1]) * x + (pl[k + 1, 2] - pl[k, 2]) * z


@njit(cache=True)
def _locate(x, y, z, i, pl, hx, hz, dx, dz, nx, nz, S):
    ix = int((x + hx) / dx)
    if ix < 0:
        ix = 0
    elif ix >= nx:
        ix = nx - 1
    iz = int((z + hz) / dz)
    if iz < 0:
        iz = 0
    elif iz >= nz:
        iz = nz - 1
    t = _thick(pl, i, x, z)
    s = 0
    if t > 1.0e-12:
        lo = pl[i, 0] + pl[i, 1] * x + pl[i, 2] * z
        s = int((y - lo) / t * S)
        if s < 0:
            s = 0
        elif s >= S:
            s = S - 1
    return ix, iz, s


@njit(cache=True)
def _march(x, y, z, ux, uy, uz, length, w, mua, clear, i, ix, iz, s, pl, hx, hz, dx, dz, nx, nz, S,
           L, absorbed, flu, row):
    """Exact voxel walk of a straight segment (track-length estimator).

    Adds ``∫ w(s) ds`` of the segment to ``flu`` voxel by voxel (units mW·mm) and returns the
    final weight.  In a CLEAR layer (no scattering) the weight decays deterministically,
    ``w(s) = w exp(-mua s)``, the absorbed power ``w (1 - e^{-mua l})`` is deposited per piece and
    ``∫ w ds = w (1 - e^{-mua l}) / mua``.  In a turbid layer ``w`` is constant along the segment
    (implicit-capture bookkeeping happens at the collision, in the caller).  Voxel boundaries
    (lateral grid planes and the linear sub-layer planes) are crossed exactly; after each crossing
    the position is nudged by 1e-9 mm and the voxel is re-derived from the position.
    """
    A0 = pl[i, 0]
    B0 = pl[i, 1]
    C0 = pl[i, 2]
    dA = pl[i + 1, 0] - A0
    dB = pl[i + 1, 1] - B0
    dC = pl[i + 1, 2] - C0
    rem = length
    it = 0
    cap = 4 * (nx + nz + S) + 16
    while True:
        d = rem
        if ux > 1.0e-14:
            dd = (-hx + (ix + 1) * dx - x) / ux
            if dd < d:
                d = dd
        elif ux < -1.0e-14:
            dd = (-hx + ix * dx - x) / ux
            if dd < d:
                d = dd
        if uz > 1.0e-14:
            dd = (-hz + (iz + 1) * dz - z) / uz
            if dd < d:
                d = dd
        elif uz < -1.0e-14:
            dd = (-hz + iz * dz - z) / uz
            if dd < d:
                d = dd
        f = s / S
        Af = A0 + f * dA
        Bf = B0 + f * dB
        Cf = C0 + f * dC
        rate = uy - Bf * ux - Cf * uz
        if rate < -1.0e-14:
            gap = y - (Af + Bf * x + Cf * z)
            if gap < 0.0:
                gap = 0.0
            dd = gap / (-rate)
            if dd < d:
                d = dd
        f = (s + 1) / S
        Af = A0 + f * dA
        Bf = B0 + f * dB
        Cf = C0 + f * dC
        rate = uy - Bf * ux - Cf * uz
        if rate > 1.0e-14:
            gap = Af + Bf * x + Cf * z - y
            if gap < 0.0:
                gap = 0.0
            dd = gap / rate
            if dd < d:
                d = dd
        if d < 0.0:
            d = 0.0
        if it >= cap:
            d = rem
        vi = ((ix * nz + iz) * L + i) * S + s
        if clear and mua > 1.0e-12:
            att = math.exp(-mua * d)
            dep = w * (1.0 - att)
            flu[row, vi] += dep / mua
            absorbed[row, vi] += dep
            w *= att
        else:
            flu[row, vi] += w * d
        rem -= d
        if rem <= 1.0e-10:
            break
        step = d + 1.0e-9
        x += ux * step
        y += uy * step
        z += uz * step
        rem -= 1.0e-9
        if rem <= 0.0:
            break
        ix, iz, s = _locate(x, y, z, i, pl, hx, hz, dx, dz, nx, nz, S)
        it += 1
    return w


@njit(cache=True)
def _segment(x, y, z, ux, uy, uz, d, w, mua, clear, i, cix, ciz, cs, cl, pl, hx, hz, dx, dz, nx, nz,
             S, L, absorbed, flu, row):
    """Tally the straight segment of length ``d`` from (x, y, z): fast path when it starts and ends in
    the same (convex) voxel, exact voxel walk otherwise.  ``(cix, ciz, cs)`` is the cached voxel of
    the start point, valid when ``cl == i``.  Returns ``(w_end, ix_end, iz_end, s_end)``."""
    if cl != i:
        cix, ciz, cs = _locate(x, y, z, i, pl, hx, hz, dx, dz, nx, nz, S)
    ex, ez, es = _locate(x + ux * d, y + uy * d, z + uz * d, i, pl, hx, hz, dx, dz, nx, nz, S)
    if ex == cix and ez == ciz and es == cs:
        vi = ((ex * nz + ez) * L + i) * S + es
        if clear and mua > 1.0e-12:
            att = math.exp(-mua * d)
            dep = w * (1.0 - att)
            flu[row, vi] += dep / mua
            absorbed[row, vi] += dep
            w *= att
        else:
            flu[row, vi] += w * d
    else:
        w = _march(x, y, z, ux, uy, uz, d, w, mua, clear, i, cix, ciz, cs, pl, hx, hz, dx, dz, nx,
                   nz, S, L, absorbed, flu, row)
    return w, ex, ez, es


@njit(cache=True)
def _tally_exit(face, u, v, ox, oy, oz, wex, mode, C, coef, wscale, fg, fgi, cnorm, rad, orig, row,
                l0, depth0, cx, cy, cz, cos_a):
    """Tally escaping weight ``wex`` (leaving in direction ``o``) if inside the camera cone."""
    if ox * cx + oy * cy + oz * cz < cos_a:
        return
    nu = fgi[face, 0]
    nv = fgi[face, 1]
    iu = int((u - fg[face, 0]) / fg[face, 1])
    if iu < 0:
        iu = 0
    elif iu >= nu:
        iu = nu - 1
    iv = int((v - fg[face, 2]) / fg[face, 3])
    if iv < 0:
        iv = 0
    elif iv >= nv:
        iv = nv - 1
    cell = fgi[face, 2] + iu * nv + iv
    cn = cnorm[cell]
    if cn <= 0.0:
        return
    if mode == 0:
        rad[row, cell, 0] += wex * cn
    else:
        for c in range(C):
            rad[row, cell, c] += wex * coef[c] * cn
        ws = wex * wscale
        orig[row, cell, 0] += ws
        orig[row, cell, 1] += ws * depth0
        orig[row, cell, 2 + l0] += ws


@njit(cache=True)
def _tau_min(x, y, z, i, pl, lay, L):
    """Lower bound of the optical depth of ANY line of sight from the point to the top surface: the vertical
    optical depth (a slanted path is only longer).  Used to skip peel-off events that cannot contribute."""
    t = (pl[i + 1, 0] - y) + pl[i + 1, 1] * x + pl[i + 1, 2] * z
    if t < 0.0:
        t = 0.0
    tau = (lay[0, i] + lay[1, i]) * t
    for k in range(i + 1, L):
        th = _thick(pl, k, x, z)
        if th > 0.0:
            tau += (lay[0, k] + lay[1, k]) * th
    return tau


@njit(cache=True)
def _peel_top(x, y, z, i, pl, pn, lay, fres, cx, cy, cz, hx, hz, L, ch, pk, du, gx, gz, prev, res):
    """Next-event (peel-off) transfer factor from the point ``(x, y, z)`` in layer ``i`` to the camera
    through the TOP face.

    Finds the internal direction ``d`` that refracts, through the chain of planar interfaces above the
    point (each with its own tilted normal), into the camera direction ``c`` in air -- the inverse of the
    forward refraction used by the walk -- and returns ``F = T J exp(-tau)`` such that a packet of weight
    ``w`` scattering (or being emitted) at the point contributes the power per unit solid angle IN AIR
    ``dP/dOmega_air = w p(d) F`` towards the camera (see :mod:`spectraforge.studio.transport3d`).

    * ``T`` = product of the unpolarised Fresnel transmittances of the interfaces crossed (incl. exit),
    * ``J = dOmega_inside / dOmega_air = prod (n_up^2 cos_up) / (n_in^2 cos_in)`` (etendue conservation
      across each interface, ``n_a^2 cos(a) dOmega_a = n_b^2 cos(b) dOmega_b``),
    * ``tau = sum mu_t,j s_j`` over the straight segments in every layer crossed.

    The chain of layers crossed depends on where the ray crosses each plane (a layer may be absent --
    zero thickness -- there); it is found iteratively (<= 4 passes) from the previous crossing points.
    Result in ``res``: ``F, d_x, d_y, d_z, x_exit, z_exit, J`` (``F = 0`` when no direct path exists: camera
    below the plane, ray leaving through a side wall, ...)."""
    for p in range(L + 1):
        gx[p] = x
        gz[p] = z
    m_prev = -1
    F = 0.0
    xe = 0.0
    ze = 0.0
    for it in range(4):
        # ---- layer chain above the point, found at the current crossing-point guesses
        m = 0
        ch[0] = i
        pk[0] = i + 1
        kcur = i
        while True:
            pk[m] = kcur + 1
            k = -1
            px = gx[kcur + 1]
            pz = gz[kcur + 1]
            for kk in range(kcur + 1, L):
                if _thick(pl, kk, px, pz) > EPS_T:
                    k = kk
                    break
            if k < 0:
                break
            m += 1
            ch[m] = k
            kcur = k
        if it > 0 and m == m_prev:
            same = True
            for j in range(m + 1):
                if prev[j] != ch[j]:
                    same = False
            if same:
                break
        m_prev = m
        for j in range(m + 1):
            prev[j] = ch[j]
        # ---- inverse refraction air -> layer i (records the direction in every layer)
        ux = cx
        uy = cy
        uz = cz
        Tf = 1.0
        Jf = 1.0
        for j in range(m, -1, -1):
            p = pk[j]
            mx = pn[p, 0]
            my = pn[p, 1]
            mz = pn[p, 2]
            cu = ux * mx + uy * my + uz * mz
            if cu <= 1.0e-9:
                res[0] = 0.0
                return
            n_in = lay[3, ch[j]]
            n_up = 1.0
            if j < m:
                n_up = lay[3, ch[j + 1]]
            if fres and n_in != n_up:
                r = n_up / n_in
                tx = (ux - cu * mx) * r
                ty = (uy - cu * my) * r
                tz = (uz - cu * mz) * r
                s2 = tx * tx + ty * ty + tz * tz
                if s2 >= 1.0:
                    res[0] = 0.0
                    return
                ci = math.sqrt(1.0 - s2)
                ux = tx + ci * mx
                uy = ty + ci * my
                uz = tz + ci * mz
                Tf *= 1.0 - fresnel_R(n_in, n_up, ci)
                Jf *= (n_up * n_up * cu) / (n_in * n_in * ci)
            du[j, 0] = ux
            du[j, 1] = uy
            du[j, 2] = uz
        # ---- forward trace: optical depth and crossing points
        tau = 0.0
        xx = x
        yy = y
        zz = z
        for j in range(m + 1):
            p = pk[j]
            ux = du[j, 0]
            uy = du[j, 1]
            uz = du[j, 2]
            rate = uy - pl[p, 1] * ux - pl[p, 2] * uz
            if rate <= 1.0e-12:
                res[0] = 0.0
                return
            gap = pl[p, 0] + pl[p, 1] * xx + pl[p, 2] * zz - yy
            if gap < 0.0:
                gap = 0.0
            t = gap / rate
            kj = ch[j]
            tau += (lay[0, kj] + lay[1, kj]) * t
            xx += ux * t
            yy += uy * t
            zz += uz * t
            gx[p] = xx
            gz[p] = zz
        xe = xx
        ze = zz
        F = Tf * Jf * math.exp(-tau)
        res[6] = Jf
        res[1] = du[0, 0]
        res[2] = du[0, 1]
        res[3] = du[0, 2]
    if not (xe < hx and xe > -hx and ze < hz and ze > -hz):
        res[0] = 0.0
        return
    res[0] = F
    res[4] = xe
    res[5] = ze


@njit(cache=True)
def _peel_deposit(xe, ze, pc, mode, C, coef, wscale, fg, fgi, cnorm, rad, orig, row, l0, depth0):
    """Add the peel-off estimate ``pc`` of the POWER INTO THE ACCEPTANCE CONE to the top-face cell containing
    the exit point (same ``cnorm`` as the cone estimator)."""
    nu = fgi[0, 0]
    nv = fgi[0, 1]
    iu = int((xe - fg[0, 0]) / fg[0, 1])
    if iu < 0:
        iu = 0
    elif iu >= nu:
        iu = nu - 1
    iv = int((ze - fg[0, 2]) / fg[0, 3])
    if iv < 0:
        iv = 0
    elif iv >= nv:
        iv = nv - 1
    cell = fgi[0, 2] + iu * nv + iv
    cn = cnorm[cell]
    if cn <= 0.0:
        return
    if mode == 0:
        rad[row, cell, 0] += pc * cn
    else:
        pcn = pc * cn
        for c in range(C):
            rad[row, cell, c] += pcn * coef[c]
        ws = pc * wscale
        orig[row, cell, 0] += ws
        orig[row, cell, 1] += ws * depth0
        orig[row, cell, 2 + l0] += ws


@njit(cache=True)
def _peel_event(x, y, z, i, ux, uy, uz, g, w, iso, pl, pn, lay, fres, fp, hx, hz, L, ch, pk, du, gx, gz, prv,
                pres, mode, C, coef, wscale, fg, fgi, cnorm, rad, orig, row, l0, depth0):
    """Next-event estimate at one scattering (or emission) event, with multiple-importance sampling.

    The camera direction is drawn uniformly from the acceptance cone (pdf ``1/Omega_c`` per sr in air); the
    inverse chain (:func:`_peel_top`) gives the internal direction, the Henyey-Greenstein density ``p`` of
    scattering ``u -> d`` and the Jacobian ``J = dOmega_in/dOmega_air``, so the competing technique
    (random-walk scattering followed by an exit into the cone) would have produced that air direction with
    density ``p J``.  The balance heuristic gives the peel-off the weight ``(1/Omega_c)/(1/Omega_c + p J)``::

        power into the cone  =  w p F Omega_c / (1 + Omega_c p J)         (F = T J e^-tau)

    The weight is bounded by ``w F Omega_c`` however forward-peaked the phase function is; the walk's own
    exit tally takes the complementary weight ``Omega_c p J / (1 + Omega_c p J)``."""
    cosa = fp[F_COSA]
    om = fp[F_OM]
    cz = 1.0 - np.random.random() * (1.0 - cosa)
    sz = math.sqrt(max(0.0, 1.0 - cz * cz))
    ph = 2.0 * math.pi * np.random.random()
    cph = sz * math.cos(ph)
    sph = sz * math.sin(ph)
    dx = cz * fp[F_CX] + cph * fp[F_T1X] + sph * fp[F_T2X]
    dy = cz * fp[F_CY] + cph * fp[F_T1Y] + sph * fp[F_T2Y]
    dz = cz * fp[F_CZ] + cph * fp[F_T1Z] + sph * fp[F_T2Z]
    _peel_top(x, y, z, i, pl, pn, lay, fres, dx, dy, dz, hx, hz, L, ch, pk, du, gx, gz, prv, pres)
    F = pres[0]
    if F <= 0.0:
        return
    if iso:
        p = 0.25 / math.pi
    else:
        csc = ux * pres[1] + uy * pres[2] + uz * pres[3]
        den = 1.0 + g * g - 2.0 * g * csc
        if den < 1.0e-12:
            den = 1.0e-12
        p = (1.0 - g * g) / (4.0 * math.pi * den * math.sqrt(den))
    pj = p * pres[6]
    _peel_deposit(pres[4], pres[5], w * p * F * om / (1.0 + om * pj), mode, C, coef, wscale, fg, fgi, cnorm, rad,
                  orig, row, l0, depth0)


@njit(cache=True)
def _nearest_boundary(x, y, z, ux, uy, uz, i, pl, hx, hz):
    """Distance along the ray to the closest boundary of layer ``i`` and its kind
    (0 lower plane, 1 upper plane, 2 +x, 3 -x, 4 +z, 5 -z; -1 = none)."""
    db = BIG
    kind = -1
    a0 = pl[i, 0]
    b0 = pl[i, 1]
    c0 = pl[i, 2]
    rate = uy - b0 * ux - c0 * uz
    if rate < -1.0e-14:
        gap = y - (a0 + b0 * x + c0 * z)
        if gap < 0.0:
            gap = 0.0
        dd = gap / (-rate)
        if dd < db:
            db = dd
            kind = 0
    a1 = pl[i + 1, 0]
    b1 = pl[i + 1, 1]
    c1 = pl[i + 1, 2]
    rate = uy - b1 * ux - c1 * uz
    if rate > 1.0e-14:
        gap = a1 + b1 * x + c1 * z - y
        if gap < 0.0:
            gap = 0.0
        dd = gap / rate
        if dd < db:
            db = dd
            kind = 1
    if ux > 1.0e-14:
        dd = (hx - x) / ux
        if dd < 0.0:
            dd = 0.0
        if dd < db:
            db = dd
            kind = 2
    elif ux < -1.0e-14:
        dd = (-hx - x) / ux
        if dd < 0.0:
            dd = 0.0
        if dd < db:
            db = dd
            kind = 3
    if uz > 1.0e-14:
        dd = (hz - z) / uz
        if dd < 0.0:
            dd = 0.0
        if dd < db:
            db = dd
            kind = 4
    elif uz < -1.0e-14:
        dd = (-hz - z) / uz
        if dd < 0.0:
            dd = 0.0
        if dd < db:
            db = dd
            kind = 5
    return db, kind


@njit(cache=True)
def _run_batch(row, p0, p1, seed, fp, ip, pl, pn, lay, fg, fgi, cnorm, pidx, lp, ld, llay, wph,
               src, qv, wscale, absorbed, flu, rad, orig, diag):
    np.random.seed(seed)
    hx = fp[F_HX]
    hz = fp[F_HZ]
    dx = fp[F_DX]
    dz = fp[F_DZ]
    cx = fp[F_CX]
    cy = fp[F_CY]
    cz = fp[F_CZ]
    cos_a = fp[F_COSA]
    nx = ip[I_NX]
    nz = ip[I_NZ]
    L = ip[I_L]
    S = ip[I_S]
    fres = ip[I_FRES] != 0
    mode = ip[I_MODE]
    C = ip[I_C]
    do_vox = ip[I_VOX] != 0
    peel = ip[I_PEEL] != 0
    om = fp[F_OM]
    coef = np.zeros(max(C, 1))
    ch = np.zeros(L + 2, np.int64)
    pk = np.zeros(L + 2, np.int64)
    prv = np.zeros(L + 2, np.int64)
    du = np.zeros((L + 2, 3))
    gx = np.zeros(L + 2)
    gz = np.zeros(L + 2)
    pres = np.zeros(7)   # _peel_top result slots: F, d_xyz, x_exit, z_exit, J
    for p in range(p0, p1):
        # ---------------------------------------------------------------- launch
        l0 = 0
        depth0 = 0.0
        if mode == 0:
            j = pidx[p]
            x = lp[j, 0]
            y = lp[j, 1]
            z = lp[j, 2]
            ux = ld[j, 0]
            uy = ld[j, 1]
            uz = ld[j, 2]
            i = llay[j]
            w = wph
        else:
            v = pidx[p]
            s0 = v % S
            r1 = v // S
            i = r1 % L
            r2 = r1 // L
            iz0 = r2 % nz
            ix0 = r2 // nz
            xlo = -hx + ix0 * dx
            zlo = -hz + iz0 * dz
            tmax = 0.0
            for a in range(2):
                for b in range(2):
                    tc = _thick(pl, i, xlo + a * dx, zlo + b * dz)
                    if tc > tmax:
                        tmax = tc
            x = xlo + 0.5 * dx
            z = zlo + 0.5 * dz
            for _ in range(200):
                xt = xlo + np.random.random() * dx
                zt = zlo + np.random.random() * dz
                if np.random.random() * tmax <= _thick(pl, i, xt, zt):
                    x = xt
                    z = zt
                    break
            t = _thick(pl, i, x, z)
            y = pl[i, 0] + pl[i, 1] * x + pl[i, 2] * z + (s0 + np.random.random()) / S * t
            uy = 2.0 * np.random.random() - 1.0
            st = math.sqrt(max(0.0, 1.0 - uy * uy))
            ph = 2.0 * math.pi * np.random.random()
            ux = st * math.cos(ph)
            uz = st * math.sin(ph)
            w = 1.0
            l0 = i
            depth0 = pl[L, 0] + pl[L, 1] * x + pl[L, 2] * z - y
            q = qv[v]
            for c in range(C):
                coef[c] = src[v, c] / q * wscale
        # ``refl``: the packet has no direct-path peel-off estimate pending, so a camera-cone exit through the
        # top face must be tallied by the cone estimator.  True after any Fresnel reflection since the last
        # scattering event, and at the launch of an excitation packet (the un-scattered entry beam is a delta
        # in direction, which the peel-off cannot sample).  False after a scatter / emission event.
        refl = mode == 0
        pj_last = 0.25 / math.pi       # density of the last scattering direction (isotropic emission at the launch)
        jc = 1.0                       # prod (n_up^2 cos_up)/(n_in^2 cos_in) of the transmissions since the last event
        if peel and mode == 1 and _tau_min(x, y, z, i, pl, lay, L) < TAU_CUT:
            _peel_event(x, y, z, i, ux, uy, uz, 0.0, w, True, pl, pn, lay, fres, fp, hx, hz, L, ch, pk, du, gx, gz,
                        prv, pres, mode, C, coef, wscale, fg, fgi, cnorm, rad, orig, row, l0, depth0)
        s_left = -math.log(1.0 - np.random.random())
        wmin = 1.0e-3 * w
        cl = -1
        cix = 0
        ciz = 0
        cs = 0
        steps = 0
        while True:
            # ---- Russian roulette at 1e-4 of the launch weight (MCML scheme; exact energy bookkeeping)
            if w < wmin:
                if w <= 0.0:
                    break
                if np.random.random() < 0.1:
                    diag[row, D_ROUL] += 9.0 * w * wscale
                    w *= 10.0
                else:
                    diag[row, D_ROUL] -= w * wscale
                    break
            steps += 1
            if steps > MAX_STEPS:
                diag[row, D_LOST] += w * wscale
                break
            mua = lay[0, i]
            mus = lay[1, i]
            g = lay[2, i]
            ni = lay[3, i]
            mut = mua + mus
            clear = mus <= 1.0e-9
            # ---- boundary distance: skipped when the whole free flight stays inside the layer
            #      (the layer is convex, so testing the end point suffices)
            inside = False
            if not clear:
                d_free = s_left / mut
                xe = x + ux * d_free
                ye = y + uy * d_free
                ze = z + uz * d_free
                inside = (xe < hx and xe > -hx and ze < hz and ze > -hz
                          and ye > pl[i, 0] + pl[i, 1] * xe + pl[i, 2] * ze
                          and ye < pl[i + 1, 0] + pl[i + 1, 1] * xe + pl[i + 1, 2] * ze)
            if inside:
                db = BIG
                kind = 0
            else:
                db, kind = _nearest_boundary(x, y, z, ux, uy, uz, i, pl, hx, hz)
                if kind < 0:                  # cannot happen inside a closed block
                    diag[row, D_LOST] += w * wscale
                    break
            # ---- free flight
            if (not clear) and s_left / mut < db:
                d = s_left / mut
                dw = w * mua / mut
                if do_vox:
                    w, cix, ciz, cs = _segment(x, y, z, ux, uy, uz, d, w, mua, False, i, cix, ciz, cs,
                                               cl, pl, hx, hz, dx, dz, nx, nz, S, L, absorbed, flu,
                                               row)
                    cl = i
                    if dw > 0.0:
                        absorbed[row, ((cix * nz + ciz) * L + i) * S + cs] += dw
                x += ux * d
                y += uy * d
                z += uz * d
                if dw > 0.0:
                    diag[row, D_ABS] += dw * wscale
                    w -= dw
                if mus > 0.0:
                    if peel and w > 0.0 and _tau_min(x, y, z, i, pl, lay, L) < TAU_CUT:
                        _peel_event(x, y, z, i, ux, uy, uz, g, w, False, pl, pn, lay, fres, fp, hx, hz, L, ch, pk,
                                    du, gx, gz, prv, pres, mode, C, coef, wscale, fg, fgi, cnorm, rad, orig, row,
                                    l0, depth0)
                    uxo = ux
                    uyo = uy
                    uzo = uz
                    ux, uy, uz = _hg(ux, uy, uz, g)
                    refl = False
                    jc = 1.0
                    csn = uxo * ux + uyo * uy + uzo * uz
                    den = 1.0 + g * g - 2.0 * g * csn
                    if den < 1.0e-12:
                        den = 1.0e-12
                    pj_last = (1.0 - g * g) / (4.0 * math.pi * den * math.sqrt(den))
                s_left = -math.log(1.0 - np.random.random())
                continue
            # ---- move to the boundary
            if clear:
                w0 = w
                if do_vox:
                    w, cix, ciz, cs = _segment(x, y, z, ux, uy, uz, db, w, mua, True, i, cix, ciz,
                                               cs, cl, pl, hx, hz, dx, dz, nx, nz, S, L, absorbed,
                                               flu, row)
                    cl = i
                elif mua > 0.0:
                    w *= math.exp(-mua * db)
                diag[row, D_ABS] += (w0 - w) * wscale
            else:
                if do_vox:
                    w, cix, ciz, cs = _segment(x, y, z, ux, uy, uz, db, w, mua, False, i, cix, ciz,
                                               cs, cl, pl, hx, hz, dx, dz, nx, nz, S, L, absorbed,
                                               flu, row)
                    cl = i
                s_left -= db * mut
            x += ux * db
            y += uy * db
            z += uz * db
            # ---- boundary event
            if kind >= 2:
                # vertical side face -> air
                if kind == 2:
                    mx = 1.0
                    mz = 0.0
                    face = 1
                    uc = z
                elif kind == 3:
                    mx = -1.0
                    mz = 0.0
                    face = 2
                    uc = z
                elif kind == 4:
                    mx = 0.0
                    mz = 1.0
                    face = 3
                    uc = x
                else:
                    mx = 0.0
                    mz = -1.0
                    face = 4
                    uc = x
                ci = ux * mx + uz * mz
                if ci < 0.0:
                    ci = 0.0
                R = 0.0
                if fres:
                    R = fresnel_R(ni, 1.0, ci)
                wex = w * (1.0 - R)
                if wex > 0.0:
                    if fres:
                        ct = math.sqrt(max(0.0, 1.0 - ni * ni * (1.0 - ci * ci)))
                        kf = ct - ni * ci
                        ox = ni * ux + kf * mx
                        oy = ni * uy
                        oz = ni * uz + kf * mz
                    else:
                        ox = ux
                        oy = uy
                        oz = uz
                    diag[row, D_T_SIDE] += wex * wscale
                    _tally_exit(face, uc, y, ox, oy, oz, wex, mode, C, coef, wscale, fg, fgi,
                                cnorm, rad, orig, row, l0, depth0, cx, cy, cz, cos_a)
                w *= R
                if w <= 0.0:
                    break
                refl = True
                ux -= 2.0 * ci * mx
                uz -= 2.0 * ci * mz
            elif kind == 1:
                # upper interface of layer i
                k = -1
                for kk in range(i + 1, L):
                    if _thick(pl, kk, x, z) > EPS_T:
                        k = kk
                        break
                mx = pn[i + 1, 0]
                my = pn[i + 1, 1]
                mz = pn[i + 1, 2]
                ci = ux * mx + uy * my + uz * mz
                if ci < 0.0:
                    ci = 0.0
                if k < 0:
                    R = 0.0
                    if fres:
                        R = fresnel_R(ni, 1.0, ci)
                    wex = w * (1.0 - R)
                    if wex > 0.0:
                        if fres:
                            ct = math.sqrt(max(0.0, 1.0 - ni * ni * (1.0 - ci * ci)))
                            kf = ct - ni * ci
                            ox = ni * ux + kf * mx
                            oy = ni * uy + kf * my
                            oz = ni * uz + kf * mz
                        else:
                            ox = ux
                            oy = uy
                            oz = uz
                        diag[row, D_R_TOP] += wex * wscale
                        wtal = wex
                        if peel and not refl:
                            # direct path: shares the estimate with the peel-off (balance-heuristic weight)
                            jt = jc
                            if fres and ci > 1.0e-9:
                                jt *= ct / (ni * ni * ci)
                            pjt = om * pj_last * jt
                            wtal = wex * pjt / (1.0 + pjt)
                        _tally_exit(0, x, z, ox, oy, oz, wtal, mode, C, coef, wscale, fg, fgi,
                                    cnorm, rad, orig, row, l0, depth0, cx, cy, cz, cos_a)
                    w *= R
                    if w <= 0.0:
                        break
                    refl = True
                    ux -= 2.0 * ci * mx
                    uy -= 2.0 * ci * my
                    uz -= 2.0 * ci * mz
                else:
                    n2 = lay[3, k]
                    if (not fres) or n2 == ni:
                        i = k
                    else:
                        R = fresnel_R(ni, n2, ci)
                        if np.random.random() < R:
                            refl = True
                            ux -= 2.0 * ci * mx
                            uy -= 2.0 * ci * my
                            uz -= 2.0 * ci * mz
                        else:
                            eta = ni / n2
                            ct = math.sqrt(max(0.0, 1.0 - eta * eta * (1.0 - ci * ci)))
                            kf = ct - eta * ci
                            ux = eta * ux + kf * mx
                            uy = eta * uy + kf * my
                            uz = eta * uz + kf * mz
                            if ci > 1.0e-9:
                                jc *= (n2 * n2 * ct) / (ni * ni * ci)
                            i = k
            else:
                # lower interface of layer i (kind 0); outward-down normal = -pn
                k = -1
                for kk in range(i - 1, -1, -1):
                    if _thick(pl, kk, x, z) > EPS_T:
                        k = kk
                        break
                mx = -pn[i, 0]
                my = -pn[i, 1]
                mz = -pn[i, 2]
                ci = ux * mx + uy * my + uz * mz
                if ci < 0.0:
                    ci = 0.0
                if k < 0:
                    R = 0.0
                    if fres:
                        R = fresnel_R(ni, 1.0, ci)
                    wex = w * (1.0 - R)
                    diag[row, D_T_BOT] += wex * wscale
                    w *= R
                    if w <= 0.0:
                        break
                    refl = True
                    ux -= 2.0 * ci * mx
                    uy -= 2.0 * ci * my
                    uz -= 2.0 * ci * mz
                else:
                    n2 = lay[3, k]
                    if (not fres) or n2 == ni:
                        i = k
                    else:
                        R = fresnel_R(ni, n2, ci)
                        if np.random.random() < R:
                            refl = True
                            ux -= 2.0 * ci * mx
                            uy -= 2.0 * ci * my
                            uz -= 2.0 * ci * mz
                        else:
                            eta = ni / n2
                            ct = math.sqrt(max(0.0, 1.0 - eta * eta * (1.0 - ci * ci)))
                            kf = ct - eta * ci
                            ux = eta * ux + kf * mx
                            uy = eta * uy + kf * my
                            uz = eta * uz + kf * mz
                            i = k
        diag[row, D_STEPS] += steps


@parallel_njit(cache=True)
def run_chunk(first_batch, nb, bs, n_total, seeds, fp, ip, pl, pn, lay, fg, fgi, cnorm, pidx, lp,
              ld, llay, wph, src, qv, wscale, absorbed, flu, rad, orig, diag):
    """Run ``nb`` consecutive batches; batch ``b`` owns row ``b - first_batch`` of every output."""
    for cb in prange(nb):
        b = first_batch + cb
        p0 = b * bs
        p1 = p0 + bs
        if p1 > n_total:
            p1 = n_total
        _run_batch(cb, p0, p1, seeds[b], fp, ip, pl, pn, lay, fg, fgi, cnorm, pidx, lp, ld, llay,
                   wph, src, qv, wscale, absorbed, flu, rad, orig, diag)
