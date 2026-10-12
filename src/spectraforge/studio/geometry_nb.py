"""Numba kernels for the exact ray/CSG queries of :mod:`spectraforge.studio.geometry`.

A scene is flattened into plain arrays (:func:`compile_solids`): each solid is one of

* mode 0 – a single convex primitive,
* mode 1 – ``A ∩ B``       (the droplet: sphere ∩ half-space),
* mode 2 – ``A \\ B``       (petri-dish / cuvette shells),

with primitives typed 0 = box ``(lo, hi)``, 1 = sphere ``(c, r)``, 2 = Y-cylinder ``(y0, y1, r)``,
3 = half-space ``(n, c)``. The per-ray scalar code is a literal port of the vectorised NumPy
intersections in ``geometry.py`` (same tie-breaking, same epsilons), parallelised over rays with
``prange``. Every ray is independent, so results do not depend on the thread count.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from spectraforge.studio._numba_env import parallel_njit

_INF = np.inf
PT_BOX, PT_SPHERE, PT_CYL, PT_HALF = 0, 1, 2, 3
MODE_SINGLE, MODE_AND, MODE_DIFF = 0, 1, 2


# --------------------------------------------------------------------------------------------
# Primitive intervals (local frame) -> 8-tuple (t_in, t_out, nin_xyz, nout_xyz); empty = (+inf, -inf).
# Scalars/tuples only (no per-call array allocation) so the kernels stay allocation-free.
# --------------------------------------------------------------------------------------------
@njit(cache=True)
def _box(prm, ox, oy, oz, dx, dy, dz):
    tn_best = -_INF
    tf_best = _INF
    ai = 0
    ao = 0
    sai = 1.0
    sao = 1.0
    for k in range(3):
        dk = dx if k == 0 else (dy if k == 1 else dz)
        ok = ox if k == 0 else (oy if k == 1 else oz)
        if abs(dk) < 1e-12:
            dk = -1e-12 if dk < 0 else 1e-12
        inv = 1.0 / dk
        t1 = (prm[k] - ok) * inv
        t2 = (prm[3 + k] - ok) * inv
        tn = min(t1, t2)
        tf = max(t1, t2)
        sg = 1.0 if dk > 0 else -1.0
        if k == 0 or tn > tn_best:
            tn_best = tn
            ai = k
            sai = sg
        if k == 0 or tf < tf_best:
            tf_best = tf
            ao = k
            sao = sg
    if not (tf_best > tn_best):
        tn_best, tf_best = _INF, -_INF
    return (tn_best, tf_best,
            -sai if ai == 0 else 0.0, -sai if ai == 1 else 0.0, -sai if ai == 2 else 0.0,
            sao if ao == 0 else 0.0, sao if ao == 1 else 0.0, sao if ao == 2 else 0.0)


@njit(cache=True)
def _sphere(prm, ox, oy, oz, dx, dy, dz):
    r = prm[3]
    oc0, oc1, oc2 = ox - prm[0], oy - prm[1], oz - prm[2]
    b = oc0 * dx + oc1 * dy + oc2 * dz
    disc = b * b - (oc0 * oc0 + oc1 * oc1 + oc2 * oc2 - r * r)
    if not (disc > 0):
        return (_INF, -_INF, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sq = np.sqrt(disc)
    t_in = -b - sq
    t_out = -b + sq
    if not (t_out > t_in):
        return (_INF, -_INF, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (t_in, t_out,
            (ox + t_in * dx - prm[0]) / r, (oy + t_in * dy - prm[1]) / r, (oz + t_in * dz - prm[2]) / r,
            (ox + t_out * dx - prm[0]) / r, (oy + t_out * dy - prm[1]) / r, (oz + t_out * dz - prm[2]) / r)


@njit(cache=True)
def _cyl(prm, ox, oy, oz, dx, dy, dz):
    y0, y1, rad = prm[0], prm[1], prm[2]
    a = dx * dx + dz * dz
    b = ox * dx + oz * dz
    c = ox * ox + oz * oz - rad * rad
    par = a < 1e-14
    a_s = 1.0 if par else a
    disc = b * b - a_s * c
    hit = (disc > 0) and (not par)
    if hit:
        sq = np.sqrt(disc)
        r_in = (-b - sq) / a_s
        r_out = (-b + sq) / a_s
    elif par and c <= 0:
        r_in, r_out = -_INF, _INF
    else:
        r_in, r_out = _INF, -_INF
    pdy = abs(dy) < 1e-14
    dy_s = 1.0 if pdy else dy
    ta = (y0 - oy) / dy_s
    tb = (y1 - oy) / dy_s
    in_slab = (oy >= y0) and (oy <= y1)
    if pdy:
        if in_slab:
            y_in, y_out = -_INF, _INF
        else:
            y_in, y_out = _INF, -_INF
    else:
        y_in, y_out = min(ta, tb), max(ta, tb)
    sgn = 1.0 if pdy else dy
    cin = -(1.0 if sgn > 0 else (-1.0 if sgn < 0 else 0.0))
    ri = r_in if np.isfinite(r_in) else 0.0
    ro = r_out if np.isfinite(r_out) else 0.0
    if r_in >= y_in:
        t_in = r_in
        n0, n1, n2 = (ox + ri * dx) / rad, 0.0, (oz + ri * dz) / rad
    else:
        t_in = y_in
        n0, n1, n2 = 0.0, cin, 0.0
    if r_out <= y_out:
        t_out = r_out
        m0, m1, m2 = (ox + ro * dx) / rad, 0.0, (oz + ro * dz) / rad
    else:
        t_out = y_out
        m0, m1, m2 = 0.0, -cin, 0.0
    if not (t_out > t_in):
        t_in, t_out = _INF, -_INF
    return (t_in, t_out, n0, n1, n2, m0, m1, m2)


@njit(cache=True)
def _half(prm, ox, oy, oz, dx, dy, dz):
    num = prm[6] - (ox * prm[3] + oy * prm[4] + oz * prm[5])
    den = dx * prm[3] + dy * prm[4] + dz * prm[5]
    par = abs(den) < 1e-14
    den_s = 1.0 if par else den
    t = num / den_s
    inside = num >= 0
    if par:
        if inside:
            t_in, t_out = -_INF, _INF
        else:
            t_in, t_out = _INF, -_INF
    else:
        t_in = t if den < 0 else -_INF
        t_out = t if den > 0 else _INF
    if not (t_out > t_in):
        t_in, t_out = _INF, -_INF
    return (t_in, t_out, prm[3], prm[4], prm[5], prm[3], prm[4], prm[5])


@njit(cache=True)
def _prim(kind, prm, ox, oy, oz, dx, dy, dz):
    if kind == 0:
        return _box(prm, ox, oy, oz, dx, dy, dz)
    if kind == 1:
        return _sphere(prm, ox, oy, oz, dx, dy, dz)
    if kind == 2:
        return _cyl(prm, ox, oy, oz, dx, dy, dz)
    return _half(prm, ox, oy, oz, dx, dy, dz)


@njit(cache=True)
def _prim_contains(kind, prm, x, y, z):
    if kind == 0:
        return (x >= prm[0] and x <= prm[3] and y >= prm[1] and y <= prm[4]
                and z >= prm[2] and z <= prm[5])
    if kind == 1:
        a0, a1, a2 = x - prm[0], y - prm[1], z - prm[2]
        return a0 * a0 + a1 * a1 + a2 * a2 <= prm[3] * prm[3]
    if kind == 2:
        return ((x * x + z * z <= prm[2] * prm[2]) and y >= prm[0] and y <= prm[1])
    return x * prm[3] + y * prm[4] + z * prm[5] <= prm[6]


@njit(cache=True)
def _solid_intervals(mode, pk, pp, ox, oy, oz, dx, dy, dz):
    """Up to two intervals of a local-frame solid -> ``(n_iv, iv0, iv1)`` (8-tuples)."""
    a = _prim(pk[0], pp[0], ox, oy, oz, dx, dy, dz)
    empty = (_INF, -_INF, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if mode == 0:
        return 1, a, empty
    b = _prim(pk[1], pp[1], ox, oy, oz, dx, dy, dz)
    if mode == 1:                                    # A ∩ B
        if a[0] >= b[0]:
            ti, n0, n1, n2 = a[0], a[2], a[3], a[4]
        else:
            ti, n0, n1, n2 = b[0], b[2], b[3], b[4]
        if a[1] <= b[1]:
            to, m0, m1, m2 = a[1], a[5], a[6], a[7]
        else:
            to, m0, m1, m2 = b[1], b[5], b[6], b[7]
        if not (to > ti):
            ti, to = _INF, -_INF
        return 1, (ti, to, n0, n1, n2, m0, m1, m2), empty
    # A \ B
    a_ok = a[1] > a[0]
    b_ok = b[1] > b[0]
    cut = b[0] if b_ok else _INF
    if a[1] <= cut:
        to1, m0, m1, m2 = a[1], a[5], a[6], a[7]
    else:
        to1, m0, m1, m2 = cut, -b[2], -b[3], -b[4]
    ti1 = a[0] if a_ok else _INF
    to1 = to1 if a_ok else -_INF
    if not (to1 > ti1):
        ti1, to1 = _INF, -_INF
    iv0 = (ti1, to1, a[2], a[3], a[4], m0, m1, m2)
    if a[0] >= b[1]:
        ti2, n0, n1, n2 = a[0], a[2], a[3], a[4]
    else:
        ti2, n0, n1, n2 = b[1], -b[5], -b[6], -b[7]
    ti2 = max(a[0], b[1]) if b_ok else _INF
    ok2 = a_ok and b_ok
    ti2 = ti2 if ok2 else _INF
    to2 = a[1] if ok2 else -_INF
    if not (to2 > ti2):
        ti2, to2 = _INF, -_INF
    return 2, iv0, (ti2, to2, n0, n1, n2, a[5], a[6], a[7])


# --------------------------------------------------------------------------------------------
# Scene queries
# --------------------------------------------------------------------------------------------
@njit(cache=True)
def _loc(R, p, x, y, z, translate):
    if translate:
        x, y, z = x - p[0], y - p[1], z - p[2]
    return (x * R[0, 0] + y * R[1, 0] + z * R[2, 0],
            x * R[0, 1] + y * R[1, 1] + z * R[2, 1],
            x * R[0, 2] + y * R[1, 2] + z * R[2, 2])


@njit(cache=True)
def _first_hit_one(ox, oy, oz, dx, dy, dz, tm, excl, R, P, centers, radius, mode, pkind, pparam):
    """Nearest crossing beyond ``tm`` -> (t, nx, ny, nz, solid, entering, t_exit)."""
    S = R.shape[0]
    bt = _INF
    bn0 = bn1 = bn2 = 0.0
    bs = -1
    be = False
    bx = _INF
    for si in range(S):
        if excl == si:
            continue
        oc0, oc1, oc2 = centers[si, 0] - ox, centers[si, 1] - oy, centers[si, 2] - oz
        b = oc0 * dx + oc1 * dy + oc2 * dz
        d2 = oc0 * oc0 + oc1 * oc1 + oc2 * oc2 - b * b
        r2 = radius[si] * radius[si]
        if d2 > r2:
            continue
        if not (b + np.sqrt(max(r2 - d2, 0.0)) > tm):
            continue
        lx, ly, lz = _loc(R[si], P[si], ox, oy, oz, True)
        mx, my, mz = _loc(R[si], P[si], dx, dy, dz, False)
        niv, iv0, iv1 = _solid_intervals(mode[si], pkind[si], pparam[si], lx, ly, lz, mx, my, mz)
        for k in range(niv):
            iv = iv0 if k == 0 else iv1
            if not (iv[1] > iv[0]):
                continue
            for ent in range(2):
                t = iv[0] if ent == 0 else iv[1]
                if np.isfinite(t) and t > tm and t < bt:
                    bt = t
                    q0 = iv[2] if ent == 0 else iv[5]
                    q1 = iv[3] if ent == 0 else iv[6]
                    q2 = iv[4] if ent == 0 else iv[7]
                    bn0 = R[si, 0, 0] * q0 + R[si, 0, 1] * q1 + R[si, 0, 2] * q2
                    bn1 = R[si, 1, 0] * q0 + R[si, 1, 1] * q1 + R[si, 1, 2] * q2
                    bn2 = R[si, 2, 0] * q0 + R[si, 2, 1] * q1 + R[si, 2, 2] * q2
                    bs = si
                    be = ent == 0
                    bx = iv[1]
    return bt, bn0, bn1, bn2, bs, be, bx


@njit(cache=True)
def _locate_one(x, y, z, R, P, centers, radius, mode, pkind, pparam):
    for si in range(R.shape[0]):
        oc0, oc1, oc2 = x - centers[si, 0], y - centers[si, 1], z - centers[si, 2]
        if oc0 * oc0 + oc1 * oc1 + oc2 * oc2 > radius[si] * radius[si]:
            continue
        lx, ly, lz = _loc(R[si], P[si], x, y, z, True)
        m = _prim_contains(pkind[si, 0], pparam[si, 0], lx, ly, lz)
        if mode[si] == 1:
            m = m and _prim_contains(pkind[si, 1], pparam[si, 1], lx, ly, lz)
        elif mode[si] == 2:
            m = m and (not _prim_contains(pkind[si, 1], pparam[si, 1], lx, ly, lz))
        if m:
            return si
    return -1


@parallel_njit(cache=True)
def first_hit_nb(origins, dirs, t_min, exclude, R, P, centers, radius, mode, pkind, pparam):
    n = origins.shape[0]
    best_t = np.empty(n)
    best_n = np.zeros((n, 3))
    best_s = np.empty(n, np.int64)
    best_e = np.empty(n, np.bool_)
    best_x = np.empty(n)
    for i in prange(n):
        t, n0, n1, n2, s, e, x = _first_hit_one(origins[i, 0], origins[i, 1], origins[i, 2],
                                                dirs[i, 0], dirs[i, 1], dirs[i, 2], t_min[i], exclude[i],
                                                R, P, centers, radius, mode, pkind, pparam)
        best_t[i] = t
        best_n[i, 0], best_n[i, 1], best_n[i, 2] = n0, n1, n2
        best_s[i] = s
        best_e[i] = e
        best_x[i] = x
    return best_t, best_n, best_s, best_e, best_x


@parallel_njit(cache=True)
def locate_nb(points, R, P, centers, radius, mode, pkind, pparam):
    n = points.shape[0]
    res = np.empty(n, np.int64)
    for i in prange(n):
        res[i] = _locate_one(points[i, 0], points[i, 1], points[i, 2], R, P, centers, radius, mode, pkind, pparam)
    return res


# --------------------------------------------------------------------------------------------
# Fused volumetric shadow ray (tracer ``_shadow(volume=True)`` for clear media)
# --------------------------------------------------------------------------------------------
@njit(cache=True)
def _fresnel_R(cos_i, n1, n2):
    cos_i = min(max(cos_i, 0.0), 1.0)
    eta = n1 / n2
    rad = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if rad < 0.0:
        return 1.0
    cos_t = np.sqrt(rad)
    a = n1 * cos_i
    b = n2 * cos_t
    rs = (a - b) / (a + b) if a + b > 0 else 1.0
    c = n2 * cos_i
    d = n1 * cos_t
    rp = (c - d) / (c + d) if c + d > 0 else 1.0
    return 0.5 * (rs * rs + rp * rp)


@njit(cache=True)
def _refract(dx, dy, dz, nx, ny, nz, n1, n2):
    dn = dx * nx + dy * ny + dz * nz
    if dn > 0:
        nx, ny, nz = -nx, -ny, -nz
    cos_i = min(max(-(dx * nx + dy * ny + dz * nz), 0.0), 1.0)
    eta = n1 / n2
    rad = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if rad < 0.0:
        k = 2.0 * (dx * nx + dy * ny + dz * nz)
        tx, ty, tz = dx - k * nx, dy - k * ny, dz - k * nz
    else:
        c = eta * cos_i - np.sqrt(rad)
        tx, ty, tz = eta * dx + c * nx, eta * dy + c * ny, eta * dz + c * nz
    nrm = np.sqrt(tx * tx + ty * ty + tz * tz)
    if nrm > 0:
        tx, ty, tz = tx / nrm, ty / nrm, tz / nrm
    return tx, ty, tz


@parallel_njit(cache=True)
def volume_shadow_nb(pts, u, dist, lit, start, solid_n, solid_clear, fresnel, shadows, eps_t, probe,
                     max_cross, R, P, centers, radius, mode, pkind, pparam):
    """Per sample point: ``T`` (product of Fresnel T from the light-entry face outward, 0 if occluded),
    ``face_t``, ``face_cos_u``, ``face_cos_ext``, ``face_cos_int`` (see ``tracer._shadow``)."""
    n = pts.shape[0]
    T = np.zeros(n)
    face_t = np.full(n, np.nan)
    face_cu = np.zeros(n)
    face_ce = np.zeros(n)
    face_ci = np.zeros(n)
    for i in prange(n):
        if not lit[i]:
            continue
        ox, oy, oz = pts[i, 0], pts[i, 1], pts[i, 2]
        ux, uy, uz = u[i, 0], u[i, 1], u[i, 2]
        cur = start[i]
        t_prev = 0.0
        ncr = 0
        blocked = False
        nrm = np.empty((max_cross, 3))
        nsrc = np.empty(max_cross)
        nx_ = np.empty(max_cross)
        max_it = max_cross if shadows else 1
        done = False
        for it in range(max_it):
            ht, h0, h1, h2, _hs, _he, _hx = _first_hit_one(ox, oy, oz, ux, uy, uz, t_prev + eps_t, -1,
                                                          R, P, centers, radius, mode, pkind, pparam)
            if not (np.isfinite(ht) and ht < dist[i]):
                done = True
                break
            after = _locate_one(ox + ux * (ht + probe), oy + uy * (ht + probe), oz + uz * (ht + probe),
                                R, P, centers, radius, mode, pkind, pparam)
            if h0 * ux + h1 * uy + h2 * uz < 0.0:
                h0, h1, h2 = -h0, -h1, -h2
            nrm[ncr, 0], nrm[ncr, 1], nrm[ncr, 2] = h0, h1, h2
            nsrc[ncr] = solid_n[after] if after >= 0 else 1.0
            nx_[ncr] = solid_n[cur] if cur >= 0 else 1.0
            if ncr == 0:
                face_t[i] = ht
                face_cu[i] = abs(ux * h0 + uy * h1 + uz * h2)
            ncr += 1
            if shadows and after >= 0 and not solid_clear[after]:
                blocked = True
                break
            cur = after
            t_prev = ht
        if (not done) and (not blocked) and shadows:
            blocked = True                           # too many crossings: treat as occluded
        t_acc = 1.0
        if fresnel:
            if not blocked:
                dlx, dly, dlz = -ux, -uy, -uz
                for k in range(ncr - 1, -1, -1):
                    ci = abs(dlx * nrm[k, 0] + dly * nrm[k, 1] + dlz * nrm[k, 2])
                    ci = min(max(ci, 0.0), 1.0)
                    t_acc *= 1.0 - _fresnel_R(ci, nsrc[k], nx_[k])
                    dlx, dly, dlz = _refract(dlx, dly, dlz, nrm[k, 0], nrm[k, 1], nrm[k, 2], nsrc[k], nx_[k])
                    if k == 0:
                        face_ce[i] = ci
                        face_ci[i] = abs(dlx * nrm[0, 0] + dly * nrm[0, 1] + dlz * nrm[0, 2])
        elif ncr > 0:
            c = abs(-ux * nrm[0, 0] - uy * nrm[0, 1] - uz * nrm[0, 2])
            face_ce[i] = c
            face_ci[i] = c
        T[i] = 0.0 if blocked else t_acc
    return T, face_t, face_cu, face_ce, face_ci


# --------------------------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------------------------
def _flat_prim(prim):
    """``(kind, params[8])`` for a leaf primitive, or None if unsupported."""
    from spectraforge.studio import geometry as g
    prm = np.zeros(8)
    if isinstance(prim, g.Box):
        prm[:3], prm[3:6] = prim.lo, prim.hi
        return PT_BOX, prm
    if isinstance(prim, g.Sphere):
        prm[:3], prm[3] = prim.c, prim.r
        return PT_SPHERE, prm
    if isinstance(prim, g.Cylinder):
        prm[:3] = prim.y0, prim.y1, prim.r
        return PT_CYL, prm
    if isinstance(prim, g.HalfSpace):
        prm[3:6], prm[6] = prim.n, prim.c
        return PT_HALF, prm
    return None


def compile_solids(solids):
    """Flatten ``solids`` into the array tuple used by the kernels, or ``None`` if any solid has a
    structure the kernels do not support (callers then fall back to the NumPy path)."""
    from spectraforge.studio import geometry as g
    S = len(solids)
    R = np.zeros((S, 3, 3))
    P = np.zeros((S, 3))
    centers = np.zeros((S, 3))
    radius = np.zeros(S)
    mode = np.zeros(S, np.int64)
    pkind = np.zeros((S, 2), np.int64)
    pparam = np.zeros((S, 2, 8))
    for i, s in enumerate(solids):
        R[i], P[i], centers[i], radius[i] = s.R, s.p, s.center, s.radius
        pr = s.prim
        if isinstance(pr, g.Intersection):
            if len(pr.parts) != 2:
                return None
            parts, mode[i] = pr.parts, MODE_AND
        elif isinstance(pr, g.Difference):
            parts, mode[i] = (pr.a, pr.b), MODE_DIFF
        else:
            parts, mode[i] = (pr,), MODE_SINGLE
        for j, leaf in enumerate(parts):
            fp = _flat_prim(leaf)
            if fp is None:
                return None
            pkind[i, j], pparam[i, j] = fp
    return R, P, centers, radius, mode, pkind, pparam
