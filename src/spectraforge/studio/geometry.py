"""Analytic solids with exact, vectorised ray intersections (N rays at once).

World frame: right-handed, Y up, millimetres (see :mod:`spectraforge.studio.schema`).

Every solid is a *constructive* combination of convex primitives (box, sphere, finite Y-cylinder,
half-space) in its local frame, placed by a rigid transform. A ray ``o + t d`` (``|d| = 1``) meets a
convex primitive in one interval ``[t_in, t_out]``; a CSG difference ``A \\ B`` of two convex
primitives yields at most two intervals. Everything else (first hit, medium look-up) is built on
those intervals, so rays may start anywhere, including inside a solid, and chord lengths are exact.

Object rotation convention: ``R = Rx(a) Ry(b) Rz(c)`` for ``rotation_deg = (a, b, c)``, identical to
three.js ``Matrix4.makeRotationFromEuler(new Euler(a, b, c, 'XYZ'))``.

Composite objects (local frames per ``schema.py``):

* ``petri_dish``  → *vessel* solid = outer cylinder minus the inner cavity (the cavity is open at
  the top, so the glass base and the annular wall are one CSG solid) + *fill* cylinder
  (material = object ``material_id``, ``fill_height_mm`` above the inner base).
* ``cuvette``     → *vessel* = outer box minus the open-top cavity + *fill* box.
* ``layered_block`` → ONE convex solid (the whole stack, any layer thickness profile): the box
  ``[-X/2, X/2] x [0, y_max] x [-Z/2, Z/2]`` intersected with the half-space below the (possibly
  tilted) top interface plane ``y = a + b x + c z``. It is an opaque/turbid occluder for the camera and
  shadow rays (``object_index`` = the block, ``material`` = the TOP layer's material); the interior is
  solved by the 3D Monte Carlo (:mod:`spectraforge.studio.transport3d`), not by ray tracing.
* ``droplet``     → spherical cap = sphere ∩ half-space ``y >= 0`` (also valid for h > base radius).

Performance: :meth:`Scene.first_hit` / :meth:`Scene.locate` run on parallel numba kernels
(:mod:`spectraforge.studio.geometry_nb`, a literal scalar port of the NumPy intersections below,
parity-tested to 1e-10); set ``Scene.backend = "numpy"`` to use the vectorised NumPy reference.

The background (``Background.kind == "plane"``) is a box slab ``y ∈ [-thickness, 0]`` with
``object_index = BACKGROUND_OBJECT``; it is the last solid, so on ties objects win.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectraforge.studio.schema import (
    BoxShape,
    CuvetteShape,
    CylinderShape,
    DropletShape,
    Experiment,
    LayeredBlockShape,
    PetriDishShape,
    SphereShape,
    Transform,
)

#: ``object_index`` of the background slab in :class:`Solid` / ``HitBuffer``.
BACKGROUND_OBJECT = -2
#: Rays are not allowed to re-hit anything closer than this (mm) after leaving a surface.
EPS_T_MM = 1e-6
#: Offset (mm) along a ray used to decide which solid lies just beyond a boundary crossing.
PROBE_MM = 1e-5

_INF = np.inf


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _rowmat(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``a @ m`` for (N,3) rows and a 3x3 matrix (BLAS, with the spurious floating-point warnings
    Apple Accelerate's matmul raises under numpy 2.x suppressed)."""
    with np.errstate(all="ignore"):
        return a @ m


def _rowvec(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``a @ v`` for (N,3) rows and a 3-vector."""
    with np.errstate(all="ignore"):
        return a @ v


# --------------------------------------------------------------------------------------------
# Rotation
# --------------------------------------------------------------------------------------------
def euler_xyz_matrix(rotation_deg) -> np.ndarray:
    """Rotation matrix equal to three.js ``makeRotationFromEuler(Euler(x, y, z, 'XYZ'))``."""
    a, b, c = np.radians(np.asarray(rotation_deg, dtype=float))
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rx @ ry @ rz


def object_transform(t: Transform) -> tuple[np.ndarray, np.ndarray]:
    """``(R, p)``: local→world is ``x_world = R @ x_local + p``."""
    return euler_xyz_matrix(t.rotation_deg), np.asarray(t.position_mm, dtype=float)


# --------------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------------
@dataclass
class Interval:
    """Ray/solid overlap ``[t_in, t_out]`` with the *outward* solid normals at both ends.
    Empty where ``t_out <= t_in`` (stored as ``+inf, -inf``). ``t_in = -inf`` / ``t_out = +inf``
    mean the ray starts inside / never leaves."""

    t_in: np.ndarray
    t_out: np.ndarray
    n_in: np.ndarray
    n_out: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return self.t_out > self.t_in


def _finalize(t_in, t_out, n_in, n_out) -> Interval:
    bad = ~(t_out > t_in)
    if bad.any():
        t_in = np.where(bad, _INF, t_in)
        t_out = np.where(bad, -_INF, t_out)
    return Interval(t_in, t_out, n_in, n_out)


def _empty(n: int) -> Interval:
    return Interval(np.full(n, _INF), np.full(n, -_INF), np.zeros((n, 3)), np.zeros((n, 3)))


def _intersect(a: Interval, b: Interval) -> Interval:
    use_a = a.t_in >= b.t_in
    t_in = np.where(use_a, a.t_in, b.t_in)
    n_in = np.where(use_a[:, None], a.n_in, b.n_in)
    use_a = a.t_out <= b.t_out
    t_out = np.where(use_a, a.t_out, b.t_out)
    n_out = np.where(use_a[:, None], a.n_out, b.n_out)
    return _finalize(t_in, t_out, n_in, n_out)


def _difference(a: Interval, b: Interval) -> list[Interval]:
    """``A \\ B`` for convex A, B → up to two disjoint intervals."""
    a_ok = a.valid
    b_ok = b.valid
    cut = np.where(b_ok, b.t_in, _INF)
    first_exit_a = a.t_out <= cut
    t_out1 = np.minimum(a.t_out, cut)
    n_out1 = np.where(first_exit_a[:, None], a.n_out, -b.n_in)
    iv1 = _finalize(np.where(a_ok, a.t_in, _INF), np.where(a_ok, t_out1, -_INF), a.n_in, n_out1)
    start_a = a.t_in >= b.t_out
    t_in2 = np.where(b_ok, np.maximum(a.t_in, b.t_out), _INF)
    n_in2 = np.where(start_a[:, None], a.n_in, -b.n_out)
    ok2 = a_ok & b_ok
    iv2 = _finalize(np.where(ok2, t_in2, _INF), np.where(ok2, a.t_out, -_INF), n_in2, a.n_out)
    return [iv1, iv2]


# --------------------------------------------------------------------------------------------
# Convex primitives (local frame)
# --------------------------------------------------------------------------------------------
class Primitive:
    def interval(self, o: np.ndarray, d: np.ndarray) -> Interval:  # pragma: no cover
        raise NotImplementedError

    def intervals(self, o: np.ndarray, d: np.ndarray) -> list[Interval]:
        return [self.interval(o, d)]

    def contains(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def bound(self) -> tuple[np.ndarray, float]:  # pragma: no cover
        raise NotImplementedError


class Box(Primitive):
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def interval(self, o, d):
        n = len(o)
        d = np.where(np.abs(d) < 1e-12, np.where(d < 0, -1e-12, 1e-12), d)
        inv = 1.0 / d
        t1 = (self.lo - o) * inv
        t2 = (self.hi - o) * inv
        tn = np.minimum(t1, t2)
        tf = np.maximum(t1, t2)
        r = np.arange(n)
        ax_in = tn.argmax(axis=1)
        ax_out = tf.argmin(axis=1)
        n_in = np.zeros((n, 3))
        n_out = np.zeros((n, 3))
        n_in[r, ax_in] = -np.sign(d[r, ax_in])
        n_out[r, ax_out] = np.sign(d[r, ax_out])
        return _finalize(tn[r, ax_in], tf[r, ax_out], n_in, n_out)

    def contains(self, p):
        return np.all((p >= self.lo) & (p <= self.hi), axis=1)

    def bound(self):
        return 0.5 * (self.lo + self.hi), float(0.5 * np.linalg.norm(self.hi - self.lo))


class Sphere(Primitive):
    def __init__(self, center, radius):
        self.c = np.asarray(center, dtype=float)
        self.r = float(radius)

    def interval(self, o, d):
        oc = o - self.c
        b = _dot(oc, d)
        disc = b * b - (_dot(oc, oc) - self.r * self.r)
        hit = disc > 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t_in = np.where(hit, -b - sq, _INF)
        t_out = np.where(hit, -b + sq, -_INF)
        with np.errstate(invalid="ignore"):
            n_in = (o + t_in[:, None] * d - self.c) / self.r
            n_out = (o + t_out[:, None] * d - self.c) / self.r
        return _finalize(t_in, t_out, np.nan_to_num(n_in), np.nan_to_num(n_out))

    def contains(self, p):
        return _dot(p - self.c, p - self.c) <= self.r * self.r

    def bound(self):
        return self.c, self.r


class Cylinder(Primitive):
    """Finite cylinder, axis = local Y through x = z = 0, ``y ∈ [y0, y1]``."""

    def __init__(self, y0, y1, radius):
        self.y0, self.y1, self.r = float(y0), float(y1), float(radius)

    def interval(self, o, d):
        n = len(o)
        ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        a = dx * dx + dz * dz
        b = ox * dx + oz * dz
        c = ox * ox + oz * oz - self.r * self.r
        par = a < 1e-14
        a_s = np.where(par, 1.0, a)
        disc = b * b - a_s * c
        hit = (disc > 0) & ~par
        sq = np.sqrt(np.where(hit, disc, 0.0))
        r_in = np.where(hit, (-b - sq) / a_s, np.where(par & (c <= 0), -_INF, _INF))
        r_out = np.where(hit, (-b + sq) / a_s, np.where(par & (c <= 0), _INF, -_INF))
        # caps
        pdy = np.abs(dy) < 1e-14
        dy_s = np.where(pdy, 1.0, dy)
        ta = (self.y0 - oy) / dy_s
        tb = (self.y1 - oy) / dy_s
        in_slab = (oy >= self.y0) & (oy <= self.y1)
        y_in = np.where(pdy, np.where(in_slab, -_INF, _INF), np.minimum(ta, tb))
        y_out = np.where(pdy, np.where(in_slab, _INF, -_INF), np.maximum(ta, tb))
        cap_in_n = np.zeros((n, 3))
        cap_in_n[:, 1] = -np.sign(np.where(pdy, 1.0, dy))
        cap_out_n = -cap_in_n
        with np.errstate(invalid="ignore"):
            pin = o + np.where(np.isfinite(r_in), r_in, 0.0)[:, None] * d
            pout = o + np.where(np.isfinite(r_out), r_out, 0.0)[:, None] * d
        rn_in = np.stack([pin[:, 0], np.zeros(n), pin[:, 2]], axis=1) / self.r
        rn_out = np.stack([pout[:, 0], np.zeros(n), pout[:, 2]], axis=1) / self.r
        radial = Interval(r_in, r_out, rn_in, rn_out)
        slab = Interval(y_in, y_out, cap_in_n, cap_out_n)
        return _intersect(radial, slab)

    def contains(self, p):
        return (p[:, 0] ** 2 + p[:, 2] ** 2 <= self.r ** 2) & (p[:, 1] >= self.y0) & (p[:, 1] <= self.y1)

    def bound(self):
        h = 0.5 * (self.y1 - self.y0)
        return np.array([0.0, 0.5 * (self.y0 + self.y1), 0.0]), float(np.hypot(self.r, h))


class HalfSpace(Primitive):
    """``{x : n·x <= c}``."""

    def __init__(self, normal, c):
        n = np.asarray(normal, dtype=float)
        self.n = n / np.linalg.norm(n)
        self.c = float(c)

    def interval(self, o, d):
        num = self.c - _rowvec(o, self.n)
        den = _rowvec(d, self.n)
        par = np.abs(den) < 1e-14
        den_s = np.where(par, 1.0, den)
        t = num / den_s
        inside = num >= 0
        t_in = np.where(par, np.where(inside, -_INF, _INF), np.where(den < 0, t, -_INF))
        t_out = np.where(par, np.where(inside, _INF, -_INF), np.where(den > 0, t, _INF))
        nn = np.broadcast_to(self.n, (len(o), 3)).copy()
        return _finalize(t_in, t_out, nn, nn)

    def contains(self, p):
        return _rowvec(p, self.n) <= self.c

    def bound(self):  # only used inside Intersection (never as the first part)
        return np.zeros(3), float("inf")


class Intersection(Primitive):
    def __init__(self, *parts: Primitive):
        self.parts = parts

    def interval(self, o, d):
        iv = self.parts[0].interval(o, d)
        for p in self.parts[1:]:
            iv = _intersect(iv, p.interval(o, d))
        return iv

    def contains(self, p):
        m = self.parts[0].contains(p)
        for q in self.parts[1:]:
            m &= q.contains(p)
        return m

    def bound(self):
        return self.parts[0].bound()


class Difference(Primitive):
    """``a \\ b`` (both convex) — non-convex; yields two intervals."""

    def __init__(self, a: Primitive, b: Primitive):
        self.a, self.b = a, b

    def interval(self, o, d):  # pragma: no cover - use intervals()
        raise TypeError("Difference has two intervals; call intervals()")

    def intervals(self, o, d):
        return _difference(self.a.interval(o, d), self.b.interval(o, d))

    def contains(self, p):
        return self.a.contains(p) & ~self.b.contains(p)

    def bound(self):
        return self.a.bound()


# --------------------------------------------------------------------------------------------
# Solid = primitive + rigid transform + material
# --------------------------------------------------------------------------------------------
@dataclass
class Solid:
    name: str
    object_index: int            # index into experiment.objects; BACKGROUND_OBJECT for the slab
    material_index: int          # index into experiment.materials
    role: str                    # "body" | "vessel" | "fill" | "block" | "background"
    prim: Primitive
    R: np.ndarray
    p: np.ndarray
    center: np.ndarray = field(init=False)
    radius: float = field(init=False)

    def __post_init__(self):
        c, r = self.prim.bound()
        self.center = self.R @ c + self.p
        self.radius = float(r) * (1.0 + 1e-9) + 1e-6

    def intervals(self, o: np.ndarray, d: np.ndarray) -> list[Interval]:
        """World-frame ray intervals (``d`` unit; rotation preserves ``t``)."""
        ol = _rowmat(o - self.p, self.R)
        dl = _rowmat(d, self.R)
        return [Interval(iv.t_in, iv.t_out, _rowmat(iv.n_in, self.R.T), _rowmat(iv.n_out, self.R.T))
                for iv in self.prim.intervals(ol, dl)]

    def contains(self, pts: np.ndarray) -> np.ndarray:
        return self.prim.contains(_rowmat(pts - self.p, self.R))

    def cull(self, o: np.ndarray, d: np.ndarray, t_min: np.ndarray) -> np.ndarray:
        """Indices of rays that can possibly reach this solid beyond ``t_min`` (bounding sphere)."""
        oc = self.center - o
        b = _dot(oc, d)
        d2 = _dot(oc, oc) - b * b
        r2 = self.radius * self.radius
        ok = d2 <= r2
        reach = b + np.sqrt(np.maximum(r2 - d2, 0.0)) > t_min
        return np.nonzero(ok & reach)[0]


@dataclass
class Hit:
    """Nearest boundary crossing beyond ``t_min`` for N rays (``t = inf`` where nothing is hit)."""

    t: np.ndarray            # (N,)
    normal: np.ndarray       # (N,3) OUTWARD unit normal of the crossed solid at the hit
    solid: np.ndarray        # (N,) int index into Scene.solids (-1 none)
    entering: np.ndarray     # (N,) bool: ray enters the solid (False: leaves it)
    t_exit: np.ndarray       # (N,) distance at which the ray leaves the solid (== t if leaving)

    @property
    def found(self) -> np.ndarray:
        return np.isfinite(self.t)


class Scene:
    """Collection of solids with exact ray queries."""

    #: ``"numba"`` (compiled parallel kernels, default when the CSG structure is supported) or
    #: ``"numpy"`` (reference implementation). Both give the same result to float tolerance.
    backend = "numba"

    def __init__(self, solids: list[Solid], warnings: list[str] | None = None):
        self.solids = solids
        self.warnings = warnings or []
        self._flat = None
        self._flat_done = False

    def _nb(self):
        """Flattened kernel arrays, or None when the numba backend is off/unsupported."""
        if self.backend != "numba":
            return None
        if not self._flat_done:
            from spectraforge.studio import geometry_nb
            self._flat = geometry_nb.compile_solids(self.solids) if self.solids else None
            self._flat_done = True
        return self._flat

    def __len__(self):
        return len(self.solids)

    # ---------------------------------------------------------------------------------------
    def first_hit(self, origins: np.ndarray, dirs: np.ndarray, t_min=EPS_T_MM,
                  exclude: np.ndarray | None = None) -> Hit:
        """Nearest boundary crossing with ``t > t_min`` (scalar or per-ray array).

        Rays may start inside solids: the next *entry or exit* is returned. ``exclude`` (optional
        int array (N,)) names one solid per ray that is ignored entirely."""
        n = len(origins)
        t_min = np.broadcast_to(np.asarray(t_min, dtype=float), (n,))
        flat = self._nb()
        if flat is not None and n:
            from spectraforge.studio import geometry_nb
            ex = (np.full(n, -1, np.int64) if exclude is None
                  else np.ascontiguousarray(exclude, dtype=np.int64))
            bt, bn, bs, be, bx = geometry_nb.first_hit_nb(
                np.ascontiguousarray(origins, dtype=np.float64), np.ascontiguousarray(dirs, dtype=np.float64),
                np.ascontiguousarray(t_min), ex, *flat)
            return Hit(bt, bn, bs, be, bx)
        best_t = np.full(n, _INF)
        best_n = np.zeros((n, 3))
        best_s = np.full(n, -1, dtype=np.int64)
        best_e = np.zeros(n, dtype=bool)
        best_x = np.full(n, _INF)
        for si, s in enumerate(self.solids):
            idx = s.cull(origins, dirs, t_min)
            if exclude is not None and idx.size:
                idx = idx[exclude[idx] != si]
            if idx.size == 0:
                continue
            tm = t_min[idx]
            for iv in s.intervals(origins[idx], dirs[idx]):
                ok_iv = iv.t_out > iv.t_in
                for entering in (True, False):
                    t = iv.t_in if entering else iv.t_out
                    upd = ok_iv & np.isfinite(t) & (t > tm) & (t < best_t[idx])
                    if not upd.any():
                        continue
                    sel = idx[upd]
                    best_t[sel] = t[upd]
                    best_n[sel] = (iv.n_in if entering else iv.n_out)[upd]
                    best_s[sel] = si
                    best_e[sel] = entering
                    best_x[sel] = iv.t_out[upd]
        return Hit(best_t, best_n, best_s, best_e, best_x)

    def locate(self, points: np.ndarray) -> np.ndarray:
        """Index of the first solid containing each point (priority = list order), −1 = air."""
        n = len(points)
        flat = self._nb()
        if flat is not None and n:
            from spectraforge.studio import geometry_nb
            return geometry_nb.locate_nb(np.ascontiguousarray(points, dtype=np.float64), *flat)
        res = np.full(n, -1, dtype=np.int64)
        for si, s in enumerate(self.solids):
            todo = res < 0
            if not todo.any():
                break
            oc = points - s.center
            cand = np.nonzero(todo & (_dot(oc, oc) <= s.radius * s.radius))[0]
            if cand.size:
                inside = s.contains(points[cand])
                res[cand[inside]] = si
        return res

    def locate_after(self, origins: np.ndarray, dirs: np.ndarray, t: np.ndarray,
                     probe_mm: float = PROBE_MM) -> np.ndarray:
        """Medium (solid index, −1 air) just beyond the crossing at distance ``t`` along each ray."""
        return self.locate(origins + dirs * (np.asarray(t) + probe_mm)[:, None])


# --------------------------------------------------------------------------------------------
# Building solids from an Experiment
# --------------------------------------------------------------------------------------------
def _droplet_primitive(base_r: float, h: float) -> Primitive:
    rs = (base_r * base_r + h * h) / (2.0 * h)
    return Intersection(Sphere([0.0, h - rs, 0.0], rs), HalfSpace([0.0, -1.0, 0.0], 0.0))


def block_plane_coeffs(shape: LayeredBlockShape) -> np.ndarray:
    """``(L, 3)`` ``[a, b, c]`` of the TOP interface ``y = a + b x + c z`` of every layer (bottom → top).

    Layer thickness profiles are linear in x or z, so each cumulative interface is an exact plane:
    ``t_i(x, z) = t0 + (t1 - t0) (u / size + 1/2)`` with ``u`` the profile axis coordinate (the
    profile clamps to [0, 1] outside the footprint, which is irrelevant inside it)."""
    X, Z = shape.footprint_mm
    acc = np.zeros(3)
    out = []
    for layer in shape.layers:
        pr = layer.thickness
        if pr.kind == "uniform":
            acc = acc + np.array([pr.thickness_mm, 0.0, 0.0])
        else:
            size = X if pr.axis == "x" else Z
            slope = (pr.end_thickness_mm - pr.thickness_mm) / size
            d = np.array([pr.thickness_mm + 0.5 * (pr.end_thickness_mm - pr.thickness_mm), 0.0, 0.0])
            d[1 if pr.axis == "x" else 2] = slope
            acc = acc + d
        out.append(acc.copy())
    return np.array(out)


def block_primitive(shape: LayeredBlockShape) -> Primitive:
    """Convex solid of a layered block in its local frame (see module docstring)."""
    X, Z = shape.footprint_mm
    a, b, c = block_plane_coeffs(shape)[-1]
    corners = [a + b * sx * X / 2 + c * sz * Z / 2 for sx in (-1, 1) for sz in (-1, 1)]
    nvec = np.array([-b, 1.0, -c])
    nn = float(np.linalg.norm(nvec))
    return Intersection(Box([-X / 2, 0.0, -Z / 2], [X / 2, max(corners) + 1.0, Z / 2]),
                        HalfSpace(nvec, a / nn))


def build_solids(experiment: Experiment) -> list[Solid]:
    """Instantiate every visible object (+ background slab) as :class:`Solid` s.

    Returned list order = ``locate`` priority: objects in order (vessel before fill), then the
    background. Warnings (missing vessel material, clamped fill) are in :func:`build_scene`."""
    return build_scene(experiment).solids


def build_scene(experiment: Experiment) -> Scene:
    mat_idx = {m.id: i for i, m in enumerate(experiment.materials)}
    solids: list[Solid] = []
    warns: list[str] = []
    for oi, o in enumerate(experiment.objects):
        if not o.visible:
            continue
        R, p = object_transform(o.transform)
        mi = mat_idx[o.material_id]
        sh = o.shape

        def add(name, prim, material, role):
            solids.append(Solid(f"{o.id}:{name}", oi, material, role, prim, R, p))

        vessel_id = o.vessel_material_id
        if vessel_id is None and isinstance(sh, (PetriDishShape, CuvetteShape)):
            vessel_id = "glass" if "glass" in mat_idx else None
            if vessel_id is None:
                warns.append(f"object '{o.id}': no vessel_material_id and no 'glass' material; "
                             "vessel walls omitted")
        vi = mat_idx[vessel_id] if vessel_id is not None else None

        if isinstance(sh, BoxShape):
            h = 0.5 * np.asarray(sh.size_mm, dtype=float)
            add("body", Box(-h, h), mi, "body")
        elif isinstance(sh, SphereShape):
            add("body", Sphere([0, 0, 0], sh.radius_mm), mi, "body")
        elif isinstance(sh, CylinderShape):
            add("body", Cylinder(-0.5 * sh.height_mm, 0.5 * sh.height_mm, sh.radius_mm), mi, "body")
        elif isinstance(sh, DropletShape):
            add("body", _droplet_primitive(sh.base_radius_mm, sh.height_mm), mi, "body")
        elif isinstance(sh, PetriDishShape):
            r_in = sh.inner_radius_mm
            r_out = r_in + sh.wall_mm
            if vi is not None:
                add("vessel", Difference(Cylinder(0.0, sh.height_mm, r_out),
                                         Cylinder(sh.base_mm, sh.height_mm + 1.0, r_in)), vi, "vessel")
            cap = sh.height_mm - sh.base_mm
            fill = min(sh.fill_height_mm, cap)
            if sh.fill_height_mm > cap:
                warns.append(f"object '{o.id}': fill_height clamped to {cap:g} mm")
            if fill > 0:
                add("fill", Cylinder(sh.base_mm, sh.base_mm + fill, r_in), mi, "fill")
        elif isinstance(sh, CuvetteShape):
            ix, iy, iz = sh.inner_mm
            w = sh.wall_mm
            if vi is not None:
                outer = Box([-ix / 2 - w, 0.0, -iz / 2 - w], [ix / 2 + w, iy + w, iz / 2 + w])
                cavity = Box([-ix / 2, w, -iz / 2], [ix / 2, iy + w + 1.0, iz / 2])
                add("vessel", Difference(outer, cavity), vi, "vessel")
            fill = min(sh.fill_height_mm, iy)
            if sh.fill_height_mm > iy:
                warns.append(f"object '{o.id}': fill_height clamped to {iy:g} mm")
            if fill > 0:
                add("fill", Box([-ix / 2, w, -iz / 2], [ix / 2, w + fill, iz / 2]), mi, "fill")
        elif isinstance(sh, LayeredBlockShape):
            top_mat = mat_idx[sh.layers[-1].material_id]
            solids.append(Solid(f"{o.id}:block", oi, top_mat, "block", block_primitive(sh), R, p))
        else:  # pragma: no cover
            raise TypeError(f"unsupported shape {type(sh).__name__}")

    bg = experiment.background
    if bg.kind == "plane" and bg.material_id is not None:
        e = 0.5 * bg.extent_mm
        solids.append(Solid("background", BACKGROUND_OBJECT, mat_idx[bg.material_id], "background",
                            Box([-e, -bg.thickness_mm, -e], [e, 0.0, e]), np.eye(3), np.zeros(3)))
    return Scene(solids, warns)
