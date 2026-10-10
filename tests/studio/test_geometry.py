"""Analytic solids: transforms, intersections, chords, CSG."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from spectraforge.studio import geometry as G
from spectraforge.studio.presets import builtin_materials
from spectraforge.studio.schema import (
    Background, BoxShape, CuvetteShape, CylinderShape, DropletShape, Experiment, PetriDishShape,
    SceneObject, SphereShape, Transform,
)

MATS = builtin_materials()
MIDX = {m.id: i for i, m in enumerate(MATS)}


def scene_of(shape, pos=(0, 0, 0), rot=(0, 0, 0), material="water", vessel=None, background=False):
    exp = Experiment(
        materials=MATS,
        objects=[SceneObject(id="o", name="o", shape=shape, material_id=material,
                             transform=Transform(position_mm=pos, rotation_deg=rot), vessel_material_id=vessel)],
        background=Background(kind="plane" if background else "none", material_id="black_anodized" if background else None))
    return G.build_scene(exp)


def ray(o, d):
    o = np.atleast_2d(np.asarray(o, float))
    d = np.atleast_2d(np.asarray(d, float))
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


# --------------------------------------------------------------------------------------------
def test_euler_matches_threejs_reference():
    expected = np.array([[0.3535533906, -0.6123724357, 0.7071067812],
                         [0.9267766953, 0.1268264840, -0.3535533906],
                         [0.1268264840, 0.7803300859, 0.6123724357]])
    assert np.allclose(G.euler_xyz_matrix((30, 45, 60)), expected, atol=1e-9)
    # independent re-derivation of three.js makeRotationFromEuler 'XYZ' (Matrix4.js)
    a, b, c = np.radians([-20.0, 70.0, 135.0])
    A, B, C, D, E, F = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    ref = np.array([[C * E, -C * F, D], [A * F + B * E * D, A * E - B * F * D, -B * C],
                    [B * F - A * E * D, B * E + A * F * D, A * C]])
    assert np.allclose(G.euler_xyz_matrix((-20, 70, 135)), ref, atol=1e-12)
    assert np.allclose(G.euler_xyz_matrix((0, 0, 0)), np.eye(3))
    R = G.euler_xyz_matrix((33, -71, 12))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_rotation_moves_box_corner():
    # Rz(90°): local +X → world +Y
    sc = scene_of(BoxShape(size_mm=(10, 2, 2)), rot=(0, 0, 90))
    o, d = ray([0, 20, 0], [0, -1, 0])
    h = sc.first_hit(o, d)
    assert h.t[0] == pytest.approx(20 - 5)               # box now extends ±5 along world Y


# ------------------------------------------------------------------ chords
def test_sphere_chords():
    sc = scene_of(SphereShape(radius_mm=7.0), pos=(1, 7, 2))
    o, d = ray([1, 30, 2], [0, -1, 0])
    h = sc.first_hit(o, d)
    assert h.t[0] == pytest.approx(16.0) and h.t_exit[0] == pytest.approx(30.0)
    assert h.entering[0] and np.allclose(h.normal[0], [0, 1, 0])
    for b in (1.0, 3.5, 6.9):                             # offset rays: chord = 2 sqrt(R² - b²)
        o, d = ray([1 + b, 30, 2], [0, -1, 0])
        h = sc.first_hit(o, d)
        assert h.t_exit[0] - h.t[0] == pytest.approx(2 * np.sqrt(49 - b * b), rel=1e-9)
    o, d = ray([1 + 7.01, 30, 2], [0, -1, 0])
    assert not sc.first_hit(o, d).found[0]


def test_box_chords_and_normals():
    sc = scene_of(BoxShape(size_mm=(20, 6, 10)), pos=(0, 3, 0), rot=(0, 0, 0))
    o, d = ray([[0, 20, 0], [-30, 3, 0], [0, 3, 30]], [[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    h = sc.first_hit(o, d)
    assert np.allclose(h.t, [14, 20, 25])
    assert np.allclose(h.t_exit - h.t, [6, 20, 10])
    assert np.allclose(h.normal, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    # slanted chord through a rotated box: 45° around Y, diagonal of a square footprint
    sc = scene_of(BoxShape(size_mm=(10, 4, 10)), rot=(0, 45, 0))
    o, d = ray([-30, 0, 0], [1, 0, 0])
    h = sc.first_hit(o, d)
    assert h.t_exit[0] - h.t[0] == pytest.approx(10 * np.sqrt(2))


def test_cylinder_chords():
    sc = scene_of(CylinderShape(radius_mm=5, height_mm=8), pos=(0, 4, 0))
    o, d = ray([[0, 30, 0], [-20, 4, 0], [-20, 4, 3.0]], [[0, -1, 0], [1, 0, 0], [1, 0, 0]])
    h = sc.first_hit(o, d)
    assert np.allclose(h.t_exit - h.t, [8, 10, 8])
    assert np.allclose(h.normal, [[0, 1, 0], [-1, 0, 0], [-0.8, 0, 0.6]])
    # axis-parallel ray outside the radius misses; inside it spans the full height
    o, d = ray([[6.0, 30, 0], [4.0, 30, 0]], [[0, -1, 0], [0, -1, 0]])
    h = sc.first_hit(o, d)
    assert not h.found[0] and h.found[1]


def test_droplet_cap():
    # sessile: base radius 3, height 1.5 → sphere radius (9+2.25)/3 = 3.75
    sc = scene_of(DropletShape(base_radius_mm=3.0, height_mm=1.5), pos=(0, 0, 0))
    o, d = ray([0, 10, 0], [0, -1, 0])
    h = sc.first_hit(o, d)
    assert h.t[0] == pytest.approx(10 - 1.5) and h.t_exit[0] == pytest.approx(10.0)
    assert np.allclose(h.normal[0], [0, 1, 0])
    # horizontal ray just above the base sees the full base diameter
    o, d = ray([-10, 1e-4, 0], [1, 0, 0])
    h = sc.first_hit(o, d)
    assert h.t_exit[0] - h.t[0] == pytest.approx(6.0, abs=2e-3)
    # exit through the flat base has a downward outward normal
    o, d = ray([0, 0.5, 0], [0, -1, 0])
    h = sc.first_hit(o, d)
    assert not h.entering[0] and h.t[0] == pytest.approx(0.5) and np.allclose(h.normal[0], [0, -1, 0])
    # taller than a hemisphere (h > a): sphere radius (a²+h²)/(2h)
    sc = scene_of(DropletShape(base_radius_mm=2.0, height_mm=5.0))
    o, d = ray([0, 12, 0], [0, -1, 0])
    h = sc.first_hit(o, d)
    assert h.t[0] == pytest.approx(7.0) and h.t_exit[0] - h.t[0] == pytest.approx(5.0)
    Rs = (4 + 25) / 10
    o, d = ray([-10, 5 - Rs, 0], [1, 0, 0])               # through the equator
    h = sc.first_hit(o, d)
    assert h.t_exit[0] - h.t[0] == pytest.approx(2 * Rs)


def test_petri_dish_csg_annulus_and_fill():
    shp = PetriDishShape(inner_radius_mm=30.0, wall_mm=1.2, height_mm=15.0, base_mm=1.2, fill_height_mm=4.0)
    sc = scene_of(shp, material="fluorescein_10uM", vessel="glass")
    roles = {s.role: i for i, s in enumerate(sc.solids)}
    assert set(roles) == {"vessel", "fill"}
    glass, fill = roles["vessel"], roles["fill"]
    # horizontal ray at y = 10 (above the liquid, in the wall zone) crossing the whole dish:
    o = np.array([[-50.0, 10.0, 0.0]])
    d = np.array([[1.0, 0.0, 0.0]])
    events, t0 = [], 0.0
    for _ in range(6):
        h = sc.first_hit(o, d, t0 + G.EPS_T_MM)
        if not h.found[0]:
            break
        events.append((h.t[0], bool(h.entering[0]), int(h.solid[0])))
        t0 = h.t[0]
    ts = [e[0] for e in events]
    assert np.allclose(ts, [50 - 31.2, 50 - 30, 50 + 30, 50 + 31.2])
    assert [e[1] for e in events] == [True, False, True, False] and all(e[2] == glass for e in events)
    # wall chord is the wall thickness
    h = sc.first_hit(o, d)
    assert h.t_exit[0] - h.t[0] == pytest.approx(1.2)
    # vertical ray at the centre: liquid surface y=5.2, liquid/glass interface y=1.2, bottom y=0
    o = np.array([[0.0, 20.0, 0.0]])
    d = np.array([[0.0, -1.0, 0.0]])
    h = sc.first_hit(o, d)
    assert h.t[0] == pytest.approx(20 - 5.2) and h.solid[0] == fill and h.t_exit[0] - h.t[0] == pytest.approx(4.0)
    # media look-up
    pts = np.array([[0, 3.0, 0], [0, 0.5, 0], [0, 10, 0], [30.6, 8, 0], [0, -1, 0], [0, 5.0, 29.9]])
    loc = sc.locate(pts)
    assert list(loc) == [fill, glass, -1, glass, -1, fill]
    # fill clamped
    big = scene_of(PetriDishShape(fill_height_mm=99.0), material="water", vessel="glass")
    assert any("clamped" in w for w in big.warnings)


def test_cuvette_open_top_shell():
    shp = CuvetteShape(inner_mm=(10, 40, 10), wall_mm=1.25, fill_height_mm=30.0)
    sc = scene_of(shp, material="fluorescein_10uM", vessel="glass")
    glass = [i for i, s in enumerate(sc.solids) if s.role == "vessel"][0]
    fill = [i for i, s in enumerate(sc.solids) if s.role == "fill"][0]
    # from above: open top → liquid first (y = 1.25 + 30)
    o, d = ray([0, 100, 0], [0, -1, 0])
    h = sc.first_hit(o, d)
    assert h.solid[0] == fill and h.t[0] == pytest.approx(100 - 31.25)
    # above the liquid the cavity is air; the wall top is at y=41.25
    assert sc.locate(np.array([[0, 35, 0], [6.0, 35, 0], [0, 0.5, 0], [0, 10, 0]])).tolist() == [-1, glass, glass, fill]
    # through both side walls at y = 35: 2 glass chords of 1.25
    o, d = ray([-20, 35, 0], [1, 0, 0])
    h = sc.first_hit(o, d)
    assert h.solid[0] == glass and h.t_exit[0] - h.t[0] == pytest.approx(1.25)


def test_background_slab_and_priority():
    exp = Experiment(materials=MATS,
                     objects=[SceneObject(id="b", name="b", shape=BoxShape(size_mm=(10, 2, 10)), material_id="water",
                                          transform=Transform(position_mm=(0, 1, 0)))],
                     background=Background(kind="plane", material_id="black_anodized", thickness_mm=5.0, extent_mm=100.0))
    sc = G.build_scene(exp)
    assert sc.solids[-1].object_index == G.BACKGROUND_OBJECT
    o, d = ray([30, 10, 0], [0, -1, 0])
    h = sc.first_hit(o, d)
    assert h.t[0] == pytest.approx(10) and h.t_exit[0] == pytest.approx(15) and sc.solids[h.solid[0]].object_index == -2
    assert sc.locate(np.array([[0, 0.5, 0], [0, -0.5, 0], [60, 1, 0]])).tolist()[:2] == [0, 1]
    # outside the slab extent nothing is there
    o, d = ray([80, 10, 0], [0, -1, 0])
    assert not sc.first_hit(o, d).found[0]


def test_ray_starting_inside_returns_exit():
    sc = scene_of(SphereShape(radius_mm=5.0), pos=(0, 5, 0))
    o, d = ray([0, 5, 0], [0, 1, 0])
    h = sc.first_hit(o, d)
    assert not h.entering[0] and h.t[0] == pytest.approx(5.0) and h.t_exit[0] == pytest.approx(5.0)
    assert np.allclose(h.normal[0], [0, 1, 0])
    # t_min and exclude
    h = sc.first_hit(o, d, t_min=6.0)
    assert not h.found[0]
    h = sc.first_hit(o, d, exclude=np.array([0]))
    assert not h.found[0]


def test_invisible_objects_skipped():
    exp = Experiment(materials=MATS, background=Background(kind="none", material_id=None),
                     objects=[SceneObject(id="b", name="b", shape=BoxShape(), material_id="water", visible=False)])
    assert G.build_solids(exp) == []


# ------------------------------------------------------------------ property tests
SHAPES = [
    BoxShape(size_mm=(12, 5, 9)), SphereShape(radius_mm=6.0), CylinderShape(radius_mm=4.0, height_mm=7.0),
    DropletShape(base_radius_mm=3.0, height_mm=1.5), DropletShape(base_radius_mm=2.0, height_mm=5.0),
    PetriDishShape(inner_radius_mm=20.0, fill_height_mm=4.0), CuvetteShape(),
]
angle = st.floats(-180, 180)


@given(idx=st.integers(0, len(SHAPES) - 1), rx=angle, ry=angle, rz=angle,
       oth=st.floats(0, np.pi), oph=st.floats(0, 2 * np.pi), jit=st.tuples(*[st.floats(-1, 1)] * 3))
@settings(max_examples=150, deadline=None)
def test_hits_lie_on_surface_with_outward_unit_normals(idx, rx, ry, rz, oth, oph, jit):
    sc = scene_of(SHAPES[idx], pos=(3, 1, -2), rot=(rx, ry, rz), material="water", vessel="glass")
    centre = np.array([3.0, 1.0, -2.0]) + 0.5 * np.array(jit) * 10
    org = centre + 90 * np.array([np.sin(oth) * np.cos(oph), np.cos(oth), np.sin(oth) * np.sin(oph)])
    o, d = ray(org, centre - org)
    t0, k = G.EPS_T_MM, 0
    for _ in range(8):
        h = sc.first_hit(o, d, t0)
        if not h.found[0]:
            break
        p = (o + d * h.t[0])[0]
        n = h.normal[0]
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-9)
        s = sc.solids[h.solid[0]]
        eps = 1e-5
        inside = s.contains((p - eps * n)[None, :])[0]
        outside = s.contains((p + eps * n)[None, :])[0]
        assert inside and not outside, (s.name, p, n)
        assert (np.dot(d[0], n) < 0) == bool(h.entering[0]) or abs(np.dot(d[0], n)) < 1e-6
        assert h.t_exit[0] >= h.t[0] - 1e-12
        t0 = h.t[0] + 1e-6


# --------------------------------------------------------------------------------------------
# numba backend == NumPy reference
# --------------------------------------------------------------------------------------------
def _all_shapes_scene():
    exp = Experiment(
        materials=MATS,
        objects=[
            SceneObject(id="b", name="b", shape=BoxShape(size_mm=(20, 8, 12)), material_id="water",
                        transform=Transform(position_mm=(-30, 4, 0), rotation_deg=(10, 25, -15))),
            SceneObject(id="s", name="s", shape=SphereShape(radius_mm=7), material_id="water",
                        transform=Transform(position_mm=(0, 8, 0))),
            SceneObject(id="c", name="c", shape=CylinderShape(radius_mm=6, height_mm=10), material_id="water",
                        transform=Transform(position_mm=(25, 5, 5), rotation_deg=(0, 0, 70))),
            SceneObject(id="d", name="d", shape=DropletShape(base_radius_mm=5, height_mm=3), material_id="water",
                        transform=Transform(position_mm=(0, 0, 25), rotation_deg=(0, 40, 0))),
            SceneObject(id="p", name="p", shape=PetriDishShape(), material_id="water", vessel_material_id="glass",
                        transform=Transform(position_mm=(-30, 0, -40), rotation_deg=(5, 0, 8))),
            SceneObject(id="v", name="v", shape=CuvetteShape(), material_id="water", vessel_material_id="glass",
                        transform=Transform(position_mm=(30, 0, -40), rotation_deg=(0, 30, 3))),
        ],
        background=Background(kind="plane", material_id="black_anodized"))
    return G.build_scene(exp)


def test_numba_backend_matches_numpy_reference():
    sc = _all_shapes_scene()
    rng = np.random.default_rng(3)
    n = 4000
    o = rng.uniform([-60, -2, -70], [60, 40, 60], (n, 3))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:200] = np.eye(3)[rng.integers(0, 3, 200)] * rng.choice([-1, 1], (200, 1))     # axis-parallel rays
    tm = rng.choice([1e-6, 0.5], n)
    ex = np.where(rng.random(n) < 0.3, rng.integers(0, len(sc.solids), n), -1)
    sc.backend = "numpy"
    ref = sc.first_hit(o, d, tm, exclude=ex)
    loc_ref = sc.locate(o)
    sc.backend = "numba"
    try:
        got = sc.first_hit(o, d, tm, exclude=ex)
        loc = sc.locate(o)
    finally:
        sc.backend = "numba"
    assert (got.solid == ref.solid).all() and (got.entering == ref.entering).all()
    assert (np.isfinite(got.t) == np.isfinite(ref.t)).all()
    f = np.isfinite(ref.t)
    np.testing.assert_allclose(got.t[f], ref.t[f], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got.normal[f], ref.normal[f], atol=1e-9)
    np.testing.assert_allclose(got.t_exit[f], ref.t_exit[f], rtol=1e-10, atol=1e-10)
    assert (loc == loc_ref).all()
    assert f.mean() > 0.3 and (loc >= 0).any()
