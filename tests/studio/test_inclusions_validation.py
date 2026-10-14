"""Physics validation of the heterogeneous (inclusion) 3D Monte Carlo and of its lateral transport
(SpectralForge Studio v1.2, brief W2 section 3).

1. null inclusion (material == host)          -> identical to the host-only run (exactly, same seed)
2. filling inclusion (box covers a whole layer) -> identical to replacing that layer's material
3. energy conservation with inclusions          -> ``tests/studio/test_inclusions.py::test_energy_conservation_*``
4. first-order perturbation theory              -> small absorbing sphere, dR = -dmua * int Phi Phi^dagger dV
5. spatially resolved reflectance R(rho)        -> independent MCML-type reference + Farrell, Patterson & Wilson 1992
6. buried fluorescent sphere                    -> contrast falls monotonically with depth (see the figure script)
7. determinism / thread invariance / no NaN    -> ``tests/studio/test_inclusions.py``

Every test prints "number vs reference" (run with ``-s``).  Items 4-6 are marked ``slow`` (1e7 photons).

References
----------
Farrell T.J., Patterson M.S., Wilson B.C., Med. Phys. 19, 879-888 (1992), doi:10.1118/1.596777 (diffusion dipole R(rho)).
Groenhuis R.A.J. et al., Appl. Opt. 22, 2456-2462 (1983); Egan W.G., Hilgeman T., Appl. Opt. 18, 3875 (1979) (internal
reflection parameter A of the extrapolated boundary, as used by Farrell 1992).
Wang L., Jacques S.L., Zheng L., Comput. Methods Programs Biomed. 47, 131-146 (1995) (MCML; the reference code below
follows its scheme: implicit capture, Russian roulette, Fresnel weight splitting at the surface).
Arridge S.R., Inverse Problems 15, R41 (1999); Hielscher A.H. et al., Phys. Med. Biol. 43, 1285 (1998)
(perturbation / Born expansion of the transport equation; pMC: Hayakawa C.K. et al., Opt. Lett. 26, 1335 (2001)).
"""
from __future__ import annotations

import math

import numba
import numpy as np
import pytest

from spectraforge.studio import transport3d as T
from spectraforge.studio.contracts import BlockGeometry, BlockLaunch, RegionOptics
from spectraforge.studio.schema import Inclusion, ThicknessProfile, VolumeMCSettings

from .test_inclusions import HOST, UNI, WEDGE, cfg, igeom, iopt, sph


# ============================================================================================
# 1. null inclusion
# ============================================================================================
@pytest.mark.parametrize("peel", [False, True])
def test_null_inclusion_equals_host_only(peel):
    """Inclusions whose material equals the host (a sphere and a voxel-aligned box): same seed, same random
    numbers, so the result is identical to the host-only run (no random number is consumed by a voxel crossing)."""
    layers = [UNI(5.0)]
    incs = [sph(c=(0, 2.5, 0), r=1.5),
            Inclusion(name="b", material_id="m", kind="box", center_mm=(5, 1.5, 2), size_mm=(3, 2, 3))]
    G, G0 = igeom(layers, incs), igeom(layers, [])
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 20_000)
    a = T.excitation_pass(G0, iopt([HOST]), launch, [0.1, 1, 0], cfg(), n_photons=200_000, seed=7, peel_off=peel)
    b = T.excitation_pass(G, iopt([HOST], [HOST[:3], HOST[:3]]), launch, [0.1, 1, 0], cfg(), n_photons=200_000, seed=7,
                          peel_off=peel)
    ra, rb = a.elastic_radiance["top"], b.elastic_radiance["top"]
    dr = float(np.abs(ra - rb).max() / ra.max())
    A_tot = float(b.absorbed_mW.sum() + b.inclusion_absorbed_mW.sum())
    print(f"\nnull inclusion (peel={peel}): R {a.diffuse_reflectance:.8f} vs {b.diffuse_reflectance:.8f}; "
          f"T {a.transmittance:.8f} vs {b.transmittance:.8f}; A {a.absorbed_fraction:.8f} vs {b.absorbed_fraction:.8f}; "
          f"max |dL|/max L = {dr:.1e}; absorbed (layer + inclusions) {A_tot:.8f} vs {a.absorbed_mW.sum():.8f}")
    assert b.diffuse_reflectance == a.diffuse_reflectance and b.transmittance == a.transmittance
    assert b.absorbed_fraction == a.absorbed_fraction
    assert dr < 1e-12
    assert A_tot == pytest.approx(float(a.absorbed_mW.sum()), rel=1e-12)
    assert b.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)


# ============================================================================================
# 2. filling inclusion
# ============================================================================================
FILL_LAYERS = [UNI(1.0), WEDGE(1.0, 3.0)]
FILL_L0 = (0.05, 10.0, 0.9, 1.4)
FILL_L1 = (0.2, 14.0, 0.8, 1.4)          # the host material of layer 1 (replaced by the inclusion)
FILL_INC = (0.6, 6.0, 0.5)               # optics of the inclusion == the replacement material of layer 1


@pytest.mark.parametrize("peel", [False, True])
def test_filling_inclusion_equals_layer_replacement(peel):
    """A box that covers the whole (wedge-shaped) layer 1 is that layer made of the inclusion material.  The box
    boundary y = 1 coincides with the layer plane; the slanted top surface cuts through voxels (the voxels that only
    graze the block must still become inclusion, not a host crust)."""
    box = Inclusion(name="f", material_id="m", kind="box", center_mm=(0, 3.0, 0), size_mm=(30, 4.0, 30))
    G, Gr = igeom(FILL_LAYERS, [box]), igeom(FILL_LAYERS, [])
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 20_000)
    ref = T.excitation_pass(Gr, iopt([FILL_L0, FILL_INC + (1.4,)]), launch, [0.1, 1, 0], cfg(), n_photons=300_000,
                            seed=7, peel_off=peel)
    fil = T.excitation_pass(G, iopt([FILL_L0, FILL_L1], [FILL_INC]), launch, [0.1, 1, 0], cfg(), n_photons=300_000,
                            seed=7, peel_off=peel)
    rel = lambda a, b: abs(a - b) / abs(a)
    ra, rb = ref.elastic_radiance["top"].sum(), fil.elastic_radiance["top"].sum()
    print(f"\nfilling inclusion (peel={peel}): R {ref.diffuse_reflectance:.6f} vs {fil.diffuse_reflectance:.6f} "
          f"({rel(ref.diffuse_reflectance, fil.diffuse_reflectance):.1e}); T {ref.transmittance:.6f} vs "
          f"{fil.transmittance:.6f}; A {ref.absorbed_fraction:.6f} vs {fil.absorbed_fraction:.6f}; "
          f"camera radiance {ra:.6e} vs {rb:.6e} ({rel(ra, rb):.1e}); absorbed in layer 1 / inclusion "
          f"{ref.info['layer_absorbed_fraction'][1]:.6f} vs {fil.info['inclusion_absorbed_fraction'][0]:.6f}")
    assert rel(ref.diffuse_reflectance, fil.diffuse_reflectance) < 0.01
    assert rel(ref.transmittance, fil.transmittance) < 0.02
    assert rel(ref.absorbed_fraction, fil.absorbed_fraction) < 0.01
    assert rel(ra, rb) < 0.01
    assert rel(ref.info["layer_absorbed_fraction"][1], fil.info["inclusion_absorbed_fraction"][0]) < 0.01
    # the host material of layer 1 absorbs nothing any more (the box covers all of it)
    assert float(fil.absorbed_mW[:, :, 1, :].sum()) < 1e-3 * float(fil.inclusion_absorbed_mW.sum())
    if not peel:
        assert fil.diffuse_reflectance == ref.diffuse_reflectance          # same random numbers: exactly equal
        assert fil.absorbed_fraction == ref.absorbed_fraction


# ============================================================================================
# 4. first-order perturbation theory
# ============================================================================================
PERT_MEDIUM = dict(mua=0.1, mus=2.0, g=0.0, n=1.4)          # albedo 0.95, mfp' = 0.5 mm, diffusion length 1.3 mm
PERT_THICKNESS, PERT_FOOTPRINT, PERT_RADIUS, PERT_DEPTH = 14.0, 40.0, 1.5, 3.0
PERT_CENTRES = [(x, z) for x in (-6.0, 0.0, 6.0) for z in (-6.0, 0.0, 6.0)]      # 3 x 3 spheres, 6 mm apart


def perturbation_study(n_photons=20_000_000, deltas=(0.01, 0.02, 0.04, 0.08), seeds=(1, 2, 3), voxel=0.15):
    """Nine identical spheres (r = 1.5 mm, centres 3 mm below the surface) in a turbid semi-infinite-like block under
    wide illumination; their absorption is raised by ``delta`` (all other optics unchanged).

    * measured: ``dR = R(null) - R(delta)`` with the SAME seed (common random numbers: photons that miss the spheres
      follow identical paths, which removes most of the Monte Carlo noise of the difference);
    * predicted (first order; Born/perturbation expansion of the transport equation, e.g. Arridge, Inverse Problems
      15, R41, 1999; Hielscher et al., Phys. Med. Biol. 43, 1285, 1998):
          dR = -delta * int Phi(r) Phi^dagger(r) dV ,
      with Phi from the null run (the absorbed power of the null inclusions divided by mu_a,host gives int Phi dV
      voxel by voxel) and Phi^dagger, the importance (adjoint) function of the top-reflectance detector, obtained by
      reciprocity from a second Monte Carlo run in which the power absorbed in the (null) spheres is re-emitted
      isotropically from the same voxels: the fraction that leaves through the top face is
      <Phi^dagger>_Phi = int Phi Phi^dagger dV / int Phi dV  (isotropic-importance approximation: exact for isotropic
      scattering when the radiance is nearly isotropic, i.e. away from the surface and the sources).
    """
    m = PERT_MEDIUM
    H = (m["mua"], m["mus"], m["g"], m["n"])
    incs = [sph(name=f"s{i}", c=(x, PERT_THICKNESS - PERT_DEPTH, z), r=PERT_RADIUS) for i, (x, z) in enumerate(PERT_CENTRES)]
    G = igeom([UNI(PERT_THICKNESS)], incs, fp=(PERT_FOOTPRINT, PERT_FOOTPRINT))
    s = VolumeMCSettings(lateral_cell_mm=10.0, exit_cell_mm=10.0, sublayers=2, acceptance_half_angle_deg=12.0,
                         inclusion_voxel_mm=voxel)
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 20_000)
    cam = [0.0, 1.0, 0.0]
    imap = T.make_inclusion_map(G, s)
    null_opt = iopt([H], [H[:3]] * len(incs))
    per_seed = []
    base = None
    for seed in seeds:
        b = T.excitation_pass(G, null_opt, launch, cam, s, n_photons=n_photons, seed=seed, peel_off=False,
                              per_photon_streams=True)
        if base is None:
            base = b
        row = []
        for d in deltas:
            p = T.excitation_pass(G, iopt([H], [(m["mua"] + d, m["mus"], m["g"])] * len(incs)), launch, cam, s,
                                  n_photons=n_photons, seed=seed, peel_off=False, per_photon_streams=True)
            row.append(b.diffuse_reflectance - p.diffuse_reflectance)
        per_seed.append(row)
    per_seed = np.array(per_seed)
    # first-order prediction from the seed-1 null run + the adjoint run
    absd = base.inclusion_absorbed_mW.reshape(-1)[imap.occ_flat]
    int_phi = float(absd.sum()) / m["mua"]                                 # int Phi dV / P_inc  [mm]
    isrc = np.zeros((1,) + imap.shape)
    isrc[0].reshape(-1)[imap.occ_flat] = absd
    em = T.emission_pass(G, [iopt([H], [H[:3]] * len(incs))], np.zeros((1,) + T.make_voxel_grid(G, s).shape), ["x"],
                         cam, s, n_photons=n_photons, seed=seeds[0] + 100, peel_off=False, inclusion_sources=isrc)
    E_phi = float(em.info["escape_top"][0])
    slope = int_phi * E_phi
    deltas = np.asarray(deltas, float)
    mean = per_seed.mean(0)
    err = per_seed.std(0, ddof=1) / math.sqrt(len(seeds)) if len(seeds) > 1 else np.full(len(deltas), np.nan)
    # quadratic through the origin  dR = a d + b d^2  (the second-order term is the nonlinearity of the large delta)
    A = np.stack([deltas, deltas ** 2], 1)
    (a_fit, b_fit), *_ = np.linalg.lstsq(A, mean, rcond=None)
    return {"deltas": deltas.tolist(), "dR_mean": mean.tolist(), "dR_err": err.tolist(), "dR_per_seed": per_seed.tolist(),
            "int_phi_mm": int_phi, "E_phi": E_phi, "slope_pred": slope,
            "dR_pred_linear": (slope * deltas).tolist(), "ratio": (mean / (slope * deltas)).tolist(),
            "a_fit": float(a_fit), "b_fit": float(b_fit), "ratio_a_fit": float(a_fit / slope),
            "R_null": base.diffuse_reflectance, "sphere_volume_mm3": float(imap.occ_volume_mm3.sum()),
            "analytic_volume_mm3": len(incs) * 4.0 / 3.0 * math.pi * PERT_RADIUS ** 3,
            "n_photons": n_photons, "seeds": list(seeds)}


@pytest.mark.slow
def test_first_order_perturbation_of_diffuse_reflectance():
    out = perturbation_study(n_photons=5_000_000, deltas=(0.01, 0.02, 0.04, 0.08), seeds=(1, 2))
    print("\nperturbation: int Phi dV / P = %.4e mm, <Phi^dagger> = %.4f, slope = %.4e per (1/mm)"
          % (out["int_phi_mm"], out["E_phi"], out["slope_pred"]))
    for d, mean, err, pr, ra in zip(out["deltas"], out["dR_mean"], out["dR_err"], out["dR_pred_linear"], out["ratio"]):
        print(f"  dmua {d:5.2f}: measured dR {mean:.4e} +- {err:.1e}   first order {pr:.4e}   ratio {ra:.3f}")
    print(f"  small-delta limit (quadratic fit): a = {out['a_fit']:.4e}, ratio to first-order slope {out['ratio_a_fit']:.3f}")
    assert out["ratio_a_fit"] == pytest.approx(1.0, abs=0.15)
    assert all(r > 0 for r in out["dR_mean"])


# ============================================================================================
# 5. spatially resolved diffuse reflectance R(rho): lateral transport
# ============================================================================================
def boundary_parameter_A(n_rel, exact=False):
    """Internal-reflection parameter ``A = (1 + r_2)/(1 - r_1)`` of the extrapolated boundary ``z_b = 2 A D``.

    ``exact=False``: A from the empirical diffuse reflection ``r_d = -1.440 n^-2 + 0.710 n^-1 + 0.668 + 0.0636 n``,
    ``A = (1 + r_d)/(1 - r_d)`` (Groenhuis et al., Appl. Opt. 22, 2456, 1983; used by Farrell, Patterson & Wilson 1992).
    ``exact=True``: from the unpolarised Fresnel reflectance R(mu) of a ray inside the medium,
    ``r_1 = 2 int R mu dmu``, ``r_2 = 3 int R mu^2 dmu`` (Haskell et al., J. Opt. Soc. Am. A 11, 2727, 1994; Keijzer,
    Star & Storchi, Appl. Opt. 27, 1820, 1988)."""
    if not exact:
        rd = -1.440 / n_rel ** 2 + 0.710 / n_rel + 0.668 + 0.0636 * n_rel
        return (1.0 + rd) / (1.0 - rd)
    mu = np.linspace(1e-7, 1.0, 400_001)
    st = n_rel * np.sqrt(1.0 - mu ** 2)
    R = np.ones_like(mu)
    m = st < 1.0
    ct = np.sqrt(1.0 - st[m] ** 2)
    c = mu[m]
    R[m] = 0.5 * (((n_rel * c - ct) / (n_rel * c + ct)) ** 2 + ((n_rel * ct - c) / (n_rel * ct + c)) ** 2)
    r1 = 2.0 * np.trapezoid(R * mu, mu)
    r2 = 3.0 * np.trapezoid(R * mu ** 2, mu)
    return (1.0 + r2) / (1.0 - r1)


def farrell_reflectance(rho, mua, musp, n_rel, exact_A=False):
    """Diffusion-theory R(rho) [1/mm^2 per unit incident (transmitted) power] of a semi-infinite medium for a normally
    incident pencil beam, with the extrapolated-boundary dipole of Farrell, Patterson & Wilson (1992), eqs. 11-12:

        R(rho) = a'/(4 pi) [ z0 (mu_eff + 1/r1) e^(-mu_eff r1) / r1^2 + (z0 + 2 zb)(mu_eff + 1/r2) e^(-mu_eff r2) / r2^2 ]

    with mu_t' = mu_a + mu_s', a' = mu_s'/mu_t', z0 = 1/mu_t', mu_eff = sqrt(3 mu_a mu_t'), zb = 2 A D, D = 1/(3 mu_t'),
    r1 = sqrt(z0^2 + rho^2), r2 = sqrt((z0 + 2 zb)^2 + rho^2); A from :func:`boundary_parameter_A`
    (``exact_A=False``: the formula of the original paper)."""
    rho = np.asarray(rho, float)
    mut = mua + musp
    D = 1.0 / (3.0 * mut)
    mueff = math.sqrt(3.0 * mua * mut)
    ap, z0 = musp / mut, 1.0 / mut
    zb = 2.0 * boundary_parameter_A(n_rel, exact_A) * D
    r1 = np.sqrt(z0 ** 2 + rho ** 2)
    r2 = np.sqrt((z0 + 2 * zb) ** 2 + rho ** 2)
    return ap / (4 * np.pi) * (z0 * (mueff + 1 / r1) * np.exp(-mueff * r1) / r1 ** 2
                               + (z0 + 2 * zb) * (mueff + 1 / r2) * np.exp(-mueff * r2) / r2 ** 2)


@numba.njit(cache=True)
def _fresnel_unpolarised(n1, n2, ci):
    if ci > 0.999999:
        r = (n1 - n2) / (n1 + n2)
        return r * r
    si = math.sqrt(1.0 - ci * ci)
    st = n1 / n2 * si
    if st >= 1.0:
        return 1.0
    ct = math.sqrt(1.0 - st * st)
    rs = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)) ** 2
    rp = ((n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)) ** 2
    return 0.5 * (rs + rp)


@numba.njit(cache=True)
def _ref_batch(seed, n, mua, mus, g, n_rel, dr, nb, zmax, out):
    np.random.seed(seed)
    mut = mua + mus
    alb = mus / mut
    w0 = 1.0 - _fresnel_unpolarised(1.0, n_rel, 1.0)
    for _ in range(n):
        w = w0
        x = y = z = 0.0
        ux = uy = 0.0
        uz = 1.0
        while True:
            s = -math.log(1.0 - np.random.random()) / mut
            if uz < 0.0 and z + s * uz < 0.0:                 # reaches the surface z = 0
                d = -z / uz
                x += ux * d
                y += uy * d
                z = 0.0
                R = _fresnel_unpolarised(n_rel, 1.0, -uz)
                k = int(math.sqrt(x * x + y * y) / dr)
                if k < nb:
                    out[k] += w * (1.0 - R)
                w *= R
                uz = -uz
                rest = s - d                                  # memoryless: continue the step in the reflected direction
                x += ux * rest
                y += uy * rest
                z += uz * rest
            else:
                x += ux * s
                y += uy * s
                z += uz * s
            if z > zmax:
                break
            w *= alb
            if g == 0.0:
                cs = 2.0 * np.random.random() - 1.0
            else:
                t = (1.0 - g * g) / (1.0 - g + 2.0 * g * np.random.random())
                cs = (1.0 + g * g - t * t) / (2.0 * g)
            sn = math.sqrt(max(0.0, 1.0 - cs * cs))
            ph = 2.0 * math.pi * np.random.random()
            cp, sp = math.cos(ph), math.sin(ph)
            if abs(uz) > 0.99999:
                nux, nuy, nuz = sn * cp, sn * sp, cs * (1.0 if uz > 0 else -1.0)
            else:
                tmp = math.sqrt(1.0 - uz * uz)
                nux = sn * (ux * uz * cp - uy * sp) / tmp + ux * cs
                nuy = sn * (uy * uz * cp + ux * sp) / tmp + uy * cs
                nuz = -sn * cp * tmp + uz * cs
            ux, uy, uz = nux, nuy, nuz
            if w < 1e-4:
                if np.random.random() < 0.1:
                    w *= 10.0
                else:
                    break


@numba.njit(parallel=True, cache=True)
def _ref_run(nphot, nbatch, mua, mus, g, n_rel, dr, nb, zmax):
    out = np.zeros((nbatch, nb))
    per = nphot // nbatch
    for b in numba.prange(nbatch):
        _ref_batch(1000 + b, per, mua, mus, g, n_rel, dr, nb, zmax, out[b])
    return out.sum(0)


def reference_radial_reflectance(mua, musp, g, n_rel, n_photons, dr=0.05, rmax=40.0, zmax=80.0):
    """Independent MCML-style pencil-beam semi-infinite Monte Carlo (numba, ~60 lines, no code shared with the studio
    engine): returns the annulus centres and R(rho) [1/mm^2 per unit INCIDENT power; includes the entry Fresnel loss]."""
    nb = int(rmax / dr)
    tot = _ref_run(int(n_photons), 140, mua, musp / (1.0 - g), g, n_rel, dr, nb, zmax) / (140 * (int(n_photons) // 140))
    edges = np.arange(nb + 1) * dr
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 0.5 * (edges[1:] + edges[:-1]), tot / area


def engine_radial_reflectance(mua, musp, g, n_rel, n_photons, *, footprint=100.0, thickness=60.0, exit_cell=0.25, seed=1):
    """R(rho) of the 3D engine: pencil beam on a ``footprint^2 x thickness`` homogeneous block, top-exit power per exit
    cell (``ExcitationTallies.top_exit_mW``) -> returns ``(rho cell centres, R per cell [1/mm^2], tallies)``."""
    G = BlockGeometry(0, (footprint, footprint), ["m"], [0], T.planes_from_layers((footprint, footprint),
                      [ThicknessProfile(thickness_mm=thickness)]), np.eye(4))
    s = VolumeMCSettings(lateral_cell_mm=20.0, exit_cell_mm=exit_cell, sublayers=2)
    opt = RegionOptics(500.0, np.array([mua]), np.array([musp / (1.0 - g)]), np.array([g]), np.array([n_rel]))
    launch = BlockLaunch(500.0, np.array([[0.0, thickness, 0.0]]), np.array([[0.0, -1.0, 0.0]]), np.array([1.0]))
    r = T.excitation_pass(G, opt, launch, [0.0, 1.0, 0.0], s, n_photons=n_photons, seed=seed)
    fg = r.faces["top"]
    uc = 0.5 * (np.asarray(fg.u_edges_mm)[1:] + np.asarray(fg.u_edges_mm)[:-1])
    vc = 0.5 * (np.asarray(fg.v_edges_mm)[1:] + np.asarray(fg.v_edges_mm)[:-1])
    rho = np.hypot(uc[:, None], vc[None, :])
    area = float(np.asarray(fg.u_edges_mm)[1] - np.asarray(fg.u_edges_mm)[0]) * float(
        np.asarray(fg.v_edges_mm)[1] - np.asarray(fg.v_edges_mm)[0])
    return rho, r.top_exit_mW / area, r


RHO_EDGES = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])      # annuli [mm]; mfp' = 1 mm


def annulus_means(rho, R, edges=RHO_EDGES):
    return np.array([R[(rho >= a) & (rho < b)].mean() for a, b in zip(edges[:-1], edges[1:])])


def ref_annulus_means(rc, Rr, edges=RHO_EDGES):
    """Annulus means of the reference rings (area-weighted: ring area ~ rho), like the mean over the engine's cells."""
    return np.array([np.average(Rr[(rc >= a) & (rc < b)], weights=rc[(rc >= a) & (rc < b)])
                     for a, b in zip(edges[:-1], edges[1:])])


def farrell_annulus_means(mua, musp, n_rel, edges=RHO_EDGES, exact_A=False):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = np.linspace(a, b, 600)
        out.append(np.trapezoid(farrell_reflectance(r, mua, musp, n_rel, exact_A) * r, r) / np.trapezoid(r, r))
    return np.array(out)


@pytest.mark.slow
def test_radial_reflectance_matches_independent_mc_and_farrell():
    """Pencil beam on a semi-infinite homogeneous block (no inclusions; mua = 0.01, mus' = 1 /mm, n = 1.4, isotropic
    scattering): the lateral transport of the 3D engine against (a) an independent MCML-type reference code
    (must agree to the Monte Carlo noise) and (b) Farrell-Patterson-Wilson diffusion theory (rho > 2 mfp': within 10 %
    with the exact Fresnel boundary parameter; the original paper's empirical A overshoots by up to 13 %)."""
    mua, musp, g, n = 0.01, 1.0, 0.0, 1.4
    rho, R, tal = engine_radial_reflectance(mua, musp, g, n, 2_000_000)
    assert tal.info["energy_balance"] == pytest.approx(1.0, abs=1e-9)
    rc, Rr = reference_radial_reflectance(mua, musp, g, n, 8_000_000)
    eng = annulus_means(rho, R)
    ref = ref_annulus_means(rc, Rr)
    T0 = 1.0 - tal.info["specular"]                                    # Farrell: per unit TRANSMITTED power
    far = farrell_annulus_means(mua, musp, n) * T0                     # original paper (Groenhuis A)
    farx = farrell_annulus_means(mua, musp, n, exact_A=True) * T0      # A from the exact Fresnel moments
    print(f"\n A (Groenhuis) = {boundary_parameter_A(n):.3f}, A (exact Fresnel) = {boundary_parameter_A(n, True):.3f}")
    print(" rho [mm]      engine        reference   Farrell(1992) Farrell(exact A)  eng/ref  eng/F92  eng/Fx   ref/F92")
    for k, (a, b) in enumerate(zip(RHO_EDGES[:-1], RHO_EDGES[1:])):
        print(f" {a:4.1f}-{b:4.1f}   {eng[k]:.4e}   {ref[k]:.4e}   {far[k]:.4e}   {farx[k]:.4e}   {eng[k] / ref[k]:.3f}   "
              f"{eng[k] / far[k]:.3f}   {eng[k] / farx[k]:.3f}   {ref[k] / far[k]:.3f}")
    print(f" total diffuse R: engine {tal.diffuse_reflectance:.5f}")
    np.testing.assert_allclose(eng, ref, rtol=0.03)                    # engine == independent MC (noise ~0.5 %)
    far_ok = RHO_EDGES[:-1] >= 2.0                                     # rho > 2 mfp'
    np.testing.assert_allclose(eng[far_ok], farx[far_ok], rtol=0.10)   # diffusion theory with the exact boundary A
    np.testing.assert_allclose(eng[far_ok], far[far_ok], rtol=0.15)    # the original (empirical-A) formula: up to 13 %


# ============================================================================================
# 6. buried fluorescent sphere: contrast against depth
# ============================================================================================
BEAD_R = 1.0
HOST_THICKNESS = 10.0
BEAD_WL = 490.0


def buried_bead_experiment(mus_prime_500: float, depth_mm: float, photons: int = 1_500_000, voxel_mm: float = 0.2,
                           seed: int = 0):
    """A fluorescein bead (sphere, radius 1 mm, top at ``depth_mm`` below the surface) inside a non-fluorescent turbid
    host of reduced scattering ``mus_prime_500`` (1/mm at 500 nm, g = 0.8, intralipid-like).  One excitation at
    490 nm, deterministic (noise sources off)."""
    from spectraforge.studio.presets import PRESETS, _layer, _obj
    from spectraforge.studio.schema import LayeredBlockShape, ScatteringSpec

    exp = PRESETS["layered_wedge_phantom"]()
    host = next(m for m in exp.materials if m.id == "intralipid_cover")
    host = host.model_copy(update={"id": "turbid_host", "scattering": ScatteringSpec(
        model="power_law", mus_prime_500_per_mm=mus_prime_500, power_b=1.0, g=0.8)})
    shape = LayeredBlockShape(
        footprint_mm=(30.0, 30.0), layers=[_layer("host", "turbid_host", HOST_THICKNESS)],
        inclusions=[Inclusion(name="bead", material_id="fluorescein_agarose_layer", kind="sphere",
                              center_mm=(0.0, HOST_THICKNESS - depth_mm - BEAD_R, 0.0), radius_mm=BEAD_R)])
    src = exp.source()
    seq = [s for s in src.sequence if s.enabled and abs(s.wavelength_nm - BEAD_WL) < 1e-6]
    tg = exp.acquisition.physics.model_copy(update={
        "power_drift": False, "shot_noise": False, "read_noise": False, "dark_current": False,
        "fixed_pattern_noise": False, "photobleaching": False})
    vs = exp.acquisition.volume_mc.model_copy(update={
        "excitation_photons": photons, "emission_photons": photons, "emission_node_step_nm": 40.0,
        "lateral_cell_mm": 1.0, "exit_cell_mm": 0.5, "sublayers": 4, "inclusion_voxel_mm": voxel_mm})
    acq = exp.acquisition.model_copy(update={"physics": tg, "volume_mc": vs, "seed": seed})
    return exp.model_copy(update={
        "materials": list(exp.materials) + [host], "acquisition": acq,
        "objects": [_obj("block", "Buried bead", shape, "turbid_host", (0.0, 0.0, 0.0))],
        "sources": [src.model_copy(update={"sequence": seq}) if s.id == src.id else s for s in exp.sources]})


def bead_signal(res, ring_mm=(3.0, 5.0)):
    """(mean radiance over the bead's footprint, mean over a ring of block pixels around it, Michelson contrast,
    per-pixel size mm).  Radiance integrated over the emission bands 505-565 nm (fluorescein; keeps the water Raman line at ~587 nm out)."""
    ex = res.excitations_nm[0]
    gt = res.ground_truth
    mask = gt["inclusion_mask:bead"]
    block = np.isfinite(gt["layer_thickness:host"])
    wl = res.emission_nm[ex]
    img = res.radiance[ex][..., (wl > 505.0) & (wl < 565.0)].sum(-1)
    ys, xs = np.nonzero(mask == 1)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.indices(img.shape)
    # pixel pitch from the block footprint (30 mm) seen through the mask of the block
    cols = np.nonzero(block.any(0))[0]
    pitch = 30.0 / (cols.max() - cols.min() + 1)
    r = np.hypot(yy - cy, xx - cx) * pitch
    on = float(img[mask == 1].mean())
    ring = float(img[block & (r >= ring_mm[0]) & (r < ring_mm[1])].mean())
    return on, ring, (on - ring) / (on + ring), pitch


@pytest.mark.slow
def test_buried_bead_contrast_decreases_monotonically_with_depth():
    from spectraforge.studio import block_capture, capture

    depths = [0.5, 1.0, 2.0, 3.0]
    out = {}
    for mp in (0.5, 1.5, 3.0):
        row = []
        for d in depths:
            block_capture.clear_cache()
            res = capture.capture(buried_bead_experiment(mp, d), scale=0.5)
            row.append(bead_signal(res)[:3])
        out[mp] = row
    print("\n buried bead: on-bead radiance | ring radiance | Michelson contrast, against depth of the bead top")
    for mp, row in out.items():
        print(f" mus'={mp:3.1f}/mm: " + "  ".join(f"d={d:.1f}: {on:.3e}|{rg:.3e}|{c:.3f}" for d, (on, rg, c) in zip(depths, row)))
    for mp, row in out.items():
        on = np.array([r[0] for r in row])
        c = np.array([r[2] for r in row])
        assert np.all(np.diff(on) < 0), (mp, on)
        assert np.all(np.diff(c) < 0), (mp, c)
    # (no ordering across mus' is asserted: scattering raises the near-surface fluence by back-scattering but blurs and
    # attenuates the emission of deeper beads, so the curves cross; they are reported in the figure)


# ============================================================================================
# performance: 4 inclusions at 0.2 mm against the same block without inclusions (brief target <= 1.6x)
# ============================================================================================
def timing_study(n_photons=2_000_000, reps=3):
    """Wall-clock ratio of the heterogeneous against the homogeneous block, interleaved runs (A B A B, so that the load
    of other jobs cancels; best of ``reps``), on the reference configuration of
    ``test_transport3d_validation.test_performance_reference_configuration`` (40 x 40 mm, 3 layers, wedge cover,
    0.5 mm lateral cells, 8 sublayers) with 4 inclusions (sphere, ellipsoid, cylindrical vessel, box) at 0.2 mm."""
    import time

    from .test_transport3d import geom, opt

    fp = (40.0, 40.0)
    layers = [UNI(2.0), UNI(0.5), WEDGE(0.0, 2.0, "x")]
    incs = [sph("s1", (-8, 3.0, -6), 1.5),
            Inclusion(name="e", material_id="m", kind="ellipsoid", center_mm=(6, 2.2, 5), semi_axes_mm=(2.5, 1.0, 1.5)),
            Inclusion(name="v", material_id="m", kind="cylinder", center_mm=(0, 1.6, 0), radius_mm=0.6, axis="x",
                      length_mm=30),
            Inclusion(name="b", material_id="m", kind="box", center_mm=(8, 1.0, -8), size_mm=(5, 1.0, 5))]
    G0 = geom(layers, fp)
    G1 = BlockGeometry(0, fp, G0.layer_names, G0.layer_material, G0.plane_coeffs, G0.world_from_local,
                       [i.name for i in incs], [0] * 4, incs)
    s = VolumeMCSettings(lateral_cell_mm=0.5, exit_cell_mm=0.5, sublayers=8, inclusion_voxel_mm=0.2)
    ip = [(0.5, 8.0, 0.8), (0.3, 20.0, 0.9), (1.5, 6.0, 0.7), (0.2, 15.0, 0.85)]
    add = lambda o: RegionOptics(o.lambda_nm, o.mua_per_mm, o.mus_per_mm, o.g, o.n, np.array([p[0] for p in ip]),
                                 np.array([p[1] for p in ip]), np.array([p[2] for p in ip]))
    o0 = opt([0.1, 0.8, 0.05], [10.0, 12.0, 20.0], [0.9, 0.8, 0.85], [1.4, 1.4, 1.4], 365.0)
    o1 = add(o0)
    o2 = opt([0.05, 0.2, 0.03], [9.0, 10.0, 18.0], [0.9, 0.8, 0.85], [1.4, 1.4, 1.4], 520.0)
    o2i = add(o2)
    launch = T.uniform_beam_launch(G0, 365.0, 1.0, 200_000, beam_half_size_mm=(15, 15))
    imap = T.make_inclusion_map(G1, s)
    grid = T.make_voxel_grid(G0, s)
    ex0 = T.excitation_pass(G0, o0, launch, [0, 1, 0], s, n_photons=200_000, seed=1)
    src = np.zeros((1,) + grid.shape)
    src[0] = ex0.absorbed_mW * 0.01
    isrc = np.zeros((1,) + imap.shape)
    isrc[0].reshape(-1)[imap.occ_flat] = 0.01 * imap.occ_volume_mm3
    runs = {
        "excitation": (lambda: T.excitation_pass(G0, o0, launch, [0, 1, 0], s, n_photons=n_photons, seed=1),
                       lambda: T.excitation_pass(G1, o1, launch, [0, 1, 0], s, n_photons=n_photons, seed=1)),
        "emission": (lambda: T.emission_pass(G0, [o2], src, ["a"], [0, 1, 0], s, n_photons=n_photons, seed=1),
                     lambda: T.emission_pass(G1, [o2i], src, ["a"], [0, 1, 0], s, n_photons=n_photons, seed=1,
                                             inclusion_sources=isrc))}
    out = {"map_shape": list(imap.shape), "n_occupied_voxels": int(imap.n_occupied),
           "occupied_fraction": float(imap.n_occupied / np.prod(imap.shape)), "n_photons": n_photons}
    for name, (f0, f1) in runs.items():
        f0(), f1()                                       # warm-up (JIT)
        t0, t1 = [], []
        for _ in range(reps):
            a = time.perf_counter(); f0(); t0.append(time.perf_counter() - a)
            a = time.perf_counter(); f1(); t1.append(time.perf_counter() - a)
        out[name] = {"seconds_homogeneous": min(t0), "seconds_4_inclusions": min(t1),
                     "ratio_best": min(t1) / min(t0), "ratio_median": float(np.median(t1) / np.median(t0)),
                     "all_homogeneous": t0, "all_4_inclusions": t1}
    return out


@pytest.mark.slow
def test_performance_four_inclusions():
    out = timing_study(n_photons=300_000, reps=3)
    print("\n4 inclusions at 0.2 mm: map %s (%.1f %% of the voxels occupied)" % (out["map_shape"], 100 * out["occupied_fraction"]))
    for k in ("excitation", "emission"):
        r = out[k]
        print(f"  {k}: x{r['ratio_best']:.2f} best, x{r['ratio_median']:.2f} median (target <= 1.6 on an idle machine)")
    # loose: the 1.6x target is judged on an idle machine; here only a regression guard on a shared one
    assert out["excitation"]["ratio_median"] < 3.5 and out["emission"]["ratio_median"] < 3.5
