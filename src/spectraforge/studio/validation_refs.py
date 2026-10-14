"""Independent references for validating the lateral transport of the 3D block Monte Carlo (Studio v1.2).

* :func:`farrell_reflectance` -- spatially resolved diffuse reflectance of a semi-infinite turbid medium for a normally
  incident pencil beam, standard diffusion theory with the extrapolated boundary condition
  (T. J. Farrell, M. S. Patterson, B. Wilson, "A diffusion theory model of spatially resolved, steady-state diffuse
  reflectance for the noninvasive determination of tissue optical properties in vivo", Med. Phys. 19, 879-888 (1992),
  doi:10.1118/1.596777, eqs. 11-14).  The internal-reflection parameter is the polynomial of R. A. J. Groenhuis,
  H. A. Ferwerda, J. J. ten Bosch, Appl. Opt. 22, 2456 (1983), doi:10.1364/AO.22.002456.
* :func:`reference_radial_mc` -- a small, separately written MCML-type code (Wang, Jacques, Zheng, Comput. Methods
  Programs Biomed. 47, 131 (1995), doi:10.1016/0169-2607(95)01640-F: one photon packet at a time, probabilistic
  Fresnel exit, implicit capture, Henyey-Greenstein, roulette; no layers, no planes, no voxels) that bins the exit
  weight in annuli.  It shares no code with ``transport3d_mc`` apart from numba.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


def farrell_reflectance(rho, mua: float, musp: float, n: float, n_ext: float = 1.0):
    """Diffuse reflectance ``R(rho)`` [1/mm^2] per unit power ENTERING the medium (Farrell 1992, eq. 11-14)."""
    rho = np.asarray(rho, float)
    mut = mua + musp
    D = 1.0 / (3.0 * mut)
    a = musp / mut
    mueff = math.sqrt(3.0 * mua * mut)
    nr = n / n_ext
    rd = -1.440 / nr ** 2 + 0.710 / nr + 0.668 + 0.0636 * nr          # Groenhuis et al. 1983
    A = (1.0 + rd) / (1.0 - rd)
    zb = 2.0 * A * D
    z0 = 1.0 / mut
    r1 = np.sqrt(z0 ** 2 + rho ** 2)
    r2 = np.sqrt((z0 + 2.0 * zb) ** 2 + rho ** 2)
    return a / (4.0 * math.pi) * (z0 * (mueff + 1.0 / r1) * np.exp(-mueff * r1) / r1 ** 2
                                  + (z0 + 2.0 * zb) * (mueff + 1.0 / r2) * np.exp(-mueff * r2) / r2 ** 2)


def farrell_annulus_average(edges, mua, musp, n, n_ext=1.0, m=400):
    """Farrell ``R`` averaged over the annuli ``[edges[i], edges[i+1]]`` (area weighted)."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = np.linspace(a + 1e-9, b, m)
        out.append(np.trapezoid(2 * math.pi * r * farrell_reflectance(r, mua, musp, n, n_ext), r)
                   / (math.pi * (b * b - a * a)))
    return np.array(out)


@njit(cache=True)
def _fresnel(n1, n2, ci):
    if ci > 0.99999:
        r = (n1 - n2) / (n1 + n2)
        return r * r
    s1 = math.sqrt(max(0.0, 1 - ci * ci))
    s2 = n1 / n2 * s1
    if s2 >= 1.0:
        return 1.0
    c2 = math.sqrt(1 - s2 * s2)
    cap = ci * c2 - s1 * s2
    cam = ci * c2 + s1 * s2
    sap = s1 * c2 + ci * s2
    sam = s1 * c2 - ci * s2
    return 0.5 * sam * sam / (sap * sap) * (cam * cam + cap * cap) / (cam * cam)


@njit(cache=True)
def _chunk(seed, nph, mua, mus, g, n, nbin, dr, out):
    np.random.seed(seed)
    mut = mua + mus
    for _ in range(nph):
        x = y = z = 0.0
        ux = uy = 0.0
        uz = 1.0
        w = 1.0
        while True:
            s = -math.log(1.0 - np.random.random()) / mut
            if uz < 0.0:
                db = -z / uz
                if db < s:                                  # reaches the surface z = 0 (up is -z)
                    x += ux * db
                    y += uy * db
                    z = 0.0
                    if np.random.random() < _fresnel(n, 1.0, -uz):
                        uz = -uz                            # internal reflection (memoryless: redraw the step)
                        continue
                    ib = int(math.sqrt(x * x + y * y) / dr)
                    if ib < nbin:
                        out[ib] += w
                    break
            x += ux * s
            y += uy * s
            z += uz * s
            w -= w * mua / mut
            if g == 0.0:
                c = 2 * np.random.random() - 1
            else:
                t = (1 - g * g) / (1 - g + 2 * g * np.random.random())
                c = (1 + g * g - t * t) / (2 * g)
            st = math.sqrt(max(0.0, 1 - c * c))
            ph = 2 * math.pi * np.random.random()
            cp = math.cos(ph)
            sp = math.sin(ph)
            if abs(uz) > 0.99999:
                ux, uy, uz = st * cp, st * sp, (1.0 if uz > 0 else -1.0) * c
            else:
                tt = math.sqrt(1 - uz * uz)
                nux = st * (ux * uz * cp - uy * sp) / tt + ux * c
                nuy = st * (uy * uz * cp + ux * sp) / tt + uy * c
                nuz = -st * cp * tt + uz * c
                ux, uy, uz = nux, nuy, nuz
            if w < 1e-4:
                if np.random.random() < 0.1:
                    w *= 10.0
                else:
                    break


@njit(parallel=True, cache=True)
def _run(nph_total, nchunk, mua, mus, g, n, nbin, dr, seed):
    outs = np.zeros((nchunk, nbin))
    per = nph_total // nchunk
    for c in prange(nchunk):
        _chunk(seed + 977 * c, per, mua, mus, g, n, nbin, dr, outs[c])
    return outs.sum(0) / (per * nchunk)


def reference_radial_mc(mua, musp, g, n, n_photons, *, edges, seed=5, n_chunks=256):
    """Radial diffuse reflectance ``R(rho)`` [1/mm^2] per unit power ENTERING a semi-infinite medium (annulus averages
    over ``edges``, uniform spacing), plus the total diffuse reflectance, from the independent MCML-type code."""
    edges = np.asarray(edges, float)
    dr = float(edges[1] - edges[0])
    mus = musp / (1.0 - g)
    tot = _run(int(n_photons), int(n_chunks), float(mua), float(mus), float(g), float(n), len(edges) - 1, dr, int(seed))
    area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return tot / area, float(tot.sum())


def buried_sphere_image(cover_mm: float, cover_musp: float, *, n_exc: int, n_em: int, seed: int = 1,
                        radius_mm: float = 0.8, host_musp: float = 1.0, mua: float = 0.01, g: float = 0.9,
                        n_idx: float = 1.4, footprint_mm: float = 24.0, cell_mm: float = 1.0,
                        host_fraction: float = 0.03, voxel_mm: float = 0.1):
    """Fluorescence image of a fluorescent sphere buried under a turbid cover (transport level, no spectra).

    Block: substrate 6 mm (``host_musp``) + cover of thickness ``cover_mm`` (``cover_musp``), uniform wide beam.  A
    sphere of radius ``radius_mm`` is centred on the axis at ``y = 5.2`` (top at 5.2 + r = 6.0 mm: touching the
    cover).  Sources: dye in the sphere with absorbed-power yield 1, plus a weak host autofluorescence
    (``host_fraction`` of the sphere's per-volume rate, in the substrate only), both proportional to the local
    fluence.  Returns ``(image, contrast, info)``: the top-face radiance ``(nu, nv)`` (arbitrary units) and the Weber
    contrast ``(L_centre - L_ring)/L_ring`` of the central 1x1 cell against the cells 6-9 mm from the axis."""
    from spectraforge.studio import transport3d as T
    from spectraforge.studio.contracts import BlockGeometry, RegionOptics
    from spectraforge.studio.schema import Inclusion, ThicknessProfile, VolumeMCSettings

    fp = (footprint_mm, footprint_mm)
    layers = [ThicknessProfile(thickness_mm=6.0), ThicknessProfile(thickness_mm=cover_mm)]
    sph = Inclusion(name="bead", material_id="m", kind="sphere", center_mm=(0.0, 5.2, 0.0), radius_mm=radius_mm)
    G = BlockGeometry(0, fp, ["substrate", "cover"], [0, 1], T.planes_from_layers(fp, layers), np.eye(4),
                      ["bead"], [0], [sph])
    s = VolumeMCSettings(lateral_cell_mm=cell_mm, exit_cell_mm=cell_mm, sublayers=3, inclusion_voxel_mm=voxel_mm)
    mus = lambda musp: musp / (1.0 - g)
    opt = RegionOptics(500.0, np.array([mua, mua]), np.array([mus(host_musp), mus(cover_musp)]), np.array([g, g]),
                       np.array([n_idx, n_idx]), np.array([mua]), np.array([mus(host_musp)]), np.array([g]))
    launch = T.uniform_beam_launch(G, 500.0, 1.0, 100_000, seed=seed)
    exc = T.excitation_pass(G, opt, launch, [0.0, 1.0, 0.0], s, n_photons=n_exc, seed=seed)
    imap = exc.inclusion_map
    phi_i = exc.inclusion_fluence_mW_mm2.reshape(-1)[imap.occ_flat]
    isrc = np.zeros((1,) + imap.shape)
    isrc[0].reshape(-1)[imap.occ_flat] = phi_i * imap.occ_volume_mm3
    vol = T.host_voxel_volumes(G, exc.grid, imap)
    src = np.zeros((1,) + exc.grid.shape)
    src[0, :, :, 0, :] = host_fraction * exc.fluence_mW_mm2[:, :, 0, :] * vol[:, :, 0, :]
    em = T.emission_pass(G, [opt], src, ["x"], [0.0, 1.0, 0.0], s, n_photons=n_em, seed=seed + 1,
                         inclusion_sources=isrc)
    img = em.radiance["top"][:, :, 0, 0]
    e = em.faces["top"].u_edges_mm
    c = 0.5 * (e[1:] + e[:-1])
    X, Z = np.meshgrid(c, c, indexing="ij")
    rho = np.hypot(X, Z)
    centre = float(img[rho < 0.6 * cell_mm + 1e-9].mean()) if (rho < 0.6 * cell_mm + 1e-9).any() else float(img[rho == rho.min()].mean())
    ring = float(img[(rho > 6.0) & (rho < 9.0)].mean())
    return img, (centre - ring) / ring, {"exc": exc, "em": em, "rho": rho}
