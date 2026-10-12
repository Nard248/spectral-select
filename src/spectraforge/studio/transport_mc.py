"""Numba Monte Carlo engines behind :mod:`spectraforge.studio.transport`.

Two independent engines live here:

1. **White Monte Carlo bake** (``bake_arrays``) -- the engine that feeds the fast kernel tables.
   Photon packets random-walk in a plane-parallel slab in *units of the scattering mean free path*
   (``mu_s = 1``) with **no absorption**; every collision and every exit is histogrammed against
   the packet's *accumulated path length* ``s`` (in scattering mfp).  Absorption is applied
   afterwards, exactly, as the weight ``exp(-r s)`` with ``r = mu_a / mu_s``
   (Alerstam et al., "White Monte Carlo for time-resolved photon migration", J. Biomed. Opt. 13,
   041304, 2008).  One run therefore serves *every* absorption level; the tables depend only on
   (g, n, n_outside, cos(theta_internal), tau = mu_s d).

   Fluence from the collision estimator.  For one packet launched per unit entering power per
   area, the scattered power density in a depth bin is ``mu_s * Phi * dz``; the white packets
   scatter at every collision with weight ``exp(-r s)``, so the (dimensionless, per unit entering
   irradiance) fluence is ``f(z) = sum_collisions exp(-r s) / (N dz')``, ``z' = mu_s z``.  The
   estimator is unbiased (also for unscattered/ballistic light) and exactly separable in ``r``.

2. **Conventional MCML** (``mcml_slab``) -- the classical implicit-capture + Russian-roulette
   code (Wang, Jacques & Zheng 1995).  It is used ONLY for verification: the MCML benchmark,
   energy conservation, the brute-force forward emitter run that checks the optical-reciprocity
   normalisation, and ``reabsorbed_fraction``.

Geometry: slab ``0 < z < d`` (z down, into the medium), refractive index ``n`` inside and
``n_out`` outside on BOTH faces (the bottom face of a finite slab sees ``n_out`` too).  Boundaries
use unpolarised Fresnel reflection with total internal reflection, decided probabilistically
(packet weight is preserved).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from spectraforge.studio._numba_env import parallel_njit

# --------------------------------------------------------------------------------------------
# Shared grids (path-length bins etc.)
# --------------------------------------------------------------------------------------------
#: path-length bins (units of scattering mfp): bin 0 = [0, S0), bin k>=1 = [S0 R^(k-1), S0 R^k)
S0 = 1.0e-3
S_RATIO = 1.06
S_CAP = 2.0e4                      # packets longer than this are discarded (weight e^{-r_min S_CAP})
S_SPLIT = 100.0                    # beyond this path length: Russian roulette (p=1/2, w x2) at every
                                   # doubling of s -- cuts the cost of the long diffusive tail ~10x
N_S = 1 + int(math.ceil(math.log(S_CAP / S0) / math.log(S_RATIO)))
LN_R = math.log(S_RATIO)
N_COSBIN = 16                      # top-exit bins, uniform in cos^2(theta_exit,external)


def s_edges() -> np.ndarray:
    e = np.empty(N_S + 1)
    e[0] = 0.0
    e[1:] = S0 * S_RATIO ** np.arange(N_S)
    return e


# --------------------------------------------------------------------------------------------
# Elementary helpers
# --------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def fresnel_R(n1, n2, c1):
    """Unpolarised Fresnel reflectance for light in medium n1 hitting an n2 interface at
    cos(incidence) = c1 >= 0.  Returns 1 for total internal reflection."""
    if n1 == n2:
        return 0.0
    s2sq = (n1 / n2) ** 2 * (1.0 - c1 * c1)
    if s2sq >= 1.0:
        return 1.0
    c2 = math.sqrt(1.0 - s2sq)
    if c1 > 0.99999:
        r = (n1 - n2) / (n1 + n2)
        return r * r
    a = n1 * c1
    b = n2 * c2
    rs = ((a - b) / (a + b)) ** 2
    a2 = n2 * c1
    b2 = n1 * c2
    rp = ((a2 - b2) / (a2 + b2)) ** 2
    return 0.5 * (rs + rp)


@njit(cache=True, inline="always")
def _hg(ux, uy, uz, g):
    """Sample a Henyey-Greenstein scattering event; return the new direction cosines."""
    r = np.random.random()
    if g == 0.0:
        cost = 2.0 * r - 1.0
    else:
        tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * r)
        cost = (1.0 + g * g - tmp * tmp) / (2.0 * g)
    if cost > 1.0:
        cost = 1.0
    elif cost < -1.0:
        cost = -1.0
    sint = math.sqrt(1.0 - cost * cost)
    phi = 2.0 * math.pi * np.random.random()
    cp = math.cos(phi)
    sp = math.sin(phi)
    if abs(uz) > 0.99999:
        nux = sint * cp
        nuy = sint * sp
        nuz = cost if uz > 0 else -cost
    else:
        t = math.sqrt(1.0 - uz * uz)
        nux = sint * (ux * uz * cp - uy * sp) / t + ux * cost
        nuy = sint * (uy * uz * cp + ux * sp) / t + uy * cost
        nuz = -sint * cp * t + uz * cost
    return nux, nuy, nuz


@njit(cache=True, inline="always")
def _sbin(s):
    if s < S0:
        return 0
    k = int(math.log(s / S0) / LN_R) + 1
    return k if k < N_S else N_S - 1


# --------------------------------------------------------------------------------------------
# 1. White Monte Carlo
# --------------------------------------------------------------------------------------------
@njit(cache=True)
def _white_run(mu0, tau, n, no, g, n_ph, zed, nzb, rad_edges, nrb, do_rad, m, rtop, tt, rad):
    """One white-MC run: ``n_ph`` packets launched at the top face, internal cosine ``mu0``,
    normal-plane azimuth 0.  Accumulates raw counts into ``m`` (collisions, [depth bin, s bin]),
    ``rtop`` (top exits, [cos^2 bin, s bin]), ``tt`` (bottom exits, [s bin]) and ``rad`` (top exits
    vs radius, [r bin, s bin]; only if ``do_rad``)."""
    ux0 = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    for _ in range(n_ph):
        x = 0.0
        y = 0.0
        z = 0.0
        ux = ux0
        uy = 0.0
        uz = mu0
        s = 0.0
        w = 1.0
        s_next = S_SPLIT
        while True:
            step = -math.log(1.0 - np.random.random())
            if uz > 0.0:
                db = (tau - z) / uz
            elif uz < 0.0:
                db = -z / uz
            else:
                db = np.inf
            if step < db:
                x += ux * step
                y += uy * step
                z += uz * step
                s += step
                if s >= S_CAP:
                    break
                if s >= s_next:
                    if np.random.random() < 0.5:
                        w *= 2.0
                        s_next *= 2.0
                    else:
                        break
                zb = np.searchsorted(zed, z, side="right") - 1
                if zb < 0:
                    zb = 0
                elif zb >= nzb:
                    zb = nzb - 1
                m[zb, _sbin(s)] += w
                ux, uy, uz = _hg(ux, uy, uz, g)
            else:
                x += ux * db
                y += uy * db
                s += db
                if uz > 0.0:
                    z = tau
                else:
                    z = 0.0
                if s >= S_CAP:
                    break
                if s >= s_next:
                    if np.random.random() < 0.5:
                        w *= 2.0
                        s_next *= 2.0
                    else:
                        break
                if np.random.random() < fresnel_R(n, no, abs(uz)):
                    uz = -uz
                    continue
                sb = _sbin(s)
                if uz < 0.0:
                    mue2 = 1.0 - (n / no) ** 2 * (1.0 - uz * uz)
                    b = int(mue2 * N_COSBIN)
                    if b >= N_COSBIN:
                        b = N_COSBIN - 1
                    elif b < 0:
                        b = 0
                    rtop[b, sb] += w
                    if do_rad:
                        r = math.sqrt(x * x + y * y)
                        rb = np.searchsorted(rad_edges, r, side="right") - 1
                        if rb >= nrb:
                            rb = nrb - 1
                        elif rb < 0:
                            rb = 0
                        rad[rb, sb] += w
                else:
                    tt[sb] += w
                break


@njit(nogil=True, cache=True)
def _bake_task(task, mu_nodes, tau_nodes, n, no, g, n_base, seed, u_edges, zinf_edges, rad_edges,
               m_fin, m_inf, rtop_fin, rtop_inf, tt_fin, tt_inf, rad_fin, rad_inf):
    """Bake one (mu, tau) cell -- or the semi-infinite cell ``j == nt`` -- into the shared output
    arrays. The RNG stream is seeded from ``(seed, task)`` only, so the result is independent of the
    order in which / the thread on which tasks run."""
    nc = mu_nodes.shape[0]
    nt = tau_nodes.shape[0]
    nu = u_edges.shape[0] - 1
    nz_inf = zinf_edges.shape[0] - 1
    nrb = rad_edges.shape[0] - 1
    np.random.seed(seed * 1000003 + task)
    c = task // (nt + 1)
    j = task % (nt + 1)
    mu0 = mu_nodes[c]
    do_rad = (c == nc - 1)               # normal incidence runs also record the radial profile
    if j < nt:
        tau = tau_nodes[j]
        n_ph = int(n_base * min(64.0, max(1.0, 1.0 / tau)))
        zed = tau * u_edges
        m = np.zeros((nu, m_fin.shape[3]))
        rt = np.zeros((rtop_fin.shape[2], rtop_fin.shape[3]))
        tb = np.zeros(tt_fin.shape[2])
        rd = np.zeros((nrb, rad_fin.shape[2]))
        _white_run(mu0, tau, n, no, g, n_ph, zed, nu, rad_edges, nrb, do_rad, m, rt, tb, rd)
        m_fin[c, j] = m / n_ph
        rtop_fin[c, j] = rt / n_ph
        tt_fin[c, j] = tb / n_ph
        if do_rad:
            rad_fin[j] = rd / n_ph
    else:
        n_ph = n_base
        m = np.zeros((nz_inf, m_inf.shape[2]))
        rt = np.zeros((rtop_inf.shape[1], rtop_inf.shape[2]))
        tb = np.zeros(tt_inf.shape[1])
        rd = np.zeros((nrb, rad_inf.shape[1]))
        _white_run(mu0, np.inf, n, no, g, n_ph, zinf_edges, nz_inf, rad_edges, nrb, do_rad,
                   m, rt, tb, rd)
        m_inf[c] = m / n_ph
        rtop_inf[c] = rt / n_ph
        tt_inf[c] = tb / n_ph
        if do_rad:
            rad_inf[:] = rd / n_ph


def _task_order(nc: int, nt: int, n_base: int, tau_nodes: np.ndarray) -> np.ndarray:
    """Task ids sorted by estimated cost, longest first (LPT scheduling for the thread pool)."""
    ntask = nc * (nt + 1)
    j = np.arange(ntask) % (nt + 1)
    tau = np.concatenate([np.asarray(tau_nodes, float), [np.inf]])[j]
    n_ph = np.where(np.isfinite(tau), n_base * np.minimum(64.0, np.maximum(1.0, 1.0 / tau)), n_base)
    cost = n_ph * (1.0 + np.minimum(np.where(np.isfinite(tau), tau, 1e4) ** 2, 4000.0))
    return np.argsort(-cost, kind="stable").astype(np.int64)


def bake_arrays(g, n, n_out, mu_nodes, tau_nodes, u_edges, zinf_edges, rad_edges, n_photons, seed):
    """Run the white-MC bake; returns a dict of float32 arrays (masses per launched packet)."""
    mu_nodes = np.ascontiguousarray(mu_nodes, dtype=np.float64)
    tau_nodes = np.ascontiguousarray(tau_nodes, dtype=np.float64)
    nc, nt = len(mu_nodes), len(tau_nodes)
    nu, nz, nrb = len(u_edges) - 1, len(zinf_edges) - 1, len(rad_edges) - 1
    out = dict(
        m_fin=np.zeros((nc, nt, nu, N_S), np.float32),
        m_inf=np.zeros((nc, nz, N_S), np.float32),
        rtop_fin=np.zeros((nc, nt, N_COSBIN, N_S), np.float32),
        rtop_inf=np.zeros((nc, N_COSBIN, N_S), np.float32),
        tt_fin=np.zeros((nc, nt, N_S), np.float32),
        tt_inf=np.zeros((nc, N_S), np.float32),
        rad_fin=np.zeros((nt, nrb, N_S), np.float32),
        rad_inf=np.zeros((nrb, N_S), np.float32),
    )
    args = (mu_nodes, tau_nodes, float(n), float(n_out), float(g), int(n_photons), int(seed),
            np.ascontiguousarray(u_edges, np.float64), np.ascontiguousarray(zinf_edges, np.float64),
            np.ascontiguousarray(rad_edges, np.float64),
            out["m_fin"], out["m_inf"], out["rtop_fin"], out["rtop_inf"], out["tt_fin"],
            out["tt_inf"], out["rad_fin"], out["rad_inf"])
    order = _task_order(nc, nt, int(n_photons), tau_nodes)
    from concurrent.futures import ThreadPoolExecutor
    from numba import get_num_threads
    with ThreadPoolExecutor(max_workers=max(1, int(get_num_threads()))) as ex:
        # dynamic (work-stealing) schedule: heterogeneous task cost and P/E cores make a static
        # parallel-for split ~2x slower than the ideal
        list(ex.map(lambda t: _bake_task(int(t), *args), order))
    return out


# --------------------------------------------------------------------------------------------
# 2. Conventional MCML (verification engine)
# --------------------------------------------------------------------------------------------
@njit(cache=True)
def _mcml_chunk(mua, mus, g, n, no, d, mode, mu0_ext, z0, zrate, n_ph, zmax, nzb, res, hist, absz):
    """Implicit-capture + roulette MC for ``n_ph`` packets.

    mode 0: collimated beam from outside at external cosine ``mu0_ext`` (specular reflection is
            NOT simulated; the caller multiplies by 1 - Rsp);
    mode 1: isotropic point emitters at depth ``z0``;
    mode 2: isotropic emitters with depth drawn from the (truncated) exponential rate ``zrate``.
    res = [Rd, Tt, A_total]; hist = top-exit weight per cos^2 bin; absz = absorbed weight per z bin.
    """
    mut = mua + mus
    albedo_abs = mua / mut
    for _ in range(n_ph):
        x = 0.0
        y = 0.0
        if mode == 0:
            z = 0.0
            sin_e = math.sqrt(max(0.0, 1.0 - mu0_ext * mu0_ext))
            sin_i = sin_e * no / n
            ux = sin_i
            uy = 0.0
            uz = math.sqrt(max(0.0, 1.0 - sin_i * sin_i))
        else:
            if mode == 1:
                z = z0
            else:
                if math.isinf(d):
                    z = -math.log(1.0 - np.random.random()) / zrate
                else:
                    z = -math.log(1.0 - np.random.random() * (1.0 - math.exp(-zrate * d))) / zrate
            uz = 2.0 * np.random.random() - 1.0
            st = math.sqrt(max(0.0, 1.0 - uz * uz))
            ph = 2.0 * math.pi * np.random.random()
            ux = st * math.cos(ph)
            uy = st * math.sin(ph)
        w = 1.0
        while True:
            step = -math.log(1.0 - np.random.random()) / mut
            if uz > 0.0:
                db = (d - z) / uz
            elif uz < 0.0:
                db = -z / uz
            else:
                db = np.inf
            if step < db:
                x += ux * step
                y += uy * step
                z += uz * step
                dw = w * albedo_abs
                res[2] += dw
                zb = int(z / zmax * nzb)
                if zb >= nzb:
                    zb = nzb - 1
                elif zb < 0:
                    zb = 0
                absz[zb] += dw
                w -= dw
                ux, uy, uz = _hg(ux, uy, uz, g)
                if w < 1.0e-4:
                    if np.random.random() < 0.1:
                        w *= 10.0
                    else:
                        break
            else:
                z = d if uz > 0.0 else 0.0
                if np.random.random() < fresnel_R(n, no, abs(uz)):
                    uz = -uz
                    continue
                if uz < 0.0:
                    res[0] += w
                    mue2 = 1.0 - (n / no) ** 2 * (1.0 - uz * uz)
                    b = int(mue2 * N_COSBIN)
                    if b >= N_COSBIN:
                        b = N_COSBIN - 1
                    elif b < 0:
                        b = 0
                    hist[b] += w
                else:
                    res[1] += w
                break


@parallel_njit(cache=True)
def _mcml_all(mua, mus, g, n, no, d, mode, mu0_ext, z0, zrate, n_per, nchunks, seed, zmax, nzb,
              res, hist, absz):
    for c in prange(nchunks):
        np.random.seed(seed * 7919 + c)
        r = np.zeros(3)
        h = np.zeros(N_COSBIN)
        a = np.zeros(nzb)
        _mcml_chunk(mua, mus, g, n, no, d, mode, mu0_ext, z0, zrate, n_per, zmax, nzb, r, h, a)
        res[c] = r
        hist[c] = h
        absz[c] = a


def mcml_slab(mua, mus, g, n, n_out, d, *, mode="collimated", mu0_ext=1.0, z0=0.0, zrate=0.0,
              n_photons=200_000, seed=1, nz_bins=50, zmax=None):
    """Conventional MCML run.  Returns a dict with (per unit launched power, per packet):

    ``Rd`` diffuse reflectance (top exits, transmitted through the Fresnel boundary), ``Tt``
    transmittance (bottom), ``A`` absorbed fraction, ``Rsp`` specular reflection (collimated mode
    only; ``Rd``/``Tt``/``A`` are already multiplied by ``1 - Rsp``), ``hist_cos2`` top-exit weight
    per cos^2(external exit angle) bin, ``absz`` absorbed weight per depth bin and the bin edges.
    """
    modes = {"collimated": 0, "point": 1, "exp": 2}
    md = modes[mode]
    nchunks = 64
    n_per = max(1, int(math.ceil(n_photons / nchunks)))
    if zmax is None:
        zmax = d if np.isfinite(d) else 10.0 / max(mua + mus * (1 - g), 1e-12)
    res = np.zeros((nchunks, 3))
    hist = np.zeros((nchunks, N_COSBIN))
    absz = np.zeros((nchunks, nz_bins))
    _mcml_all(float(mua), float(mus), float(g), float(n), float(n_out), float(d), md,
              float(mu0_ext), float(z0), float(zrate), n_per, nchunks, int(seed), float(zmax),
              int(nz_bins), res, hist, absz)
    tot = n_per * nchunks
    scale = 1.0 / tot
    rsp = float(fresnel_R(n_out, n, mu0_ext)) if md == 0 else 0.0
    f = scale * (1.0 - rsp)
    out = dict(Rd=res[:, 0].sum() * f, Tt=res[:, 1].sum() * f, A=res[:, 2].sum() * f, Rsp=rsp,
               hist_cos2=hist.sum(0) * f, absz=absz.sum(0) * f,
               z_edges=np.linspace(0.0, zmax, nz_bins + 1), n_photons=tot)
    return out


# --------------------------------------------------------------------------------------------
# 3. Fast pixel interpolation of the baked kernel tables
# --------------------------------------------------------------------------------------------
@njit(cache=True, error_model="numpy")
def _fres(n1, n2, c1):
    """Unpolarised Fresnel reflectance (TIR / non-finite -> 1); scalar twin of ``transport.fresnel_reflectance``."""
    c1 = min(max(c1, 0.0), 1.0)
    s2 = (n1 / n2) ** 2 * (1.0 - c1 * c1)
    if s2 >= 1.0:
        return 1.0
    c2 = math.sqrt(min(max(1.0 - s2, 0.0), 1.0))
    rs = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)) ** 2
    rp = ((n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)) ** 2
    r = 0.5 * (rs + rp)
    if not math.isfinite(r):
        return 1.0
    return r


@njit(cache=True)
def _lin_idx(v, nodes):
    """Lower index and fractional weight of ``v`` on an ascending node axis (clamped)."""
    i = np.searchsorted(nodes, v, side="right") - 1
    i = min(max(i, 0), nodes.shape[0] - 2)
    w = min(max((v - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0), 1.0)
    return i, w


@parallel_njit(cache=True, error_model="numpy")
def turbid_pixels(cos_in, cos_out, thick, no, n, mu_nodes, d_nodes, lnj, w_idx, w_wgt, em_sel,
                  ln_zbar, rho_tab, out_G, out_rho, out_zbar):
    """Fused per-pixel turbid-medium kernel evaluation (numba twin of the NumPy pipeline
    ``refract -> Fresnel -> axis lookup -> tri-linear log interpolation`` in ``transport.py``).

    ``lnj`` (Nc, Nc, K+1, Nn) holds ``ln(mu_i mu_o J)`` at the 4 nm emission *nodes*; the fine-grid
    columns ``em_sel`` (indices into the optics grid, possibly a subset) are obtained by linear
    interpolation in ``ln J`` between the two surrounding nodes (``w_idx``/``w_wgt``), i.e. the same
    log-linear interpolation as a pre-expanded fine table, at a quarter of the memory."""
    P = cos_in.shape[0]
    nc = lnj.shape[0]
    nb = lnj.shape[1]
    nk = lnj.shape[2]
    nn = lnj.shape[3]
    nsel = em_sel.shape[0]
    K = d_nodes.shape[0]
    nbin = rho_tab.shape[2]
    ln_d = np.log(d_nodes)
    r2n = (no / n) ** 2
    for p in prange(P):
        cin = cos_in[p]
        cout = cos_out[p]
        th_raw = thick[p]
        th = np.inf if math.isnan(th_raw) else th_raw
        s2i = r2n * (1.0 - cin * cin)
        ok_i = (cin > 0.0) and (s2i < 1.0)
        ci = math.sqrt(min(max(1.0 - s2i, 0.0), 1.0))
        s2o = r2n * (1.0 - cout * cout)
        ok_o = (cout > 0.0) and (s2o < 1.0)
        co = math.sqrt(min(max(1.0 - s2o, 0.0), 1.0))
        t_in = 1.0 - _fres(no, n, cin)
        t_out = 1.0 - _fres(n, no, co)
        ok = ok_i and ok_o and (not (th_raw <= 0.0)) and (t_in > 0.0) and (t_out > 0.0)
        ok_r = ok_i and (not (th_raw <= 0.0)) and (cout == cout)
        if not (ok or ok_r):
            for j in range(nsel):
                out_G[p, j] = 0.0
            out_rho[p] = 0.0
            out_zbar[p] = 0.0
            continue
        ia, fa = _lin_idx(min(max(ci, mu_nodes[0]), 1.0), mu_nodes)
        ib, fb = _lin_idx(min(max(co, mu_nodes[0]), 1.0), mu_nodes)
        # thickness axis (K finite nodes + semi-infinite node K)
        shift = 0.0
        if not (th < d_nodes[K - 1]):
            kk = K
            fk = 0.0
        else:
            tt = min(max(th, d_nodes[0]), d_nodes[K - 1])
            kk = np.searchsorted(d_nodes, tt, side="right") - 1
            kk = min(max(kk, 0), K - 2)
            fk = min(max((math.log(tt) - ln_d[kk]) / (ln_d[kk + 1] - ln_d[kk]), 0.0), 1.0)
            if th < d_nodes[0]:
                shift = math.log(max(th, 1e-300) / d_nodes[0])
        a1 = min(ia + 1, nc - 1)
        b1 = min(ib + 1, nb - 1)
        k1 = min(kk + 1, nk - 1)
        c000 = (1 - fa) * (1 - fb) * (1 - fk)
        c001 = (1 - fa) * (1 - fb) * fk
        c010 = (1 - fa) * fb * (1 - fk)
        c011 = (1 - fa) * fb * fk
        c100 = fa * (1 - fb) * (1 - fk)
        c101 = fa * (1 - fb) * fk
        c110 = fa * fb * (1 - fk)
        c111 = fa * fb * fk
        if ok:
            pref = t_in * t_out / (4.0 * math.pi * n * n * max(ci * co, 1e-9))
            tmp = np.empty(nn)
            for q in range(nn):
                tmp[q] = (c000 * lnj[ia, ib, kk, q] + c001 * lnj[ia, ib, k1, q]
                          + c010 * lnj[ia, b1, kk, q] + c011 * lnj[ia, b1, k1, q]
                          + c100 * lnj[a1, ib, kk, q] + c101 * lnj[a1, ib, k1, q]
                          + c110 * lnj[a1, b1, kk, q] + c111 * lnj[a1, b1, k1, q])
            for j in range(nsel):
                w = em_sel[j]
                i0 = w_idx[w]
                t = w_wgt[w]
                out_G[p, j] = pref * math.exp((1.0 - t) * tmp[i0] + t * tmp[i0 + 1] + shift)
            zb = (c000 * ln_zbar[ia, ib, kk] + c001 * ln_zbar[ia, ib, k1]
                  + c010 * ln_zbar[ia, b1, kk] + c011 * ln_zbar[ia, b1, k1]
                  + c100 * ln_zbar[a1, ib, kk] + c101 * ln_zbar[a1, ib, k1]
                  + c110 * ln_zbar[a1, b1, kk] + c111 * ln_zbar[a1, b1, k1])
            out_zbar[p] = math.exp(zb + shift)
        else:
            for j in range(nsel):
                out_G[p, j] = 0.0
            out_zbar[p] = 0.0
        # elastic remission: (mu_i', d, cos^2 of the external view angle)
        if ok_r:
            m2 = min(max(cout * cout, 0.0), 1.0) * nbin - 0.5
            b0 = int(min(max(math.floor(m2), 0.0), nbin - 2.0))
            wbn = min(max(m2 - b0, 0.0), 1.0)
            v = 0.0
            for ai in range(2):
                aa = ia if ai == 0 else a1
                fa_ = (1 - fa) if ai == 0 else fa
                for ki in range(2):
                    kc = kk if ki == 0 else k1
                    fk_ = (1 - fk) if ki == 0 else fk
                    for bi in range(2):
                        bb = b0 if bi == 0 else b0 + 1
                        fb_ = (1 - wbn) if bi == 0 else wbn
                        v += fa_ * fk_ * fb_ * rho_tab[aa, kc, bb]
            out_rho[p] = t_in * math.exp(v + shift)
        else:
            out_rho[p] = 0.0


@parallel_njit(cache=True)
def clear_G(mut_em, mut_ex, ci_p, co_p, length, pref, out, mean_depth, sel):
    """Closed-form clear-medium kernel ``pref * L * (1 - e^{-x})/x`` with
    ``x = (mut_ex co'/ci' + mut_em) L`` for all pixels/wavelengths, plus the G-weighted mean depth
    of origin (``<z>`` of ``exp(-a z)`` on ``[0, L co']`` with ``a = mut_ex/ci' + mut_em/co'``).
    ``sel`` lists the grid indices to evaluate (output column j <-> ``mut_em[sel[j]]``); the mean
    depth is G-weighted over those columns."""
    nw = sel.shape[0]
    for p in prange(ci_p.shape[0]):
        if pref[p] <= 0.0 or length[p] <= 0.0:
            for w in range(nw):
                out[p, w] = 0.0
            mean_depth[p] = 0.0
            continue
        L = length[p]
        co = co_p[p]
        kap0 = mut_ex * co / ci_p[p]
        az0 = mut_ex / ci_p[p]
        zl = L * co
        gs = 0.0
        gz = 0.0
        for w in range(nw):
            x = (kap0 + mut_em[sel[w]]) * L
            if x < 1.0e-9:
                ratio = 1.0
                zb = 0.5 * zl
            else:
                ratio = -math.expm1(-x) / x
                a = az0 + mut_em[sel[w]] / co
                if x > 600.0:
                    zb = 1.0 / a
                else:
                    zb = 1.0 / a - zl * math.exp(-x) / (-math.expm1(-x))
            g = pref[p] * L * ratio
            out[p, w] = g
            gs += g
            gz += g * zb
        mean_depth[p] = gz / gs if gs > 0.0 else 0.0
