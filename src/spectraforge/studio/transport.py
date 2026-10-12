"""Light transport inside uniform media for SpectralForge Studio.

Public API (see :mod:`spectraforge.studio.contracts` for the radiometric definitions)::

    clear_medium_kernel(optics, ex_nm, cos_in, cos_out, thickness_mm, n_outside=1.0, path_mm=None)
    get_transport_model(n_photons=200_000, seed=0, cache_dir=None) -> MCTransportModel
    MCTransportModel.kernels(optics, ex_nm, cos_in, cos_out, thickness_mm, n_outside=1.0, path_mm=None,
                             em_index=None)
    MCTransportModel.reabsorbed_fraction(optics, em_nm, ex_nm=..., ...)
    warmup(presets=None, *, jit=True, bake=True, gtables=True, progress=None, cache_dir=None)
    start_warmup(**kw) -> Thread ; warmup_status() -> dict          # server start-up hook

Physics and normalisations (derived once here, used everywhere)
================================================================
Slab ``0 < z < d`` (z into the medium), index ``n`` inside, ``n_o`` outside.  ``E`` is the
excitation irradiance on the surface (mW/mm^2, includes the incidence cosine, before Fresnel).

*Excitation leg.*  Let ``f(z; mu', lam)`` be the fluence rate (mW/mm^2) at depth z per unit power
per area *entering* the medium, for a collimated beam with internal cosine ``mu'`` (clear medium:
``f = exp(-mu_t z/mu')/mu'`` because the beam cross-section shrinks by ``mu'``).  Fluence per unit
irradiance is ``E T_in f`` with ``T_in`` the entry Fresnel transmittance.

*Emission leg.*  Isotropic emitters with source density ``S(z)`` (mW/mm^3) radiate ``S/4 pi``
per sr.  The radiance leaving the top face towards external cosine ``mu_o`` is ``S dz eta(z)``
with (clear medium)
``eta = T_out exp(-mu_t,em z/mu_o') / (4 pi n^2 mu_o')`` -- the ``1/n^2`` is the basic-radiance
(etendue) factor ``L/n^2`` = const, quoted relative to air (so it telescopes across glass layers).

*Optical reciprocity (Helmholtz).*  The adjoint of "isotropic point source at z, detector =
radiance leaving at ``mu_o``" is "collimated beam entering at ``mu_o``, fluence at z".  Comparing
with the clear limit fixes the constant, and the identity is exact for any scattering, internal
reflection or absorption::

        eta(z; mu_o, lam_em) = T_out f(z; mu_o', lam_em) / (4 pi n^2)           (R)

(``T_out`` = Fresnel transmittance at the exit angle, equal to ``T_in`` at the conjugate angle).
So ONE family of collimated-incidence fluence profiles gives both legs and

    G(lam_em) = T_in T_out / (4 pi n^2)  *  J,    J = integral_0^d f_ex(z) f_em(z) dz     [mm]  (G)

which reproduces the contracts.py closed form for ``mu_s -> 0`` (``J = (1-e^{-kL})/(kappa) /
(mu_i' mu_o')``-type integral, verified in the tests).  Reciprocity (R) is verified numerically
against a brute-force forward Monte Carlo that launches isotropic emitters (tests).

Elastic diffuse remission ``rho(theta_i, theta_o)``: of the power ``E A T_in`` entering, the
fraction exiting the top face in the external cosine bin ``[mu, mu+dmu]`` (azimuth-averaged) is
``dP``.  Radiance (relative to air) = ``dP / (pi d(mu^2) n_o^2)`` per unit entering power, so
``rho = T_in * that`` (1/sr).  A Lambertian remitter of total reflectance R gives ``R/pi``.

How it is made fast (bake + cache)
===================================
*White Monte Carlo* (:mod:`transport_mc`): photons walk with ``mu_s = 1`` and no absorption; path
length ``s`` is histogrammed and absorption enters afterwards as the weight ``exp(-(mu_a/mu_s) s)``.
Tables depend only on ``(g, n, n_o)`` and the dimensionless variables ``(mu', tau = mu_s d,
depth, s)``, so they are baked once per (g, n, n_o) -- not per material, wavelength or dye
concentration -- and cached on disk.  Per (material, excitation) a small table
``J(mu_i', mu_o', d, lam_em)`` is assembled in ~50 ms (matrix products over ``s``) on the 4 nm emission
nodes and cached on the (shared) model; per pixel ``kernels()`` is then one fused parallel numba kernel
(refraction + Fresnel + tri-linear log-interpolation + node->grid ``ln J`` lerp + elastic rho + mean
depth), ~30 ms for 90k pixels x 651 wavelengths and proportionally less with ``em_index``.
The table assembly, baked tables and radial profiles are shared across preview / capture / explain
calls (see :func:`get_transport_model`); :func:`warmup` pre-pays JIT compilation and the bakes.

Approximations (documented, all tested)
  * HG phase function with a single ``g`` per material (not wavelength dependent).
  * ``r = mu_a/mu_s`` is floored at 5e-4 (paths are truncated at 2e4 mfp; weight e^-10).
  * The bottom face of a finite slab borders ``n_outside`` (e.g. air) as well.
  * Elastic ``rho`` is azimuth-averaged; emission wavelengths are computed on a 4 nm node grid
    and interpolated (log-linear) to the 1 nm internal grid.
  * Below ``tau = 0.02`` the thin-slab (single-scattering) profile is used with absorption
    ``mu_a d`` preserved, so the kernel tends smoothly to the clear closed form.
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import transport_mc as mc
from .contracts import MaterialOptics, TransportKernels

ENGINE_VERSION = "transport-1.0"

# ---- table grids -------------------------------------------------------------------------
MU_NODES = np.array([0.04, 0.12, 0.22, 0.34, 0.48, 0.62, 0.76, 0.90, 1.0])   # internal cosine
TAU_NODES = 0.02 * 2.0 ** (np.arange(35) / 2.0)          # 0.02 .. ~2600 scattering mfp
TAU_CUT = 700.0                                          # finite d-nodes up to here; above: infinite
N_U = 40
U_EDGES = np.sin(np.pi * np.arange(N_U + 1) / (2 * N_U)) ** 2         # fraction of thickness
ZINF_EDGES = np.concatenate([[0.0], 0.02 * 1.3 ** np.arange(54)])     # semi-infinite depth bins
RAD_EDGES = ZINF_EDGES                                                # radial bins (mfp)
R_MIN = 5.0e-4
EM_NODE_STEP_NM = 4.0
N_RADIAL = 48
_TAU0 = float(TAU_NODES[0])
_TAU_MAX = float(TAU_NODES[-1])
_NB = mc.N_COSBIN


def default_cache_dir() -> Path:
    env = os.environ.get("SPECTRALFORGE_MC_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "spectralforge" / "mc"


# ============================================================================================
# Fresnel helpers (own implementation; does not depend on surface.py)
# ============================================================================================
def fresnel_reflectance(n1, n2, c1):
    """Unpolarised Fresnel reflectance, light in medium ``n1`` hitting ``n2`` with incidence cosine
    ``c1`` (>=0).  Vectorised; total internal reflection -> 1."""
    n1, n2, c1 = np.broadcast_arrays(np.asarray(n1, float), np.asarray(n2, float),
                                     np.clip(np.asarray(c1, float), 0.0, 1.0))
    s2 = (n1 / n2) ** 2 * (1.0 - c1 * c1)
    tir = s2 >= 1.0
    c2 = np.sqrt(np.clip(1.0 - s2, 0.0, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)) ** 2
        rp = ((n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)) ** 2
    r = 0.5 * (rs + rp)
    r = np.where(np.isfinite(r), r, 1.0)
    return np.where(tir, 1.0, r)


def _refract_cos(cos_ext, n_from, n_to):
    """Cosine of the refracted ray (Snell) and a validity mask (cos>0 and no TIR)."""
    c = np.asarray(cos_ext, float)
    s2 = (n_from / n_to) ** 2 * (1.0 - c * c)
    ok = (c > 0.0) & (s2 < 1.0)
    return np.sqrt(np.clip(1.0 - s2, 0.0, 1.0)), ok


# ============================================================================================
# 1. Clear media -- exact closed form
# ============================================================================================
def _em_selection(optics: MaterialOptics, em_index) -> np.ndarray:
    """Validated int64 index array into ``optics.wavelengths_nm`` (``None`` = all wavelengths)."""
    if em_index is None:
        return np.arange(len(optics.wavelengths_nm), dtype=np.int64)
    sel = np.ascontiguousarray(np.asarray(em_index, dtype=np.int64).ravel())
    if sel.size and (sel.min() < 0 or sel.max() >= len(optics.wavelengths_nm)):
        raise IndexError("em_index outside the optics wavelength grid")
    return sel


def clear_medium_kernel(optics: MaterialOptics, ex_nm: float, cos_in, cos_out, thickness_mm,
                        n_outside=1.0, path_mm=None, em_index=None) -> TransportKernels:
    """Closed-form transport of a non-scattering medium (contracts.py)::

        G = T_in T_out / (4 pi n^2 cos(th_i')) * (1 - exp(-kappa L)) / kappa,
        kappa = mu_t,ex cos(th_o')/cos(th_i') + mu_t,em

    ``cos_in`` / ``cos_out`` are EXTERNAL cosines (refracted here with ``n_outside`` ->
    ``optics.n``); ``T_in`` is the entry Fresnel transmittance (outside -> inside, excitation
    angle) and ``T_out`` the exit Fresnel transmittance (inside -> outside at the view angle; 0 on
    TIR).  ``L = path_mm`` or ``thickness / cos(th_o')``.  Elastic remission is 0.  Returns G on
    ``optics.wavelengths_nm`` (shape (P, W), float32).  ``(1-e^{-x})/x`` is evaluated with expm1.

    ``em_index`` (optional int array) restricts the output to those columns of the optics grid,
    giving ``G`` of shape (P, len(em_index)) with identical values; ``mean_depth_mm`` is then the
    G-weighted depth over the selected columns.
    """
    sel = _em_selection(optics, em_index)
    cos_in, cos_out, thick, n_o = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(a, float)) for a in (cos_in, cos_out, thickness_mm, n_outside)))
    n = float(optics.n)
    ci_p, ok_i = _refract_cos(cos_in, n_o, n)
    co_p, ok_o = _refract_cos(cos_out, n_o, n)
    t_in = 1.0 - fresnel_reflectance(n_o, n, cos_in)
    t_out = 1.0 - fresnel_reflectance(n, n_o, co_p)
    if path_mm is None:
        length = thick / np.maximum(co_p, 1e-12)
    else:
        length = np.broadcast_to(np.atleast_1d(np.asarray(path_mm, float)), cos_in.shape)
    length = np.where(np.isnan(length), 0.0, np.minimum(length, 1e9))
    ok = ok_i & ok_o & (length > 0.0) & np.isfinite(t_in)
    pref = np.where(ok, t_in * t_out / (4.0 * np.pi * n * n * np.maximum(ci_p, 1e-12)), 0.0)
    length = np.where(ok, length, 0.0)

    mut_ex = float(optics.interp(optics.mut_per_mm, ex_nm))
    G = np.empty((cos_in.shape[0], sel.size), np.float32)
    mean_depth = np.empty(cos_in.shape[0])
    mc.clear_G(np.ascontiguousarray(optics.mut_per_mm, dtype=np.float64), mut_ex,
               np.ascontiguousarray(np.maximum(ci_p, 1e-12)), np.ascontiguousarray(np.maximum(co_p, 1e-12)),
               np.ascontiguousarray(length), np.ascontiguousarray(pref), G, mean_depth, sel)
    return TransportKernels(fluorescence_G=G, elastic_rho=np.zeros(cos_in.shape), mean_depth_mm=mean_depth)


# ============================================================================================
# 2. Baked white-MC tables
# ============================================================================================
@dataclass
class BakedTables:
    """White-MC tables for one (g, n, n_out); all masses are per launched packet."""

    g: float
    n: float
    n_out: float
    arrays: dict

    def __getitem__(self, k):
        return self.arrays[k]


def _bake_key(g, n, n_out, n_photons, seed) -> str:
    h = hashlib.sha1()
    h.update(f"{ENGINE_VERSION}|{g:.4f}|{n:.4f}|{n_out:.4f}|{n_photons}|{seed}|".encode())
    for a in (MU_NODES, TAU_NODES, U_EDGES, ZINF_EDGES, mc.s_edges()):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:20]


def bake_tables(g: float, n: float, n_out: float, n_photons: int = 200_000, seed: int = 0,
                cache_dir: Path | None = None) -> BakedTables:
    """Bake (or load from ``cache_dir``) the white-MC tables.

    ``n_photons`` is the quality knob: each (angle, thickness) cell launches ``n_photons // 8``
    packets (x up to 64 for optically thin cells, so every cell sees ~n_photons/8 collisions).  Default ~25k per cell -> ~300 cells, a few
    seconds on a laptop.  Deterministic for a given seed (per-cell RNG streams)."""
    g, n, n_out = round(float(g), 4), round(float(n), 4), round(float(n_out), 4)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    key = _bake_key(g, n, n_out, n_photons, seed)
    path = cache_dir / f"tables_{key}.npz"
    if path.exists():
        try:
            with np.load(path) as z:
                return BakedTables(g, n, n_out, {k: z[k] for k in z.files})
        except Exception:                                  # corrupt cache -> rebake
            path.unlink(missing_ok=True)
    arrays = mc.bake_arrays(g, n, n_out, MU_NODES, TAU_NODES, U_EDGES, ZINF_EDGES, RAD_EDGES,
                            max(1, int(n_photons) // 8), seed)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except OSError:
        pass                                               # read-only cache dir: keep in memory
    return BakedTables(g, n, n_out, arrays)


def _bin_weights(r: np.ndarray) -> np.ndarray:
    """(N_S, len(r)) bin-averaged absorption weights ``<exp(-r s)>`` over each path-length bin."""
    e = mc.s_edges()
    a = e[:-1, None]
    b = e[1:, None]
    rr = np.asarray(r, float)[None, :]
    x = rr * (b - a)
    xs = np.where(x < 1e-8, 1.0, x)
    f = np.where(x < 1e-8, 1.0 - 0.5 * x, -np.expm1(-xs) / xs)
    w = np.exp(-rr * a) * f
    w[w < 1e-200] = 0.0                       # avoid denormals (BLAS flags them)
    return w


# ============================================================================================
# 3. Per-(material, excitation) kernel table
# ============================================================================================
@dataclass
class GTable:
    d_nodes: np.ndarray          # (K,) finite thickness nodes, mm
    ln_j: np.ndarray             # (Nc, Nc, K+1, Nn) float32, ln(mu_i mu_o J) at the Nn emission nodes;
                                 # index K = semi-infinite. Fine-grid columns are interpolated in ln J.
    w_idx: np.ndarray            # (W,) int64 lower node of every fine-grid wavelength
    w_wgt: np.ndarray            # (W,) float64 fractional position between nodes w_idx and w_idx+1
    ln_zbar: np.ndarray          # (Nc, Nc, K+1) ln mean origin depth, mm
    rho: np.ndarray              # (Nc, K+1, NB) ln of unit-power remission radiance (1/sr)
    mus_ex: float
    mua_ex: float


def _em_nodes(wl: np.ndarray) -> np.ndarray:
    nodes = np.arange(wl[0], wl[-1], EM_NODE_STEP_NM)
    return np.unique(np.concatenate([nodes, [wl[-1]]]))


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product with the spurious Accelerate-BLAS FP flags silenced (result checked finite)."""
    with np.errstate(all="ignore"):
        out = np.matmul(a, b)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite value in transport table assembly")
    return out


def _tau_weights(tau: np.ndarray):
    """Interpolation index/weight on the baked tau axis (log-linear), tau clamped to the nodes."""
    te = np.clip(tau, _TAU0, _TAU_MAX)
    jl = np.clip(np.searchsorted(TAU_NODES, te, side="right") - 1, 0, len(TAU_NODES) - 2)
    ln_tau = np.log(TAU_NODES)
    w = np.clip((np.log(te) - ln_tau[jl]) / (ln_tau[jl + 1] - ln_tau[jl]), 0.0, 1.0)
    return jl, w


def _build_gtable(bt: BakedTables, optics: MaterialOptics, ex_nm: float) -> GTable:
    """Assemble J(mu_i', mu_o', d, lam_em), the mean origin depth and the elastic table for one
    (material, excitation).  ``d`` nodes are geometric (factor sqrt 2) from the optically thin
    limit up to ``tau = TAU_CUT`` plus a semi-infinite node.  Profiles at an arbitrary
    ``tau = mu_s d`` are log-interpolated between the baked tau nodes; below ``tau_0`` the thin-slab
    profile is used with the absorption ``mu_a d`` preserved (``r_eff = r tau / tau_0``)."""
    wl = optics.wavelengths_nm
    mua_ex = max(float(optics.interp(optics.mua_per_mm, ex_nm)), 0.0)
    mus_ex = float(optics.interp(optics.mus_per_mm, ex_nm))
    nodes = _em_nodes(wl)
    mua_n = np.maximum(np.interp(nodes, wl, optics.mua_per_mm), 0.0)
    mus_n = np.maximum(np.interp(nodes, wl, optics.mus_per_mm), 1e-12)
    nn = len(nodes)
    nc = len(MU_NODES)
    nt = len(TAU_NODES)
    # index 0 = excitation, 1.. = emission nodes (all processed identically)
    mua_a = np.concatenate([[mua_ex], mua_n])
    mus_a = np.concatenate([[mus_ex], mus_n])
    r_a = np.maximum(mua_a / mus_a, R_MIN)
    na = nn + 1
    mut_max = float((mua_a + mus_a).max())

    d_hi = TAU_CUT / mus_ex
    d_lo = min(_TAU0 / mus_ex, 0.02 / mut_max)
    K = 1 + int(np.ceil(2.0 * np.log2(d_hi / d_lo)))
    d_nodes = np.sort(d_hi * 2.0 ** (-0.5 * np.arange(K)))            # ascending, last = d_hi
    du = np.diff(U_EDGES)
    uc = 0.5 * (U_EDGES[:-1] + U_EDGES[1:])

    mf = bt["m_fin"].astype(np.float64)                        # (Nc, Nt, Nu, Ns)
    mi = bt["m_inf"].astype(np.float64)                        # (Nc, Nz, Ns)
    e_all = _bin_weights(r_a)                                  # (Ns, na)
    p_all = _dot(mf.reshape(-1, mf.shape[-1]), e_all).reshape(nc, nt, N_U, na)
    dens_all = p_all / (TAU_NODES[None, :, None, None] * du[None, None, :, None])
    dens_all = np.ascontiguousarray(dens_all.transpose(0, 3, 1, 2))   # (Nc, na, Nt, Nu)
    ar = np.arange(na)

    # elastic remission per unit entering power: [ci, tau node, cos^2 bin] (ex weights only)
    rt_all = _dot(bt["rtop_fin"].astype(np.float64).reshape(-1, mc.N_S), e_all[:, :1]).reshape(nc, nt, _NB)
    rti = _dot(bt["rtop_inf"].astype(np.float64).reshape(-1, mc.N_S), e_all[:, :1]).reshape(nc, _NB)
    rho_unit = 1.0 / (np.pi * (1.0 / _NB) * bt.n_out ** 2)

    J = np.zeros((nc, nc, K + 1, nn))
    Jz = np.zeros((nc, nc, K + 1, nn))
    rho = np.zeros((nc, K + 1, _NB))
    for ik in range(K):
        d = d_nodes[ik]
        tau = mus_a * d
        thin = tau < _TAU0
        jl, w = _tau_weights(tau)
        dens = ((1 - w)[None, :, None] * dens_all[:, ar, jl, :]
                + w[None, :, None] * dens_all[:, ar, jl + 1, :])               # (Nc, na, Nu)
        if thin.any():                                       # thin-slab limit, mu_a d preserved
            idx = np.where(thin)[0]
            e_thin = _bin_weights(r_a[idx] * tau[idx] / _TAU0)
            p_thin = np.einsum("cus,sn->cnu", mf[:, 0], e_thin, optimize=True)
            dens[:, idx, :] = p_thin / (_TAU0 * du[None, None, :])
        ex = dens[:, 0, :]                                    # (Nc_i, Nu)
        em = dens[:, 1:, :]                                   # (Nc_o, Nn, Nu)
        J[:, :, ik, :] = d * np.einsum("iu,onu,u->ion", ex, em, du, optimize=True)
        Jz[:, :, ik, :] = d * np.einsum("iu,onu,u->ion", ex, em, du * d * uc, optimize=True)
        # elastic remission at this thickness
        if thin[0]:
            r_eff = r_a[0] * tau[0] / _TAU0
            rho[:, ik, :] = (np.einsum("cbs,s->cb", bt["rtop_fin"][:, 0].astype(np.float64),
                                       _bin_weights(np.array([r_eff]))[:, 0]) * (tau[0] / _TAU0))
        else:
            rho[:, ik, :] = (1 - w[0]) * rt_all[:, jl[0], :] + w[0] * rt_all[:, jl[0] + 1, :]
    rho[:, K, :] = rti
    rho = rho * rho_unit

    # semi-infinite node: cumulative emission mass interpolated onto the excitation depth grid
    dzp = np.diff(ZINF_EDGES)
    dens_ex_inf = _dot(mi.reshape(-1, mc.N_S), e_all[:, :1]).reshape(nc, -1) / dzp[None, :]
    p_inf = _dot(mi.reshape(-1, mc.N_S), e_all[:, 1:]).reshape(nc, -1, nn)  # mass per bin (z' units)
    cum = np.concatenate([np.zeros((nc, 1, nn)), np.cumsum(p_inf, axis=1)], axis=1)
    z_phys = ZINF_EDGES / mus_ex                               # (Nz+1,) mm
    zc_phys = 0.5 * (z_phys[:-1] + z_phys[1:])
    for o in range(nc):
        for n_ in range(nn):
            q = z_phys * mus_n[n_]
            cq = np.interp(q, ZINF_EDGES, cum[o, :, n_])
            mass_em = np.diff(cq) / mus_n[n_]                  # int f_em dz over each ex bin (mm)
            J[:, o, K, n_] = np.einsum("iz,z->i", dens_ex_inf, mass_em)
            Jz[:, o, K, n_] = np.einsum("iz,z->i", dens_ex_inf, mass_em * zc_phys)

    mu_i = MU_NODES[:, None, None, None]
    mu_o = MU_NODES[None, :, None, None]
    ln_nodes = np.log(np.maximum(J * mu_i * mu_o, 1e-30)).astype(np.float32)
    zbar = np.log(np.maximum(Jz.sum(-1) / np.maximum(J.sum(-1), 1e-300), 1e-12))
    # nodes -> fine grid: linear in ln J, evaluated inside the pixel kernel (mc.turbid_pixels)
    idx = np.clip(np.searchsorted(nodes, wl, side="right") - 1, 0, nn - 2)
    wgt = np.clip((wl - nodes[idx]) / (nodes[idx + 1] - nodes[idx]), 0.0, 1.0)
    return GTable(d_nodes=d_nodes, ln_j=np.ascontiguousarray(ln_nodes), w_idx=idx.astype(np.int64),
                  w_wgt=wgt, ln_zbar=np.ascontiguousarray(zbar), rho=np.ascontiguousarray(np.log(np.maximum(rho, 1e-30))),
                  mus_ex=mus_ex, mua_ex=mua_ex)


# ============================================================================================
# 5. The model
# ============================================================================================
def _optics_key(optics: MaterialOptics) -> tuple:
    wl = optics.wavelengths_nm
    return (optics.cache_key(), len(wl), round(float(wl[0]), 6), round(float(wl[-1]), 6))


class MCTransportModel:
    """Turbid-media transport via baked white-MC tables (+ exact closed form for clear media).

    Instances are thread-safe. :func:`get_transport_model` hands out one *shared* instance per
    ``(n_photons, seed, cache_dir)``, so baked tables, the per-(material, excitation) ``J`` tables and
    radial profiles are reused across preview / capture / explain calls."""

    #: LRU capacities (a GTable is ~1.7 MB, a baked table set ~24 MB)
    MAX_GTABLES = 128
    MAX_BAKED = 16
    MAX_RADIAL = 512

    def __init__(self, n_photons: int = 200_000, seed: int = 0, cache_dir: str | Path | None = None):
        self.n_photons = int(n_photons)
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._baked: dict[tuple, BakedTables] = {}
        self._gtables: dict[tuple, GTable] = {}
        self._radial: dict[tuple, tuple] = {}
        self._lock = threading.RLock()

    # ---- table access --------------------------------------------------------------------
    def baked(self, g: float, n: float, n_out: float) -> BakedTables:
        key = (round(float(g), 4), round(float(n), 4), round(float(n_out), 4))
        with self._lock:
            bt = self._baked.pop(key, None)
            if bt is None:
                bt = bake_tables(*key, n_photons=self.n_photons, seed=self.seed, cache_dir=self.cache_dir)
                while len(self._baked) >= self.MAX_BAKED:
                    self._baked.pop(next(iter(self._baked)))
            self._baked[key] = bt                      # most-recently-used last
            return bt

    def gtable(self, optics: MaterialOptics, ex_nm: float, n_out: float) -> GTable:
        key = (_optics_key(optics), round(float(ex_nm), 3), round(float(n_out), 4))
        with self._lock:
            gt = self._gtables.pop(key, None)
            if gt is None:
                bt = self.baked(optics.g, optics.n, n_out)
                gt = _build_gtable(bt, optics, ex_nm)
                while len(self._gtables) >= self.MAX_GTABLES:
                    self._gtables.pop(next(iter(self._gtables)))
            self._gtables[key] = gt
            return gt

    # ---- main entry ----------------------------------------------------------------------
    def kernels(self, optics: MaterialOptics, ex_nm: float, cos_in, cos_out, thickness_mm,
                n_outside=1.0, path_mm=None, em_index=None) -> TransportKernels:
        """Transport kernels for ``P`` pixels (contracts.TransportModel.kernels).

        ``em_index`` (optional int array into ``optics.wavelengths_nm``) restricts ``fluorescence_G``
        to those emission wavelengths, shape (P, len(em_index)); values are identical to the
        corresponding columns of the full result (use it when only some wavelengths are needed, e.g.
        a band-limited capture: cost scales with the number of columns). ``mean_depth_mm`` of
        a *clear* medium is then weighted over the selected columns only."""
        mus_ex = float(optics.interp(optics.mus_per_mm, ex_nm))
        if optics.clear or mus_ex <= 1e-9:
            return clear_medium_kernel(optics, ex_nm, cos_in, cos_out, thickness_mm, n_outside, path_mm,
                                       em_index)
        cos_in, cos_out, thick, n_o = np.broadcast_arrays(
            *(np.atleast_1d(np.asarray(a, float)) for a in (cos_in, cos_out, thickness_mm, n_outside)))
        P = cos_in.shape[0]
        sel_w = _em_selection(optics, em_index)
        n = float(optics.n)
        G = np.empty((P, sel_w.size), np.float32)
        rho = np.empty(P)
        zbar = np.empty(P)
        n_o_key = np.round(n_o, 4)
        uniq = np.unique(n_o_key)
        for no in uniq:
            gt = self.gtable(optics, ex_nm, float(no))
            if uniq.size == 1:
                args = (np.ascontiguousarray(cos_in), np.ascontiguousarray(cos_out), np.ascontiguousarray(thick))
                mc.turbid_pixels(*args, float(no), n, MU_NODES, gt.d_nodes, gt.ln_j, gt.w_idx, gt.w_wgt,
                                 sel_w, gt.ln_zbar, gt.rho, G, rho, zbar)
            else:
                sel = np.where(n_o_key == no)[0]
                g_, r_, z_ = (np.empty((sel.size, sel_w.size), np.float32), np.empty(sel.size),
                              np.empty(sel.size))
                mc.turbid_pixels(np.ascontiguousarray(cos_in[sel]), np.ascontiguousarray(cos_out[sel]),
                                 np.ascontiguousarray(thick[sel]), float(no), n, MU_NODES, gt.d_nodes,
                                 gt.ln_j, gt.w_idx, gt.w_wgt, sel_w, gt.ln_zbar, gt.rho, g_, r_, z_)
                G[sel], rho[sel], zbar[sel] = g_, r_, z_
        res = TransportKernels(fluorescence_G=G, elastic_rho=rho, mean_depth_mm=zbar)
        self._attach_radial(res, optics, ex_nm, thick, float(np.median(n_o)))
        return res

    # ---- radial subsurface spread -----------------------------------------------------------
    def _attach_radial(self, res: TransportKernels, optics: MaterialOptics, ex_nm: float,
                       thick: np.ndarray, n_out: float) -> None:
        fin = thick[np.isfinite(thick) & (thick > 0)]
        d_med = float(np.median(fin)) if fin.size == thick.size and fin.size else np.inf
        key = (_optics_key(optics), round(float(ex_nm), 3), round(n_out, 4), d_med)
        with self._lock:
            hit = self._radial.pop(key, None)
            if hit is None:
                hit = radial_profile(self.baked(optics.g, optics.n, round(n_out, 4)), optics, ex_nm, d_med)
                while len(self._radial) >= self.MAX_RADIAL:
                    self._radial.pop(next(iter(self._radial)))
            self._radial[key] = hit
        res.radial_r_mm, res.radial_profile, res.radial_lambda_nm = hit

    # ---- optional: secondary-emission helper ------------------------------------------------------
    def reabsorbed_fraction(self, optics: MaterialOptics, em_nm: float, ex_nm: float | None = None,
                            cos_in: float = 1.0, thickness_mm: float = np.inf, n_outside: float = 1.0,
                            n_photons: int = 40_000, seed: int = 1) -> float:
        """Fraction of fluorescence emitted at ``em_nm`` that is absorbed inside the medium
        (neither leaves through the top nor the bottom face).  Emitters are distributed like the
        un-scattered excitation beam, ``exp(-mu_t,ex z/cos(th_i'))``, truncated to the slab; the
        emission transport is a conventional implicit-capture MC (``transport_mc.mcml_slab``).
        Includes the dye's own absorption only through ``optics.mua_per_mm`` (total).  ~0.1 s."""
        ex = em_nm if ex_nm is None else ex_nm
        mua = max(float(optics.interp(optics.mua_per_mm, em_nm)), 0.0)
        mus = max(float(optics.interp(optics.mus_per_mm, em_nm)), 1e-9)
        mut_ex = float(optics.interp(optics.mut_per_mm, ex))
        ci, _ = _refract_cos(np.array([cos_in]), n_outside, optics.n)
        rate = mut_ex / max(float(ci[0]), 1e-3)
        r = mc.mcml_slab(mua, mus, optics.g, optics.n, n_outside, float(thickness_mm), mode="exp",
                         zrate=rate, n_photons=n_photons, seed=seed)
        return float(r["A"])


_MODELS: dict[tuple, MCTransportModel] = {}
_MODELS_LOCK = threading.Lock()


def get_transport_model(n_photons: int = 200_000, seed: int = 0, cache_dir=None) -> MCTransportModel:
    """Factory: returns a model whose ``kernels(...)`` dispatches on ``optics.clear``.

    The model is a process-wide *shared* instance per ``(n_photons, seed, cache_dir)`` (thread-safe),
    so every preview / capture / explain call reuses the in-memory baked tables, assembled ``J``
    tables and radial profiles instead of rebuilding them. ``cache_dir=None`` -> the default disk
    cache (``$SPECTRALFORGE_MC_CACHE`` or ``~/.cache/spectralforge/mc``)."""
    cdir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    key = (int(n_photons), int(seed), str(cdir))
    with _MODELS_LOCK:
        m = _MODELS.get(key)
        if m is None:
            m = _MODELS[key] = MCTransportModel(n_photons=n_photons, seed=seed, cache_dir=cdir)
        return m


# ============================================================================================
# 6. Radial profile
# ============================================================================================
def radial_profile(bt: BakedTables, optics: MaterialOptics, ex_nm: float, thickness_mm: float = np.inf,
                   n_nodes: int = 7):
    """Diffuse reflectance vs radius for a normally incident point beam, normalised so that
    ``trapz(2 pi r p, r) = 1`` on the returned grid.  Returns ``(r_mm (n_r,), profile (W_r, n_r),
    lambda_nm (W_r,))``; nodes are the excitation wavelength plus ``n_nodes`` evenly spaced over
    the spectral grid.  Thickness enters through ``tau = mu_s d`` (interpolated in log tau)."""
    wl = optics.wavelengths_nm
    lam = np.unique(np.concatenate([[ex_nm], np.linspace(wl[0], wl[-1], n_nodes)]))
    mua = np.maximum(np.interp(lam, wl, optics.mua_per_mm), 0.0)
    mus = np.maximum(np.interp(lam, wl, optics.mus_per_mm), 1e-12)
    r_lo = 0.03 / mus.max()
    r_hi = min(300.0 / mus.min(), 1e4)
    r_grid = np.geomspace(r_lo, r_hi, N_RADIAL)
    edges = np.concatenate([[0.0], np.sqrt(r_grid[:-1] * r_grid[1:]), [r_grid[-1] * 1.2]])
    out = np.zeros((len(lam), N_RADIAL))
    rf = bt["rad_fin"].astype(np.float64)
    ri = bt["rad_inf"].astype(np.float64)
    for i in range(len(lam)):
        r_eff = max(mua[i] / mus[i], R_MIN)
        tau = mus[i] * thickness_mm if np.isfinite(thickness_mm) else np.inf
        if not tau <= _TAU_MAX:
            mass = _dot(ri, _bin_weights(np.array([r_eff]))[:, 0])
        else:
            te = max(tau, _TAU0)
            r_use = r_eff * (tau / _TAU0 if tau < _TAU0 else 1.0)
            e = _bin_weights(np.array([r_use]))[:, 0]
            jl = int(np.clip(np.searchsorted(TAU_NODES, te, side="right") - 1, 0, len(TAU_NODES) - 2))
            w = float(np.clip((np.log(te) - np.log(TAU_NODES[jl])) / np.log(TAU_NODES[jl + 1] / TAU_NODES[jl]), 0, 1))
            mass = (1 - w) * _dot(rf[jl], e) + w * _dot(rf[jl + 1], e)
        cum = np.concatenate([[0.0], np.cumsum(mass)])
        cq = np.interp(edges * mus[i], RAD_EDGES, cum)
        dm = np.diff(cq)
        prof = dm / (np.pi * np.diff(edges ** 2))
        norm = np.trapezoid(2 * np.pi * r_grid * prof, r_grid)
        out[i] = prof / norm if norm > 0 else prof
    return r_grid, out, lam


# ============================================================================================
# 7. Warm-up (cold-start hiding)
# ============================================================================================
_WARMUP_LOCK = threading.Lock()
_WARMUP_STATE: dict = {"state": "idle", "fraction": 0.0, "message": "", "result": None}


def warmup_status() -> dict:
    """Snapshot of the background warm-up: ``{"state": "idle"|"running"|"done"|"error",
    "fraction": 0..1, "message": str, "result": dict | None}`` (JSON-serialisable)."""
    with _WARMUP_LOCK:
        return dict(_WARMUP_STATE)


def _compile_kernels() -> None:
    """Force-load / compile every numba kernel with tiny inputs (``cache=True`` makes later
    processes load them from disk in ~1 s instead of compiling for ~20 s)."""
    wl = np.arange(300.0, 340.0, 1.0)
    z = np.zeros(wl.size)
    optics = MaterialOptics("warmup", wl, np.full(wl.size, 0.1), {}, np.full(wl.size, 10.0), 0.8, 1.4,
                            False, {}, {})
    # clear closed form (+ em_index path)
    clear = MaterialOptics("warmup_clear", wl, np.full(wl.size, 0.1), {}, z, 0.0, 1.4, True, {}, {})
    clear_medium_kernel(clear, 310.0, [0.9], [0.8], [1.0], 1.0, None)
    clear_medium_kernel(clear, 310.0, [0.9], [0.8], [1.0], 1.0, None, em_index=[0, 3])
    # turbid pixel kernel with a synthetic GTable (same dtypes / layouts as production)
    nn, K = 6, 4
    mc.turbid_pixels(np.array([0.9]), np.array([0.8]), np.array([1.0]), 1.0, 1.4, MU_NODES,
                     np.geomspace(0.01, 10.0, K), np.zeros((len(MU_NODES), len(MU_NODES), K + 1, nn), np.float32),
                     np.zeros(wl.size, np.int64), np.zeros(wl.size), np.arange(wl.size, dtype=np.int64),
                     np.zeros((len(MU_NODES), len(MU_NODES), K + 1)), np.zeros((len(MU_NODES), K + 1, _NB)),
                     np.empty((1, wl.size), np.float32), np.empty(1), np.empty(1))
    # white-MC bake + MCML (1 packet per cell: compile only)
    mc.bake_arrays(0.8, 1.4, 1.0, MU_NODES, TAU_NODES, U_EDGES, ZINF_EDGES, RAD_EDGES, 1, 0)
    mc.mcml_slab(0.1, 10.0, 0.8, 1.4, 1.0, 1.0, n_photons=64, seed=0)


def builtin_bake_plan(preset_names=None, grid_px: tuple[int, int] = (32, 24)) -> list[dict]:
    """The ``(g, n, n_outside)`` white-MC table combinations (and excitation wavelengths) the builtin
    presets actually use, found by tracing each preset at a tiny resolution.

    Returns one dict per preset: ``{"preset", "n_photons", "seed", "combos": [(g, n, n_out), ...],
    "excitations": [(material_id, ex_nm, n_out), ...]}``, combos ordered by first use."""
    from . import optics as optics_mod
    from . import presets as presets_mod
    from .tracer import trace_scene
    plan = []
    order = ["mehsi_lab_replica", "generic_macro_bench", "led_multispectral", "laser_line"]   # UI default first
    order += [k for k in presets_mod.PRESETS if k not in order]
    for name in (preset_names or order):
        exp = presets_mod.preset(name)
        acq = exp.acquisition
        hit = trace_scene(exp, *grid_px, volume_samples=0)
        fl = optics_mod.merged_fluorophores(exp)
        opts = {m.id: optics_mod.material_optics(m, fl, optics_mod.SPECTRAL_GRID_NM, acq.physics)
                for m in exp.materials}
        combos: list[tuple] = []
        exc: list[tuple] = []
        try:
            waves = sorted({float(s.wavelength_nm) for s in exp.source().sequence if s.enabled})
        except ValueError:
            waves = []
        for k in range(hit.material.shape[0]):
            for mi in np.unique(hit.material[k][hit.material[k] >= 0]):
                op = opts[exp.materials[int(mi)].id]
                if op.clear:
                    continue
                sel = hit.material[k] == mi
                for no in np.unique(np.round(hit.n_outside[k][sel], 4)):
                    c = (round(float(op.g), 4), round(float(op.n), 4), round(float(no), 4))
                    if c not in combos:
                        combos.append(c)
                    exc.extend((op.material_id, w, float(no)) for w in waves)
        plan.append({"preset": name, "n_photons": int(acq.mc_photons), "seed": int(acq.seed),
                     "combos": combos, "excitations": exc})
    return plan


def warmup(presets=None, *, jit: bool = True, bake: bool = True, gtables: bool = True,
           progress=None, cache_dir=None) -> dict:
    """Pre-pay every cold-start cost so the first preview is as fast as the tenth.

    1. ``jit``: load/compile all numba kernels (tracer geometry, transport, bake) with tiny inputs.
       Compiled code persists in numba's on-disk cache (``cache=True``), so this is ~1 s from the
       second process onwards.
    2. ``bake``: bake -- or load from the disk cache ``cache_dir`` (default
       ``$SPECTRALFORGE_MC_CACHE`` / ``~/.cache/spectralforge/mc``) -- the white-MC tables of every
       ``(g, n, n_outside)`` combination used by the builtin ``presets`` (default: all, the UI's default
       ``mehsi_lab_replica`` first). A bake costs ~10 s on 14 cores and is only ever done once per
       machine: the first run writes the cache ("shipped on first run"), later runs just load it.
    3. ``gtables``: assemble the per-(material, excitation) kernel tables for every source step of
       those presets (~40 ms each) into the shared in-memory model.

    Blocking; call it from a background thread (see :func:`start_warmup`). Idempotent and thread-safe
    with concurrent captures (they share the same model). ``progress(fraction, message)`` is called
    between stages. Returns timings: ``{"jit_s", "bake_s", "gtable_s", "total_s", "bakes_done",
    "gtables_built", "combos"}``. Exceptions in one stage do not stop the others (the stage name and
    error are listed under ``"errors"``)."""
    import time as _time
    t_all = _time.perf_counter()
    res: dict = {"jit_s": 0.0, "bake_s": 0.0, "gtable_s": 0.0, "bakes_done": 0, "gtables_built": 0,
                 "combos": [], "errors": []}

    def note(frac, msg):
        with _WARMUP_LOCK:
            _WARMUP_STATE.update(fraction=float(frac), message=msg)
        if progress is not None:
            try:
                progress(float(frac), msg)
            except Exception:                                    # progress sinks must never break warm-up
                pass

    if jit:
        note(0.0, "compiling numerical kernels")
        t = _time.perf_counter()
        try:
            _compile_kernels()
            from . import presets as presets_mod
            from .tracer import trace_scene
            trace_scene(presets_mod.preset("laser_line"), 24, 18)   # compiles geometry + volume-shadow kernels
        except Exception as exc:                                  # noqa: BLE001
            res["errors"].append(("jit", repr(exc)))
        res["jit_s"] = _time.perf_counter() - t
    plan = []
    if bake or gtables:
        try:
            note(0.1, "planning transport tables")
            plan = builtin_bake_plan(presets)
        except Exception as exc:                                  # noqa: BLE001
            res["errors"].append(("plan", repr(exc)))
    jobs = [(p["n_photons"], p["seed"], c) for p in plan for c in p["combos"]]
    jobs = list(dict.fromkeys(jobs))
    res["combos"] = [c for _, _, c in jobs]
    if bake:
        t = _time.perf_counter()
        for i, (nph, seed, (g, n, no)) in enumerate(jobs):
            note(0.15 + 0.7 * i / max(len(jobs), 1), f"transport tables {i + 1}/{len(jobs)}")
            try:
                get_transport_model(nph, seed, cache_dir).baked(g, n, no)
                res["bakes_done"] += 1
            except Exception as exc:                              # noqa: BLE001
                res["errors"].append(("bake", repr(exc)))
        res["bake_s"] = _time.perf_counter() - t
    if gtables:
        t = _time.perf_counter()
        from . import optics as optics_mod
        from . import presets as presets_mod
        for p in plan:
            try:
                exp = presets_mod.preset(p["preset"])
                fl = optics_mod.merged_fluorophores(exp)
                mats = {m.id: m for m in exp.materials}
                mdl = get_transport_model(p["n_photons"], p["seed"], cache_dir)
                note(0.9, f"kernel tables: {p['preset']}")
                for mid, w, no in dict.fromkeys(p["excitations"]):
                    op = optics_mod.material_optics(mats[mid], fl, optics_mod.SPECTRAL_GRID_NM, exp.acquisition.physics)
                    mdl.gtable(op, w, no)
                    res["gtables_built"] += 1
            except Exception as exc:                              # noqa: BLE001
                res["errors"].append(("gtable", repr(exc)))
        res["gtable_s"] = _time.perf_counter() - t
    res["total_s"] = _time.perf_counter() - t_all
    note(1.0, "ready")
    return res


def start_warmup(*, presets=None, **kw) -> threading.Thread:
    """Run :func:`warmup` in a daemon thread (the call the server makes at start-up) and return it.
    Progress / result are available from :func:`warmup_status` (``state`` = running -> done|error)."""
    def run():
        with _WARMUP_LOCK:
            _WARMUP_STATE.update(state="running", fraction=0.0, message="starting", result=None)
        try:
            out = warmup(presets, **kw)
            with _WARMUP_LOCK:
                _WARMUP_STATE.update(state="done", fraction=1.0, message="ready", result=out)
        except Exception as exc:                                  # noqa: BLE001
            with _WARMUP_LOCK:
                _WARMUP_STATE.update(state="error", message=repr(exc))
    th = threading.Thread(target=run, name="spectraforge-warmup", daemon=True)
    th.start()
    return th
