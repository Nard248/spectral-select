"""Interface optics: unpolarised Fresnel, Snell refraction, GGX microfacet specular reflection.

All functions are vectorised over NumPy arrays and broadcast their scalar arguments.

Equations
---------
Snell's law ``n1 sin θ1 = n2 sin θ2``; with cosines only::

    cosθt = sqrt(1 - (n1/n2)² (1 - cosθi²))          (TIR when the radicand < 0)

Fresnel amplitude coefficients for s and p polarisation (Born & Wolf, *Principles of Optics*,
§1.5)::

    r_s = (n1 cosθi − n2 cosθt) / (n1 cosθi + n2 cosθt)
    r_p = (n2 cosθi − n1 cosθt) / (n2 cosθi + n1 cosθt)

Unpolarised reflectance ``R = (r_s² + r_p²) / 2``; transmittance ``T = 1 − R`` (non-absorbing
interface). Beyond the critical angle ``R = 1``.

GGX / Trowbridge-Reitz microfacet BRDF (Walter et al. 2007, *Microfacet Models for Refraction
through Rough Surfaces*; Heitz 2014 for the height-correlated Smith masking)::

    f(ωi, ωo) = D(h) G2(ωi, ωo) F(ωi·h) / (4 |n·ωi| |n·ωo|)
    D(h)  = α² / (π ((n·h)² (α² − 1) + 1)²)             ∫ D(h) (n·h) dω_h = 1
    Λ(ω)  = (−1 + sqrt(1 + α² tan²θ)) / 2
    G2    = 1 / (1 + Λ(ωi) + Λ(ωo))

Here ``α`` is the material's ``surface_roughness`` (clamped to ``>= ALPHA_MIN``).
"""
from __future__ import annotations

import numpy as np

ALPHA_MIN = 0.005


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", a, b)


# --------------------------------------------------------------------------------------------
# Snell + Fresnel
# --------------------------------------------------------------------------------------------
def snell_cos_t(cos_i, n1, n2):
    """Return ``(cos_t, tir)`` for incidence cosine ``cos_i`` going from index ``n1`` to ``n2``.

    ``cos_t`` is 0 where total internal reflection occurs (``tir`` True)."""
    cos_i = np.clip(np.asarray(cos_i, dtype=float), 0.0, 1.0)
    eta = np.asarray(n1, dtype=float) / np.asarray(n2, dtype=float)
    rad = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    tir = rad < 0.0
    cos_t = np.sqrt(np.where(tir, 0.0, rad))
    return cos_t, tir


def critical_angle_cos(n1: float, n2: float) -> float:
    """Cosine of the critical angle for ``n1 > n2`` (``nan`` if no TIR is possible)."""
    return float(np.sqrt(1.0 - (n2 / n1) ** 2)) if n1 > n2 else float("nan")


def fresnel_reflectance(cos_i, n1, n2):
    """Unpolarised Fresnel reflectance (1.0 under total internal reflection)."""
    cos_i = np.clip(np.asarray(cos_i, dtype=float), 0.0, 1.0)
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    cos_t, tir = snell_cos_t(cos_i, n1, n2)
    a = n1 * cos_i
    b = n2 * cos_t
    rs = np.where(a + b > 0, (a - b) / np.where(a + b > 0, a + b, 1.0), 1.0)
    c = n2 * cos_i
    d = n1 * cos_t
    rp = np.where(c + d > 0, (c - d) / np.where(c + d > 0, c + d, 1.0), 1.0)
    return np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp))


def fresnel_transmittance(cos_i, n1, n2):
    """``1 - R`` (0 under total internal reflection)."""
    return 1.0 - fresnel_reflectance(cos_i, n1, n2)


def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Mirror reflection of direction(s) ``d`` about unit normal(s) ``n`` (sign-agnostic)."""
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    return d - 2.0 * _dot(d, n)[..., None] * n


def refract(d: np.ndarray, n: np.ndarray, n1, n2):
    """Vector Snell refraction.

    ``d``: unit propagation direction(s) in the medium of index ``n1``; ``n``: unit interface
    normal(s) (either orientation — it is flipped internally so that it opposes ``d``);
    ``n2``: index of the medium being entered. Returns ``(t, tir)`` with ``t`` the unit refracted
    direction (the mirror reflection where ``tir``)."""
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    dn = _dot(d, n)
    n_opp = np.where((dn > 0)[..., None], -n, n)       # now d·n_opp <= 0
    cos_i = np.clip(-_dot(d, n_opp), 0.0, 1.0)
    eta = np.broadcast_to(np.asarray(n1, dtype=float) / np.asarray(n2, dtype=float), dn.shape)
    cos_t, tir = snell_cos_t(cos_i, n1, n2)
    t = eta[..., None] * d + (eta * cos_i - cos_t)[..., None] * n_opp
    t = np.where(tir[..., None], d - 2.0 * (d * n_opp).sum(-1, keepdims=True) * n_opp, t)
    norm = np.linalg.norm(t, axis=-1, keepdims=True)
    return t / np.where(norm > 0, norm, 1.0), tir


# --------------------------------------------------------------------------------------------
# GGX microfacet
# --------------------------------------------------------------------------------------------
def ggx_D(cos_h, alpha):
    """GGX normal distribution function (1/sr); normalised: ``∫ D cosθh dω = 1``."""
    a2 = np.maximum(np.asarray(alpha, dtype=float), ALPHA_MIN) ** 2
    c2 = np.clip(np.asarray(cos_h, dtype=float), 0.0, 1.0) ** 2
    den = c2 * (a2 - 1.0) + 1.0
    return a2 / (np.pi * den * den)


def smith_lambda(cos_theta, alpha):
    """Smith Λ for GGX: ``(−1 + sqrt(1 + α² tan²θ)) / 2``."""
    a2 = np.maximum(np.asarray(alpha, dtype=float), ALPHA_MIN) ** 2
    c2 = np.clip(np.asarray(cos_theta, dtype=float), 1e-9, 1.0) ** 2
    return 0.5 * (-1.0 + np.sqrt(1.0 + a2 * (1.0 - c2) / c2))


def smith_G2(cos_i, cos_o, alpha):
    """Height-correlated Smith masking-shadowing."""
    return 1.0 / (1.0 + smith_lambda(cos_i, alpha) + smith_lambda(cos_o, alpha))


def ggx_brdf(normal: np.ndarray, wi: np.ndarray, wo: np.ndarray, alpha, n_ior, n_outside=1.0):
    """GGX specular BRDF (1/sr) of a dielectric interface.

    ``normal``, ``wi`` (toward the light), ``wo`` (toward the viewer): unit vectors (N,3);
    ``alpha``: roughness (clamped to ``>= ALPHA_MIN``); ``n_ior``: index of the surface medium,
    ``n_outside`` that of the incident medium. Zero if either direction is below the surface.
    Reciprocal: ``f(wi, wo) = f(wo, wi)``."""
    normal = np.asarray(normal, dtype=float)
    wi = np.asarray(wi, dtype=float)
    wo = np.asarray(wo, dtype=float)
    ci = _dot(normal, wi)
    co = _dot(normal, wo)
    h = wi + wo
    hn = np.linalg.norm(h, axis=-1)
    h = h / np.where(hn > 0, hn, 1.0)[..., None]
    ch = _dot(normal, h)
    vh = np.clip(_dot(wi, h), 0.0, 1.0)
    F = fresnel_reflectance(vh, n_outside, n_ior)
    ok = (ci > 1e-6) & (co > 1e-6) & (hn > 0)
    ci_s = np.where(ok, ci, 1.0)
    co_s = np.where(ok, co, 1.0)
    f = ggx_D(ch, alpha) * smith_G2(ci_s, co_s, alpha) * F / (4.0 * ci_s * co_s)
    return np.where(ok, f, 0.0)
