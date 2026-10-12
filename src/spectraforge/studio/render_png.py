"""Server-side colour rendering of captures to PNG.

* ``true_color`` — band radiance L(λb) [mW/(mm²·sr)] is projected on the CIE 1931 2° colour-matching
  functions, ``X = Σ_b x̄(λb) L_b`` (likewise Y, Z; bands already integrate their passband so no extra
  Δλ factor), converted to linear sRGB (D65 matrix, IEC 61966-2-1), auto-exposed so the 99th
  percentile of luminance Y maps to 1 (times ``gain``) and gamma-encoded. x̄ȳz̄ use the multi-lobe
  Gaussian fit of Wyman, Sloan & Shirley (2013), J. Comput. Graph. Tech. 2(2), max error ~1 %.
* ``false_color`` — three bands mapped to R, G, B with a per-channel p1..p99 stretch.
* ``band`` — one band mapped through the viridis colormap.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

_XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

_VIRIDIS = np.array([
    (0x44, 0x01, 0x54), (0x48, 0x28, 0x78), (0x3e, 0x49, 0x89), (0x31, 0x68, 0x8e), (0x26, 0x82, 0x8e),
    (0x1f, 0x9e, 0x89), (0x35, 0xb7, 0x79), (0x6d, 0xcd, 0x59), (0xb4, 0xde, 0x2c), (0xfd, 0xe7, 0x25),
], dtype=np.float64) / 255.0


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_cmf(lam_nm) -> np.ndarray:
    """CIE 1931 colour-matching functions at ``lam_nm`` -> (N, 3) [x̄, ȳ, z̄] (Wyman et al. 2013)."""
    lam = np.asarray(lam_nm, dtype=np.float64)
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def _u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def true_color(cube: np.ndarray, emission_nm, gain: float = 1.0) -> np.ndarray:
    """(H, W, B) band radiance -> (H, W, 3) uint8 sRGB."""
    cmf = cie_cmf(emission_nm)                       # (B, 3)
    xyz = np.tensordot(np.asarray(cube, dtype=np.float64), cmf, axes=([2], [0]))   # (H, W, 3)
    y = xyz[..., 1]
    ref = float(np.percentile(y, 99.0)) if y.size else 0.0
    if not np.isfinite(ref) or ref <= 0:
        ref = float(y.max()) if y.size and y.max() > 0 else 1.0
    lin = (xyz / ref * 0.9 * gain) @ _XYZ_TO_SRGB.T
    return _u8(srgb_encode(lin))


def auto_false_color_bands(n_bands: int) -> tuple[int, int, int]:
    """Auto triplet for (R, G, B): bands at 85 %, 50 %, 15 % of the band list."""
    if n_bands <= 1:
        return (0, 0, 0)
    return tuple(int(round(f * (n_bands - 1))) for f in (0.85, 0.5, 0.15))  # type: ignore[return-value]


def _stretch(a: np.ndarray, gain: float, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(a, [lo_q, hi_q]) if a.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo) * gain, 0.0, 1.0)


def false_color(cube: np.ndarray, bands: tuple[int, int, int] | None = None, gain: float = 1.0) -> np.ndarray:
    b = bands or auto_false_color_bands(cube.shape[2])
    ch = [_stretch(np.asarray(cube[..., int(np.clip(i, 0, cube.shape[2] - 1))], dtype=np.float64), gain) for i in b]
    return _u8(np.stack(ch, axis=-1))


def viridis(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    pos = t * (len(_VIRIDIS) - 1)
    i0 = np.minimum(pos.astype(int), len(_VIRIDIS) - 2)
    f = (pos - i0)[..., None]
    return _VIRIDIS[i0] * (1 - f) + _VIRIDIS[i0 + 1] * f


def band_image(plane: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """(H, W) -> (H, W, 3) uint8 viridis, min..p99.5 stretch."""
    a = np.asarray(plane, dtype=np.float64)
    lo, hi = float(a.min()), float(np.percentile(a, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return _u8(viridis(np.clip((a - lo) / (hi - lo) * gain, 0.0, 1.0)))


def encode_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()
