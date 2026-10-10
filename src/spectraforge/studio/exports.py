"""Capture exports.

``spectradata_pkl`` writes the exact on-disk layout of ``spectral_select.types.SpectraData.to_pickle``
(``{'data': {str(ex): {'cube', 'wavelengths'}}, 'excitation_wavelengths', 'mask', 'exposure_times',
'laser_powers'}``; times in seconds) plus extra top-level keys (experiment JSON, engine version, seed,
ground-truth layer names). ``SpectraData.from_pickle`` puts every unknown top-level key into
``SpectraData.metadata``, so the file loads with the existing tools unchanged.
"""
from __future__ import annotations

import io
import json
import pickle
import re
import zipfile
from urllib.parse import quote

import numpy as np

FORMATS = {
    "spectradata_pkl": ("pkl", "application/octet-stream"),
    "npz": ("npz", "application/octet-stream"),
    "tiff_zip": ("tiff.zip", "application/zip"),
    "ground_truth_zip": ("ground_truth.zip", "application/zip"),
    "experiment_json": ("json", "application/json"),
}


_SAFE_RE = re.compile(r"[^A-Za-z0-9._@:+-]")


def safe_name(text: object, maxlen: int = 80, default: str = "item") -> str:
    """Whitelist ``[A-Za-z0-9._@:+-]`` (everything else, incl. path separators and non-ASCII, -> ``_``).

    Never returns an empty string, a name containing ``..`` or with leading dots (hidden files) or one longer than
    ``maxlen``; safe as a zip member stem and as a download filename component."""
    s = re.sub(r"\.{2,}", "_", _SAFE_RE.sub("_", str(text))).lstrip(".")
    s = s[:maxlen].rstrip(".")
    return s if s.strip("_") else default


def slug(text: str) -> str:
    return safe_name(text, 60, "capture")


def content_disposition(filename: str) -> str:
    """RFC 6266 ``attachment`` header value with a sanitized ASCII filename (no CR/LF/quotes/backslashes)."""
    fn = safe_name(filename, 120, "download")
    return f'attachment; filename="{fn}"; filename*=UTF-8\'\'{quote(fn, safe="")}'


def _ex_key(ex: float) -> str:
    return f"{ex:g}"


def _offset_adu(result) -> float:
    try:
        exp = json.loads(result.experiment_json)
        cams = exp.get("cameras") or []
        cid = exp.get("active_camera_id")
        cam = next((c for c in cams if c["id"] == cid), cams[0])
        return float(cam["sensor"]["offset_adu"])
    except Exception:  # noqa: BLE001 - metadata only; fall back to no offset removal
        return 0.0


def _seed(result) -> int | None:
    try:
        return int(json.loads(result.experiment_json)["acquisition"]["seed"])
    except Exception:  # noqa: BLE001
        return None


def spectradata_payload(result, sample_name: str) -> dict:
    """Dict in the ``SpectraData.to_pickle`` layout (see module docstring)."""
    off = _offset_adu(result)
    data, expo, power = {}, {}, {}
    for ex in result.excitations_nm:
        k = str(float(ex))
        cube = (np.asarray(result.adu[ex], dtype=np.float32) - np.float32(off)).astype(np.float32)
        data[k] = {"cube": cube, "wavelengths": [float(w) for w in result.emission_nm[ex]]}
        expo[k] = float(result.exposure_ms[ex]) / 1000.0
        power[k] = float(result.power_mW[ex])
    return {
        "data": data,
        "excitation_wavelengths": sorted(float(e) for e in result.excitations_nm),
        "mask": np.ones((result.height, result.width), dtype=bool),
        "exposure_times": expo,
        "laser_powers": power,
        # ---- extra metadata (lands in SpectraData.metadata on load) ----
        "source": "SpectralForge Studio",
        "engine_version": result.engine_version,
        "capture_mode": result.mode,
        "seed": _seed(result),
        "adu_offset_subtracted": off,
        "experiment_json": result.experiment_json,
        "ground_truth_layers": sorted(result.ground_truth.keys()),
        "sample_label": sample_name,
    }


def to_spectradata_pkl(result, sample_name: str) -> bytes:
    return pickle.dumps(spectradata_payload(result, sample_name), protocol=pickle.HIGHEST_PROTOCOL)


def to_npz(result) -> bytes:
    arrs: dict[str, np.ndarray] = {}
    for ex in result.excitations_nm:
        k = _ex_key(ex)
        arrs[f"adu_{k}"] = np.asarray(result.adu[ex], dtype=np.float32)
        arrs[f"radiance_{k}"] = np.asarray(result.radiance[ex], dtype=np.float32)
        arrs[f"emission_nm_{k}"] = np.asarray(result.emission_nm[ex], dtype=np.float64)
    for name, g in result.ground_truth.items():
        arrs[f"gt_{safe_name(name, 100, 'layer')}"] = np.asarray(g)
    arrs["excitations_nm"] = np.asarray(result.excitations_nm, dtype=np.float64)
    arrs["experiment_json"] = np.asarray(result.experiment_json)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return buf.getvalue()


def to_tiff_zip(result) -> bytes:
    import tifffile
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for ex in result.excitations_nm:
            adu = np.clip(np.rint(np.asarray(result.adu[ex])), 0, 65535).astype(np.uint16)   # (H, W, B)
            pages = np.ascontiguousarray(np.moveaxis(adu, 2, 0))                              # (B, H, W)
            desc = json.dumps({"excitation_nm": float(ex), "emission_nm": [float(w) for w in result.emission_nm[ex]],
                               "exposure_ms": float(result.exposure_ms[ex]), "power_mW": float(result.power_mW[ex]),
                               "mode": result.mode, "unit": "ADU"})
            buf = io.BytesIO()
            tifffile.imwrite(buf, pages, photometric="minisblack", description=desc)
            zf.writestr(f"adu_ex{safe_name(_ex_key(ex), 20, 'x')}nm.tif", buf.getvalue())
        zf.writestr("experiment.json", result.experiment_json)
    return out.getvalue()


def to_ground_truth_zip(result) -> bytes:
    out = io.BytesIO()
    index = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        used: set[str] = set()
        for name, g in result.ground_truth.items():
            a = np.asarray(g)
            b = io.BytesIO()
            np.save(b, a, allow_pickle=False)
            stem = safe_name(name, 100, "layer")
            n, i = stem, 1
            while n in used or n == "ground_truth":          # keep members unique after sanitizing
                i += 1
                n = f"{stem}_{i}"
            used.add(n)
            zf.writestr(f"{n}.npy", b.getvalue())
            index[name] = {"file": f"{n}.npy", "shape": list(a.shape), "dtype": str(a.dtype)}
        zf.writestr("ground_truth.json", json.dumps(index, indent=1))
    return out.getvalue()


def build_export(result, capture_id: str, name: str, fmt: str) -> tuple[bytes, str, str]:
    """-> (bytes, attachment filename, media type). Raises ``KeyError`` for unknown formats."""
    if fmt not in FORMATS:
        raise KeyError(fmt)
    ext, media = FORMATS[fmt]
    base = f"{slug(name)}_{safe_name(capture_id, 32, 'id')}"
    if fmt == "spectradata_pkl":
        payload = to_spectradata_pkl(result, name)
    elif fmt == "npz":
        payload = to_npz(result)
    elif fmt == "tiff_zip":
        payload = to_tiff_zip(result)
    elif fmt == "ground_truth_zip":
        payload = to_ground_truth_zip(result)
    else:
        payload = json.dumps(json.loads(result.experiment_json), indent=2).encode()
    return payload, f"{base}.{ext}", media
