"""Provenance: canonical experiment hash and a software / hardware environment record.

Every export (npz, study manifest, dataset index) embeds :func:`provenance` so a result can be traced to the
exact experiment (``experiment_hash``), code (package version + git commit + dirty flag) and environment.

* :func:`canonical_json` - deterministic JSON (sorted keys, no whitespace, floats via ``repr`` round-trip).
* :func:`experiment_hash` - SHA-256 of the canonical JSON of the validated experiment (hex, 64 chars).
* :func:`provenance` - dict with versions, git state, platform, CPU count, timestamp, numba threading layer.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _to_plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: keys sorted, compact separators, numpy scalars converted."""
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")
    return json.dumps(_to_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=default)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def experiment_hash(exp: Any) -> str:
    """SHA-256 (hex) of the canonical JSON of an ``Experiment`` (or an equivalent JSON dict)."""
    return sha256_hex(canonical_json(exp))


def package_version() -> str:
    try:
        from importlib.metadata import version
        return version("spectral-select")
    except Exception:  # noqa: BLE001
        pass
    try:
        import tomllib
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            return str(tomllib.load(f)["project"]["version"])
    except Exception:  # noqa: BLE001
        return "unknown"


@lru_cache(maxsize=1)
def git_state() -> dict:
    """``{commit, branch, dirty}`` of the checkout this package runs from (``None`` values if not a git repo)."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None
    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status)}


def _numba_info() -> dict:
    try:
        import numba
        info: dict[str, Any] = {"version": numba.__version__}
        try:
            info["threading_layer"] = numba.threading_layer()
        except Exception:  # noqa: BLE001 - not yet chosen (no parallel kernel launched)
            info["threading_layer"] = None
        info["num_threads"] = int(numba.get_num_threads())
        return info
    except Exception:  # noqa: BLE001
        return {"version": None, "threading_layer": None, "num_threads": None}


def provenance(extra: dict | None = None) -> dict:
    """Environment record (JSON-safe).  ``extra`` entries are merged under their own keys."""
    try:
        from .server import ENGINE_VERSION
    except Exception:  # noqa: BLE001
        ENGINE_VERSION = "studio"
    out = {
        "package": "spectral-select",
        "package_version": package_version(),
        "engine_version": ENGINE_VERSION,
        "git": dict(git_state()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": _numba_info(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import scipy
        out["scipy"] = scipy.__version__
    except Exception:  # noqa: BLE001
        pass
    if extra:
        out.update(extra)
    return out


def attach(result: Any, exp: Any = None, **extra: Any) -> dict:
    """Store a provenance record in ``result.meta['provenance']`` (``meta`` is created on demand, so this
    works with the current ``CaptureResult`` dataclass) and return it."""
    meta = getattr(result, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
        try:
            result.meta = meta
        except Exception:  # noqa: BLE001 - frozen / slotted: give up silently, provenance is advisory
            pass
    rec = provenance(extra or None)
    if exp is not None:
        rec["experiment_hash"] = experiment_hash(exp)
    meta["provenance"] = rec
    return rec
