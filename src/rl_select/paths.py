"""Central configuration of the rl_select study: results root, cache dir, dataset list.

Environment variables (read at import time, inherited by worker processes):
  RL_SELECT_RESULTS   results root (absolute, or relative to the repo root). Default ``results/rl_select``
                      (the legacy 2026-10-19 results). The v2 rerun sets ``results/rl_select_v2``.
  RL_SELECT_CACHE     dataset cache dir (absolute, or relative to ``results/rl_select``). Default ``cache_recwl``
                      = caches with the recorded emission wavelengths (+ the cleaned ``lichens2_sub3``).
  RL_SELECT_DATASETS  comma-separated dataset list for the study. Default = ``DEFAULT_DATASETS``.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = ROOT / "results" / "rl_select"
DEFAULT_DATASETS = ("lichens1", "lichens2_sub3", "collagen_acetic", "collagen_pepsin")
# loadable for comparison (the full, partly mislabelled Lichens 2)
KNOWN_DATASETS = ("lichens1", "lichens2", "lichens2_sub3", "collagen_acetic", "collagen_pepsin")


def _resolve(p: str, base: Path) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else base / q


def results_root() -> Path:
    v = os.environ.get("RL_SELECT_RESULTS", "").strip()
    return _resolve(v, ROOT) if v else LEGACY_ROOT


def cache_dir() -> Path:
    return _resolve(os.environ.get("RL_SELECT_CACHE", "").strip() or "cache_recwl", LEGACY_ROOT)


def datasets() -> tuple[str, ...]:
    v = os.environ.get("RL_SELECT_DATASETS", "").strip()
    ds = tuple(s.strip() for s in v.split(",") if s.strip()) if v else DEFAULT_DATASETS
    bad = [d for d in ds if d not in KNOWN_DATASETS]
    if bad:
        raise ValueError(f"unknown datasets {bad}; known: {KNOWN_DATASETS}")
    return ds


def family(key: str) -> str:
    """'lichens' or 'collagen' (cross-family transfer groups)."""
    return "lichens" if key.startswith("lichens") else "collagen"


def cross_pools(ds: tuple[str, ...] | None = None) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """{'C2L': (collagen datasets, lichens datasets), 'L2C': (...)} for the configured dataset list."""
    ds = ds or datasets()
    lic = tuple(d for d in ds if family(d) == "lichens")
    col = tuple(d for d in ds if family(d) == "collagen")
    out = {}
    if lic and col:
        out["C2L"] = (col, lic)
        out["L2C"] = (lic, col)
    return out


def cross_of(ds: tuple[str, ...] | None = None) -> dict[str, str]:
    """dataset -> cross-family source tag (C2L for lichens targets, L2C for collagen targets)."""
    ds = ds or datasets()
    return {d: ("C2L" if family(d) == "lichens" else "L2C") for d in ds}


def seeds() -> tuple[int, ...]:
    """Policy training seeds (env RL_SELECT_SEEDS, comma list; default 0,1,2)."""
    v = os.environ.get("RL_SELECT_SEEDS", "").strip()
    return tuple(int(s) for s in v.split(",") if s.strip() != "") if v else (0, 1, 2)


def paper_out() -> Path:
    """Where paper_figures.py writes figures/ and tables/. Legacy root -> publications/rl_band_selection;
    any other results root -> <root>/paper (never overwrites the publication folder)."""
    v = os.environ.get("RL_SELECT_PAPER_OUT", "").strip()
    if v:
        return _resolve(v, ROOT)
    rr = results_root()
    return ROOT / "publications" / "rl_band_selection" if rr == LEGACY_ROOT else rr / "paper"


def workers(default: int | None = None) -> int:
    """Worker processes (env RL_SELECT_WORKERS, else ``default``, else cpu_count - 1)."""
    v = os.environ.get("RL_SELECT_WORKERS", "").strip()
    if v:
        return max(1, int(v))
    return max(1, default if default is not None else (os.cpu_count() or 2) - 1)
