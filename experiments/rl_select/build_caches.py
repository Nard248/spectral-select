"""Build the rl_select dataset caches from the ME-attention loader (single source of truth).

  PYTHONPATH=src python experiments/rl_select/build_caches.py [--out DIR] [--keys ...] [--verify-against OLD_DIR]

* lichens1 / lichens2 / collagen_*: ``me_attention.data.load(key)`` with ME_ATTENTION_RECORDED_WL=1 (recorded emission
  wavelengths; loads results/me_attention/cache/<key>_recwl.npz, or rebuilds from Data/processed if missing).
* lichens2_sub3: no env var (recorded wavelengths and the 400 nm exposure fix are built into its config).
Output: <out>/<key>.npz (default results/rl_select/cache_recwl) with the keys of the legacy RL caches
(X, ex_nm, em_nm, y, groups, coords, specimen_classes, excitations, emissions, grid_index, specimen_map, meta).
NEVER commit these files (the repo is public, Lichens 2 is unpublished).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEYS = ("lichens1", "lichens2_sub3", "collagen_acetic", "collagen_pepsin", "lichens2")
CACHE_KEYS = ("X", "ex_nm", "em_nm", "y", "groups", "coords", "specimen_classes", "excitations", "emissions",
              "grid_index", "specimen_map", "meta")


def load_me(key: str):
    """ME-attention MEData with the recorded wavelengths (env var set only around the call)."""
    from me_attention import data as D
    old = os.environ.get("ME_ATTENTION_RECORDED_WL")
    try:
        if key == "lichens2_sub3":
            os.environ.pop("ME_ATTENTION_RECORDED_WL", None)
        else:
            os.environ["ME_ATTENTION_RECORDED_WL"] = "1"
        return D.load(key)
    finally:
        if old is None:
            os.environ.pop("ME_ATTENTION_RECORDED_WL", None)
        else:
            os.environ["ME_ATTENTION_RECORDED_WL"] = old


def write_cache(key: str, out: Path) -> Path:
    d = load_me(key)
    info = dict(d.info)
    assert info.get("recorded_wavelengths", False), f"{key}: loader did not use recorded wavelengths"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{key}.npz"
    np.savez(p, X=d.X, ex_nm=d.ex_nm, em_nm=d.em_nm, y=d.y, groups=d.groups, coords=d.coords,
             specimen_classes=d.specimen_classes, excitations=d.excitations, emissions=d.emissions,
             grid_index=d.grid_index, specimen_map=d.specimen_map,
             meta=np.array(json.dumps(dict(name=d.name, class_names=d.class_names, image_shape=list(d.image_shape),
                                           info=info))))
    return p


def verify(key: str, new_dir: Path, old_dir: Path) -> str:
    """X-identity check against the legacy RL cache. Returns a one-line report; raises on mismatch."""
    n = np.load(new_dir / f"{key}.npz", allow_pickle=True)
    if key == "lichens2_sub3":
        o = np.load(old_dir / "lichens2.npz", allow_pickle=True)
        sub = np.load(new_dir / "lichens2_sub3.npz", allow_pickle=True)
        # subset pixels: match by image coordinates
        full = {tuple(c): i for i, c in enumerate(o["coords"])}
        idx = np.array([full[tuple(c)] for c in sub["coords"]])
        Xo = o["X"][idx]
        m400 = o["ex_nm"] == 400.0
        assert Xo.shape == n["X"].shape and np.array_equal(o["ex_nm"], n["ex_nm"])
        assert np.array_equal(Xo[:, ~m400], n["X"][:, ~m400]), "non-400 nm columns differ"
        a = Xo[:, m400].astype(np.float64) * 0.1
        rel = float(np.max(np.abs(a - n["X"][:, m400]) / np.maximum(np.abs(a), 1e-12)))
        assert rel < 1e-6, f"400 nm block is not x0.1 (max rel diff {rel})"
        return (f"{key}: X == old lichens2 X[{len(idx)} subset px]: {int((~m400).sum())} non-400 nm columns bit-identical, "
                f"{int(m400.sum())} 400 nm columns = x0.1 (max rel diff {rel:.2e}, float32 rounding)")
    o = np.load(old_dir / f"{key}.npz", allow_pickle=True)
    assert o["X"].shape == n["X"].shape, (o["X"].shape, n["X"].shape)
    assert np.array_equal(o["X"], n["X"]), f"{key}: X differs from legacy cache"
    for k in ("y", "groups", "coords", "specimen_classes", "ex_nm"):
        assert np.array_equal(o[k], n[k]), f"{key}: {k} differs"
    ch = int((o["em_nm"] != n["em_nm"]).sum())
    return f"{key}: X bit-identical {o['X'].shape}; em_nm labels changed for {ch}/{len(o['em_nm'])} features"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "rl_select" / "cache_recwl"))
    ap.add_argument("--keys", nargs="*", default=list(DEFAULT_KEYS))
    ap.add_argument("--verify-against", default=None, help="legacy RL cache dir (results/rl_select/cache)")
    a = ap.parse_args()
    out = Path(a.out)
    for k in a.keys:
        p = write_cache(k, out)
        z = np.load(p, allow_pickle=True)
        assert set(CACHE_KEYS) <= set(z.files)
        sc = z["specimen_classes"]
        print(f"{k}: X{z['X'].shape} classes={int(z['y'].max()) + 1} specimens/class={np.bincount(sc).tolist()} -> {p}", flush=True)
        if a.verify_against:
            print("  VERIFY", verify(k, out, Path(a.verify_against)), flush=True)
