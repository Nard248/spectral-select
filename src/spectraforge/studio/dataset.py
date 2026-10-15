"""Synthetic dataset generator for machine learning.

``generate_dataset(spec, out_dir)`` draws ``n`` parameter vectors from a randomised design over the parameter
ranges, simulates each (own seed), and writes::

    out_dir/datasets.json            index: spec, provenance, class names, one entry per sample
    out_dir/samples/s00000.pkl       SpectraData pickle (``spectral_select.types.SpectraData.from_pickle`` loads it)
    out_dir/samples/s00000.npz       labels (int16 class map), selected ground-truth maps, scalar targets
    out_dir/samples/s00000.json      sidecar written last (its presence marks a finished sample; resumable)
    out_dir/samples/s00000_capture.npz   full capture (only with ``save_npz``)

The pickle carries ``labels`` (the class map) and ``class_names`` as extra top-level keys, which
``SpectraData.from_pickle`` places in ``SpectraData.metadata``.  :func:`load_sample` / :func:`iter_dataset` read the
dataset back as ``(SpectraData, labels, targets)``.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import analysis, exports
from .provenance import experiment_hash, provenance
from .study import StudyError, build_experiment, make_design, param_unit, validate_spec  # noqa: F401
from .study_schema import DatasetSpec, TargetSpec

BACKGROUND = 0


# ------------------------------------------------------------------------------------------ labels
def label_map(res, scheme: str) -> tuple[np.ndarray, dict[int, str]]:
    """Per-pixel integer class map (H, W) int16 and ``{class id: name}`` for a label ``scheme``
    (see :class:`DatasetSpec`)."""
    gt = res.ground_truth
    h, w = res.height, res.width
    exp = json.loads(res.experiment_json)
    if scheme == "object":
        o = np.asarray(gt["object"])
        lab = np.where(np.isfinite(o) & (o >= 0), o + 1, BACKGROUND).astype(np.int16)
        names = {0: "background", **{i + 1: ob["name"] for i, ob in enumerate(exp["objects"])}}
        return lab, names
    if scheme == "material":
        m = np.asarray(gt["material"])
        lab = np.where(np.isfinite(m), m, -1).astype(np.int16) + 1
        return lab, {int(v): ("background" if v == 0 else f"material_{int(v) - 1}") for v in np.unique(lab)}
    if scheme in ("origin_layer", "top_layer"):
        layers = [k.split(":", 1)[1] for k in gt if k.startswith("origin_layer_fraction:")]
        if not layers:
            raise StudyError(f"label scheme {scheme!r} needs a layered-block object (no layer ground truth found); "
                             "use 'object' or 'gt_key:<name>'")
        names = {0: "background", **{i + 1: n for i, n in enumerate(layers)}}
        if scheme == "origin_layer":
            fr = np.stack([np.nan_to_num(np.asarray(gt[f"origin_layer_fraction:{n}"], float), nan=-1.0)
                           for n in layers], axis=0)
            lab = np.where(fr.max(axis=0) > 0, fr.argmax(axis=0) + 1, BACKGROUND).astype(np.int16)
        else:
            th = np.stack([np.nan_to_num(np.asarray(gt[f"layer_thickness:{n}"], float), nan=0.0) for n in layers])
            top = np.full((h, w), BACKGROUND, dtype=np.int16)
            for i in range(len(layers)):          # bottom -> top: later layers overwrite
                top = np.where(th[i] > 1e-6, i + 1, top)
            lab = top.astype(np.int16)
        return lab, names
    if scheme.startswith("gt_key:"):
        key = scheme.split(":", 1)[1]
        g = analysis._gt_map(res, key)
        lab = np.where(np.isfinite(g), np.rint(g), 0).astype(np.int16)
        return lab, {int(v): f"{key}={int(v)}" for v in np.unique(lab)}
    raise StudyError(f"unknown label scheme {scheme!r}")


def compute_target(res, t: TargetSpec) -> float:
    from .study import _reduce, _roi_px
    g = analysis._gt_map(res, t.gt)
    if t.roi is None:
        return _reduce(g[np.isfinite(g)], t.stat)
    rows, cols = analysis._roi_pixels(res, _roi_px(t.roi, t.roi_units, res.width, res.height))[2]
    return _reduce(g[rows, cols], t.stat)


# ------------------------------------------------------------------------------------------ generation
def _draw(spec: DatasetSpec) -> tuple[list[list[Any]], np.ndarray]:
    ps = spec.parameters
    if not ps:
        return [[] for _ in range(spec.n)], np.zeros((spec.n, 0))
    d = make_design(spec.as_study())
    return d.values, d.U


def generate_dataset(spec: DatasetSpec, out_dir: str | Path, *, resume: bool = True,
                     progress: Callable[[float, str], None] | None = None, cancel: Callable[[], bool] | None = None,
                     capture_fn: Callable | None = None) -> dict:
    """Simulate and write the dataset (see module docstring).  Returns the ``datasets.json`` index dict."""
    from . import api
    t0 = time.perf_counter()
    out = Path(out_dir)
    sdir = out / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    base = spec.resolve_base()
    if spec.parameters:
        validate_spec(spec.as_study(), base)
    values, U = _draw(spec)
    base_doc = base.model_dump(mode="json")
    seed0 = int(base.acquisition.seed)
    labels_all: dict[int, str] = {}
    entries: list[dict] = []
    for i in range(spec.n):
        if cancel is not None and cancel():
            break
        sid = f"s{i:05d}"
        side = sdir / f"{sid}.json"
        if resume and side.exists():
            e = json.loads(side.read_text())
        else:
            exp = build_experiment(base_doc, spec.parameters, values[i], seed0 + i)
            if capture_fn is not None:
                res = capture_fn(exp, spec.mode, spec.excitations)
            else:
                res = api.capture(exp, mode=spec.mode if spec.mode in ("preview", "fast") else "full",
                                  excitations=spec.excitations)
            lab, names = label_map(res, spec.label_scheme)
            maps = {f"map:{k}": np.asarray(analysis._gt_map(res, k), dtype=np.float32) for k in spec.maps}
            targets = {t.name: compute_target(res, t) for t in spec.targets}
            payload = exports.spectradata_payload(res, f"{spec.name}_{sid}")
            payload.update(labels=lab, class_names={str(k): v for k, v in names.items()},
                           params={p.label: v for p, v in zip(spec.parameters, values[i])}, targets=targets,
                           sample_id=sid)
            with open(sdir / f"{sid}.pkl", "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            np.savez_compressed(sdir / f"{sid}.npz", labels=lab, **maps,
                                targets_json=np.asarray(json.dumps(targets)))
            if spec.save_npz:
                api.save(res, sdir / f"{sid}_capture.npz")
            e = {"id": sid, "index": i, "seed": seed0 + i, "exp_hash": experiment_hash(exp),
                 "params": {p.label: v for p, v in zip(spec.parameters, values[i])},
                 "targets": targets, "class_names": {str(k): v for k, v in names.items()},
                 "class_counts": {str(int(k)): int((lab == k).sum()) for k in np.unique(lab)},
                 "shape": [int(res.height), int(res.width)],
                 "excitations_nm": [float(x) for x in res.excitations_nm],
                 "files": {"spectradata": f"samples/{sid}.pkl", "labels": f"samples/{sid}.npz"}}
            side.write_text(json.dumps(e, default=float))
        for k, v in e["class_names"].items():
            labels_all.setdefault(int(k), v)
        entries.append(e)
        if progress:
            progress((i + 1) / spec.n, f"sample {i + 1}/{spec.n}")
    index = {"name": spec.name, "n": len(entries), "n_requested": spec.n, "label_scheme": spec.label_scheme,
             "class_names": {str(k): labels_all[k] for k in sorted(labels_all)},
             "parameters": [p.label for p in spec.parameters], "targets": [t.name for t in spec.targets],
             "maps": list(spec.maps), "spec": spec.model_dump(mode="json"), "provenance": provenance(),
             "wall_time_s": time.perf_counter() - t0, "samples": entries}
    (out / "datasets.json").write_text(json.dumps(index, indent=2, default=float))
    return index


# ------------------------------------------------------------------------------------------ reading
def load_index(out_dir: str | Path) -> dict:
    return json.loads((Path(out_dir) / "datasets.json").read_text())


def load_sample(out_dir: str | Path, entry: dict | str):
    """``(SpectraData, labels (H, W) int16, extras)`` of one sample, loaded with the repo's own
    ``SpectraData.from_pickle``.  ``extras`` holds ``targets`` and the saved ``maps``."""
    from spectral_select.types import SpectraData
    out = Path(out_dir)
    if isinstance(entry, str):
        entry = next(e for e in load_index(out)["samples"] if e["id"] == entry)
    sd = SpectraData.from_pickle(out / entry["files"]["spectradata"])
    with np.load(out / entry["files"]["labels"]) as z:
        labels = z["labels"]
        maps = {k[4:]: z[k] for k in z.files if k.startswith("map:")}
    return sd, labels, {"targets": entry["targets"], "maps": maps, "params": entry["params"]}


def iter_dataset(out_dir: str | Path) -> Iterator[tuple]:
    for e in load_index(out_dir)["samples"]:
        yield load_sample(out_dir, e)
