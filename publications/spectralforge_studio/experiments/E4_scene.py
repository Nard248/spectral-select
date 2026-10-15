"""E4 shared helpers: scene, capture with common random numbers, ROI/noise extraction.

Scene: 16 x 16 mm block (4 mm dermis_reticular substrate | 7 mm tissue layer (E2 'buried' NADH/FAD/collagen/elastin
autofluorescent dermis_papillary) | dense-collagen cover of thickness d).  An inclusion whose TOP touches the
cover/tissue interface (so 'depth' = cover thickness d = depth of the inclusion top below the surface):
  nodule: sphere of radius r with `dysplasia_nodule` tissue optics, NADH 400 / FAD 150 / collagen 60 uM
  vessel: cylinder (axis z) of radius r filled with `blood_oxygenated`
Lab TLS lines + Nuance LCTF camera (unchanged), field of view = whole 348x256 sensor (75 mm), full resolution.
"""
from __future__ import annotations

import functools
import sys

import numpy as np

from spectraforge.studio import api, block_capture, capture as cap, presets as P, transport3d as t3
from spectraforge.studio.schema import DyeAmount, Inclusion, LayeredBlockShape

EXC = [340.0, 365.0, 400.0, 430.0]
FOOT = 16.0
H_SUB, H_TIS = 4.0, 7.0
NODULE_NADH, NODULE_FAD = 400.0, 150.0     # uM: metabolically active nodule (6.7x / 3.8x the host tissue)


def install_crn():
    """Make block_capture's passes use per-photon streams (common random numbers). Idempotent."""
    if getattr(t3.excitation_pass, "_crn", False):
        return
    for name in ("excitation_pass", "emission_pass"):
        f = getattr(t3, name)

        @functools.wraps(f)
        def g(*a, _f=f, **k):
            k.setdefault("per_photon_streams", True)
            return _f(*a, **k)
        g._crn = True
        setattr(t3, name, g)


def uninstall_crn():
    for name in ("excitation_pass", "emission_pass"):
        f = getattr(t3, name)
        if getattr(f, "_crn", False):
            setattr(t3, name, f.__wrapped__)


def build(kind: str | None, r: float, d: float, seed: int = 0, photons: int = 1_000_000, voxel: float = 0.1,
          excitations=EXC, null: bool = False):
    """kind: 'nodule' | 'vessel' | None (no inclusion). null=True: same inclusion geometry but filled with the host tissue
    (matched control: identical voxelisation, channel layout and, with CRN, photon histories)."""
    mats = {m.id: m for m in P.builtin_materials()}
    cover = mats["collagen_dense"].model_copy(update={"id": "cover"})
    tissue = mats["dermis_papillary"].model_copy(update={"id": "tissue", "dyes": [
        DyeAmount(fluorophore="NADH", concentration_uM=60.0), DyeAmount(fluorophore="FAD", concentration_uM=40.0),
        DyeAmount(fluorophore="collagen", concentration_uM=250.0), DyeAmount(fluorophore="elastin", concentration_uM=40.0)]})
    nodule = mats["dysplasia_nodule"].model_copy(update={"id": "nodule", "dyes": [
        DyeAmount(fluorophore="NADH", concentration_uM=NODULE_NADH), DyeAmount(fluorophore="FAD", concentration_uM=NODULE_FAD),
        DyeAmount(fluorophore="collagen", concentration_uM=60.0)]})
    incs = []
    if kind == "nodule":
        incs = [Inclusion(name="inc", material_id="tissue" if null else "nodule", kind="sphere",
                          center_mm=(0.0, H_SUB + H_TIS - r, 0.0), radius_mm=r)]
    elif kind == "vessel":
        incs = [Inclusion(name="inc", material_id="tissue" if null else "blood_oxygenated", kind="cylinder", axis="z",
                          center_mm=(0.0, H_SUB + H_TIS - r, 0.0), radius_mm=r, length_mm=FOOT)]
    layers = [P._layer("substrate", "dermis_reticular", H_SUB), P._layer("tissue", "tissue", H_TIS)]
    if d > 0:
        layers.append(P._layer("cover", "cover", d))
    block = LayeredBlockShape(footprint_mm=(FOOT, FOOT), layers=layers, inclusions=incs)
    exp = P.tissue_under_collagen()
    seq = [s for s in exp.sources[0].sequence if float(s.wavelength_nm) in excitations]
    src = exp.sources[0].model_copy(update={"sequence": seq})
    vs = exp.acquisition.volume_mc.model_copy(update={
        "excitation_photons": photons, "emission_photons": photons, "preview_fraction": 1.0,
        "inclusion_voxel_mm": voxel, "emission_node_step_nm": 40.0})
    exp = exp.model_copy(update={
        "name": "E4 inclusion detectability",
        "materials": list(mats.values()) + [cover, tissue, nodule],
        "objects": [P._obj("block", "block", block, "cover" if d > 0 else "tissue", (0, 0, 0))],
        "sources": [src]})
    exp = exp.model_copy(update={"acquisition": exp.acquisition.model_copy(update={"volume_mc": vs, "seed": seed})})
    return exp
