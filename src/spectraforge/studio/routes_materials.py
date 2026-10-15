"""Engine-backed optical properties of one material (``POST /api/material_optics``) and the calibrated-fluorophore library.

The web material inspector charts mu_a(lambda) and mu_s'(lambda) from the *same* :func:`optics.material_optics` and
:mod:`tissue_optics` tables the simulator uses (the client-side copy in ``tissueSpectra.ts`` is only an offline fallback).

``POST /api/material_optics``  body ``{material: MaterialSpec, fluorophores?: [FluorophoreSpec], nm_min?, nm_max?, step?}``
-> ``{nm, mua_per_mm, mus_prime_per_mm, g, n, components: {name: mua_per_mm[]}}``.  ``components`` holds one curve per
dye (``dye:<name>``), the background absorber (``absorber``) and each tissue chromophore of Jacques (2013, Phys. Med.
Biol. 58, R37, eq. 1): ``hbo2, hb, water, lipid, melanosome, bilirubin, beta_carotene`` (their sum with the dyes and the
absorber is ``mua_per_mm``, up to the engine's absorption floor).

``GET /api/calibrated_fluorophores`` -> ``[{spec: FluorophoreSpec, provenance: {...}}]`` (sim-to-real library, ``calibration.py``).
"""
from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from . import optics, spectra, tissue_optics
from .presets import builtin_fluorophores
from .schema import FluorophoreSpec, MaterialSpec

router = APIRouter()

MAX_POINTS = 2001
MAX_FLUOROPHORES = 64
MAX_BODY_BYTES = 1_000_000


def _issues(exc: ValidationError) -> list[dict]:
    return [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", ""))} for e in exc.errors()[:20]]


def _num(body: dict, key: str, default: float) -> float:
    v = body.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise HTTPException(422, detail=[{"loc": [key], "msg": "must be a finite number"}])
    return float(v)


def material_optics_payload(material: MaterialSpec, fluorophores: dict[str, FluorophoreSpec], nm: np.ndarray) -> dict:
    """The response of ``/api/material_optics`` for ``material`` on the wavelength grid ``nm``."""
    mo = optics.material_optics(material, fluorophores, grid=nm)
    comps: dict[str, list[float]] = {}
    for name, mua in mo.mua_dyes_per_mm.items():
        comps[f"dye:{name}"] = np.asarray(mua, float).tolist()
    ab = material.background_absorption
    if ab.preset != "none":
        comps["absorber"] = np.asarray(spectra.absorber_mua_per_mm(ab.preset, ab.scale, nm, ab.custom), float).tolist()
    t = material.tissue
    if t is not None:
        amounts = {"hbo2": (t.blood_volume_fraction * t.oxygen_saturation * t.hemoglobin_g_per_L),
                   "hb": (t.blood_volume_fraction * (1.0 - t.oxygen_saturation) * t.hemoglobin_g_per_L),
                   "water": t.water_fraction, "lipid": t.lipid_fraction, "melanosome": t.melanosome_fraction,
                   "bilirubin": t.bilirubin_uM, "beta_carotene": t.beta_carotene_uM}
        for k, amount in amounts.items():
            if amount > 0:
                comps[k] = np.asarray(tissue_optics.chromophore_mua_per_mm(k, nm, amount), float).tolist()
    sc = material.scattering
    musp = np.zeros_like(nm) if (mo.clear or sc.model == "none") else optics.reduced_scattering_per_mm(sc, nm)
    return {"nm": nm.tolist(), "mua_per_mm": np.asarray(mo.mua_per_mm, float).tolist(),
            "mus_prime_per_mm": np.asarray(musp, float).tolist(), "g": float(mo.g), "n": float(mo.n),
            "components": comps}


@router.post("/api/material_optics")
async def material_optics_endpoint(request: Request):
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(413, "request body too large")
    try:
        import json
        body: Any = json.loads(raw)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "request body is not valid JSON")
    if not isinstance(body, dict) or "material" not in body:
        raise HTTPException(422, detail=[{"loc": ["material"], "msg": "field required"}])
    nm_min, nm_max, step = _num(body, "nm_min", 350.0), _num(body, "nm_max", 900.0), _num(body, "step", 5.0)
    if not (100.0 <= nm_min < nm_max <= 2500.0):
        raise HTTPException(422, detail=[{"loc": ["nm_min"], "msg": "need 100 <= nm_min < nm_max <= 2500"}])
    if step <= 0 or (nm_max - nm_min) / step + 1 > MAX_POINTS:
        raise HTTPException(422, detail=[{"loc": ["step"], "msg": f"step must be > 0 and give at most {MAX_POINTS} points"}])
    try:
        mat = MaterialSpec.model_validate(body["material"])
        extra = body.get("fluorophores") or []
        if not isinstance(extra, list) or len(extra) > MAX_FLUOROPHORES:
            raise HTTPException(422, detail=[{"loc": ["fluorophores"], "msg": f"must be a list of at most {MAX_FLUOROPHORES}"}])
        fl = {f.name: f for f in builtin_fluorophores()}
        for f in extra:
            spec = FluorophoreSpec.model_validate(f)
            fl[spec.name] = spec
    except ValidationError as exc:
        raise HTTPException(422, detail=_issues(exc))
    nm = np.arange(nm_min, nm_max + 0.5 * step, step)
    try:
        return await asyncio.to_thread(material_optics_payload, mat, fl, nm)
    except KeyError as exc:
        raise HTTPException(422, detail=[{"loc": ["material", "dyes"], "msg": str(exc.args[0])}])


@router.get("/api/calibrated_fluorophores")
def calibrated_fluorophores_endpoint():
    from .calibration import calibrated_fluorophores

    def clean(x: Any) -> Any:
        if isinstance(x, float):
            return x if math.isfinite(x) else None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    return [{"spec": clean(s.model_dump(mode="json")), "provenance": clean(p)}
            for s, p in calibrated_fluorophores(with_provenance=True)]
