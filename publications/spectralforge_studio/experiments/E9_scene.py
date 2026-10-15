"""Scene of E9: three side-by-side tissue types (different buried fluorophore mixtures) under a common
collagen cover wedge (thickness t0 -> t1 along x, randomised per capture)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C3_common import *  # noqa: F401,F403
from C3_common import P, DyeAmount, LayeredBlockShape, api  # noqa: E402

CLASSES = {  # name -> (NADH, FAD, collagen, elastin) uM of the buried layer
    "NADH-rich": (150.0, 25.0, 150.0, 20.0),
    "FAD-rich": (25.0, 120.0, 150.0, 20.0),
    "stroma": (30.0, 20.0, 700.0, 120.0),
}
SCALE = 0.25


def build_e9(t0=0.0, t1=1.5, seed=0, preview_scale=SCALE, excitations=None):
    mats = {m.id: m for m in P.builtin_materials()}
    cover = mats["collagen_dense"].model_copy(update={"id": "cover"})
    objs, new = [], [cover]
    for i, (name, (n, f, c, e)) in enumerate(CLASSES.items()):
        mid = f"buried_{i}"
        new.append(mats["dermis_papillary"].model_copy(update={"id": mid, "dyes": [
            DyeAmount(fluorophore="NADH", concentration_uM=n), DyeAmount(fluorophore="FAD", concentration_uM=f),
            DyeAmount(fluorophore="collagen", concentration_uM=c), DyeAmount(fluorophore="elastin", concentration_uM=e)]}))
        block = LayeredBlockShape(footprint_mm=(36.0, 12.0), layers=[
            P._layer(f"substrate_{i}", "dermis_reticular", 4.0), P._layer(f"buried_{i}", mid, 1.0),
            P._layer(f"cover_{i}", "cover", t0, t1)])
        objs.append(P._obj(f"tissue_{i}", name, block, "cover", (0.0, 0.0, (i - 1) * 12.5)))
    exp = P.tissue_under_collagen()
    src = exp.sources[0]
    seq = [s for s in src.sequence if excitations is None or float(s.wavelength_nm) in excitations]
    exp = exp.model_copy(update={"name": "E9 tissue types under collagen wedge",
                                 "materials": list(mats.values()) + new, "objects": objs,
                                 "sources": [src.model_copy(update={"sequence": seq})]})
    exp = api.set(exp, "/acquisition/seed", seed)
    exp = api.set(exp, "/acquisition/volume_mc/preview_fraction", PREVIEW_FRACTION)
    return api.set(exp, "/acquisition/preview_scale", preview_scale)
