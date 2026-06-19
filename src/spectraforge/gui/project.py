"""Save/load a Forge project (ForgeState) as a single compressed .npz file.

Metadata (fluorophores, materials, layer descriptors, acquisition, artifacts, seed) is
stored as a JSON string; each layer's amount_map is stored as a compressed array.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.material import Material
from spectraforge.gui.layer import Layer
from spectraforge.gui.state import ForgeState


def _acq_to_meta(acq: AcquisitionConfig) -> dict:
    return {
        "excitations": [float(e) for e in acq.excitations],
        "em_min": acq.em_min, "em_max": acq.em_max, "em_step": acq.em_step,
        "lamp": {str(k): v for k, v in acq.lamp.items()},
        "exposure": {str(k): v for k, v in acq.exposure.items()},
        "power": {str(k): v for k, v in acq.power.items()},
    }


def _acq_from_meta(m: dict) -> AcquisitionConfig:
    fkeys = lambda d: {float(k): v for k, v in d.items()}
    return AcquisitionConfig(
        excitations=[float(e) for e in m["excitations"]],
        em_min=m["em_min"], em_max=m["em_max"], em_step=m["em_step"],
        lamp=fkeys(m["lamp"]), exposure=fkeys(m["exposure"]), power=fkeys(m["power"]),
    )


def save_project(state: ForgeState, path) -> None:
    meta = {
        "height": state.height, "width": state.width,
        "seed": state.seed, "active_layer": state.active_layer,
        "fluorophores": [asdict(f) for f in state.library.values()],
        "materials": [{"name": m.name, "recipe": m.recipe} for m in state.materials.values()],
        "layers": [
            {"name": layer.name,
             "material": {"name": layer.material.name, "recipe": layer.material.recipe},
             "visible": bool(layer.visible)}
            for layer in state.layers
        ],
        "acquisition": _acq_to_meta(state.acquisition),
        "artifacts": asdict(state.artifacts),
    }
    arrays = {f"_layer{i}": layer.amount_map for i, layer in enumerate(state.layers)}
    np.savez_compressed(path, _meta=np.array(json.dumps(meta)), **arrays)


def load_project(path) -> ForgeState:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["_meta"]))

    library = {f["name"]: Fluorophore(**f) for f in meta["fluorophores"]}
    materials = {m["name"]: Material(m["name"], dict(m["recipe"])) for m in meta["materials"]}

    state = ForgeState(
        height=meta["height"], width=meta["width"],
        library=library, materials=materials,
        acquisition=_acq_from_meta(meta["acquisition"]),
        artifacts=ArtifactConfig(**meta["artifacts"]),
        seed=meta["seed"], active_layer=meta["active_layer"],
    )
    state.layers = []
    for i, lmeta in enumerate(meta["layers"]):
        mat = Material(lmeta["material"]["name"], dict(lmeta["material"]["recipe"]))
        state.layers.append(
            Layer(name=lmeta["name"], material=mat,
                  amount_map=np.asarray(data[f"_layer{i}"], dtype=float),
                  visible=lmeta["visible"])
        )
    return state
