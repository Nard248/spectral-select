"""Export the Studio contract (JSON schema, presets, builtin library) for the web app.

Run after editing ``spectraforge/studio/schema.py`` or ``presets.py``::

    PYTHONPATH=src python scripts/export_studio_contract.py && (cd apps/studio-web && npm run gen:types)
"""
import json
from pathlib import Path

from spectraforge.studio.presets import PRESETS, builtin_fluorophores, builtin_materials
from spectraforge.studio.schema import EXTINCTION_REFERENCE_M_CM, SCHEMA_VERSION, Experiment

OUT = Path(__file__).resolve().parents[1] / "apps" / "studio-web" / "src" / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def _draft7_tuples(node):
    """json-schema-to-typescript understands draft-7 tuple ``items: [...]``, not ``prefixItems``."""
    if isinstance(node, dict):
        if "prefixItems" in node:
            node["items"] = node.pop("prefixItems")
            node["additionalItems"] = False
        for v in node.values():
            _draft7_tuples(v)
    elif isinstance(node, list):
        for v in node:
            _draft7_tuples(v)
    return node


schema = _draft7_tuples(Experiment.model_json_schema(mode="serialization"))
(OUT / "experiment.schema.json").write_text(json.dumps(schema, indent=1))
(OUT / "presets.json").write_text(json.dumps({
    "schema_version": SCHEMA_VERSION,
    "extinction_reference_M_cm": EXTINCTION_REFERENCE_M_CM,
    "fluorophores": [f.model_dump() for f in builtin_fluorophores()],
    "materials": [m.model_dump() for m in builtin_materials()],
    "experiments": {k: f().model_dump() for k, f in PRESETS.items()},
}, indent=1))
print("wrote", *sorted(p.name for p in OUT.iterdir()))
