"""Built-in fluorophore starter library."""
from __future__ import annotations

import json
from pathlib import Path

from spectraforge.fluorophore import Fluorophore

_DATA = Path(__file__).parent / "data" / "fluorophores.json"


def load_builtin_library() -> dict[str, Fluorophore]:
    """Load the bundled starter fluorophores as ``{name: Fluorophore}``."""
    records = json.loads(_DATA.read_text())
    return {r["name"]: Fluorophore(**r) for r in records}
