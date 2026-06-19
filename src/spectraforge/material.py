"""Material = a named recipe of fluorophores with relative concentrations (the 'brush')."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Material:
    name: str
    recipe: dict[str, float] = field(default_factory=dict)  # fluorophore_name -> concentration

    def fluorophores(self) -> list[str]:
        return list(self.recipe.keys())
