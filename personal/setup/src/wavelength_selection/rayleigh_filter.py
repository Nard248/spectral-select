"""Physical filtering for excitation/emission wavelength pairs."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class WavelengthPair:
    excitation_nm: float
    emission_nm: float

    @property
    def pair_id(self) -> str:
        return f"ex{self.excitation_nm:g}_em{self.emission_nm:g}"


@dataclass(frozen=True)
class RejectedPair:
    pair: WavelengthPair
    reasons: tuple[str, ...]


def rejection_reasons(pair: WavelengthPair, offset_nm: float = 40.0) -> tuple[str, ...]:
    """Return physical rejection reasons used by the original preprocessing code."""
    reasons: list[str] = []
    if pair.emission_nm < pair.excitation_nm + offset_nm:
        reasons.append("first_order_rayleigh")
    if 2 * pair.excitation_nm - offset_nm <= pair.emission_nm <= 2 * pair.excitation_nm + offset_nm:
        reasons.append("second_order_rayleigh")
    return tuple(reasons)


def apply_rayleigh_filter(
    pairs: Iterable[WavelengthPair], offset_nm: float = 40.0
) -> tuple[list[WavelengthPair], list[RejectedPair]]:
    accepted: list[WavelengthPair] = []
    rejected: list[RejectedPair] = []
    for pair in pairs:
        reasons = rejection_reasons(pair, offset_nm)
        (rejected if reasons else accepted).append(
            RejectedPair(pair, reasons) if reasons else pair  # type: ignore[arg-type]
        )
    return accepted, rejected
