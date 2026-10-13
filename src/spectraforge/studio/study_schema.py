"""Pydantic schema of a computational study (design of experiments over an :class:`Experiment`).

A :class:`StudySpec` names a base experiment (inline or a preset), the *parameters* to vary (RFC 6901 pointers
with ranges / choices, optional linked pointers), the *design* (grid, random, LHS, scrambled Sobol, Saltelli for
Sobol' indices, Morris trajectories), the number of Monte Carlo *replicates* per design point, the capture
*mode* and the scalar *metrics* extracted from every capture (see :mod:`.study`).

JSON example::

    {"name": "cover thickness x fluorophore", "base_experiment": "layered_wedge_phantom",
     "parameters": [
       {"pointer": "/objects/0/shape/layers/2/thickness/thickness_mm", "label": "cover_mm",
        "kind": "continuous", "low": 0.1, "high": 1.0, "scale": "linear"}],
     "design": {"kind": "lhs", "n": 8, "seed": 1}, "replicates": 2, "mode": "preview",
     "metrics": [{"name": "signal", "kind": "band_sum", "roi": {"shape": "rect", "x0": .3, "y0": .3, "x1": .7, "y1": .7}}]}
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analysis import BandRef, NormSpec, RoiSpec
from .schema import Experiment

F = Annotated[float, Field(allow_inf_nan=False)]
Stat = Literal["mean", "median", "std", "min", "max"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParameterSpec(_M):
    """One varied parameter.  ``continuous`` / ``integer`` use ``low``/``high`` (``scale``: ``linear`` or ``log``);
    ``choice`` uses ``values`` (any JSON scalars / strings).  ``also_set`` pointers receive the same value
    (e.g. both ends of a wedge)."""

    pointer: str
    label: str | None = None
    kind: Literal["continuous", "integer", "choice"] = "continuous"
    low: F | None = None
    high: F | None = None
    scale: Literal["linear", "log"] = "linear"
    values: list[Any] | None = None
    also_set: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ParameterSpec":
        if not self.pointer.startswith("/"):
            raise ValueError(f"pointer must be an RFC 6901 pointer starting with '/', got {self.pointer!r}")
        if self.kind == "choice":
            if not self.values or len(self.values) < 2:
                raise ValueError("choice parameter needs at least 2 'values'")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.kind} parameter needs 'low' and 'high'")
            if not self.high > self.low:
                raise ValueError("'high' must be greater than 'low'")
            if self.scale == "log" and self.low <= 0:
                raise ValueError("log scale needs low > 0")
            if self.values is not None and len(self.values) < 1:
                raise ValueError("'values' must not be empty")
        if self.label is None:
            self.label = self.pointer.strip("/").replace("/", ".")
        return self


class DesignSpec(_M):
    """``grid``: ``levels`` per continuous/integer parameter (choices use all their values).
    ``random`` / ``lhs`` / ``sobol``: ``n`` points.  ``saltelli``: base sample size ``n`` (rows = n*(k+2));
    ``morris``: ``n`` trajectories on a ``levels``-level grid (rows = n*(k+1))."""

    kind: Literal["grid", "random", "lhs", "sobol", "saltelli", "morris"] = "lhs"
    n: int = Field(8, ge=1, le=1_000_000)
    levels: int = Field(4, ge=2, le=1000)
    seed: int = 0


class MetricSpec(_M):
    """A scalar summary of one capture.

    kinds: ``roi_band`` (stat over the ROI of one band: ``ex``, ``band_nm``), ``band_ratio`` (stat of band ``a`` /
    stat of band ``b``), ``band_sum`` (stat of the per-pixel sum of bands in [``from_nm``, ``to_nm``] of ``ex``),
    ``peak_wavelength`` (peak position, nm, of the ROI-mean spectrum at ``ex``), ``eem_integral`` (integral of the
    ROI-mean EEM over excitation index x emission wavelength: sum over excitations of the trapezoid integral over
    the emission grid), ``gt_stat`` (stat of the ground-truth map ``gt`` in the ROI), ``contrast`` (between
    ``roi`` and ``roi_b`` of the per-pixel metric ``inner``: Michelson (A-B)/(A+B) or Weber (A-B)/B) and ``cnr``
    (|mean_A - mean_B| / sqrt((var_A + var_B)/2)).  ROI coordinates are image *fractions* (0..1) by default
    (``roi_units='fraction'``) so one study works at preview and full resolution, or ``pixels``.
    ``ex`` defaults to the first captured excitation."""

    name: str
    kind: Literal["roi_band", "band_ratio", "band_sum", "peak_wavelength", "eem_integral", "gt_stat",
                  "contrast", "cnr"]
    roi: RoiSpec | None = None
    roi_b: RoiSpec | None = None
    roi_units: Literal["fraction", "pixels"] = "fraction"
    norm: NormSpec = Field(default_factory=NormSpec)
    stat: Stat = "mean"
    ex: F | None = None
    band_nm: F | None = None
    a: BandRef | None = None
    b: BandRef | None = None
    from_nm: F | None = None
    to_nm: F | None = None
    gt: str | None = None
    inner: dict | None = None            # analysis metric dict for contrast / cnr (default: band_sum of ``ex``)
    contrast: Literal["michelson", "weber"] = "michelson"

    @model_validator(mode="after")
    def _check(self) -> "MetricSpec":
        k = self.kind
        if k == "roi_band" and self.band_nm is None:
            raise ValueError("roi_band needs band_nm")
        if k == "band_ratio" and (self.a is None or self.b is None):
            raise ValueError("band_ratio needs a and b ({ex, band_nm})")
        if k == "gt_stat" and not self.gt:
            raise ValueError("gt_stat needs gt (ground-truth layer name)")
        if k in ("contrast", "cnr") and (self.roi is None or self.roi_b is None):
            raise ValueError(f"{k} needs two ROIs: roi and roi_b")
        return self


class StudySpec(_M):
    name: str = "study"
    base_experiment: Union[Experiment, str]
    parameters: list[ParameterSpec] = Field(min_length=1, max_length=64)
    design: DesignSpec = Field(default_factory=DesignSpec)
    replicates: int = Field(1, ge=1, le=1000)
    mode: Literal["preview", "full"] = "preview"
    excitations: list[F] | None = None
    metrics: list[MetricSpec] = Field(min_length=1)
    max_runs: int = Field(20_000, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "StudySpec":
        labels = [p.label for p in self.parameters]
        if len(set(labels)) != len(labels):
            raise ValueError(f"parameter labels must be unique, got {labels}")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"metric names must be unique, got {names}")
        return self

    def resolve_base(self) -> Experiment:
        """The base experiment (a preset name is looked up in :mod:`.presets`)."""
        if isinstance(self.base_experiment, Experiment):
            return self.base_experiment
        from .presets import PRESETS
        if self.base_experiment not in PRESETS:
            raise ValueError(f"unknown preset {self.base_experiment!r}; available: {sorted(PRESETS)}")
        return PRESETS[self.base_experiment]()


# ------------------------------------------------------------------------------------------ datasets
class TargetSpec(_M):
    """A continuous per-sample regression target: ``stat`` of ground-truth map ``gt`` inside ``roi`` (whole image
    if omitted), e.g. mean cover thickness (``layer_thickness:cover``)."""

    name: str
    gt: str
    stat: Stat = "mean"
    roi: RoiSpec | None = None
    roi_units: Literal["fraction", "pixels"] = "fraction"


class DatasetSpec(_M):
    """Synthetic ML dataset: ``n`` captures of the base experiment with the parameters drawn from a randomised
    design (``random`` | ``lhs`` | ``sobol``) over their ranges; every sample gets its own Monte Carlo / sensor
    seed ``base_seed + i``.

    ``label_scheme`` selects the per-pixel class map (0 = background / no signal source):
    ``origin_layer`` (argmax of ``origin_layer_fraction:<layer>``: the layer most of the signal originates in),
    ``top_layer`` (outermost layer with non-zero thickness), ``object`` (1 + scene-object index),
    ``material`` (raw engine material code), or ``gt_key:<name>`` (any ground-truth map, rounded to int, NaN -> 0).
    ``maps`` lists extra ground-truth maps saved as float32 regression targets; ``targets`` are scalar targets."""

    name: str = "dataset"
    base_experiment: Union[Experiment, str]
    parameters: list[ParameterSpec] = Field(default_factory=list)
    n: int = Field(16, ge=1, le=100_000)
    design: Literal["random", "lhs", "sobol"] = "lhs"
    seed: int = 0
    mode: Literal["preview", "full"] = "preview"
    excitations: list[F] | None = None
    label_scheme: str = "origin_layer"
    maps: list[str] = Field(default_factory=list)
    targets: list[TargetSpec] = Field(default_factory=list)
    save_npz: bool = False

    resolve_base = StudySpec.resolve_base

    def as_study(self) -> StudySpec:
        """Equivalent :class:`StudySpec` (for design generation / validation; its metric is a placeholder)."""
        return StudySpec(name=self.name, base_experiment=self.base_experiment,
                         parameters=self.parameters,
                         design=DesignSpec(kind=self.design, n=self.n, seed=self.seed), mode=self.mode,
                         excitations=self.excitations, metrics=[MetricSpec(name="_", kind="band_sum")])
