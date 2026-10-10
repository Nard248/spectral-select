"""SpectralForge Studio experiment schema — the single contract between the 3D editor and the engine.

An :class:`Experiment` fully describes a virtual multi-excitation hyperspectral acquisition:
fluorophores, materials (uniform optical media), 3D objects, a background, one or more light
sources with an excitation sequence, and one or more hyperspectral cameras. The browser edits it,
the Python engine simulates it, and it is embedded verbatim in every export.

Conventions (shared with the web app — keep ``apps/studio-web/src/types/experiment.ts`` in sync):

* Right-handed world frame, **Y up** (three.js convention). The optical table top is ``y = 0``.
* Lengths in **millimetres**, wavelengths in **nanometres**, angles in **degrees** in the schema
  (radians internally), power in **milliwatts**, exposure in **milliseconds**, concentration in
  **micromolar**, absorption / scattering coefficients in **1/mm**.
* ``Transform.rotation_deg`` is an intrinsic Euler XYZ rotation (three.js default order ``'XYZ'``).
* Shapes are defined in their local frame (see each shape for its origin), then transformed.
* Sources and cameras are posed by ``position_mm`` + ``target_mm`` (look-at) + ``roll_deg``.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

# Relative library extinction 1.0 corresponds to this molar extinction (EGFP-like), M^-1 cm^-1.
EXTINCTION_REFERENCE_M_CM = 55_000.0

Vec3 = tuple[float, float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              json_schema_serialization_defaults_required=True)


# --------------------------------------------------------------------------------------------
# Fluorophores
# --------------------------------------------------------------------------------------------
class FluorophoreSpec(_Model):
    """Parametric fluorophore (same parameters as :class:`spectraforge.Fluorophore`) plus the
    photophysics the 3D engine adds. ``extinction`` is relative: ``1.0 == EXTINCTION_REFERENCE``."""

    name: str
    ex_peak_nm: float = Field(gt=150, lt=1200)
    ex_fwhm_nm: float = Field(gt=0)
    em_peak_nm: float = Field(gt=150, lt=1500)
    em_fwhm_nm: float = Field(gt=0)
    quantum_yield: float = Field(0.5, ge=0, le=1)
    extinction: float = Field(1.0, ge=0)
    em_skew: float = Field(0.0, ge=0)
    vibronic: float = Field(0.0, ge=0)
    # Photobleaching quantum yield: probability that one absorbed photon destroys the molecule.
    bleach_quantum_yield: float = Field(1e-7, ge=0, le=1)
    display_color: str | None = None   # UI only, "#rrggbb"


# --------------------------------------------------------------------------------------------
# Materials — every material is a uniform optical medium (volume), even "surfaces":
# white paper = highly scattering volume + optical brightener dye; black anodized aluminium =
# strongly absorbing volume with a rough interface.
# --------------------------------------------------------------------------------------------
class DyeAmount(_Model):
    fluorophore: str
    concentration_uM: float = Field(ge=0)


class ScatteringSpec(_Model):
    """Reduced-scattering power law ``mus'(λ) = mus_prime_500 · (λ/500 nm)^-power_b`` (1/mm) with
    Henyey-Greenstein anisotropy ``g`` (``mus = mus' / (1 - g)``). ``model='none'`` = clear medium."""

    model: Literal["none", "power_law"] = "none"
    mus_prime_500_per_mm: float = Field(0.0, ge=0)
    power_b: float = Field(1.0, ge=0)
    g: float = Field(0.9, ge=-0.99, le=0.99)


class AbsorberSpec(_Model):
    """Non-fluorescent background absorption added to the dyes' absorption.

    ``preset`` spectra are tabulated in :mod:`spectraforge.studio.spectra`; ``scale`` multiplies
    them (for ``water`` scale 1 = pure water; for ``hemoglobin`` / ``melanin`` scale is the
    absorption coefficient at 500 nm in 1/mm). ``custom`` is a list of ``(nm, mua_per_mm)``."""

    preset: Literal["none", "water", "hemoglobin", "melanin", "carbon_black", "custom"] = "none"
    scale: float = Field(1.0, ge=0)
    custom: list[tuple[float, float]] = Field(default_factory=list)


class FretPair(_Model):
    donor: str
    acceptor: str
    k_per_uM: float = Field(ge=0)   # transfer efficiency E = k c_A / (1 + k c_A)


class MaterialSpec(_Model):
    id: str
    name: str
    dyes: list[DyeAmount] = Field(default_factory=list)
    refractive_index: float = Field(1.33, ge=1.0, le=3.0)
    scattering: ScatteringSpec = Field(default_factory=ScatteringSpec)
    background_absorption: AbsorberSpec = Field(default_factory=AbsorberSpec)
    surface_roughness: float = Field(0.15, ge=0.0, le=1.0)   # GGX alpha of the interface
    aqueous: bool = False                   # enables the water-Raman line
    raman_strength: float = Field(1.0, ge=0)  # relative water-Raman cross-section multiplier
    quenching_ksv_per_uM: float = Field(0.0, ge=0)  # Stern-Volmer self-quenching constant
    fret: list[FretPair] = Field(default_factory=list)
    display_color: str | None = None        # UI only


# --------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------
class Transform(_Model):
    position_mm: Vec3 = (0.0, 0.0, 0.0)
    rotation_deg: Vec3 = (0.0, 0.0, 0.0)


class BoxShape(_Model):
    """Axis-aligned box centred on the local origin."""
    kind: Literal["box"] = "box"
    size_mm: Vec3 = (20.0, 5.0, 20.0)


class SphereShape(_Model):
    """Sphere centred on the local origin."""
    kind: Literal["sphere"] = "sphere"
    radius_mm: float = Field(10.0, gt=0)


class CylinderShape(_Model):
    """Solid cylinder, axis along local +Y, centred on the local origin."""
    kind: Literal["cylinder"] = "cylinder"
    radius_mm: float = Field(10.0, gt=0)
    height_mm: float = Field(10.0, gt=0)


class DropletShape(_Model):
    """Spherical cap: flat circular base on local ``y = 0`` (radius ``base_radius_mm``), apex at
    ``y = height_mm``. ``height_mm <= base_radius_mm`` is a sessile drop (contact angle <= 90°)."""
    kind: Literal["droplet"] = "droplet"
    base_radius_mm: float = Field(3.0, gt=0)
    height_mm: float = Field(1.5, gt=0)


class PetriDishShape(_Model):
    """Glass/plastic dish: a flat base disc plus a cylindrical wall, base bottom on local ``y = 0``,
    axis +Y. The object's ``material_id`` fills it to ``fill_height_mm`` above the inner base;
    the vessel uses ``SceneObject.vessel_material_id``."""
    kind: Literal["petri_dish"] = "petri_dish"
    inner_radius_mm: float = Field(43.0, gt=0)
    wall_mm: float = Field(1.2, gt=0)
    height_mm: float = Field(15.0, gt=0)
    base_mm: float = Field(1.2, gt=0)
    fill_height_mm: float = Field(4.0, ge=0)


class CuvetteShape(_Model):
    """Rectangular cuvette: inner cavity ``inner_mm`` (x, y, z), walls ``wall_mm``; outer bottom
    on local ``y = 0``. Filled with the object's material to ``fill_height_mm``."""
    kind: Literal["cuvette"] = "cuvette"
    inner_mm: Vec3 = (10.0, 40.0, 10.0)
    wall_mm: float = Field(1.25, gt=0)
    fill_height_mm: float = Field(30.0, ge=0)


Shape = Annotated[
    Union[BoxShape, SphereShape, CylinderShape, DropletShape, PetriDishShape, CuvetteShape],
    Field(discriminator="kind"),
]


class SceneObject(_Model):
    id: str
    name: str
    shape: Shape
    transform: Transform = Field(default_factory=Transform)
    material_id: str
    vessel_material_id: str | None = None   # petri_dish / cuvette walls; default "glass"
    visible: bool = True


class RoomLight(_Model):
    enabled: bool = False
    spectrum: Literal["fluorescent_tube", "white_led", "daylight", "incandescent"] = "fluorescent_tube"
    illuminance_lux: float = Field(5.0, ge=0)   # ambient illuminance on the table


class Background(_Model):
    """``none``: rays that miss every object return nothing (void). ``plane``: a slab of
    ``material_id`` covering the table top (``y in [-thickness, 0]``), ``extent_mm`` square."""
    kind: Literal["none", "plane"] = "plane"
    material_id: str | None = "black_anodized"
    thickness_mm: float = Field(5.0, gt=0)
    extent_mm: float = Field(600.0, gt=0)
    room_light: RoomLight = Field(default_factory=RoomLight)


# --------------------------------------------------------------------------------------------
# Light source
# --------------------------------------------------------------------------------------------
class ExcitationStep(_Model):
    wavelength_nm: float = Field(gt=200, lt=1100)
    power_mW: float = Field(1.0, ge=0)
    exposure_ms: float = Field(100.0, gt=0)
    enabled: bool = True


class BeamSpec(_Model):
    profile: Literal["gaussian", "flat_top"] = "gaussian"
    aperture_diameter_mm: float = Field(8.0, gt=0)          # beam diameter at the exit (1/e^2 for gaussian)
    divergence_half_angle_deg: float = Field(6.0, ge=0, lt=80)


class LightSourceSpec(_Model):
    id: str
    name: str
    kind: Literal["tunable_lamp", "led_array", "laser"] = "tunable_lamp"
    position_mm: Vec3 = (-180.0, 220.0, 0.0)
    target_mm: Vec3 = (0.0, 0.0, 0.0)
    roll_deg: float = 0.0
    beam: BeamSpec = Field(default_factory=BeamSpec)
    line_fwhm_nm: float = Field(10.0, gt=0)       # monochromator bandpass / LED FWHM / laser linewidth
    lamp_curve: Literal["flat", "xenon"] = "xenon"  # spectral power shape (tunable_lamp only)
    excitation_filter_od: float = Field(4.0, ge=0)  # out-of-band rejection of the clean-up filter
    power_drift_pct: float = Field(0.0, ge=0)        # 1-sigma step-to-step power fluctuation
    sequence: list[ExcitationStep] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------------
class LensSpec(_Model):
    focal_length_mm: float = Field(35.0, gt=0)
    f_number: float = Field(4.0, gt=0.5)
    focus_distance_mm: float = Field(300.0, gt=0)
    transmission: float = Field(0.9, gt=0, le=1)
    axial_chromatic_mm_per_100nm: float = Field(0.05, ge=0)   # focal shift per 100 nm from 550 nm


class SensorSpec(_Model):
    width_px: int = Field(348, ge=8, le=4096)
    height_px: int = Field(256, ge=8, le=4096)
    pixel_pitch_um: float = Field(6.45, gt=0)   # native pitch; effective pitch = pitch * binning
    binning: int = Field(4, ge=1, le=16)
    qe_curve: Literal["ccd_bv", "ccd_uv", "scmos", "flat"] = "ccd_bv"
    full_well_e: float = Field(60_000.0, gt=0)   # per output (binned) pixel
    read_noise_e: float = Field(8.0, ge=0)
    dark_current_e_per_s: float = Field(0.5, ge=0)
    gain_e_per_adu: float = Field(1.5, gt=0)
    offset_adu: float = Field(100.0, ge=0)
    bit_depth: Literal[12, 14, 16] = 16
    prnu_pct: float = Field(1.0, ge=0)          # photo-response non-uniformity (1-sigma, %)
    dsnu_e: float = Field(2.0, ge=0)            # dark-signal non-uniformity (1-sigma, e-)
    hot_pixel_fraction: float = Field(1e-4, ge=0, le=0.1)


class SpectralFilterSpec(_Model):
    """Emission-side spectral selection. ``lctf``: liquid-crystal tunable filter (Nuance-like),
    ``filter_wheel``: interference band-pass filters (angle-dependent blue shift).
    ``band_start_policy='excitation_offset'`` starts the band list at ``max(band_start_nm,
    excitation + start_offset_nm)`` (the real Nuance protocol for 415/430 nm excitation)."""

    kind: Literal["lctf", "filter_wheel"] = "lctf"
    band_start_nm: float = 420.0
    band_step_nm: float = Field(10.0, gt=0)
    n_bands: int = Field(31, ge=1, le=512)
    band_start_policy: Literal["fixed", "excitation_offset"] = "excitation_offset"
    start_offset_nm: float = 20.0
    fwhm_nm: float = Field(20.0, gt=0)
    peak_transmission_curve: Literal["lctf_vis", "flat"] = "lctf_vis"
    interference_n_eff: float = Field(2.0, gt=1.0)
    emission_longpass_nm: float | None = 415.0
    longpass_od: float = Field(5.0, ge=0)


class CameraSpec(_Model):
    id: str
    name: str
    position_mm: Vec3 = (0.0, 320.0, 0.0)
    target_mm: Vec3 = (0.0, 0.0, 0.0)
    roll_deg: float = 0.0
    lens: LensSpec = Field(default_factory=LensSpec)
    sensor: SensorSpec = Field(default_factory=SensorSpec)
    filter: SpectralFilterSpec = Field(default_factory=SpectralFilterSpec)


# --------------------------------------------------------------------------------------------
# Acquisition / physics switches
# --------------------------------------------------------------------------------------------
class PhysicsToggles(_Model):
    """Every effect can be switched off (for teaching / ablation). All default ON except
    ``indirect_bounce`` (costly) and ``room_light`` (controlled by ``Background.room_light``)."""

    fluorescence: bool = True
    absorption: bool = True              # dye + background absorption (inner filter / reabsorption)
    scattering: bool = True              # volume multiple scattering (Monte Carlo transport)
    elastic_leak: bool = True            # reflected / backscattered excitation through the filters
    raman: bool = True
    fresnel: bool = True                 # interface reflection / refraction / TIR
    specular_glints: bool = True
    shadows: bool = True
    subsurface_spread: bool = True       # lateral light spread under the surface
    indirect_bounce: bool = False        # one diffuse bounce between objects / background
    secondary_emission: bool = True      # re-emission of reabsorbed fluorescence
    quenching: bool = True
    fret: bool = True
    photobleaching: bool = True
    power_drift: bool = True
    defocus: bool = True
    diffraction: bool = True
    chromatic_aberration: bool = True
    vignetting: bool = True
    filter_angle_shift: bool = True
    shot_noise: bool = True
    read_noise: bool = True
    dark_current: bool = True
    fixed_pattern_noise: bool = True
    saturation: bool = True
    quantization: bool = True


class AcquisitionSettings(_Model):
    mode: Literal["sample", "dark", "white_reference"] = "sample"
    seed: int = 0
    mc_photons: int = Field(200_000, ge=1_000, le=50_000_000)   # Monte Carlo quality per bake
    preview_scale: float = Field(0.25, gt=0, le=1)
    physics: PhysicsToggles = Field(default_factory=PhysicsToggles)


# --------------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------------
class Experiment(_Model):
    schema_version: str = SCHEMA_VERSION
    name: str = "Untitled experiment"
    description: str = ""
    fluorophores: list[FluorophoreSpec] = Field(default_factory=list)   # added to / override builtins
    materials: list[MaterialSpec] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    background: Background = Field(default_factory=Background)
    sources: list[LightSourceSpec] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)
    active_source_id: str | None = None
    active_camera_id: str | None = None
    acquisition: AcquisitionSettings = Field(default_factory=AcquisitionSettings)

    @model_validator(mode="after")
    def _check_references(self) -> "Experiment":
        mats = {m.id for m in self.materials}
        for o in self.objects:
            if o.material_id not in mats:
                raise ValueError(f"object '{o.id}' references unknown material '{o.material_id}'")
            if o.vessel_material_id is not None and o.vessel_material_id not in mats:
                raise ValueError(f"object '{o.id}' references unknown vessel material "
                                 f"'{o.vessel_material_id}'")
        if self.background.kind == "plane" and self.background.material_id not in mats:
            raise ValueError(f"background references unknown material '{self.background.material_id}'")
        for label, ids, active in (("source", [s.id for s in self.sources], self.active_source_id),
                                   ("camera", [c.id for c in self.cameras], self.active_camera_id)):
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {label} ids")
            if active is not None and active not in ids:
                raise ValueError(f"active_{label}_id '{active}' not found")
        for coll, label in ((self.materials, "material"), (self.objects, "object")):
            ids = [x.id for x in coll]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {label} ids")
        return self

    # Convenience accessors -----------------------------------------------------------------
    def material(self, material_id: str) -> MaterialSpec:
        for m in self.materials:
            if m.id == material_id:
                return m
        raise KeyError(material_id)

    def source(self) -> LightSourceSpec:
        sid = self.active_source_id or (self.sources[0].id if self.sources else None)
        for s in self.sources:
            if s.id == sid:
                return s
        raise ValueError("experiment has no light source")

    def camera(self) -> CameraSpec:
        cid = self.active_camera_id or (self.cameras[0].id if self.cameras else None)
        for c in self.cameras:
            if c.id == cid:
                return c
        raise ValueError("experiment has no camera")
