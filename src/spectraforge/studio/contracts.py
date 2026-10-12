"""Internal engine contracts for SpectralForge Studio — the data passed between engine stages.

Pipeline (one excitation step)::

    Experiment ──trace_scene()──► HitBuffer            (tracer.py      — geometry & ray optics)
    MaterialSpec ─material_optics()─► MaterialOptics   (optics.py      — absorption/scattering spectra)
    (HitBuffer layer, MaterialOptics) ─TransportModel.kernels()─► TransportKernels   (transport.py)
    radiance per pixel per emission wavelength ──sensor.expose()──► electrons ─► ADU   (sensor.py)
    everything ──capture()──► CaptureResult            (capture.py     — orchestration)

Radiometric definitions (all modules MUST use these):

* ``E``  — excitation irradiance on a surface point, mW/mm², *outside* the medium, measured on the
  surface (includes the incidence cosine), *before* Fresnel transmission.
* Fluorescence radiance leaving a surface toward the camera, outside the medium:
  ``L(λem) = E · Σ_k μa_k(λex) · Φ_k · e_k(λem) · G(λem)``  in mW/(mm²·sr·nm), where ``e_k`` is
  the area-normalised emission density (1/nm), ``μa_k`` the dye's absorption coefficient (1/mm)
  at the (line-shape-averaged) excitation, and ``G`` (mm/sr) the transport kernel which contains
  *everything else*: Fresnel in/out, refraction, the 1/n² radiance factor, 1/(4π), depth
  integration, excitation attenuation and emission re-absorption along the paths.
  Clear medium closed form (reference implementation, ``transport.clear_medium_kernel``)::

      G = T_in·T_out / (4π · n² · cosθi') · (1 − exp(−κ·L)) / κ,   κ = μt_ex·cosθo'/cosθi' + μt_em

  with θi', θo' the refracted internal angles of the excitation and view rays, ``L`` the view
  chord inside the object and μt = μa + μs.
* Elastic (excitation-wavelength) diffuse remission radiance: ``L_el = E · ρ`` with ``ρ`` in 1/sr
  (BRDF·cos-free radiance factor, excludes the specular glint which the tracer provides).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

#: Fine internal spectral grid used by every module (nm). Band signals are integrals over it.
SPECTRAL_GRID_NM = np.arange(250.0, 900.0 + 1e-9, 1.0)

#: Layers tracked per camera ray (primary surface + up to 3 hits behind clear objects).
MAX_LAYERS = 4

ProgressFn = Callable[[float, str], None]     # (fraction 0..1, message)


@dataclass
class MaterialOptics:
    """Optical properties of one uniform material on ``wavelengths_nm`` (the fine grid)."""

    material_id: str
    wavelengths_nm: np.ndarray            # (W,)
    mua_per_mm: np.ndarray                # (W,) total absorption: dyes + background absorber
    mua_dyes_per_mm: dict[str, np.ndarray]  # fluorophore name -> (W,) its absorption coefficient
    mus_per_mm: np.ndarray                # (W,) scattering coefficient (NOT reduced); 0 if clear
    g: float                              # Henyey-Greenstein anisotropy
    n: float                              # refractive index
    clear: bool                           # True -> no volume scattering (closed-form transport)
    quantum_yields: dict[str, float]      # effective Φ per dye after self-quenching
    emission_density: dict[str, np.ndarray]  # dye -> (W,) emission shape normalised to ∫ = 1 (1/nm)

    def interp(self, arr: np.ndarray, lam_nm) -> np.ndarray:
        return np.interp(lam_nm, self.wavelengths_nm, arr)

    @property
    def mut_per_mm(self) -> np.ndarray:
        return self.mua_per_mm + self.mus_per_mm

    def cache_key(self) -> str:
        """Stable hash of the transport-relevant properties (for Monte Carlo bake caching)."""
        import hashlib
        h = hashlib.sha1()
        for a in (self.mua_per_mm, self.mus_per_mm):
            h.update(np.ascontiguousarray(np.round(a, 9)).tobytes())
        h.update(f"{self.g:.6f}|{self.n:.6f}|{self.clear}".encode())
        return h.hexdigest()[:16]


@dataclass
class HitBuffer:
    """Per-pixel ray-optics result for one camera at one resolution. ``P = width*height`` pixels
    in row-major order (row 0 = top of the image). Arrays with a leading ``K`` axis are per
    *layer*: layer 0 is the first surface the camera ray hits; layer j>0 is the next surface
    reached after the ray (refracted) exits a CLEAR object of layer j-1. A turbid material, the
    background slab, or ``MAX_LAYERS`` terminates the chain. Unused layers have ``material=-1``."""

    width: int
    height: int
    # ---- per layer (K, P) --------------------------------------------------------------
    material: np.ndarray            # int, index into experiment.materials, -1 = none
    object_index: np.ndarray        # int, index into experiment.objects, -2 = background, -1 = none
    point_mm: np.ndarray            # (K, P, 3) entry point of the view ray into the layer
    normal: np.ndarray              # (K, P, 3) outward unit normal at the entry point
    path_mm: np.ndarray             # view-ray chord inside the layer's medium (background: thickness)
    cos_view: np.ndarray            # EXTERNAL cosine of the view direction to the outward normal
    cos_view_internal: np.ndarray   # cos of refracted view ray vs inward normal (cosθo')
    cos_light_internal: np.ndarray  # cos of refracted excitation ray vs inward normal (cosθi')
    cos_light: np.ndarray           # external incidence cosine of the excitation at the entry point
    thickness_mm: np.ndarray        # medium thickness along the inward normal at the entry point
                                    #   (distance to the far side; background: slab thickness)
    T_view: np.ndarray              # cumulative Fresnel transmittance of all interfaces the view
                                    #   ray crossed BEFORE this layer's entry interface (1 for layer 0).
                                    #   The entry interface itself is inside the transport kernel G.
    T_light: np.ndarray             # same for the shadow ray source -> entry point: product of the
                                    #   Fresnel T of clear interfaces crossed BEFORE the entry
                                    #   (excluding the entry), 0 if occluded by a non-clear object
    n_outside: np.ndarray           # refractive index of the medium outside the entry interface
                                    #   (1.0 for layer 0, the previous clear layer's n behind glass)
    irradiance_per_mW: np.ndarray   # E per 1 mW source power at the entry point (mW/mm² per mW):
                                    #   beam profile × inverse-square × cos incidence, before Fresnel
    light_clear_path_mm: np.ndarray  # (K, P, M) path length of the shadow ray through each clear
                                    #   material m (M = len(experiment.materials)), for Beer-Lambert
    # ---- per pixel (P,) ----------------------------------------------------------------
    depth_mm: np.ndarray            # camera -> first hit distance along the ray (inf if none)
    glint_per_mW: np.ndarray        # specular radiance factor of the source at the first interface
                                    #   (mW/(mm²·sr) per mW source, excitation wavelength, pre-filter)
    pixel_footprint_mm: np.ndarray  # projected pixel side length on the first surface
    view_dir: np.ndarray            # (P, 3) unit vector from the first hit toward the camera
    ray_dir: np.ndarray             # (P, 3) camera ray direction (world)
    field_angle_rad: np.ndarray     # angle of the chief ray to the optical axis (vignetting, filter)
    # ---- frame-level ------------------------------------------------------------------
    focus_depth_mm: float = 0.0     # distance of the plane of focus along the optical axis
    info: dict = field(default_factory=dict)
    # ---- volumetric samples for CLEAR layers (K, P, Q), optional ------------------------
    # Exact single-pass fluorescence of a clear medium for ARBITRARY illumination geometry (beam
    # entering through a different face than the camera looks through, e.g. a laser crossing a
    # cuvette at 90°). Q quadrature points along the view chord inside each clear layer; for each,
    # the excitation path is traced back to the source through its actual entry face.
    #   L_out(λem) = T_out_view/n² · Σ_q w_q · F_q · Σ_k μa_k Φ_k e_k(λem)/(4π)
    #                · exp(−μt(λex)·l_q) · exp(−μt(λem)·s_q)          [× P_source, × T_view prior]
    # Turbid layers and the background leave these at 0 / NaN and use TransportKernels as before.
    vol_s_mm: np.ndarray | None = None          # distance from the view-entry point along the chord
    vol_weight_mm: np.ndarray | None = None     # quadrature weight (ds), 0 for unused samples
    vol_fluence_per_mW: np.ndarray | None = None  # excitation fluence rate at the sample per 1 mW,
                                                #   just inside the light-entry face: beam intensity
                                                #   perpendicular to propagation × Fresnel T of the
                                                #   light-entry face × T of prior clear occluders ×
                                                #   refraction beam-compression cosθ_ext/cosθ_int
                                                #   (already included — do not re-apply);
                                                #   0 if occluded / outside the beam
    vol_light_path_mm: np.ndarray | None = None  # excitation path length INSIDE this layer's medium
                                                #   from its light-entry face to the sample
    vol_T_out: np.ndarray | None = None          # (K, P) Fresnel T leaving the medium toward the
                                                #   camera through the view-entry face (0 on TIR)

    @property
    def n_pixels(self) -> int:
        return self.width * self.height


@dataclass
class TransportKernels:
    """Transport factors for one material layer at one excitation, for a set of pixels."""

    fluorescence_G: np.ndarray      # (P_sel, W) mm/sr on the fine emission grid (see module doc)
    elastic_rho: np.ndarray         # (P_sel,) 1/sr diffuse remission at the excitation wavelength
    radial_r_mm: np.ndarray | None = None       # subsurface-spread kernel radius grid
    radial_profile: np.ndarray | None = None    # (W_r, n_r) normalised (∫ 2πr p dr = 1) per node λ
    radial_lambda_nm: np.ndarray | None = None  # node wavelengths of the radial profile
    mean_depth_mm: np.ndarray | None = None     # (P_sel,) fluorescence-weighted mean origin depth


class TransportModel(Protocol):
    def kernels(self, optics: MaterialOptics, ex_nm: float, cos_in: np.ndarray, cos_out: np.ndarray,
                thickness_mm: np.ndarray, n_outside: np.ndarray | float = 1.0,
                path_mm: np.ndarray | None = None) -> TransportKernels:
        """``cos_in`` / ``cos_out``: EXTERNAL cosines of the excitation and view directions to the
        surface normal (the kernel refracts them itself using ``n_outside`` and ``optics.n``);
        ``thickness_mm``: geometric thickness of the medium along the inward normal (np.inf for
        semi-infinite). ``path_mm``: exact view-ray chord inside the medium (from the tracer);
        clear media use it as ``L`` in the closed form (turbid media use the slab ``thickness_mm``;
        if ``path_mm`` is None, ``L = thickness_mm / cosθo'``). G includes the entry-interface Fresnel transmission in and out, internal
        reflections at the boundaries, refraction and the radiance n² factor (relative to air).
        """
        ...


@dataclass
class CaptureResult:
    """One capture (all enabled excitation steps) from one camera."""

    experiment_json: str                   # the exact experiment that produced this
    mode: str                              # sample | dark | white_reference
    width: int
    height: int
    excitations_nm: list[float]
    emission_nm: dict[float, np.ndarray]   # ex -> (B,) band centres
    adu: dict[float, np.ndarray]           # ex -> (H, W, B) float32 raw counts (incl. offset)
    radiance: dict[float, np.ndarray]      # ex -> (H, W, B) float32 noise-free band radiance
                                           #   mW/(mm²·sr) integrated over each band's passband
    electrons_mean: dict[float, np.ndarray]  # ex -> (H, W, B) float32 expected signal electrons
    exposure_ms: dict[float, float]
    power_mW: dict[float, float]           # delivered (drifted) power per step
    ground_truth: dict[str, np.ndarray]    # per-pixel maps, see capture.GROUND_TRUTH_KEYS
    warnings: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    engine_version: str = "studio-1.0"
    # ---- v1.1: layered blocks (3D Monte Carlo) --------------------------------------------
    blocks: list[dict] = field(default_factory=list)
    #   [{"object_id", "layer_names", "footprint_mm", "world_from_local" (4x4 list)}]
    volumes: dict[str, dict] = field(default_factory=dict)
    #   object_id -> {"grid": VoxelGrid, "layer_names": [...],
    #                 "fluence": {ex: (nx, nz, L, S)}, "absorbed": {ex: (nx, nz, L, S)},
    #                 "source": {(ex, channel): (nx, nz, L, S)}, "units": {kind: str}}


# ============================================================================================
# 3D layered-block Monte Carlo (transport3d) — contracts
# ============================================================================================
# A ``layered_block`` object is solved by an explicit 3D photon-transport simulation instead of
# the per-pixel transport kernel G. All quantities below are in the block's LOCAL frame: origin at
# the centre of the bottom face, x ∈ [−X/2, X/2], z ∈ [−Z/2, Z/2], y up. Layer i (0 = bottom) is
# bounded by the planes y = h_{i−1}(x,z) and y = h_i(x,z) with h_i = a_i + b_i·x + c_i·z
# (h_{−1} ≡ 0) and by the four vertical side planes. Outside the block is air (n = 1).
#
# Two passes per capture:
#   1. excitation pass (one per excitation step): photons from the source beam enter the block,
#      scatter / refract / are absorbed; tallies = absorbed power and fluence per voxel, plus the
#      elastic (excitation-wavelength) radiance leaving each face toward the camera.
#   2. emission pass (one per emission wavelength node): photons are launched isotropically from
#      the fluorescence SOURCE volumes (built by capture.py from the absorbed power, quantum
#      yields, FRET, quenching and bleaching) and carry one weight per source channel; tallies =
#      radiance leaving each face toward the camera per source channel.
# Voxels are LAYER-CONFORMING: (ix, iz, layer, sub) with ``sublayers`` cells across each layer's
# local thickness, so thin and wedge-shaped layers keep full depth resolution.

@dataclass
class BlockGeometry:
    """Planar-interface description of one layered block (local frame)."""

    object_index: int                       # index into experiment.objects
    footprint_mm: tuple[float, float]       # (X, Z)
    layer_names: list[str]
    layer_material: list[int]               # index into experiment.materials, bottom → top
    plane_coeffs: np.ndarray                # (L, 3) [a, b, c] of the TOP interface of each layer
    world_from_local: np.ndarray            # (4, 4) homogeneous transform (rotation + translation)

    def top_height(self, layer: int, x, z):
        a, b, c = self.plane_coeffs[layer]
        return a + b * np.asarray(x) + c * np.asarray(z)


@dataclass
class VoxelGrid:
    """Layer-conforming tally grid."""

    nx: int
    nz: int
    n_layers: int
    sublayers: int
    x_edges_mm: np.ndarray                  # (nx+1,)
    z_edges_mm: np.ndarray                  # (nz+1,)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.nx, self.nz, self.n_layers, self.sublayers)


@dataclass
class FaceGrid:
    """Exit-radiance tally grid on one block face. Faces: 'top' (the top interface plane), '+x',
    '-x', '+z', '-z' (vertical sides). (u, v) are face-local coordinates: for 'top' u = x, v = z;
    for ±x faces u = z, v = y; for ±z faces u = x, v = y (all mm, local frame)."""

    face: str
    u_edges_mm: np.ndarray
    v_edges_mm: np.ndarray


@dataclass
class BlockLaunch:
    """Excitation photons entering a block for one excitation step (sampled by capture.py from the
    source beam, the block's faces and shadowing by other objects)."""

    ex_nm: float
    points_local: np.ndarray                # (N, 3) points on the block surface
    dirs_local: np.ndarray                  # (N, 3) unit propagation directions (into the block)
    power_mW: np.ndarray                    # (N,) power carried by each launch sample (sum =
                                            #   power incident on the block, before Fresnel)
    line_weights: np.ndarray | None = None  # optional (n_lambda,) excitation line shape samples


@dataclass
class RegionOptics:
    """Optics of every layer at ONE wavelength (arrays of length L)."""

    lambda_nm: float
    mua_per_mm: np.ndarray                  # (L,)
    mus_per_mm: np.ndarray                  # (L,)
    g: np.ndarray                           # (L,)
    n: np.ndarray                           # (L,)


@dataclass
class ExcitationTallies:
    ex_nm: float
    grid: VoxelGrid
    absorbed_mW: np.ndarray                 # (nx, nz, L, S) power absorbed per voxel (all absorbers)
    fluence_mW_mm2: np.ndarray              # (nx, nz, L, S) fluence rate (track-length estimator)
    elastic_radiance: dict[str, np.ndarray]  # face -> (nu, nv) mW/(mm²·sr) toward the camera
    faces: dict[str, FaceGrid]
    diffuse_reflectance: float              # total remitted / incident (diagnostic)
    transmittance: float                    # left through bottom / sides (diagnostic)
    absorbed_fraction: float                # (diagnostic; R + T + A ≈ 1 − specular)
    n_photons: int = 0
    info: dict = field(default_factory=dict)


@dataclass
class EmissionTallies:
    em_nodes_nm: np.ndarray                 # (Ne,)
    channels: list[str]                     # source channel labels, e.g. "365|tissue|NADH"
    radiance: dict[str, np.ndarray]         # face -> (nu, nv, Ne, C) radiance toward the camera
                                            #   per unit emitted spectral density: mW/(mm²·sr) per
                                            #   (mW/nm) → multiply by e_k(λ) [1/nm] and the channel
                                            #   source power to get mW/(mm²·sr·nm)
    faces: dict[str, FaceGrid]
    origin_depth_mm: dict[str, np.ndarray] | None = None  # face -> (nu, nv) mean source depth
                                            #   below the top surface of escaping light (diagnostic)
    origin_layer_fraction: dict[str, np.ndarray] | None = None  # face -> (nu, nv, L)
    n_photons: int = 0
    info: dict = field(default_factory=dict)
