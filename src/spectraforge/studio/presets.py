"""Built-in fluorophores, materials and experiment presets for SpectralForge Studio.

Python is the single source of truth: the web app fetches these from ``GET /api/presets`` (and a
bundled copy is generated into ``apps/studio-web/src/presets/`` by ``scripts/export_presets.py``).
"""
from __future__ import annotations

from spectraforge.library import load_builtin_library
from spectraforge.studio.schema import (
    AbsorberSpec,
    Background,
    BeamSpec,
    BoxShape,
    CameraSpec,
    CuvetteShape,
    CylinderShape,
    DropletShape,
    DyeAmount,
    ExcitationStep,
    Experiment,
    FluorophoreSpec,
    LensSpec,
    LightSourceSpec,
    MaterialSpec,
    PetriDishShape,
    ScatteringSpec,
    SceneObject,
    SensorSpec,
    SpectralFilterSpec,
    SphereShape,
    Transform,
)

# Display colours for the builtin dyes (approximate perceived emission colour).
_DYE_COLORS = {
    "collagen": "#7f6bff", "elastin": "#6f7dff", "NADH": "#4f8dff", "FAD": "#57e08a",
    "lipofuscin": "#ffb347", "porphyrin": "#ff4f6a", "DAPI": "#4d7cff", "EBFP2": "#5a86ff",
    "EGFP": "#3df07a", "fluorescein": "#5cff6a", "mCherry": "#ff5a5a", "mKate2": "#ff3d6e",
    "optical_brightener": "#6aa8ff", "chlorophyll_a": "#d42f2f",
}

# Fluorophores the studio adds on top of the 2D library (real lab confounds / lichen pigments).
_EXTRA_FLUOROPHORES = [
    FluorophoreSpec(name="optical_brightener", ex_peak_nm=350, ex_fwhm_nm=45, em_peak_nm=435,
                    em_fwhm_nm=55, quantum_yield=0.7, extinction=0.9, em_skew=0.4),
    FluorophoreSpec(name="chlorophyll_a", ex_peak_nm=430, ex_fwhm_nm=40, em_peak_nm=675,
                    em_fwhm_nm=25, quantum_yield=0.3, extinction=1.6, em_skew=0.3, vibronic=0.25),
]

# Typical photobleaching quantum yields (order of magnitude): fluorescein ~3e-5, EGFP ~1e-5,
# endogenous fluorophores ~1e-6 .. 1e-5.
_BLEACH = {"fluorescein": 3e-5, "EGFP": 1e-5, "mCherry": 5e-6, "mKate2": 5e-6, "DAPI": 2e-6,
           "NADH": 5e-6, "FAD": 5e-6}


# Peak molar extinction coefficients (M^-1 cm^-1) of the 2-D library dyes replaced by literature values
# (relative extinction = eps / 55 000): fluorescein dianion pH 9, 490 nm 76 900 (Sjoback 1995); mCherry
# 72 000 and mKate2 62 500 (Shaner 2004; Shcherbo 2007); DAPI bound to DNA ~27 000 (Kapuscinski 1995);
# EBFP2 31 000 (Ai 2007); NADH 6 220 at 340 nm; FAD 11 300 at 450 nm (Lakowicz 2006).
_EXTINCTION_MCM = {"fluorescein": 76_900.0, "mCherry": 72_000.0, "mKate2": 62_500.0, "DAPI": 27_000.0,
                   "EBFP2": 31_000.0, "NADH": 6_220.0, "FAD": 11_300.0}


# Emission-shape refinements over the 2D library's pure Gaussians (studio only): real fluorescein
# keeps ~35 % of its peak at 550 nm (red tail + vibronic shoulder); a pure Gaussian gives 4 %.
_EM_SHAPE = {"fluorescein": (0.4, 0.2)}


def builtin_fluorophores() -> list[FluorophoreSpec]:
    from spectraforge.studio.schema import EXTINCTION_REFERENCE_M_CM
    out = []
    for f in load_builtin_library().values():
        ext = _EXTINCTION_MCM.get(f.name, 0.0) / EXTINCTION_REFERENCE_M_CM or f.extinction
        out.append(FluorophoreSpec(
            name=f.name, ex_peak_nm=f.ex_peak_nm, ex_fwhm_nm=f.ex_fwhm_nm, em_peak_nm=f.em_peak_nm,
            em_fwhm_nm=f.em_fwhm_nm, quantum_yield=f.quantum_yield, extinction=ext,
            em_skew=_EM_SHAPE.get(f.name, (f.em_skew, f.vibronic))[0],
            vibronic=_EM_SHAPE.get(f.name, (f.em_skew, f.vibronic))[1],
            bleach_quantum_yield=_BLEACH.get(f.name, 1e-6),
            display_color=_DYE_COLORS.get(f.name)))
    for f in _EXTRA_FLUOROPHORES:
        out.append(f.model_copy(update={"display_color": _DYE_COLORS.get(f.name)}))
    return out


def _mat(id_, name, color, dyes=(), **kw) -> MaterialSpec:
    return MaterialSpec(id=id_, name=name, display_color=color,
                        dyes=[DyeAmount(fluorophore=f, concentration_uM=c) for f, c in dyes], **kw)


def builtin_materials() -> list[MaterialSpec]:
    """Base + layer materials + the literature tissue library (:mod:`presets_tissue`)."""
    from spectraforge.studio.presets_tissue import tissue_materials

    out = {m.id: m for m in _base_materials() + _layered_materials()}
    for m in tissue_materials():
        out.setdefault(m.id, m)
    return list(out.values())


def _layered_materials() -> list[MaterialSpec]:
    """Layer materials for layered blocks (substrate / fluorescent layer / cover)."""
    water = AbsorberSpec(preset="water", scale=1.0)
    return [
        _mat("tissue_substrate", "Tissue substrate (turbid, weakly fluorescent)", "#c98f86",
             dyes=[("NADH", 8.0), ("collagen", 80.0)], refractive_index=1.38,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=1.5, power_b=1.2, g=0.85),
             background_absorption=AbsorberSpec(preset="hemoglobin", scale=0.05), surface_roughness=0.3),
        _mat("tissue_fluor_layer", "Tissue layer (NADH/FAD-rich)", "#f2b35c",
             dyes=[("NADH", 120.0), ("FAD", 60.0)], refractive_index=1.38,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=1.2, power_b=1.2, g=0.85),
             background_absorption=AbsorberSpec(preset="hemoglobin", scale=0.02), surface_roughness=0.3),
        _mat("fluorescein_agarose_layer", "Fluorescein agarose layer (5 µM)", "#6cff7a",
             dyes=[("fluorescein", 5.0)], refractive_index=1.335,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=0.3, power_b=1.0, g=0.9),
             background_absorption=water, aqueous=True, surface_roughness=0.05),
        _mat("collagen_cover", "Collagen cover (dense, autofluorescent)", "#b8a9ff",
             dyes=[("collagen", 600.0), ("elastin", 60.0)], refractive_index=1.41,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=2.0, power_b=1.2, g=0.9),
             background_absorption=water, surface_roughness=0.25),
        _mat("intralipid_cover", "Turbid cover (intralipid-like, non-fluorescent)", "#f3f1ea",
             refractive_index=1.36,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=1.5, power_b=1.0, g=0.8),
             background_absorption=water, aqueous=True, surface_roughness=0.1),
        _mat("absorbing_cover", "Absorbing cover (blood-tinted, turbid)", "#a33a3a",
             refractive_index=1.38,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=1.0, power_b=1.2, g=0.85),
             background_absorption=AbsorberSpec(preset="hemoglobin", scale=0.4), surface_roughness=0.2),
        _mat("clear_cover", "Clear cover (transparent gel)", "#d8f0ff", refractive_index=1.40,
             background_absorption=water, surface_roughness=0.02),
    ]


def _base_materials() -> list[MaterialSpec]:
    water = AbsorberSpec(preset="water", scale=1.0)
    return [
        # ---- optics / stage --------------------------------------------------------------
        _mat("glass", "Borosilicate glass", "#cfe8ff", refractive_index=1.52, surface_roughness=0.01),
        _mat("black_anodized", "Black anodized aluminium", "#1a1b1f", refractive_index=1.5,
             background_absorption=AbsorberSpec(preset="carbon_black", scale=40.0),
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=2.0, power_b=0.5, g=0.6),
             surface_roughness=0.55),
        _mat("matte_grey", "Matte grey card (18%)", "#7b7d82", refractive_index=1.5,
             background_absorption=AbsorberSpec(preset="carbon_black", scale=1.2),
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=20.0, power_b=0.3, g=0.7),
             surface_roughness=0.7),
        _mat("white_paper", "White office paper (brightened)", "#f4f6ff",
             dyes=[("optical_brightener", 30.0)], refractive_index=1.5,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=25.0, power_b=1.0, g=0.8),
             background_absorption=AbsorberSpec(preset="carbon_black", scale=0.05),
             surface_roughness=0.65),
        _mat("spectralon", "Spectralon white standard (99%)", "#ffffff", refractive_index=1.35,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=120.0, power_b=0.1, g=0.7),
             surface_roughness=0.85),
        # ---- liquids ---------------------------------------------------------------------
        _mat("water", "Water", "#9fd3ff", refractive_index=1.333, background_absorption=water,
             aqueous=True, surface_roughness=0.0),
        _mat("fluorescein_10uM", "Fluorescein 10 µM (aq.)", "#5cff6a", dyes=[("fluorescein", 10.0)],
             refractive_index=1.333, background_absorption=water, aqueous=True, surface_roughness=0.0,
             quenching_ksv_per_uM=0.002),
        _mat("fluorescein_200uM", "Fluorescein 200 µM (aq., inner-filter)", "#3fd14f",
             dyes=[("fluorescein", 200.0)], refractive_index=1.333, background_absorption=water,
             aqueous=True, surface_roughness=0.0, quenching_ksv_per_uM=0.002),
        _mat("mcherry_solution", "mCherry 20 µM (aq.)", "#ff5a5a", dyes=[("mCherry", 20.0)],
             refractive_index=1.333, background_absorption=water, aqueous=True, surface_roughness=0.0),
        _mat("dapi_solution", "DAPI 15 µM (aq.)", "#4d7cff", dyes=[("DAPI", 15.0)],
             refractive_index=1.333, background_absorption=water, aqueous=True, surface_roughness=0.0),
        _mat("fret_egfp_mcherry", "EGFP→mCherry FRET pair", "#c7b04a",
             dyes=[("EGFP", 8.0), ("mCherry", 8.0)], refractive_index=1.34, background_absorption=water,
             aqueous=True, surface_roughness=0.0,
             fret=[{"donor": "EGFP", "acceptor": "mCherry", "k_per_uM": 0.15}]),
        # ---- turbid / biological --------------------------------------------------------
        _mat("tissue_phantom", "Tissue phantom (intralipid + NADH/FAD)", "#ffb8a6",
             dyes=[("NADH", 100.0), ("FAD", 25.0), ("collagen", 300.0)], refractive_index=1.38,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=1.2, power_b=1.2, g=0.85),
             background_absorption=AbsorberSpec(preset="hemoglobin", scale=0.03), surface_roughness=0.25),
        _mat("collagen_gel", "Collagen gel", "#a89bff", dyes=[("collagen", 800.0), ("elastin", 100.0)],
             refractive_index=1.36,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=0.6, power_b=1.3, g=0.9),
             background_absorption=water, surface_roughness=0.2),
        _mat("egfp_agarose", "EGFP in agarose (1%)", "#3df07a", dyes=[("EGFP", 4.0)],
             refractive_index=1.335,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=0.25, power_b=1.0, g=0.9),
             background_absorption=water, aqueous=True, surface_roughness=0.05),
        _mat("lichen_green", "Lichen thallus (green, chlorophyll-rich)", "#5f8f3a",
             dyes=[("chlorophyll_a", 60.0), ("lipofuscin", 20.0), ("collagen", 150.0)],
             refractive_index=1.45,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=6.0, power_b=1.0, g=0.85),
             background_absorption=AbsorberSpec(preset="melanin", scale=0.4), surface_roughness=0.6),
        _mat("lichen_orange", "Lichen thallus (orange, parietin-like)", "#d9822b",
             dyes=[("lipofuscin", 80.0), ("porphyrin", 10.0), ("chlorophyll_a", 15.0)],
             refractive_index=1.45,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=8.0, power_b=1.0, g=0.85),
             background_absorption=AbsorberSpec(preset="melanin", scale=0.8), surface_roughness=0.6),
        _mat("lichen_grey", "Lichen thallus (grey, low pigment)", "#9aa093",
             dyes=[("collagen", 400.0), ("NADH", 70.0), ("chlorophyll_a", 8.0)], refractive_index=1.45,
             scattering=ScatteringSpec(model="power_law", mus_prime_500_per_mm=10.0, power_b=0.9, g=0.85),
             background_absorption=AbsorberSpec(preset="melanin", scale=0.2), surface_roughness=0.6),
    ]


# --------------------------------------------------------------------------------------------
# Hardware presets
# --------------------------------------------------------------------------------------------
_TLS_SEQUENCE = [  # (nm, mW, exposure ms) — the lab's 8-line protocol, long exposures in the UV
    (310, 0.42, 4000), (325, 0.85, 2000), (340, 1.3, 1000), (365, 2.6, 250),
    (385, 3.4, 60), (400, 3.9, 50), (415, 4.2, 80), (430, 4.5, 80),
]


def nuance_camera(id_="cam_nuance", name="Nuance-class HSI camera", position=(0.0, 330.0, 0.0),
                  target=(0.0, 0.0, 0.0)) -> CameraSpec:
    return CameraSpec(id=id_, name=name, position_mm=position, target_mm=target,
                      lens=LensSpec(focal_length_mm=35.0, f_number=4.0, focus_distance_mm=330.0),
                      sensor=SensorSpec(), filter=SpectralFilterSpec())


def tunable_lamp(id_="src_tls", position=(-200.0, 240.0, 60.0), target=(0.0, 0.0, 0.0),
                 sequence=_TLS_SEQUENCE) -> LightSourceSpec:
    """Xe arc + monochromator.  ``excitation_filter_od = 6``: a single grating monochromator rejects
    ~1e-3..1e-4 of the out-of-band continuum, the excitation band-pass / short-pass clean-up filter stack
    another OD 2-3 (a nominal OD 4 left a visible-continuum leak at ~30 % of the lichen fluorescence behind
    the 415 nm long-pass LCTF, because the Xe visible output is 5x the 310 nm line power)."""
    return LightSourceSpec(
        id=id_, name="Tunable light source (Xe + monochromator)", kind="tunable_lamp",
        position_mm=position, target_mm=target, line_fwhm_nm=10.0, lamp_curve="xenon",
        beam=BeamSpec(profile="gaussian", aperture_diameter_mm=10.0, divergence_half_angle_deg=7.0),
        excitation_filter_od=6.0, power_drift_pct=0.8,
        sequence=[ExcitationStep(wavelength_nm=w, power_mW=p, exposure_ms=e) for w, p, e in sequence])


def _obj(id_, name, shape, material, pos, rot=(0.0, 0.0, 0.0), vessel=None) -> SceneObject:
    return SceneObject(id=id_, name=name, shape=shape, material_id=material,
                       transform=Transform(position_mm=pos, rotation_deg=rot), vessel_material_id=vessel)


def generic_macro_bench() -> Experiment:
    return Experiment(
        name="Generic macro bench",
        description="Top-down HSI camera, oblique tunable lamp, a mixed set of fluorescent samples "
                    "on a black anodized stage — including a brightened paper card as a confound.",
        materials=builtin_materials(),
        objects=[
            _obj("dish", "Petri dish — tissue phantom", PetriDishShape(inner_radius_mm=30.0, fill_height_mm=4.0),
                 "tissue_phantom", (-35.0, 0.0, -20.0), vessel="glass"),
            _obj("drop", "Fluorescein droplet", DropletShape(base_radius_mm=6.0, height_mm=3.5),
                 "fluorescein_10uM", (30.0, 0.0, -30.0)),
            _obj("bead", "EGFP agarose bead", SphereShape(radius_mm=7.0), "egfp_agarose", (35.0, 7.0, 15.0)),
            _obj("cuvette", "mCherry cuvette", CuvetteShape(), "mcherry_solution", (0.0, 0.0, 45.0),
                 vessel="glass"),
            _obj("card", "Brightened paper card", BoxShape(size_mm=(40.0, 0.3, 25.0)), "white_paper",
                 (-40.0, 0.15, 40.0), rot=(0.0, 18.0, 0.0)),
        ],
        background=Background(kind="plane", material_id="black_anodized"),
        # liquid-light-guide lamp (10-25 mW over the bench), exposures set so the brightest samples reach
        # ~20-30 k e- (full well 60 k e-)
        sources=[tunable_lamp(sequence=[(365, 10.0, 500), (405, 24.0, 1000), (450, 10.0, 400),
                                        (488, 6.0, 200), (532, 24.0, 1000), (587, 6.0, 250)])],
        cameras=[CameraSpec(id="cam_main", name="HSI camera", position_mm=(-12.0, 450.0, 6.0), target_mm=(-12.0, 0.0, 6.0),
                            # 25 mm lens at 450 mm: field of view 153 x 112 mm covers the whole bench; the focus
                            # plane sits at the mid-height of the samples (dish rim y~5, bead top y~14)
                            lens=LensSpec(focal_length_mm=25.0, f_number=4.0, focus_distance_mm=443.0),
                            filter=SpectralFilterSpec(band_start_nm=400.0, start_offset_nm=35.0,
                                                      n_bands=36, emission_longpass_nm=395.0))],
        active_source_id="src_tls", active_camera_id="cam_main",
    )


def mehsi_lab_replica() -> Experiment:
    return Experiment(
        name="ME-HSI lab replica (TLS + Nuance)",
        description="8-line tunable light source 310–430 nm and a Nuance-class LCTF camera "
                    "(348×256, 31 bands from 420 nm, 10 nm step), three lichen thalli on black stage.",
        materials=builtin_materials(),
        objects=[
            _obj("lichen_a", "Lichen A (green)", DropletShape(base_radius_mm=11.0, height_mm=2.5),
                 "lichen_green", (-22.0, 0.0, -10.0)),
            _obj("lichen_b", "Lichen B (orange)", CylinderShape(radius_mm=9.0, height_mm=2.0),
                 "lichen_orange", (18.0, 1.0, -14.0)),
            _obj("lichen_c", "Lichen C (grey)", BoxShape(size_mm=(16.0, 2.2, 12.0)), "lichen_grey",
                 (2.0, 1.1, 18.0), rot=(0.0, 25.0, 0.0)),
        ],
        background=Background(kind="plane", material_id="black_anodized"),
        sources=[tunable_lamp()],
        cameras=[nuance_camera()],
        active_source_id="src_tls", active_camera_id="cam_nuance",
    )


def led_multispectral() -> Experiment:
    led = LightSourceSpec(
        id="src_led", name="LED array (6 channels)", kind="led_array", position_mm=(0.0, 260.0, -110.0),
        target_mm=(0.0, 0.0, 0.0), line_fwhm_nm=18.0, lamp_curve="flat", excitation_filter_od=5.0,
        beam=BeamSpec(profile="flat_top", aperture_diameter_mm=40.0, divergence_half_angle_deg=14.0),
        # per-channel power and exposure chosen so the brightest sample reaches ~25-35 k e- (30 k e- well
        # x 2 binning: weak 405 / 525 lines get longer exposures, the strong blue channels shorter)
        sequence=[ExcitationStep(wavelength_nm=w, power_mW=p, exposure_ms=e)
                  for w, p, e in ((365, 90, 40), (405, 90, 360), (450, 80, 40), (470, 30, 40),
                                  (525, 70, 400), (590, 60, 110))])
    # A filter-wheel camera with a UV-block long-pass (400 nm, OD 5) and an emission-filter wheel whose
    # channels start 40 nm above the LED line in use ("excitation_offset": the operator only selects the
    # filters that do not transmit the excitation, exactly as a real LED + filter-wheel system is run).
    # 16 mm lens at 400 mm: 6.6 x 5.0 mm sensor -> 158 x 119 mm field of view covering the bench.
    cam = CameraSpec(id="cam_fw", name="Filter-wheel multispectral camera", position_mm=(-12.0, 400.0, 6.0),
                     target_mm=(-12.0, 0.0, 6.0),
                     lens=LensSpec(focal_length_mm=16.0, f_number=2.8, focus_distance_mm=393.0),
                     sensor=SensorSpec(width_px=512, height_px=384, binning=2, qe_curve="scmos",
                                       full_well_e=30_000.0, read_noise_e=1.6, gain_e_per_adu=0.5),
                     filter=SpectralFilterSpec(kind="filter_wheel", band_start_nm=420.0, band_step_nm=30.0,
                                               n_bands=10, band_start_policy="excitation_offset",
                                               start_offset_nm=40.0, fwhm_nm=25.0,
                                               peak_transmission_curve="flat", emission_longpass_nm=400.0))
    exp = generic_macro_bench()
    return exp.model_copy(update={"name": "LED multispectral bench", "sources": [led], "cameras": [cam],
                                  "active_source_id": "src_led", "active_camera_id": "cam_fw",
                                  "description": "Six-channel LED array next to a filter-wheel sCMOS camera."})


def laser_line() -> Experiment:
    """Classic fluorometer geometry: a horizontal laser beam crosses a row of cuvettes (in series, so
    each cuvette sees the beam transmitted by the previous one) and the camera looks sideways, 90 deg to
    the beam.  Order along the beam: weakest absorber first (EGFP->mCherry FRET pair), then fluorescein
    10 uM, then fluorescein 200 uM last.  Lines: 457 nm sits on the blue wing of the fluorescein band
    (eps ~ 3.5e3 M-1cm-1 in this model), where the 200 uM cuvette shows the textbook inner-filter gradient
    resolved across its 10 mm (1/e length ~6 mm); at 488 nm (band peak) the same cuvette absorbs everything
    within ~0.3 mm of the entrance wall (a thin bright sliver of front-face emission) while the 10 uM cuvette
    shows an exponential decay across the cell; 561 nm lights only the mCherry of the FRET pair, 640 nm
    nothing (control).  Beam: 3 mm (1/e^2) collimated Gaussian, 0.1 deg half-angle (0.5 mrad-class laser),
    so it stays inside the 10 mm cavity.  Camera: 50 mm f/2.8 at 440 mm, field of view ~68 mm x 50 mm covering
    the whole row (the beam axis is at y = 16 mm, mid-liquid).  NOTE: fluorescein absorption is a single
    Gaussian here (~0 at 405 nm, real ~3 % of peak)."""
    laser = LightSourceSpec(
        id="src_laser", name="Laser combiner (457/488/561/640)", kind="laser",
        position_mm=(-170.0, 16.0, 0.0), target_mm=(0.0, 16.0, 0.0), line_fwhm_nm=1.0, lamp_curve="flat",
        excitation_filter_od=6.0, beam=BeamSpec(profile="gaussian", aperture_diameter_mm=3.0,
                                                divergence_half_angle_deg=0.1),
        sequence=[ExcitationStep(wavelength_nm=w, power_mW=p, exposure_ms=e)
                  for w, p, e in ((457, 20.0, 100), (488, 1.0, 30), (561, 5.0, 60), (640, 5.0, 200))])
    cam = CameraSpec(id="cam_side", name="HSI camera (90° side view)", position_mm=(0.0, 20.0, 440.0),
                     target_mm=(0.0, 20.0, 0.0), lens=LensSpec(focal_length_mm=50.0, f_number=2.8,
                                                              focus_distance_mm=440.0),
                     filter=SpectralFilterSpec(band_start_nm=420.0, n_bands=34, start_offset_nm=25.0))
    return Experiment(
        name="Laser-line cuvettes (90° geometry)",
        description="Classic fluorometer geometry: a laser beam through a row of cuvettes, viewed at 90°.",
        materials=builtin_materials(),
        objects=[
            _obj("cuv_fret", "EGFP→mCherry FRET", CuvetteShape(), "fret_egfp_mcherry", (-16.0, 0.0, 0.0), vessel="glass"),
            _obj("cuv_fluo", "Fluorescein 10 µM", CuvetteShape(), "fluorescein_10uM", (0.0, 0.0, 0.0), vessel="glass"),
            _obj("cuv_high", "Fluorescein 200 µM (inner filter)", CuvetteShape(), "fluorescein_200uM",
                 (16.0, 0.0, 0.0), vessel="glass"),
        ],
        background=Background(kind="plane", material_id="black_anodized"),
        sources=[laser], cameras=[cam], active_source_id="src_laser", active_camera_id="cam_side",
    )


def _layer(name, material, t0, t1=None, axis="x"):
    from spectraforge.studio.schema import BlockLayer, ThicknessProfile
    prof = (ThicknessProfile(kind="uniform", thickness_mm=t0, end_thickness_mm=t0) if t1 is None else
            ThicknessProfile(kind="linear", thickness_mm=t0, end_thickness_mm=t1, axis=axis))
    return BlockLayer(name=name, material_id=material, thickness=prof)


def layered_wedge_phantom() -> Experiment:
    """Generic study scene: a fluorescent layer buried under a turbid cover whose thickness falls
    linearly from 2 mm (−x edge) to 0 (+x edge) — read the attenuation / spectral distortion of the
    buried fluorophore against the local cover thickness with the Explore analysis tools."""
    from spectraforge.studio.schema import LayeredBlockShape
    block = LayeredBlockShape(footprint_mm=(40.0, 40.0), layers=[
        _layer("substrate", "tissue_substrate", 4.0),
        _layer("fluorescent layer", "fluorescein_agarose_layer", 0.5),
        _layer("cover", "intralipid_cover", 2.0, 0.0, "x"),
    ])
    return Experiment(
        name="Layered wedge phantom",
        description="Fluorescein layer (0.5 mm) on a turbid substrate, under a turbid cover that thins "
                    "linearly from 2 mm to 0 across the block. Solved with the 3D layered Monte Carlo.",
        materials=builtin_materials(),
        objects=[_obj("block", "Layered block (cover wedge)", block, "intralipid_cover", (0.0, 0.0, 0.0))],
        background=Background(kind="plane", material_id="black_anodized"),
        sources=[tunable_lamp(target=(0.0, 3.0, 0.0), sequence=[
            (340, 1.3, 400), (380, 3.6, 120), (420, 4.4, 80), (450, 4.6, 60), (470, 4.7, 60),
            (490, 4.8, 60)])],
        cameras=[nuance_camera(target=(0.0, 3.0, 0.0))],
        active_source_id="src_tls", active_camera_id="cam_nuance",
    )


def tissue_under_collagen() -> Experiment:
    """The target scenario: an NADH/FAD tissue layer under a collagen cover of varying thickness,
    imaged with the lab's TLS (310–430 nm) + Nuance hardware."""
    from spectraforge.studio.schema import LayeredBlockShape
    block = LayeredBlockShape(footprint_mm=(40.0, 40.0), layers=[
        _layer("substrate", "tissue_substrate", 5.0),
        _layer("tissue", "tissue_fluor_layer", 1.0),
        _layer("collagen", "collagen_cover", 1.5, 0.05, "x"),
    ])
    exp = mehsi_lab_replica()
    return exp.model_copy(update={
        "name": "Tissue under collagen (wedge)",
        "description": "NADH/FAD tissue layer under a collagen cover thinning from 1.5 mm to 50 µm, "
                       "on a turbid substrate; lab TLS + Nuance hardware. 3D layered Monte Carlo.",
        "objects": [_obj("block", "Tissue / collagen block", block, "collagen_cover", (0.0, 0.0, 0.0))],
    })


def pencil_beam_penetration() -> Experiment:
    """Spatially resolved light transport: a 1 mm laser beam hits a tissue block under a uniform
    collagen cover at 45°. The excitation fluence inside spreads into the classic banana-shaped
    penetration volume and the surface fluorescence spot is broadened by subsurface scattering —
    inspect it with the cutaway light field (cut along z = 0, through the beam) and line profiles."""
    from spectraforge.studio.schema import LayeredBlockShape
    block = LayeredBlockShape(footprint_mm=(30.0, 30.0), layers=[
        _layer("substrate", "tissue_substrate", 5.0),
        _layer("tissue", "tissue_fluor_layer", 1.0),
        _layer("collagen", "collagen_cover", 0.5),
    ])
    laser = LightSourceSpec(
        id="src_pencil", name="Pencil-beam laser (405/450/488)", kind="laser",
        position_mm=(-140.0, 146.5, 0.0), target_mm=(0.0, 6.5, 0.0), line_fwhm_nm=1.0, lamp_curve="flat",
        excitation_filter_od=6.0,
        beam=BeamSpec(profile="gaussian", aperture_diameter_mm=1.0, divergence_half_angle_deg=0.1),
        sequence=[ExcitationStep(wavelength_nm=w, power_mW=2.0, exposure_ms=e)
                  for w, e in ((405, 100.0), (450, 50.0), (488, 50.0))])
    exp = mehsi_lab_replica()
    return exp.model_copy(update={
        "name": "Pencil-beam penetration (tissue + collagen)",
        "description": "1 mm laser beam at 45° into a tissue block under 0.5 mm of collagen: visualise "
                       "the banana-shaped 3D light field and the scattering-broadened fluorescence spot.",
        "objects": [_obj("block", "Tissue / collagen block", block, "collagen_cover", (0.0, 0.0, 0.0))],
        "sources": [laser], "active_source_id": "src_pencil",
        "cameras": [nuance_camera(target=(0.0, 6.5, 0.0))],
    })


PRESETS = {
    "generic_macro_bench": generic_macro_bench,
    "mehsi_lab_replica": mehsi_lab_replica,
    "led_multispectral": led_multispectral,
    "laser_line": laser_line,
    "layered_wedge_phantom": layered_wedge_phantom,
    "tissue_under_collagen": tissue_under_collagen,
    "pencil_beam_penetration": pencil_beam_penetration,
}


def _register_tissue_presets() -> None:
    from spectraforge.studio.presets_tissue import TISSUE_PRESETS

    PRESETS.update(TISSUE_PRESETS)


_register_tissue_presets()


def preset(name: str) -> Experiment:
    return PRESETS[name]()
