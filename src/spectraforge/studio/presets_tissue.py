"""Literature-parameterised tissue materials and tissue experiments (SpectralForge Studio v1.2).

Absorption comes from :class:`~spectraforge.studio.schema.TissueComposition` (Jacques 2013 eq. 1:
blood fraction ``B`` and saturation ``S``, water ``W``, lipid ``F``, melanosomes ``M``), scattering from
the Jacques 2013 two-component law (``mie_rayleigh``, eq. 2) with the cited ``a'``, ``f_ray``, ``b_mie``.
Jacques SL, Phys. Med. Biol. 58 (2013) R37, doi:10.1088/0031-9155/58/11/R37; the individual
Table-1 / Table-2 / Table-3 entries are cited per material below (table CSVs at
omlc.org/news/dec14/Jacques_PMB2013/).

Conventions / approximations
----------------------------
* ``a'`` is converted from the paper's 1/cm to the engine's 1/mm.
* ``g`` (HG anisotropy) is not tabulated by Jacques 2013; typical soft-tissue values (0.8-0.9 at visible
  wavelengths; Cheong, Prahl & Welch 1990, IEEE J. Quantum Electron. 26, 2166) are used and noted.
* Refractive indices: soft tissue 1.37-1.40 (Bolin et al. 1989, Appl. Opt. 28, 2297), fat 1.44, water 1.333.
* Fluorophore concentrations are *effective* concentrations in the units of the existing Studio
  presets (collagen = fluorescent cross-links; NADH/FAD of the order of 10-100 / 5-50 uM in
  metabolically active cells, Chance et al. 1979, J. Biol. Chem. 254, 4764; Croce & Bottiroli 2014,
  Eur. J. Histochem. 58, 2461).  They are literature-informed *typical* values, not measurements.
* The dysplasia nodule uses a 3x NADH enrichment and a 1.3x higher scattering than its host
  (nuclear crowding). Both ratios are illustrative choices for a demonstration scene (the direction of
  the effect is documented, e.g. Georgakoudi et al. 2002, Gastroenterology 122, 42), not measured values.
* Keratin is not in ``builtin_fluorophores()`` so it is not modelled.
"""
from __future__ import annotations

import numpy as np

from spectraforge.studio.schema import (
    AbsorberSpec,
    DyeAmount,
    Experiment,
    Inclusion,
    LayeredBlockShape,
    MaterialSpec,
    ScatteringSpec,
    TissueComposition,
)

# Jacques 2013 Table 1/2 entries: (a' [1/cm], f_ray, b_mie)
_T = {
    "epidermis": (66.7, 0.29, 0.689),        # Table 1 #7  Salomatina et al. 2006
    "papillary": (48.0, 0.409, 0.702),       # Table 2 skin average
    "reticular": (43.6, 0.41, 0.562),        # Table 1 #6  dermis, Salomatina et al. 2006
    "muscle": (13.0, 0.0, 0.926),            # Table 1 #29 Tromberg 1996 (a' = a, f_ray = 0)
    "adipose": (15.4, 0.0, 0.68),            # Table 1 #53 subcutaneous adipose, Bashkatov et al. 2005
    "oral_mucosa": (18.8, 0.0, 1.62),        # Table 1 #31 mucous tissue, Bashkatov et al. 2005
    "colon_mucosa": (19.1, 0.153, 1.091),    # Table 2 'other soft tissues'
    "liver": (16.5, 0.0, 1.64),              # Table 1 #43 Yi & Backman 2012
    "dense_collagen": (29.2, 0.489, 0.644),  # Table 2 'other fibrous tissues'
    "breast": (18.7, 0.288, 0.685),          # Table 2 breast
    "brain": (27.4, 0.315, 1.087),           # Table 2 brain
    "blood": (22.0, 0.0, 0.66),              # Table 1 #42 whole blood, Alexandrakis et al. 2005 (a, b)
}


def _scat(key: str, g: float, scale: float = 1.0) -> ScatteringSpec:
    a, f, b = _T[key]
    return ScatteringSpec(model="mie_rayleigh", mus_prime_500_per_mm=scale * a / 10.0, power_b=b,
                          f_rayleigh=f, g=g)


def _m(id_, name, color, dyes, tissue, scat, n, rough=0.2, aqueous=False) -> MaterialSpec:
    return MaterialSpec(id=id_, name=name, display_color=color,
                        dyes=[DyeAmount(fluorophore=f, concentration_uM=c) for f, c in dyes],
                        tissue=tissue, scattering=scat, refractive_index=n, surface_roughness=rough,
                        aqueous=aqueous)


def intralipid_scattering(percent: float = 1.0, ref_nm: float = 500.0) -> ScatteringSpec:
    """Power-law fit (400-900 nm) of the van Staveren 1991 Intralipid mus' = mus (1 - g)
    (see :func:`tissue_optics.intralipid_mus_prime_per_mm`), with ``g`` evaluated at ``ref_nm``."""
    from spectraforge.studio import tissue_optics as to
    nm = np.linspace(400.0, 900.0, 101)
    a, b = to.fit_power_law(nm, to.intralipid_mus_prime_per_mm(nm, percent), ref_nm)
    return ScatteringSpec(model="power_law", mus_prime_500_per_mm=a, power_b=b,
                          g=float(to.intralipid_g(ref_nm)))


def tissue_materials() -> list[MaterialSpec]:
    """Literature-parameterised tissue materials (see module docstring for the sources)."""
    skin_ar = [("NADH", 40.0), ("FAD", 15.0)]
    mats = [
        # ---- skin ----------------------------------------------------------------------------
        # epidermis: Choudhury 2010 / Jacques 1996 melanosome fractions: light 2.5 %, medium 13 %, dark 30 %
        # (Table 3 'Epidermis' M = 2.5 %; medium and dark from the Jacques skin-optics ranges 11-16 % and
        # 18-43 %), water ~50 %, keratin scattering (Table 1 #7)
        _m("epidermis_light", "Epidermis (light skin, M=2.5%)", "#f3c9b0", skin_ar,
           TissueComposition(water_fraction=0.5, melanosome_fraction=0.025), _scat("epidermis", 0.85), 1.40),
        _m("epidermis_medium", "Epidermis (medium skin, M=13%)", "#c98f6a", skin_ar,
           TissueComposition(water_fraction=0.5, melanosome_fraction=0.13), _scat("epidermis", 0.85), 1.40),
        _m("epidermis_dark", "Epidermis (dark skin, M=30%)", "#6b4a35", skin_ar,
           TissueComposition(water_fraction=0.5, melanosome_fraction=0.30), _scat("epidermis", 0.85), 1.40),
        # dermis: Table 3 #8 (Choudhury 2010): B = 0.2 %, S = 39 %, W = 65 % (mean dermis); papillary has the
        # richer capillary loop (B ~ 2 %, within the 0.2-4 % range), reticular is collagen-rich, low blood
        _m("dermis_papillary", "Papillary dermis", "#e7a79a",
           [("collagen", 250.0), ("elastin", 40.0), ("NADH", 20.0), ("FAD", 8.0)],
           TissueComposition(blood_volume_fraction=0.02, oxygen_saturation=0.75, water_fraction=0.65),
           _scat("papillary", 0.85), 1.39),
        _m("dermis_reticular", "Reticular dermis", "#dba096",
           [("collagen", 450.0), ("elastin", 60.0), ("NADH", 10.0), ("FAD", 4.0)],
           TissueComposition(blood_volume_fraction=0.005, oxygen_saturation=0.65, water_fraction=0.65),
           _scat("reticular", 0.85), 1.39),
        _m("subcutis_fat", "Subcutis (adipose)", "#f7e3a4", [("NADH", 4.0), ("FAD", 3.0), ("collagen", 40.0)],
           TissueComposition(blood_volume_fraction=0.005, oxygen_saturation=0.7, water_fraction=0.10,
                             lipid_fraction=0.80, beta_carotene_uM=3.0),
           _scat("adipose", 0.9), 1.44),
        # ---- other soft tissue ----------------------------------------------------------------
        # skeletal muscle: Table 1 #29 scattering; B ~ 1 % (Table 3 #36), W ~ 75 % (Woodard & White 1986)
        _m("muscle_skeletal", "Skeletal muscle", "#b5524f", [("NADH", 60.0), ("FAD", 30.0), ("collagen", 80.0)],
           TissueComposition(blood_volume_fraction=0.01, oxygen_saturation=0.65, water_fraction=0.75),
           _scat("muscle", 0.9), 1.38),
        _m("mucosa_oral", "Oral mucosa", "#e58f8f",
           [("NADH", 35.0), ("FAD", 14.0), ("collagen", 120.0), ("elastin", 15.0)],
           TissueComposition(blood_volume_fraction=0.02, oxygen_saturation=0.7, water_fraction=0.70),
           _scat("oral_mucosa", 0.85), 1.38),
        _m("mucosa_colon", "Colon mucosa", "#e9a29c",
           [("NADH", 45.0), ("FAD", 20.0), ("collagen", 150.0), ("elastin", 15.0)],
           TissueComposition(blood_volume_fraction=0.03, oxygen_saturation=0.6, water_fraction=0.80),
           _scat("colon_mucosa", 0.85), 1.38),
        _m("dysplasia_nodule", "Dysplastic nodule (NADH-rich, crowded nuclei)", "#ff7d5c",
           [("NADH", 135.0), ("FAD", 14.0), ("collagen", 60.0)],
           TissueComposition(blood_volume_fraction=0.03, oxygen_saturation=0.55, water_fraction=0.80),
           _scat("colon_mucosa", 0.85, scale=1.3), 1.38),
        # liver: Table 1 #43 scattering; sheep/horse liver B in Table 3 (49.5 %) is an ex-vivo perfusion
        # figure, a more conservative B = 10 % is used (in-vivo hepatic blood volume 10-30 %); W ~ 70 %
        _m("liver", "Liver", "#7a2f2a", [("NADH", 80.0), ("FAD", 40.0), ("lipofuscin", 4.0), ("porphyrin", 0.1)],
           TissueComposition(blood_volume_fraction=0.10, oxygen_saturation=0.70, water_fraction=0.70,
                             bilirubin_uM=2.0),
           _scat("liver", 0.9), 1.39),
        # breast: Table 2 scattering; Table 3 #1-5 normal breast: B ~ 1 %, S ~ 68 %, W ~ 15 %, F ~ 60 %
        _m("breast_normal", "Breast (normal)", "#f0bdb0", [("NADH", 25.0), ("FAD", 10.0), ("collagen", 150.0)],
           TissueComposition(blood_volume_fraction=0.01, oxygen_saturation=0.68, water_fraction=0.15,
                             lipid_fraction=0.60),
           _scat("breast", 0.9), 1.40),
        # brain grey matter: Table 2 brain; B ~ 3 % (Table 3 #26-27 rat cortex), W ~ 80 %
        _m("brain_grey", "Brain grey matter", "#c9b2b6", [("NADH", 70.0), ("FAD", 35.0), ("lipofuscin", 2.0)],
           TissueComposition(blood_volume_fraction=0.03, oxygen_saturation=0.60, water_fraction=0.80),
           _scat("brain", 0.9), 1.38),
        # ---- collagen -------------------------------------------------------------------------
        # acellular gel: hydrated collagen network (~98 % water); mus'(500) taken as 1/5 of the Table-2
        # fibrous-tissue value -- an ESTIMATE for a 2-4 mg/mL gel (no table value), the same order as the
        # existing 'collagen_gel' preset (0.6 /mm)
        _m("collagen_gel_acellular", "Collagen gel (acellular)", "#a89bff", [("collagen", 700.0), ("elastin", 40.0)],
           TissueComposition(blood_volume_fraction=0.0005, oxygen_saturation=0.7, water_fraction=0.97),
           _scat("dense_collagen", 0.9, scale=0.2), 1.36, rough=0.15),
        # dense (tendon-like) collagen: Table 2 'other fibrous tissues', W ~ 65 %
        _m("collagen_dense", "Dense collagen (tendon-like)", "#8d7bff", [("collagen", 1200.0), ("elastin", 80.0)],
           TissueComposition(blood_volume_fraction=0.001, oxygen_saturation=0.7, water_fraction=0.65),
           _scat("dense_collagen", 0.93), 1.41, rough=0.25),
        # ---- blood ----------------------------------------------------------------------------
        _m("blood_oxygenated", "Whole blood (S=95 %)", "#a0121a", [],
           TissueComposition(blood_volume_fraction=1.0, oxygen_saturation=0.95), _scat("blood", 0.98), 1.38),
        # ---- phantoms -------------------------------------------------------------------------
        MaterialSpec(id="intralipid_1pct", name="Intralipid 1 % phantom (van Staveren 1991)",
                     display_color="#f3f1ea", scattering=intralipid_scattering(1.0),
                     tissue=TissueComposition(water_fraction=0.99), refractive_index=1.34, aqueous=True,
                     surface_roughness=0.05),
        MaterialSpec(id="india_ink_absorber", name="India-ink absorber phantom (carbon black, mua(500)=0.1 /mm)",
                     display_color="#1c1c20", refractive_index=1.34, aqueous=True, surface_roughness=0.05,
                     background_absorption=AbsorberSpec(preset="carbon_black", scale=0.1),
                     tissue=TissueComposition(water_fraction=0.99)),
    ]
    return mats


# --------------------------------------------------------------------------------------------
# experiments (copy of the tissue_under_collagen scene structure: TLS + Nuance, block on black stage)
# --------------------------------------------------------------------------------------------
def _scene(name, description, block, footprint_target_y, sequence=None) -> Experiment:
    from spectraforge.studio import presets as P
    exp = P.mehsi_lab_replica()
    mats = {m.id: m for m in exp.materials}
    for m in tissue_materials():
        mats[m.id] = m
    sources = [P.tunable_lamp(target=(0.0, footprint_target_y, 0.0), **({"sequence": sequence} if sequence else {}))]
    return exp.model_copy(update={
        "name": name, "description": description, "materials": list(mats.values()),
        "objects": [P._obj("block", name, block, "collagen_dense", (0.0, 0.0, 0.0))],
        "sources": sources, "cameras": [P.nuance_camera(target=(0.0, footprint_target_y, 0.0))],
    })


def skin_three_layer() -> Experiment:
    """Epidermis (light skin, 0.1 mm) over papillary (0.2 mm) and reticular (1.2 mm) dermis over a 3 mm
    subcutis.  Thicknesses: typical forearm skin (Meglinski & Matcher 2002, Physiol. Meas. 23, 741)."""
    from spectraforge.studio import presets as P
    block = LayeredBlockShape(footprint_mm=(20.0, 20.0), layers=[
        P._layer("subcutis", "subcutis_fat", 3.0), P._layer("reticular dermis", "dermis_reticular", 1.2),
        P._layer("papillary dermis", "dermis_papillary", 0.2), P._layer("epidermis", "epidermis_light", 0.1)])
    return _scene("Skin three-layer (epidermis / dermis / subcutis)",
                  "Literature-parameterised skin: melanosome-absorbing epidermis, blood-perfused dermis "
                  "(NADH/FAD/collagen/elastin autofluorescence), fatty subcutis. 3D layered Monte Carlo.",
                  block, 2.25)


def mucosa_dysplasia_inclusion() -> Experiment:
    """Colon mucosa (0.8 mm) over submucosal dense collagen (1.0 mm) over muscle (2 mm) with a buried
    spherical NADH-rich dysplastic nodule (r = 0.35 mm, centre 0.45 mm below the surface)."""
    from spectraforge.studio import presets as P
    block = LayeredBlockShape(footprint_mm=(16.0, 16.0), layers=[
        P._layer("muscle", "muscle_skeletal", 2.0), P._layer("submucosa", "collagen_dense", 1.0),
        P._layer("mucosa", "mucosa_colon", 0.8)],
        inclusions=[Inclusion(name="nodule", material_id="dysplasia_nodule", kind="sphere",
                              center_mm=(0.0, 3.35, 0.0), radius_mm=0.35)])
    return _scene("Mucosa with buried dysplastic nodule",
                  "Colon mucosa over submucosa and muscle with a buried 0.7 mm NADH-rich dysplastic "
                  "nodule: can the TLS + Nuance chain see a metabolic lesion through 0.45 mm of mucosa?",
                  block, 1.9)


def vessel_under_collagen() -> Experiment:
    """A 0.4 mm blood vessel (S = 95 %) running along z in reticular dermis, 0.3 mm under a 0.5 mm dense-
    collagen cover: haemoglobin absorption imprints a dip on the collagen/dermis fluorescence."""
    from spectraforge.studio import presets as P
    block = LayeredBlockShape(footprint_mm=(16.0, 16.0), layers=[
        P._layer("dermis", "dermis_reticular", 3.0), P._layer("collagen", "collagen_dense", 0.5)],
        inclusions=[Inclusion(name="vessel", material_id="blood_oxygenated", kind="cylinder",
                              center_mm=(0.0, 2.6, 0.0), radius_mm=0.2, axis="z", length_mm=1000.0)])
    return _scene("Blood vessel under collagen",
                  "Cylindrical blood vessel (0.4 mm) in reticular dermis under a 0.5 mm dense-collagen "
                  "cover: light absorbed by oxy-haemoglobin attenuates the autofluorescence above it.",
                  block, 1.75)


def tissue_presets() -> list[Experiment]:
    return [skin_three_layer(), mucosa_dysplasia_inclusion(), vessel_under_collagen()]


TISSUE_PRESETS = {
    "skin_three_layer": skin_three_layer,
    "mucosa_dysplasia_inclusion": mucosa_dysplasia_inclusion,
    "vessel_under_collagen": vessel_under_collagen,
}
