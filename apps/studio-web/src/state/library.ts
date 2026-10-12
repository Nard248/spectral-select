/** Library of insertable things: shapes / lab-ware (SceneObject factories), fresh materials, light sources. */
import type { SceneObject, MaterialSpec, LightSourceSpec, CameraSpec, BlockLayer, LayeredBlockShape } from '../generated/experiment'

export interface LibraryShape {
  key: string
  label: string
  hint: string
  /** library materials (ids from the presets bundle) that must exist in the experiment for this item (layered blocks) */
  requires?: string[]
  /** y offset so the item rests on the table (z=0 plane) */
  make: (id: string, materialId: string, vesselId: string | null) => SceneObject
}

const tf = (x = 0, y = 0, z = 0) => ({ position_mm: [x, y, z] as [number, number, number], rotation_deg: [0, 0, 0] as [number, number, number] })

const uni = (t: number): BlockLayer['thickness'] => ({ kind: 'uniform', thickness_mm: t, end_thickness_mm: t, axis: 'x' })
const wedge = (t0: number, t1: number, axis: 'x' | 'z' = 'x'): BlockLayer['thickness'] => ({ kind: 'linear', thickness_mm: t0, end_thickness_mm: t1, axis })
const block = (id: string, name: string, layers: [string, string, BlockLayer['thickness']][], coverMat: string, footprint: [number, number] = [40, 40]): SceneObject => ({
  id, name, shape: { kind: 'layered_block', footprint_mm: footprint, layers: layers.map(([n, m, t]) => ({ name: n, material_id: m, thickness: t })) } as LayeredBlockShape,
  transform: tf(), material_id: coverMat, vessel_material_id: null, visible: true,
})

/** Layered-block items: `material_id` is ignored by the engine for blocks (each layer has its own) but must reference an existing material. */
export const BLOCK_ITEMS: LibraryShape[] = [
  { key: 'block_wedge', label: 'Covered fluorophore', hint: 'wedge cover 2 → 0 mm', requires: ['tissue_substrate', 'fluorescein_agarose_layer', 'intralipid_cover'],
    make: (id) => block(id, 'Covered fluorophore (wedge)', [['substrate', 'tissue_substrate', uni(4)], ['fluorescent layer', 'fluorescein_agarose_layer', uni(0.5)], ['cover', 'intralipid_cover', wedge(2, 0)]], 'intralipid_cover') },
  { key: 'block_tissue', label: 'Tissue under collagen', hint: 'NADH/FAD, cover 1.5 → 0.05', requires: ['tissue_substrate', 'tissue_fluor_layer', 'collagen_cover'],
    make: (id) => block(id, 'Tissue / collagen block', [['substrate', 'tissue_substrate', uni(5)], ['tissue', 'tissue_fluor_layer', uni(1)], ['collagen', 'collagen_cover', wedge(1.5, 0.05)]], 'collagen_cover') },
  { key: 'block_two', label: 'Two-layer slab', hint: 'fluor slab + uniform cover', requires: ['fluorescein_agarose_layer', 'intralipid_cover'],
    make: (id) => block(id, 'Two-layer slab', [['fluorescent slab', 'fluorescein_agarose_layer', uni(3)], ['cover', 'intralipid_cover', uni(1)]], 'intralipid_cover', [30, 30]) },
]

/** Default shape + required library materials when a plain object is switched to `layered_block` in the inspector. */
export const DEFAULT_BLOCK = BLOCK_ITEMS[1]

export const LIBRARY_SHAPES: LibraryShape[] = [
  { key: 'petri', label: 'Petri dish', hint: '35 mm dish, glass', make: (id, m, v) => ({ id, name: 'Petri dish', shape: { kind: 'petri_dish', inner_radius_mm: 17, wall_mm: 1, height_mm: 10, base_mm: 1, fill_height_mm: 4 }, transform: tf(), material_id: m, vessel_material_id: v ?? 'glass', visible: true }) },
  { key: 'cuvette', label: 'Cuvette', hint: '10 × 10 mm quartz-like', make: (id, m, v) => ({ id, name: 'Cuvette', shape: { kind: 'cuvette', inner_mm: [10, 40, 10], wall_mm: 1.25, fill_height_mm: 30 }, transform: tf(), material_id: m, vessel_material_id: v ?? 'glass', visible: true }) },
  { key: 'slide', label: 'Glass slide', hint: '76 × 26 × 1 mm', make: (id, m) => ({ id, name: 'Glass slide', shape: { kind: 'box', size_mm: [76, 1, 26] }, transform: tf(0, 0.5, 0), material_id: m, vessel_material_id: null, visible: true }) },
  { key: 'droplet', label: 'Droplet', hint: 'sessile drop', make: (id, m) => ({ id, name: 'Droplet', shape: { kind: 'droplet', base_radius_mm: 8, height_mm: 4 }, transform: tf(), material_id: m, vessel_material_id: null, visible: true }) },
  { key: 'bead', label: 'Bead', hint: '7 mm sphere', make: (id, m) => ({ id, name: 'Bead', shape: { kind: 'sphere', radius_mm: 7 }, transform: tf(0, 7, 0), material_id: m, vessel_material_id: null, visible: true }) },
  { key: 'block', label: 'Block', hint: 'box 30 × 10 × 30', make: (id, m) => ({ id, name: 'Block', shape: { kind: 'box', size_mm: [30, 10, 30] }, transform: tf(0, 5, 0), material_id: m, vessel_material_id: null, visible: true }) },
  { key: 'puck', label: 'Puck', hint: 'cylinder r10 h6', make: (id, m) => ({ id, name: 'Puck', shape: { kind: 'cylinder', radius_mm: 10, height_mm: 6 }, transform: tf(0, 3, 0), material_id: m, vessel_material_id: null, visible: true }) },
  ...BLOCK_ITEMS,
]

export const newMaterial = (id: string, name = 'New material'): MaterialSpec => ({
  id, name, dyes: [], refractive_index: 1.4,
  scattering: { model: 'none', mus_prime_500_per_mm: 0, power_b: 1, g: 0.9 },
  background_absorption: { preset: 'none', scale: 1, custom: [] },
  surface_roughness: 0.3, aqueous: false, raman_strength: 1, quenching_ksv_per_uM: 0, fret: [], display_color: '#8aa0b8',
})

export const newSource = (id: string, kind: LightSourceSpec['kind'] = 'tunable_lamp'): LightSourceSpec => ({
  id, name: kind === 'laser' ? 'Laser' : kind === 'led_array' ? 'LED array' : 'Tunable lamp', kind,
  position_mm: [-180, 220, 60], target_mm: [0, 0, 0], roll_deg: 0,
  beam: { profile: 'gaussian', aperture_diameter_mm: kind === 'laser' ? 2 : 10, divergence_half_angle_deg: kind === 'laser' ? 0.5 : 7 },
  line_fwhm_nm: kind === 'laser' ? 1 : kind === 'led_array' ? 20 : 10,
  lamp_curve: kind === 'tunable_lamp' ? 'xenon' : 'flat', excitation_filter_od: 4, power_drift_pct: 0.5,
  sequence: [{ wavelength_nm: 405, power_mW: 3, exposure_ms: 100, enabled: true }, { wavelength_nm: 488, power_mW: 4, exposure_ms: 100, enabled: true }],
})

export const newCamera = (id: string, from: CameraSpec): CameraSpec => ({ ...structuredClone(from), id, name: 'HSI camera' })
