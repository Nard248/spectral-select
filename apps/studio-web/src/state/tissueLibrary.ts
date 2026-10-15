/** Tissue-optics library: the engine bundle (`presets.json`, W1) ships the chromophore-composed tissue materials; this file adds
 *  only the client-side fallbacks the bundle lacks (a fluorescent bead) and resolves library ids. */
import type { MaterialSpec } from '../generated/experiment'
import { BUNDLE } from './presets'

type Sc = MaterialSpec['scattering']
const sc = (mus500: number, b: number, fRay: number, g = 0.9): Sc => ({ model: 'mie_rayleigh', mus_prime_500_per_mm: mus500, power_b: b, f_rayleigh: fRay, g })
const base = (id: string, name: string, color: string, n: number, scattering: Sc, tissue: MaterialSpec['tissue'], extra: Partial<MaterialSpec> = {}): MaterialSpec => ({
  id, name, dyes: [], refractive_index: n, scattering, background_absorption: { preset: 'none', scale: 1, custom: [] }, tissue,
  surface_roughness: 0.25, aqueous: false, raman_strength: 1, quenching_ksv_per_uM: 0, fret: [], display_color: color, ...extra,
})
export const LOCAL_TISSUE_MATERIALS: MaterialSpec[] = [
  base('fluorescent_bead', 'Fluorescent bead (EGFP)', '#5cff8a', 1.45, sc(3.0, 1.0, 0), null, { dyes: [{ fluorophore: 'EGFP', concentration_uM: 50 }] }),
]

/** Library material by id: engine bundle first, then the client-side tissue materials. */
export function libraryMaterial(id: string): MaterialSpec | undefined {
  return BUNDLE.materials.find((m) => m.id === id) ?? LOCAL_TISSUE_MATERIALS.find((m) => m.id === id)
}
/** Every tissue-optics material available to insert (bundle materials with a `tissue` block, plus the local ones not shadowed). */
export function tissueLibrary(): MaterialSpec[] {
  const fromBundle = BUNDLE.materials.filter((m) => m.tissue)
  return [...fromBundle, ...LOCAL_TISSUE_MATERIALS.filter((m) => !fromBundle.some((b) => b.id === m.id) && !BUNDLE.materials.some((b) => b.id === m.id))]
}
