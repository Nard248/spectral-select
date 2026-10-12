import raw from '../generated/presets.json'
import type { Experiment, FluorophoreSpec, MaterialSpec } from '../generated/experiment'
import type { PresetsBundle } from '../api/types'

export const BUNDLE = raw as unknown as PresetsBundle

export const PRESET_ORDER: { key: string; label: string; blurb: string }[] = [
  { key: 'mehsi_lab_replica', label: 'ME-HSI lab replica', blurb: 'Nuance-style LCTF, 8 UV/violet excitations, lichen thalli' },
  { key: 'generic_macro_bench', label: 'Generic macro bench', blurb: 'Mixed fluorescent samples under a tunable lamp' },
  { key: 'led_multispectral', label: 'LED multispectral', blurb: 'LED array + filter wheel, six bands' },
  { key: 'laser_line', label: 'Laser line, side view', blurb: 'Cuvette row: dilute, FRET and inner-filter regimes' },
  { key: 'layered_wedge_phantom', label: 'Layered wedge phantom', blurb: '3D: fluorescein layer under a turbid cover thinning 2 → 0 mm' },
  { key: 'tissue_under_collagen', label: 'Tissue under collagen', blurb: '3D: NADH/FAD layer under a collagen cover 1.5 → 0.05 mm' },
  { key: 'pencil_beam_penetration', label: 'Pencil-beam penetration', blurb: '3D: 1 mm laser at 45° into tissue under 0.5 mm collagen — the banana-shaped light field' },
]

export const clone = <T,>(v: T): T => structuredClone(v)

export function presetExperiment(key: string): Experiment {
  const e = BUNDLE.experiments[key] ?? Object.values(BUNDLE.experiments)[0]
  return clone(e)
}

export const DEFAULT_PRESET = 'mehsi_lab_replica'

/** Builtin fluorophores overridden by the experiment's own. */
export function fluorophoreMap(exp: Pick<Experiment, 'fluorophores'>): Record<string, FluorophoreSpec> {
  const m: Record<string, FluorophoreSpec> = {}
  for (const f of BUNDLE.fluorophores) m[f.name] = f
  for (const f of exp.fluorophores) m[f.name] = f
  return m
}

export function materialMap(exp: Pick<Experiment, 'materials'>): Record<string, MaterialSpec> {
  const m: Record<string, MaterialSpec> = {}
  for (const x of exp.materials) m[x.id] = x
  return m
}
