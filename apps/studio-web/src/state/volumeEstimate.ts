/** Planning estimates for the 3D layered-block Monte Carlo (cost model, not a measurement). */
import type { Experiment, VolumeMCSettings } from '../generated/experiment'

/** Assumed engine throughput in photon packets per second (all cores): ≈0.36 M/s measured on the first transport3d build
 *  (24 M excitation + 9.6 M emission packets in 94 s). A planning figure — it varies with hardware, layer optics and cell sizes. */
export const ASSUMED_PHOTONS_PER_S = 3.6e5

export const QUALITY_PRESETS: { key: string; label: string; photons: number; hint: string }[] = [
  { key: 'draft', label: 'Draft', photons: 200_000, hint: 'noisy tails, seconds' },
  { key: 'standard', label: 'Standard', photons: 2_000_000, hint: 'default' },
  { key: 'high', label: 'High', photons: 20_000_000, hint: 'smooth deep tails, minutes' },
  { key: 'max', label: 'Max', photons: 100_000_000, hint: 'publication, tens of minutes' },
]
export const qualityOf = (v: Pick<VolumeMCSettings, 'excitation_photons' | 'emission_photons'>) =>
  v.excitation_photons === v.emission_photons ? QUALITY_PRESETS.find((p) => p.photons === v.excitation_photons)?.key ?? 'custom' : 'custom'

export interface VolumeCost {
  blocks: number
  excitationSteps: number
  emissionNodes: number
  totalPhotons: number
  seconds: number
  previewSeconds: number
  voxels: number
  megabytes: number
}

/** Photon packets = N_ex·(excitation steps) + N_em·(emission nodes), per block; voxels = nx·nz·layers·sublayers. */
export function estimateVolumeCost(exp: Experiment): VolumeCost {
  const v = exp.acquisition.volume_mc
  const blocks = exp.objects.filter((o) => o.visible && o.shape.kind === 'layered_block')
  const src = exp.sources.find((s) => s.id === exp.active_source_id) ?? exp.sources[0]
  const steps = src ? src.sequence.filter((s) => s.enabled).length : 0
  const cam = exp.cameras.find((c) => c.id === exp.active_camera_id) ?? exp.cameras[0]
  const span = cam ? Math.max(0, cam.filter.n_bands * cam.filter.band_step_nm) : 300
  const nodes = Math.floor(span / v.emission_node_step_nm) + 1
  let voxels = 0
  for (const o of blocks) if (o.shape.kind === 'layered_block') {
    const [fx, fz] = o.shape.footprint_mm
    voxels += Math.ceil(fx / v.lateral_cell_mm) * Math.ceil(fz / v.lateral_cell_mm) * o.shape.layers.length * v.sublayers
  }
  const per = v.excitation_photons * steps + v.emission_photons * nodes
  const total = per * blocks.length
  return { blocks: blocks.length, excitationSteps: steps, emissionNodes: nodes, totalPhotons: total, seconds: total / ASSUMED_PHOTONS_PER_S, previewSeconds: (total * v.preview_fraction) / ASSUMED_PHOTONS_PER_S, voxels, megabytes: (voxels * 4 * (2 + 1)) / 1e6 }
}

export function fmtDuration(s: number): string {
  if (!(s > 0)) return '0 s'
  if (s < 1) return '< 1 s'
  if (s < 90) return `${Math.round(s)} s`
  if (s < 5400) return `${Math.round(s / 60)} min`
  return `${(s / 3600).toFixed(1)} h`
}
