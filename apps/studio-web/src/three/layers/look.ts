/** Physical-looking appearance of a block layer from its optical spec (no engine needed). */
import * as THREE from 'three'
import type { MaterialSpec } from '../../generated/experiment'

export interface LayerLook {
  color: THREE.Color
  /** apparent extinction per mm used for the see-through opacity  α = 1 − exp(−κ·t) */
  kappa: number
  roughness: number
  sheen: number
  clearcoat: number
  turbid: boolean
  clear: boolean
}

export function layerLook(m: MaterialSpec): LayerLook {
  const mus = m.scattering.model === 'none' ? 0 : m.scattering.mus_prime_500_per_mm
  const clear = mus < 0.05
  const color = new THREE.Color(m.display_color ?? '#bdbdbd').multiplyScalar(0.72)
  return {
    color,
    kappa: 0.3 + 0.42 * mus,
    roughness: THREE.MathUtils.clamp(0.22 + 0.7 * m.surface_roughness, 0.08, 1),
    sheen: clear ? 0 : Math.min(1, 0.4 + 0.25 * mus),
    clearcoat: clear ? 1 : m.surface_roughness < 0.3 ? 0.35 : 0,
    turbid: !clear,
    clear,
  }
}

/** See-through opacity of a slab of thickness `t` (mm). */
export const layerAlpha = (kappa: number, t: number) => 1 - Math.exp(-kappa * t)
