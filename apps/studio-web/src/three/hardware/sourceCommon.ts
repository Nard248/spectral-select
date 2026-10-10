import * as THREE from 'three'
import type { LightSourceSpec } from '../../generated/experiment'
import { wavelengthToRGB } from '../../lib/color'
import { lookAtQuaternion } from '../pose'

export interface SourcePartProps {
  source: LightSourceSpec
  /** active excitation wavelength, null = off */
  wavelengthNm: number | null
  q: THREE.Quaternion
  selected?: boolean
  hovered?: boolean
}
export type V3 = [number, number, number]

export function sourceQuat(s: LightSourceSpec) { return lookAtQuaternion(s.position_mm, s.target_mm, s.roll_deg) }

/** emissive display colour for the active line (HDR multiplied for bloom) */
export function glowColor(nm: number | null, gain = 2.4): THREE.Color {
  if (nm == null) return new THREE.Color('#0b0d10')
  return new THREE.Color(...wavelengthToRGB(nm)).multiplyScalar(gain)
}

/** Horizontal unit vector pointing away from the target (where the table-standing post and cabling go). */
export function outwardXZ(s: LightSourceSpec): THREE.Vector2 {
  const v = new THREE.Vector2(s.position_mm[0] - s.target_mm[0], s.position_mm[2] - s.target_mm[2])
  if (v.length() < 1e-3) v.set(1, 0)
  return v.normalize()
}

/** Anchor (world orientation, relative to the source origin) for a local point + post location pushed `push` mm outward. */
export function mountFor(s: LightSourceSpec, q: THREE.Quaternion, local: V3, push = 46) {
  const a = new THREE.Vector3(...local).applyQuaternion(q)
  const o = outwardXZ(s)
  return { anchor: [a.x, a.y, a.z] as V3, post: [a.x + o.x * push, a.z + o.y * push] as [number, number] }
}
