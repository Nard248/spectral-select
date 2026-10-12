/** Approximate beam irradiance (relative) of a source at a world point — used for the fluorescence-glow
 *  preview and the mock engine. Gaussian beam: w(z) = a/2 + z·tan(θ), I ∝ (w(L)/w(z))² exp(−2 r²/w²), normalised to 1 at the aim point. */
import type { LightSourceSpec } from '../generated/experiment'

export interface BeamSample { irradiance: number; incidenceDeg: number; distanceMm: number }

export function beamAt(src: LightSourceSpec, p: readonly number[], normal: readonly number[] | null = [0, 1, 0]): BeamSample {
  const o = src.position_mm, t = src.target_mm
  let ax = t[0] - o[0], ay = t[1] - o[1], az = t[2] - o[2]
  const al = Math.hypot(ax, ay, az) || 1
  ax /= al; ay /= al; az /= al
  const dx = p[0] - o[0], dy = p[1] - o[1], dz = p[2] - o[2]
  const z = dx * ax + dy * ay + dz * az
  const dist = Math.hypot(dx, dy, dz) || 1e-6
  if (z <= 1) return { irradiance: 0, incidenceDeg: 90, distanceMm: dist }
  const rx = dx - z * ax, ry = dy - z * ay, rz = dz - z * az
  const r2 = rx * rx + ry * ry + rz * rz
  const w0 = Math.max(src.beam.aperture_diameter_mm / 2, 0.1)
  const w = w0 + z * Math.tan((src.beam.divergence_half_angle_deg * Math.PI) / 180)
  const prof = src.beam.profile === 'gaussian' ? Math.exp((-2 * r2) / (w * w)) : (r2 <= w * w ? 1 : Math.exp(-((Math.sqrt(r2) - w) ** 2) / 20))
  // normalised so irradiance ≈ 1 at the aim point (throw L): geom = (w(L)/w(z))²
  const wL = w0 + al * Math.tan((src.beam.divergence_half_angle_deg * Math.PI) / 180)
  const geom = Math.min(8, (wL / w) ** 2)
  const cos = normal ? Math.max(0, -(dx * normal[0] + dy * normal[1] + dz * normal[2]) / dist) : 1   // null normal = volume illumination (no surface cosine)
  return {
    irradiance: prof * geom * cos,
    incidenceDeg: (Math.acos(Math.min(1, cos)) * 180) / Math.PI,
    distanceMm: dist,
  }
}
