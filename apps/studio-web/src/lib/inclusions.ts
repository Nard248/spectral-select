/** Geometry helpers for block inclusions (`LayeredBlockShape.inclusions`), in the BLOCK's local frame: footprint centred on
 *  x = z = 0, bottom face at y = 0, top surface at y = Σ layer thicknesses (a plane for uniform / linear profiles).
 *  Mirrors the schema doc of `Inclusion` (sphere / ellipsoid / cylinder / box; the part outside the block is ignored). */
import type { Inclusion, LayeredBlockShape } from '../generated/experiment'
import { totalRange, totalThickness } from '../three/layers/profile'

/** The generated `Inclusions` type is a 0..16-tuple union; this views a plain array as that type. */
export const asInclusions = (a: Inclusion[]): LayeredBlockShape['inclusions'] => a as unknown as LayeredBlockShape['inclusions']
export type Kind = Inclusion['kind']
export type Vec3 = [number, number, number]
export const MAX_INCLUSIONS = 16
export const KINDS: { value: Kind; label: string }[] = [
  { value: 'sphere', label: 'Sphere' }, { value: 'ellipsoid', label: 'Ellipsoid' }, { value: 'cylinder', label: 'Cylinder' }, { value: 'box', label: 'Box' },
]
const AX = { x: 0, z: 2 } as const

/** Top surface height of the block at (x, z). */
export const topAt = (shape: LayeredBlockShape, x: number, z: number) => totalThickness(shape, x, z)

/** Depth of the inclusion centre below the block's top surface (mm, positive downwards). */
export const depthBelowTop = (shape: LayeredBlockShape, inc: Pick<Inclusion, 'center_mm'>) =>
  topAt(shape, inc.center_mm[0], inc.center_mm[2]) - inc.center_mm[1]
/** Centre y that puts the inclusion centre `depth` mm below the top surface. */
export const centerYForDepth = (shape: LayeredBlockShape, x: number, z: number, depth: number) => topAt(shape, x, z) - depth

/** Half extents (x, y, z) of the inclusion's axis-aligned bounding box (cylinder length NOT clipped to the block). */
export function halfExtents(inc: Inclusion): Vec3 {
  switch (inc.kind) {
    case 'sphere': return [inc.radius_mm, inc.radius_mm, inc.radius_mm]
    case 'ellipsoid': return [...inc.semi_axes_mm] as Vec3
    case 'box': return [inc.size_mm[0] / 2, inc.size_mm[1] / 2, inc.size_mm[2] / 2]
    case 'cylinder': return inc.axis === 'x' ? [inc.length_mm / 2, inc.radius_mm, inc.radius_mm] : [inc.radius_mm, inc.radius_mm, inc.length_mm / 2]
  }
}

/** Characteristic size (mm) used for sanity warnings and sizing the 3D label. */
export const characteristicSize = (inc: Inclusion) => 2 * (inc.kind === 'cylinder' ? inc.radius_mm : Math.min(...halfExtents(inc)))

/** Axis-aligned bounds [min, max] per axis; a cylinder is clipped to the footprint along its axis (that is how the engine treats it). */
function bounds(shape: LayeredBlockShape, inc: Inclusion): { lo: Vec3; hi: Vec3; clippedAxis: number } | null {
  const h = halfExtents(inc), c = inc.center_mm
  const lo: Vec3 = [c[0] - h[0], c[1] - h[1], c[2] - h[2]], hi: Vec3 = [c[0] + h[0], c[1] + h[1], c[2] + h[2]]
  let clippedAxis = -1
  if (inc.kind === 'cylinder') {
    const seg = cylinderSegment(shape, inc); if (!seg) return null
    clippedAxis = AX[inc.axis]; lo[clippedAxis] = seg[0]; hi[clippedAxis] = seg[1]
  }
  return { lo, hi, clippedAxis }
}
/** Does the inclusion overlap the block volume at all (a cheap, conservative "partly inside" on the bounding box)? */
export function overlapsBlock(shape: LayeredBlockShape, inc: Inclusion): boolean {
  const b = bounds(shape, inc); if (!b) return false
  const { max } = totalRange(shape)
  const [fx, fz] = shape.footprint_mm
  return b.hi[0] > -fx / 2 && b.lo[0] < fx / 2 && b.hi[1] > 0 && b.lo[1] < max && b.hi[2] > -fz / 2 && b.lo[2] < fz / 2
}
/** Is the whole bounding box inside the block volume (a cylinder's length is clipped by design and never counts as outside)? */
export function fullyInside(shape: LayeredBlockShape, inc: Inclusion): boolean {
  const b = bounds(shape, inc); if (!b) return false
  const { min } = totalRange(shape)
  const [fx, fz] = shape.footprint_mm
  return b.lo[0] >= -fx / 2 && b.hi[0] <= fx / 2 && b.lo[2] >= -fz / 2 && b.hi[2] <= fz / 2 && b.lo[1] >= 0 && b.hi[1] <= min
}

export interface InclusionIssue { severity: 'error' | 'warn'; message: string }
/** UI-level validation of one inclusion against its block (the engine re-validates; this is for instant feedback). */
export function validateInclusion(shape: LayeredBlockShape, inc: Inclusion, materialIds: Iterable<string>, voxelMm?: number): InclusionIssue[] {
  const out: InclusionIssue[] = []
  if (!new Set(materialIds).has(inc.material_id)) out.push({ severity: 'error', message: `Material "${inc.material_id}" does not exist.` })
  const h = halfExtents(inc)
  if (!(h[0] > 0 && h[1] > 0 && h[2] > 0)) out.push({ severity: 'error', message: 'All sizes must be positive.' })
  if (!overlapsBlock(shape, inc)) out.push({ severity: 'error', message: 'The inclusion lies completely outside the block, so it has no effect.' })
  else if (!fullyInside(shape, inc)) out.push({ severity: 'warn', message: 'Partly outside the block: only the part inside is simulated.' })
  if (voxelMm && voxelMm > 0 && characteristicSize(inc) < 2 * voxelMm) out.push({ severity: 'warn', message: `Thinner than 2 voxels (${voxelMm} mm): it will be voxelised coarsely. Reduce the inclusion voxel size in Acquisition.` })
  return out
}

/** Segment [a, b] of a cylinder along its axis after clipping to the block footprint (null if it misses the block). */
export function cylinderSegment(shape: LayeredBlockShape, inc: Inclusion): [number, number] | null {
  const i = AX[inc.axis], half = shape.footprint_mm[inc.axis === 'x' ? 0 : 1] / 2
  const a = Math.max(inc.center_mm[i] - inc.length_mm / 2, -half), b = Math.min(inc.center_mm[i] + inc.length_mm / 2, half)
  return b > a ? [a, b] : null
}

/** Vertical chord [y0, y1] of the inclusion along the line (x, z) (null if the line misses it); not clipped to the block. */
export function chordAt(inc: Inclusion, x: number, z: number): [number, number] | null {
  const [cx, cy, cz] = inc.center_mm
  switch (inc.kind) {
    case 'sphere': {
      const q = inc.radius_mm ** 2 - (x - cx) ** 2 - (z - cz) ** 2
      return q > 0 ? [cy - Math.sqrt(q), cy + Math.sqrt(q)] : null
    }
    case 'ellipsoid': {
      const [ax, ay, az] = inc.semi_axes_mm
      const s = 1 - ((x - cx) / ax) ** 2 - ((z - cz) / az) ** 2
      return s > 0 ? [cy - ay * Math.sqrt(s), cy + ay * Math.sqrt(s)] : null
    }
    case 'cylinder': {
      const along = inc.axis === 'x' ? x - cx : z - cz, across = inc.axis === 'x' ? z - cz : x - cx
      if (Math.abs(along) > inc.length_mm / 2 || Math.abs(across) >= inc.radius_mm) return null
      const h = Math.sqrt(inc.radius_mm ** 2 - across ** 2)
      return [cy - h, cy + h]
    }
    case 'box':
      return Math.abs(x - cx) < inc.size_mm[0] / 2 && Math.abs(z - cz) < inc.size_mm[2] / 2 ? [cy - inc.size_mm[1] / 2, cy + inc.size_mm[1] / 2] : null
  }
}

/** Planar cross-section of an inclusion cut by the plane `axis = p`, in plane coordinates (w = the other horizontal axis, y). */
export interface CrossSection { shape: 'ellipse' | 'rect'; cw: number; cy: number; hw: number; hy: number }
export function crossSection(inc: Inclusion, axis: 'x' | 'z', p: number, halfW?: number): CrossSection | null {
  const [cx, cy, cz] = inc.center_mm
  const ca = axis === 'x' ? cx : cz, cw = axis === 'x' ? cz : cx
  const d = p - ca
  switch (inc.kind) {
    case 'sphere': {
      const q = inc.radius_mm ** 2 - d * d
      return q > 0 ? { shape: 'ellipse', cw, cy, hw: Math.sqrt(q), hy: Math.sqrt(q) } : null
    }
    case 'ellipsoid': {
      const [ax, ay, az] = inc.semi_axes_mm
      const sa = axis === 'x' ? ax : az, sw = axis === 'x' ? az : ax
      const s = 1 - (d / sa) ** 2
      return s > 0 ? { shape: 'ellipse', cw, cy, hw: sw * Math.sqrt(s), hy: ay * Math.sqrt(s) } : null
    }
    case 'cylinder': {
      if (inc.axis === axis) return Math.abs(d) <= inc.length_mm / 2 ? { shape: 'ellipse', cw, cy, hw: inc.radius_mm, hy: inc.radius_mm } : null
      if (Math.abs(d) >= inc.radius_mm) return null
      let lo = cw - inc.length_mm / 2, hi = cw + inc.length_mm / 2
      if (halfW != null) { lo = Math.max(lo, -halfW); hi = Math.min(hi, halfW) }
      return hi > lo ? { shape: 'rect', cw: (lo + hi) / 2, cy, hw: (hi - lo) / 2, hy: Math.sqrt(inc.radius_mm ** 2 - d * d) } : null
    }
    case 'box': {
      const sa = axis === 'x' ? inc.size_mm[0] : inc.size_mm[2], sw = axis === 'x' ? inc.size_mm[2] : inc.size_mm[0]
      return Math.abs(d) < sa / 2 ? { shape: 'rect', cw, cy, hw: sw / 2, hy: inc.size_mm[1] / 2 } : null
    }
  }
}

/** One-line size description for lists and labels. */
export function describeSize(inc: Inclusion): string {
  const f = (v: number) => `${+v.toFixed(3)}`
  switch (inc.kind) {
    case 'sphere': return `Ø ${f(inc.radius_mm * 2)} mm`
    case 'ellipsoid': return `${f(inc.semi_axes_mm[0] * 2)} × ${f(inc.semi_axes_mm[1] * 2)} × ${f(inc.semi_axes_mm[2] * 2)} mm`
    case 'cylinder': return `Ø ${f(inc.radius_mm * 2)} mm along ${inc.axis}`
    case 'box': return `${f(inc.size_mm[0])} × ${f(inc.size_mm[1])} × ${f(inc.size_mm[2])} mm`
  }
}

/** Unique inclusion name based on `base`. */
export function uniqueName(base: string, taken: string[]): string {
  if (!taken.includes(base)) return base
  for (let i = 2; ; i++) if (!taken.includes(`${base} ${i}`)) return `${base} ${i}`
}

export function makeInclusion(kind: Kind, materialId: string, shape: LayeredBlockShape, name = 'Inclusion'): Inclusion {
  const total = totalThickness(shape, 0, 0)
  const r = Math.max(0.2, Math.min(1, total / 6))
  const inc: Inclusion = {
    name, material_id: materialId, kind, center_mm: [0, Math.max(0.05, total - 2 * r), 0],
    radius_mm: r, semi_axes_mm: [1.5 * r, r, r], size_mm: [3 * r, 2 * r, 3 * r], axis: 'z', length_mm: 1000,
  }
  return inc
}

export interface InclusionPreset { key: string; label: string; hint: string; materialId: string; make: (shape: LayeredBlockShape, materialId: string, name: string) => Inclusion }
const atDepth = (shape: LayeredBlockShape, inc: Inclusion, depth: number): Inclusion => {
  inc.center_mm[1] = Math.max(centerYForDepth(shape, inc.center_mm[0], inc.center_mm[2], depth), 0.02)
  return inc
}
export const INCLUSION_PRESETS: InclusionPreset[] = [
  { key: 'vessel', label: 'Blood vessel', hint: 'Ø 0.8 mm cylinder, 0.9 mm deep', materialId: 'blood_oxygenated',
    make: (s, m, name) => atDepth(s, { ...makeInclusion('cylinder', m, s, name), radius_mm: 0.4, axis: 'z', length_mm: 1000, center_mm: [0, 0, 0] }, 0.9) },
  { key: 'tumour', label: 'Tumour nodule', hint: 'Ø 3 mm sphere, 2 mm deep', materialId: 'dysplasia_nodule',
    make: (s, m, name) => atDepth(s, { ...makeInclusion('sphere', m, s, name), radius_mm: 1.5, center_mm: [0, 0, 0] }, 2) },
  { key: 'bead', label: 'Fluorescent bead', hint: 'Ø 0.5 mm sphere, 1.2 mm deep', materialId: 'fluorescent_bead',
    make: (s, m, name) => atDepth(s, { ...makeInclusion('sphere', m, s, name), radius_mm: 0.25, center_mm: [0, 0, 0] }, 1.2) },
]
