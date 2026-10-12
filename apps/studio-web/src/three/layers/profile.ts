/** Layer-thickness math for layered blocks. Mirrors `ThicknessProfile.at` in
 *  `src/spectraforge/studio/schema.py` exactly (a linear profile is clamped at the footprint edges):
 *
 *      t(x, z) = t0 + (t1 − t0) · clamp(u / size + ½, 0, 1),   u = x (axis "x") or z (axis "z")
 *
 *  Every interface  y_i(x, z) = Σ_{j ≤ i} t_j(x, z)  is therefore an exact plane. */
import type { BlockLayer, LayeredBlockShape, ThicknessProfile } from '../../generated/experiment'

export type Footprint = readonly [number, number]
export type Axis = 'x' | 'z'

export function thicknessAt(p: ThicknessProfile, x: number, z: number, fp: Footprint): number {
  if (p.kind === 'uniform') return p.thickness_mm
  const u = p.axis === 'x' ? x : z
  const size = p.axis === 'x' ? fp[0] : fp[1]
  const f = Math.min(Math.max(u / size + 0.5, 0), 1)
  return p.thickness_mm + (p.end_thickness_mm - p.thickness_mm) * f
}

/** Thickness of every layer (bottom → top) at local (x, z). */
export const layerThicknesses = (shape: Pick<LayeredBlockShape, 'layers' | 'footprint_mm'>, x: number, z: number): number[] =>
  shape.layers.map((l) => thicknessAt(l.thickness, x, z, shape.footprint_mm))

/** Interface heights [0, y_1, …, y_n] (length n+1): bottom face, then the top of each layer. */
export function interfaceHeights(shape: Pick<LayeredBlockShape, 'layers' | 'footprint_mm'>, x: number, z: number): number[] {
  const out = [0]
  for (const t of layerThicknesses(shape, x, z)) out.push(out[out.length - 1] + t)
  return out
}

export const totalThickness = (shape: Pick<LayeredBlockShape, 'layers' | 'footprint_mm'>, x: number, z: number) =>
  layerThicknesses(shape, x, z).reduce((a, b) => a + b, 0)

/** Depth of the TOP interface of layer `i` below the block's top surface at (x, z) — the engine's `layer_depth:<layer>`. */
export function layerDepth(shape: Pick<LayeredBlockShape, 'layers' | 'footprint_mm'>, i: number, x: number, z: number): number {
  const h = interfaceHeights(shape, x, z)
  return h[h.length - 1] - h[i + 1]
}

export const footprintCorners = (fp: Footprint): [number, number][] => {
  const hx = fp[0] / 2, hz = fp[1] / 2
  return [[-hx, -hz], [hx, -hz], [-hx, hz], [hx, hz]]
}

/** Min / max total thickness over the footprint (attained at a corner, the profiles being linear). */
export function totalRange(shape: Pick<LayeredBlockShape, 'layers' | 'footprint_mm'>): { min: number; max: number } {
  const ts = footprintCorners(shape.footprint_mm).map(([x, z]) => totalThickness(shape, x, z))
  return { min: Math.min(...ts), max: Math.max(...ts) }
}

/** Thickness of one layer at its two profile ends (same value twice for a uniform layer). */
export const profileEnds = (l: BlockLayer): [number, number] =>
  l.thickness.kind === 'uniform' ? [l.thickness.thickness_mm, l.thickness.thickness_mm] : [l.thickness.thickness_mm, l.thickness.end_thickness_mm]

export interface BlockIssue {
  severity: 'error' | 'warn'
  message: string
  /** layer index the issue belongs to, if any */
  layer?: number
}

/** Same rules as the schema validators plus UI-level hints. `materialIds` (optional) flags dangling references. */
export function validateBlock(shape: LayeredBlockShape, materialIds?: Iterable<string>): BlockIssue[] {
  const issues: BlockIssue[] = []
  const [fx, fz] = shape.footprint_mm
  if (!(fx > 0) || !(fz > 0)) issues.push({ severity: 'error', message: 'Footprint must be positive in x and z.' })
  if ((shape.layers as BlockLayer[]).length === 0) issues.push({ severity: 'error', message: 'A block needs at least one layer.' })
  if ((shape.layers as BlockLayer[]).length > 8) issues.push({ severity: 'error', message: 'At most 8 layers are supported.' })
  const seen = new Map<string, number>()
  shape.layers.forEach((l, i) => {
    if (seen.has(l.name)) issues.push({ severity: 'error', layer: i, message: `Layer name "${l.name}" is used twice — names must be unique.` })
    seen.set(l.name, i)
    if (!l.name.trim()) issues.push({ severity: 'error', layer: i, message: `Layer ${i + 1} has no name.` })
  })
  if (fx > 0 && fz > 0 && (shape.layers as BlockLayer[]).length) {
    for (const [x, z] of footprintCorners(shape.footprint_mm)) {
      if (totalThickness(shape, x, z) <= 0) {
        issues.push({ severity: 'error', message: 'Total thickness is zero somewhere on the footprint — give at least one layer some thickness there.' })
        break
      }
    }
  }
  if (materialIds) {
    const ids = new Set(materialIds)
    shape.layers.forEach((l, i) => { if (!ids.has(l.material_id)) issues.push({ severity: 'error', layer: i, message: `Layer "${l.name}" uses unknown material "${l.material_id}".` }) })
  }
  shape.layers.forEach((l, i) => {
    const [a, b] = profileEnds(l)
    if (a === 0 && b === 0) issues.push({ severity: 'warn', layer: i, message: `Layer "${l.name}" has zero thickness everywhere (it is ignored).` })
  })
  return issues
}

/** Wedge slope in degrees for a linear profile over the footprint (null for uniform). */
export function wedgeAngleDeg(l: BlockLayer, fp: Footprint): number | null {
  if (l.thickness.kind !== 'linear') return null
  const size = l.thickness.axis === 'x' ? fp[0] : fp[1]
  return (Math.atan2(Math.abs(l.thickness.end_thickness_mm - l.thickness.thickness_mm), size) * 180) / Math.PI
}
