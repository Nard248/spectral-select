/** Geometry of a layered block (planar interfaces) — shared by the mock engine and the light-field viewer.
 *  Local frame: x, z across the footprint (centred), y up; the bottom face sits on y = 0, layers listed bottom → top. */
import type { BlockLayer, LayeredBlockShape, ThicknessProfile } from '../generated/experiment'

/** Thickness (mm) of one profile at local (x, z) for footprint (Lx, Lz): uniform, or linear from `thickness_mm`
 *  at the negative edge of `axis` to `end_thickness_mm` at the positive edge. */
export function profileAt(p: ThicknessProfile, x: number, z: number, Lx: number, Lz: number): number {
  if (p.kind !== 'linear') return Math.max(0, p.thickness_mm)
  const u = p.axis === 'x' ? x / Lx + 0.5 : z / Lz + 0.5
  return Math.max(0, p.thickness_mm + (p.end_thickness_mm - p.thickness_mm) * Math.min(1, Math.max(0, u)))
}

export interface LayerStack {
  /** thickness per layer, bottom → top */
  t: number[]
  /** y of the bottom of each layer (above the block bottom) */
  bottom: number[]
  /** y of the top surface */
  top: number
}

export function stackAt(shape: Pick<LayeredBlockShape, 'footprint_mm' | 'layers'>, x: number, z: number): LayerStack {
  const [Lx, Lz] = shape.footprint_mm
  const t: number[] = [], bottom: number[] = []
  let y = 0
  for (const l of shape.layers as BlockLayer[]) {
    const th = profileAt(l.thickness, x, z, Lx, Lz)
    bottom.push(y); t.push(th); y += th
  }
  return { t, bottom, top: y }
}

/** The top surface plane y = a + b·x + c·z (exact because every profile is linear). */
export function topPlane(shape: Pick<LayeredBlockShape, 'footprint_mm' | 'layers'>): { a: number; b: number; c: number; max: number } {
  const [Lx, Lz] = shape.footprint_mm
  let a = 0, b = 0, c = 0
  for (const l of shape.layers as BlockLayer[]) {
    const p = l.thickness
    if (p.kind !== 'linear') { a += p.thickness_mm; continue }
    a += (p.thickness_mm + p.end_thickness_mm) / 2
    const s = (p.end_thickness_mm - p.thickness_mm)
    if (p.axis === 'x') b += s / Lx; else c += s / Lz
  }
  const max = Math.max(a + Math.abs(b) * Lx / 2 + Math.abs(c) * Lz / 2, 1e-6)
  return { a, b, c, max }
}

/** Depth of the TOP interface of layer `i` below the block's top surface (= Σ thickness of the layers above it). */
export function layerDepthTop(st: LayerStack, i: number): number {
  let d = 0
  for (let j = i + 1; j < st.t.length; j++) d += st.t[j]
  return d
}
