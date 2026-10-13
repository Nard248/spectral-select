import { describe, expect, it } from 'vitest'
import type { Inclusion, LayeredBlockShape } from '../generated/experiment'
import { asInclusions, centerYForDepth, chordAt, crossSection, cylinderSegment, depthBelowTop, fullyInside, overlapsBlock, validateInclusion, INCLUSION_PRESETS } from './inclusions'

const uni = (t: number) => ({ kind: 'uniform' as const, thickness_mm: t, end_thickness_mm: t, axis: 'x' as const })
const wedgeX = (a: number, b: number) => ({ kind: 'linear' as const, thickness_mm: a, end_thickness_mm: b, axis: 'x' as const })
type Layers = LayeredBlockShape['layers']
const block = (incs: Inclusion[] = [], layers: Layers = [{ name: 's', material_id: 'm', thickness: uni(4) }, { name: 'c', material_id: 'm', thickness: uni(2) }]): LayeredBlockShape =>
  ({ kind: 'layered_block', footprint_mm: [20, 10], layers, inclusions: asInclusions(incs) })
const inc = (o: Partial<Inclusion>): Inclusion => ({ name: 'i', material_id: 'm', kind: 'sphere', center_mm: [0, 5, 0], radius_mm: 1, semi_axes_mm: [2, 1, 1], size_mm: [2, 2, 2], axis: 'z', length_mm: 1000, ...o })

describe('depth below top', () => {
  it('is top − centre.y and round-trips through centerYForDepth', () => {
    const s = block()
    expect(depthBelowTop(s, inc({ center_mm: [0, 5, 0] }))).toBeCloseTo(1, 12)
    expect(centerYForDepth(s, 0, 0, 1.5)).toBeCloseTo(4.5, 12)
  })
  it('follows a wedge: the top surface slopes with x', () => {
    const s = block([], [{ name: 's', material_id: 'm', thickness: uni(2) }, { name: 'w', material_id: 'm', thickness: wedgeX(0, 4) }])
    // total thickness is 2 + 0 at x = -10, 2 + 4 at x = +10
    expect(centerYForDepth(s, -10, 0, 0.5)).toBeCloseTo(1.5, 12)
    expect(centerYForDepth(s, 10, 0, 0.5)).toBeCloseTo(5.5, 12)
    expect(depthBelowTop(s, inc({ center_mm: [10, 5.5, 0] }))).toBeCloseTo(0.5, 12)
  })
})

describe('inside-block tests', () => {
  const s = block()   // 20 × 10 footprint, top at y = 6
  it('inside / partly inside / outside', () => {
    expect(fullyInside(s, inc({ center_mm: [0, 3, 0], radius_mm: 1 }))).toBe(true)
    expect(overlapsBlock(s, inc({ center_mm: [0, 6, 0], radius_mm: 1 }))).toBe(true)   // pokes through the top
    expect(fullyInside(s, inc({ center_mm: [0, 6, 0], radius_mm: 1 }))).toBe(false)
    expect(overlapsBlock(s, inc({ center_mm: [30, 3, 0] }))).toBe(false)
    expect(overlapsBlock(s, inc({ center_mm: [0, 20, 0] }))).toBe(false)
  })
  it('a cylinder is bounded by its radius across and its length along', () => {
    const c = inc({ kind: 'cylinder', axis: 'x', radius_mm: 1, length_mm: 4, center_mm: [8, 3, 0] })
    expect(overlapsBlock(s, c)).toBe(true)
    expect(cylinderSegment(s, c)).toEqual([6, 10])         // 6..10, clipped at the +x edge (x = 10)
    expect(cylinderSegment(s, inc({ kind: 'cylinder', axis: 'z', length_mm: 1000 }))).toEqual([-5, 5])
    expect(cylinderSegment(s, inc({ kind: 'cylinder', axis: 'x', length_mm: 2, center_mm: [40, 3, 0] }))).toBeNull()
  })
  it('validation reports missing materials, outside and partly-outside inclusions, thin sizes', () => {
    expect(validateInclusion(s, inc({}), ['m'])).toEqual([])
    expect(validateInclusion(s, inc({ material_id: 'zz' }), ['m'])[0].severity).toBe('error')
    expect(validateInclusion(s, inc({ center_mm: [50, 3, 0] }), ['m'])[0].message).toMatch(/outside/)
    expect(validateInclusion(s, inc({ center_mm: [0, 6, 0] }), ['m'])[0].severity).toBe('warn')
    expect(validateInclusion(s, inc({ radius_mm: 0.05 }), ['m'], 0.2).some((i) => /voxel/.test(i.message))).toBe(true)
  })
})

describe('analytic chords and cross-sections', () => {
  it('sphere chord is 2·sqrt(r² − d²)', () => {
    const c = chordAt(inc({ radius_mm: 2, center_mm: [0, 5, 0] }), 1.2, 0)!
    expect(c[1] - c[0]).toBeCloseTo(2 * Math.sqrt(4 - 1.44), 12)
    expect(chordAt(inc({ radius_mm: 2 }), 2.1, 0)).toBeNull()
  })
  it('ellipsoid, box and cylinder chords', () => {
    expect(chordAt(inc({ kind: 'ellipsoid', semi_axes_mm: [2, 1, 1], center_mm: [0, 5, 0] }), 0, 0)).toEqual([4, 6])
    expect(chordAt(inc({ kind: 'box', size_mm: [2, 3, 2], center_mm: [0, 5, 0] }), 0.5, -0.5)).toEqual([3.5, 6.5])
    expect(chordAt(inc({ kind: 'box', size_mm: [2, 3, 2] }), 1.5, 0)).toBeNull()
    const cyl = inc({ kind: 'cylinder', axis: 'z', radius_mm: 1, center_mm: [0, 5, 0], length_mm: 4 })
    expect(chordAt(cyl, 0, 1.9)).toEqual([4, 6])
    expect(chordAt(cyl, 0, 2.1)).toBeNull()      // past the end of the cylinder
    expect(chordAt(cyl, 1.01, 0)).toBeNull()     // outside the radius
  })
  it('sphere cut at d from its centre has radius sqrt(r² − d²); a cylinder cut along its axis is a circle, across it a strip', () => {
    const cs = crossSection(inc({ radius_mm: 2, center_mm: [1, 5, 0] }), 'x', 2)!
    expect(cs.shape).toBe('ellipse'); expect(cs.hw).toBeCloseTo(Math.sqrt(3), 12); expect(cs.cy).toBe(5)
    const cyl = inc({ kind: 'cylinder', axis: 'z', radius_mm: 1, center_mm: [0, 5, 0], length_mm: 1000 })
    expect(crossSection(cyl, 'z', 0)).toMatchObject({ shape: 'ellipse', hw: 1, hy: 1 })
    const strip = crossSection(cyl, 'x', 0.6, 5)!
    expect(strip.shape).toBe('rect'); expect(strip.hy).toBeCloseTo(0.8, 12); expect(strip.hw).toBe(5)
    expect(crossSection(cyl, 'x', 1.2)).toBeNull()
  })
})

describe('library presets', () => {
  it('place each preset inside the block, buried below the top surface', () => {
    const s = block([], [{ name: 's', material_id: 'm', thickness: uni(4) }, { name: 'c', material_id: 'm', thickness: uni(3) }])
    for (const p of INCLUSION_PRESETS) {
      const i = p.make(s, p.materialId, p.label)
      expect(overlapsBlock(s, i)).toBe(true)
      expect(depthBelowTop(s, i)).toBeGreaterThan(0)
    }
  })
})
