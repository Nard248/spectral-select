import { describe, expect, it } from 'vitest'
import { thicknessAt, interfaceHeights, layerDepth, totalRange, validateBlock, wedgeAngleDeg } from './profile'
import { BUNDLE } from '../../state/presets'
import type { LayeredBlockShape, ThicknessProfile } from '../../generated/experiment'

const FP: [number, number] = [40, 20]
const lin = (t0: number, t1: number, axis: 'x' | 'z'): ThicknessProfile => ({ kind: 'linear', thickness_mm: t0, end_thickness_mm: t1, axis })
const uni = (t: number): ThicknessProfile => ({ kind: 'uniform', thickness_mm: t, end_thickness_mm: t, axis: 'x' })

/** Independent re-statement of Python `ThicknessProfile.at` for cross-checking. */
function pyAt(p: ThicknessProfile, x: number, z: number, fp: [number, number]) {
  if (p.kind === 'uniform') return p.thickness_mm
  const u = p.axis === 'x' ? x : z, size = p.axis === 'x' ? fp[0] : fp[1]
  const f = Math.min(Math.max(u / size + 0.5, 0.0), 1.0)
  return p.thickness_mm + (p.end_thickness_mm - p.thickness_mm) * f
}

describe('thicknessAt (mirrors schema.ThicknessProfile.at)', () => {
  it('uniform ignores position', () => { expect(thicknessAt(uni(1.25), 17, -3, FP)).toBe(1.25) })
  it('linear x: start at -L/2, end at +L/2, midpoint average', () => {
    const p = lin(2, 0, 'x')
    expect(thicknessAt(p, -20, 5, FP)).toBeCloseTo(2, 12)
    expect(thicknessAt(p, 20, -5, FP)).toBeCloseTo(0, 12)
    expect(thicknessAt(p, 0, 0, FP)).toBeCloseTo(1, 12)
    expect(thicknessAt(p, 10, 0, FP)).toBeCloseTo(0.5, 12)
  })
  it('linear z uses the z size', () => {
    const p = lin(0.5, 1.5, 'z')
    expect(thicknessAt(p, 99, -10, FP)).toBeCloseTo(0.5, 12)
    expect(thicknessAt(p, -99, 10, FP)).toBeCloseTo(1.5, 12)
    expect(thicknessAt(p, 0, 5, FP)).toBeCloseTo(1.25, 12)
  })
  it('clamps outside the footprint', () => {
    expect(thicknessAt(lin(3, 1, 'x'), -1000, 0, FP)).toBe(3)
    expect(thicknessAt(lin(3, 1, 'x'), 1000, 0, FP)).toBe(1)
  })
  it('matches the reference formula on a grid', () => {
    for (const p of [lin(2, 0, 'x'), lin(0.05, 1.5, 'z'), uni(0.7)])
      for (let x = -25; x <= 25; x += 6.25) for (let z = -12; z <= 12; z += 3) expect(thicknessAt(p, x, z, FP)).toBeCloseTo(pyAt(p, x, z, FP), 12)
  })
})

describe('interfaces', () => {
  const shape: LayeredBlockShape = {
    kind: 'layered_block', footprint_mm: [40, 40],
    layers: [{ name: 's', material_id: 'a', thickness: uni(4) }, { name: 'f', material_id: 'b', thickness: uni(0.5) }, { name: 'c', material_id: 'c', thickness: lin(2, 0, 'x') }],
  }
  it('cumulative heights', () => {
    expect(interfaceHeights(shape, -20, 0)).toEqual([0, 4, 4.5, 6.5])
    expect(interfaceHeights(shape, 20, 0)).toEqual([0, 4, 4.5, 4.5])
  })
  it('interfaces are planes: height at the centre = mean of opposite corners', () => {
    for (let i = 0; i < 4; i++) {
      const c = interfaceHeights(shape, 0, 0)[i]
      const m = (interfaceHeights(shape, -20, -20)[i] + interfaceHeights(shape, 20, 20)[i]) / 2
      expect(c).toBeCloseTo(m, 12)
    }
  })
  it('layer depth below the top surface', () => {
    expect(layerDepth(shape, 1, -20, 0)).toBeCloseTo(2, 12)   // fluorescent layer top is 2 mm below the surface at the thick end
    expect(layerDepth(shape, 1, 20, 0)).toBeCloseTo(0, 12)    // and 0 mm at the pinched end
    expect(layerDepth(shape, 2, 0, 0)).toBeCloseTo(0, 12)     // top layer
  })
  it('total range', () => {
    expect(totalRange(shape)).toEqual({ min: 4.5, max: 6.5 })
  })
  it('presets: wedge geometry matches the documented numbers', () => {
    const o = BUNDLE.experiments['layered_wedge_phantom'].objects[0].shape
    if (o.kind !== 'layered_block') throw new Error('expected layered block')
    expect(totalRange(o)).toEqual({ min: 4.5, max: 6.5 })
    expect(wedgeAngleDeg(o.layers[2]!, o.footprint_mm)).toBeCloseTo(Math.atan(2 / 40) * 180 / Math.PI, 10)
    expect(wedgeAngleDeg(o.layers[0]!, o.footprint_mm)).toBeNull()
  })
})

describe('validateBlock', () => {
  const base = (): LayeredBlockShape => ({ kind: 'layered_block', footprint_mm: [10, 10], layers: [{ name: 'a', material_id: 'm', thickness: uni(1) }] })
  it('accepts a valid block', () => { expect(validateBlock(base(), ['m'])).toEqual([]) })
  it('flags duplicate names', () => {
    const s = base(); s.layers.push({ name: 'a', material_id: 'm', thickness: uni(1) })
    expect(validateBlock(s).some((i) => i.severity === 'error' && /twice/.test(i.message))).toBe(true)
  })
  it('flags zero total thickness at a corner only', () => {
    const s = base(); s.layers[0].thickness = lin(0, 1, 'x')
    expect(validateBlock(s).some((i) => /zero somewhere/.test(i.message))).toBe(true)
    s.layers.push({ name: 'b', material_id: 'm', thickness: uni(0.1) })
    expect(validateBlock(s).filter((i) => i.severity === 'error')).toEqual([])
  })
  it('flags dangling material', () => { expect(validateBlock(base(), ['other']).length).toBe(1) })
  it('too many layers', () => {
    const s = base(); for (let i = 0; i < 8; i++) s.layers.push({ name: `l${i}`, material_id: 'm', thickness: uni(1) })
    expect(validateBlock(s).some((i) => /At most 8/.test(i.message))).toBe(true)
  })
})
