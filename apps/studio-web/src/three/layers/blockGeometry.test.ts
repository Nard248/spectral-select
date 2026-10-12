import { describe, expect, it } from 'vitest'
import { buildBlockMesh, cutRegion, cutSide, buildSectionStrips, niceStep, type Patch } from './blockGeometry'
import { interfaceHeights } from './profile'
import type { LayeredBlockShape } from '../../generated/experiment'

const uni = (t: number) => ({ kind: 'uniform' as const, thickness_mm: t, end_thickness_mm: t, axis: 'x' as const })
const shape: LayeredBlockShape = {
  kind: 'layered_block', footprint_mm: [40, 30],
  layers: [
    { name: 's', material_id: 'a', thickness: uni(4) },
    { name: 'f', material_id: 'b', thickness: uni(0.5) },
    { name: 'c', material_id: 'c', thickness: { kind: 'linear', thickness_mm: 2, end_thickness_mm: 0, axis: 'x' } },
  ],
}

const triArea = (p: Patch, t: number) => {
  const i = [p.indices[3 * t], p.indices[3 * t + 1], p.indices[3 * t + 2]]
  const v = i.map((k) => [p.positions[3 * k], p.positions[3 * k + 1], p.positions[3 * k + 2]])
  const a = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]], b = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]]
  const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
  return { n: c, area: Math.hypot(...c) / 2 }
}
const allPatches = (m: ReturnType<typeof buildBlockMesh>) =>
  m.layers.flatMap((l) => [l.top, l.bottom, ...Object.values(l.sides)]).filter((p): p is Patch => !!p)

describe('buildBlockMesh', () => {
  const m = buildBlockMesh(shape)
  it('top vertices lie exactly on the interface planes', () => {
    m.layers.forEach((l, i) => {
      for (let v = 0; v < l.top.positions.length / 3; v++) {
        const x = l.top.positions[3 * v], y = l.top.positions[3 * v + 1], z = l.top.positions[3 * v + 2]
        expect(y).toBeCloseTo(interfaceHeights(shape, x, z)[i + 1], 4)
      }
    })
  })
  it('has no degenerate triangles and winding agrees with the stored normals', () => {
    for (const p of allPatches(m)) {
      expect(p.indices.length % 3).toBe(0)
      for (let t = 0; t < p.indices.length / 3; t++) {
        const { n, area } = triArea(p, t)
        expect(area).toBeGreaterThan(1e-9)
        const k = p.indices[3 * t]
        expect(n[0] * p.normals[3 * k] + n[1] * p.normals[3 * k + 1] + n[2] * p.normals[3 * k + 2]).toBeGreaterThan(0)
      }
    }
  })
  it('a layer pinching to zero thickness emits side-wall triangles only where it has area', () => {
    const cover = m.layers[2]
    const wall = cover.sides['+x']   // thickness is 0 at +x edge
    expect(wall).toBeNull()
    expect(cover.sides['-x']).not.toBeNull()
    expect(cover.sides['+z']).not.toBeNull()
    expect(m.layers[0].bottom).not.toBeNull()
    expect(m.layers[1].bottom).toBeNull()
  })
  it('block extents', () => {
    const xs = m.layers[0].top.positions
    let minX = 1e9, maxX = -1e9
    for (let i = 0; i < xs.length; i += 3) { minX = Math.min(minX, xs[i]); maxX = Math.max(maxX, xs[i]) }
    expect([minX, maxX]).toEqual([-20, 20])
  })
})

describe('cutaway', () => {
  it('cutRegion clamps and keeps the right side', () => {
    expect(cutRegion([40, 30], 'x', 5, 'low')).toMatchObject({ x0: -20, x1: 5, z0: -15, z1: 15 })
    expect(cutRegion([40, 30], 'z', -3, 'high')).toMatchObject({ z0: -3, z1: 15 })
    expect(cutRegion([40, 30], 'x', 99, 'low').x1).toBe(20)
    expect(cutSide('x', 'low')).toBe('+x'); expect(cutSide('z', 'high')).toBe('-z')
  })
  it('cut wall of the wedge sits at the cut with the interpolated thickness', () => {
    const m = buildBlockMesh(shape, cutRegion(shape.footprint_mm, 'x', 10, 'low'))
    const wall = m.layers[2].sides['+x']!
    const ys: number[] = []
    for (let i = 0; i < wall.positions.length; i += 3) { expect(wall.positions[i]).toBeCloseTo(10, 5); ys.push(wall.positions[i + 1]) }
    expect(Math.max(...ys)).toBeCloseTo(4.5 + 0.5, 4)   // cover thickness at x = 10 is 0.5 mm
  })
  it('section strips tile each layer between its interfaces', () => {
    const strips = buildSectionStrips(shape, 'x', -10, 'low', 8, 4)
    expect(strips).toHaveLength(3)
    const s = strips[2]
    const ys = Array.from({ length: s.positions.length / 3 }, (_, i) => s.positions[3 * i + 1])
    expect(Math.min(...ys)).toBeCloseTo(4.5, 5); expect(Math.max(...ys)).toBeCloseTo(4.5 + 1.5, 5)
    expect(Math.max(...s.uvs)).toBeLessThanOrEqual(1 + 1e-9)
  })
})

describe('thickness contours', () => {
  it('niceStep rounds up to 1-2-5', () => {
    expect(niceStep(0.3)).toBe(0.5); expect(niceStep(0.12)).toBe(0.2); expect(niceStep(0.04)).toBe(0.05); expect(niceStep(7)).toBe(10)
  })
  it('wedge layer gets iso-lines of constant thickness across its axis, on the top plane', () => {
    const m = buildBlockMesh(shape)
    const c = m.contours
    expect(c.length).toBeGreaterThan(0)
    expect(c.length % 6).toBe(0)
    for (let i = 0; i < c.length; i += 6) {
      expect(c[i]).toBeCloseTo(c[i + 3], 6)                         // constant x (wedge along x)
      expect(c[i + 1]).toBeCloseTo(interfaceHeights(shape, c[i], c[i + 2])[3], 4)   // on the cover's top plane
      const t = interfaceHeights(shape, c[i], 0)[3] - interfaceHeights(shape, c[i], 0)[2]
      expect(t).toBeGreaterThan(0); expect(t).toBeLessThan(2)       // strictly inside the wedge range
    }
  })
  it('uniform stacks have no contours', () => {
    const u: LayeredBlockShape = { ...shape, layers: [{ name: 'a', material_id: 'a', thickness: uni(2) }] }
    expect(buildBlockMesh(u).contours.length).toBe(0)
  })
})
