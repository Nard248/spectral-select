import { describe, expect, it } from 'vitest'
import { robustMax, sliceTexture, defaultChannel, demoVolume, ROWS, type Volume } from './lightfield'
import { lut } from '../ui/colormaps'
import { BUNDLE } from '../state/presets'
import { spreadLabels } from '../three/layers/BlockAnnotations'
import type { VolumeInfo } from '../api/types'

/** 4 (x) × 3 (z) × 2 layers × 2 sublayers; fluence ∝ 10^(index along x) so the slice must show a decade per x cell. */
function ramp(): Volume {
  const nx = 4, nz = 3, L = 2, S = 2, nd = L * S
  const data = new Float32Array(nx * nz * nd)
  for (let ix = 0; ix < nx; ix++) for (let iz = 0; iz < nz; iz++) for (let d = 0; d < nd; d++) data[(ix * nz + iz) * nd + d] = Math.pow(10, ix) * (d === 0 ? 1 : 1)
  const info: VolumeInfo = { object_id: 'b', kind: 'fluence', excitations_nm: [365], channels: [], dims: [nx, nz, nd], x_edges_mm: [0, 1, 2, 3, 4], z_edges_mm: [0, 1, 2, 3], layer_names: ['a', 'b'], sublayers: S, unit: 'x' }
  return { info, data, dims: [nx, nz, nd], vmax: 1000 }
}
const px = (t: { image: { data: ArrayLike<number> | null; width: number } }, x: number, y: number) => { const o = (y * t.image.width + x) * 4; const d = t.image.data!; return [d[o], d[o + 1], d[o + 2]] }

describe('light-field slicing', () => {
  it('robustMax ignores a single hot voxel at the 99.9th percentile', () => {
    const a = new Float32Array(5000).fill(1); a[17] = 1e9
    expect(robustMax(a)).toBe(1)
    expect(robustMax(new Float32Array(10))).toBe(0)
  })
  it('texture stacks layers vertically at ROWS rows each', () => {
    const t = sliceTexture(ramp(), { axis: 'z', position_mm: 1.5, decades: 3, cmap: 'inferno' })
    expect(t.image.height).toBe(2 * ROWS)
  })
  it('log colour mapping: brighter cells (larger x) get higher colormap indices along the horizontal axis of a z-cut', () => {
    const t = sliceTexture(ramp(), { axis: 'z', position_mm: 1.5, decades: 3, cmap: 'inferno' })
    const lum = (c: number[]) => c[0] + c[1] + c[2]
    const left = lum(px(t, 2, ROWS / 2)), right = lum(px(t, t.image.width - 3, ROWS / 2))
    expect(right).toBeGreaterThan(left)
    // the maximum (1000 = vmax) maps to the top of the colormap, 3 decades lower to the bottom
    const cm = lut('inferno')
    expect(px(t, t.image.width - 1, ROWS / 2)).toEqual([cm[255 * 3], cm[255 * 3 + 1], cm[255 * 3 + 2]].map(Math.round))
  })
  it('x-cut picks the x cell containing the plane', () => {
    const v = ramp()
    const lo = sliceTexture(v, { axis: 'x', position_mm: 0.5, decades: 3, cmap: 'gray' })
    const hi = sliceTexture(v, { axis: 'x', position_mm: 3.5, decades: 3, cmap: 'gray' })
    expect(px(hi, 5, 5)[0]).toBeGreaterThan(px(lo, 5, 5)[0])
  })
  it('reuses the buffer when the size is unchanged', () => {
    const v = ramp()
    const a = sliceTexture(v, { axis: 'z', position_mm: 1, decades: 3, cmap: 'gray' })
    const b = sliceTexture(v, { axis: 'z', position_mm: 2, decades: 3, cmap: 'gray' }, a)
    expect(b).toBe(a)
  })
  it('defaultChannel prefers a dye over Raman', () => {
    expect(defaultChannel(['cover|raman', 'layer|dye'], null)).toBe('layer|dye')
    expect(defaultChannel(['a|raman', 'b|dye'], 'a|raman')).toBe('a|raman')
    expect(defaultChannel([], null)).toBeNull()
  })
  it('demo volume has the layer-conforming layout and a decaying field', () => {
    const o = BUNDLE.experiments['layered_wedge_phantom'].objects[0].shape
    if (o.kind !== 'layered_block') throw new Error('block expected')
    const v = demoVolume(o, 'block', [1.5, 0.3, 1.5], [0.02, 0.02, 0.02])
    expect(v.dims).toEqual([80, 80, 24])
    expect(v.data.length).toBe(80 * 80 * 24)
    expect(v.vmax).toBeGreaterThan(0)
  })
})

describe('callout spreading', () => {
  it('keeps neighbours at least `gap` apart without reordering', () => {
    const out = spreadLabels([1, 1.1, 1.2, 5], 2)
    for (let i = 1; i < out.length; i++) expect(out[i] - out[i - 1]).toBeGreaterThanOrEqual(2 - 1e-9)
    expect(out[3]).toBeGreaterThan(out[2])
  })
  it('leaves already-separated labels where they were', () => {
    expect(spreadLabels([0, 10, 20], 2)).toEqual([0, 10, 20])
  })
})
