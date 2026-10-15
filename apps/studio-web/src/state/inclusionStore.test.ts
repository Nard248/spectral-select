import { beforeEach, describe, expect, it } from 'vitest'
import { useStudio } from './store'
import { parseExperiment, serialize } from './io'
import { columnSegments } from '../api/mockBlock'
import { blockPixelComponents } from '../api/mockBlock'
import { INCLUSION_PRESETS, depthBelowTop, overlapsBlock } from '../lib/inclusions'
import { tissueLibrary } from './tissueLibrary'
import type { Inclusion, LayeredBlockShape } from '../generated/experiment'

const st = () => useStudio.getState()
const blockOf = (id: string) => { const o = st().exp.objects.find((x) => x.id === id)!; return o.shape as LayeredBlockShape }
beforeEach(() => { st().loadPreset('generic_macro_bench') })

describe('v1.2 library items and inclusion reducers', () => {
  it('"Vessel under collagen" brings its materials and buries a vessel inside the block', () => {
    const id = st().addObject('block_vessel')
    const b = blockOf(id)
    const ids = new Set(st().exp.materials.map((m) => m.id))
    for (const l of b.layers) expect(ids.has(l.material_id)).toBe(true)
    const v = (b.inclusions as Inclusion[])[0]
    expect(v.kind).toBe('cylinder'); expect(ids.has(v.material_id)).toBe(true)
    expect(overlapsBlock(b, v)).toBe(true); expect(depthBelowTop(b, v)).toBeCloseTo(0.9, 9)
  })
  it('"Buried tumour nodule" and "Skin (3-layer)" are consistent; tissue materials carry a composition', () => {
    const t = blockOf(st().addObject('block_tumour')); const s = blockOf(st().addObject('block_skin'))
    expect((t.inclusions as Inclusion[])[0].kind).toBe('sphere'); expect(s.layers).toHaveLength(3); expect(s.inclusions).toHaveLength(0)
    const ids = new Set(st().exp.materials.map((m) => m.id))
    for (const sh of [t, s]) for (const l of sh.layers) expect(ids.has(l.material_id)).toBe(true)
    expect(st().exp.materials.find((m) => m.id === 'dermis_reticular')!.tissue).not.toBeNull()
    expect(tissueLibrary().length).toBeGreaterThanOrEqual(18)
  })
  it('adding / editing / deleting an inclusion is undoable and survives a save-load round trip', () => {
    const id = st().addObject('block_two')
    const n0 = st().past.length
    const p = INCLUSION_PRESETS[2]
    st().edit((e) => {
      const o = e.objects.find((x) => x.id === id)!; if (o.shape.kind !== 'layered_block') return
      ;(o.shape.inclusions as Inclusion[]).push(p.make(o.shape, o.shape.layers[0].material_id, 'bead'))
    })
    expect(blockOf(id).inclusions).toHaveLength(1)
    const back = parseExperiment(serialize(st().exp))
    expect((back.objects.find((o) => o.id === id)!.shape as LayeredBlockShape).inclusions).toHaveLength(1)
    st().undo(); expect(blockOf(id).inclusions).toHaveLength(0); expect(st().past.length).toBe(n0)
  })
})

describe('mock engine: inclusions', () => {
  const setup = () => {
    st().addObject('block_vessel')
    const e = st().exp
    const o = e.objects.find((x) => x.id.startsWith('block_vessel'))!
    return { e, shape: o.shape as LayeredBlockShape, mats: new Map(e.materials.map((m) => [m.id, m] as const)) }
  }
  it('splits the column at the vessel chord and keeps the total thickness', () => {
    const { shape, mats } = setup()
    const plain = columnSegments({ ...shape, inclusions: [] as unknown as LayeredBlockShape['inclusions'] }, mats, 0, 0)
    const segs = columnSegments(shape, mats, 0, 0)
    expect(plain).toHaveLength(2)
    expect(segs.some((s) => s.name.startsWith('inc:'))).toBe(true)
    expect(segs.reduce((a, s) => a + s.t, 0)).toBeCloseTo(plain.reduce((a, s) => a + s.t, 0), 9)
    expect(columnSegments(shape, mats, 10, 0).some((s) => s.name.startsWith('inc:'))).toBe(false) // beside the vessel
  })
  it('a buried absorber darkens the fluorescence from below it', () => {
    const { e, shape } = setup()
    const step = { ex: 488, powerMW: 4, exposureMs: 100 }
    const bands = [520, 540, 560]
    const o = { gain: 1, bandStep: 10, sigmaBand: 5, lineFwhm: 10 }
    const sum = (r: Record<string, Float32Array>) => Object.values(r).reduce((a, v) => a + v.reduce((x, y) => x + y, 0), 0)
    const withVessel = sum(blockPixelComponents(e, shape, 0, 0, step, bands, o))
    const without = sum(blockPixelComponents(e, { ...shape, inclusions: [] as unknown as LayeredBlockShape['inclusions'] }, 0, 0, step, bands, o))
    expect(withVessel).toBeLessThan(without)
  })
})
