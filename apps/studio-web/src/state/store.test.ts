import { beforeEach, describe, expect, it } from 'vitest'
import { useStudio } from './store'
import { PRESET_ORDER } from './presets'
import { parseExperiment, serialize } from './io'

const st = () => useStudio.getState()
beforeEach(() => { st().loadPreset('generic_macro_bench') })

describe('experiment store', () => {
  it('loads every preset with consistent references', () => {
    for (const p of PRESET_ORDER) {
      st().loadPreset(p.key)
      const e = st().exp
      const mats = new Set(e.materials.map((m) => m.id))
      expect(e.objects.length).toBeGreaterThan(0)
      for (const o of e.objects) expect(mats.has(o.material_id)).toBe(true)
      expect(st().past).toHaveLength(0)
      expect(st().dirty).toBe(false)
    }
  })

  it('undo / redo restores exact experiment snapshots', () => {
    const before = st().exp
    st().edit((e) => { e.name = 'A' })
    st().edit((e) => { e.name = 'B' })
    expect(st().exp.name).toBe('B')
    st().undo(); expect(st().exp.name).toBe('A')
    st().undo(); expect(st().exp).toBe(before)
    st().redo(); st().redo(); expect(st().exp.name).toBe('B')
    expect(st().future).toHaveLength(0)
  })

  it('coalesces slider drags with the same key into one undo step', () => {
    const n0 = st().past.length
    for (let v = 1; v <= 20; v++) st().edit((e) => { e.objects[0].transform.position_mm[0] = v }, 'drag:x')
    expect(st().past.length).toBe(n0 + 1)
    st().undo()
    expect(st().exp.objects[0].transform.position_mm[0]).not.toBe(20)
  })

  it('a new edit clears the redo stack', () => {
    st().edit((e) => { e.name = 'A' }); st().undo()
    expect(st().future).toHaveLength(1)
    st().edit((e) => { e.name = 'C' })
    expect(st().future).toHaveLength(0)
  })

  it('adds objects with unique ids, selects them, and deletes them', () => {
    const n = st().exp.objects.length
    const a = st().addObject('petri'), b = st().addObject('petri')
    expect(a).not.toBe(b)
    expect(st().exp.objects).toHaveLength(n + 2)
    expect(st().selection).toEqual({ kind: 'object', id: b })
    st().deleteSelected()
    expect(st().exp.objects).toHaveLength(n + 1)
    expect(st().selection).toBeNull()
    st().undo()
    expect(st().exp.objects.some((o) => o.id === b)).toBe(true)
  })

  it('duplicates the selected object at an offset with a fresh id', () => {
    const o = st().exp.objects[0]
    st().select({ kind: 'object', id: o.id })
    st().duplicateSelected()
    const c = st().exp.objects.at(-1)!
    expect(c.id).not.toBe(o.id)
    expect(c.shape).toEqual(o.shape)
    expect(c.transform.position_mm[0]).toBe(o.transform.position_mm[0] + 25)
  })

  it('refuses to delete the last source / camera and in-use materials', () => {
    st().select({ kind: 'source', id: st().exp.sources[0].id }); st().deleteSelected()
    expect(st().exp.sources).toHaveLength(1)
    st().select({ kind: 'camera', id: st().exp.cameras[0].id }); st().deleteSelected()
    expect(st().exp.cameras).toHaveLength(1)
    const used = st().exp.objects[0].material_id
    st().select({ kind: 'material', id: used }); st().deleteSelected()
    expect(st().exp.materials.some((m) => m.id === used)).toBe(true)
  })

  it('undo drops a selection that no longer exists', () => {
    const id = st().addObject('bead')
    expect(st().selection?.id).toBe(id)
    st().undo()
    expect(st().selection).toBeNull()
  })
})

describe('.sfx.json io', () => {
  it('round-trips an experiment', () => {
    const e = st().exp
    expect(parseExperiment(serialize(e))).toEqual(e)
  })
  it('rejects malformed input with a readable message', () => {
    expect(() => parseExperiment('nope')).toThrow(/JSON/)
    expect(() => parseExperiment('{"objects":[]}')).toThrow(/missing/)
  })
})
