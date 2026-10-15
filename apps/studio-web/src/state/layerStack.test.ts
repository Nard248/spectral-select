import { describe, expect, it } from 'vitest'
import { addLayer, removeLayer, duplicateLayer, moveLayer, setProfileKind, setUniformThickness, uniqueLayerName, flipWedge, MAX_LAYERS } from './layerStack'
import type { LayeredBlockShape } from '../generated/experiment'

const mk = (): LayeredBlockShape => ({
  kind: 'layered_block', footprint_mm: [20, 20], inclusions: [],
  layers: [
    { name: 'a', material_id: 'm1', thickness: { kind: 'uniform', thickness_mm: 2, end_thickness_mm: 2, axis: 'x' } },
    { name: 'b', material_id: 'm2', thickness: { kind: 'linear', thickness_mm: 1, end_thickness_mm: 0, axis: 'z' } },
  ],
})
const names = (s: LayeredBlockShape) => (s.layers as { name: string }[]).map((l) => l.name)

describe('layer stack reducers', () => {
  it('uniqueLayerName', () => {
    expect(uniqueLayerName('layer', [])).toBe('layer')
    expect(uniqueLayerName('layer', ['layer', 'layer 2'])).toBe('layer 3')
  })
  it('addLayer appends / inserts with unique names and caps at 8', () => {
    const s = mk()
    expect(addLayer(s, 'm3')).toBe(2)
    expect(addLayer(s, 'm3')).toBe(3)
    expect(names(s)).toEqual(['a', 'b', 'layer', 'layer 2'])
    expect(addLayer(s, 'm3', 0, 'x')).toBe(1)
    expect(names(s)[1]).toBe('x')
    while (s.layers.length < MAX_LAYERS) addLayer(s, 'm1')
    expect(addLayer(s, 'm1')).toBe(-1)
    expect(s.layers.length).toBe(MAX_LAYERS)
  })
  it('removeLayer keeps at least one layer', () => {
    const s = mk()
    expect(removeLayer(s, 0)).toBe(true)
    expect(names(s)).toEqual(['b'])
    expect(removeLayer(s, 0)).toBe(false)
    expect(removeLayer(s, 5)).toBe(false)
  })
  it('duplicateLayer inserts a deep copy right after', () => {
    const s = mk()
    expect(duplicateLayer(s, 1)).toBe(2)
    expect(names(s)).toEqual(['a', 'b', 'b copy'])
    const L = s.layers as LayeredBlockShape['layers'][number][]
    L[2].thickness.thickness_mm = 9
    expect(L[1].thickness.thickness_mm).toBe(1)
  })
  it('moveLayer reorders (drag-to-reorder semantics)', () => {
    const s = mk(); addLayer(s, 'm3'); addLayer(s, 'm3')
    moveLayer(s, 0, 3)
    expect(names(s)).toEqual(['b', 'layer', 'layer 2', 'a'])
    moveLayer(s, 3, 1)
    expect(names(s)).toEqual(['b', 'a', 'layer', 'layer 2'])
    expect(moveLayer(s, 1, 1)).toBe(false)
  })
  it('profile kind switching keeps thickness continuous', () => {
    const s = mk(); const L = s.layers as LayeredBlockShape['layers'][number][]
    setProfileKind(L[0], 'linear')
    expect(L[0].thickness.kind).toBe('linear')
    expect(L[0].thickness.thickness_mm).toBe(2)
    expect(L[0].thickness.end_thickness_mm).toBe(0.5)
    setProfileKind(L[1], 'uniform')
    expect(L[1].thickness).toMatchObject({ kind: 'uniform', thickness_mm: 1, end_thickness_mm: 1 })
    setUniformThickness(L[1], 3)
    expect(L[1].thickness).toMatchObject({ thickness_mm: 3, end_thickness_mm: 3 })
    setProfileKind(L[0], 'linear')   // no-op when already linear
    flipWedge(L[0]); expect(L[0].thickness).toMatchObject({ thickness_mm: 0.5, end_thickness_mm: 2 })
  })
})
