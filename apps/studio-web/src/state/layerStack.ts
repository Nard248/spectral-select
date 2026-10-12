/** Pure editing operations for the layer stack of a `layered_block` (operate on a plain object or an immer draft). */
import type { BlockLayer, LayeredBlockShape, ThicknessProfile } from '../generated/experiment'

export const MAX_LAYERS = 8
const layersOf = (s: LayeredBlockShape) => s.layers as BlockLayer[]

export function uniqueLayerName(base: string, taken: Iterable<string>): string {
  const set = new Set(taken)
  if (!set.has(base)) return base
  let n = 2
  while (set.has(`${base} ${n}`)) n++
  return `${base} ${n}`
}

export const uniformProfile = (t = 1): ThicknessProfile => ({ kind: 'uniform', thickness_mm: t, end_thickness_mm: t, axis: 'x' })

/** Append (or insert after `after`) a uniform 1 mm layer. Returns the new layer's index, or -1 when the stack is full. */
export function addLayer(shape: LayeredBlockShape, materialId: string, after?: number, name = 'layer'): number {
  const L = layersOf(shape)
  if (L.length >= MAX_LAYERS) return -1
  const at = after == null ? L.length : Math.min(Math.max(after + 1, 0), L.length)
  L.splice(at, 0, { name: uniqueLayerName(name, L.map((l) => l.name)), material_id: materialId, thickness: uniformProfile(1) })
  return at
}

export function removeLayer(shape: LayeredBlockShape, i: number): boolean {
  const L = layersOf(shape)
  if (L.length <= 1 || i < 0 || i >= L.length) return false
  L.splice(i, 1)
  return true
}

export function duplicateLayer(shape: LayeredBlockShape, i: number): number {
  const L = layersOf(shape)
  if (L.length >= MAX_LAYERS || i < 0 || i >= L.length) return -1
  const src = L[i]
  L.splice(i + 1, 0, { name: uniqueLayerName(`${src.name} copy`, L.map((l) => l.name)), material_id: src.material_id, thickness: { ...src.thickness } })
  return i + 1
}

/** Move layer `from` so that it ends up at index `to` (both in the current list). */
export function moveLayer(shape: LayeredBlockShape, from: number, to: number): boolean {
  const L = layersOf(shape)
  if (from === to || from < 0 || to < 0 || from >= L.length || to >= L.length) return false
  const [l] = L.splice(from, 1)
  L.splice(to, 0, l)
  return true
}

/** Switch a layer between a uniform slab and a linear wedge, keeping the thickness continuous. */
export function setProfileKind(layer: BlockLayer, kind: ThicknessProfile['kind']) {
  if (layer.thickness.kind === kind) return
  const t = layer.thickness
  layer.thickness = kind === 'uniform'
    ? { kind: 'uniform', thickness_mm: t.thickness_mm, end_thickness_mm: t.thickness_mm, axis: t.axis }
    : { kind: 'linear', thickness_mm: t.thickness_mm, end_thickness_mm: t.end_thickness_mm !== t.thickness_mm ? t.end_thickness_mm : Math.max(0, t.thickness_mm * 0.25), axis: t.axis }
}

/** Set the uniform thickness (keeps the unused end in sync so toggling to linear starts flat). */
export function setUniformThickness(layer: BlockLayer, t: number) {
  layer.thickness.thickness_mm = t
  if (layer.thickness.kind === 'uniform') layer.thickness.end_thickness_mm = t
}

export const flipWedge = (layer: BlockLayer) => {
  const t = layer.thickness
  ;[t.thickness_mm, t.end_thickness_mm] = [t.end_thickness_mm, t.thickness_mm]
}
