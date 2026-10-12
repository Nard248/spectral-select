import type { SceneObject } from '../generated/experiment'
import { viewApi } from './viewApi'
import { useLayerView, DEFAULT_SECTION, type SectionState } from '../state/layerView'

/** Fly the camera to a 3/4 view that looks straight onto the cut face of a block's section (from the removed side). */
export function frameSection(obj: SceneObject, sec: Pick<SectionState, 'axis' | 'keep'>, ms = 900) {
  if (obj.shape.kind !== 'layered_block') return
  const [fx, fz] = obj.shape.footprint_mm
  const p = obj.transform.position_mm
  const sign = sec.keep === 'low' ? 1 : -1
  const dir = sec.axis === 'z' ? [0.32, 0.5, sign] : [sign, 0.5, 0.32]
  const n = Math.hypot(dir[0], dir[1], dir[2])
  const dist = 1.85 * Math.max(fx, fz) + 26
  const d = [dir[0] / n, dir[1] / n, dir[2] / n]
  // slide the view so the block sits right of centre: the layer callouts hang off its left end, clear of the toolbar
  const f = [-d[0], -d[1], -d[2]]
  const r = [f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0]   // forward × up
  const rl = Math.hypot(r[0], r[1], r[2]) || 1
  const sh = (-0.05 * dist) / rl
  const target: [number, number, number] = [p[0] + r[0] * sh, p[1] + 3, p[2] + r[2] * sh]
  void viewApi.current?.flyTo([target[0] + d[0] * dist, target[1] + d[1] * dist, target[2] + d[2] * dist], target, ms)
}

/** Switch the cutaway of a block on/off (framing the camera when it turns on). */
export function setCutaway(obj: SceneObject, on: boolean, patch: Partial<SectionState> = {}) {
  const lv = useLayerView.getState()
  const cur = lv.sections[obj.id] ?? DEFAULT_SECTION
  lv.setSection(obj.id, { ...patch, on })
  if (on && !cur.on) frameSection(obj, { axis: patch.axis ?? cur.axis, keep: patch.keep ?? cur.keep })
}
