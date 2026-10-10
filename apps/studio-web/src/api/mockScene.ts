/** Tiny analytic ray-caster used by the mock engine: renders per-pixel material / depth / incidence maps
 *  of the experiment from the HSI camera's point of view (same intrinsics as the 3D viewport). */
import * as THREE from 'three'
import type { Experiment, SceneObject } from '../generated/experiment'
import { buildThreeCamera } from '../lib/camera'
import { beamAt } from '../lib/lighting'
import { activeCamera, activeSource } from '../state/store'

export interface SceneMaps {
  width: number
  height: number
  /** index into `materials`, −1 = void */
  mat: Int16Array
  /** index into exp.objects, −1 = background plane / void */
  obj: Int16Array
  depth: Float32Array
  incidence: Float32Array
  irradiance: Float32Array
  materials: string[]
}

interface Hit { t: number; n: [number, number, number]; mat: string; vol?: boolean; chord?: number }

function rayBox(o: number[], d: number[], lo: number[], hi: number[]): { t: number; n: [number, number, number]; chord: number } | null {
  let t0 = -Infinity, t1 = Infinity
  let axis = 0, sign = 1
  for (let a = 0; a < 3; a++) {
    if (Math.abs(d[a]) < 1e-12) { if (o[a] < lo[a] || o[a] > hi[a]) return null; continue }
    let ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a], s = -1
    if (ta > tb) { [ta, tb] = [tb, ta]; s = 1 }
    if (ta > t0) { t0 = ta; axis = a; sign = s }
    if (tb < t1) t1 = tb
    if (t0 > t1) return null
  }
  if (t0 < 0) return null
  const n: [number, number, number] = [0, 0, 0]; n[axis] = sign
  return { t: t0, n, chord: Math.max(0, t1 - t0) }
}

function rayEllipsoid(o: number[], d: number[], r: number[], yMin = -Infinity): { t: number; n: [number, number, number] } | null {
  const ox = o[0] / r[0], oy = o[1] / r[1], oz = o[2] / r[2]
  const dx = d[0] / r[0], dy = d[1] / r[1], dz = d[2] / r[2]
  const a = dx * dx + dy * dy + dz * dz, b = 2 * (ox * dx + oy * dy + oz * dz), c = ox * ox + oy * oy + oz * oz - 1
  const disc = b * b - 4 * a * c
  if (disc < 0) return null
  const sq = Math.sqrt(disc)
  for (const t of [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]) {
    if (t <= 0) continue
    const y = o[1] + t * d[1]
    if (y < yMin) continue
    const px = (o[0] + t * d[0]) / (r[0] * r[0]), py = y / (r[1] * r[1]), pz = (o[2] + t * d[2]) / (r[2] * r[2])
    const l = Math.hypot(px, py, pz) || 1
    return { t, n: [px / l, py / l, pz / l] }
  }
  return null
}

function rayCylinder(o: number[], d: number[], R: number, y0: number, y1: number): { t: number; n: [number, number, number] } | null {
  let best: { t: number; n: [number, number, number] } | null = null
  const a = d[0] * d[0] + d[2] * d[2], b = 2 * (o[0] * d[0] + o[2] * d[2]), c = o[0] * o[0] + o[2] * o[2] - R * R
  if (a > 1e-12) {
    const disc = b * b - 4 * a * c
    if (disc >= 0) {
      const sq = Math.sqrt(disc)
      for (const t of [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]) {
        if (t <= 0) continue
        const y = o[1] + t * d[1]
        if (y >= y0 && y <= y1) { const x = o[0] + t * d[0], z = o[2] + t * d[2]; best = { t, n: [x / R, 0, z / R] }; break }
      }
    }
  }
  if (Math.abs(d[1]) > 1e-12) {
    for (const [yy, ny] of [[y1, 1], [y0, -1]] as const) {
      const t = (yy - o[1]) / d[1]
      if (t <= 0 || (best && t >= best.t)) continue
      const x = o[0] + t * d[0], z = o[2] + t * d[2]
      if (x * x + z * z <= R * R) best = { t, n: [0, ny, 0] }
    }
  }
  return best
}

function intersectObject(obj: SceneObject, o: number[], d: number[]): Hit | null {
  const s = obj.shape
  const m = obj.material_id, v = obj.vessel_material_id ?? obj.material_id
  switch (s.kind) {
    case 'box': {
      const h = s.size_mm.map((x) => x / 2)
      const r = rayBox(o, d, h.map((x) => -x), h)
      return r ? { ...r, mat: m } : null
    }
    case 'sphere': { const r = rayEllipsoid(o, d, [s.radius_mm, s.radius_mm, s.radius_mm]); return r ? { ...r, mat: m } : null }
    case 'cylinder': { const r = rayCylinder(o, d, s.radius_mm, -s.height_mm / 2, s.height_mm / 2); return r ? { ...r, mat: m } : null }
    case 'droplet': { const r = rayEllipsoid([o[0], o[1], o[2]], d, [s.base_radius_mm, s.height_mm, s.base_radius_mm], 0); return r ? { ...r, mat: m } : null }
    case 'petri_dish': {
      const Ro = s.inner_radius_mm + s.wall_mm
      const liq = rayCylinder(o, d, s.inner_radius_mm, s.base_mm, s.base_mm + s.fill_height_mm)
      if (liq) return { ...liq, mat: m, vol: true, chord: 2 * s.inner_radius_mm }      // glass walls are thin and clear: show the contents
      const r = rayCylinder(o, d, Ro, 0, s.height_mm)
      return r ? { ...r, mat: v } : null
    }
    case 'cuvette': {
      const [ix, iy, iz] = s.inner_mm, w = s.wall_mm
      const tl = rayBox(o, d, [-ix / 2, w, -iz / 2], [ix / 2, w + s.fill_height_mm, iz / 2])
      if (tl) return { ...tl, mat: m, vol: true }  // tl.chord = path length through the liquid
      const r = rayBox(o, d, [-(ix / 2 + w), 0, -(iz / 2 + w)], [ix / 2 + w, iy + w, iz / 2 + w])
      return r ? { ...r, mat: v } : null
    }
  }
}

const VOLUME_SAMPLES = 24

export function renderMaps(exp: Experiment, width: number, height: number): SceneMaps {
  const cam = activeCamera(exp), src = activeSource(exp)
  const tc = buildThreeCamera(cam)
  const tanY = Math.tan((tc.fov * Math.PI) / 360)
  const aspect = tc.aspect
  const camMat = new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(tc.quaternion))
  const ce = camMat.elements
  const eye = cam.position_mm
  const objs = exp.objects.map((ob) => {
    const R = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(...ob.transform.rotation_deg.map((x) => (x * Math.PI) / 180) as [number, number, number], 'XYZ'))
    const Ri = new THREE.Matrix3().setFromMatrix4(R.clone().invert()).elements
    const Rw = new THREE.Matrix3().setFromMatrix4(R).elements
    return { ob, Ri, Rw, p: ob.transform.position_mm }
  })
  const materials: string[] = []
  const mid = (id: string) => { let i = materials.indexOf(id); if (i < 0) { materials.push(id); i = materials.length - 1 } return i }
  const bg = exp.background
  const N = width * height
  const out: SceneMaps = {
    width, height, materials, mat: new Int16Array(N).fill(-1), obj: new Int16Array(N).fill(-1),
    depth: new Float32Array(N), incidence: new Float32Array(N), irradiance: new Float32Array(N),
  }
  const mul = (e: number[], x: number, y: number, z: number): [number, number, number] =>
    [e[0] * x + e[3] * y + e[6] * z, e[1] * x + e[4] * y + e[7] * z, e[2] * x + e[5] * y + e[8] * z]
  for (let j = 0; j < height; j++) {
    const ny = (1 - ((j + 0.5) / height) * 2) * tanY
    for (let i = 0; i < width; i++) {
      const nx = (((i + 0.5) / width) * 2 - 1) * tanY * aspect
      const dl = mul(ce, nx, ny, -1)
      const dl_len = Math.hypot(dl[0], dl[1], dl[2])
      const d = [dl[0] / dl_len, dl[1] / dl_len, dl[2] / dl_len]
      let best: { t: number; n: [number, number, number]; mat: string; oi: number; vol?: boolean; chord?: number } | null = null
      objs.forEach(({ ob, Ri, Rw, p }, oi) => {
        if (!ob.visible) return
        const ol = mul(Ri, eye[0] - p[0], eye[1] - p[1], eye[2] - p[2])
        const dd = mul(Ri, d[0], d[1], d[2])
        const h = intersectObject(ob, ol, dd)
        if (h && (!best || h.t < best.t)) best = { t: h.t, n: mul(Rw, ...h.n), mat: h.mat, oi, vol: h.vol, chord: h.chord }
      })
      if (bg.kind === 'plane' && bg.material_id && Math.abs(d[1]) > 1e-9) {
        const t = (0 - eye[1]) / d[1]
        if (t > 0 && (!best || t < (best as { t: number }).t)) {
          const x = eye[0] + t * d[0], z = eye[2] + t * d[2]
          if (Math.abs(x) < bg.extent_mm / 2 && Math.abs(z) < bg.extent_mm / 2) best = { t, n: [0, 1, 0], mat: bg.material_id, oi: -1 }
        }
      }
      if (!best) continue
      const b = best as { t: number; n: [number, number, number]; mat: string; oi: number; vol?: boolean; chord?: number }
      const k = j * width + i
      out.mat[k] = mid(b.mat); out.obj[k] = b.oi; out.depth[k] = b.t
      const pt = [eye[0] + b.t * d[0], eye[1] + b.t * d[1], eye[2] + b.t * d[2]]
      const s = beamAt(src, pt, b.vol ? null : b.n)
      let irr = s.irradiance
      if (b.vol && b.chord && b.chord > 0) {
        // A clear liquid is lit throughout its volume, not just on the face the camera sees: a narrow beam
        // (laser) crosses the cuvette mid-plane and would be missed by a surface sample. Average the beam
        // irradiance along the view chord (fluorescence signal ∝ ∫ I dl).
        const n = VOLUME_SAMPLES
        let acc = 0
        for (let q = 0; q < n; q++) {
          const tq = b.t + ((q + 0.5) / n) * b.chord
          acc += beamAt(src, [eye[0] + tq * d[0], eye[1] + tq * d[1], eye[2] + tq * d[2]], null).irradiance
        }
        irr = acc / n
      }
      out.irradiance[k] = irr; out.incidence[k] = s.incidenceDeg
    }
  }
  return out
}
