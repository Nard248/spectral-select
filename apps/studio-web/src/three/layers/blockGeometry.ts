/** Exact mesh data for a layered block — pure typed arrays (no three.js), unit-testable.
 *
 *  Every interface  y_i(x, z) = Σ_{j≤i} t_j(x, z)  is a plane, so a layer is a hexahedron with a planar top and bottom.
 *  Interfaces are emitted ONCE (as the TOP face of the layer below them) so nothing z-fights, and side walls skip
 *  zero-area triangles where a layer pinches out to nothing — no degenerate geometry. A `region` (sub-rectangle of
 *  the footprint) lets the cutaway view build the kept half of the block with the same code; the wall at the cut is
 *  the capped section face. */
import type { LayeredBlockShape } from '../../generated/experiment'
import { interfaceHeights, layerThicknesses, type Axis } from './profile'

export interface Region { x0: number; x1: number; z0: number; z1: number }
export type Side = '-x' | '+x' | '-z' | '+z'

export interface Patch {
  positions: Float32Array
  normals: Float32Array
  /** thickness of the owning layer at each vertex (drives the physical opacity) */
  thickness: Float32Array
  indices: Uint32Array
}
export interface LayerPatches {
  index: number
  top: Patch
  sides: Record<Side, Patch | null>
  /** only layer 0 carries a bottom face (the block underside) */
  bottom: Patch | null
}
export interface BlockMeshData {
  region: Region
  layers: LayerPatches[]
  /** line segments (pairs of xyz) tracing every interface outline and the vertical corner edges */
  edges: Float32Array
  /** outline of the whole kept block (12 edges) */
  hull: Float32Array
  /** iso-thickness lines on the top face of every wedge layer (every 'nice' step), as line-segment pairs */
  contours: Float32Array
}

const EPS_AREA = 1e-10
const sub3 = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross3 = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const dot3 = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const norm3 = (a: number[]) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l] }

class PatchBuilder {
  pos: number[] = []; nor: number[] = []; thk: number[] = []; idx: number[] = []
  vertex(p: number[], n: number[], t: number) { this.pos.push(p[0], p[1], p[2]); this.nor.push(n[0], n[1], n[2]); this.thk.push(t); return this.pos.length / 3 - 1 }
  /** triangle wound so its geometric normal agrees with `n`; zero-area triangles are dropped */
  tri(a: number, b: number, c: number, n: number[]) {
    const P = (i: number) => [this.pos[3 * i], this.pos[3 * i + 1], this.pos[3 * i + 2]]
    const g = cross3(sub3(P(b), P(a)), sub3(P(c), P(a)))
    if (Math.hypot(g[0], g[1], g[2]) < EPS_AREA) return
    if (dot3(g, n) >= 0) this.idx.push(a, b, c); else this.idx.push(a, c, b)
  }
  build(): Patch | null {
    if (!this.idx.length) return null
    return { positions: new Float32Array(this.pos), normals: new Float32Array(this.nor), thickness: new Float32Array(this.thk), indices: new Uint32Array(this.idx) }
  }
}

/** Grid face over `region` at height h(x, z) (a plane). `nSeg` segments per axis. */
function gridFace(region: Region, h: (x: number, z: number) => number, thick: (x: number, z: number) => number, nSeg: number, up: boolean): Patch {
  const b = new PatchBuilder()
  const { x0, x1, z0, z1 } = region
  // plane normal from the gradient
  const dhx = (h(x1, (z0 + z1) / 2) - h(x0, (z0 + z1) / 2)) / (x1 - x0 || 1)
  const dhz = (h((x0 + x1) / 2, z1) - h((x0 + x1) / 2, z0)) / (z1 - z0 || 1)
  const n0 = norm3([-dhx, 1, -dhz]); const n = up ? n0 : [-n0[0], -n0[1], -n0[2]]
  const id: number[][] = []
  for (let i = 0; i <= nSeg; i++) {
    id.push([])
    for (let j = 0; j <= nSeg; j++) {
      const x = x0 + ((x1 - x0) * i) / nSeg, z = z0 + ((z1 - z0) * j) / nSeg
      id[i].push(b.vertex([x, h(x, z), z], n, thick(x, z)))
    }
  }
  for (let i = 0; i < nSeg; i++) for (let j = 0; j < nSeg; j++) {
    b.tri(id[i][j], id[i + 1][j], id[i][j + 1], n)
    b.tri(id[i + 1][j], id[i + 1][j + 1], id[i][j + 1], n)
  }
  return b.build()!
}

/** Vertical wall at one side of `region`, between the layer's bottom and top interface. */
function sideWall(region: Region, side: Side, lo: (x: number, z: number) => number, hi: (x: number, z: number) => number,
  thick: (x: number, z: number) => number, nSeg: number): Patch | null {
  const { x0, x1, z0, z1 } = region
  const alongX = side === '-z' || side === '+z'
  const fixed = side === '-x' ? x0 : side === '+x' ? x1 : side === '-z' ? z0 : z1
  const a0 = alongX ? x0 : z0, a1 = alongX ? x1 : z1
  const out = side === '-x' ? [-1, 0, 0] : side === '+x' ? [1, 0, 0] : side === '-z' ? [0, 0, -1] : [0, 0, 1]
  const P = (s: number) => { const u = a0 + ((a1 - a0) * s) / nSeg; return alongX ? { x: u, z: fixed } : { x: fixed, z: u } }
  const b = new PatchBuilder()
  const bot: number[] = [], top: number[] = []
  for (let s = 0; s <= nSeg; s++) {
    const { x, z } = P(s); const t = thick(x, z)
    bot.push(b.vertex([x, lo(x, z), z], out, t))
    top.push(b.vertex([x, hi(x, z), z], out, t))
  }
  for (let s = 0; s < nSeg; s++) {
    b.tri(bot[s], bot[s + 1], top[s + 1], out)
    b.tri(bot[s], top[s + 1], top[s], out)
  }
  return b.build()
}

/** 1–2–5 step ≥ x (mm) for contour spacing. */
export function niceStep(x: number): number {
  if (!(x > 0)) return 1
  const e = Math.pow(10, Math.floor(Math.log10(x))), m = x / e
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * e
}

/** Build the full mesh data for `shape` over `region` (default: whole footprint). */
export function buildBlockMesh(shape: LayeredBlockShape, region?: Partial<Region>, nSeg = 24): BlockMeshData {
  const fp = shape.footprint_mm
  const r: Region = { x0: -fp[0] / 2, x1: fp[0] / 2, z0: -fp[1] / 2, z1: fp[1] / 2, ...region }
  const n = shape.layers.length
  const H = (i: number) => (x: number, z: number) => interfaceHeights(shape, x, z)[i]
  const T = (i: number) => (x: number, z: number) => layerThicknesses(shape, x, z)[i]
  const layers: LayerPatches[] = []
  for (let i = 0; i < n; i++) {
    const sides = {} as Record<Side, Patch | null>
    for (const s of ['-x', '+x', '-z', '+z'] as Side[]) sides[s] = sideWall(r, s, H(i), H(i + 1), T(i), nSeg)
    layers.push({
      index: i,
      top: gridFace(r, H(i + 1), T(i), nSeg, true),
      sides,
      bottom: i === 0 ? gridFace(r, () => 0, T(0), 1, false) : null,
    })
  }
  const edges: number[] = []
  const seg = (a: number[], b: number[]) => { if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) > 1e-9) edges.push(...a, ...b) }
  const cs: [number, number][] = [[r.x0, r.z0], [r.x1, r.z0], [r.x1, r.z1], [r.x0, r.z1]]
  for (let k = 0; k <= n; k++) {
    for (let c = 0; c < 4; c++) {
      const [xa, za] = cs[c], [xb, zb] = cs[(c + 1) % 4]
      seg([xa, H(k)(xa, za), za], [xb, H(k)(xb, zb), zb])
    }
  }
  for (const [x, z] of cs) for (let k = 0; k < n; k++) seg([x, H(k)(x, z), z], [x, H(k + 1)(x, z), z])
  const hull: number[] = []
  const hs = (a: number[], b: number[]) => hull.push(...a, ...b)
  for (let c = 0; c < 4; c++) {
    const [xa, za] = cs[c], [xb, zb] = cs[(c + 1) % 4]
    hs([xa, 0, za], [xb, 0, zb]); hs([xa, H(n)(xa, za), za], [xb, H(n)(xb, zb), zb]); hs([xa, 0, za], [xa, H(n)(xa, za), za])
  }
  // iso-thickness contours of wedge layers: the top face of layer i is a plane, lines of constant t_i run across the wedge axis
  const contours: number[] = []
  shape.layers.forEach((l, i) => {
    const t = l.thickness
    if (t.kind !== 'linear' || t.thickness_mm === t.end_thickness_mm) return
    const size = t.axis === 'x' ? fp[0] : fp[1]
    const lo = Math.min(t.thickness_mm, t.end_thickness_mm), hi = Math.max(t.thickness_mm, t.end_thickness_mm)
    const step = niceStep((hi - lo) / 6)
    for (let v = Math.ceil((lo + 1e-9) / step) * step; v < hi - 1e-9; v += step) {
      const f = (v - t.thickness_mm) / (t.end_thickness_mm - t.thickness_mm)
      const u = (f - 0.5) * size
      if (t.axis === 'x') { if (u < r.x0 || u > r.x1) continue; contours.push(u, H(i + 1)(u, r.z0), r.z0, u, H(i + 1)(u, r.z1), r.z1) }
      else { if (u < r.z0 || u > r.z1) continue; contours.push(r.x0, H(i + 1)(r.x0, u), u, r.x1, H(i + 1)(r.x1, u), u) }
    }
  })
  return { region: r, layers, edges: new Float32Array(edges), hull: new Float32Array(hull), contours: new Float32Array(contours) }
}

/** The kept sub-rectangle of the footprint for a cutaway at local coordinate `c` along `axis`.
 *  keep 'low' retains u ≤ c (the cut wall is the '+axis' side); 'high' retains u ≥ c (the '-axis' side). */
export function cutRegion(fp: readonly [number, number], axis: Axis, c: number, keep: 'low' | 'high'): Region {
  const half = axis === 'x' ? fp[0] / 2 : fp[1] / 2
  const cc = Math.min(Math.max(c, -half), half)
  const full: Region = { x0: -fp[0] / 2, x1: fp[0] / 2, z0: -fp[1] / 2, z1: fp[1] / 2 }
  if (axis === 'x') return keep === 'low' ? { ...full, x1: cc } : { ...full, x0: cc }
  return keep === 'low' ? { ...full, z1: cc } : { ...full, z0: cc }
}
export const cutSide = (axis: Axis, keep: 'low' | 'high'): Side => (keep === 'low' ? (axis === 'x' ? '+x' : '+z') : (axis === 'x' ? '-x' : '-z'))

/** Strip geometry laid ON the section plane for the light-field texture. The strip of layer `l` is a (nw+1)×(rows+1)
 *  grid; `uv.x` = fraction along the horizontal axis, `uv.y` = (l + fraction across the layer) / nLayers. */
export interface StripData { positions: Float32Array; uvs: Float32Array; indices: Uint32Array; normal: [number, number, number] }
export function buildSectionStrips(shape: LayeredBlockShape, axis: Axis, c: number, keep: 'low' | 'high', nw = 48, rows = 8, padV = 0): StripData[] {
  const fp = shape.footprint_mm
  const w0 = axis === 'x' ? -fp[1] / 2 : -fp[0] / 2, w1 = -w0
  const nL = shape.layers.length
  const nrm: [number, number, number] = axis === 'x' ? [keep === 'low' ? 1 : -1, 0, 0] : [0, 0, keep === 'low' ? 1 : -1]
  const out: StripData[] = []
  for (let l = 0; l < nL; l++) {
    const pos: number[] = [], uv: number[] = [], idx: number[] = []
    for (let j = 0; j <= nw; j++) {
      const w = w0 + ((w1 - w0) * j) / nw
      const [x, z] = axis === 'x' ? [c, w] : [w, c]
      const h = interfaceHeights(shape, x, z)
      for (let s = 0; s <= rows; s++) {
        const f = s / rows
        pos.push(x, h[l] + (h[l + 1] - h[l]) * f, z)
        uv.push(j / nw, (l + padV + f * (1 - 2 * padV)) / nL)
      }
    }
    for (let j = 0; j < nw; j++) for (let s = 0; s < rows; s++) {
      const a = j * (rows + 1) + s, b = (j + 1) * (rows + 1) + s, c2 = a + 1, d = b + 1
      // orient towards `nrm`
      const P = (i: number) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]
      const g = cross3(sub3(P(b), P(a)), sub3(P(c2), P(a)))
      if (dot3(g, nrm) >= 0) idx.push(a, b, c2, b, d, c2); else idx.push(a, c2, b, b, c2, d)
    }
    out.push({ positions: new Float32Array(pos), uvs: new Float32Array(uv), indices: new Uint32Array(idx), normal: nrm })
  }
  return out
}
