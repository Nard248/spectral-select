/** Pure geometry / rasterisation for the light-field viewer. Volumes are float32 (nx, nz, nd) C-order with
 *  nd = n_layers · sublayers; depth index d = layer·S + s, layers bottom → top, sub-layers bottom → top inside a layer.
 *  Cells are layer-conforming, so drawing in PHYSICAL coordinates maps every canvas pixel back to (layer, sublayer)
 *  through the local layer thicknesses — wedges therefore look like wedges. */
import type { LayeredBlockShape } from '../generated/experiment'
import type { VolumeInfo } from '../api/types'
import { layerDepthTop, stackAt, topPlane } from './blockGeometry'
import { lut, type CmapName } from '../ui/colormaps'

export interface VolGeom {
  nx: number; nz: number; nd: number; S: number; L: number
  x0: number; x1: number; z0: number; z1: number
  Lx: number; Lz: number
  shape: Pick<LayeredBlockShape, 'footprint_mm' | 'layers'>
  ymax: number
}

export function geometry(info: VolumeInfo, shape: Pick<LayeredBlockShape, 'footprint_mm' | 'layers'>): VolGeom {
  const [nx, nz, nd] = info.dims
  const x0 = info.x_edges_mm[0], x1 = info.x_edges_mm[nx], z0 = info.z_edges_mm[0], z1 = info.z_edges_mm[nz]
  return { nx, nz, nd, S: info.sublayers, L: info.layer_names.length, x0, x1, z0, z1, Lx: x1 - x0, Lz: z1 - z0, shape, ymax: topPlane(shape).max }
}

/** Does the shape's layer list agree with the volume's layer names? */
export const shapeMatches = (info: VolumeInfo, shape: Pick<LayeredBlockShape, 'layers'>) =>
  shape.layers.length === info.layer_names.length && shape.layers.every((l, i) => l.name === info.layer_names[i])

/** ImageData where available (browser), a structurally identical buffer elsewhere (unit tests). */
const mkImg = (w: number, h: number): ImageData => (typeof ImageData !== 'undefined' ? new ImageData(w, h) : ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h } as ImageData))
const clampI = (v: number, n: number) => Math.min(n - 1, Math.max(0, v))

/** Volume cell (ix, iz, d) containing the physical point (x, y, z), or null above the surface / outside the footprint. */
export function locate(g: VolGeom, x: number, y: number, z: number): { ix: number; iz: number; d: number; layer: number } | null {
  if (x < g.x0 || x > g.x1 || z < g.z0 || z > g.z1 || y < 0) return null
  const st = stackAt(g.shape, x, z)
  if (y > st.top) return null
  let layer = st.t.length - 1
  for (let i = 0; i < st.t.length; i++) if (y < st.bottom[i] + st.t[i] || i === st.t.length - 1) { if (st.t[i] > 1e-9 && y >= st.bottom[i] - 1e-9) { layer = i; break } }
  const t = st.t[layer]
  const s = clampI(Math.floor(((y - st.bottom[layer]) / (t || 1)) * g.S), g.S)
  return { ix: clampI(Math.floor(((x - g.x0) / g.Lx) * g.nx), g.nx), iz: clampI(Math.floor(((z - g.z0) / g.Lz) * g.nz), g.nz), d: layer * g.S + s, layer }
}

export interface ColorScale { vmin: number; vmax: number; log: boolean; cmap: CmapName }

export function autoScale(data: Float32Array, log: boolean, decades: number, cmap: CmapName): ColorScale {
  let mx = 0
  for (let i = 0; i < data.length; i += 3) if (data[i] > mx) mx = data[i]
  if (!(mx > 0)) mx = 1
  return { vmax: mx, vmin: log ? mx * 10 ** -decades : 0, log, cmap }
}

export function colorIndex(v: number, sc: ColorScale): number {
  if (!(v > 0) && sc.log) return 0
  const t = sc.log ? Math.log10(Math.max(v, 1e-300) / sc.vmin) / Math.log10(sc.vmax / sc.vmin) : (v - sc.vmin) / (sc.vmax - sc.vmin || 1)
  return Math.round(Math.min(1, Math.max(0, t)) * 255)
}

const BG: [number, number, number] = [10, 12, 16]

/** XZ slice (depth vs x) at footprint position `z` into an RGBA buffer of size w×h; y spans [0, ymax] with the top at row 0. */
export function renderXZ(g: VolGeom, data: Float32Array, z: number, w: number, h: number, sc: ColorScale): ImageData {
  const img = mkImg(w, h)
  const L = lut(sc.cmap)
  const iz = clampI(Math.floor(((z - g.z0) / g.Lz) * g.nz), g.nz)
  for (let u = 0; u < w; u++) {
    const x = g.x0 + ((u + 0.5) / w) * g.Lx
    const ix = clampI(Math.floor(((x - g.x0) / g.Lx) * g.nx), g.nx)
    const st = stackAt(g.shape, x, z)
    const base = (ix * g.nz + iz) * g.nd
    for (let v = 0; v < h; v++) {
      const y = g.ymax * (1 - (v + 0.5) / h)
      const o = (v * w + u) * 4
      if (y > st.top) { img.data[o] = BG[0]; img.data[o + 1] = BG[1]; img.data[o + 2] = BG[2]; img.data[o + 3] = 255; continue }
      let layer = 0
      for (let i = st.t.length - 1; i >= 0; i--) if (st.t[i] > 1e-9 && y >= st.bottom[i]) { layer = i; break }
      const s = clampI(Math.floor(((y - st.bottom[layer]) / (st.t[layer] || 1)) * g.S), g.S)
      const val = data[base + layer * g.S + s]
      const ci = colorIndex(val, sc) * 3
      img.data[o] = L[ci]; img.data[o + 1] = L[ci + 1]; img.data[o + 2] = L[ci + 2]; img.data[o + 3] = 255
    }
  }
  return img
}

/** XY (plan) slice at height `y` above the block bottom; boundaries between layers are drawn as light edges. */
export function renderXY(g: VolGeom, data: Float32Array, y: number, w: number, h: number, sc: ColorScale): ImageData {
  const img = mkImg(w, h)
  const L = lut(sc.cmap)
  const layerMap = new Int8Array(w * h).fill(-1)
  for (let v = 0; v < h; v++) {
    const z = g.z0 + ((v + 0.5) / h) * g.Lz
    for (let u = 0; u < w; u++) {
      const x = g.x0 + ((u + 0.5) / w) * g.Lx
      const o = (v * w + u) * 4
      const c = locate(g, x, y, z)
      if (!c) { img.data[o] = BG[0]; img.data[o + 1] = BG[1]; img.data[o + 2] = BG[2]; img.data[o + 3] = 255; continue }
      layerMap[v * w + u] = c.layer
      const ci = colorIndex(data[(c.ix * g.nz + c.iz) * g.nd + c.d], sc) * 3
      img.data[o] = L[ci]; img.data[o + 1] = L[ci + 1]; img.data[o + 2] = L[ci + 2]; img.data[o + 3] = 255
    }
  }
  for (let v = 0; v < h - 1; v++) for (let u = 0; u < w - 1; u++) {
    const a = layerMap[v * w + u]
    if (a !== layerMap[v * w + u + 1] || a !== layerMap[(v + 1) * w + u]) {
      const o = (v * w + u) * 4
      for (let c = 0; c < 3; c++) img.data[o + c] = img.data[o + c] * 0.35 + 255 * 0.65
    }
  }
  return img
}

export interface DepthProfile {
  layer: number
  /** mean depth below the top surface of each sub-layer centre, mean / std of the value over the lateral cells that have this layer */
  depth: number[]; mean: number[]; std: number[]
  top: number; bottom: number   // mean depth span of the layer
}

/** Lateral-average depth profile per layer (cells where a layer has ~zero thickness are skipped). */
export function depthProfiles(g: VolGeom, data: Float32Array): DepthProfile[] {
  const out: DepthProfile[] = []
  const stacks: ReturnType<typeof stackAt>[] = []
  for (let ix = 0; ix < g.nx; ix++) for (let iz = 0; iz < g.nz; iz++) stacks.push(stackAt(g.shape, g.x0 + ((ix + 0.5) / g.nx) * g.Lx, g.z0 + ((iz + 0.5) / g.nz) * g.Lz))
  for (let i = 0; i < g.L; i++) {
    const depth = new Array<number>(g.S).fill(0), mean = new Array<number>(g.S).fill(0), m2 = new Array<number>(g.S).fill(0)
    let n = 0, dTop = 0, dBot = 0
    for (let ix = 0; ix < g.nx; ix++) for (let iz = 0; iz < g.nz; iz++) {
      const st = stacks[ix * g.nz + iz]
      if (st.t[i] < 1e-6) continue
      n++
      const top = layerDepthTop(st, i)
      dTop += top; dBot += top + st.t[i]
      const base = (ix * g.nz + iz) * g.nd + i * g.S
      for (let s = 0; s < g.S; s++) {
        const val = data[base + s]
        depth[s] += top + st.t[i] * (1 - (s + 0.5) / g.S)
        mean[s] += val; m2[s] += val * val
      }
    }
    if (!n) continue
    out.push({
      layer: i, depth: depth.map((v) => v / n), mean: mean.map((v) => v / n),
      std: mean.map((v, s) => Math.sqrt(Math.max(0, m2[s] / n - (v / n) ** 2))), top: dTop / n, bottom: dBot / n,
    })
  }
  return out
}
