/** Mock implementations of the v1.1 analysis endpoints (EEM / ROI / line / response) on top of an abstract capture accessor. */
import type { CaptureMeta, EemRequest, EemResult, LineProfile, LineRequest, MetricSpec, NormSpec, ResponseCurve, ResponseRequest, RoiRequest, RoiSpec, RoiStats } from './types'
import { maskIndices } from '../analysis/roi'
import { normUnit } from '../analysis/norm'

export interface CaptureAccess {
  meta: CaptureMeta
  /** Float32 H·W·B (band fastest) in normalised units for excitation `ex` */
  cube(ex: number, norm: NormSpec): Float32Array
  /** numeric ground-truth raster H·W (NaN where undefined) or null if unknown */
  gt(name: string): Float32Array | null
  /** mm per pixel along the image, per pixel (for line profiles) */
  pxMm: Float32Array | null
  /** average component decomposition over the given pixel indices (subsampled by the caller) */
  components(indices: ArrayLike<number>): Record<string, Record<string, number[]>> | null
}

const normOf = (n?: NormSpec | null): NormSpec => n ?? { mode: 'adu' }

function median(a: number[]): number {
  if (!a.length) return NaN
  const s = Float64Array.from(a).sort()
  const m = s.length >> 1
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m])
}

/** Resample the per-excitation EEM onto the union emission grid (2 nm), NaN where a band set does not reach. */
export function gridEem(exs: number[], emission: Record<string, number[]>, value: Record<string, number[]>): { grid: number[]; z: number[][] } {
  let lo = Infinity, hi = -Infinity
  for (const e of exs) { const em = emission[String(e)]; if (em?.length) { lo = Math.min(lo, em[0]); hi = Math.max(hi, em[em.length - 1]) } }
  if (!Number.isFinite(lo)) return { grid: [], z: [] }
  const grid: number[] = []
  for (let g = lo; g <= hi + 1e-9; g += 2) grid.push(g)
  const z = exs.map((e) => {
    const em = emission[String(e)], v = value[String(e)]
    return grid.map((g) => {
      if (g < em[0] - 1e-9 || g > em[em.length - 1] + 1e-9) return NaN
      let j = 0
      while (j < em.length - 2 && em[j + 1] < g) j++
      const t = em.length > 1 ? (g - em[j]) / (em[j + 1] - em[j] || 1) : 0
      return em.length > 1 ? v[j] * (1 - t) + v[j + 1] * t : v[0]
    })
  })
  return { grid, z }
}

export function mockEem(c: CaptureAccess, req: EemRequest): EemResult {
  const norm = normOf(req.norm)
  const { width: W, height: H } = c.meta
  const roi = req.roi
  const idx = maskIndices(roi as Exclude<RoiSpec, never>, W, H)
  if (!idx.length) throw new Error('ROI contains no pixels inside the image')
  const exs = c.meta.excitations_nm
  const value: Record<string, number[]> = {}, std: Record<string, number[]> = {}
  for (const ex of exs) {
    const key = String(ex), B = c.meta.emission_nm[key].length
    const cube = c.cube(ex, norm)
    const v: number[] = [], sd: number[] = []
    for (let b = 0; b < B; b++) {
      let s = 0, s2 = 0
      const vals: number[] | null = req.stat === 'median' ? [] : null
      for (let q = 0; q < idx.length; q++) { const x = cube[idx[q] * B + b]; s += x; s2 += x * x; vals?.push(x) }
      const mean = s / idx.length
      v.push(vals ? median(vals) : mean)
      sd.push(Math.sqrt(Math.max(0, s2 / idx.length - mean * mean)))
    }
    value[key] = v; std[key] = sd
  }
  const g = gridEem(exs, c.meta.emission_nm, value)
  return { roi, n_pixels: idx.length, norm, unit: normUnit(norm.mode), excitations_nm: exs, emission_nm: c.meta.emission_nm, value, std, grid_emission_nm: g.grid, grid_value: g.z }
}

export function mockRoi(c: CaptureAccess, req: RoiRequest): RoiStats {
  const base = mockEem(c, req)
  const { width: W, height: H } = c.meta
  const idx = maskIndices(req.roi, W, H)
  const ground_truth: RoiStats['ground_truth'] = {}
  for (const name of c.meta.ground_truth_layers) {
    const r = c.gt(name); if (!r) continue
    let n = 0, s = 0, s2 = 0, mn = Infinity, mx = -Infinity
    for (let q = 0; q < idx.length; q++) { const v = r[idx[q]]; if (!Number.isFinite(v)) continue; n++; s += v; s2 += v * v; mn = Math.min(mn, v); mx = Math.max(mx, v) }
    if (!n) continue
    const mean = s / n
    ground_truth[name] = { mean, std: Math.sqrt(Math.max(0, s2 / n - mean * mean)), min: mn, max: mx }
  }
  let components: RoiStats['components'] = null
  if (req.components) {
    const stride = Math.max(1, Math.ceil(idx.length / 64))
    const sub: number[] = []
    for (let q = 0; q < idx.length; q += stride) sub.push(idx[q])
    components = c.components(sub)
  }
  return { ...base, ground_truth, components }
}

export function mockLine(c: CaptureAccess, req: LineRequest): LineProfile {
  const norm = normOf(req.norm)
  const { width: W, height: H } = c.meta
  const [x0, y0] = req.p0, [x1, y1] = req.p1
  const len = Math.hypot(x1 - x0, y1 - y0)
  const n = Math.max(2, Math.ceil(len) + 1)
  const ux = (x1 - x0) / (len || 1), uy = (y1 - y0) / (len || 1)
  const half = Math.max(0, ((req.width_px ?? 1) - 1) / 2)
  const exs = (req.excitations_nm?.length ? req.excitations_nm : c.meta.excitations_nm).filter((e) => c.meta.excitations_nm.includes(e))
  const px = (xx: number, yy: number) => Math.min(W - 1, Math.max(0, Math.round(xx))) + Math.min(H - 1, Math.max(0, Math.round(yy))) * W
  const taps = Math.max(1, Math.round((req.width_px ?? 1)))
  const sampleIdx: number[][] = []
  const distance_px: number[] = []
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * len
    const cx = x0 + ux * t, cy = y0 + uy * t
    distance_px.push(t)
    const row: number[] = []
    for (let k = 0; k < taps; k++) { const o = taps > 1 ? -half + (2 * half * k) / (taps - 1) : 0; row.push(px(cx - uy * o, cy + ux * o)) }
    sampleIdx.push(row)
  }
  const value: Record<string, number[][]> = {}, emission: Record<string, number[]> = {}
  for (const ex of exs) {
    const key = String(ex), B = c.meta.emission_nm[key].length
    const cube = c.cube(ex, norm)
    emission[key] = c.meta.emission_nm[key]
    value[key] = sampleIdx.map((row) => { const o = new Array<number>(B).fill(0); for (const p of row) for (let b = 0; b < B; b++) o[b] += cube[p * B + b] / row.length; return o })
  }
  const ground_truth: Record<string, number[]> = {}
  for (const name of req.ground_truth ?? []) {
    const r = c.gt(name); if (!r) continue
    ground_truth[name] = sampleIdx.map((row) => { let s = 0, m = 0; for (const p of row) if (Number.isFinite(r[p])) { s += r[p]; m++ } return m ? s / m : NaN })
  }
  let distance_mm: number[] | null = null
  if (c.pxMm) {
    let acc = 0
    distance_mm = [0]
    for (let i = 1; i < n; i++) { const ds = distance_px[i] - distance_px[i - 1]; acc += ds * 0.5 * (c.pxMm[sampleIdx[i][0]] + c.pxMm[sampleIdx[i - 1][0]]); distance_mm.push(acc) }
    if (!distance_mm.every((v) => Number.isFinite(v)) || acc <= 0) distance_mm = null
  }
  return { distance_px, distance_mm, value, emission_nm: emission, ground_truth, unit: normUnit(norm.mode) }
}

/** Per-pixel scalar for a metric from a normalised cube set. */
function metricMap(c: CaptureAccess, m: MetricSpec, norm: NormSpec, N: number): Float32Array {
  const out = new Float32Array(N)
  const nearest = (em: number[], nm: number) => { let bi = 0; for (let i = 1; i < em.length; i++) if (Math.abs(em[i] - nm) < Math.abs(em[bi] - nm)) bi = i; return bi }
  const bandMap = (ex: number, nm: number) => {
    const key = String(ex), em = c.meta.emission_nm[key]
    if (!em) throw new Error(`capture has no excitation ${ex} nm`)
    const b = nearest(em, nm), B = em.length, cube = c.cube(ex, norm)
    const o = new Float32Array(N)
    for (let k = 0; k < N; k++) o[k] = cube[k * B + b]
    return o
  }
  switch (m.kind) {
    case 'band': return bandMap(m.ex, m.band_nm)
    case 'band_ratio': { const a = bandMap(m.a.ex, m.a.band_nm), b = bandMap(m.b.ex, m.b.band_nm); for (let k = 0; k < N; k++) out[k] = b[k] !== 0 ? a[k] / b[k] : NaN; return out }
    case 'band_sum': {
      const em = c.meta.emission_nm[String(m.ex)], B = em.length, cube = c.cube(m.ex, norm)
      for (let k = 0; k < N; k++) { let s = 0; for (let b = 0; b < B; b++) if (em[b] >= (m.from_nm ?? -Infinity) && em[b] <= (m.to_nm ?? Infinity)) s += cube[k * B + b]; out[k] = s }
      return out
    }
    case 'peak_wavelength': {
      const em = c.meta.emission_nm[String(m.ex)], B = em.length, cube = c.cube(m.ex, norm)
      for (let k = 0; k < N; k++) { let bi = 0; for (let b = 1; b < B; b++) if (cube[k * B + b] > cube[k * B + bi]) bi = b; out[k] = em[bi] }
      return out
    }
  }
}

export function mockResponse(c: CaptureAccess, req: ResponseRequest): ResponseCurve {
  const norm = normOf(req.norm)
  const { width: W, height: H } = c.meta
  const N = W * H
  const gt = c.gt(req.x_ground_truth)
  if (!gt) throw new Error(`unknown or non-numeric ground-truth layer "${req.x_ground_truth}"`)
  const y = metricMap(c, req.metric, norm, N)
  const sel: number[] = []
  if (req.roi) { const idx = maskIndices(req.roi, W, H); for (const k of idx) sel.push(k) } else for (let k = 0; k < N; k++) sel.push(k)
  const ok = sel.filter((k) => Number.isFinite(gt[k]) && Number.isFinite(y[k]))
  let lo = Infinity, hi = -Infinity
  for (const k of ok) { lo = Math.min(lo, gt[k]); hi = Math.max(hi, gt[k]) }
  const bins = Math.max(1, req.bins ?? 24)
  const cnt = new Array<number>(bins).fill(0), s1 = new Array<number>(bins).fill(0), s2 = new Array<number>(bins).fill(0)
  const span = hi - lo || 1
  for (const k of ok) { const b = Math.min(bins - 1, Math.floor(((gt[k] - lo) / span) * bins)); cnt[b]++; s1[b] += y[k]; s2[b] += y[k] * y[k] }
  const bin_centres: number[] = [], mean: number[] = [], std: number[] = [], count: number[] = []
  for (let b = 0; b < bins; b++) {
    if (!cnt[b]) continue
    const m = s1[b] / cnt[b]
    bin_centres.push(lo + ((b + 0.5) / bins) * span); mean.push(m); std.push(Math.sqrt(Math.max(0, s2[b] / cnt[b] - m * m))); count.push(cnt[b])
  }
  const stride = Math.max(1, Math.ceil(ok.length / 4000))
  const scatter_x: number[] = [], scatter_y: number[] = []
  for (let q = 0; q < ok.length; q += stride) { scatter_x.push(gt[ok[q]]); scatter_y.push(y[ok[q]]) }
  const unit = req.metric.kind === 'peak_wavelength' ? 'nm' : req.metric.kind === 'band_ratio' ? 'ratio' : normUnit(norm.mode)
  return { x_ground_truth: req.x_ground_truth, metric: req.metric, bin_centres, mean, std, count, scatter_x, scatter_y, unit }
}
