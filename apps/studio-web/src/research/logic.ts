/** Pure logic of the Research workspace: design sizes and runtime estimates, ROI conversion, run-table helpers
 *  (replicate aggregation, sorting, CSV) and the small linear algebra behind the optimal-design baselines. No React, no I/O. */
import type { RoiSpec } from '../api/types'
import type {
  DesignKind, DesignSpec, FisherResult, MetricSpec, ParameterSpec, StudyMode, StudyRunRow, StudySpec,
} from '../api/researchTypes'

// ---------------------------------------------------------------------------------------------- design size
export const DESIGN_INFO: Record<DesignKind, { label: string; blurb: string }> = {
  grid: { label: 'Grid', blurb: 'Full factorial: every combination of `levels` values per parameter. Exact but explodes with the number of parameters.' },
  random: { label: 'Random', blurb: 'Independent uniform draws. Simple, but clusters and leaves gaps.' },
  lhs: { label: 'Latin hypercube', blurb: 'Stratified: every parameter is covered evenly in N slices. The default for surrogate training sets.' },
  sobol: { label: 'Sobol sequence', blurb: 'Scrambled low-discrepancy points: fills space evenly and can be extended.' },
  saltelli: { label: 'Saltelli (Sobol′ indices)', blurb: 'Radial design A, B, ABᵢ with n(k+2) runs. Required for variance-based first-order and total Sobol′ indices.' },
  morris: { label: 'Morris screening', blurb: 'Elementary-effects trajectories, n(k+1) runs. Cheap ranking (μ*, σ) when the model is expensive.' },
}
export const DESIGN_ORDER: DesignKind[] = ['lhs', 'sobol', 'saltelli', 'morris', 'grid', 'random']

/** Number of parameter levels the grid design uses for one parameter. */
export function gridLevels(p: ParameterSpec, levels: number): number {
  if (p.kind === 'choice') return p.values?.length ?? 1
  if (p.kind === 'integer' && p.low != null && p.high != null) return Math.max(1, Math.min(levels, Math.floor(p.high) - Math.ceil(p.low) + 1))
  return levels
}

/** Design points (distinct experiments) and total runs (points × replicates) of a study. */
export function designSize(design: Pick<DesignSpec, 'kind' | 'n' | 'levels'>, params: ParameterSpec[], replicates: number): { points: number; runs: number } {
  const k = Math.max(0, params.length)
  let points = 0
  switch (design.kind) {
    case 'grid': points = k ? params.reduce((a, p) => a * gridLevels(p, design.levels), 1) : 0; break
    case 'saltelli': points = design.n * (k + 2); break
    case 'morris': points = design.n * (k + 1); break
    default: points = design.n
  }
  return { points, runs: points * Math.max(1, replicates) }
}

export interface TimingBasis {
  /** seconds for one preview capture (first excitation, preview scale) */
  previewS?: number | null
  /** seconds for one full capture, if known */
  fullS?: number | null
  /** mean (t_capture + t_metrics) of finished runs of each mode, measured on this engine */
  observed?: Partial<Record<StudyMode, number>>
  nExcitations: number
  previewScale: number
}
/** Estimated seconds per run for a mode: observed value if any, else scaled from the preview timing. */
export function secondsPerRun(mode: StudyMode, t: TimingBasis): { s: number; basis: 'observed' | 'preview-scaled' | 'default' } {
  const obs = t.observed?.[mode]
  if (obs != null && obs > 0) return { s: obs, basis: 'observed' }
  const prev = t.previewS != null && t.previewS > 0 ? t.previewS : null
  const base = prev ?? 1.5
  const ne = Math.max(1, t.nExcitations)
  let s: number
  if (mode === 'preview') s = base
  else if (mode === 'fast') s = base * ne
  else s = t.fullS != null && t.fullS > 0 ? t.fullS : base * ne / Math.max(0.05, t.previewScale) ** 2
  return { s, basis: prev ? 'preview-scaled' : 'default' }
}
export function estimateRuntime(runs: number, mode: StudyMode, t: TimingBasis): { seconds: number; basis: 'observed' | 'preview-scaled' | 'default' } {
  const { s, basis } = secondsPerRun(mode, t)
  return { seconds: runs * s, basis }
}
export function fmtDuration(sec: number): string {
  if (!Number.isFinite(sec)) return '—'
  if (sec < 1) return '<1 s'
  if (sec < 90) return `${Math.round(sec)} s`
  const m = sec / 60
  if (m < 90) return `${Math.round(m)} min`
  const h = m / 60
  return h < 48 ? `${h.toFixed(1)} h` : `${(h / 24).toFixed(1)} d`
}

// ---------------------------------------------------------------------------------------------- parameters / ROIs
/** The linked pointers of a parameter: the end thickness of a wedge layer follows its start thickness. */
export function suggestLinks(pointer: string, all: string[]): string[] {
  const m = /^(.*\/thickness)\/thickness_mm$/.exec(pointer)
  if (!m) return []
  const end = `${m[1]}/end_thickness_mm`
  return all.includes(end) ? [end] : []
}

/** Default sampling range of a parameter whose current value is `v` (`unit` as in the sweep picker). */
export function defaultRange(v: number, unit: string, label: string): { low: number; high: number; scale: 'linear' | 'log' } {
  if (unit === 'mm' && /thickness/.test(label)) return { low: 0.05, high: Math.max(1, +(v * 2).toPrecision(2)), scale: 'linear' }
  if (v > 0) return { low: +(v * 0.25).toPrecision(2), high: +(v * 2).toPrecision(2), scale: 'linear' }
  if (v < 0) return { low: +(v * 2).toPrecision(2), high: +(v * 0.25).toPrecision(2), scale: 'linear' }
  return { low: 0, high: 1, scale: 'linear' }
}

export function parameterValid(p: ParameterSpec): string | null {
  if (p.kind === 'choice') return (p.values?.length ?? 0) >= 2 ? null : 'a choice needs at least 2 values'
  if (p.low == null || p.high == null || !Number.isFinite(p.low) || !Number.isFinite(p.high)) return 'set low and high'
  if (!(p.high > p.low)) return 'high must exceed low'
  if (p.scale === 'log' && p.low <= 0) return 'log scale needs low > 0'
  return null
}

/** Convert a region ROI in capture pixels to the image fractions a study uses (engine convention: x / (width − 1)). */
export function roiToFraction(r: RoiSpec, w: number, h: number): RoiSpec {
  const fx = Math.max(w - 1, 1), fy = Math.max(h - 1, 1)
  const q = (v: number, f: number) => +(v / f).toFixed(5)
  switch (r.shape) {
    case 'point': return { shape: 'point', x: q(r.x, fx), y: q(r.y, fy) }
    case 'disk': return { shape: 'disk', x: q(r.x, fx), y: q(r.y, fy), radius_px: q(r.radius_px, fx) }
    case 'rect': return { shape: 'rect', x0: q(r.x0, fx), y0: q(r.y0, fy), x1: q(r.x1, fx), y1: q(r.y1, fy) }
    case 'ellipse': return { shape: 'ellipse', cx: q(r.cx, fx), cy: q(r.cy, fy), rx: q(r.rx, fx), ry: q(r.ry, fy) }
    case 'polygon': return { shape: 'polygon', points: r.points.map(([x, y]) => [q(x, fx), q(y, fy)] as [number, number]) }
  }
}
export const rectFrac = (x0: number, y0: number, x1: number, y1: number): RoiSpec => ({ shape: 'rect', x0, y0, x1, y1 })
export const ROI_PRESETS: { id: string; label: string; roi: RoiSpec }[] = [
  { id: 'whole', label: 'Whole image', roi: rectFrac(0, 0, 1, 1) },
  { id: 'centre', label: 'Centre 40 %', roi: rectFrac(0.3, 0.3, 0.7, 0.7) },
  { id: 'left', label: 'Left third', roi: rectFrac(0, 0, 0.33, 1) },
  { id: 'right', label: 'Right third', roi: rectFrac(0.67, 0, 1, 1) },
]
export function roiSummary(r: RoiSpec | null | undefined): string {
  if (!r) return 'whole image'
  const f = (v: number) => (+v.toFixed(2)).toString()
  switch (r.shape) {
    case 'rect': return `rect ${f(r.x0)}–${f(r.x1)} × ${f(r.y0)}–${f(r.y1)}`
    case 'disk': return `disk (${f(r.x)}, ${f(r.y)}) r ${f(r.radius_px)}`
    case 'ellipse': return `ellipse (${f(r.cx)}, ${f(r.cy)})`
    case 'point': return `point (${f(r.x)}, ${f(r.y)})`
    case 'polygon': return `polygon · ${r.points.length} vertices`
  }
}

export const METRIC_KIND_INFO: Record<MetricSpec['kind'], { label: string; hint: string; rois: 0 | 1 | 2 }> = {
  band_sum: { label: 'Band sum', hint: 'Statistic over the ROI of the per-pixel sum of emission bands in a range.', rois: 1 },
  roi_band: { label: 'Single band', hint: 'Statistic over the ROI of one emission band.', rois: 1 },
  band_ratio: { label: 'Band ratio', hint: 'Statistic of band A divided by statistic of band B.', rois: 1 },
  peak_wavelength: { label: 'Peak wavelength', hint: 'Position (nm) of the maximum of the ROI-mean spectrum at one excitation.', rois: 1 },
  eem_integral: { label: 'EEM integral', hint: 'Integral of the ROI-mean excitation–emission matrix.', rois: 1 },
  gt_stat: { label: 'Ground-truth statistic', hint: 'Statistic of a ground-truth map (e.g. layer thickness) in the ROI.', rois: 1 },
  contrast: { label: 'Contrast', hint: 'Michelson (A−B)/(A+B) or Weber (A−B)/B between two ROIs.', rois: 2 },
  cnr: { label: 'CNR', hint: 'Contrast-to-noise ratio |μA−μB| / √((σA²+σB²)/2) between two ROIs.', rois: 2 },
}
export const METRIC_ORDER: MetricSpec['kind'][] = ['band_sum', 'roi_band', 'band_ratio', 'peak_wavelength', 'contrast', 'cnr', 'eem_integral', 'gt_stat']

/** A reason a metric is not runnable, or null. */
export function metricProblem(m: MetricSpec): string | null {
  if (!m.name.trim()) return 'name the metric'
  switch (m.kind) {
    case 'roi_band': return m.band_nm == null ? `${m.name}: pick a band` : null
    case 'band_ratio': return m.a && m.b ? null : `${m.name}: pick bands A and B`
    case 'gt_stat': return m.gt ? null : `${m.name}: pick a ground-truth map`
    case 'contrast': case 'cnr': return m.roi && m.roi_b ? null : `${m.name}: needs two ROIs`
    default: return null
  }
}

/** Validation of a whole spec before it is sent: a list of human-readable problems. */
export function specProblems(s: Pick<StudySpec, 'parameters' | 'metrics' | 'design' | 'replicates' | 'max_runs'>): string[] {
  const out: string[] = []
  if (!s.parameters.length) out.push('Add at least one parameter.')
  s.parameters.forEach((p) => { const e = parameterValid(p); if (e) out.push(`${p.label ?? p.pointer}: ${e}`) })
  const labels = s.parameters.map((p) => p.label)
  if (new Set(labels).size !== labels.length) out.push('Parameter labels must be unique.')
  if (!s.metrics.length) out.push('Add at least one metric.')
  const names = s.metrics.map((m) => m.name)
  if (new Set(names).size !== names.length) out.push('Metric names must be unique.')
  s.metrics.forEach((m) => { const e = metricProblem(m); if (e) out.push(e) })
  const { runs } = designSize(s.design, s.parameters, s.replicates)
  if (runs > s.max_runs) out.push(`${runs.toLocaleString()} runs exceeds the limit of ${s.max_runs.toLocaleString()}.`)
  if (s.design.kind === 'saltelli' && s.design.n < 8) out.push('Saltelli needs a base sample size n of at least 8 for Sobol\u2019 indices.')
  if (s.design.kind === 'morris' && s.design.levels % 2 !== 0) out.push('Morris needs an even number of levels.')
  return out
}

// ---------------------------------------------------------------------------------------------- runs
export const pCols = (columns: string[]) => columns.filter((c) => c.startsWith('p:')).map((c) => c.slice(2))
export const mCols = (columns: string[]) => columns.filter((c) => c.startsWith('m:')).map((c) => c.slice(2))
export const numOf = (r: StudyRunRow, col: string): number => { const v = r[col]; return typeof v === 'number' && Number.isFinite(v) ? v : NaN }

export interface PointAgg { point: number; x: number; y: number; std: number; n: number; rows: StudyRunRow[] }
/** Group finished runs by design point; mean and sample std of metric `m` over the replicates, x = parameter `p`. */
export function aggregatePoints(rows: StudyRunRow[], p: string, m: string): PointAgg[] {
  const byPoint = new Map<number, StudyRunRow[]>()
  for (const r of rows) { if (r.status !== 'ok') continue; const a = byPoint.get(r.point) ?? []; a.push(r); byPoint.set(r.point, a) }
  const out: PointAgg[] = []
  for (const [point, rs] of byPoint) {
    const ys = rs.map((r) => numOf(r, `m:${m}`)).filter(Number.isFinite)
    const x = numOf(rs[0], `p:${p}`)
    if (!ys.length || !Number.isFinite(x)) continue
    const mean = ys.reduce((a, b) => a + b, 0) / ys.length
    const std = ys.length > 1 ? Math.sqrt(ys.reduce((a, b) => a + (b - mean) ** 2, 0) / (ys.length - 1)) : 0
    out.push({ point, x, y: mean, std, n: ys.length, rows: rs })
  }
  return out.sort((a, b) => a.x - b.x)
}

export type SortDir = 'asc' | 'desc'
export function sortRows(rows: StudyRunRow[], col: string, dir: SortDir): StudyRunRow[] {
  const k = dir === 'asc' ? 1 : -1
  return [...rows].sort((a, b) => {
    const x = a[col], y = b[col]
    const xn = typeof x === 'number' && Number.isFinite(x), yn = typeof y === 'number' && Number.isFinite(y)
    if (xn && yn) return ((x as number) - (y as number)) * k
    if (xn) return -1
    if (yn) return 1
    return String(x ?? '').localeCompare(String(y ?? '')) * k
  })
}

const csvCell = (v: unknown): string => {
  if (v == null) return ''
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : ''
  const s = String(v)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}
export function runsToCsv(columns: string[], rows: StudyRunRow[], header: string[] = []): string {
  const lines = header.map((h) => `# ${h}`)
  lines.push(columns.map(csvCell).join(','))
  for (const r of rows) lines.push(columns.map((c) => csvCell(r[c])).join(','))
  return lines.join('\n') + '\n'
}
/** Parse CSV text (RFC 4180 quoting; `#` comment lines skipped) into columns and rows; numeric-looking cells become numbers. */
export function parseCsv(text: string): { columns: string[]; rows: Record<string, string | number | null>[] } {
  const recs: string[][] = []
  let cur: string[] = [], cell = '', q = false
  const body = text.replace(/^\uFEFF/, '')
  const flushCell = () => { cur.push(cell); cell = '' }
  const flushRec = () => { flushCell(); if (!(cur.length === 1 && cur[0] === '')) recs.push(cur); cur = [] }
  for (let i = 0; i < body.length; i++) {
    const c = body[i]
    if (q) { if (c === '"') { if (body[i + 1] === '"') { cell += '"'; i++ } else q = false } else cell += c }
    else if (c === '"') q = true
    else if (c === ',') flushCell()
    else if (c === '\n') flushRec()
    else if (c !== '\r') cell += c
  }
  if (cell !== '' || cur.length) flushRec()
  const data = recs.filter((r) => !(r[0] ?? '').startsWith('#'))
  const columns = data[0] ?? []
  const rows = data.slice(1).map((r) => {
    const o: Record<string, string | number | null> = {}
    columns.forEach((c, i) => { const s = r[i] ?? ''; o[c] = s === '' ? null : /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(s) ? Number(s) : s })
    return o
  })
  return { columns, rows }
}

/** Parameter values of a run as `{pointer: value}` using the study's parameter specs (also_set pointers get the same value). */
export function runAssignments(spec: Pick<StudySpec, 'parameters'>, row: StudyRunRow): { pointer: string; value: number | string | boolean }[] {
  const out: { pointer: string; value: number | string | boolean }[] = []
  for (const p of spec.parameters) {
    const v = row[`p:${p.label}`]
    if (v == null || typeof v === 'undefined') continue
    for (const ptr of [p.pointer, ...p.also_set]) out.push({ pointer: ptr, value: v as number | string | boolean })
  }
  return out
}

// ---------------------------------------------------------------------------------------------- parallel coordinates
export interface Axis { key: string; label: string; lo: number; hi: number; log?: boolean }
export function axisOf(rows: StudyRunRow[], key: string, label: string, log = false): Axis {
  let lo = Infinity, hi = -Infinity
  for (const r of rows) { const v = numOf(r, key); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v) } }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1 }
  if (lo === hi) { lo -= 0.5; hi += 0.5 }
  return { key, label, lo, hi, log: log && lo > 0 }
}
export function axisT(a: Axis, v: number): number {
  if (!Number.isFinite(v)) return NaN
  if (a.log) return (Math.log(v) - Math.log(a.lo)) / (Math.log(a.hi) - Math.log(a.lo))
  return (v - a.lo) / (a.hi - a.lo)
}

// ---------------------------------------------------------------------------------------------- CRLB algebra (optimal-design baselines)
/** Inverse of a small symmetric positive-(semi)definite matrix by Gauss-Jordan with a relative ridge; null if singular. */
export function invertSym(F: number[][]): number[][] | null {
  const n = F.length
  const tr = F.reduce((a, r, i) => a + r[i], 0) / Math.max(1, n)
  if (!(tr > 0)) return null
  const ridge = 1e-12 * tr
  const A = F.map((r, i) => [...r.map((v, j) => v + (i === j ? ridge : 0)), ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))])
  for (let c = 0; c < n; c++) {
    let piv = c
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r
    if (Math.abs(A[piv][c]) < 1e-300) return null
    ;[A[c], A[piv]] = [A[piv], A[c]]
    const d = A[c][c]
    for (let j = 0; j < 2 * n; j++) A[c][j] /= d
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; if (f !== 0) for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j] }
  }
  return A.map((r) => r.slice(n))
}

/** CRLB standard deviation per parameter if only channels `idx` were acquired: sqrt(diag((Σ_c J_c J_cᵀ / var_c)⁻¹)). */
export function crlbSdFor(jac: number[][], varE: number[], idx: number[]): number[] {
  const P = jac[0]?.length ?? 0
  const F = Array.from({ length: P }, () => new Array<number>(P).fill(0))
  for (const c of idx) { const J = jac[c], w = 1 / varE[c]; for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) F[i][j] += J[i] * J[j] * w }
  const inv = invertSym(F)
  return inv ? inv.map((r, i) => Math.sqrt(Math.max(r[i], 0))) : new Array(P).fill(NaN)
}

/** Deterministic PRNG (mulberry32) so baseline curves are reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }
}

export interface DesignBaselines { ks: number[]; random: { mean: number[]; lo: number[]; hi: number[] }; uniform: number[] }
/**
 * CRLB sd of parameter `p` against the number of channels k, for (a) random subsets (median and 10-90 % band over `draws`
 * draws) and (b) a uniform subset: k channels evenly spaced through the channel list (excitation-major, as the engine orders
 * them). `jac` is the (C, P) Jacobian in electrons and `varE` the per-channel variance of the Fisher result.
 */
export function designBaselines(fr: Pick<FisherResult, 'jac_e' | 'var_e'>, p: number, kmax: number, draws = 120, seed = 1): DesignBaselines {
  const C = fr.jac_e.length
  const ks = Array.from({ length: Math.min(kmax, C) }, (_, i) => i + 1)
  const rand = rng(seed)
  const mean: number[] = [], lo: number[] = [], hi: number[] = [], uniform: number[] = []
  for (const k of ks) {
    const sds: number[] = []
    for (let d = 0; d < draws; d++) {
      const pool = Array.from({ length: C }, (_, i) => i)
      for (let i = 0; i < k; i++) { const j = i + Math.floor(rand() * (C - i)); [pool[i], pool[j]] = [pool[j], pool[i]] }
      const s = crlbSdFor(fr.jac_e, fr.var_e, pool.slice(0, k))[p]
      if (Number.isFinite(s)) sds.push(s)
    }
    sds.sort((a, b) => a - b)
    const q = (f: number) => (sds.length ? sds[Math.min(sds.length - 1, Math.floor(f * sds.length))] : NaN)
    mean.push(q(0.5)); lo.push(q(0.1)); hi.push(q(0.9))
    const idx = Array.from({ length: k }, (_, i) => Math.min(C - 1, Math.floor(((i + 0.5) * C) / k)))
    uniform.push(crlbSdFor(fr.jac_e, fr.var_e, [...new Set(idx)])[p])
  }
  return { ks, random: { mean, lo, hi }, uniform }
}

/** Heat-map grid (excitation rows × emission columns) of a per-channel value; missing cells are NaN. */
export function channelGrid(ch: { ex_nm: number[]; em_nm: number[] }, v: ArrayLike<number>): { z: number[][]; xs: number[]; ys: number[]; index: number[][] } {
  const ys = [...new Set(ch.ex_nm)].sort((a, b) => a - b)
  const xs = [...new Set(ch.em_nm.map((e) => +e.toFixed(3)))].sort((a, b) => a - b)
  const z = ys.map(() => new Array<number>(xs.length).fill(NaN))
  const index = ys.map(() => new Array<number>(xs.length).fill(-1))
  ch.ex_nm.forEach((ex, c) => { const i = ys.indexOf(ex), j = xs.indexOf(+ch.em_nm[c].toFixed(3)); z[i][j] = v[c]; index[i][j] = c })
  return { z, xs, ys, index }
}

/** Heuristic: Sobol' estimates from tiny Saltelli designs (n_rows < 16) or with ΣS1 far outside [0, 1] are dominated by estimator noise. */
export function sobolUnreliable(m: { sum_S1?: number | null; n_rows?: number | null }): boolean {
  const s = m.sum_S1
  return (typeof m.n_rows === 'number' && m.n_rows < 16) || (typeof s === 'number' && Number.isFinite(s) && (s < -0.2 || s > 1.3))
}

export type MdBlock = { kind: 'h'; text: string } | { kind: 'p'; text: string } | { kind: 'table'; head: string[]; rows: string[][] }
const cells = (l: string) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim())
/** Split the (small subset of) Markdown the Methods generator emits into headings, paragraphs and GFM pipe tables. */
export function parseMdBlocks(text: string): MdBlock[] {
  const out: MdBlock[] = []
  for (const chunk of text.split(/\n{2,}/)) {
    const lines = chunk.split('\n').filter((l) => l.trim())
    let para: string[] = []
    const flush = () => { if (para.length) { out.push({ kind: 'p', text: para.join(' ') }); para = [] } }
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i]
      if (/^#{1,4} /.test(l)) { flush(); out.push({ kind: 'h', text: l.replace(/^#{1,4} /, '') }); continue }
      if (l.trim().startsWith('|') && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1]) && lines[i + 1].includes('-')) {
        flush()
        const head = cells(l), rows: string[][] = []
        i += 2
        while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(cells(lines[i++]))
        i--
        out.push({ kind: 'table', head, rows })
        continue
      }
      para.push(l)
    }
    flush()
  }
  return out
}
