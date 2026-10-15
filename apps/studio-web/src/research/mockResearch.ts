/** In-browser stand-in for the research endpoints, used when the Python engine is unreachable.
 *  The designs and the Sobol' / Morris estimators are real (`designs.ts`); the "simulator" is a smooth synthetic response
 *  of the unit-cube coordinates, so every UI path (progress, results, sensitivity, inverse, methods) works offline.
 *  Nothing here is physics. */
import type { Experiment } from '../generated/experiment'
import type { Raster } from '../api/client'
import { getEngine } from '../api/client'
import type { JobEvent } from '../api/types'
import type {
  CalibratedFluorophore, DesignRequest, DesignResult, FisherRequest, FisherResult, FitMapResult, FitRequest, FitResult, MaterialOptics,
  MaterialOpticsRequest, MethodsRequest, MethodsText, Provenance, SensitivityResult, SobolMetric, MorrisMetric, StudyEvent, StudyRunRow,
  StudyRuns, StudySpec, StudyStart, StudyStatus, SurrogateRequest, SurrogateSummary, UnmixRequest, UnmixResult,
} from '../api/researchTypes'
import type { ResearchApi } from './api'
import { designSize, crlbSdFor, rng } from './logic'
import { morrisIndices, sobolBootstrap, sobolIndices, unitDesign, unitToValue, varianceOf } from './designs'
import { CALIBRATED_OFFLINE } from './calibratedData'
import { BUNDLE, fluorophoreMap } from '../state/presets'
import { materialMua, musPrime, bandCentres } from '../lib/spectral'
import { tissueMuaTerms } from '../lib/tissueSpectra'
import type { FluorophoreSpec, MaterialSpec } from '../generated/experiment'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
let seq = 0
const uid = (p: string) => `${p}${(++seq).toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}`
const hash01 = (s: string) => { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return ((h >>> 0) % 10007) / 10007 }

// ------------------------------------------------------------------------------------------------ studies
interface MockStudy {
  id: string; spec: StudySpec; keep: boolean
  state: StudyStatus['state']; fraction: number; message: string; error: string | null
  columns: string[]; rows: StudyRunRow[]; points: ReturnType<typeof unitDesign>
  listeners: Set<(e: StudyEvent) => void>; timer: ReturnType<typeof setTimeout> | null; started: number
}
const studies = new Map<string, MockStudy>()

function metricScale(kind: string): { base: number; span: number } {
  switch (kind) {
    case 'peak_wavelength': return { base: 470, span: 40 }
    case 'band_ratio': return { base: 0.6, span: 1.6 }
    case 'contrast': return { base: 0.05, span: 0.9 }
    case 'cnr': return { base: 1, span: 30 }
    case 'gt_stat': return { base: 0.2, span: 1.4 }
    default: return { base: 1800, span: 24000 }
  }
}
/** Smooth synthetic response f(u) in [0, 1]: parameter i has weight w_i (descending), alternating monotone / saturating /
 *  U-shaped transfer, plus one interaction between the two leading parameters. */
export function syntheticResponse(u: number[], seedKey: string): number {
  let f = 0, wsum = 0
  u.forEach((x, i) => {
    const w = (hash01(`${seedKey}:w${i}`) * 0.5 + 0.5) / (1 + 0.9 * i)
    const shape = Math.floor(hash01(`${seedKey}:s${i}`) * 3)
    const g = shape === 0 ? x : shape === 1 ? 1 - Math.exp(-3 * x) : 1 - 4 * (x - 0.5) ** 2
    f += w * g; wsum += w
  })
  if (u.length > 1) { const w = 0.35 / (1 + u.length * 0.1); f += w * u[0] * u[1]; wsum += w }
  return f / (wsum || 1)
}

function buildRows(spec: StudySpec, points: MockStudy['points']): { columns: string[]; rows: StudyRunRow[] } {
  const labels = spec.parameters.map((p) => p.label ?? p.pointer)
  const columns = ['run_id', 'point', 'replicate', 'seed', 'exp_hash', 'block', 'row', 'traj', 'step', 'step_param', 'direction',
    ...labels.map((l) => `p:${l}`), ...labels.map((l) => `u:${l}`), ...spec.metrics.map((m) => `m:${m.name}`), 'status', 'error', 't_capture_s', 't_metrics_s', 'warnings']
  const rows: StudyRunRow[] = []
  const reps = Math.max(1, spec.replicates)
  points.forEach((pt, i) => {
    for (let r = 0; r < reps; r++) {
      const rand = rng(i * 131 + r * 17 + 7)
      const row: StudyRunRow = {
        run_id: `${i.toString().padStart(5, '0')}_${r}`, point: i, replicate: r, seed: r, exp_hash: (hash01(`h${i}`) * 1e16).toString(16).slice(0, 12),
        block: pt.block ?? null, row: pt.row ?? null, traj: pt.traj ?? null, step: pt.step ?? null, step_param: pt.stepParam ?? null, direction: pt.direction ?? null,
        status: 'ok', error: null, t_capture_s: +(0.4 + rand() * 0.3).toFixed(3), t_metrics_s: +(0.01 + rand() * 0.02).toFixed(3), warnings: null,
      }
      spec.parameters.forEach((p, j) => { row[`p:${labels[j]}`] = unitToValue(p, pt.u[j]); row[`u:${labels[j]}`] = pt.u[j] })
      spec.metrics.forEach((m) => {
        const { base, span } = metricScale(m.kind)
        const f = syntheticResponse(pt.u, m.name)
        const noise = (rand() + rand() + rand() - 1.5) * 0.04
        row[`m:${m.name}`] = +(base + span * (f + noise)).toPrecision(6)
      })
      rows.push(row)
    }
  })
  return { columns, rows }
}

const statusOf = (s: MockStudy): StudyStatus => ({
  study_id: s.id, name: s.spec.name, state: s.state, fraction: s.fraction, message: s.message, n_runs: s.rows.length, error: s.error,
  ...(s.state === 'done' || s.state === 'cancelled' || s.state === 'error' ? { manifest: { spec: s.spec, spec_hash: (hash01(s.id) * 1e16).toString(16).slice(0, 16), n_runs: s.rows.length, n_completed: s.rows.filter((r) => r.status === 'ok').length, n_failed: 0, provenance: { ...mockProvenance, experiment_hash: (hash01(`${s.id}e`) * 1e32).toString(16).padEnd(32, 'a') + (hash01(`${s.id}f`) * 1e32).toString(16).padEnd(32, 'b') } } } : {}),
})
const emit = (s: MockStudy, e: StudyEvent) => s.listeners.forEach((l) => l(e))

// ------------------------------------------------------------------------------------------------ sensitivity
function mockSensitivity(s: MockStudy, boot: number): SensitivityResult {
  const spec = s.spec, k = spec.parameters.length
  const labels = spec.parameters.map((p) => p.label ?? p.pointer)
  const design = spec.design.kind
  if (design !== 'saltelli' && design !== 'morris') throw new Error('sensitivity analysis needs a saltelli (Sobol′) or morris design')
  const metrics: SensitivityResult['metrics'] = {}
  for (const m of spec.metrics) {
    const col = `m:${m.name}`
    const meanBy = (rows: StudyRunRow[]) => {
      const byPoint = new Map<number, number[]>()
      rows.forEach((r) => { const v = r[col]; if (typeof v === 'number') (byPoint.get(r.point) ?? byPoint.set(r.point, []).get(r.point)!).push(v) })
      return byPoint
    }
    const per = meanBy(s.rows)
    const f = s.points.map((_, i) => { const v = per.get(i) ?? []; return v.reduce((a, b) => a + b, 0) / (v.length || 1) })
    const withinVar = (() => {
      if (spec.replicates < 2) return null
      let ss = 0, dof = 0
      per.forEach((v) => { const m0 = v.reduce((a, b) => a + b, 0) / v.length; ss += v.reduce((a, b) => a + (b - m0) ** 2, 0); dof += v.length - 1 })
      return dof > 0 ? ss / dof : null
    })()
    const total = varianceOf(f)
    const noise = { replicates: spec.replicates, within_var: withinVar, noise_floor: withinVar != null && total > 0 ? Math.min(1, withinVar / spec.replicates / total) : null }
    if (design === 'saltelli') {
      const n = spec.design.n
      const fA = f.slice(0, n), fB = f.slice(n, 2 * n), fAB = Array.from({ length: k }, (_, i) => f.slice(2 * n + i * n, 2 * n + (i + 1) * n))
      const est = sobolIndices(fA, fB, fAB), ci = sobolBootstrap(fA, fB, fAB, Math.min(boot, 400))
      const indices: SobolMetric['indices'] = {}
      labels.forEach((l, i) => { indices[l] = { S1: est[i].S1, S1_ci: ci.S1[i], ST: est[i].ST, ST_ci: ci.ST[i] } })
      metrics[m.name] = { method: 'sobol', n_rows: f.length, variance: total, sum_S1: est.reduce((a, e) => a + e.S1, 0), noise, indices }
    } else {
      const p = Math.max(2, spec.design.levels), delta = p / (2 * (p - 1))
      const mi = morrisIndices(f, s.points, labels, delta)
      const indices: MorrisMetric['indices'] = {}
      labels.forEach((l, i) => { indices[l] = { mu: mi[i].mu, mu_star: mi[i].mu_star, mu_star_ci: [mi[i].mu_star - 1.96 * mi[i].sigma / Math.sqrt(Math.max(1, mi[i].n)), mi[i].mu_star + 1.96 * mi[i].sigma / Math.sqrt(Math.max(1, mi[i].n))], sigma: mi[i].sigma, n_effects: mi[i].n } })
      metrics[m.name] = { method: 'morris', n_rows: f.length, variance: total, noise, indices }
    }
  }
  return { design, parameters: labels, metrics }
}

// ------------------------------------------------------------------------------------------------ methods / provenance
const mockProvenance: Provenance = {
  package: 'spectral-select', package_version: '0.1.0 (mock)', engine_version: 'mock', git: { commit: null, branch: 'offline', dirty: null },
  python: 'n/a (browser mock)', numpy: 'n/a', scipy: 'n/a', numba: null, platform: 'browser', machine: 'n/a', cpu_count: navigator.hardwareConcurrency ?? 1,
  timestamp_utc: new Date().toISOString(),
}

function mockMethods(req: MethodsRequest): MethodsText {
  const { experiment: e, study, style } = req
  const L = style === 'latex'
  const b = (t: string) => (L ? `\\textbf{${t}}` : `**${t}**`)
  const num = (v: number) => String(+v.toPrecision(4))
  const src = e.sources.find((s) => s.id === e.active_source_id) ?? e.sources[0]
  const cam = e.cameras.find((c) => c.id === e.active_camera_id) ?? e.cameras[0]
  const exs = src.sequence.filter((s) => s.enabled).map((s) => s.wavelength_nm)
  const lines: string[] = []
  lines.push(`${b('Simulation.')} Multi-excitation fluorescence hyperspectral images of the scene “${e.name}” were simulated with SpectralForge Studio (offline mock engine; the text below is a template, not engine output). `
    + `The scene contains ${e.objects.length} object${e.objects.length === 1 ? '' : 's'} made of ${e.materials.length} material${e.materials.length === 1 ? '' : 's'}. `
    + `The source (${src.name}) was stepped through ${exs.length} excitation wavelength${exs.length === 1 ? '' : 's'} (${exs.join(', ')} nm); `
    + `the camera (${cam.name}) recorded ${cam.filter.n_bands} emission bands in ${num(cam.filter.band_step_nm)} nm steps with a ${num(cam.filter.fwhm_nm)} nm filter bandwidth. `
    + `Monte Carlo transport used ${e.acquisition.mc_photons.toLocaleString('en-US')} photons.`)
  if (study) {
    const d = study.design
    lines.push(`${b('Study design.')} ${study.parameters.length} parameter${study.parameters.length === 1 ? ' was' : 's were'} varied (${study.parameters.map((p) => `${p.label}${p.kind === 'choice' ? '' : ` ${num(p.low ?? 0)}–${num(p.high ?? 0)}`}`).join('; ')}) `
      + `with a ${d.kind} design of ${designSize(d, study.parameters, study.replicates).points} points and ${study.replicates} replicate${study.replicates === 1 ? '' : 's'} per point (${study.mode} mode, base seed ${d.seed}). `
      + `Metrics: ${study.metrics.map((m) => `${m.name} (${m.kind.replace('_', ' ')})`).join(', ')}.`
      + (d.kind === 'saltelli' ? ' First-order and total Sobol′ indices were estimated with the Saltelli and Jansen estimators with bootstrap confidence intervals.' : d.kind === 'morris' ? ' Parameters were screened with Morris elementary effects (μ*, σ).' : ''))
  }
  lines.push(`${b('Software.')} ${L ? 'SpectralForge \\cite{meloyan2026spectralforge}' : 'SpectralForge (Meloyan, WHISPERS 2026)'}, offline mock engine.`)
  return { style, text: lines.join(L ? '\n\n' : '\n\n') }
}

// ------------------------------------------------------------------------------------------------ material optics
function mockMaterialOptics(req: MaterialOpticsRequest): MaterialOptics {
  const mat = req.material as MaterialSpec
  const extra = (req.fluorophores ?? []) as FluorophoreSpec[]
  const fl = fluorophoreMap({ fluorophores: extra })
  const lo = req.nm_min ?? 350, hi = req.nm_max ?? 900, step = req.step ?? 5
  const nm: number[] = []
  for (let w = lo; w <= hi + 1e-9; w += step) nm.push(w)
  const components: Record<string, number[]> = {}
  mat.dyes.forEach((d) => {
    const single: MaterialSpec = { ...mat, dyes: [d], tissue: null, background_absorption: { ...mat.background_absorption, preset: 'none' } } as MaterialSpec
    components[`dye:${d.fluorophore}`] = nm.map((w) => materialMua(single, fl, w))
  })
  if (mat.background_absorption.preset !== 'none') {
    const only: MaterialSpec = { ...mat, dyes: [], tissue: null } as MaterialSpec
    components.absorber = nm.map((w) => materialMua(only, fl, w))
  }
  if (mat.tissue) {
    const terms = nm.map((w) => tissueMuaTerms(mat.tissue!, w))
    ;(['hbo2', 'hb', 'water', 'lipid', 'melanosome', 'bilirubin', 'carotene'] as const).forEach((k) => {
      if (terms.some((t) => t[k] > 0)) components[k === 'carotene' ? 'beta_carotene' : k] = terms.map((t) => t[k])
    })
  }
  return { nm, mua_per_mm: nm.map((w) => materialMua(mat, fl, w)), mus_prime_per_mm: nm.map((w) => musPrime(mat.scattering, w)), g: mat.scattering.g, n: mat.refractive_index, components }
}

// ------------------------------------------------------------------------------------------------ inverse
interface Job { listeners: Set<(e: JobEvent) => void>; timer: ReturnType<typeof setTimeout> | null; done: boolean }
const jobs = new Map<string, Job>()
const results = new Map<string, unknown>()
function runJob<T>(label: string, steps: number, make: () => T, ms = 260): string {
  const id = uid('mj')
  const job: Job = { listeners: new Set(), timer: null, done: false }
  jobs.set(id, job)
  let i = 0
  const tick = () => {
    if (i < steps) { const f = i / steps; job.listeners.forEach((l) => l({ type: 'progress', fraction: f, message: `${label} ${i + 1}/${steps}` })); i++; job.timer = setTimeout(tick, ms) }
    else {
      try { const rid = uid('res'); results.set(rid, make()); job.done = true; job.listeners.forEach((l) => l({ type: 'done', capture_id: rid })) }
      catch (e) { job.listeners.forEach((l) => l({ type: 'error', message: String(e) })) }
    }
  }
  job.timer = setTimeout(tick, 60)
  return id
}

function channelsOf(e: Experiment, exOverride?: number[]): { ex_nm: number[]; em_nm: number[] } {
  const src = e.sources.find((s) => s.id === e.active_source_id) ?? e.sources[0]
  const cam = e.cameras.find((c) => c.id === e.active_camera_id) ?? e.cameras[0]
  const exs = (exOverride?.length ? exOverride : src.sequence.filter((s) => s.enabled).map((s) => s.wavelength_nm))
  const ex_nm: number[] = [], em_nm: number[] = []
  for (const ex of exs) for (const em of bandCentres(cam.filter.band_start_nm, cam.filter.band_step_nm, cam.filter.n_bands, cam.filter.band_start_policy, cam.filter.start_offset_nm, ex)) { ex_nm.push(ex); em_nm.push(em) }
  return { ex_nm, em_nm }
}

function mockFisher(req: FisherRequest): FisherResult {
  const e = req.experiment ?? (BUNDLE.experiments[Object.keys(BUNDLE.experiments)[0]] as Experiment)
  const ch = channelsOf(e, req.excitations)
  const params = req.parameters ?? [{ name: 'theta0' }]
  const names = params.map((p, i) => p.name ?? (p.path ?? p.paths?.[0] ?? `p${i}`).split('/').slice(-2).join('.'))
  const P = names.length, C = ch.ex_nm.length
  const theta = params.map((p) => (p.lo != null && p.hi != null ? (p.lo + p.hi) / 2 : 1))
  const rand = rng(11)
  // each parameter couples to a band of (ex, em): a Gaussian ridge in the excitation-emission plane, parameter-specific position
  const jac: number[][] = [], mean: number[] = [], varE: number[] = []
  for (let c = 0; c < C; c++) {
    const em = ch.em_nm[c], ex = ch.ex_nm[c]
    const base = 4e3 * Math.exp(-0.5 * ((em - (ex + 120)) / 70) ** 2) * (0.4 + 0.6 * Math.exp(-0.5 * ((ex - 365) / 60) ** 2))
    mean.push(base)
    varE.push(base + 25 ** 2)
    jac.push(Array.from({ length: P }, (_, p) => {
      const c0 = ex + 60 + 55 * p, w = 40 + 25 * p
      return base * Math.exp(-0.5 * ((em - c0) / w) ** 2) * (p % 2 ? -1 : 1) * (0.7 + 0.6 * rand()) / Math.max(theta[p], 1e-3) * 0.4
    }))
  }
  const F = Array.from({ length: P }, () => new Array<number>(P).fill(0))
  jac.forEach((J, c) => { for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) F[i][j] += (J[i] * J[j]) / varE[c] })
  const sd = crlbSdFor(jac, varE, jac.map((_, i) => i))
  return {
    fisher_id: uid('fis'), names, theta, fisher: F, crlb_covariance: F.map((_, i) => F.map((__, j) => (i === j ? sd[i] ** 2 : 0))),
    crlb_sd: sd, crlb_rel_sd: sd.map((s, i) => s / Math.max(Math.abs(theta[i]), 1e-9)), identifiable: sd.every(Number.isFinite), n_channels: C,
    channels: ch, mean_e: mean, jac_e: jac, var_e: varE, info_per_channel: jac.map((J, c) => J.map((v) => (v * v) / varE[c])), meta: { mock: true },
  }
}

/** Greedy D/A/target channel selection (the engine adds an exchange refinement and an exhaustive check). */
function mockDesign(fr: FisherResult, req: DesignRequest): DesignResult {
  const k = Math.min(req.k, fr.n_channels), P = fr.names.length
  const crit = req.criterion ?? 'D'
  const tIdx = crit.startsWith('target:') ? Math.max(0, fr.names.indexOf(crit.slice(7))) : -1
  const score = (idx: number[]) => {
    const sd = crlbSdFor(fr.jac_e, fr.var_e, idx)
    if (!sd.every(Number.isFinite)) return -Infinity
    if (tIdx >= 0) return -sd[tIdx]
    if (crit === 'A') return -sd.reduce((a, s) => a + s * s, 0)
    return -sd.reduce((a, s) => a + Math.log(s), 0)
  }
  const sel: number[] = [], objective: number[] = [], sds: (number | null)[][] = []
  for (let n = 0; n < k; n++) {
    let best = -1, bs = -Infinity
    for (let c = 0; c < fr.n_channels; c++) {
      if (sel.includes(c)) continue
      const s = score([...sel, c]) + (n === 0 && P > 1 && !Number.isFinite(score([c])) ? 0 : 0)
      const t = Number.isFinite(s) ? s : -1e300 + c
      if (t > bs) { bs = t; best = c }
    }
    sel.push(best); objective.push(bs)
    sds.push(crlbSdFor(fr.jac_e, fr.var_e, sel).map((v) => (Number.isFinite(v) ? v : null)))
  }
  const ch = { ex_nm: sel.map((i) => fr.channels.ex_nm[i]), em_nm: sel.map((i) => fr.channels.em_nm[i]) }
  let exposure: DesignResult['exposure'] = null
  if (req.budget === 'exposure_time') {
    const T = req.total_time_s ?? k * 0.5
    const w = sel.map((c) => Math.sqrt(fr.info_per_channel[c].reduce((a, b) => a + b, 0)))
    const ws = w.reduce((a, b) => a + b, 0) || 1
    const equal = sel.map(() => T / k), opt = w.map((x) => (T * x) / ws)
    exposure = {
      total_time_s: T, channels: ch, t_opt_s: opt, t_reference_s: sel.map(() => 0.5), t_equal_s: equal, t_max_s: sel.map(() => T),
      crlb_sd_opt: crlbSdFor(fr.jac_e.map((J, c) => { const j = sel.indexOf(c); return j < 0 ? J : J.map((v) => v * Math.sqrt(opt[j] / 0.5)) }), fr.var_e, sel),
      crlb_sd_reference: sds[k - 1].map((v) => v ?? NaN), crlb_sd_equal: crlbSdFor(fr.jac_e.map((J, c) => { const j = sel.indexOf(c); return j < 0 ? J : J.map((v) => v * Math.sqrt(equal[j] / 0.5)) }), fr.var_e, sel), names: fr.names,
    }
  }
  return {
    criterion: crit, k, selected: sel, channels: ch, labels: sel.map((i) => `${fr.channels.ex_nm[i]}->${fr.channels.em_nm[i]}`), objective, names: fr.names,
    crlb_sd_vs_k: sds, fisher: fr.fisher, exhaustive: null, exposure, all_channels_sd: fr.crlb_sd,
  }
}

const unmixStore = new Map<string, { names: string[]; w: number; h: number; maps: Record<string, Float32Array> }>()

async function mockUnmix(req: UnmixRequest): Promise<UnmixResult> {
  const eng = getEngine()
  if (!eng) throw new Error('engine unavailable')
  const meta = await eng.getCapture(req.capture_id)
  const ex = meta.excitations_nm[0]
  const band = await eng.band(req.capture_id, ex, Math.floor((meta.emission_nm[String(ex)]?.length ?? 2) / 2))
  const K = Math.max(2, Math.min(4, req.endmembers?.n_components ?? 3))
  const names = req.endmembers?.source === 'rois' ? Object.keys(req.endmembers.rois ?? {}) : ['collagen', 'NADH', 'FAD', 'background'].slice(0, K)
  if (!names.length) throw new Error('endmembers.rois must be an object {name: roi}')
  const { width: w, height: h, data } = band
  let mx = 1e-9
  for (let i = 0; i < data.length; i++) mx = Math.max(mx, data[i])
  const raw = names.map((_, k) => { const m = new Float32Array(w * h); for (let i = 0; i < m.length; i++) { const x = (i % w) / w, y = Math.floor(i / w) / h; m[i] = Math.max(0, (data[i] / mx) * (0.4 + 0.6 * Math.sin((k + 1) * 2.2 * x + k) ** 2) + 0.15 * Math.cos((k + 1) * 3 * y) ** 2) } return m })
  const maps: Record<string, Float32Array> = {}
  const tot = new Float32Array(w * h)
  raw.forEach((m) => m.forEach((v, i) => { tot[i] += v }))
  names.forEach((n, k) => { const m = new Float32Array(w * h); for (let i = 0; i < m.length; i++) m[i] = tot[i] > 0 ? raw[k][i] / tot[i] : 0; maps[n] = m })
  const rm = new Float32Array(w * h); const rand = rng(5)
  for (let i = 0; i < rm.length; i++) rm[i] = 0.02 + 0.03 * rand() + 0.04 * (data[i] / mx)
  maps.rmse = rm; maps.residual_fraction = rm
  const id = uid('unm')
  unmixStore.set(id, { names, w, h, maps })
  const em = meta.emission_nm[String(ex)] ?? []
  return {
    names, method: req.method ?? 'nnls', shape: [h, w], mean_residual_fraction: 0.046, meta: { mock: true }, unmix_id: id,
    mean_abundance: Object.fromEntries(names.map((n) => [n, maps[n].reduce((a, b) => a + b, 0) / (w * h)])),
    endmembers: {
      names, source: req.endmembers?.source ?? 'forward_model', signal: req.signal ?? 'adu_net', meta: {},
      channels: { ex_nm: em.map(() => ex), em_nm: em },
      matrix: names.map((_, k) => em.map((e) => Math.exp(-0.5 * ((e - (460 + 40 * k)) / (35 + 8 * k)) ** 2))),
    },
    ...(req.compare_gt ? { gt_comparison: Object.fromEntries(names.map((n, k) => [n, { pearson_r: 0.97 - 0.03 * k, slope: 0.98 - 0.02 * k, rmse: 0.04 + 0.01 * k, rel_rmse: 0.09 + 0.02 * k }])) } : {}),
  }
}

interface MockSurrogate { names: string[]; bounds: number[][] }
const surrogates = new Map<string, MockSurrogate>()

function mockFit(req: FitRequest): FitResult {
  const s = surrogates.get(req.surrogate_id)
  const names = s?.names ?? ['theta0']
  const b = s?.bounds ?? names.map(() => [0, 1])
  const theta = b.map(([lo, hi], i) => lo + (hi - lo) * (0.4 + 0.15 * i))
  const se = b.map(([lo, hi]) => (hi - lo) * 0.012)
  const ci = Object.fromEntries(names.map((n, i) => [n, { lo: theta[i] - 1.96 * se[i], hi: theta[i] + 1.96 * se[i], level: req.level ?? 0.95 }]))
  const profile = req.profile ? Object.fromEntries(names.map((n, i) => {
    const values = Array.from({ length: 31 }, (_, j) => theta[i] + ((j - 15) / 15) * 4 * se[i])
    return [n, { values, delta_chi2: values.map((v) => ((v - theta[i]) / se[i]) ** 2) }]
  })) : null
  return { names, theta, se, cov: names.map((_, i) => names.map((__, j) => (i === j ? se[i] ** 2 : 0))), chi2: 41.2, dof: 40, reduced_chi2: 1.03, success: true, at_bound: names.map(() => false), ci, profile, n_roi_pixels: 1024 }
}

function mockFitMap(req: FitRequest): FitMapResult {
  const s = surrogates.get(req.surrogate_id)
  const names = s?.names ?? ['theta0']
  const [lo, hi] = s?.bounds?.[0] ?? [0, 1]
  const R = 24, Cc = 32, rand = rng(9)
  const gt: (number | null)[][] = [], th: (number | null)[][] = [], c2: (number | null)[][] = []
  for (let r = 0; r < R; r++) { const g: (number | null)[] = [], t: (number | null)[] = [], q: (number | null)[] = []; for (let c = 0; c < Cc; c++) { const v = lo + (hi - lo) * (0.1 + 0.8 * (c / (Cc - 1))); g.push(v); t.push(v + (hi - lo) * 0.03 * (rand() + rand() - 1)); q.push(0.9 + 0.3 * rand()) } gt.push(g); th.push(t); c2.push(q) }
  return { names, tile: req.map?.tile ?? 8, shape: [R, Cc], stats: { bias: 0.002 * (hi - lo), rmse: 0.03 * (hi - lo), pearson_r: 0.992, coverage_1p96se: 0.94, n_tiles: R * Cc }, theta: th, se: null, chi2_red: c2, gt }
}

// ------------------------------------------------------------------------------------------------ the API
export const mockResearch: ResearchApi = {
  kind: 'mock',
  async startStudy(spec, keep): Promise<StudyStart> {
    const base = typeof spec.base_experiment === 'string' ? spec.base_experiment : spec.base_experiment.name
    void base
    const d = spec.design
    const points = unitDesign(d.kind, spec.parameters, d.n, d.levels, d.seed)
    const { points: np, runs } = designSize(d, spec.parameters, spec.replicates)
    if (runs > spec.max_runs) throw new Error(`422 ${runs} runs exceeds max_runs ${spec.max_runs}`)
    if (!spec.parameters.length || !spec.metrics.length) throw new Error('422 a study needs parameters and metrics')
    const { columns, rows } = buildRows(spec, points)
    const s: MockStudy = { id: uid('st'), spec: structuredClone(spec), keep, state: 'running', fraction: 0, message: 'starting', error: null, columns, rows, points, listeners: new Set(), timer: null, started: Date.now() }
    studies.set(s.id, s)
    const per = Math.max(1, Math.ceil(rows.length / 40))
    let done = 0
    const tick = () => {
      if (s.state !== 'running') return
      done = Math.min(rows.length, done + per)
      s.fraction = done / rows.length; s.message = `run ${done}/${rows.length}`
      emit(s, { type: 'progress', fraction: s.fraction, message: s.message })
      if (done >= rows.length) { s.state = 'done'; s.fraction = 1; emit(s, { type: 'done', study_id: s.id, n_failed: 0 }) }
      else s.timer = setTimeout(tick, 70)
    }
    s.timer = setTimeout(tick, 120)
    void np
    return { study_id: s.id, n_runs: rows.length, n_points: np }
  },
  async listStudies() { return [...studies.values()].reverse().map(statusOf) },
  async studyStatus(id) { const s = studies.get(id); if (!s) throw new Error('404 unknown study'); return statusOf(s) },
  watchStudy(id, onEvent) {
    const s = studies.get(id)
    if (!s) { queueMicrotask(() => onEvent({ type: 'error', message: 'unknown study' })); return () => {} }
    s.listeners.add(onEvent)
    if (s.state === 'done') queueMicrotask(() => onEvent({ type: 'done', study_id: id, n_failed: 0 }))
    return () => s.listeners.delete(onEvent)
  },
  async cancelStudy(id) {
    const s = studies.get(id); if (!s) throw new Error('404 unknown study')
    if (s.state === 'running') { s.state = 'cancelled'; if (s.timer) clearTimeout(s.timer); emit(s, { type: 'cancelled' }) }
    return statusOf(s)
  },
  async studyRuns(id, offset, limit): Promise<StudyRuns> {
    const s = studies.get(id); if (!s) throw new Error('404 unknown study')
    const n = Math.floor(s.fraction * s.rows.length)
    if (n < 1) throw new Error('409 the study has not written results yet')
    return { total: n, columns: s.columns, rows: s.rows.slice(offset, Math.min(n, offset + limit)) }
  },
  async sensitivity(id, boot) {
    const s = studies.get(id); if (!s) throw new Error('404 unknown study')
    await sleep(150)
    return mockSensitivity(s, boot)
  },
  async methodsText(req) { await sleep(120); return mockMethods(req) },
  async provenance() { return { ...mockProvenance, timestamp_utc: new Date().toISOString() } },
  async materialOptics(req) { return mockMaterialOptics(req) },
  async calibratedFluorophores(): Promise<CalibratedFluorophore[]> { return CALIBRATED_OFFLINE },
  unmix: mockUnmix,
  async unmixMap(uid0, name): Promise<Raster> {
    const u = unmixStore.get(uid0); const m = u?.maps[name]
    if (!u || !m) throw new Error('404 unknown map')
    return { width: u.w, height: u.h, data: m }
  },
  async createSurrogate(req: SurrogateRequest) {
    const names = req.parameters.map((p, i) => p.name ?? (p.path ?? p.paths?.[0] ?? `p${i}`).split('/').slice(-2).join('.'))
    const bounds = req.parameters.map((p, i) => { const g = req.grid[names[i]]; return [p.lo ?? (g ? Math.min(...g) : 0), p.hi ?? (g ? Math.max(...g) : 1)] })
    const n = req.mode === 'sobol' ? req.n_points ?? 16 : names.reduce((a, nm) => a * (req.grid[nm]?.length ?? 1), 1)
    const job = runJob('simulating', Math.min(8, n), () => { const id = uid('sur'); surrogates.set(id, { names, bounds }); return id }, 220)
    // the "result" of a surrogate job is its id
    jobs.get(job)!.listeners.add((e) => { if (e.type === 'done') { const sid = results.get(e.capture_id) as string; results.set(sid, { surrogate_id: sid, names, bounds, n_points: n, n_channels: 80, mode: req.mode ?? 'grid', loo: { loo_rel_rms_mean: 0.004, loo_rel_rms_max: 0.011, n } }) } })
    return { job_id: job, n_simulator_runs: n }
  },
  async surrogate(id): Promise<SurrogateSummary> { return results.get(id) as SurrogateSummary },
  async fit(req) { await sleep(200); return mockFit(req) },
  async fitMap(req) { await sleep(400); return mockFitMap(req) },
  async fisher(req) {
    if (req.surrogate_id) { await sleep(80); const r = mockFisher({ ...req, parameters: surrogates.get(req.surrogate_id)?.names.map((n) => ({ name: n })) }); results.set(r.fisher_id, r); return r }
    const names = req.parameters?.length ?? 1
    const job = runJob('finite differences', Math.min(8, 2 * names + 1), () => { const r = mockFisher(req); results.set(r.fisher_id, r); return r.fisher_id }, 200)
    return { job_id: job }
  },
  async design(req) {
    await sleep(60)
    const fr = results.get(req.fisher_id ?? '') as FisherResult | undefined
    if (!fr) throw new Error('404 unknown fisher result')
    return mockDesign(fr, req)
  },
  async inverseResult<T>(id: string) {
    const r = results.get(id)
    if (typeof r === 'string') return results.get(r) as T       // fisher job: id → fisher_id → result
    return r as T
  },
  watchJob(jobId, onEvent) {
    const job = jobs.get(jobId)
    if (!job) { queueMicrotask(() => onEvent({ type: 'error', message: 'unknown job' })); return () => {} }
    job.listeners.add(onEvent)
    return () => job.listeners.delete(onEvent)
  },
}
