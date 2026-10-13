/** In-browser stand-in for the Python engine, used when `/api/health` is unreachable.
 *  Not physically rigorous: Gaussian dye spectra over analytically ray-cast material regions, but it speaks
 *  the exact `Engine` contract so every UI path (preview, capture progress, explorer, pixel explain) works offline. */
import type { Experiment, Inclusion, MaterialSpec } from '../generated/experiment'
import { chordAt } from '../lib/inclusions'
import type {
  BandStats, CaptureMeta, CaptureRequest, JobEvent, JobStatus, NormSpec, PhysicsWarning, PixelExplain, PreviewRequest, SweepEvent, SweepRequest, SweepStatus, ValidationResult,
} from './types'
import type { Cube, Engine, Raster } from './client'
import { blockGround, blockGtNames, blockInfos, blockPixelComponents, blocksOf, volumeData, volumeInfos } from './mockBlock'
import { mockEem, mockLine, mockResponse, mockRoi, type CaptureAccess } from './mockAnalysis'
import { stackAt } from '../analysis/blockGeometry'
import { setByPointer } from '../sweep/paths'
import { BUNDLE, fluorophoreMap } from '../state/presets'
import { bandCentres, emissionShape, EXTINCTION_REFERENCE, lineAveragedExcitation, FWHM_TO_SIGMA, muaDye } from '../lib/spectral'
import { renderMaps, type SceneMaps } from './mockScene'
import { activeCamera, activeSource } from '../state/store'

const GAIN = 5e4
const BLEACH_K = 1.5e4

type CompSpectra = Record<string, Float32Array>   // component → per band (pre-offset ADU at unit shade)
interface Step { ex: number; powerMW: number; exposureMs: number }
interface MockCapture {
  meta: CaptureMeta
  exp: Experiment
  maps: SceneMaps
  steps: Step[]
  /** [step][materialIndex] → components */
  spectra: CompSpectra[][]
  cubes: Map<string, Float32Array>
  /** layered-block spectra cached by (step, quantised layer-thickness vector) */
  blockSpectra: Map<string, Record<string, Float32Array>>
  normCubes: Map<string, Float32Array>
  gts: Map<string, Float32Array>
}
const store = new Map<string, MockCapture>()
const order: string[] = []
let seq = 0

/** Deterministic hash noise in [−1, 1). */
function hash(a: number, b: number, c: number): number {
  let h = (a * 374761393 + b * 668265263 + c * 2147483647) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return (((h ^ (h >>> 16)) >>> 0) / 4294967296) * 2 - 1
}
const gnoise = (a: number, b: number, c: number) => (hash(a, b, c) + hash(a, b + 7, c + 3) + hash(a + 5, b, c + 11)) * 0.577

function computeSpectra(exp: Experiment, matIds: string[], steps: Step[], bandsPerStep: number[][]): CompSpectra[][] {
  const fl = fluorophoreMap(exp)
  const src = activeSource(exp)
  const cam = activeCamera(exp)
  const mats = new Map(exp.materials.map((m) => [m.id, m] as const))
  const physics = exp.acquisition.physics
  const cumDose: Record<string, number> = {}
  return steps.map((st, si) => {
    const bands = bandsPerStep[si]
    const step = cam.filter.band_step_nm
    const sigmaBand = cam.filter.fwhm_nm * FWHM_TO_SIGMA
    const perMat: CompSpectra[] = matIds.map((id) => (mats.has(id) ? compsForMaterial(mats.get(id)!) : {}))
    function compsForMaterial(m: MaterialSpec): CompSpectra {
      const out: CompSpectra = {}
      const gain = (st.powerMW * st.exposureMs) / 100
      if (physics.fluorescence) {
        for (const d of m.dyes) {
          const f = fl[d.fluorophore]; if (!f) continue
          const eps = f.extinction * EXTINCTION_REFERENCE * lineAveragedExcitation(f, st.ex, src.line_fwhm_nm)
          const absP = physics.absorption ? 1 - Math.pow(10, -eps * d.concentration_uM * 1e-6 * 0.3) : 0.05
          const key = `${m.id}|${d.fluorophore}`
          const bf = physics.photobleaching ? Math.exp(-Math.min(6, f.bleach_quantum_yield * BLEACH_K * (cumDose[key] ?? 0))) : 1
          const quench = physics.quenching ? 1 / (1 + m.quenching_ksv_per_uM * d.concentration_uM) : 1
          const sig = f.em_fwhm_nm * FWHM_TO_SIGMA * (1 + f.em_skew)
          const arr = new Float32Array(bands.length)
          bands.forEach((wl, j) => {
            arr[j] = GAIN * gain * absP * f.quantum_yield * quench * bf * (emissionShape(f, wl) / (sig * Math.sqrt(2 * Math.PI))) * step
          })
          out[`dye:${d.fluorophore}`] = arr
          cumDose[key] = (cumDose[key] ?? 0) + (st.powerMW * st.exposureMs / 1000) * absP
        }
      }
      if (physics.elastic_leak) {
        const refl = 0.02 + 0.2 * (parseInt((m.display_color ?? '#808080').slice(1, 3), 16) / 255)
        const arr = new Float32Array(bands.length)
        bands.forEach((wl, j) => { arr[j] = GAIN * gain * refl * 2e-3 * Math.exp(-0.5 * ((wl - st.ex) / (sigmaBand * 1.2)) ** 2) })
        out['elastic'] = arr
      }
      if (physics.raman && m.aqueous) {
        const wr = 1e7 / (1e7 / st.ex - 3400)
        const arr = new Float32Array(bands.length)
        bands.forEach((wl, j) => { arr[j] = GAIN * gain * 8e-4 * m.raman_strength * Math.exp(-0.5 * ((wl - wr) / (6.4 + sigmaBand)) ** 2) })
        out['raman'] = arr
      }
      if (exp.background.room_light.enabled) {
        const arr = new Float32Array(bands.length)
        bands.forEach((wl, j) => { arr[j] = 25 * exp.background.room_light.illuminance_lux * Math.exp(-0.5 * ((wl - 545) / 80) ** 2) / 100 })
        out['room_light'] = arr
      }
      return out
    }
    void muaDye
    return perMat
  })
}

function stepsOf(exp: Experiment, only?: number | null): Step[] {
  const src = activeSource(exp)
  return src.sequence.filter((s) => s.enabled && (only == null || Math.abs(s.wavelength_nm - only) < 1e-6))
    .map((s) => ({ ex: s.wavelength_nm, powerMW: s.power_mW, exposureMs: s.exposure_ms }))
}

function bandsFor(exp: Experiment, ex: number): number[] {
  const f = activeCamera(exp).filter
  return bandCentres(f.band_start_nm, f.band_step_nm, f.n_bands, f.band_start_policy, f.start_offset_nm, ex)
}

/** Spectra of the layered-block pixel k (cached by layer-thickness vector). */
function blockComps(c: MockCapture, si: number, k: number): Record<string, Float32Array> | null {
  if (c.maps.face[k] === 0 || c.maps.obj[k] < 0) return null
  const ob = c.exp.objects[c.maps.obj[k]]
  if (ob.shape.kind !== 'layered_block') return null
  const lx = c.maps.lx[k], lz = c.maps.lz[k]
  const stk = stackAt(ob.shape, lx, lz)
  const incKey = (ob.shape.inclusions as Inclusion[]).map((inc) => { const ch = chordAt(inc, lx, lz); return ch ? `${Math.round(ch[0] * 100)}:${Math.round(ch[1] * 100)}` : '-' }).join(';')
  const key = `${si}|${c.maps.obj[k]}|${stk.t.map((t) => Math.round(t * 200)).join(',')}|${incKey}`
  let r = c.blockSpectra.get(key)
  if (!r) {
    const st = c.steps[si]
    const cam = activeCamera(c.exp), src = activeSource(c.exp)
    const f = cam.filter
    r = blockPixelComponents(c.exp, ob.shape, lx, lz, st, bandsFor(c.exp, st.ex), {
      gain: GAIN * (st.powerMW * st.exposureMs) / 100, bandStep: f.band_step_nm, sigmaBand: f.fwhm_nm * FWHM_TO_SIGMA, lineFwhm: src.line_fwhm_nm,
    })
    c.blockSpectra.set(key, r)
  }
  return r
}

function valueAt(c: MockCapture, si: number, k: number, j: number, mode: CaptureMeta['mode'], comps?: Record<string, number>, pre?: Record<string, Float32Array> | null): number {
  const sens = activeCamera(c.exp).sensor
  const mi = c.maps.mat[k]
  const st = c.steps[si]
  let sig = 0
  const blk = mode === 'sample' ? (pre !== undefined ? pre : blockComps(c, si, k)) : null
  if (mode === 'dark') sig = 0
  else if (mode === 'white_reference') sig = c.maps.irradiance[k] > 0 ? 38000 * Math.min(1, c.maps.irradiance[k] * 1.4) * (st.powerMW * st.exposureMs) / 400 : 0
  else if (blk) {
    const sh = c.maps.irradiance[k]
    for (const [name, arr] of Object.entries(blk)) { const v = arr[j] * sh; sig += v; if (comps) comps[name] = v }
    if (c.exp.background.room_light.enabled) {
      const wl = bandsFor(c.exp, st.ex)[j]
      const v = 25 * c.exp.background.room_light.illuminance_lux * Math.exp(-0.5 * ((wl - 545) / 80) ** 2) / 100
      sig += v; if (comps) comps.room_light = v
    }
  } else if (mi >= 0) {
    const sh = c.maps.irradiance[k]
    for (const [name, arr] of Object.entries(c.spectra[si][mi])) {
      const v = arr[j] * (name === 'room_light' ? 1 : sh)
      sig += v
      if (comps) comps[name] = v
    }
  }
  const dark = 0.5 * st.exposureMs / 1000
  const shot = c.exp.acquisition.physics.shot_noise ? Math.sqrt(Math.max(sig, 0) * sens.gain_e_per_adu) / sens.gain_e_per_adu * gnoise(k, j, si) : 0
  const read = c.exp.acquisition.physics.read_noise ? (sens.read_noise_e / sens.gain_e_per_adu) * gnoise(k + 1, j, si + 100) : 0
  const v = sens.offset_adu + sig + dark + shot + read
  return Math.min(65535, Math.max(0, v))
}

function cubeFor(c: MockCapture, si: number, mode: CaptureMeta['mode']): Float32Array {
  const key = `${si}`
  let cube = c.cubes.get(key)
  if (cube) return cube
  const nb = bandsFor(c.exp, c.steps[si].ex).length
  const N = c.maps.width * c.maps.height
  cube = new Float32Array(N * nb)
  for (let k = 0; k < N; k++) {
    const pre = mode === 'sample' ? blockComps(c, si, k) : null
    for (let j = 0; j < nb; j++) cube[k * nb + j] = valueAt(c, si, k, j, mode, undefined, pre)
  }
  c.cubes.set(key, cube)
  return cube
}

function pct(sorted: Float32Array, p: number) { return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] }

function build(exp: Experiment, mode: CaptureMeta['mode'], opts: { scale: number; only?: number | null }): MockCapture {
  const t0 = performance.now()
  const sensor = activeCamera(exp).sensor
  const W = Math.max(16, Math.round(sensor.width_px * opts.scale)), H = Math.max(16, Math.round(sensor.height_px * opts.scale))
  const steps = stepsOf(exp, opts.only)
  if (!steps.length) throw new Error('no enabled excitation step in the active source sequence')
  const maps = renderMaps(exp, W, H)
  const bandsPerStep = steps.map((s) => bandsFor(exp, s.ex))
  const spectra = computeSpectra(exp, maps.materials, steps, bandsPerStep)
  const id = `${mode === 'preview' ? 'pv' : 'cap'}_${(++seq).toString().padStart(3, '0')}`
  const cap: MockCapture = { exp, maps, steps, spectra, cubes: new Map(), blockSpectra: new Map(), normCubes: new Map(), gts: new Map(), meta: null as unknown as CaptureMeta }
  const stats: Record<string, BandStats[]> = {}, sat: Record<string, number> = {}, emission: Record<string, number[]> = {}
  const exposure: Record<string, number> = {}, power: Record<string, number> = {}
  const N = W * H
  steps.forEach((st, si) => {
    const key = String(st.ex)
    const bands = bandsPerStep[si]
    emission[key] = bands; exposure[key] = st.exposureMs; power[key] = st.powerMW
    const cube = cubeFor(cap, si, mode === 'preview' ? 'sample' : mode)
    stats[key] = bands.map((_, j) => {
      const sample: number[] = []
      for (let k = 0; k < N; k += 29) sample.push(cube[k * bands.length + j])
      const s = Float32Array.from(sample).sort()
      return { min: s[0], max: s[s.length - 1], p01: pct(s, 0.01), p99: pct(s, 0.99), mean: s.reduce((a, b) => a + b, 0) / s.length }
    })
    let nsat = 0, cnt = 0
    for (let k = 0; k < N; k += 29) for (let j = 0; j < bands.length; j++) { cnt++; if (cube[k * bands.length + j] >= 65535) nsat++ }
    sat[key] = nsat / Math.max(1, cnt)
  })
  const warnings: string[] = []
  Object.entries(sat).forEach(([k, v]) => { if (v > 0.001) warnings.push(`${(v * 100).toFixed(2)}% of samples saturated at ${k} nm — lower power or exposure`) })
  warnings.push('Mock engine: synthetic spectra, not a physical simulation')
  cap.meta = {
    id, created: new Date().toISOString(), name: exp.name, mode, width: W, height: H,
    excitations_nm: steps.map((s) => s.ex), emission_nm: emission, exposure_ms: exposure, power_mW: power, stats,
    saturated_fraction: sat, ground_truth_layers: ['material_index', 'object_index', 'depth_mm', 'incidence_deg', 'irradiance', ...blockGtNames(exp)],
    warnings, timings_s: { total: (performance.now() - t0) / 1000 }, camera_id: activeCamera(exp).id, source_id: activeSource(exp).id,
  }
  if (blocksOf(exp).length) Object.assign(cap.meta, { blocks: blockInfos(exp), volumes: volumeInfos(exp, steps.map((s) => s.ex)) })
  store.set(id, cap); order.unshift(id)
  return cap
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const need = (id: string) => { const c = store.get(id); if (!c) throw new Error(`unknown capture ${id}`); return c }
const modeOf = (c: MockCapture): CaptureMeta['mode'] => c.meta.mode === 'preview' ? 'sample' : c.meta.mode

// ---- jobs ---------------------------------------------------------------------------------------
interface Job { status: JobStatus; listeners: Set<(e: JobEvent) => void>; timer: ReturnType<typeof setTimeout> | null; cancelled: boolean }
const jobs = new Map<string, Job>()
let jobSeq = 0

function emit(job: Job, e: JobEvent) { job.listeners.forEach((l) => l(e)) }

function physicsWarnings(exp: Experiment): PhysicsWarning[] {
  const w: PhysicsWarning[] = []
  const src = activeSource(exp), cam = activeCamera(exp)
  const fl = fluorophoreMap(exp)
  if (!src.sequence.some((s) => s.enabled)) w.push({ code: 'empty_sequence', severity: 'warn', message: 'No enabled excitation step: nothing to acquire.' })
  for (const s of src.sequence) {
    if (s.enabled && s.wavelength_nm < 380) w.push({ code: 'uv_glass', severity: 'info', message: `${s.wavelength_nm} nm is UV: borosilicate dish/lens transmission falls off — check the vessel material.`, target_id: src.id })
  }
  if (cam.filter.band_start_policy === 'excitation_offset')
    w.push({ code: 'band_policy', severity: 'info', message: `Emission bands start at max(${cam.filter.band_start_nm}, excitation + ${cam.filter.start_offset_nm}) nm (Nuance protocol), so band lists differ per excitation.`, target_id: cam.id })
  if (exp.sources.length > 1 || exp.cameras.length > 1) w.push({ code: 'multi_hw', severity: 'info', message: 'Only the active source and camera are simulated.' })
  for (const m of exp.materials) {
    if (!exp.objects.some((o) => o.material_id === m.id || o.vessel_material_id === m.id) && exp.background.material_id !== m.id) continue
    let worst = 0, worstNm = 0
    for (const d of m.dyes) {
      const f = fl[d.fluorophore]; if (!f) continue
      for (const s of src.sequence) {
        if (!s.enabled) continue
        const A = f.extinction * EXTINCTION_REFERENCE * Math.exp(-0.5 * ((s.wavelength_nm - f.ex_peak_nm) / (f.ex_fwhm_nm * FWHM_TO_SIGMA)) ** 2) * d.concentration_uM * 1e-6 * 0.3
        if (A > worst) { worst = A; worstNm = s.wavelength_nm }
      }
    }
    if (worst > 0.1 && exp.acquisition.physics.absorption)
      w.push({ code: 'inner_filter', severity: 'warn', message: `${m.name}: absorbance up to ${worst.toFixed(2)} (at ${worstNm} nm) — inner-filter effect will distort spectra.`, target_id: m.id })
  }
  return w
}

export const mockEngine: Engine = {
  kind: 'mock',
  async health() { return { ok: true, engine_version: 'mock', schema_version: BUNDLE.schema_version } },
  async presets() { return BUNDLE },
  async validate(e): Promise<ValidationResult> {
    const errors: ValidationResult['errors'] = []
    const matIds = new Set(e.materials.map((m) => m.id))
    e.objects.forEach((o, i) => {
      if (!matIds.has(o.material_id)) errors.push({ loc: ['objects', i, 'material_id'], msg: `unknown material "${o.material_id}"` })
      if (o.vessel_material_id && !matIds.has(o.vessel_material_id)) errors.push({ loc: ['objects', i, 'vessel_material_id'], msg: `unknown material "${o.vessel_material_id}"` })
    })
    e.sources.forEach((s, i) => s.sequence.forEach((st, k) => {
      if (!(st.wavelength_nm > 200 && st.wavelength_nm < 1100)) errors.push({ loc: ['sources', i, 'sequence', k, 'wavelength_nm'], msg: 'must be between 200 and 1100 nm' })
      if (!(st.exposure_ms > 0)) errors.push({ loc: ['sources', i, 'sequence', k, 'exposure_ms'], msg: 'must be > 0' })
    }))
    return { ok: errors.length === 0, errors, warnings: physicsWarnings(e) }
  },
  async preview(r: PreviewRequest) {
    await sleep(120)
    const exp = r.experiment
    const only = r.excitation_nm ?? stepsOf(exp)[0]?.ex ?? null
    return build(exp, 'preview', { scale: r.scale ?? exp.acquisition.preview_scale, only }).meta
  },
  async startCapture(r: CaptureRequest) {
    const id = `job_${++jobSeq}`
    const job: Job = { status: { job_id: id, state: 'running', fraction: 0, message: 'starting', capture_id: null, error: null }, listeners: new Set(), timer: null, cancelled: false }
    jobs.set(id, job)
    const exp = r.experiment, mode = r.mode ?? 'sample'
    const steps = stepsOf(exp)
    if (!steps.length) throw new Error('no enabled excitation step in the active source sequence')
    const per = 520
    let i = 0
    const tick = () => {
      if (job.cancelled) return
      if (i < steps.length) {
        const f = i / steps.length
        job.status = { ...job.status, fraction: f, message: `Exposing ${steps[i].ex} nm (${i + 1}/${steps.length})` }
        emit(job, { type: 'progress', fraction: f, message: job.status.message })
        i++
        job.timer = setTimeout(tick, per)
      } else {
        try {
          const cap = build(exp, mode, { scale: 1 })
          job.status = { ...job.status, state: 'done', fraction: 1, message: 'done', capture_id: cap.meta.id }
          emit(job, { type: 'done', capture_id: cap.meta.id })
        } catch (err) {
          job.status = { ...job.status, state: 'error', error: String(err) }
          emit(job, { type: 'error', message: String(err) })
        }
      }
    }
    job.timer = setTimeout(tick, 80)
    return { job_id: id }
  },
  watchJob(jobId, onEvent) {
    const job = jobs.get(jobId)
    if (!job) { queueMicrotask(() => onEvent({ type: 'error', message: 'unknown job' })); return () => {} }
    job.listeners.add(onEvent)
    return () => job.listeners.delete(onEvent)
  },
  async jobStatus(id) { return jobs.get(id)!.status },
  async cancelJob(id) {
    const job = jobs.get(id)!
    if (job.status.state === 'running') {
      job.cancelled = true
      if (job.timer) clearTimeout(job.timer)
      job.status = { ...job.status, state: 'cancelled' }
      emit(job, { type: 'cancelled' })
    }
    return job.status
  },
  async listCaptures() { return order.map((i) => store.get(i)!.meta).filter((m) => m.mode !== 'preview') },
  async getCapture(id) { return need(id).meta },
  async band(id, ex, band) {
    const c = need(id); const si = c.steps.findIndex((s) => s.ex === ex)
    if (si < 0) throw new Error(`no excitation ${ex}`)
    const nb = c.meta.emission_nm[String(ex)].length, N = c.maps.width * c.maps.height
    const cube = cubeFor(c, si, modeOf(c)), out = new Float32Array(N)
    for (let k = 0; k < N; k++) out[k] = cube[k * nb + band]
    return { width: c.maps.width, height: c.maps.height, data: out }
  },
  async cube(id, ex): Promise<Cube> {
    const c = need(id); const si = c.steps.findIndex((s) => s.ex === ex)
    if (si < 0) throw new Error(`no excitation ${ex}`)
    return { width: c.maps.width, height: c.maps.height, bands: c.meta.emission_nm[String(ex)].length, data: cubeFor(c, si, modeOf(c)) }
  },
  async pixel(id, x, y): Promise<PixelExplain> {
    const c = need(id)
    const k = Math.min(c.maps.height - 1, Math.max(0, y)) * c.maps.width + Math.min(c.maps.width - 1, Math.max(0, x))
    const adu: PixelExplain['adu'] = {}, rad: PixelExplain['radiance'] = {}, components: PixelExplain['components'] = {}, em: PixelExplain['emission_nm'] = {}
    c.steps.forEach((st, si) => {
      const key = String(st.ex), bands = c.meta.emission_nm[key]
      em[key] = bands
      const a: number[] = [], r: number[] = []
      const comp: Record<string, number[]> = {}
      bands.forEach((_, j) => {
        const parts: Record<string, number> = {}
        a.push(valueAt(c, si, k, j, modeOf(c), parts))
        let tot = 0
        for (const [n, v] of Object.entries(parts)) { (comp[n] ??= [])[j] = v; tot += v }
        r.push(tot)
      })
      adu[key] = a; rad[key] = r; components[key] = comp
    })
    const mi = c.maps.mat[k]
    const blkGt: Record<string, number> = {}
    if (c.maps.face[k] > 0 && c.maps.obj[k] >= 0) {
      const sh = c.exp.objects[c.maps.obj[k]].shape
      if (sh.kind === 'layered_block') Object.assign(blkGt, blockGround(c.exp, sh, c.maps.lx[k], c.maps.lz[k], c.maps.face[k]))
    }
    return {
      x, y, emission_nm: em, adu, radiance: rad, components,
      ground_truth: {
        material: mi >= 0 ? c.maps.materials[mi] : null,
        object: c.maps.obj[k] >= 0 ? c.exp.objects[c.maps.obj[k]].id : (mi >= 0 ? 'background' : null),
        depth_mm: mi >= 0 ? c.maps.depth[k] : null,
        incidence_deg: mi >= 0 ? c.maps.incidence[k] : null,
        irradiance: c.maps.irradiance[k],
        ...blkGt,
      },
    }
  },
  async groundTruth(id, name): Promise<Raster> {
    const c = need(id)
    const data = gtRaster(c, name)
    if (!data) throw new Error(`unknown ground-truth layer "${name}"`)
    return { width: c.maps.width, height: c.maps.height, data }
  },
  imageUrl: () => '',
  downloadUrl: () => '#',

  // ---- v1.1 analysis ------------------------------------------------------------------------
  async eem(id, r) { await sleep(10); return mockEem(access(id), r) },
  async roi(id, r) { await sleep(10); return mockRoi(access(id), r) },
  async line(id, r) { await sleep(10); return mockLine(access(id), r) },
  async response(id, r) { await sleep(10); return mockResponse(access(id), r) },
  async volumes(id) { return (need(id).meta as CaptureMeta & { volumes?: ReturnType<typeof volumeInfos> }).volumes ?? [] },
  async volume(id, q) { await sleep(10); return volumeData(need(id).exp, q) },
  async captureExperiment(id) { return need(id).exp },

  // ---- v1.1 sweeps --------------------------------------------------------------------------
  async startSweep(r) { return startMockSweep(r) },
  watchSweep(sweepId, onEvent) {
    const sw = sweeps.get(sweepId)
    if (!sw) { queueMicrotask(() => onEvent({ type: 'error', message: 'unknown sweep' })); return () => {} }
    sw.listeners.add(onEvent)
    sw.backlog.forEach((e) => onEvent(e))      // late subscribers get everything that already happened
    return () => sw.listeners.delete(onEvent)
  },
  async sweepStatus(id) { return sweeps.get(id)!.status },
  async cancelSweep(id) {
    const sw = sweeps.get(id)!
    if (sw.status.state === 'running' || sw.status.state === 'queued') { sw.cancelled = true; if (sw.timer) clearTimeout(sw.timer); sw.status = { ...sw.status, state: 'cancelled' }; sweepEmit(sw, { type: 'cancelled' }) }
    return sw.status
  },
}

// ---- helpers for the analysis endpoints -------------------------------------------------------------

function gtRaster(c: MockCapture, name: string): Float32Array | null {
  const m = c.maps
  const cached = c.gts.get(name); if (cached) return cached
  let out: Float32Array | null = null
  if (name === 'material_index') out = Float32Array.from(m.mat)
  else if (name === 'object_index') out = Float32Array.from(m.obj)
  else if (name === 'depth_mm') out = m.depth
  else if (name === 'incidence_deg') out = m.incidence
  else if (name === 'irradiance') out = m.irradiance
  else if (c.meta.ground_truth_layers.includes(name)) {
    // layered-block ground truth: fill every block key in one pass over the pixels
    const keys = c.meta.ground_truth_layers.filter((n) => n.includes(':') || n === 'block_face' || n === 'origin_depth_mm')
    const arrs = new Map(keys.map((k) => [k, new Float32Array(m.width * m.height).fill(NaN)] as const))
    for (let k = 0; k < m.width * m.height; k++) {
      if (m.face[k] === 0 || m.obj[k] < 0) continue
      const sh = c.exp.objects[m.obj[k]].shape
      if (sh.kind !== 'layered_block') continue
      const g = blockGround(c.exp, sh, m.lx[k], m.lz[k], m.face[k])
      for (const [key, v] of Object.entries(g)) { const a = arrs.get(key); if (a) a[k] = v }
    }
    arrs.forEach((a, key) => c.gts.set(key, a))
    out = arrs.get(name) ?? null
    return out
  }
  if (out) c.gts.set(name, out)
  return out
}

/** Mock dark/white/exposure normalisation of a cube — mirrors what the engine documents for NormSpec. */
function normCube(c: MockCapture, ex: number, norm: NormSpec): Float32Array {
  const key = `${ex}|${norm.mode}|${norm.dark_capture_id ?? ''}|${norm.white_capture_id ?? ''}`
  const hit = c.normCubes.get(key); if (hit) return hit
  const si = c.steps.findIndex((s) => s.ex === ex)
  if (si < 0) throw new Error(`no excitation ${ex}`)
  const raw = cubeFor(c, si, modeOf(c))
  const sens = activeCamera(c.exp).sensor, st = c.steps[si]
  const other = (id: string | null | undefined, role: string) => {
    if (!id) throw new Error(`${norm.mode} needs a ${role} capture`)
    const o = need(id)
    if (o.maps.width !== c.maps.width || o.maps.height !== c.maps.height) throw new Error(`${role} capture ${id} has a different image size`)
    const osi = o.steps.findIndex((s) => s.ex === ex)
    if (osi < 0) throw new Error(`${role} capture ${id} lacks excitation ${ex} nm`)
    const cu = cubeFor(o, osi, modeOf(o))
    if (cu.length !== raw.length) throw new Error(`${role} capture ${id} has a different band set at ${ex} nm`)
    return cu
  }
  const out = new Float32Array(raw.length)
  switch (norm.mode) {
    case 'adu': out.set(raw); break
    case 'electrons': for (let i = 0; i < raw.length; i++) out[i] = (raw[i] - sens.offset_adu) * sens.gain_e_per_adu; break
    case 'radiance': { const k = 1 / (GAIN * (st.powerMW * st.exposureMs) / 100); for (let i = 0; i < raw.length; i++) out[i] = (raw[i] - sens.offset_adu) * k; break }
    case 'exposure_power': { const k = 1 / (st.exposureMs * st.powerMW); for (let i = 0; i < raw.length; i++) out[i] = (raw[i] - sens.offset_adu) * k; break }
    case 'irradiance_normalized': {
      // E_ex (mW/mm²) ≈ relative irradiance map × source power; NaN where E_ex < 1e-3 of its maximum (mirrors analysis.py)
      const nb = raw.length / c.maps.irradiance.length, t = st.exposureMs / 1000
      let eMax = 0
      for (let k = 0; k < c.maps.irradiance.length; k++) eMax = Math.max(eMax, c.maps.irradiance[k] * st.powerMW)
      for (let k = 0; k < c.maps.irradiance.length; k++) {
        const e = c.maps.irradiance[k] * st.powerMW
        for (let b = 0; b < nb; b++) out[k * nb + b] = e > 1e-3 * eMax ? (raw[k * nb + b] - sens.offset_adu) / (t * e) : NaN
      }
      break
    }
    case 'dark_subtracted': { const d = other(norm.dark_capture_id, 'dark'); for (let i = 0; i < raw.length; i++) out[i] = raw[i] - d[i]; break }
    case 'white_normalized': {
      const d = other(norm.dark_capture_id, 'dark'), w = other(norm.white_capture_id, 'white')
      for (let i = 0; i < raw.length; i++) { const den = w[i] - d[i]; out[i] = Math.abs(den) > 1e-6 ? (raw[i] - d[i]) / den : NaN }
      break
    }
  }
  c.normCubes.set(key, out)
  if (c.normCubes.size > 12) c.normCubes.delete(c.normCubes.keys().next().value as string)
  return out
}

function access(id: string): CaptureAccess {
  const c = need(id)
  return {
    meta: c.meta,
    cube: (ex, norm) => normCube(c, ex, norm),
    gt: (name) => gtRaster(c, name),
    pxMm: c.maps.pxMm,
    components: (indices) => {
      const acc: Record<string, Record<string, number[]>> = {}
      c.steps.forEach((st, si) => {
        const key = String(st.ex), B = c.meta.emission_nm[key].length
        const sum: Record<string, number[]> = {}
        for (let q = 0; q < indices.length; q++) {
          const k = indices[q]
          const pre = modeOf(c) === 'sample' ? blockComps(c, si, k) : null
          for (let b = 0; b < B; b++) {
            const parts: Record<string, number> = {}
            valueAt(c, si, k, b, modeOf(c), parts, pre)
            for (const [n, v] of Object.entries(parts)) { (sum[n] ??= new Array<number>(B).fill(0))[b] += v / indices.length }
          }
        }
        acc[key] = sum
      })
      return acc
    },
  }
}

// ---- sweeps -----------------------------------------------------------------------------------------
interface Sweep { status: SweepStatus; listeners: Set<(e: SweepEvent) => void>; backlog: SweepEvent[]; timer: ReturnType<typeof setTimeout> | null; cancelled: boolean }
const sweeps = new Map<string, Sweep>()
let sweepSeq = 0
function sweepEmit(sw: Sweep, e: SweepEvent) { sw.backlog.push(e); sw.listeners.forEach((l) => l(e)) }

function startMockSweep(r: SweepRequest): { sweep_id: string } {
  const id = `sweep_${++sweepSeq}`
  const values = r.parameter.values
  if (!values.length) throw new Error('sweep needs at least one parameter value')
  const sw: Sweep = {
    status: { sweep_id: id, state: 'running', fraction: 0, parameter: r.parameter, capture_ids: values.map(() => null), error: null },
    listeners: new Set(), backlog: [], timer: null, cancelled: false,
  }
  sweeps.set(id, sw)
  // validate the path up front so a bad parameter fails the request, not the middle of the run
  setByPointer(r.experiment, r.parameter.path, values[0])
  let i = 0
  const mode = r.mode ?? 'preview'
  const tick = () => {
    if (sw.cancelled) return
    if (i >= values.length) {
      sw.status = { ...sw.status, state: 'done', fraction: 1 }
      sweepEmit(sw, { type: 'done', sweep_id: id })
      return
    }
    const idx = i++
    sw.status = { ...sw.status, fraction: idx / values.length }
    sweepEmit(sw, { type: 'progress', fraction: idx / values.length, message: `point ${idx + 1}/${values.length}`, index: idx })
    setTimeout(() => {
      if (sw.cancelled) return
      try {
        const exp = setByPointer(r.experiment, r.parameter.path, values[idx])
        const scale = mode === 'preview' ? Math.min(0.5, exp.acquisition.preview_scale) : 0.7
        // like the engine: a preview point holds ONE excitation (the first enabled step)
        const cap = build(exp, 'sample', { scale, only: mode === 'preview' ? stepsOf(exp)[0]?.ex ?? null : null })
        sw.status.capture_ids[idx] = cap.meta.id
        sweepEmit(sw, { type: 'point', index: idx, value: values[idx], capture_id: cap.meta.id })
        sw.timer = setTimeout(tick, 30)
      } catch (err) {
        sw.status = { ...sw.status, state: 'error', error: String(err instanceof Error ? err.message : err) }
        sweepEmit(sw, { type: 'error', message: sw.status.error ?? 'sweep failed' })
      }
    }, 20)
  }
  sw.timer = setTimeout(tick, 60)
  return { sweep_id: id }
}
