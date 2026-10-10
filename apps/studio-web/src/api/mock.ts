/** In-browser stand-in for the Python engine, used when `/api/health` is unreachable.
 *  Not physically rigorous: Gaussian dye spectra over analytically ray-cast material regions, but it speaks
 *  the exact `Engine` contract so every UI path (preview, capture progress, explorer, pixel explain) works offline. */
import type { Experiment, MaterialSpec } from '../generated/experiment'
import type {
  BandStats, CaptureMeta, CaptureRequest, JobEvent, JobStatus, PhysicsWarning, PixelExplain, PreviewRequest, ValidationResult,
} from './types'
import type { Cube, Engine, Raster } from './client'
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

function valueAt(c: MockCapture, si: number, k: number, j: number, mode: CaptureMeta['mode'], comps?: Record<string, number>): number {
  const sens = activeCamera(c.exp).sensor
  const mi = c.maps.mat[k]
  const st = c.steps[si]
  let sig = 0
  if (mode === 'dark') sig = 0
  else if (mode === 'white_reference') sig = c.maps.irradiance[k] > 0 ? 38000 * Math.min(1, c.maps.irradiance[k] * 1.4) * (st.powerMW * st.exposureMs) / 400 : 0
  else if (mi >= 0) {
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
  for (let k = 0; k < N; k++) for (let j = 0; j < nb; j++) cube[k * nb + j] = valueAt(c, si, k, j, mode)
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
  const cap: MockCapture = { exp, maps, steps, spectra, cubes: new Map(), meta: null as unknown as CaptureMeta }
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
    saturated_fraction: sat, ground_truth_layers: ['material_index', 'object_index', 'depth_mm', 'incidence_deg', 'irradiance'],
    warnings, timings_s: { total: (performance.now() - t0) / 1000 }, camera_id: activeCamera(exp).id, source_id: activeSource(exp).id,
  }
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
    return {
      x, y, emission_nm: em, adu, radiance: rad, components,
      ground_truth: {
        material: mi >= 0 ? c.maps.materials[mi] : null,
        object: c.maps.obj[k] >= 0 ? c.exp.objects[c.maps.obj[k]].id : (mi >= 0 ? 'background' : null),
        depth_mm: mi >= 0 ? c.maps.depth[k] : null,
        incidence_deg: mi >= 0 ? c.maps.incidence[k] : null,
        irradiance: c.maps.irradiance[k],
      },
    }
  },
  async groundTruth(id, name): Promise<Raster> {
    const c = need(id); const m = c.maps
    const src = name === 'material_index' ? Float32Array.from(m.mat) : name === 'object_index' ? Float32Array.from(m.obj)
      : name === 'depth_mm' ? m.depth : name === 'incidence_deg' ? m.incidence : m.irradiance
    return { width: m.width, height: m.height, data: src }
  },
  imageUrl: () => '',
  downloadUrl: () => '#',
}
