/** HTTP/WS client for the SpectralForge Studio engine — exactly the contract in `./types.ts`.
 *  `getEngine()` returns the real engine when `/api/health` answers, else the in-browser mock. */
import type { Experiment } from '../generated/experiment'
import type {
  CaptureMeta, CaptureRequest, EemRequest, EemResult, Health, JobEvent, JobStatus, LineProfile, LineRequest, PixelExplain, PreviewRequest, PresetsBundle,
  ResponseCurve, ResponseRequest, RoiRequest, RoiStats, SweepEvent, SweepRequest, SweepStatus, ValidationResult, VolumeInfo,
} from './types'

export type Kind = 'adu' | 'radiance' | 'electrons'
export type DownloadFormat = 'spectradata_pkl' | 'npz' | 'tiff_zip' | 'ground_truth_zip' | 'experiment_json'

export interface Raster { width: number; height: number; data: Float32Array }
export interface Cube extends Raster { bands: number }
/** One 3D light-field volume: float32 (nx, nz, nd) C-order; `nd = n_layers · sublayers`. */
export interface VolumeData { dims: [number, number, number]; data: Float32Array }
export interface VolumeQuery { object: string; kind: VolumeInfo['kind']; ex?: number; channel?: string }

export interface Engine {
  kind: 'real' | 'mock'
  health(): Promise<Health>
  presets(): Promise<PresetsBundle>
  validate(e: Experiment): Promise<ValidationResult>
  preview(r: PreviewRequest): Promise<CaptureMeta>
  startCapture(r: CaptureRequest): Promise<{ job_id: string }>
  /** Subscribe to job events; returns an unsubscribe function. */
  watchJob(jobId: string, onEvent: (e: JobEvent) => void): () => void
  jobStatus(jobId: string): Promise<JobStatus>
  cancelJob(jobId: string): Promise<JobStatus>
  listCaptures(): Promise<CaptureMeta[]>
  getCapture(id: string): Promise<CaptureMeta>
  band(id: string, ex: number, band: number, kind?: Kind): Promise<Raster>
  cube(id: string, ex: number, kind?: Kind): Promise<Cube>
  pixel(id: string, x: number, y: number): Promise<PixelExplain>
  groundTruth(id: string, name: string): Promise<Raster>
  imageUrl(id: string, mode: 'true_color' | 'false_color' | 'band', ex?: number, band?: number, gain?: number): string
  downloadUrl(id: string, format: DownloadFormat): string

  // ---- v1.1 analysis ------------------------------------------------------------------------
  eem(id: string, r: EemRequest): Promise<EemResult>
  roi(id: string, r: RoiRequest): Promise<RoiStats>
  line(id: string, r: LineRequest): Promise<LineProfile>
  response(id: string, r: ResponseRequest): Promise<ResponseCurve>
  volumes(id: string): Promise<VolumeInfo[]>
  volume(id: string, q: VolumeQuery): Promise<VolumeData>
  /** The experiment a capture was made from (for physical layer geometry); null if unavailable. */
  captureExperiment(id: string): Promise<Experiment | null>
  // ---- v1.1 sweeps --------------------------------------------------------------------------
  startSweep(r: SweepRequest): Promise<{ sweep_id: string }>
  watchSweep(sweepId: string, onEvent: (e: SweepEvent) => void): () => void
  sweepStatus(sweepId: string): Promise<SweepStatus>
  cancelSweep(sweepId: string): Promise<SweepStatus>
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

/** JSON.parse that tolerates Python's bare NaN / Infinity tokens (mapped to null). */
export function parseLooseJson<T>(text: string): T {
  try { return JSON.parse(text) as T } catch {
    return JSON.parse(text.replace(/(?<=[:,[]\s*)-?(?:NaN|Infinity)(?=\s*[,\]}])/g, 'null')) as T
  }
}
async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) throw new ApiError(r.status, `${init?.method ?? 'GET'} ${url} → ${r.status} ${await r.text().catch(() => '')}`.slice(0, 400))
  return parseLooseJson<T>(await r.text())
}
/** Replace `null` entries of numeric arrays (how JSON carries NaN) by NaN, recursively. */
export function nanify<T>(o: T): T {
  if (Array.isArray(o)) {
    const hasNum = o.some((v) => typeof v === 'number')
    return (hasNum || o.every((v) => v === null) ? o.map((v) => (v === null ? NaN : nanify(v))) : o.map((v) => nanify(v))) as T
  }
  if (o && typeof o === 'object') { const out: Record<string, unknown> = {}; for (const [k, v] of Object.entries(o)) out[k] = nanify(v); return out as T }
  return o
}
const post = <T,>(url: string, body?: unknown) =>
  json<T>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })

async function binary(url: string): Promise<{ buf: Float32Array; w: number; h: number; b: number }> {
  const r = await fetch(url)
  if (!r.ok) throw new ApiError(r.status, `GET ${url} → ${r.status}`)
  const ab = await r.arrayBuffer()
  return {
    buf: new Float32Array(ab), w: Number(r.headers.get('X-Width') ?? 0),
    h: Number(r.headers.get('X-Height') ?? 0), b: Number(r.headers.get('X-Bands') ?? 1),
  }
}

const q = (o: Record<string, string | number | undefined>) =>
  Object.entries(o).filter(([, v]) => v !== undefined).map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join('&')

export const realEngine: Engine = {
  kind: 'real',
  health: () => json<Health>('/api/health'),
  presets: () => json<PresetsBundle>('/api/presets'),
  validate: (e) => post<ValidationResult>('/api/validate', e),
  preview: (r) => post<CaptureMeta>('/api/preview', r),
  startCapture: (r) => post<{ job_id: string }>('/api/captures', r),
  watchJob(jobId, onEvent) {
    let closed = false
    let poll: ReturnType<typeof setInterval> | null = null
    const startPolling = () => {
      if (poll || closed) return
      poll = setInterval(async () => {
        try {
          const s = await json<JobStatus>(`/api/jobs/${jobId}`)
          if (s.state === 'running' || s.state === 'queued') onEvent({ type: 'progress', fraction: s.fraction, message: s.message })
          else {
            if (poll) clearInterval(poll)
            closed = true
            if (s.state === 'done' && s.capture_id) onEvent({ type: 'done', capture_id: s.capture_id })
            else if (s.state === 'cancelled') onEvent({ type: 'cancelled' })
            else onEvent({ type: 'error', message: s.error ?? 'job failed' })
          }
        } catch { /* keep polling */ }
      }, 500)
    }
    let ws: WebSocket | null = null
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}`)
      ws.onmessage = (m) => {
        try {
          const ev = JSON.parse(String(m.data)) as JobEvent
          if (ev.type !== 'progress') closed = true
          onEvent(ev)
        } catch { /* ignore malformed frame */ }
      }
      ws.onerror = () => startPolling()
      ws.onclose = () => { if (!closed) startPolling() }
    } catch { startPolling() }
    return () => { closed = true; if (poll) clearInterval(poll); ws?.close() }
  },
  jobStatus: (id) => json<JobStatus>(`/api/jobs/${id}`),
  cancelJob: (id) => post<JobStatus>(`/api/jobs/${id}/cancel`),
  listCaptures: () => json<CaptureMeta[]>('/api/captures'),
  getCapture: (id) => json<CaptureMeta>(`/api/captures/${id}`),
  async band(id, ex, band, kind = 'adu') {
    const { buf, w, h } = await binary(`/api/captures/${id}/band?${q({ ex, band, kind })}`)
    return { width: w, height: h, data: buf }
  },
  async cube(id, ex, kind = 'adu') {
    const { buf, w, h, b } = await binary(`/api/captures/${id}/cube?${q({ ex, kind })}`)
    return { width: w, height: h, bands: b, data: buf }
  },
  pixel: (id, x, y) => json<PixelExplain>(`/api/captures/${id}/pixel?x=${x}&y=${y}`),
  async groundTruth(id, name) {
    const { buf, w, h } = await binary(`/api/captures/${id}/gt/${encodeURIComponent(name)}`)
    return { width: w, height: h, data: buf }
  },
  eem: async (id, r) => nanify(await post<EemResult>(`/api/captures/${id}/eem`, r)),
  roi: async (id, r) => nanify(await post<RoiStats>(`/api/captures/${id}/roi`, r)),
  line: async (id, r) => nanify(await post<LineProfile>(`/api/captures/${id}/line`, r)),
  response: async (id, r) => nanify(await post<ResponseCurve>(`/api/captures/${id}/response`, r)),
  volumes: (id) => json<VolumeInfo[]>(`/api/captures/${id}/volumes`),
  async volume(id, qq) {
    const r = await fetch(`/api/captures/${id}/volume?${q({ object: qq.object, kind: qq.kind, ex: qq.ex, channel: qq.channel })}`)
    if (!r.ok) throw new ApiError(r.status, `GET volume → ${r.status} ${(await r.text().catch(() => '')).slice(0, 200)}`)
    const dims = (r.headers.get('X-Dims') ?? '').split(',').map(Number)
    const data = new Float32Array(await r.arrayBuffer())
    if (dims.length !== 3 || dims.some((d) => !(d > 0)) || dims[0] * dims[1] * dims[2] !== data.length) throw new ApiError(500, `volume has inconsistent X-Dims "${dims.join(',')}" for ${data.length} values`)
    return { dims: dims as [number, number, number], data }
  },
  async captureExperiment(id) {
    try {
      const r = await fetch(`/api/captures/${id}/download?format=experiment_json`)
      if (!r.ok) return null
      return parseLooseJson<Experiment>(await r.text())
    } catch { return null }
  },
  startSweep: (r) => post<{ sweep_id: string }>('/api/sweeps', r),
  watchSweep(sweepId, onEvent) {
    let closed = false
    let poll: ReturnType<typeof setInterval> | null = null
    let seen = 0
    const startPolling = () => {
      if (poll || closed) return
      poll = setInterval(async () => {
        try {
          const s = await json<SweepStatus>(`/api/sweeps/${sweepId}`)
          s.capture_ids.forEach((cid, i) => { if (cid && i >= seen) { onEvent({ type: 'point', index: i, value: s.parameter.values[i], capture_id: cid }); seen = i + 1 } })
          if (s.state === 'running' || s.state === 'queued') onEvent({ type: 'progress', fraction: s.fraction, message: `point ${Math.min(seen + 1, s.parameter.values.length)}/${s.parameter.values.length}`, index: seen })
          else {
            if (poll) clearInterval(poll)
            closed = true
            if (s.state === 'done') onEvent({ type: 'done', sweep_id: sweepId })
            else if (s.state === 'cancelled') onEvent({ type: 'cancelled' })
            else onEvent({ type: 'error', message: s.error ?? 'sweep failed' })
          }
        } catch { /* keep polling */ }
      }, 600)
    }
    let ws: WebSocket | null = null
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/api/sweeps/${sweepId}`)
      ws.onmessage = (m) => {
        try {
          const ev = parseLooseJson<SweepEvent>(String(m.data))
          if (ev.type === 'done' || ev.type === 'error' || ev.type === 'cancelled') closed = true
          onEvent(ev)
        } catch { /* ignore malformed frame */ }
      }
      ws.onerror = () => startPolling()
      ws.onclose = () => { if (!closed) startPolling() }
    } catch { startPolling() }
    return () => { closed = true; if (poll) clearInterval(poll); ws?.close() }
  },
  sweepStatus: (id) => json<SweepStatus>(`/api/sweeps/${id}`),
  cancelSweep: (id) => post<SweepStatus>(`/api/sweeps/${id}/cancel`),
  imageUrl: (id, mode, ex, band, gain) => `/api/captures/${id}/image.png?${q({ mode, ex, band, gain })}`,
  downloadUrl: (id, format) => `/api/captures/${id}/download?format=${format}`,
}

let current: Engine | null = null
let probing: Promise<Engine> | null = null

/** Probe `/api/health` (1.2 s timeout); fall back to the mock engine. */
export function detectEngine(force = false): Promise<Engine> {
  if (current && !force) return Promise.resolve(current)
  if (probing) return probing
  probing = (async () => {
    let eng: Engine
    if (typeof location !== 'undefined' && new URLSearchParams(location.search).get('engine') === 'mock') { eng = await loadMock(); current = eng; probing = null; return eng }   // ?engine=mock forces the offline engine
    try {
      const ctl = new AbortController()
      const t = setTimeout(() => ctl.abort(), 1200)
      const r = await fetch('/api/health', { signal: ctl.signal })
      clearTimeout(t)
      const h = (await r.json()) as Health
      eng = r.ok && h.ok ? realEngine : await loadMock()
    } catch { eng = await loadMock() }
    current = eng
    probing = null
    return eng
  })()
  return probing
}
async function loadMock(): Promise<Engine> { return (await import('./mock')).mockEngine }
export const getEngine = (): Engine | null => current
