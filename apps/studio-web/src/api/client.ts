/** HTTP/WS client for the SpectralForge Studio engine — exactly the contract in `./types.ts`.
 *  `getEngine()` returns the real engine when `/api/health` answers, else the in-browser mock. */
import type { Experiment } from '../generated/experiment'
import type {
  CaptureMeta, CaptureRequest, Health, JobEvent, JobStatus, PixelExplain, PreviewRequest, PresetsBundle, ValidationResult,
} from './types'

export type Kind = 'adu' | 'radiance' | 'electrons'
export type DownloadFormat = 'spectradata_pkl' | 'npz' | 'tiff_zip' | 'ground_truth_zip' | 'experiment_json'

export interface Raster { width: number; height: number; data: Float32Array }
export interface Cube extends Raster { bands: number }

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
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) throw new ApiError(r.status, `${init?.method ?? 'GET'} ${url} → ${r.status} ${await r.text().catch(() => '')}`.slice(0, 400))
  return (await r.json()) as T
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
